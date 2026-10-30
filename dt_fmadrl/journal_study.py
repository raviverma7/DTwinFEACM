"""Reproducible journal evaluation: six core figures and three tables.

python -m dt_fmadrl.journal_study --preset smoke --output outputs/journal_smoke
python -m dt_fmadrl.journal_study --preset full --output outputs/journal_full
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import pickle
import sys
import time

# Optional workspace-local plotting and process-memory dependencies.
_LOCAL=Path(__file__).resolve().parents[1]/'.runtime'
if _LOCAL.exists():sys.path.insert(0,str(_LOCAL))
import numpy as np
from .baselines import DQNTrainer
from .config import load_config
from .evaluation import evaluate,MemorySampler
from .federated import FederatedTrainer
from .mappo import MAPPOTrainer
from .predictive import PredictiveController, RandomController

MAIN=('fmadrl','dqn','mappo','random','predictive','proposed')
ABLATIONS=('fmadrl','augmentation_only','ranking_only','proposed','dt_only','no_pretraining','ungated')
GROUPS=[dict(name='low',cpu_scale=.5,budget_scale=.5,workload_scale=.8,link_availability=.85),
        dict(name='medium',cpu_scale=1.,budget_scale=1.,workload_scale=1.,link_availability=.95),
        dict(name='high',cpu_scale=1.5,budget_scale=1.5,workload_scale=1.2,link_availability=.99)]

def plan(preset):
    cases=[]
    def add(experiment,axis,methods,overrides=None,disturbance=None):
        for method in methods:cases.append(dict(experiment=experiment,axis=axis,method=method,overrides=overrides or {},disturbance=disturbance))
    populations=(12,24,36) if preset=='smoke' else (40,80,120,160,200)
    for population in populations:
        per,extra=divmod(population,4);add('population',population,MAIN,{'environment.devices_per_edge':[per+(i<extra) for i in range(4)]})
    add('main',0,MAIN)
    for rate in (.25,.5,.75,1.):add('arrival',rate,MAIN,{'environment.arrival_probability':rate})
    for scale in (.5,1.,2.):add('workload',scale,('fmadrl','proposed'),{'environment.domain_workload_scales':[scale]*4})
    for scale in (.5,1.,2.):
        add('edge_capacity',scale,('fmadrl','proposed'),{'environment.mec_cpu_hz':20e9*scale})
        add('backhaul_capacity',scale,('fmadrl','proposed'),{'environment.cloud_backhaul_bps':100e6*scale})
    for mobile in (False,True):add('heterogeneity',int(mobile),MAIN,{'environment.device_groups':GROUPS,'environment.mobile':mobile})
    add('ablation',0,ABLATIONS)
    for kind,levels in dict(noise=(0.,.5,1.,2.,8.),bias=(.25,.5),stale=(1,5,10),workload=(2.,3.),arrival=(1.,),outage=(.8,.5),duration=(3,6),history=(64,128)).items():
        for level in levels:
            overrides={};disturbance=None
            if kind=='history':overrides={'digital_twin.history':int(level)}
            elif kind=='duration':overrides={'environment.outage_duration_slots':int(level),'environment.link_availability':.9}
            else:disturbance=dict(kind=kind,level=level)
            if kind=='arrival':overrides={'environment.arrival_probability':.25}
            add('stress_'+kind,level,('proposed','ungated'),overrides,disturbance)
    for unequal in (False,True):
        environment={'environment.domain_arrival_multipliers':[.25,.5,.75,1.] if unequal else [1.]*4,
                     'environment.domain_workload_scales':[.5,1.,1.5,2.] if unequal else [1.]*4}
        for name,overrides in {
            'independent':{'learning.federation_enabled':False},
            'sample_weighted':{},'equal_weighted':{'learning.federation_weighting':'equal'},
            'interval_1':{'learning.federation_interval':1},'interval_10':{'learning.federation_interval':10},
            'intermittent':{'learning.federation_participation':.5}}.items():
            add('federation_unequal' if unequal else 'federation_equal',name,('proposed',),{**environment,**overrides})
    return cases

def set_values(cfg,values):
    for key,value in values.items():
        section,field=key.split('.');setattr(getattr(cfg,section),field,copy.deepcopy(value))

def make_trainer(cfg,method):
    if method=='random':return RandomController(cfg)
    if method=='dqn':return DQNTrainer(cfg)
    if method=='mappo':return MAPPOTrainer(cfg)
    if method=='predictive':return PredictiveController(cfg)
    twin=method not in ('fmadrl','random')
    if method=='augmentation_only':cfg.digital_twin.ranking=False
    if method=='ranking_only':cfg.digital_twin.augmentation=False
    if method=='dt_only':cfg.digital_twin.augmentation=False;cfg.digital_twin.actor_guidance=False
    if method=='ungated':cfg.digital_twin.gating=False
    return FederatedTrainer(cfg,twin)

def warmup(trainer,slots,method):
    trainer.env.rng=np.random.default_rng(trainer.cfg.seed+40000);trainer.env.task_trace_hash=""
    states=trainer.env.reset();rng=np.random.default_rng(trainer.cfg.seed+30303);generated=0
    for _ in range(slots):
        actions=np.array([rng.choice(np.flatnonzero(f)) for f in trainer.env.feasible_actions()])
        twin=isinstance(trainer,FederatedTrainer) and trainer.use_twin and method!='no_pretraining'
        issued=trainer._issue(states,actions) if twin else None
        ns,r,done,m=trainer.env.step(actions);generated+=int(m.generated.sum())
        if twin:trainer._update_twins(states,actions,ns,r,m,issued)
        states=trainer.env.reset() if done else ns
    return generated

def fingerprint(cfg,method,rounds,warmup_slots,code_hash):
    raw=json.dumps(dict(cfg=asdict(cfg),method=method,rounds=rounds,warmup_slots=warmup_slots,code_hash=code_hash),sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

def run(config,output,preset,seeds,rounds=None,episodes=None,experiments=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'runs').mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    base=load_config(config)
    if preset=='smoke':
        base.environment.vehicles_per_mec=6;base.environment.devices_per_edge=None;base.environment.episode_steps=6
        base.learning.rounds=10;base.learning.local_steps=3;base.learning.evaluation_episodes=2
        base.digital_twin.pretrain_steps=20;base.digital_twin.refit_interval=2;base.digital_twin.history=128
    rounds=base.learning.rounds if rounds is None else rounds;episodes=base.learning.evaluation_episodes if episodes is None else episodes
    warmup_slots=base.digital_twin.pretrain_steps
    cases=plan(preset)
    if experiments:cases=[c for c in cases if c['experiment'] in experiments]
    if len(set(seeds))!=len(seeds) or len(seeds)<2:raise ValueError("Use at least two distinct independent seeds")
    source=Path(__file__).parent
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source.glob('*.py')))).hexdigest()
    manifest=dict(preset=preset,seeds=seeds,rounds=rounds,evaluation_episodes=episodes,warmup_slots=warmup_slots,
                  base_config=asdict(base),code_sha256=code_hash,cases=cases,
                  runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
                  completion_denominator='all generated tasks; inactive device slots excluded',failure_treatment='modeled full execution delay/energy included for failed tasks',
                  arrival_model='at most one Bernoulli task per device per slot; rate=probability/slot_duration_s',
                  queue_metric='unfinished CPU cycles divided by mean current task cycles; equivalent tasks, not an event queue',
                  learning_budget='identical warmup slots and policy-training slots for all methods; warmup actor/Q updates disabled',
                  twin_validation='single-device counterfactual interventions with other actions and RNG fixed; no probe transition used for training',
                  latency='joint controller call including augmentation/ranking; diagnostics and probes excluded; mean/p95/p99',
                  memory='20 ms sampled process RSS including interpreter; peak can miss shorter allocations',
                  model_energy='fixed DT energy remains an assumption; compute times are measured independently',
                  parameter_selection='fixed settings; MAPPO/DQN baseline competitiveness requires separate hyperparameter validation')
    manifest_file=out/'manifest.json'
    if manifest_file.exists():
        previous=json.loads(manifest_file.read_text())
        if previous!=manifest:raise ValueError('Output folder belongs to a different plan/code version. Use a new output folder.')
    manifest_file.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    index=[];total=len(cases)*len(seeds);completed=0
    for seed in seeds:
        for case in cases:
            cfg=copy.deepcopy(base);cfg.seed=seed;set_values(cfg,case['overrides']);cfg.digital_twin.pretrain_steps=0
            # Materialize method switches before fingerprinting.
            trainer=make_trainer(cfg,case['method'])
            case_hash=hashlib.sha256(json.dumps(dict(case=case,seed=seed,code=code_hash),sort_keys=True).encode()).hexdigest()[:20]
            result_file=out/'runs'/f'{case_hash}.json';completed+=1
            if result_file.exists():
                record=json.loads(result_file.read_text());index.append(dict(path=str(result_file.relative_to(out)),**record['case'],seed=seed));continue
            training_hash=fingerprint(cfg,case['method'],rounds,warmup_slots,code_hash)
            checkpoint=out/'checkpoints'/f'{training_hash}.pkl'
            if checkpoint.exists():
                # Only trusted checkpoints created in this exact code/config run are loaded.
                with checkpoint.open('rb') as handle:trainer,history,overhead=pickle.load(handle)
            else:
                start=time.perf_counter()
                with MemorySampler() as memory:
                    warmup_generated=warmup(trainer,warmup_slots,case['method'])
                    warmup_trace=trainer.env.task_trace_hash
                    trainer.env.rng=np.random.default_rng(cfg.seed+50000);trainer.env.task_trace_hash=""
                    history=trainer.train(rounds)
                overhead=dict(training_seconds=time.perf_counter()-start,peak_rss_bytes=memory.peak,rss_baseline_bytes=memory.baseline,
                              warmup_task_trace_hash=warmup_trace,training_task_trace_hash=trainer.env.task_trace_hash,
                              warmup_generated_tasks=warmup_generated,training_generated_tasks=trainer.training_generated_tasks,
                              training_slot_budget=rounds*cfg.learning.local_steps,warmup_slot_budget=warmup_slots,
                              centralized_training_data_bytes=getattr(trainer,'training_data_bytes',0))
                with checkpoint.open('wb') as handle:pickle.dump((trainer,history,overhead),handle)
            train_diagnostics=trainer.diagnostics() if isinstance(trainer,FederatedTrainer) else {}
            train_refits=sum(t.fit_count for t in trainer.twins) if isinstance(trainer,FederatedTrainer) and trainer.use_twin else 0
            probes=12 if case['experiment']=='main' and case['method']=='proposed' else 0
            if case['experiment']=='stress_noise':probes=6
            evaluation=evaluate(trainer,case['method'],episodes,probe_steps=probes,disturbance=case['disturbance'])
            diagnostics=evaluation['diagnostics'];overhead.update(training_dt_refit_seconds=train_diagnostics.get('dt_refit_seconds',0),evaluation_dt_refit_seconds=diagnostics.get('dt_refit_seconds',0)-train_diagnostics.get('dt_refit_seconds',0),training_synchronization_bytes=train_diagnostics.get('synchronization_bytes',0),evaluation_synchronization_bytes=diagnostics.get('synchronization_bytes',0)-train_diagnostics.get('synchronization_bytes',0),peak_rss_delta_bytes=max(0,overhead['peak_rss_bytes']-overhead['rss_baseline_bytes']),dt_refit_count_training=train_refits,federation_bytes=diagnostics.get('federation_bytes',0),synchronization_bytes=diagnostics.get('synchronization_bytes',0),dt_refit_seconds=diagnostics.get('dt_refit_seconds',0))
            result=dict(case=case,seed=seed,config=asdict(trainer.cfg),history=[asdict(r) for r in history],overhead=overhead,**evaluation)
            temporary=result_file.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(result_file)
            index.append(dict(path=str(result_file.relative_to(out)),**case,seed=seed))
            (out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
            if completed%10==0 or completed==total:print(f'Completed {completed}/{total}: {case["experiment"]} {case["method"]}, seed {seed}',flush=True)
    (out/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    from .journal_report import generate
    return generate(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/research.json');p.add_argument('--output',default='outputs/journal_smoke');p.add_argument('--preset',choices=('smoke','full'),default='smoke');p.add_argument('--seeds',default='11,22,33,44,55');p.add_argument('--rounds',type=int);p.add_argument('--episodes',type=int);p.add_argument('--experiments',nargs='+');a=p.parse_args()
    print(json.dumps(run(a.config,a.output,a.preset,[int(v) for v in a.seeds.split(',')],a.rounds,a.episodes,a.experiments),indent=2))
