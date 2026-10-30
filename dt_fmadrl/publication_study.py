"""Disjoint tuning/confirmation study. All outputs stay inside this revision.

Run: python -m dt_fmadrl.publication_study --profile extended --workers 3
An independently selected proposed method is compared with tuned baselines,
both analytical references, legacy and component ablations, and its ungated twin.
No result is discarded for being unfavorable.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'.runtime'))
import numpy as np
from .config import load_config
from .federated import FederatedTrainer
from .baselines import DQNTrainer
from .mappo import MAPPOTrainer
from .predictive import PredictiveController, ObservationPredictiveController, RandomController
from .journal_study import warmup, set_values, GROUPS
from .evaluation import evaluate
from .statistics import ci95

METHODS=('fmadrl','dqn','mappo','random','predictive_full','predictive_observed',
         'legacy','proposed','augmentation_only','ranking_only','ungated')
METRICS=('delay','energy','completion_rate','cost','decision_ms_p95','gate_zero_rate')
SCENARIOS={
    'nominal':({},None),
    'corrupted_twin':({},dict(kind='noise',level=8.)),
    'edge_cpu_half':({'environment.mec_cpu_hz':10e9},None),
    'workload_double':({'environment.domain_workload_scales':[2.]*4},None),
    'mobile_groups':({'environment.mobile':True,'environment.device_groups':GROUPS},None),
}

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def profiles(name):
    if name=='pilot':return dict(rounds=4,local_steps=5,episode_steps=20,episodes=1,warmup=30,tune_seeds=[101],seeds=[1001,1002])
    return dict(rounds=120,local_steps=25,episode_steps=100,episodes=15,warmup=300,
                tune_seeds=[101,202,303],seeds=list(range(1001,1011)))

def base_config(p,seed):
    cfg=load_config(ROOT/'config/research.json');cfg.seed=seed
    cfg.environment.vehicles_per_mec=6;cfg.environment.devices_per_edge=None
    cfg.environment.episode_steps=p['episode_steps'];cfg.learning.local_steps=p['local_steps']
    cfg.learning.rounds=p['rounds'];cfg.learning.evaluation_episodes=p['episodes']
    cfg.digital_twin.pretrain_steps=0;cfg.digital_twin.refit_interval=20
    cfg.digital_twin.history=1024
    return cfg

def candidates(method):
    if method=='proposed':
        return [dict(**{'digital_twin.model_bias':w,'digital_twin.state_action_interactions':True,
                       'digital_twin.require_fitted_guidance':True}) for w in (.25,1.,3.)]
    if method=='dqn':return [{'learning.critic_lr':v} for v in (.003,.01,.03)]
    if method=='mappo':return [{'learning.actor_lr':v,'learning.critic_lr':v*2} for v in (.0003,.001,.003)]
    if method=='fmadrl':return [{'learning.actor_lr':v,'learning.critic_lr':v*2} for v in (.003,.01,.03)]
    raise ValueError(method)

def trainer_for(cfg,method):
    if method=='predictive_observed':return ObservationPredictiveController(cfg)
    if method=='predictive_full':return PredictiveController(cfg)
    if method=='random':return RandomController(cfg)
    if method=='dqn':return DQNTrainer(cfg)
    if method=='mappo':return MAPPOTrainer(cfg)
    if method=='legacy':
        cfg.digital_twin.state_action_interactions=False;cfg.digital_twin.require_fitted_guidance=False
        cfg.digital_twin.model_bias=10.
    if method=='augmentation_only':cfg.digital_twin.ranking=False
    if method=='ranking_only':cfg.digital_twin.augmentation=False
    if method=='ungated':cfg.digital_twin.gating=False
    return FederatedTrainer(cfg,method!='fmadrl')

def configure_method(cfg,method,selection):
    key='proposed' if method in ('augmentation_only','ranking_only','ungated') else method
    if key in selection:set_values(cfg,selection[key])
    return trainer_for(cfg,method)

def train(cfg,method,p,selection):
    trainer=configure_method(cfg,method,selection)
    start=time.perf_counter();n=warmup(trainer,p['warmup'],method)
    warmhash=trainer.env.task_trace_hash
    trainer.env.rng=np.random.default_rng(cfg.seed+50000);trainer.env.task_trace_hash=''
    history=trainer.train(p['rounds'])
    return trainer,history,dict(training_seconds=time.perf_counter()-start,warmup_tasks=n,
        warmup_trace=warmhash,training_tasks=trainer.training_generated_tasks,
        training_trace=trainer.env.task_trace_hash,training_slots=p['rounds']*p['local_steps'])

def tune_job(job):
    method,index,seed,p,out=job;path=Path(out)/'tuning'/f'{method}_{index}_{seed}.json'
    if path.exists():return json.loads(path.read_text())
    cfg=base_config(p,seed);override=candidates(method)[index]
    trainer,_,overhead=train(cfg,method,p,{method:override})
    result=evaluate(trainer,method,p['episodes'],probe_steps=0)
    # Fixed, predeclared selection criterion; completion is protected explicitly.
    score=result['metrics']['cost']+2*(1-result['metrics']['completion_rate'])
    record=dict(method=method,candidate=index,seed=seed,overrides=override,score=score,metrics=result['metrics'],overhead=overhead)
    save(path,record);return record

def confirm_job(job):
    method,seed,p,out,selection=job;out=Path(out)
    checkpoint=out/'checkpoints'/f'{method}_{seed}.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as f:trained,history,overhead=pickle.load(f)
    else:
        cfg=base_config(p,seed);trained,history,overhead=train(cfg,method,p,selection)
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        with checkpoint.open('wb') as f:pickle.dump((trained,history,overhead),f)
    result_paths=[]
    for scenario,(overrides,disturbance) in SCENARIOS.items():
        path=out/'runs'/f'{method}_{seed}_{scenario}.json';result_paths.append(str(path))
        if path.exists():continue
        trainer=copy.deepcopy(trained);set_values(trainer.cfg,overrides)
        probes=20 if method=='proposed' and scenario in ('nominal','corrupted_twin') else 0
        start=time.perf_counter();result=evaluate(trainer,method,p['episodes'],probe_steps=probes,disturbance=disturbance)
        # Store all task-level summaries, probes and trajectories, not cherry-picked points.
        record=dict(method=method,seed=seed,scenario=scenario,config=asdict(trainer.cfg),
            overhead=overhead,history=[asdict(h) for h in history],evaluation_seconds=time.perf_counter()-start,**result)
        save(path,record)
    return result_paths

def parallel(jobs,fn,workers,label):
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,j):j for j in jobs}
        for count,f in enumerate(as_completed(futures),1):
            results.append(f.result());print(f'{label}: {count}/{len(jobs)} {futures[f][:2]}',flush=True)
    return results

def paired_p(values):
    """Two-sided exact paired sign-flip test; assumes exchangeability of signs."""
    v=np.asarray(values,float);n=len(v)
    if n<2:return None
    signs=2*((np.arange(2**n)[:,None]>>np.arange(n))&1)-1
    observed=abs(float(v.mean()))
    return float(np.mean(np.abs(signs@v/n)>=observed-1e-12))

def holm(rows):
    indexed=sorted(enumerate(rows),key=lambda pair:pair[1]['p_raw'])
    previous=0.;m=len(rows)
    for rank,(_,row) in enumerate(indexed):
        previous=max(previous,min(1.,(m-rank)*row['p_raw']));row['p_holm']=previous

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(out,p):
    records=[json.loads(f.read_text()) for f in sorted((out/'runs').glob('*.json'))]
    by={(r['method'],r['seed'],r['scenario']):r for r in records}
    summary=[]
    for scenario in SCENARIOS:
        for method in METHODS:
            for metric in METRICS:
                vals=[by[method,s,scenario]['metrics'][metric] for s in p['seeds']]
                summary.append(dict(scenario=scenario,method=method,metric=metric,mean=float(np.mean(vals)),ci95_half_width=ci95(vals),n=len(vals)))
    write_csv(out/'tables/summary.csv',summary)
    # Prespecified family: proposed versus every comparator, each scenario and
    # delay/energy/completion/cost. Holm adjustment across this entire family.
    comparisons=[]
    for scenario in SCENARIOS:
        for other in METHODS:
            if other=='proposed':continue
            for metric in METRICS[:4]:
                vals=[by['proposed',s,scenario]['metrics'][metric]-by[other,s,scenario]['metrics'][metric] for s in p['seeds']]
                primary=(scenario=='nominal' and other in ('fmadrl','predictive_observed') and metric=='cost') or (scenario=='corrupted_twin' and other=='ungated' and metric in ('cost','delay'))
                comparisons.append(dict(family='primary' if primary else 'exploratory',scenario=scenario,comparator=other,metric=metric,mean_difference=float(np.mean(vals)),ci95_half_width=ci95(vals),n=len(vals),p_raw=paired_p(vals)))
    for family in ('primary','exploratory'):holm([r for r in comparisons if r['family']==family])
    write_csv(out/'tables/paired_comparisons.csv',comparisons)
    mismatches=[];violations=[]
    for scenario in SCENARIOS:
        for seed in p['seeds']:
            runs=[by[m,seed,scenario] for m in METHODS]
            # Method-specific twin corruption cannot alter the physical workload.
            if len({r['evaluation_task_trace_hash'] for r in runs})!=1 or len({r['metrics']['generated_tasks'] for r in runs})!=1:
                mismatches.append([scenario,seed])
            for r in runs:
                if r['metrics']['cpu_constraint_violations'] or r['metrics']['bandwidth_constraint_violations']:violations.append([scenario,seed,r['method']])
    for seed in p['seeds']:
        runs=[by[m,seed,'nominal'] for m in METHODS]
        for key in ('warmup_trace','training_trace','warmup_tasks','training_tasks'):
            if len({r['overhead'][key] for r in runs})!=1:mismatches.append(['training_'+key,seed])
    audit=dict(cases=len(records),workload_mismatches=mismatches,resource_violations=violations,
               selection_seed_overlap=sorted(set(p['seeds'])&set(p['tune_seeds'])))
    save(out/'audit.json',audit)
    if mismatches or violations or audit['selection_seed_overlap']:raise RuntimeError(f'Invalid audit: {audit}')
    from .publication_report import generate
    generate(out,summary,comparisons,records,p)

def run(profile,workers):
    p=profiles(profile);out=ROOT/'outputs'/profile;out.mkdir(parents=True,exist_ok=True)
    code_hash=hashlib.sha256(b''.join(f.read_bytes() for f in sorted((ROOT/'dt_fmadrl').glob('*.py')))).hexdigest()
    manifest=dict(profile=profile,protocol=p,code_sha256=code_hash,methods=METHODS,scenarios=SCENARIOS,
        tuning={m:candidates(m) for m in ('proposed','fmadrl','dqn','mappo')},
        selection='Minimize seed-mean cost + 2*(1-completion), nominal validation only; ties choose candidate order.',
        primary_contrasts='Four hypotheses: nominal cost versus FMADRL/observation Predictive; corrupted-twin cost/delay versus ungated. One Holm family. Remaining 196 comparisons form a separate exploratory Holm family.',
        test='Exact two-sided paired sign-flip; descriptive Student-t 95% CIs are unadjusted.',
        interpretation='No superiority assumed; all negative and null results retained; OOD scenarios use nominal-trained frozen actors.',
        runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()))
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=json.loads(json.dumps(manifest)):
        raise RuntimeError('Plan/code differs from existing output; use another revision or output profile.')
    save(manifest_path,manifest)
    jobs=[(m,k,s,p,str(out)) for m in ('proposed','fmadrl','dqn','mappo') for k in range(3) for s in p['tune_seeds']]
    tuning=parallel(jobs,tune_job,workers,'Validation')
    selection={};scores={}
    for method in ('proposed','fmadrl','dqn','mappo'):
        means=[float(np.mean([r['score'] for r in tuning if r['method']==method and r['candidate']==k])) for k in range(3)]
        winner=int(np.argmin(means));selection[method]=candidates(method)[winner];scores[method]=dict(candidate_scores=means,selected=winner)
    save(out/'selection.json',dict(seed_partition=p,settings=selection,scores=scores))
    jobs=[(m,s,p,str(out),selection) for m in METHODS for s in p['seeds']]
    parallel(jobs,confirm_job,workers,'Confirmation')
    summarize(out,p)
    print(f'Completed study: {out}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=('pilot','extended'),default='extended');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();run(args.profile,args.workers)
