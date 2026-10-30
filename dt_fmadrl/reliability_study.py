"""Gated/ungated paired stress comparisons on common evaluation realizations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
from .config import load_config
from .federated import FederatedTrainer
from .statistics import ci95

SCENARIOS = ('nominal','delayed_sync','workload_shift','short_history','outage','prediction_bias')

def run(config, output, seeds, rounds, episodes):
    rows = []; diagnostics = []; settings = {}
    for seed in seeds:
        for scenario in SCENARIOS:
            for gating in (True,False):
                cfg = load_config(config); cfg.seed = seed; cfg.digital_twin.gating = gating
                if scenario=='short_history': cfg.digital_twin.history = 128
                trainer = FederatedTrainer(cfg,True); trainer.train(rounds)
                if scenario=='delayed_sync': cfg.digital_twin.synchronization_interval = 5
                if scenario=='workload_shift':
                    cfg.environment.cycles_per_bit_min *= 2; cfg.environment.cycles_per_bit_max *= 2
                if scenario=='outage': cfg.environment.link_availability = .5
                if scenario=='prediction_bias': cfg.digital_twin.prediction_bias = .5
                ev = trainer.evaluate(episodes)
                rows.append(dict(seed=seed,scenario=scenario,gating=gating,**ev))
                diagnostics.append(dict(seed=seed,scenario=scenario,gating=gating,**trainer.diagnostics()))
                settings[f'{scenario}_{gating}'] = asdict(cfg)
    summary = {}
    for scenario in SCENARIOS:
        selected = [r for r in rows if r['scenario']==scenario]
        summary[scenario] = {}
        for metric in ('reward','delay','energy','completion_rate','cost'):
            differences = [next(r[metric] for r in selected if r['seed']==s and r['gating'])-next(r[metric] for r in selected if r['seed']==s and not r['gating']) for s in seeds]
            summary[scenario][metric] = dict(mean_gated_minus_ungated=mean(differences),sample_sd=stdev(differences) if len(differences)>1 else None,ci95_half_width=ci95(differences))
    out = Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'stress_results.json').write_text(json.dumps(dict(rows=rows,diagnostics=diagnostics,summary=summary,settings=settings,rounds=rounds,evaluation_episodes=episodes,failure_treatment='all generated tasks',evaluation_adaptation='online twins; frozen actors'),indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/research.json');p.add_argument('--output',default='outputs/revised_stress');p.add_argument('--seeds',default='11,22,33,44,55');p.add_argument('--rounds',type=int,default=120);p.add_argument('--episodes',type=int,default=15);a=p.parse_args()
    print(json.dumps(run(a.config,a.output,[int(v) for v in a.seeds.split(',')],a.rounds,a.episodes),indent=2))
