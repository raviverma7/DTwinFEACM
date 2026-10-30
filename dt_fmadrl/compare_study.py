from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
from dataclasses import asdict
from statistics import mean, stdev

from .baselines import DDPGTrainer, DQNTrainer
from .config import load_config
from .federated import FederatedTrainer

METHODS=("random","dqn","ddpg","fmadrl","dt_fmadrl")
METRICS=("reward","delay","energy","completion_rate","cost")

from .statistics import ci95 as ci

def run(config_path,output,seeds,rounds,episodes):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rows=[];hist={}
    for seed in seeds:
        cfg=load_config(config_path);cfg.seed=seed;cfg.learning.evaluation_episodes=episodes
        random=FederatedTrainer(cfg,False); rows.append({"seed":seed,"method":"random",**random.evaluate(policy="random")})
        for name,trainer in (("dqn",DQNTrainer(cfg)),("ddpg",DDPGTrainer(cfg))):
            h=trainer.train(rounds);rows.append({"seed":seed,"method":name,**trainer.evaluate(episodes)});hist[f"{seed}_{name}"]=[r.__dict__ for r in h]
        for name,twin in (("fmadrl",False),("dt_fmadrl",True)):
            trainer=FederatedTrainer(cfg,twin);h=trainer.train(rounds);rows.append({"seed":seed,"method":name,**trainer.evaluate()})
            hist[f"{seed}_{name}"]=[r.__dict__ for r in h]
    with (out/'per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=('seed','method',*METRICS));w.writeheader();w.writerows(rows)
    (out/'metadata.json').write_text(json.dumps(dict(formulation='revised_equations_3_27',config=asdict(cfg),seeds=seeds,rounds=rounds,evaluation_episodes=episodes,checkpoint='final round',evaluation_seed_offset=100000,failure_treatment='all tasks in all metrics',ci95='Student-t interval half-width across independent seeds'),indent=2),encoding='utf-8')
    summary={m:{k:{"mean":mean(x[k] for x in rows if x['method']==m),"std":stdev(x[k] for x in rows if x['method']==m) if len(seeds)>1 else None,"ci95":ci([x[k] for x in rows if x['method']==m])} for k in METRICS} for m in METHODS}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(out/'histories.json').write_text(json.dumps(hist),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/research.json');p.add_argument('--output',required=True);p.add_argument('--seeds',required=True);p.add_argument('--rounds',type=int,default=120);p.add_argument('--episodes',type=int,default=10);a=p.parse_args()
    print(json.dumps(run(a.config,a.output,[int(x) for x in a.seeds.split(',')],a.rounds,a.episodes),indent=2))
if __name__=='__main__':main()
