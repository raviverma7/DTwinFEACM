from __future__ import annotations
import argparse,json
from pathlib import Path
from .baselines import DDPGTrainer
from .config import load_config
from .federated import FederatedTrainer

def main():
 p=argparse.ArgumentParser();p.add_argument('--devices',type=int,default=500);p.add_argument('--rounds',type=int,default=10);p.add_argument('--seed',type=int,default=11);p.add_argument('--output',default='outputs/scale500/baselines500.json');a=p.parse_args()
 cfg=load_config('config/research.json');cfg.seed=a.seed;cfg.environment.num_mec=a.devices//20;cfg.environment.vehicles_per_mec=20;cfg.environment.area_m=500*(cfg.environment.num_mec**0.5);cfg.digital_twin.history=256;cfg.digital_twin.pretrain_steps=100
 random=FederatedTrainer(cfg,False);results={'random':random.evaluate(5)}
 ddpg=DDPGTrainer(cfg);ddpg_history=ddpg.train(a.rounds);results['ddpg']=ddpg.evaluate(5)
 fm=FederatedTrainer(cfg,False);fm_history=fm.train(a.rounds);results['fmadrl']=fm.evaluate(5)
 results['histories']={'ddpg':[x.__dict__ for x in ddpg_history],'fmadrl':[x.__dict__ for x in fm_history]}
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
