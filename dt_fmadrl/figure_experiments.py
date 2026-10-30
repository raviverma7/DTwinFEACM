from __future__ import annotations
import argparse,json
from pathlib import Path
from .config import load_config
from .federated import FederatedTrainer

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--rounds',type=int,default=20);p.add_argument('--lr',type=float);p.add_argument('--devices',type=int);a=p.parse_args()
 cfg=load_config('config/research.json');cfg.seed=a.seed
 if a.lr is not None:cfg.learning.actor_lr=a.lr
 if a.devices is not None:
  cfg.environment.num_mec=max(1,a.devices//20);cfg.environment.vehicles_per_mec=20
  cfg.environment.area_m=500.0*(cfg.environment.num_mec**0.5)
  # Fixed bounded twin memory keeps the 500-device experiment computationally comparable.
  cfg.digital_twin.history=256;cfg.digital_twin.pretrain_steps=100
 trainer=FederatedTrainer(cfg,True);history=trainer.train(a.rounds);evaluation=trainer.evaluate(5);detailed=trainer.evaluate_detailed(2)
 result={'seed':a.seed,'learning_rate':a.lr,'devices':trainer.env.n,'history':[x.__dict__ for x in history],'evaluation':evaluation,'detailed':detailed}
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(json.dumps(evaluation))
if __name__=='__main__':main()
