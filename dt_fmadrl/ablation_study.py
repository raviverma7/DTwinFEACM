from __future__ import annotations
import argparse,json
from pathlib import Path
from statistics import mean
from .statistics import ci95
from .config import load_config
from .federated import FederatedTrainer

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='config/research.json');p.add_argument('--episodes',type=int,default=3);p.add_argument('--output',default='outputs/ablations');p.add_argument('--seeds',default='11,22,33');p.add_argument('--rounds',type=int,default=40);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 variants={
  'fmadrl':{'twin':False},
  'dt_no_pretrain':{'twin':True,'pretrain_steps':0},
  'dt_no_confidence':{'twin':True,'gating':False},
  'dt_augmentation_only':{'twin':True,'ranking':False},
  'dt_ranking_only':{'twin':True,'augmentation':False},
  'dt_only':{'twin':True,'augmentation':False,'actor_guidance':False},
  'dt_no_model_guidance':{'twin':True,'model_bias':0.0,'energy_bias':0.0},
  'dt_full':{'twin':True},
 }
 rows=[]
 for seed in [int(x) for x in a.seeds.split(',')]:
  for name,opt in variants.items():
   cfg=load_config(a.config);cfg.seed=seed
   for k,v in opt.items():
    if k!='twin':setattr(cfg.digital_twin,k,v)
   trainer=FederatedTrainer(cfg,opt['twin']);trainer.train(a.rounds);ev=trainer.evaluate(a.episodes)
   rows.append({'seed':seed,'variant':name,**ev})
 summary={name:{k:mean(r[k] for r in rows if r['variant']==name) for k in ('reward','delay','energy','completion_rate','cost')} for name in variants}
 (out/'ablation_intervals.json').write_text(json.dumps({name:{k:ci95([r[k] for r in rows if r['variant']==name]) for k in ('reward','delay','energy','completion_rate','cost')} for name in variants},indent=2));
 (out/'ablation_rows.json').write_text(json.dumps(rows,indent=2));(out/'ablation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
