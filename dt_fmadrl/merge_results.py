from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from statistics import mean,stdev

METHODS=('random','dqn','ddpg','fmadrl','dt_fmadrl');METRICS=('reward','delay','energy','completion_rate','cost')
from .statistics import ci95 as ci
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 rows=[];hist={}
 for folder in map(Path,a.inputs):
  with (folder/'per_seed.csv').open(encoding='utf-8') as f:
   for r in csv.DictReader(f):
    if r['method'] in METHODS:rows.append({'seed':int(r['seed']),'method':r['method'],**{k:float(r[k]) for k in METRICS}})
  hist.update({k:v for k,v in json.loads((folder/'histories.json').read_text()).items()})
 with (out/'per_seed.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=('seed','method',*METRICS));w.writeheader();w.writerows(rows)
 summary={m:{k:{'mean':mean(r[k] for r in rows if r['method']==m),'std':stdev(r[k] for r in rows if r['method']==m),'ci95':ci([r[k] for r in rows if r['method']==m])} for k in METRICS} for m in METHODS if any(r['method']==m for r in rows)}
 paired={}
 for k in METRICS:
  b={r['seed']:r[k] for r in rows if r['method']=='fmadrl'};d={r['seed']:r[k] for r in rows if r['method']=='dt_fmadrl'};v=[d[s]-b[s] for s in sorted(b)]
  paired[k]={'mean_dt_minus_fmadrl':mean(v),'ci95':ci(v),'favorable_seeds':sum(x>0 for x in v) if k in ('reward','completion_rate') else sum(x<0 for x in v)}
 summary['paired_dt_effect']=paired
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'histories.json').write_text(json.dumps(hist))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
