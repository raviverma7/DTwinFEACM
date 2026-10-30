"""Seed-level statistical tables and publication export figures from saved runs."""
import csv
import json
import os
from pathlib import Path
from statistics import mean,stdev
import sys
_LOCAL=Path(__file__).resolve().parents[1]/'.runtime'
if _LOCAL.exists():sys.path.insert(0,str(_LOCAL))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.runtime'/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .statistics import ci95

LABELS=dict(fmadrl='FMADRL',dqn='DQN',mappo='MAPPO',random='Random',predictive='Predictive',proposed='DT-assisted',
            augmentation_only='Augmentation only',ranking_only='Ranking only',dt_only='DT ranking only',no_pretraining='No pretraining',ungated='Ungated')
COLORS=dict(fmadrl='#0072B2',dqn='#E69F00',mappo='#009E73',random='#999999',predictive='#CC79A7',proposed='#D55E00',ungated='#0072B2')
BASE_METRICS=('delay','energy','completion_rate','cost')
UNITS=dict(delay='Delay (s)',energy='System energy (J/task)',completion_rate='Completion (%)',cost='Normalized cost',
           queue_length='CPU backlog (equivalent tasks)',deadline_violation_rate='Deadline violations (%)',decision_ms_p95='Joint decision p95 (ms)',decision_ms_mean='Joint decision mean (ms)')


def stats(values):
    values=[float(v) for v in values]
    return dict(mean=mean(values),sample_sd=stdev(values) if len(values)>1 else None,ci95_half_width=ci95(values),independent_seeds=len(values)) if values else dict(mean=None,sample_sd=None,ci95_half_width=None,independent_seeds=0)

def factor(metric):return 100. if metric in ('completion_rate','deadline_violation_rate') else 1.

def save(fig,folder,name,title,preset):
    # Keep captions below panels; no overall heading or bottom legend.
    for legend in list(fig.legends):legend.remove()
    if fig._suptitle is not None:fig._suptitle.remove();fig._suptitle=None
    visible=[ax for ax in fig.axes if ax.get_visible()]
    for index,ax in enumerate(visible):
        context=ax.get_title();ax.set_title('')
        caption=f'({chr(97+index)})'+(f' {context}' if context else '')
        xlabel=ax.get_xlabel()
        ax.set_xlabel((xlabel+'\n\n' if xlabel else '')+caption)
        handles,labels=ax.get_legend_handles_labels()
        if ax.get_legend() is not None:ax.get_legend().remove()
        if handles:ax.legend(handles,labels,loc='lower left',bbox_to_anchor=(0,1.02),borderaxespad=0,fontsize=8,ncol=2 if len(labels)>3 else 1,framealpha=.9)
    fig.tight_layout(pad=1.2,h_pad=2.4,w_pad=2.0)
    for extension in ('png','pdf','svg'):fig.savefig(folder/f'{name}.{extension}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def line_panels(records,experiment,metrics,xlabel,folder,name,title,preset):
    selected=[r for r in records if r['case']['experiment']==experiment]
    if not selected:return
    columns=2 if len(metrics)>1 else 1;rows=int(np.ceil(len(metrics)/columns));fig,axes=plt.subplots(rows,columns,figsize=(11,3.7*rows),squeeze=False)
    methods=list(dict.fromkeys(r['case']['method'] for r in selected))
    for ax,metric in zip(axes.flat,metrics):
        for method in methods:
            points=[]
            for axis in sorted(set(r['case']['axis'] for r in selected)):
                values=[r['metrics'][metric]*factor(metric) for r in selected if r['case']['method']==method and r['case']['axis']==axis]
                if values:points.append((axis,stats(values)))
            xx=[x for x,_ in points];yy=np.array([s['mean'] for _,s in points]);err=np.array([s['ci95_half_width'] or 0 for _,s in points])
            ax.plot(xx,yy,'o-',label=LABELS.get(method,method),color=COLORS.get(method),linewidth=1.5,markersize=4)
            ax.fill_between(xx,yy-err,yy+err,color=COLORS.get(method),alpha=.12)
        ax.set(xlabel=xlabel,ylabel=UNITS.get(metric,metric));ax.grid(alpha=.2)
        if metric=='completion_rate':ax.set_ylim(0,105)
    for ax in list(axes.flat)[len(metrics):]:ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.subplots_adjust(bottom=.15);save(fig,folder,name,title,preset)

def uncertainty_curves(records,folder,preset):
    chosen=[r for r in records if r['case']['experiment']=='main'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('fmadrl','dqn','mappo','proposed'):
        runs=[r for r in chosen if r['case']['method']==method]
        if not runs:continue
        rounds=min(len(r['history']) for r in runs)
        slots=runs[0]['config']['learning']['local_steps'];warmup=runs[0]['overhead']['warmup_slot_budget']
        x=warmup+np.arange(1,rounds+1)*slots
        for ax,metric in zip(axes,('reward','completion_rate')):
            points=[stats([r['history'][t][metric]*factor(metric) for r in runs]) for t in range(rounds)]
            y=np.array([s['mean'] for s in points]);error=np.array([s['ci95_half_width'] or 0 for s in points])
            ax.plot(x,y,label=LABELS[method],color=COLORS[method]);ax.fill_between(x,y-error,y+error,color=COLORS[method],alpha=.15)
            ax.set(xlabel='Shared physical interaction slots (warmup included)',ylabel='Training reward' if metric=='reward' else 'Training completion (%)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);save(fig,folder,'figure_3_learning','Matched-budget training curves; 95% seed intervals',preset)

def stress_figures(records,folder,preset):
    noise=[r for r in records if r['case']['experiment']=='stress_noise']
    if noise:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for method in ('proposed','ungated'):
            points=[]
            for level in sorted(set(r['case']['axis'] for r in noise)):
                rows=[r for r in noise if r['case']['method']==method and r['case']['axis']==level]
                baseline={r['seed']:r for r in noise if r['case']['method']==method and r['case']['axis']==0}
                errors=[mean(p['state_mae'] for p in r['predictions']) for r in rows if r['predictions']]
                degradation=[(baseline[r['seed']]['metrics']['completion_rate']-r['metrics']['completion_rate'])*100 for r in rows if r['seed'] in baseline]
                points.append((stats(errors),stats(degradation),stats([r['metrics']['gate_zero_rate']*100 for r in rows])))
            x=[p[0]['mean'] for p in points]
            for ax,idx in zip(axes,(1,2)):
                y=[p[idx]['mean'] for p in points];error=[p[idx]['ci95_half_width'] or 0 for p in points]
                ax.errorbar(x,y,yerr=error,fmt='o-',capsize=3,color=COLORS[method],label=LABELS[method])
                ax.set_xlabel('Observed normalized next-state MAE');ax.grid(alpha=.2)
        axes[0].set_ylabel('Completion loss versus zero noise (percentage points)');axes[1].set_ylabel('Zero-guidance decisions (%)')
        axes[0].legend();save(fig,folder,'figure_4_gating','Performance degradation versus prediction degradation',preset)
    stress=[r for r in records if r['case']['experiment']=='stress_workload']
    if stress:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        max_level=max(r['case']['axis'] for r in stress)
        for method in ('proposed','ungated'):
            rows=[r for r in stress if r['case']['method']==method and r['case']['axis']==max_level]
            maxslot=max(p['slot'] for r in rows for p in r['curves'])
            window=max(1,(maxslot+1)//12)
            for ax,metric in zip(axes,('completion_rate','gate_zero_rate')):
                points=[]
                for start in range(0,maxslot+1,window):
                    per_seed=[]
                    for r in rows:
                        events=[p for p in r['curves'] if start<=p['slot']<start+window]
                        total=sum(e['generated_tasks'] for e in events)
                        if total:per_seed.append(sum(e[metric]*e['generated_tasks'] for e in events)/total*100)
                    if per_seed:points.append((start,stats(per_seed)))
                ax.errorbar([x for x,_ in points],[s['mean'] for _,s in points],yerr=[s['ci95_half_width'] or 0 for _,s in points],fmt='o-',label=LABELS[method],color=COLORS[method],capsize=2)
                ax.axvline((maxslot+1)//3,color='black',linestyle=':',label='Workload change' if method=='proposed' else None)
                ax.set(xlabel='Evaluation slot',ylabel='Completion (%)' if metric=='completion_rate' else 'Zero-guidance decisions (%)');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);save(fig,folder,'supplement_recovery','Recovery after a sustained workload shift',preset)

def prediction_figure(records,folder,preset):
    selected=[r for r in records if r['case']['experiment']=='main' and r['case']['method']=='proposed']
    if not selected:return
    fig,axes=plt.subplots(3,2,figsize=(11,12));modes=[mode for mode in ("local","edge","cloud","peer","edge_cloud_30","edge_cloud_50","edge_cloud_70") if any(p["mode"]==mode for r in selected for p in r["probes"])]
    for ax,metric,label in zip(list(axes.flat)[:4],('state_mae','delay_mae','energy_mae','reward_mae'),('Normalized state MAE','Delay MAE (s)','Energy MAE (J)','Reward MAE')):
        points=[]
        for mode in modes:
            values=[mean(p[metric] for p in r['probes'] if p['mode']==mode) for r in selected if any(p['mode']==mode for p in r['probes'])]
            points.append(stats(values))
        ax.bar(range(len(modes)),[s['mean'] for s in points],yerr=[s['ci95_half_width'] or 0 for s in points],color='#0072B2',capsize=2)
        ax.set_xticks(range(len(modes)),[m.replace('edge_cloud_','col ') for m in modes],rotation=45,ha='right',fontsize=8);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    ax=axes.flat[4]
    ranking=stats([mean(p['ranking_accuracy'] for p in r['ranking']) for r in selected if r['ranking']]);regret=stats([mean(p['decision_regret'] for p in r['ranking']) for r in selected if r['ranking']])
    ax.bar([0],[ranking['mean']*100],yerr=[(ranking['ci95_half_width'] or 0)*100],color='#009E73',capsize=3);ax.set(xticks=[0],xticklabels=['Best predicted reward'],ylabel='Top-action agreement (%)');ax.set_title(f'Mean reward regret: {regret["mean"]:.3f}',fontsize=10)
    ax=axes.flat[5]
    reliability=[r for r in records if r['case']['experiment']=='stress_noise' and r['case']['method']=='proposed'] or selected
    bins=np.linspace(0,1,6);points=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        per_seed=[]
        for seed in sorted(set(r['seed'] for r in reliability)):
            errors=[p['state_mae'] for r in reliability if r['seed']==seed for p in r['predictions'] if lo<=p['score'] and (p['score']<hi or hi==1)]
            if errors:per_seed.append(mean(errors))
        if per_seed:points.append(((lo+hi)/2,stats(per_seed)))
    ax.errorbar([x for x,_ in points],[s['mean'] for _,s in points],yerr=[s['ci95_half_width'] or 0 for _,s in points],fmt='o-',capsize=3,color='#D55E00')
    ax.set(xlabel='Score recorded before execution (fixed bins)',ylabel='Later normalized state MAE');ax.grid(alpha=.2)
    save(fig,folder,'figure_5_prediction','Held-out action probes and predecision-score reliability',preset)

def heterogeneous_figure(records,folder,preset):
    selected=[r for r in records if r['case']['experiment']=='heterogeneity']
    if not selected:return
    fig,axes=plt.subplots(3,2,figsize=(11,12));groups=('low','medium','high')
    available={r['case']['method'] for r in selected}
    methods=tuple(m for m in ('fmadrl','dqn','mappo','predictive','proposed') if m in available)
    for row,mobile in enumerate((0,1)):
        for col,metric in enumerate(('delay','completion_rate','decision_ms_p95')):
            ax=axes[col,row]
            if col<2:
                for mi,method in enumerate(methods):
                    runs=[r for r in selected if r['case']['axis']==mobile and r['case']['method']==method]
                    points=[stats([r['groups'][g][metric]*factor(metric) for r in runs if g in r['groups'] and metric in r['groups'][g]]) for g in groups]
                    ax.bar(np.arange(3)+(mi-(len(methods)-1)/2)*.15,[s['mean'] for s in points],width=.15,yerr=[s['ci95_half_width'] or 0 for s in points],label=LABELS[method],color=COLORS[method],capsize=1)
                ax.set_xticks(range(3),groups);ax.set_ylabel(UNITS[metric])
            else:
                points=[stats([r['metrics'][metric] for r in selected if r['case']['axis']==mobile and r['case']['method']==method]) for method in methods]
                ax.bar(range(len(methods)),[s['mean'] for s in points],yerr=[s['ci95_half_width'] or 0 for s in points],color=[COLORS[m] for m in methods],capsize=2)
                ax.set_xticks(range(len(methods)),[LABELS[m] for m in methods],rotation=30,ha='right',fontsize=8);ax.set_ylabel(UNITS[metric])
            ax.set_title('Mobile' if mobile else 'Stationary',fontsize=11);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='lower center',fontsize=9)
    save(fig,folder,'figure_6_heterogeneity','Device-group service and measured joint decision latency',preset)

def write_table(folder,name,rows,fields):
    with (folder/f'{name}.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    # Full precision statistics stay in JSON/CSV; Markdown is a readable summary.
    readable=[]
    for row in rows:
        readable.append({k:f'{v:.4g}' if isinstance(v,float) else '' if v is None else str(v) for k,v in row.items()})
    lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    lines+=['| '+' | '.join(row.get(k,'') for k in fields)+' |' for row in readable]
    (folder/f'{name}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def generate(output):
    out=Path(output);manifest=json.loads((out/'manifest.json').read_text());preset=manifest['preset']
    index=json.loads((out/'index.json').read_text());records=[json.loads((out/x['path']).read_text()) for x in index]
    figures=out/'figures';tables=out/'tables';figures.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    summary=[]
    keys=sorted(set((r['case']['experiment'],str(r['case']['axis']),r['case']['method']) for r in records))
    for experiment,axis,method in keys:
        selected=[r for r in records if r['case']['experiment']==experiment and str(r['case']['axis'])==axis and r['case']['method']==method]
        for metric in selected[0]['metrics']:
            values=[r['metrics'][metric] for r in selected]
            summary.append(dict(experiment=experiment,axis=axis,method=method,metric=metric,**stats(values)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for experiment,name in (('main','table_1_main'),('ablation','table_2_ablation')):
        rows=[]
        for r in summary:
            if r['experiment']==experiment and r['metric'] in (*BASE_METRICS,'decision_ms_mean','decision_ms_p95','decision_ms_p99'):
                rows.append({k:r[k] for k in ('method','metric','mean','sample_sd','ci95_half_width','independent_seeds')})
        if rows:write_table(tables,name,rows,('method','metric','mean','sample_sd','ci95_half_width','independent_seeds'))
    overhead_rows=[]
    for experiment,axis,method in keys:
        if not experiment.startswith('federation_'):continue
        selected=[r for r in records if r['case']['experiment']==experiment and str(r['case']['axis'])==axis]
        for metric in (*BASE_METRICS,'decision_ms_mean','decision_ms_p95','training_seconds','peak_rss_bytes','federation_bytes','synchronization_bytes','dt_refit_seconds','training_dt_refit_seconds','evaluation_dt_refit_seconds','training_synchronization_bytes','evaluation_synchronization_bytes','peak_rss_delta_bytes'):
            values=[r['metrics'][metric] if metric in r['metrics'] else r['overhead'][metric] for r in selected]
            overhead_rows.append(dict(workload=experiment.replace('federation_',''),variant=axis,metric=metric,**stats(values)))
    if overhead_rows:write_table(tables,'table_3_federation',overhead_rows,('workload','variant','metric','mean','sample_sd','ci95_half_width','independent_seeds'))
    group_rows=[];prediction_rows=[];energy_rows=[];paired=[]
    for group in ('low','medium','high'):
        for mobile in (0,1):
            for method in LABELS:
                selected=[r for r in records if r['case']['experiment']=='heterogeneity' and r['case']['axis']==mobile and r['case']['method']==method]
                for metric in BASE_METRICS:
                    values=[r['groups'][group][metric] for r in selected if group in r['groups'] and metric in r['groups'][group]]
                    if values:group_rows.append(dict(mobility='mobile' if mobile else 'stationary',group=group,method=method,metric=metric,**stats(values)))
    for mode in ('local','edge','cloud','peer','edge_cloud_30','edge_cloud_50','edge_cloud_70'):
        selected=[r for r in records if r['case']['experiment']=='main' and r['case']['method']=='proposed']
        for metric in ('state_mae','delay_mae','energy_mae','reward_mae'):
            values=[mean(p[metric] for p in r['probes'] if p['mode']==mode) for r in selected if any(p['mode']==mode for p in r['probes'])]
            if values:prediction_rows.append(dict(mode=mode,metric=metric,probe_count=sum(p['mode']==mode for r in selected for p in r['probes']),**stats(values)))
        for component in ('source_cpu','source_radio','remote_cpu','remote_radio','backhaul','coordination','dt'):
            values=[r['energy_breakdown'][mode][component] for r in selected if mode in r['energy_breakdown'] and r['modes'].get(mode,{}).get('generated_tasks',0)]
            if values:energy_rows.append(dict(mode=mode,component=component,**stats(values)))
    if group_rows:write_table(tables,'supplement_device_groups',group_rows,('mobility','group','method','metric','mean','sample_sd','ci95_half_width','independent_seeds'))
    reliability_rows=[]
    reliability=[r for r in records if r['case']['experiment']=='stress_noise' and r['case']['method']=='proposed']
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        values=[];sample_count=0
        for seed in sorted(set(r['seed'] for r in reliability)):
            errors=[p['state_mae'] for r in reliability if r['seed']==seed for p in r['predictions'] if lo<=p['score'] and (p['score']<hi or hi==1)]
            sample_count+=len(errors)
            if errors:values.append(mean(errors))
        reliability_rows.append(dict(score_min=float(lo),score_max=float(hi),sample_count=sample_count,**stats(values)))
    if reliability:write_table(tables,'supplement_score_reliability',reliability_rows,('score_min','score_max','sample_count','mean','sample_sd','ci95_half_width','independent_seeds'))
    rank_rows=[]
    selected=[r for r in records if r['case']['experiment']=='main' and r['case']['method']=='proposed']
    for metric in ('ranking_accuracy','decision_regret','chosen_controller_regret'):
        values=[mean(p[metric] for p in r['ranking']) for r in selected if r['ranking']]
        if values:rank_rows.append(dict(metric=metric,**stats(values)))
    if rank_rows:write_table(tables,'supplement_ranking',rank_rows,('metric','mean','sample_sd','ci95_half_width','independent_seeds'))
    if prediction_rows:write_table(tables,'supplement_prediction_modes',prediction_rows,('mode','metric','probe_count','mean','sample_sd','ci95_half_width','independent_seeds'))
    if energy_rows:write_table(tables,'supplement_energy_modes',energy_rows,('mode','component','mean','sample_sd','ci95_half_width','independent_seeds'))
    stress_experiments=sorted(set(r['case']['experiment'] for r in records if r['case']['experiment'].startswith('stress_')))
    for experiment in stress_experiments:
        for axis in sorted(set(r['case']['axis'] for r in records if r['case']['experiment']==experiment)):
            gated={r['seed']:r for r in records if r['case']['experiment']==experiment and r['case']['axis']==axis and r['case']['method']=='proposed'}
            ungated={r['seed']:r for r in records if r['case']['experiment']==experiment and r['case']['axis']==axis and r['case']['method']=='ungated'}
            for metric in (*BASE_METRICS,'gate_zero_rate'):
                values=[gated[s]['metrics'][metric]-ungated[s]['metrics'][metric] for s in sorted(set(gated)&set(ungated))]
                if values:paired.append(dict(stress=experiment,level=axis,metric=metric,**stats(values)))
    if paired:write_table(tables,'supplement_gated_minus_ungated',paired,('stress','level','metric','mean','sample_sd','ci95_half_width','independent_seeds'))
    line_panels(records,'population',BASE_METRICS,'Number of devices',figures,'figure_1_population','Performance versus population; 95% seed intervals',preset)
    line_panels(records,'arrival',BASE_METRICS,'Arrival probability per device-slot',figures,'figure_2_arrival','Performance versus task-arrival rate; 95% seed intervals',preset)
    line_panels(records,'arrival',('queue_length',),'Arrival probability per device-slot',figures,'supplement_queue','Outstanding CPU workload',preset)
    line_panels(records,'workload',('delay','energy','deadline_violation_rate'),'Computational intensity multiplier',figures,'supplement_workload','Workload sensitivity',preset)
    line_panels(records,'edge_capacity',('delay','completion_rate','cost'),'Edge CPU capacity multiplier',figures,'supplement_edge_capacity','Edge capacity sensitivity',preset)
    line_panels(records,'backhaul_capacity',('delay','completion_rate','cost'),'Backhaul capacity multiplier',figures,'supplement_backhaul_capacity','Backhaul capacity sensitivity',preset)
    uncertainty_curves(records,figures,preset);stress_figures(records,figures,preset);prediction_figure(records,figures,preset);heterogeneous_figure(records,figures,preset)
    cpu=sum(r['metrics']['cpu_constraint_violations'] for r in records);bandwidth=sum(r['metrics']['bandwidth_constraint_violations'] for r in records)
    matched={}
    for r in records:
        key=(r['seed'],r['case']['experiment'],str(r['case']['axis']))
        matched.setdefault(key,set()).add((r['overhead']['warmup_generated_tasks'],r['overhead']['training_generated_tasks'],r['metrics']['generated_tasks'],r['overhead']['warmup_task_trace_hash'],r['overhead']['training_task_trace_hash'],r['evaluation_task_trace_hash']))
    checks=dict(cpu_constraint_violations=cpu,bandwidth_constraint_violations=bandwidth,unmatched_generated_task_budgets=sum(len(v)>1 for v in matched.values()))
    (out/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    note='This smoke package validates execution and reporting. Its short training budgets and small populations do not support journal performance claims.' if preset=='smoke' else 'These are simulator results under the documented assumptions; fixed baseline hyperparameters require competitiveness validation.'
    text=f'# Journal evaluation package\n\n{note}\n\nIndependent seeds: {manifest["seeds"]}. Training: {manifest["rounds"]} rounds, {manifest["base_config"]["learning"]["local_steps"]} slots/round; common warmup: {manifest["warmup_slots"]} slots. Evaluation: {manifest["evaluation_episodes"]} episodes of {manifest["base_config"]["environment"]["episode_steps"]} slots.\n\nFigures are exported as PNG, PDF and SVG. Tables include mean, sample SD and Student-t 95% interval half-width across independent seeds. Task samples are not independent policy replications. Completion rates in tables are fractions; figure axes convert them to percentages. Normalized cost summarizes delay and energy and is not independent evidence.\n\nCompletion uses all generated tasks. Failed tasks enter delay and energy at their modeled full-execution values; idle device slots are excluded. Both source and system-energy measurements are explicit in raw runs.\n\nLatency is measured for the joint controller call, including DT augmentation and ranking. Oracle probes, synchronization and simulator stepping are outside decision latency. Separate refit/training times and sampled process RSS are reported. Numeric communication bytes exclude protocol/security overhead. DT energy remains a modeled coefficient, not a hardware measurement.\n\nPrediction mode tables use held-out single-device interventions with the other joint actions and stochastic realization fixed. Every feasible mode is probed, including rarely selected modes. Rank accuracy is for best predicted reward; regret is against the best feasible unilateral action under the same simulator. Reliability bins use scores recorded before physical execution and errors observed afterward; sparse bins/modes report their available seed and probe counts.\n\nMAPPO uses parameter sharing, a neural centralized critic, clipped PPO updates and GAE, following [PPO](https://arxiv.org/abs/1707.06347) and [MAPPO](https://arxiv.org/abs/2103.01955). This NumPy implementation is a baseline, not published-benchmark reproduction. It requires centralized training-state sharing; FMADRL exchanges model parameters and DT histories remain domain-local.\n\nResource checks: {cpu} CPU violations; {bandwidth} bandwidth violations. Matched task-budget groups differing across methods: {checks["unmatched_generated_task_budgets"]}. Queue length is equivalent unfinished tasks from CPU cycles, not an event-level task count.\n\nSee `manifest.json` for the full plan/config and `runs/` for raw seed outcomes, recovery curves, decision latency samples, group/mode energy, gate activation and prediction errors. Checkpoints include trusted local training state and are resumable only within this code/config fingerprint.\n'
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    return dict(output=str(out.resolve()),independent_seeds=len(manifest['seeds']),runs=len(records),figures=len(list(figures.glob('*.png'))),tables=len(list(tables.glob('*.csv'))),**checks)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(generate(a.output),indent=2))
