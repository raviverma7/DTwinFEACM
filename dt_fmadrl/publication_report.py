"""Publication-revision plots, numerical results, and evidence-based claims."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .statistics import ci95

NAMES={'proposed':'Revised DT-FMARL','legacy':'Original DT-FMARL','fmadrl':'FMADRL','dqn':'DQN','mappo':'MAPPO','random':'Random','predictive_full':'Predictive (full information)','predictive_observed':'Predictive (same observations)','augmentation_only':'Augmentation only','ranking_only':'Ranking only','ungated':'Revised ungated'}
TASKS=('delay','energy','completion_rate','cost')
LABELS={'delay':'Delay (s)','energy':'System energy (J/task)','completion_rate':'Completion (%)','cost':'Normalized cost','decision_ms_p95':'Joint decision p95 (ms)','gate_zero_rate':'Zero guidance (%)'}

def generate(out,summary,comparisons,records,p):
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    figs=out/'figures';figs.mkdir(exist_ok=True)
    lookup={(r['scenario'],r['method'],r['metric']):r for r in summary}
    def save(fig,name):
        for k,ax in enumerate(fig.axes):
            ax.set_xlabel(ax.get_xlabel()+f'\n\n({chr(97+k)})')
            if ax.get_legend_handles_labels()[0]:ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),fontsize=7,ncol=2)
        fig.tight_layout(h_pad=3,w_pad=2)
        for ext in ('png','pdf','svg'):fig.savefig(figs/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    def bars(methods,scenario,name):
        fig,axes=plt.subplots(2,2,figsize=(12,10))
        for metric,ax in zip(TASKS,axes.flat):
            scale=100 if metric=='completion_rate' else 1
            vals=[lookup[scenario,m,metric] for m in methods]
            ax.bar(range(len(methods)),[v['mean']*scale for v in vals],yerr=[v['ci95_half_width']*scale for v in vals],capsize=3,color=plt.cm.tab10(np.arange(len(methods))%10))
            ax.set_xticks(range(len(methods)),[NAMES[m] for m in methods],rotation=55,ha='right');ax.set_ylabel(LABELS[metric])
        save(fig,name)
    main=('fmadrl','dqn','mappo','random','predictive_full','predictive_observed','legacy','proposed')
    bars(main,'nominal','figure_1_nominal')
    methods=('fmadrl','predictive_full','predictive_observed','proposed');scenarios=list(dict.fromkeys(r['scenario'] for r in summary))
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for metric,ax in zip(TASKS,axes.flat):
        scale=100 if metric=='completion_rate' else 1
        for m in methods:
            vals=[lookup[s,m,metric] for s in scenarios]
            ax.errorbar(range(len(scenarios)),[v['mean']*scale for v in vals],yerr=[v['ci95_half_width']*scale for v in vals],label=NAMES[m],marker='o',capsize=2)
        ax.set_xticks(range(len(scenarios)),[s.replace('_',' ') for s in scenarios],rotation=25,ha='right');ax.set_ylabel(LABELS[metric])
    save(fig,'figure_2_conditions')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for method in ('fmadrl','dqn','mappo','proposed'):
        group=[r for r in records if r['scenario']=='nominal' and r['method']==method]
        for ax,metric in zip(axes,('reward','completion_rate')):
            data=np.array([[h[metric] for h in r['history']] for r in group]);scale=100 if metric=='completion_rate' else 1
            ax.plot((np.arange(data.shape[1])+1)*p['local_steps']+p['warmup'],data.mean(0)*scale,label=NAMES[method])
            widths=np.array([ci95(v) for v in data.T])*scale
            ax.fill_between((np.arange(data.shape[1])+1)*p['local_steps']+p['warmup'],data.mean(0)*scale-widths,data.mean(0)*scale+widths,alpha=.1)
            ax.set_xlabel('Physical slots including warmup');ax.set_ylabel('Training reward' if metric=='reward' else 'Training completion (%)')
    save(fig,'figure_3_learning')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    width=max(1,p['episodes']*p['episode_steps']//20)
    for method in ('proposed','ungated'):
        group=[r for r in records if r['scenario']=='corrupted_twin' and r['method']==method]
        for ax,metric in zip(axes,('completion_rate','gate_zero_rate')):
            data=np.array([[c[metric] for c in r['curves']] for r in group]);mean=data.mean(0)*100
            xs=np.arange(len(mean));splits=list(range(0,len(mean),width))
            ax.plot([xs[k:k+width].mean() for k in splits],[mean[k:k+width].mean() for k in splits],label=NAMES[method])
            ax.axvline(p['episodes']*p['episode_steps']//3,color='black',ls=':');ax.set_xlabel('Evaluation slot');ax.set_ylabel(LABELS[metric])
    save(fig,'figure_4_corruption')
    bars(('legacy','proposed','augmentation_only','ranking_only','ungated'),'nominal','figure_5_ablations')
    fig,axes=plt.subplots(3,2,figsize=(12,13))
    prediction=[x for r in records if r['method']=='proposed' and r['scenario']=='nominal' for x in r['probes']]
    modes=sorted(set(v['mode'] for v in prediction))
    for ax,metric in zip(axes.flat,('state_mae','delay_mae','energy_mae','reward_mae')):
        vals=[np.mean([v[metric] for v in prediction if v['mode']==m]) for m in modes]
        ax.bar(range(len(modes)),vals);ax.set_xticks(range(len(modes)),modes,rotation=40,ha='right');ax.set_ylabel(metric.replace('_',' '))
    ranks=[x for r in records if r['method']=='proposed' and r['scenario']=='nominal' for x in r['ranking']]
    axes[2,0].bar([0],[np.mean([r['ranking_accuracy'] for r in ranks])*100]);axes[2,0].set_xticks([0],['Top-action agreement']);axes[2,0].set_ylabel('Agreement (%)')
    pred=[x for r in records if r['method']=='proposed' and r['scenario']=='corrupted_twin' for x in r['predictions']]
    centers=[];errors=[]
    for left in np.arange(0,1,.2):
        vals=[v['state_mae'] for v in pred if left<=v['score']<left+.2]
        if vals:centers.append(left+.1);errors.append(np.mean(vals))
    axes[2,1].plot(centers,errors,marker='o');axes[2,1].set_xlabel('Predecision reliability bin');axes[2,1].set_ylabel('Observed state MAE')
    save(fig,'figure_6_prediction')
    # Group-level results include all baselines, even when they outperform ours.
    fig,axes=plt.subplots(3,2,figsize=(12,13))
    for row,group in enumerate(('low','medium','high')):
        for col,metric in enumerate(('delay','completion_rate')):
            ax=axes[row,col];scale=100 if metric=='completion_rate' else 1
            for j,method in enumerate(methods):
                vals=[r['groups'][group][metric] for r in records if r['method']==method and r['scenario']=='mobile_groups']
                ax.bar(j,np.mean(vals)*scale,yerr=ci95(vals)*scale,label=NAMES[method],capsize=3)
            ax.set_xticks([]);ax.set_xlabel(group+' resource group');ax.set_ylabel(LABELS[metric])
    save(fig,'figure_7_mobile_groups')
    def cell(v,scale=1):return f"{v['mean']*scale:.3f} ± {v['ci95_half_width']*scale:.3f}"
    lines=['# Revised research: measured results','',f"Profile: **{out.name}**. {len(p['seeds'])} independent confirmation seeds; {p['rounds']*p['local_steps']} training slots, {p['warmup']} warmup slots, and {p['episodes']*p['episode_steps']} evaluation slots per case.",'',
           'Hyperparameters were selected on disjoint validation seeds. Values are seed means ± unadjusted Student-t 95% CI half-width. Four prespecified primary tests have a separate Holm correction; remaining comparisons are exploratory. No statistical superiority is inferred from marginal interval overlap.','',
           '| Method | Delay (s) | Energy (J/task) | Completion (%) | Cost | Joint p95 (ms) |','| --- | --- | --- | --- | --- | --- |']
    for m in main:
        lines.append('| '+NAMES[m]+' | '+' | '.join(cell(lookup['nominal',m,k],100 if k=='completion_rate' else 1) for k in (*TASKS,'decision_ms_p95'))+' |')
    lines+=['','## Prespecified paired comparisons','','Negative differences favor the proposed method for cost and delay. Intervals are descriptive and unadjusted; the adjusted p-value controls the four-test family.','','| Condition | Comparator | Metric | Proposed − comparator | Holm p | Evidence at 0.05 |','| --- | --- | --- | --- | --- | --- |']
    for r in comparisons:
        if r['family']!='primary':continue
        direction='supported reduction' if r['p_holm']<.05 and r['mean_difference']<0 else ('supported increase (adverse)' if r['p_holm']<.05 else 'not established')
        lines.append(f"| {r['scenario']} | {NAMES[r['comparator']]} | {r['metric']} | {r['mean_difference']:.4f} ± {r['ci95_half_width']:.4f} | {r['p_holm']:.4f} | {direction} |")
    lines+=['','## Figures','']
    captions=['Nominal comparison with both Predictive references and the original implementation.','Out-of-distribution resource, workload, corruption and mobility conditions.','Longer-budget learning trajectories; fluctuations alone do not establish convergence.','Response to corrupted predictions; the dotted line marks onset.','Matched complete/component ablations, including unfavorable comparisons.','Twin diagnostics from held-out probes and causal observations; plotted pooled diagnostics are descriptive, not independent task-level inference.','Mobile heterogeneous groups: delay and completion.']
    for file,caption in zip(sorted(figs.glob('*.png')),captions):lines += [f'![{caption}]({file.name if False else "figures/"+file.name})','',caption,'']
    lines+=['## Interpretation and limitations','',
        'All policies are trained on the nominal environment and then evaluated under the listed conditions with frozen policy parameters and causal online twin updates. Resource/mobility cases therefore test transfer, not condition-specific retraining. The revised twin has state–action interactions, but its architecture, fixed dynamics abstraction, state-only gate, and nominal-only hyperparameter selection may still limit transfer.','',
        'The original analytical reference uses private simulator state. The observation-based reference uses the same 25 observations and public configuration; conservative sharing and peer contact estimates are stated explicitly. Both remain visible even if they win.','',
        'Energy remains modeled, with fixed twin overhead; increased interaction-model compute energy is not hardware-measured. Source budgets remain soft penalties; completion is based on deadline, feasibility, and contact conditions. No real traces, security/privacy guarantees, packet-event validation, or convergence proof have been added.','',
        'CSV files contain every summary and paired contrast. audit.json records physical workload matching and CPU/bandwidth checks. The study is evidence for assessing the approach, not an automatic declaration of publication readiness.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
