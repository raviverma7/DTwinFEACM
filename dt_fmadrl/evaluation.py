"""Shared evaluation, causal prediction probes, and resource/overhead audit."""
from collections import defaultdict
import copy
from dataclasses import asdict
import threading
import time
import numpy as np
from .environment import ACTION_NAMES
from .federated import FederatedTrainer

METRICS=('delay','energy','completion_rate','cost','deadline_violation_rate','queue_length','decision_ms_mean','decision_ms_p95','decision_ms_p99')

class MemorySampler:
    """20 ms process RSS samples; peak may miss shorter native allocations."""
    def __enter__(self):
        import psutil
        self.process=psutil.Process(); self.baseline=self.process.memory_info().rss; self.peak=self.baseline
        self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.02): self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()

class TaskAccumulator:
    def __init__(self): self.count=0;self.values=defaultdict(float)
    def add(self,m,rewards,deadline,mask):
        mask=mask&m.generated;count=int(mask.sum())
        if not count:return
        self.count+=count
        for key,value in dict(delay=m.delay,energy=m.energy,completion_rate=m.completed,cost=m.cost,reward=rewards,
                              deadline_violation_rate=m.delay>deadline,source_energy=m.device_energy,remote_energy=m.infrastructure_energy).items():
            self.values[key]+=float(np.sum(value[mask]))
    def result(self):
        return dict(generated_tasks=self.count,**{k:v/self.count for k,v in self.values.items()}) if self.count else {'generated_tasks':0}

def controller_action(trainer,states,method):
    if isinstance(trainer,FederatedTrainer):
        return trainer._random_actions() if method=='random' else trainer._act(trainer._policy_state(states),False)
    return trainer.act(states,False)

def predicted_outcomes(trainer,states):
    result=np.zeros((trainer.env.n,7,trainer.env.base_state_dim+3));scores=np.zeros(trainer.env.n);guidance=np.zeros(trainer.env.n)
    for twin,mask in zip(trainer.twins,trainer._masks()):
        result[mask]=twin._outcomes(states[mask]);scores[mask]=twin.score;guidance[mask]=twin.confidence
    return result,scores,guidance

def prediction_errors(prediction,next_state,reward,delay,energy,state_dim):
    return dict(state_mae=float(np.mean(np.abs(prediction[:state_dim]-next_state))),
                reward_mae=float(abs(prediction[state_dim]-reward)),delay_mae=float(abs(prediction[state_dim+1]-delay)),
                energy_mae=float(abs(prediction[state_dim+2]-energy)))

def probe_actions(trainer,states,joint_actions,predictions,device,tick):
    """Exact single-device interventions; other devices' actions and random draws fixed."""
    env=trainer.env;feasible=env.feasible_actions()[device];rows=[];truth=np.full(7,-np.inf)
    for action in np.flatnonzero(feasible):
        clone=copy.deepcopy(env);actions=joint_actions.copy();actions[device]=action
        ns,r,_,m=clone.step(actions);truth[action]=r[device]
        errors=prediction_errors(predictions[device,action],ns[device],r[device],m.delay[device],m.energy[device],env.base_state_dim)
        rows.append(dict(slot=tick,device=int(device),mode=ACTION_NAMES[action],**errors))
    predicted_reward=predictions[device,:,env.base_state_dim]
    choice=int(np.argmax(np.where(feasible,predicted_reward,-np.inf)));best=float(np.max(truth))
    return rows,dict(slot=tick,device=int(device),ranking_accuracy=float(truth[choice]>=best-1e-8),decision_regret=float(best-truth[choice]),
                     chosen_controller_regret=float(best-truth[joint_actions[device]]))

def evaluate(trainer,method,episodes,probe_steps=12,disturbance=None):
    env=trainer.env;cfg=trainer.cfg;env.rng=np.random.default_rng(cfg.seed+100000);env.task_trace_hash=""
    # Training synchronization backlog is not allowed to leak into evaluation diagnostics.
    if isinstance(trainer,FederatedTrainer):trainer.pending_sync.clear();trainer.prediction_records.clear()
    totals=TaskAccumulator();groups=defaultdict(TaskAccumulator);domains=defaultdict(TaskAccumulator);modes=defaultdict(TaskAccumulator)
    component_sums=defaultdict(lambda:defaultdict(float));component_counts=defaultdict(int)
    latency=[];curves=[];prediction_rows=[];probe_rows=[];rank_rows=[];queue=[];cpu_violations=bandwidth_violations=0
    gates=0;score_count=0;fallback=0;tick=0;counts=np.zeros(7,dtype=int)
    base_workload=copy.deepcopy(cfg.environment.domain_workload_scales);base_links=cfg.environment.link_availability
    disturbance=disturbance or {};onset=disturbance.get('onset',episodes*cfg.environment.episode_steps//3)
    for episode in range(episodes):
        states=env.reset();done=False
        while not done:
            if disturbance and tick==onset:
                kind=disturbance['kind'];level=disturbance['level']
                if kind=='noise':cfg.digital_twin.prediction_noise_std=level
                elif kind=='bias':cfg.digital_twin.prediction_bias=level
                elif kind=='stale':cfg.digital_twin.synchronization_interval=int(level)
                elif kind=='workload':cfg.environment.domain_workload_scales=[level]*cfg.environment.num_mec
                elif kind=='outage':
                    env.connectivity[:]=level;cfg.environment.link_availability=level
                elif kind=='arrival':cfg.environment.arrival_probability=level
            deadline=env.deadline.copy();group_tags=env.group_names.copy();generated=env.active.copy()
            started=time.perf_counter();actions=controller_action(trainer,states,method);latency.append((time.perf_counter()-started)*1000)
            counts+=np.bincount(actions[generated],minlength=7)
            has_twin=isinstance(trainer,FederatedTrainer) and trainer.use_twin
            if has_twin:
                if probe_steps:
                    predictions,scores,guidance=predicted_outcomes(trainer,states)
                else:
                    scores=np.zeros(env.n);guidance=np.zeros(env.n)
                    for twin,mask in zip(trainer.twins,trainer._masks()):scores[mask]=twin.score;guidance[mask]=twin.confidence
                issued=trainer._issue(states,actions)
                if tick<probe_steps and generated.any():
                    devices=np.flatnonzero(generated);device=int(devices[tick%len(devices)])
                    pr,rr=probe_actions(trainer,states,actions,predictions,device,tick);probe_rows.extend(pr);rank_rows.append(rr)
                gates+=int(np.sum((guidance==0)&generated));score_count+=int(generated.sum())
            fallback+=int(np.sum(generated & ~env.edge_available & ~env.peer_available & (actions==0)))
            ns,r,done,m=env.step(actions)
            if has_twin:
                devices=np.flatnonzero(generated)
                sampled=devices[(np.arange(min(len(devices),8))+tick*8)%len(devices)] if len(devices) else []
                for device in sampled if probe_steps else []:
                    errors=prediction_errors(predictions[device,actions[device]],ns[device],r[device],m.delay[device],m.energy[device],env.base_state_dim)
                    prediction_rows.append(dict(slot=tick,mode=ACTION_NAMES[actions[device]],score=float(scores[device]),guidance=float(guidance[device]),**errors))
                trainer._update_twins(states,actions,ns,r,m,issued)
            totals.add(m,r,deadline,np.ones(env.n,dtype=bool))
            for group in sorted(set(group_tags)):groups[group].add(m,r,deadline,group_tags==group)
            for domain in range(cfg.environment.num_mec):domains[str(domain)].add(m,r,deadline,env.device_edge==domain)
            for action,name in enumerate(ACTION_NAMES):
                mask=actions==action;modes[name].add(m,r,deadline,mask)
                component_counts[name]+=int(np.sum(mask&generated))
                for component,value in m.energy_components.items():component_sums[name][component]+=float(value[mask&generated].sum())
            queue.append(m.queue_length);cpu_violations+=m.cpu_constraint_violations;bandwidth_violations+=m.bandwidth_constraint_violations
            if generated.any():curves.append(dict(slot=tick,generated_tasks=int(generated.sum()),reward=float(r[generated].mean()),delay=float(m.delay[generated].mean()),energy=float(m.energy[generated].mean()),completion_rate=float(m.completed[generated].mean()),deadline_violation_rate=float(np.mean(m.delay[generated]>deadline[generated])),gate_zero_rate=float(np.mean(guidance[generated]==0)) if has_twin else 0.))
            states=ns;tick+=1
    metrics=totals.result();metrics.update(queue_length=float(np.mean(queue)),decision_ms_mean=float(np.mean(latency)),decision_ms_p95=float(np.percentile(latency,95)),decision_ms_p99=float(np.percentile(latency,99)),cpu_constraint_violations=cpu_violations,bandwidth_constraint_violations=bandwidth_violations,gate_zero_rate=gates/max(score_count,1),outage_local_fallback_count=fallback)
    diagnostics=trainer.diagnostics() if isinstance(trainer,FederatedTrainer) else {}
    cfg.environment.domain_workload_scales=base_workload;cfg.environment.link_availability=base_links
    return dict(metrics=metrics,evaluation_task_trace_hash=env.task_trace_hash,groups={k:v.result() for k,v in groups.items()},domains={k:v.result() for k,v in domains.items()},
                modes={k:v.result() for k,v in modes.items()},energy_breakdown={name:{k:v/max(component_counts[name],1) for k,v in values.items()} for name,values in component_sums.items()},
                action_distribution={name:float(counts[i]/max(counts.sum(),1)) for i,name in enumerate(ACTION_NAMES)},
                latency_ms=latency,curves=curves,predictions=prediction_rows,probes=probe_rows,ranking=rank_rows,diagnostics=diagnostics)
