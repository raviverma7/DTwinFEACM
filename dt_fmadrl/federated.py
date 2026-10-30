from dataclasses import dataclass
import time
import tracemalloc
import numpy as np
from .agent import LinearActorCritic
from .digital_twin import DigitalTwin
from .environment import DeviceEdgeEnv, ACTION_NAMES

@dataclass
class TrainingRecord:
    round: int
    reward: float
    delay: float
    energy: float
    completion_rate: float
    twin_mae: float

class FederatedTrainer:
    """Parameter-shared domain policies; private domain twins and weighted FedAvg."""
    def __init__(self, cfg, use_twin):
        self.cfg = cfg; self.use_twin = use_twin
        self.rng = np.random.default_rng(cfg.seed)
        self.aggregation_rng = np.random.default_rng(cfg.seed+9000)
        self.env = DeviceEdgeEnv(cfg.environment,cfg.seed)
        self.env.dt_enabled = use_twin
        self.twins = [DigitalTwin(self.env.base_state_dim,cfg.digital_twin) for _ in range(cfg.environment.num_mec)]
        self.twin = self.twins[0]  # Compatibility for older diagnostics; not a global twin.
        self.augment = use_twin and cfg.digital_twin.augmentation
        dim = self.env.base_state_dim*2+8 if self.augment else self.env.base_state_dim
        self.clients = [LinearActorCritic(dim,self.rng) for _ in self.twins]
        self.sample_counts = np.zeros(len(self.clients),dtype=int)
        self.federation_bytes = 0; self.synchronization_bytes = 0; self.training_generated_tasks = 0
        self.decision_seconds = []; self.training_seconds = 0.; self.peak_memory_bytes = 0
        self.pending_sync = []; self.prediction_records = []
        if use_twin and cfg.digital_twin.pretrain_steps>0: self._pretrain_twin()

    def _masks(self):
        return [self.env.device_edge==e for e in range(len(self.clients))]

    def _random_actions(self):
        return np.array([self.rng.choice(np.flatnonzero(row)) for row in self.env.feasible_actions()])

    def _issue(self, states, actions):
        return [(*t.issue(states[m],actions[m]),t.confidence) for t,m in zip(self.twins,self._masks())]

    def _update_twins(self, states, actions, ns, r, metrics, issued):
        self.pending_sync.append((states.copy(),actions.copy(),ns.copy(),r.copy(),metrics,issued))
        if len(self.pending_sync)<self.cfg.digital_twin.synchronization_interval: return
        for item in self.pending_sync:
            self._synchronize(*item)
        self.pending_sync.clear()

    def _synchronize(self, states, actions, ns, r, metrics, issued):
        for twin,mask,prediction in zip(self.twins,self._masks(),issued):
            active = metrics.generated[mask] if metrics.generated is not None else np.ones(mask.sum(),dtype=bool)
            if not active.any(): continue
            domain = np.flatnonzero(mask)[active]
            prediction = (prediction[0][active],prediction[1][active],prediction[2])
            mask = domain
            twin.update(states[mask],actions[mask],ns[mask],r[mask],metrics.energy[mask],metrics.delay[mask],issued=prediction[:2])
            self.prediction_records.append([prediction[2],*twin.last_errors])
            self.synchronization_bytes += states[mask].nbytes+actions[mask].nbytes+ns[mask].nbytes+r[mask].nbytes+metrics.energy[mask].nbytes+metrics.delay[mask].nbytes

    def _pretrain_twin(self):
        states = self.env.reset()
        for _ in range(self.cfg.digital_twin.pretrain_steps):
            actions = self._random_actions(); issued = self._issue(states,actions)
            ns,r,done,m = self.env.step(actions)
            self._update_twins(states,actions,ns,r,m,issued)
            states = self.env.reset() if done else ns
        self.env.reset()

    def _policy_state(self, states, previous_actions=None):
        if not self.augment: return states
        result = np.empty((len(states),self.env.base_state_dim*2+8))
        feasible = self.env.feasible_actions()
        for twin,mask in zip(self.twins,self._masks()): result[mask] = twin.augment(states[mask],feasible=feasible[mask])
        return result

    def _act(self, policy_states, explore):
        started = time.perf_counter(); actions = np.empty(self.env.n,dtype=int)
        feasible = self.env.feasible_actions(); dt = self.cfg.digital_twin
        for client,twin,mask in zip(self.clients,self.twins,self._masks()):
            probs = client.probabilities(policy_states[mask],feasible[mask])
            if explore:
                actions[mask] = [self.rng.choice(7,p=p/p.sum()) for p in probs]
            else:
                scores = np.log(np.maximum(probs,1e-30)) if dt.actor_guidance or not self.use_twin else np.zeros_like(probs)
                if self.use_twin and dt.ranking:
                    raw = policy_states[mask,:self.env.base_state_dim]
                    _,reward = twin.counterfactual(raw); energy,_ = twin.counterfactual_resources(raw)
                    fm = feasible[mask]
                    def scale(v):
                        mean = np.sum(np.where(fm,v,0),axis=1,keepdims=True)/fm.sum(axis=1,keepdims=True)
                        return np.sqrt(np.sum(np.where(fm,(v-mean)**2,0),axis=1,keepdims=True)/fm.sum(axis=1,keepdims=True))+1e-6
                    scores += twin.confidence*(dt.model_bias*reward/scale(reward)-dt.energy_bias*energy/scale(energy))
                actions[mask] = np.argmax(np.where(feasible[mask],scores,-np.inf),axis=1)
        self.decision_seconds.append(time.perf_counter()-started)
        return actions

    def _local_update(self, states, actions, rewards, next_states, feasible=None, terminated=False, generated=None):
        lc = self.cfg.learning
        for idx,(client,mask) in enumerate(zip(self.clients,self._masks())):
            if generated is not None: mask = mask & generated
            if not mask.any(): continue
            client.update(states[mask],actions[mask],rewards[mask],next_states[mask],lc.gamma,lc.actor_lr,lc.critic_lr,lc.entropy,
                          terminated=terminated,feasible=None if feasible is None else feasible[mask])
            self.sample_counts[idx] += int(mask.sum())

    def aggregate(self):
        lc = self.cfg.learning
        if not lc.federation_enabled or self.sample_counts.sum()==0: return
        participants = (self.aggregation_rng.random(len(self.clients))<lc.federation_participation)&(self.sample_counts>0)
        if not participants.any(): return
        raw = np.where(participants,1 if lc.federation_weighting=='equal' else self.sample_counts,0)
        weights = raw/raw.sum()
        averaged = (np.sum(np.stack([c.actor for c in self.clients])*weights[:,None,None],axis=0),
                    np.sum(np.stack([c.critic for c in self.clients])*weights[:,None],axis=0))
        for idx,client in enumerate(self.clients):
            if not participants[idx]: continue
            self.federation_bytes += 2*(client.actor.nbytes+client.critic.nbytes)
            client.set_parameters(averaged)
            self.sample_counts[idx] = 0

    def train(self, rounds=None):
        tracing = tracemalloc.is_tracing()
        profiling = tracing or self.cfg.learning.profile_allocations
        if profiling and not tracing: tracemalloc.start()
        started = time.perf_counter(); lc = self.cfg.learning; states = self.env.reset(); records = []
        for rnd in range(1,(lc.rounds if rounds is None else rounds)+1):
            totals = [[] for _ in range(5)]
            for _ in range(lc.local_steps):
                feasible = self.env.feasible_actions().copy(); ps = self._policy_state(states)
                actions = self._act(ps,True)
                issued = self._issue(states,actions) if self.use_twin else None
                ns,r,done,m = self.env.step(actions)
                self.training_generated_tasks += int(m.generated.sum())
                # TD next observation uses the model available before learning this transition.
                nps = self._policy_state(ns)
                if self.use_twin: self._update_twins(states,actions,ns,r,m,issued)
                self._local_update(ps,actions,r,nps,feasible=feasible,terminated=False,generated=m.generated)
                for target,value in zip(totals,(r[m.generated],m.delay[m.generated],m.energy[m.generated],m.completed[m.generated],[np.mean([t.last_errors[0] for t in self.twins])])): target.extend(value)
                states = self.env.reset() if done else ns
            if rnd%lc.federation_interval==0: self.aggregate()
            records.append(TrainingRecord(rnd,*[float(np.mean(v)) if v else 0. for v in totals]))
        if profiling:
            self.peak_memory_bytes = max(self.peak_memory_bytes,tracemalloc.get_traced_memory()[1])
            if not tracing: tracemalloc.stop()
        self.training_seconds += time.perf_counter()-started
        return records

    def evaluate(self, episodes=None, policy='learned'):
        episodes = self.cfg.learning.evaluation_episodes if episodes is None else episodes
        # Common physical realizations across methods, independent of training/pretraining.
        self.env.rng = np.random.default_rng(self.cfg.seed+100000)
        self.prediction_records.clear()
        totals = {k:[] for k in ('reward','delay','energy','completion_rate','cost')}
        action_counts = np.zeros(7,dtype=int)
        detail = {k:[] for k in ('vehicle_energy','infrastructure_energy','budget_violation','deadline_penalty','energy_penalty','handover_failure')}
        for _ in range(episodes):
            states = self.env.reset(); done = False
            while not done:
                actions = self._random_actions() if policy=='random' else self._act(self._policy_state(states),False)
                action_counts += np.bincount(actions[self.env.active],minlength=7)
                issued = self._issue(states,actions) if self.use_twin else None
                ns,r,done,m = self.env.step(actions)
                # Synchronize residual/history after execution; policy parameters stay frozen.
                if self.use_twin: self._update_twins(states,actions,ns,r,m,issued)
                for key,value in zip(totals,(r,m.delay,m.energy,m.completed,m.cost)): totals[key].extend(value[m.generated])
                for key,value in zip(detail,(m.vehicle_energy,m.infrastructure_energy,m.budget_violations,m.deadline_penalty,m.energy_penalty,m.handover_failures)): detail[key].extend(value[m.generated])
                states = ns
        self.last_detail = {k:float(np.mean(v)) for k,v in detail.items()}
        self.last_actions = {name:float(action_counts[i]/max(action_counts.sum(),1)) for i,name in enumerate(ACTION_NAMES)}
        return {k:float(np.mean(v)) for k,v in totals.items()}

    def evaluate_detailed(self, episodes=3):
        metrics = self.evaluate(episodes)
        return {'metrics':{**metrics,**self.last_detail},'action_distribution':self.last_actions,'diagnostics':self.diagnostics()}

    def diagnostics(self):
        errors = np.mean(self.prediction_records,axis=0) if self.prediction_records else np.zeros(5)
        return dict(predecision_guidance_mean=float(errors[0]),state_mae=float(errors[1]),
                    reward_mae=float(errors[2]),delay_mae=float(errors[3]),energy_mae=float(errors[4]),
                    decision_seconds_mean=float(np.mean(self.decision_seconds)) if self.decision_seconds else 0.,
                    dt_refit_seconds=sum(t.fit_seconds for t in self.twins),training_seconds=self.training_seconds,
                    peak_memory_bytes=self.peak_memory_bytes,federation_bytes=self.federation_bytes,synchronization_bytes=self.synchronization_bytes)
