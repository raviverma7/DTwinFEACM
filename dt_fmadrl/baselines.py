from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ResearchConfig
from .environment import VehicularEdgeEnv


@dataclass
class BaselineRecord:
    round: int
    reward: float
    delay: float
    energy: float
    completion_rate: float


class DQNTrainer:
    """Two-layer tanh Q network, bounded replay, and a lagged target network."""
    def __init__(self, cfg):
        from collections import deque
        self.cfg = cfg; self.rng = np.random.default_rng(cfg.seed+101)
        self.env = VehicularEdgeEnv(cfg.environment,cfg.seed)
        self.w1 = self.rng.normal(0,.1,(self.env.base_state_dim+1,32))
        self.w2 = self.rng.normal(0,.1,(33,7))
        self.target = (self.w1.copy(),self.w2.copy())
        self.replay = deque(maxlen=4096); self.steps = 0; self.training_generated_tasks = 0

    def _forward(self, states, target=False):
        w1,w2 = self.target if target else (self.w1,self.w2)
        x = np.column_stack((states,np.ones(len(states))))
        h = np.tanh(x@w1)
        hx = np.column_stack((h,np.ones(len(h))))
        return x,h,hx,hx@w2

    def act(self, states, explore=True, progress=1.):
        feasible = self.env.feasible_actions()
        q = self._forward(states)[3]
        actions = np.argmax(np.where(feasible,q,-np.inf),axis=1)
        if explore:
            random = self.rng.random(len(states))<max(.05,.9*(1-progress))
            for i in np.flatnonzero(random): actions[i] = self.rng.choice(np.flatnonzero(feasible[i]))
        return actions

    def _learn(self):
        count = min(64,len(self.replay))
        samples = [self.replay[i] for i in self.rng.choice(len(self.replay),count,replace=False)]
        states,actions,rewards,next_states,masks = map(np.asarray,zip(*samples))
        x,h,hx,q = self._forward(states)
        nq = self._forward(next_states,True)[3]
        # Environment episode boundaries are truncations, hence bootstrap remains enabled.
        target = rewards+self.cfg.learning.gamma*np.max(np.where(masks,nq,-np.inf),axis=1)
        error = np.clip(q[np.arange(count),actions]-target,-10,10)
        dq = np.zeros_like(q); dq[np.arange(count),actions] = error/count
        dw2 = hx.T@dq; dw1 = x.T@((dq@self.w2[:-1].T)*(1-h*h))
        lr = self.cfg.learning.critic_lr
        self.w1 -= lr*np.clip(dw1,-5,5); self.w2 -= lr*np.clip(dw2,-5,5)
        self.steps += 1
        if self.steps%20==0: self.target = (self.w1.copy(),self.w2.copy())

    def train(self, rounds):
        states = self.env.reset(); out = []; step = 0; total = max(1,rounds*self.cfg.learning.local_steps)
        for rnd in range(1,rounds+1):
            values = [[] for _ in range(4)]
            for _ in range(self.cfg.learning.local_steps):
                actions = self.act(states,True,step/total); ns,r,done,m = self.env.step(actions)
                self.training_generated_tasks += int(m.generated.sum())
                self.replay.extend(zip(states[m.generated].copy(),actions[m.generated].copy(),r[m.generated].copy(),ns[m.generated].copy(),self.env.feasible_actions()[m.generated].copy()))
                if self.replay: self._learn()
                for destination,value in zip(values,(r,m.delay,m.energy,m.completed)): destination.extend(value[m.generated])
                states = self.env.reset() if done else ns; step += 1
            out.append(BaselineRecord(rnd,*[float(np.mean(v)) for v in values]))
        return out

    def evaluate(self, episodes):
        return evaluate_policy(self.env,lambda s:self.act(s,False),episodes)


class DDPGTrainer:
    """Single global deterministic actor-critic baseline with four action bins."""
    def __init__(self,cfg:ResearchConfig):
        self.cfg=cfg;self.rng=np.random.default_rng(cfg.seed+202);self.env=VehicularEdgeEnv(cfg.environment,cfg.seed)
        dim=self.env.base_state_dim+1;self.actor=self.rng.normal(0,.02,dim);self.critic=np.zeros(dim+1)
    def _x(self,s):return np.column_stack((s,np.ones(len(s))))
    def act(self,s,explore=True,progress=1.0):
        z=np.tanh(self._x(s)@self.actor)
        if explore:z+=self.rng.normal(0,max(.05,.5*(1-progress)),len(s))
        actions = np.clip(((z+1)*3.5).astype(int),0,6)
        feasible = self.env.feasible_actions()
        for i,a in enumerate(actions):
            if not feasible[i,a]:
                choices = np.flatnonzero(feasible[i]); actions[i] = choices[np.argmin(np.abs(choices-a))]
        return actions
    def train(self,rounds):
        s=self.env.reset();out=[];total=max(1,rounds*self.cfg.learning.local_steps);step=0
        for rnd in range(1,rounds+1):
            rr=[];dd=[];ee=[];cc=[]
            for _ in range(self.cfg.learning.local_steps):
                a=self.act(s,True,step/total);ns,r,done,m=self.env.step(a);x=self._x(s)
                cf=np.column_stack((x,a/6));ncf=np.column_stack((self._x(ns),self.act(ns,False)/6))
                td=np.clip(r+self.cfg.learning.gamma*(ncf@self.critic)-cf@self.critic,-10,10)
                self.critic+=self.cfg.learning.critic_lr*np.mean(td[:,None]*cf,axis=0)
                advantage=td*(a-3.0)/3.0;self.actor+=self.cfg.learning.actor_lr*np.mean(advantage[:,None]*x,axis=0)
                rr.extend(r);dd.extend(m.delay);ee.extend(m.energy);cc.extend(m.completed);s=self.env.reset() if done else ns;step+=1
            out.append(BaselineRecord(rnd,float(np.mean(rr)),float(np.mean(dd)),float(np.mean(ee)),float(np.mean(cc))))
        return out
    def evaluate(self,episodes):return evaluate_policy(self.env,lambda s:self.act(s,False),episodes)


def evaluate_policy(env,policy,episodes):
    env.rng = np.random.default_rng(env.cfg_seed+100000) if hasattr(env,"cfg_seed") else env.rng
    totals={k:[] for k in ('reward','delay','energy','completion_rate','cost')}
    for _ in range(episodes):
        s=env.reset();done=False
        while not done:
            s,r,done,m=env.step(policy(s));totals['reward'].extend(r[m.generated]);totals['delay'].extend(m.delay[m.generated])
            totals['energy'].extend(m.energy[m.generated]);totals['completion_rate'].extend(m.completed[m.generated]);totals['cost'].extend(m.cost[m.generated])
    return {k:float(np.mean(v)) for k,v in totals.items()}
