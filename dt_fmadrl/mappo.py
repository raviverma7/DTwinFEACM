"""Parameter-shared discrete MAPPO with a centralized critic.

PPO clipped surrogate: https://arxiv.org/abs/1707.06347
Centralized-training/decentralized-execution: https://arxiv.org/abs/2103.01955
This is a documented NumPy baseline, not a reproduction of a published benchmark.
"""
import numpy as np
from .agent import softmax
from .environment import DeviceEdgeEnv
from .federated import TrainingRecord

class MLP:
    def __init__(self, dim, hidden, outputs, rng):
        self.w1 = rng.normal(0,np.sqrt(2/(dim+hidden)),(dim+1,hidden))
        self.w2 = rng.normal(0,.05,(hidden+1,outputs))
        self.m = [np.zeros_like(self.w1),np.zeros_like(self.w2)]
        self.v = [np.zeros_like(self.w1),np.zeros_like(self.w2)]
        self.step = 0

    def forward(self, states):
        x = np.column_stack((states,np.ones(len(states))))
        h = np.tanh(x@self.w1)
        hx = np.column_stack((h,np.ones(len(h))))
        return hx@self.w2,(x,h,hx)

    def update(self, cache, derivative, lr):
        x,h,hx = cache
        gradients = [x.T@((derivative@self.w2[:-1].T)*(1-h*h)),hx.T@derivative]
        norm = np.sqrt(sum(np.sum(g*g) for g in gradients))
        self.step += 1
        for i,(parameter,gradient) in enumerate(zip((self.w1,self.w2),gradients)):
            gradient *= min(1.,.5/max(norm,1e-12))
            self.m[i] = .9*self.m[i]+.1*gradient
            self.v[i] = .999*self.v[i]+.001*gradient*gradient
            parameter -= lr*(self.m[i]/(1-.9**self.step))/(np.sqrt(self.v[i]/(1-.999**self.step))+1e-8)

class MAPPOTrainer:
    def __init__(self, cfg):
        self.cfg = cfg; self.env = DeviceEdgeEnv(cfg.environment,cfg.seed)
        self.rng = np.random.default_rng(cfg.seed+2027)
        self.actor = MLP(self.env.base_state_dim,cfg.learning.hidden_dim,7,self.rng)
        self.critic = MLP(self.env.base_state_dim*2,cfg.learning.hidden_dim,1,self.rng)
        self.training_data_bytes = 0; self.training_generated_tasks = 0

    def probabilities(self, states, feasible):
        logits,cache = self.actor.forward(states)
        p = softmax(np.where(feasible,logits,-1e30))*feasible
        return p/p.sum(axis=1,keepdims=True),cache

    @staticmethod
    def centralized(states):
        return np.column_stack((states,np.repeat(states.mean(axis=0)[None],len(states),axis=0)))

    def act(self, states, explore=False):
        p,_ = self.probabilities(states,self.env.feasible_actions())
        return np.array([self.rng.choice(7,p=row) for row in p]) if explore else p.argmax(axis=1)

    @staticmethod
    def clipped_coefficient(ratio, advantage, clip):
        keep = ((advantage>=0)&(ratio<=1+clip)) | ((advantage<0)&(ratio>=1-clip))
        return np.where(keep,ratio*advantage,0.)

    def train(self, rounds):
        lc = self.cfg.learning; states = self.env.reset(); records = []
        for rnd in range(1,rounds+1):
            batch=[]; totals=[[] for _ in range(4)]
            for _ in range(lc.local_steps):
                feasible=self.env.feasible_actions().copy(); active=self.env.active.copy()
                probs,_=self.probabilities(states,feasible); actions=self.act(states,True)
                cs=self.centralized(states); values=self.critic.forward(cs)[0][:,0]
                ns,r,done,m=self.env.step(actions)
                self.training_generated_tasks += int(m.generated.sum())
                nvalues=self.critic.forward(self.centralized(ns))[0][:,0]
                team_reward=float(r[active].mean()) if active.any() else 0.
                batch.append((states.copy(),cs,feasible,actions,probs[np.arange(self.env.n),actions],values,nvalues,team_reward,done,active))
                self.training_data_bytes += states.nbytes+actions.nbytes+r.nbytes
                for destination,value in zip(totals,(r,m.delay,m.energy,m.completed)): destination.extend(value[active])
                states=self.env.reset() if done else ns
            advantages=np.zeros((len(batch),self.env.n)); future=np.zeros(self.env.n)
            for t in range(len(batch)-1,-1,-1):
                item=batch[t]; delta=item[7]+lc.gamma*item[6]-item[5]
                future=delta+lc.gamma*lc.gae_lambda*(not item[8])*future; advantages[t]=future
            active=np.concatenate([b[9] for b in batch]); adv=advantages.reshape(-1)[active]
            if active.any():
                returns=(advantages+np.stack([b[5] for b in batch])).reshape(-1)[active]
                ss=np.concatenate([b[0] for b in batch])[active];cs=np.concatenate([b[1] for b in batch])[active]
                feasible=np.concatenate([b[2] for b in batch])[active];actions=np.concatenate([b[3] for b in batch])[active];old=np.concatenate([b[4] for b in batch])[active]
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                for _ in range(lc.ppo_epochs):
                    order=self.rng.permutation(len(adv))
                    for start in range(0,len(order),128):
                        ix=order[start:start+128]; p,cache=self.probabilities(ss[ix],feasible[ix])
                        ratio=p[np.arange(len(ix)),actions[ix]]/np.maximum(old[ix],1e-12)
                        coefficient=self.clipped_coefficient(ratio,adv[ix],lc.ppo_clip)
                        logp=np.log(np.maximum(p,1e-30)); entropy=-np.sum(p*logp,axis=1,keepdims=True)
                        gradient=coefficient[:,None]*(np.eye(7)[actions[ix]]-p)-lc.entropy*p*(logp+entropy)
                        self.actor.update(cache,-gradient/len(ix),lc.actor_lr*.1)
                        v,vc=self.critic.forward(cs[ix]); dv=np.clip(v[:,0]-returns[ix],-10,10)
                        self.critic.update(vc,dv[:,None]/len(ix),lc.critic_lr*.1)
            records.append(TrainingRecord(rnd,*[float(np.mean(v)) if v else 0. for v in totals],0.))
        return records
