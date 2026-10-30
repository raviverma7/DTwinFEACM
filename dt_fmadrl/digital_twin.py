from collections import deque
import time
import numpy as np

class DigitalTwin:
    """Domain-local action-conditioned ridge models with causal history features."""
    def __init__(self, state_dim, cfg):
        self.state_dim = state_dim; self.cfg = cfg
        self.feature_dim = state_dim*2+7+1+(7*state_dim if cfg.state_action_interactions else 0)
        self.x = deque(maxlen=cfg.history); self.y = deque(maxlen=cfg.history)
        self.weights = np.zeros((self.feature_dim,state_dim+3))
        self.weights[:state_dim,:state_dim] = np.eye(state_dim)
        self.delta_ema = np.zeros(state_dim); self.residual_ema = np.zeros(state_dim)
        self.fit_count = 0; self.fit_durations = []; self.predict_seconds = 0.
        self.updates = 0; self.fit_seconds = 0.; self.last_errors = np.zeros(4)

    def _features(self, states, actions):
        parts = [states,np.repeat(self.delta_ema[None],len(states),axis=0),np.eye(7)[actions],np.ones(len(states))]
        if self.cfg.state_action_interactions:
            parts.append((np.eye(7)[actions,:,None]*states[:,None,:]).reshape(len(states),-1))
        return np.column_stack(parts)

    def issue(self, states, actions):
        features = self._features(states,actions)
        prediction = features@self.weights
        prediction = prediction + self.cfg.prediction_bias
        if self.cfg.prediction_noise_std:
            # Fixed, state/action-conditioned signed perturbation, identical for repeated queries.
            phase = features @ np.linspace(.7,4.3,self.feature_dim)
            perturbation = np.sin(phase[:,None]*np.arange(1,self.state_dim+4)[None,:]*7.13)
            scale = np.std(np.asarray(self.y),axis=0) if len(self.y)>1 else np.ones(self.state_dim+3)
            prediction += self.cfg.prediction_noise_std*perturbation*np.maximum(scale,1e-3)
        return features,prediction

    def update(self, states, actions, next_states, rewards=None, energies=None, delays=None, issued=None):
        features,prediction = self.issue(states,actions) if issued is None else issued
        residual = np.mean(np.abs(next_states-prediction[:,:self.state_dim]),axis=0)
        decay = self.cfg.residual_decay
        self.residual_ema = decay*self.residual_ema+(1-decay)*residual
        self.last_errors[0] = residual.mean()
        targets = [next_states]
        for index,value in enumerate((rewards,delays,energies)):
            value = np.zeros(len(states)) if value is None else np.asarray(value)
            targets.append(value[:,None])
            self.last_errors[index+1] = np.mean(np.abs(value-prediction[:,self.state_dim+index]))
        self.x.extend(features); self.y.extend(np.column_stack(targets))
        self.delta_ema = decay*self.delta_ema+(1-decay)*np.mean(next_states-states,axis=0)
        self.updates += 1
        if self.updates%self.cfg.refit_interval==0 and len(self.x)>=self.feature_dim:
            started = time.perf_counter(); x,y = np.asarray(self.x),np.asarray(self.y)
            self.weights = np.linalg.solve(x.T@x+self.cfg.ridge*np.eye(self.feature_dim),x.T@y)
            duration = time.perf_counter()-started
            self.fit_seconds += duration; self.fit_durations.append(duration); self.fit_count += 1

    def predict(self, states, actions=None):
        if actions is None: actions = np.zeros(len(states),dtype=int)
        return self.issue(states,actions)[1][:,:self.state_dim]

    def _outcomes(self, states):
        return np.stack([self.issue(states,np.full(len(states),a))[1] for a in range(7)],axis=1)

    def counterfactual(self, states):
        values = self._outcomes(states)
        return values[:,:,:self.state_dim],values[:,:,self.state_dim]

    def counterfactual_resources(self, states):
        values = self._outcomes(states)
        return np.maximum(values[:,:,self.state_dim+2],0),np.maximum(values[:,:,self.state_dim+1],0)

    @property
    def score(self):
        return float(np.exp(-self.cfg.residual_sensitivity*np.mean(self.residual_ema)))

    @property
    def confidence(self):
        if not self.cfg.gating: return 1.
        if self.cfg.require_fitted_guidance and self.fit_count == 0: return 0.
        score = self.score
        return score if score>=self.cfg.confidence_gate else 0.

    def augment(self, states, actions=None, feasible=None):
        predictions,rewards = self.counterfactual(states)
        mask = np.ones_like(rewards,dtype=bool) if feasible is None else feasible
        count = np.maximum(mask.sum(axis=1,keepdims=True),1)
        mean = np.sum(predictions*mask[:,:,None],axis=1)/count
        # Per-observation maximum absolute feasible reward; no future statistics.
        scale = np.max(np.where(mask,np.abs(rewards),0),axis=1,keepdims=True)+1e-6
        normalized = np.where(mask,rewards/scale,0)
        gate = self.confidence
        return np.column_stack((states,gate*mean,gate*normalized,np.full(len(states),gate)))
