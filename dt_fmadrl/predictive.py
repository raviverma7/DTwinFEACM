"""Simple model-based reference using conservative equal-share current resources."""
import numpy as np
from .environment import DeviceEdgeEnv

class PredictiveController:
    def __init__(self,cfg):
        self.cfg=cfg; self.training_generated_tasks=0; self.env=DeviceEdgeEnv(cfg.environment,cfg.seed)

    def act(self,states,explore=False):
        delay,dev,infra=self.env._all_action_metrics()
        c=self.cfg.environment; energy=dev+infra
        cost=c.rho_delay*delay/self.env.deadline[:,None]+(1-c.rho_delay)*c.energy_weight*energy/c.energy_reference_j
        penalties=c.deadline_penalty*np.maximum(delay-self.env.deadline[:,None],0)/self.env.deadline[:,None]
        penalties+=c.energy_budget_penalty*np.maximum(dev-self.env.remaining_energy[:,None],0)/np.maximum(self.env.remaining_energy[:,None],1e-9)
        horizon=self.env._contact_time()[:,None]
        peer_horizon=c.peer_range_m/np.maximum(np.linalg.norm(self.env.velocity-self.env.velocity[self.env.helper],axis=1),1e-6)
        horizons=np.repeat(horizon,7,axis=1);horizons[:,0]=np.inf;horizons[:,3]=peer_horizon if c.mobile else np.inf
        completed=(delay<=self.env.deadline[:,None])&(delay<=horizons)
        scores=c.completion_reward*completed-cost-penalties
        return np.argmax(np.where(self.env.feasible_actions(),scores,-np.inf),axis=1)

    def train(self,rounds):
        from .federated import TrainingRecord
        s=self.env.reset();history=[]
        for rnd in range(1,rounds+1):
            values=[[] for _ in range(4)]
            for _ in range(self.cfg.learning.local_steps):
                a=self.act(s);ns,r,done,m=self.env.step(a)
                self.training_generated_tasks += int(m.generated.sum())
                for out,value in zip(values,(r,m.delay,m.energy,m.completed)):out.extend(value[m.generated])
                s=self.env.reset() if done else ns
            history.append(TrainingRecord(rnd,*[float(np.mean(v)) if v else 0. for v in values],0.))
        return history


class RandomController(PredictiveController):
    def __init__(self,cfg):
        super().__init__(cfg);self.rng=np.random.default_rng(cfg.seed+404)
    def act(self,states,explore=False):
        return np.array([self.rng.choice(np.flatnonzero(mask)) for mask in self.env.feasible_actions()])


class ObservationPredictiveController(PredictiveController):
    """Analytical reference using only the same 25 observations and public config.

    No environment rates, queues, task arrays, contact geometry, helper identity,
    or hidden device CPU/budget values are accessed by the decision function.
    Current joint demand is unknown: conservative configured population sharing
    is used, as in the original full-information reference. Clipped observed
    contact time is used for edge modes; peer contact uses a public speed bound.
    """
    def act(self, states, explore=False):
        return self.actions_from_observations(states, self.cfg.environment)

    @staticmethod
    def actions_from_observations(s, c):
        bits=s[:,0]*c.task_size_bits_max;cycles=bits*s[:,1]*c.cycles_per_bit_max
        deadline=np.maximum(s[:,2]*c.deadline_s_max,1e-9);out=s[:,18]*bits
        population=sum(c.devices_per_edge or [c.vehicles_per_mec]*c.num_mec)
        # Domain counts can differ, but domain identity is not an observation.
        # Use the maximum configured count, without reading hidden identity.
        nedge=max(c.devices_per_edge or [c.vehicles_per_mec]*c.num_mec)
        bw=c.bandwidth_hz/nedge
        power=np.clip(c.min_tx_power_w+(c.tx_power_w-c.min_tx_power_w)*bits/c.task_size_bits_max,c.min_tx_power_w,c.tx_power_w)
        er=np.maximum(s[:,3]*c.bandwidth_hz,1e-9);pr=np.maximum(s[:,4]*c.bandwidth_hz,1e-9)
        snre=np.expm1(np.minimum(er/bw,60)*np.log(2));snrp=np.expm1(np.minimum(pr/bw,60)*np.log(2))
        downrate=np.maximum(bw*np.log2(1+snre*c.infrastructure_tx_power_w/np.maximum(power,1e-9)),1e-9)
        pdownrate=np.maximum(bw*np.log2(1+snrp*c.tx_power_w/np.maximum(power,1e-9)),1e-9)
        up=bits/er;down=out/downrate;pup=bits/pr;pdown=out/pdownrate
        cpu=np.maximum(s[:,16]*c.vehicle_cpu_hz*c.device_cpu_scale_max,1.)
        helper=np.maximum(s[:,22]*c.vehicle_cpu_hz*c.device_cpu_scale_max,1.)
        budget=np.maximum(s[:,17]*c.vehicle_energy_budget_j*c.device_budget_scale_max,1e-9)
        lf=np.clip(cycles/deadline,c.min_cpu_fraction*cpu,cpu)
        ef=c.mec_cpu_hz/nedge;cf=c.cloud_cpu_hz/population
        hf=np.minimum(c.vehicle_helper_cpu_hz,helper)/nedge
        eq=(1-np.clip(s[:,5],1e-6,1))/np.clip(s[:,5],1e-6,1)*c.slot_duration_s
        hq=(1-np.clip(s[:,6],1e-6,1))/np.clip(s[:,6],1e-6,1)*c.slot_duration_s
        load=np.clip(s[:,13],0,1-1e-6);cq=load/(1-load)*c.slot_duration_s
        lq=np.maximum(s[:,23],0)*c.deadline_s_max
        bh=max(c.cloud_backhaul_bps/population,1e-9);bu=bits/bh;bd=out/bh
        delay=np.zeros((len(s),7));dev=np.zeros_like(delay);infra=np.zeros_like(delay)
        delay[:,0]=lq+cycles/lf;delay[:,1]=up+eq+cycles/ef+down
        delay[:,2]=up+bu+c.cloud_rtt_s+cq+cycles/cf+bd+down
        delay[:,3]=pup+hq+cycles/hf+pdown
        dev[:,0]=c.cpu_energy_coeff*lf**2*cycles
        radio=power*up+c.rx_power_w*down;remote=c.infrastructure_rx_power_w*up+c.infrastructure_tx_power_w*down
        dev[:,1:]=radio[:,None];dev[:,3]=power*pup+c.rx_power_w*pdown
        infra[:,1]=remote+c.mec_cpu_energy_coeff*ef**2*cycles
        infra[:,2]=remote+c.cloud_cpu_energy_coeff*cf**2*cycles+c.backhaul_energy_per_bit*(bits+out)
        infra[:,3]=c.rx_power_w*pup+c.tx_power_w*pdown+c.helper_cpu_energy_coeff*hf**2*cycles
        for a,eta in zip((4,5,6),(.3,.5,.7)):
            delay[:,a]=up+c.partition_delay_s+np.maximum(eq+eta*cycles/ef,(1-eta)*(bu+bd)+c.cloud_rtt_s+cq+(1-eta)*cycles/cf)+c.sync_delay_s+c.merge_delay_s+down
            infra[:,a]=remote+c.mec_cpu_energy_coeff*ef**2*eta*cycles+c.cloud_cpu_energy_coeff*cf**2*(1-eta)*cycles+c.backhaul_energy_per_bit*(1-eta)*(bits+out)+c.coordination_energy_j
        feasible=np.ones_like(delay,dtype=bool);feasible[:,1]=s[:,19]>.5
        feasible[:,2]=feasible[:,1]&(s[:,21]>.5);feasible[:,3]=s[:,20]>.5
        feasible[:,4:]=(feasible[:,2]&(s[:,12]>=c.parallelism_threshold))[:,None]
        feasible[s[:,24]<.5]=False;feasible[s[:,24]<.5,0]=True
        horizons=np.full_like(delay,np.inf)
        if c.mobile:
            horizons[:,1:]=s[:,15,None]*10
            horizons[:,3]=c.peer_range_m/max(2*c.speed_mps_max,1e-9)
        completed=(delay<=deadline[:,None])&(delay<=horizons)
        cost=c.rho_delay*delay/deadline[:,None]+(1-c.rho_delay)*c.energy_weight*(dev+infra)/c.energy_reference_j
        penalties=c.deadline_penalty*np.maximum(delay-deadline[:,None],0)/deadline[:,None]
        penalties+=c.energy_budget_penalty*np.maximum(dev-budget[:,None],0)/budget[:,None]
        return np.argmax(np.where(feasible,c.completion_reward*completed-cost-penalties,-np.inf),axis=1)
