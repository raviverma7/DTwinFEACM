from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .config import EnvironmentConfig

LOCAL, MEC, CLOUD, VEHICLE, EDGE_CLOUD_30, EDGE_CLOUD_50, EDGE_CLOUD_70 = range(7)
PEER = VEHICLE
COLLAB_ACTIONS = (4, 5, 6)
COLLAB_SPLITS = np.array((.3, .5, .7))
ACTION_NAMES = ('local', 'edge', 'cloud', 'peer', 'edge_cloud_30', 'edge_cloud_50', 'edge_cloud_70')

@dataclass
class StepMetrics:
    delay: np.ndarray
    energy: np.ndarray
    completed: np.ndarray
    cost: np.ndarray
    vehicle_energy: np.ndarray
    infrastructure_energy: np.ndarray
    budget_violations: np.ndarray
    deadline_penalty: np.ndarray
    energy_penalty: np.ndarray
    handover_failures: np.ndarray
    generated: np.ndarray | None = None
    queue_length: float = 0.0
    cpu_constraint_violations: int = 0
    bandwidth_constraint_violations: int = 0
    energy_components: dict | None = None

    @property
    def device_energy(self):
        return self.vehicle_energy

class DeviceEdgeEnv:
    """Equal-share processor scheduler, orthogonal radios, retained work queues."""
    base_state_dim = 25

    def __init__(self, cfg, seed=42):
        self.cfg = cfg; self.cfg_seed = seed
        self.counts = np.asarray(cfg.devices_per_edge or [cfg.vehicles_per_mec]*cfg.num_mec, dtype=int)
        if len(self.counts) != cfg.num_mec or np.any(self.counts < 1):
            raise ValueError('devices_per_edge requires a positive count for each edge')
        if min(cfg.cloud_cpu_hz, cfg.mec_cpu_hz, cfg.vehicle_cpu_hz, cfg.energy_reference_j, cfg.slot_duration_s) <= 0:
            raise ValueError('Capacities, reference energy and slot duration must be positive')
        if cfg.cloud_backhaul_bps < 0: raise ValueError('Backhaul capacity cannot be negative')
        if not 0<=cfg.arrival_probability<=1 or not 0<=cfg.link_availability<=1:
            raise ValueError('Arrival and link probabilities must be in [0,1]')
        if min(cfg.device_cpu_scale_min,cfg.device_budget_scale_min,cfg.bandwidth_hz,cfg.noise_w,cfg.area_m)<=0 or cfg.outage_duration_slots<1:
            raise ValueError('Resource scales and outage duration must be positive')
        for values in (cfg.domain_arrival_multipliers,cfg.domain_workload_scales):
            if values is not None and (len(values)!=cfg.num_mec or np.any(np.asarray(values)<0)):
                raise ValueError('Domain multipliers require one nonnegative value per edge')
        for group in cfg.device_groups or []:
            if min(group['cpu_scale'],group['budget_scale'],group.get('workload_scale',1.))<=0:
                raise ValueError('Group resource/workload scales must be positive')
        self.n = int(self.counts.sum())
        self.rng = np.random.default_rng(seed)
        side = int(np.ceil(np.sqrt(cfg.num_mec)))
        grid = np.linspace(cfg.area_m/(2*side), cfg.area_m-cfg.area_m/(2*side), side)
        self.mec_xy = np.array([(x,y) for x in grid for y in grid][:cfg.num_mec])
        self.vehicle_mec = np.repeat(np.arange(cfg.num_mec), self.counts)
        self.device_edge = self.vehicle_mec
        self.dt_enabled = False
        self.task_trace_hash = ""
        self.reset()

    def reset(self):
        c = self.cfg
        self.t = 0
        self.xy = self.rng.uniform(0, c.area_m, (self.n,2))
        angle = self.rng.uniform(0, 2*np.pi, self.n)
        speed = self.rng.uniform(c.speed_mps_min,c.speed_mps_max,self.n) if c.mobile else np.zeros(self.n)
        self.velocity = np.column_stack((np.cos(angle),np.sin(angle)))*speed[:,None]
        self.device_cpu = c.vehicle_cpu_hz*self.rng.uniform(c.device_cpu_scale_min,c.device_cpu_scale_max,self.n)
        self.energy_budget = c.vehicle_energy_budget_j*self.rng.uniform(c.device_budget_scale_min,c.device_budget_scale_max,self.n)
        self.group_names = np.array(['low' if f<.85*c.vehicle_cpu_hz else 'high' if f>1.15*c.vehicle_cpu_hz else 'medium' for f in self.device_cpu],dtype=object)
        self.workload_scale = np.ones(self.n)
        self.arrival_scale = np.ones(self.n)
        self.connectivity = np.full(self.n,c.link_availability)
        if c.device_groups:
            group_ids = np.arange(self.n)%len(c.device_groups)
            for gid, group in enumerate(c.device_groups):
                mask = group_ids==gid
                self.group_names[mask] = group['name']
                self.device_cpu[mask] = c.vehicle_cpu_hz*group['cpu_scale']
                self.energy_budget[mask] = c.vehicle_energy_budget_j*group['budget_scale']
                self.workload_scale[mask] = group.get('workload_scale',1.)
                self.arrival_scale[mask] = group.get('arrival_scale',1.)
                self.connectivity[mask] = group.get('link_availability',c.link_availability)
        self.remaining_energy = self.energy_budget.copy()
        self.outage_remaining = np.zeros(self.n,dtype=int)
        self.edge_queue = np.zeros(c.num_mec); self.cloud_queue = 0.
        self.local_queue = np.zeros(self.n); self.peer_queue = np.zeros(self.n)
        self.mec_load = np.zeros(c.num_mec); self.cloud_load = self.backhaul_load = 0.
        self.helper_load = np.zeros(self.n); self.contribution = np.zeros(self.n); self.assignments = np.zeros(self.n)
        self._sample_tasks()
        return self.observe()

    def _sample_tasks(self):
        c = self.cfg
        self.task_bits = self.rng.uniform(c.task_size_bits_min,c.task_size_bits_max,self.n)
        self.cycles_per_bit = self.rng.uniform(c.cycles_per_bit_min,c.cycles_per_bit_max,self.n)
        self.deadline = self.rng.uniform(c.deadline_s_min,c.deadline_s_max,self.n)
        self.result_ratio = self.rng.uniform(.02,.15,self.n)
        self.parallelism = self.rng.beta(2,2,self.n)
        domain_workload = np.asarray(c.domain_workload_scales or [1.]*c.num_mec)[self.device_edge]
        self.cycles_per_bit *= self.workload_scale*domain_workload
        domain_arrival = np.asarray(c.domain_arrival_multipliers or [1.]*c.num_mec)[self.device_edge]
        probability = np.clip(c.arrival_probability*self.arrival_scale*domain_arrival,0,1)
        self.active = self.rng.random(self.n)<probability
        self.tx_power = np.clip(c.min_tx_power_w+(c.tx_power_w-c.min_tx_power_w)*self.task_bits/c.task_size_bits_max,c.min_tx_power_w,c.tx_power_w)
        distance = np.linalg.norm(self.xy[:,None]-self.xy[None,:],axis=2)
        np.fill_diagonal(distance,np.inf)
        eligible = (self.device_edge[:,None]==self.device_edge[None,:]) & (distance<=c.peer_range_m)
        eligible &= self.remaining_energy[None,:]>0
        self.helper = np.argmin(np.where(eligible,distance,np.inf),axis=1)
        self.peer_available = eligible.any(axis=1) & (self.rng.random(self.n)<self.connectivity)
        new_outage = self.rng.random(self.n)>=self.connectivity
        self.outage_remaining = np.maximum(self.outage_remaining-1,0)
        self.outage_remaining = np.where(new_outage,c.outage_duration_slots,self.outage_remaining)
        self.edge_available = self.outage_remaining==0
        self.backhaul_available = bool(self.rng.random()<c.link_availability) and c.cloud_backhaul_bps>0
        ed = np.linalg.norm(self.xy-self.mec_xy[self.device_edge],axis=1)+1
        pd = distance[np.arange(self.n),self.helper]+1
        pd = np.where(np.isfinite(pd),pd,c.area_m)
        self.edge_gain = self.rng.lognormal(0,.15,self.n)/ed**c.pathloss_exponent
        self.peer_gain = self.rng.lognormal(0,.15,self.n)/pd**c.pathloss_exponent

    def _rates(self):
        c = self.cfg; bw = c.bandwidth_hz/self.counts[self.device_edge]
        return bw*np.log2(1+self.tx_power*self.edge_gain/c.noise_w), bw*np.log2(1+self.tx_power*self.peer_gain/c.noise_w)

    def _contact_time(self):
        c = self.cfg
        if not c.mobile: return np.full(self.n,np.inf)
        distance = np.linalg.norm(self.xy-self.mec_xy[self.device_edge],axis=1)
        radius = c.area_m/np.sqrt(c.num_mec)*.75
        return np.maximum(radius-distance,0)/np.maximum(np.linalg.norm(self.velocity,axis=1),1e-6)

    def feasible_actions(self):
        mask = np.ones((self.n,7),dtype=bool)
        mask[:,MEC] = self.edge_available
        mask[:,CLOUD] = self.edge_available & self.backhaul_available
        mask[:,VEHICLE] = self.peer_available
        mask[:,4:] = (self.edge_available & self.backhaul_available & (self.parallelism>=self.cfg.parallelism_threshold))[:,None]
        mask[~self.active] = False
        mask[~self.active,LOCAL] = True
        return mask

    def observe(self):
        c = self.cfg; er,pr = self._rates()
        return np.column_stack((self.task_bits/c.task_size_bits_max,self.cycles_per_bit/c.cycles_per_bit_max,
            self.deadline/c.deadline_s_max,er/c.bandwidth_hz,pr/c.bandwidth_hz,
            1-self.mec_load[self.device_edge],1-self.helper_load[self.helper],self.xy[:,0]/c.area_m,self.xy[:,1]/c.area_m,
            np.linalg.norm(self.velocity,axis=1)/max(c.speed_mps_max,1),self.contribution/(1+np.bincount(self.device_edge,weights=self.contribution,minlength=c.num_mec)[self.device_edge]),
            np.minimum(self.assignments/10,1),self.parallelism,np.full(self.n,self.cloud_load),np.full(self.n,self.backhaul_load),
            np.minimum(self._contact_time(),10)/10,self.device_cpu/(c.vehicle_cpu_hz*c.device_cpu_scale_max),
            self.remaining_energy/(c.vehicle_energy_budget_j*c.device_budget_scale_max),self.result_ratio,
            self.edge_available,self.peer_available,np.full(self.n,self.backhaul_available),
            self.device_cpu[self.helper]/(c.vehicle_cpu_hz*c.device_cpu_scale_max),
            self.local_queue/(self.device_cpu*c.deadline_s_max),self.active)).astype(float)

    def _all_action_metrics(self, actions=None):
        c = self.cfg; bits = self.task_bits; cycles = bits*self.cycles_per_bit; output = bits*self.result_ratio
        er,pr = self._rates(); bw = c.bandwidth_hz/self.counts[self.device_edge]
        ed = bw*np.log2(1+c.infrastructure_tx_power_w*self.edge_gain/c.noise_w)
        pd = bw*np.log2(1+c.tx_power_w*self.peer_gain/c.noise_w)
        up,down,pup,pdown = bits/er,output/ed,bits/pr,output/pd
        if actions is None:
            ec = self.counts[self.device_edge]; cc = self.n
            pc = np.maximum(1,np.bincount(self.helper,minlength=self.n))[self.helper]
            local_cap = self.device_cpu; hf = np.minimum(c.vehicle_helper_cpu_hz,self.device_cpu[self.helper])/pc
        else:
            edge = ((actions==MEC)|np.isin(actions,COLLAB_ACTIONS)) & self.active
            cloud = ((actions==CLOUD)|np.isin(actions,COLLAB_ACTIONS)) & self.active
            ec = np.maximum(1,np.bincount(self.device_edge,weights=edge,minlength=c.num_mec))[self.device_edge]
            cc = max(1,int(cloud.sum()))
            incoming = np.bincount(self.helper,weights=(actions==VEHICLE)&self.active,minlength=self.n)
            share = np.maximum(1,incoming+((actions==LOCAL)&self.active))
            local_cap = self.device_cpu/share
            hf = np.minimum(c.vehicle_helper_cpu_hz,self.device_cpu[self.helper])/share[self.helper]
        ef = c.mec_cpu_hz/ec; cf = c.cloud_cpu_hz/cc
        lf = np.clip(cycles/self.deadline,c.min_cpu_fraction*local_cap,local_cap)
        bh = c.cloud_backhaul_bps/cc
        bhup = np.divide(bits,bh,out=np.full(self.n,np.inf),where=bh>0)
        bhdown = np.divide(output,bh,out=np.full(self.n,np.inf),where=bh>0)
        eq = self.edge_queue[self.device_edge]/c.mec_cpu_hz; cq = self.cloud_queue/c.cloud_cpu_hz
        hq = self.peer_queue[self.helper]/self.device_cpu[self.helper]
        delay = np.zeros((self.n,7)); dev = np.zeros_like(delay); infra = np.zeros_like(delay)
        delay[:,LOCAL] = self.local_queue/self.device_cpu+cycles/lf
        delay[:,MEC] = up+eq+cycles/ef+down
        delay[:,CLOUD] = up+bhup+c.cloud_rtt_s+cq+cycles/cf+bhdown+down
        delay[:,VEHICLE] = pup+hq+cycles/hf+pdown
        dev[:,LOCAL] = c.cpu_energy_coeff*lf**2*cycles
        radio = self.tx_power*up+c.rx_power_w*down
        comm = c.infrastructure_rx_power_w*up+c.infrastructure_tx_power_w*down
        dev[:,MEC] = dev[:,CLOUD] = radio
        infra[:,MEC] = comm+c.mec_cpu_energy_coeff*ef**2*cycles
        infra[:,CLOUD] = comm+c.cloud_cpu_energy_coeff*cf**2*cycles+c.backhaul_energy_per_bit*(bits+output)
        dev[:,VEHICLE] = self.tx_power*pup+c.rx_power_w*pdown
        infra[:,VEHICLE] = c.rx_power_w*pup+c.tx_power_w*pdown+c.helper_cpu_energy_coeff*hf**2*cycles
        for a,eta in zip(COLLAB_ACTIONS,COLLAB_SPLITS):
            eb = eq+eta*cycles/ef
            cb = (1-eta)*bhup+c.cloud_rtt_s+cq+(1-eta)*cycles/cf+(1-eta)*bhdown
            delay[:,a] = up+c.partition_delay_s+np.maximum(eb,cb)+c.sync_delay_s+c.merge_delay_s+down
            dev[:,a] = radio
            infra[:,a] = comm+c.mec_cpu_energy_coeff*ef**2*eta*cycles+c.cloud_cpu_energy_coeff*cf**2*(1-eta)*cycles+c.backhaul_energy_per_bit*(1-eta)*(bits+output)+c.coordination_energy_j
        dt_share = np.zeros(self.n)
        if self.dt_enabled:
            counts = np.bincount(self.device_edge,weights=self.active,minlength=c.num_mec)
            dt_share = np.where(self.active,c.dt_sync_energy_j/np.maximum(counts[self.device_edge],1),0.)
        infra += dt_share[:,None]
        self.dt_share = dt_share
        components = {name:np.zeros_like(delay) for name in ('source_cpu','source_radio','remote_cpu','remote_radio','backhaul','coordination','dt')}
        components['source_cpu'][:,LOCAL] = dev[:,LOCAL]
        components['source_radio'][:,1:] = dev[:,1:]
        components['remote_cpu'][:,MEC] = c.mec_cpu_energy_coeff*ef**2*cycles
        components['remote_cpu'][:,CLOUD] = c.cloud_cpu_energy_coeff*cf**2*cycles
        components['remote_cpu'][:,VEHICLE] = c.helper_cpu_energy_coeff*hf**2*cycles
        components['remote_radio'][:,MEC] = comm
        components['remote_radio'][:,CLOUD] = comm
        components['remote_radio'][:,VEHICLE] = c.rx_power_w*pup+c.tx_power_w*pdown
        components['backhaul'][:,CLOUD] = c.backhaul_energy_per_bit*(bits+output)
        for a,eta in zip(COLLAB_ACTIONS,COLLAB_SPLITS):
            components['remote_cpu'][:,a] = c.mec_cpu_energy_coeff*ef**2*eta*cycles+c.cloud_cpu_energy_coeff*cf**2*(1-eta)*cycles
            components['remote_radio'][:,a] = comm
            components['backhaul'][:,a] = c.backhaul_energy_per_bit*(1-eta)*(bits+output)
            components['coordination'][:,a] = c.coordination_energy_j
        components['dt'][:] = dt_share[:,None]
        for component in components.values(): component[~self.active] = 0.
        delay[~self.active] = 0.; dev[~self.active] = 0.; infra[~self.active] = 0.
        self.energy_components = components
        self.allocations = dict(edge_cpu=ef,cloud_cpu=np.full(self.n,cf),helper_cpu=hf,local_cpu=lf,edge_bandwidth=bw,peer_bandwidth=bw)
        return delay,dev,infra

    def step(self, actions):
        actions = np.asarray(actions,dtype=int)
        if actions.shape!=(self.n,) or np.any((actions<0)|(actions>6)): raise ValueError('Invalid joint actions')
        payload = b''.join(np.ascontiguousarray(value).tobytes() for value in (self.task_bits,self.cycles_per_bit,self.deadline,self.result_ratio,self.parallelism,self.active,self.xy,self.device_cpu,self.edge_gain))
        self.task_trace_hash = hashlib.sha256(self.task_trace_hash.encode()+payload).hexdigest()
        c = self.cfg; rows = np.arange(self.n); feasible = self.feasible_actions()[rows,actions]
        da,va,ia = self._all_action_metrics(actions)
        delay,dev,infra = da[rows,actions],va[rows,actions],ia[rows,actions]; energy = dev+infra
        horizon = self._contact_time()
        ph = c.peer_range_m/np.maximum(np.linalg.norm(self.velocity-self.velocity[self.helper],axis=1),1e-6)
        horizon = np.where(actions==VEHICLE,ph,horizon)
        failures = c.mobile & (actions!=LOCAL) & (delay>horizon)
        completed = (delay<=self.deadline)&feasible&~failures&self.active
        cost = c.rho_delay*delay/self.deadline+(1-c.rho_delay)*c.energy_weight*energy/c.energy_reference_j
        violation = np.maximum(dev-self.remaining_energy,0)/np.maximum(self.remaining_energy,1e-9)
        dp = c.deadline_penalty*np.maximum(delay-self.deadline,0)/self.deadline
        ep = c.energy_budget_penalty*violation
        reward = c.completion_reward*completed-cost-dp-ep-c.infeasible_penalty*(~feasible)
        reward[~self.active] = 0.
        generated = self.active.copy()
        cycles = self.task_bits*self.cycles_per_bit*self.active; eta = np.where(actions==MEC,1.,0.)
        for a,split in zip(COLLAB_ACTIONS,COLLAB_SPLITS): eta = np.where(actions==a,split,eta)
        ew = np.bincount(self.device_edge,weights=cycles*eta*feasible,minlength=c.num_mec)
        cs = np.where(actions==CLOUD,1.,np.where(np.isin(actions,COLLAB_ACTIONS),1-eta,0.))
        pw = np.bincount(self.helper,weights=cycles*(actions==VEHICLE)*feasible,minlength=self.n)
        self.edge_queue = np.maximum(0,self.edge_queue+ew-c.mec_cpu_hz*c.slot_duration_s)
        self.cloud_queue = max(0,self.cloud_queue+np.sum(cycles*cs*feasible)-c.cloud_cpu_hz*c.slot_duration_s)
        self.local_queue = np.maximum(0,self.local_queue+cycles*(actions==LOCAL)+pw-self.device_cpu*c.slot_duration_s)
        self.peer_queue = self.local_queue.copy()
        self.mec_load = self.edge_queue/(self.edge_queue+c.mec_cpu_hz*c.slot_duration_s)
        self.cloud_load = self.cloud_queue/(self.cloud_queue+c.cloud_cpu_hz*c.slot_duration_s)
        self.helper_load = self.peer_queue/(self.peer_queue+self.device_cpu*c.slot_duration_s)
        he = np.bincount(self.helper,weights=(infra-self.dt_share)*(actions==VEHICLE),minlength=self.n)
        self.remaining_energy = np.maximum(0,self.remaining_energy-dev-he)
        self.assignments += np.bincount(self.helper,weights=(actions==VEHICLE)*feasible*self.active,minlength=self.n)
        self.contribution += pw/1e9
        self.xy += self.velocity*c.slot_duration_s
        hit = (self.xy<0)|(self.xy>c.area_m); self.velocity[hit]*=-1; self.xy=np.clip(self.xy,0,c.area_m)
        if c.replenish_energy_each_slot: self.remaining_energy = self.energy_budget.copy()
        components = {k:v[rows,actions].copy() for k,v in self.energy_components.items()}
        cpu_violations = 0
        for e in range(c.num_mec):
            served = (self.device_edge==e)&(((actions==MEC)|np.isin(actions,COLLAB_ACTIONS))&generated)
            cpu_violations += int(self.allocations['edge_cpu'][served].sum()>c.mec_cpu_hz*(1+1e-10))
        cloud_served = ((actions==CLOUD)|np.isin(actions,COLLAB_ACTIONS))&generated
        cpu_violations += int(self.allocations['cloud_cpu'][cloud_served].sum()>c.cloud_cpu_hz*(1+1e-10))
        peer_alloc = np.bincount(self.helper,weights=self.allocations['helper_cpu']*((actions==VEHICLE)&generated),minlength=self.n)
        total_device_cpu = peer_alloc+self.allocations['local_cpu']*((actions==LOCAL)&generated)
        cpu_violations += int(np.sum(total_device_cpu>self.device_cpu*(1+1e-10)))
        bandwidth_violations = 0
        for e in range(c.num_mec):
            edge_served = (self.device_edge==e)&(actions!=LOCAL)&(actions!=VEHICLE)&generated
            peer_served = (self.device_edge==e)&(actions==VEHICLE)&generated
            bandwidth_violations += int(self.allocations['edge_bandwidth'][edge_served].sum()>c.bandwidth_hz*(1+1e-10))
            bandwidth_violations += int(self.allocations['peer_bandwidth'][peer_served].sum()>c.bandwidth_hz*(1+1e-10))
        queue_length = float((self.edge_queue.sum()+self.cloud_queue+self.local_queue.sum())/max(np.mean(self.task_bits*self.cycles_per_bit),1))
        self.t += 1; self._sample_tasks()
        return self.observe(),reward,self.t>=c.episode_steps,StepMetrics(delay,energy,completed,cost,dev,infra,violation,dp,ep,failures,generated,queue_length,cpu_violations,bandwidth_violations,components)

VehicularEdgeEnv = DeviceEdgeEnv
