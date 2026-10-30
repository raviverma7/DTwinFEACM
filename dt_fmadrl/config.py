from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass
class EnvironmentConfig:
    num_mec: int = 4
    vehicles_per_mec: int = 20
    area_m: float = 1000.0
    episode_steps: int = 100
    bandwidth_hz: float = 10e6
    noise_w: float = 1e-9
    tx_power_w: float = 0.5
    pathloss_exponent: float = 2.7
    vehicle_cpu_hz: float = 2e9
    mec_cpu_hz: float = 20e9
    cloud_backhaul_bps: float = 100e6
    cloud_rtt_s: float = 0.04
    vehicle_helper_cpu_hz: float = 3e9
    cpu_energy_coeff: float = 1e-28
    mec_cpu_energy_coeff: float = 1e-31
    helper_cpu_energy_coeff: float = 5e-29
    task_size_bits_min: float = 0.2e6
    task_size_bits_max: float = 2e6
    cycles_per_bit_min: float = 500.0
    cycles_per_bit_max: float = 1500.0
    deadline_s_min: float = 0.5
    deadline_s_max: float = 3.0
    speed_mps_min: float = 5.0
    speed_mps_max: float = 25.0
    rho_delay: float = 0.7
    energy_weight: float = 1.0
    deadline_penalty: float = 8.0
    completion_reward: float = 2.0
    vehicle_energy_budget_j: float = 2.0
    system_energy_budget_j: float = 8.0
    energy_budget_penalty: float = 4.0
    min_cpu_fraction: float = 0.25
    min_tx_power_w: float = 0.08
    collaboration_edge_fraction: float = 0.6
    dt_sync_energy_j: float = 0.002
    federation_energy_j: float = 0.01
    mec_queue_delay_s: float = 0.25
    cloud_queue_delay_s: float = 0.4
    partition_delay_s: float = 0.02
    sync_delay_s: float = 0.025
    merge_delay_s: float = 0.015
    backhaul_energy_per_bit: float = 2e-9
    collaboration_slots_per_mec: int = 5
    parallelism_threshold: float = 0.35
    handover_penalty: float = 5.0
    devices_per_edge: list[int] | None = None
    mobile: bool = False
    replenish_energy_each_slot: bool = True
    slot_duration_s: float = 1.0
    device_cpu_scale_min: float = 0.5
    device_cpu_scale_max: float = 1.5
    device_budget_scale_min: float = 0.5
    device_budget_scale_max: float = 1.5
    cloud_cpu_hz: float = 100e9
    cloud_cpu_energy_coeff: float = 1e-32
    rx_power_w: float = 0.1
    infrastructure_tx_power_w: float = 1.0
    infrastructure_rx_power_w: float = 0.2
    coordination_energy_j: float = 0.001
    energy_reference_j: float = 2.0
    infeasible_penalty: float = 5.0
    link_availability: float = 0.98
    peer_range_m: float = 250.0
    arrival_probability: float = 1.0
    domain_arrival_multipliers: list[float] | None = None
    domain_workload_scales: list[float] | None = None
    device_groups: list[dict] | None = None
    outage_duration_slots: int = 1


@dataclass
class LearningConfig:
    rounds: int = 120
    local_steps: int = 25
    actor_lr: float = 0.015
    critic_lr: float = 0.03
    gamma: float = 0.95
    entropy: float = 0.01
    federation_interval: int = 5
    evaluation_episodes: int = 15
    federation_enabled: bool = True
    federation_participation: float = 1.0
    federation_weighting: str = "samples"
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    hidden_dim: int = 32
    profile_allocations: bool = False


@dataclass
class DigitalTwinConfig:
    state_action_interactions: bool = False
    require_fitted_guidance: bool = False
    history: int = 1024
    ridge: float = 0.01
    refit_interval: int = 20
    prediction_horizon: int = 1
    pretrain_steps: int = 300
    confidence_gate: float = 0.45
    model_bias: float = 10.0
    energy_bias: float = 0.1
    reservoir_dim: int = 16
    reservoir_decay: float = 0.85
    residual_decay: float = 0.9
    residual_sensitivity: float = 5.0
    augmentation: bool = True
    ranking: bool = True
    actor_guidance: bool = True
    gating: bool = True
    synchronization_interval: int = 1
    prediction_bias: float = 0.0
    prediction_noise_std: float = 0.0


@dataclass
class ResearchConfig:
    seed: int
    environment: EnvironmentConfig
    learning: LearningConfig
    digital_twin: DigitalTwinConfig


def _construct(cls, values: dict):
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ResearchConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResearchConfig(
        seed=int(raw.get("seed", 42)),
        environment=_construct(EnvironmentConfig, raw.get("environment", {})),
        learning=_construct(LearningConfig, raw.get("learning", {})),
        digital_twin=_construct(DigitalTwinConfig, raw.get("digital_twin", {})),
    )
