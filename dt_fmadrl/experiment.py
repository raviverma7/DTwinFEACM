from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .baselines import DQNTrainer

from .config import load_config
from .federated import FederatedTrainer


def write_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].__dict__))
        writer.writeheader()
        writer.writerows(r.__dict__ for r in records)


def run(config_path: str, output: str, rounds: int | None = None) -> dict:
    cfg = load_config(config_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    diagnostics = {}
    for name, use_twin in (("fmadrl", False), ("dt_fmadrl", True)):
        trainer = FederatedTrainer(cfg, use_twin=use_twin)
        history = trainer.train(rounds)
        write_records(out / f"{name}_training.csv", history)
        results[name] = trainer.evaluate()
        diagnostics[name] = trainer.diagnostics()
        np.savez(out / f'{name}_checkpoint.npz', **{f'actor_{i}':c.actor for i,c in enumerate(trainer.clients)}, **{f'critic_{i}':c.critic for i,c in enumerate(trainer.clients)}, **{f'twin_{i}':t.weights for i,t in enumerate(trainer.twins)} )
    dqn = DQNTrainer(cfg)
    dqn.train(cfg.learning.rounds if rounds is None else rounds)
    results['dqn'] = dqn.evaluate(cfg.learning.evaluation_episodes)
    random_trainer = FederatedTrainer(cfg, use_twin=False)
    results["random"] = random_trainer.evaluate(policy="random")
    baseline = results["fmadrl"]
    proposed = results["dt_fmadrl"]
    results["dt_gain_percent"] = {
        "reward": 100 * (proposed["reward"] - baseline["reward"]) / max(abs(baseline["reward"]), 1e-9),
        "delay_reduction": 100 * (baseline["delay"] - proposed["delay"]) / max(baseline["delay"], 1e-9),
        "energy_reduction": 100 * (baseline["energy"] - proposed["energy"]) / max(baseline["energy"], 1e-9),
        "completion_increase": 100 * (proposed["completion_rate"] - baseline["completion_rate"]) / max(baseline["completion_rate"], 1e-9)
    }
    (out / 'metadata.json').write_text(json.dumps({
        'formulation':'revised_equations_3_27', 'config':asdict(cfg),
        'rounds':cfg.learning.rounds if rounds is None else rounds,
        'evaluation_seed_offset':100000,'checkpoint':'final round; npz policy snapshots',
        'aggregation':'equal weighting of tasks within a run; independent seed outcomes across runs',
        'failure_treatment':'delay and energy include all generated tasks; completion denominator is all tasks',
        'evaluation_adaptation':'actors frozen; domain twins synchronize and refit online',
        'diagnostics':diagnostics},indent=2),encoding='utf-8')
    (out / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FMADRL versus digital-twin FMADRL")
    parser.add_argument("--config", default="config/research.json")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--rounds", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.rounds), indent=2))


if __name__ == "__main__":
    main()

