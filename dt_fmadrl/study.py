from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from dataclasses import asdict
from statistics import mean, stdev

from .config import load_config
from .federated import FederatedTrainer


METRICS = ("reward", "delay", "energy", "completion_rate", "cost")


from .statistics import ci95


def summarize(rows: list[dict]) -> dict:
    result = {}
    for method in ("random", "fmadrl", "dt_fmadrl"):
        selected = [r for r in rows if r["method"] == method]
        result[method] = {
            metric: {"mean": mean(r[metric] for r in selected),
                     "std": stdev(r[metric] for r in selected) if len(selected) > 1 else 0.0,
                     "ci95": ci95([r[metric] for r in selected])}
            for metric in METRICS
        }
    paired = {}
    for metric in METRICS:
        base = {r["seed"]: r[metric] for r in rows if r["method"] == "fmadrl"}
        dt = {r["seed"]: r[metric] for r in rows if r["method"] == "dt_fmadrl"}
        differences = [dt[s] - base[s] for s in sorted(base)]
        paired[metric] = {
            "mean_dt_minus_baseline": mean(differences),
            "ci95": ci95(differences),
            "consistent_seeds": sum(d > 0 for d in differences) if metric in ("reward", "completion_rate") else sum(d < 0 for d in differences),
            "total_seeds": len(differences),
        }
    result["paired_effect"] = paired
    return result


def run_study(config_path: str, output: str, seeds: list[int], rounds: int, eval_episodes: int) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}
    for seed in seeds:
        cfg = load_config(config_path)
        cfg.seed = seed
        cfg.learning.evaluation_episodes = eval_episodes
        random_eval = FederatedTrainer(cfg, False).evaluate(policy="random")
        rows.append({"seed": seed, "method": "random", **random_eval})
        for name, use_twin in (("fmadrl", False), ("dt_fmadrl", True)):
            trainer = FederatedTrainer(cfg, use_twin)
            history = trainer.train(rounds)
            evaluation = trainer.evaluate()
            rows.append({"seed": seed, "method": name, **evaluation})
            histories[f"{seed}_{name}"] = [record.__dict__ for record in history]
    with (out / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "method", *METRICS))
        writer.writeheader(); writer.writerows(rows)
    (out / 'metadata.json').write_text(json.dumps({'formulation':'revised_equations_3_27',
        'config':asdict(cfg),'seeds':seeds,'rounds':rounds,'evaluation_episodes':eval_episodes,
        'evaluation_seed_offset':100000,'checkpoint':'final round','twin_evaluation':'online synchronization',
        'failure_treatment':'all generated tasks included in every metric','aggregation':'seed-level sample SD and Student-t CI half-width'},indent=2),encoding='utf-8')
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "histories.json").write_text(json.dumps(histories), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research.json")
    parser.add_argument("--output", default="outputs/study")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    print(json.dumps(run_study(args.config, args.output, seeds, args.rounds, args.eval_episodes), indent=2))


if __name__ == "__main__":
    main()
