"""Freeze classical baselines for the Track 1 instance ladder.

Instances (frozen by construction, recorded verbatim in the output):
- n=6:   lab backpack example [3, 5, 7, 9, 11, 13]
- n=12:  random.Random(42), randint(10, 99)
- n=24:  random.Random(7),  randint(10, 99)   (matches baselines.py demo seed)
- n=48/96/160: prefixes of utils/numbers_large.npy (the lab's PCE instance),
  so the scaling chapter compares against the exact instance the notebook uses.

Solvers: exact DP, simulated annealing (3 seeds), greedy LPT.
Annealing steps scale with n so the comparison stays honest but cheap.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from baselines import (
    anneal_partition,
    energy,
    exact_partition,
    greedy_partition,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def anneal_best(numbers, seeds, steps):
    best = None
    for s in seeds:
        signs, diff = anneal_partition(numbers, steps=steps, seed=s)
        if best is None or diff < best[1]:
            best = (signs, diff)
    return best


def run_instance(name, numbers):
    numbers = [int(x) for x in numbers]
    t0 = time.time()
    signs_opt, d_opt = exact_partition(numbers)
    _, d_greedy = greedy_partition(numbers)
    steps = max(20_000, 400 * len(numbers))
    _, d_sa = anneal_best(numbers, seeds=(42, 7, 123), steps=steps)
    return {
        "instance": name,
        "n": len(numbers),
        "numbers": numbers,
        "total_sum": sum(numbers),
        "diff_optimal": d_opt,
        "diff_greedy": d_greedy,
        "diff_anneal_best_of_3": d_sa,
        "energy_optimal": d_opt * d_opt,
        "energy_greedy": d_greedy * d_greedy,
        "energy_anneal": d_sa * d_sa,
        "optimal_signs": signs_opt,
        "anneal_steps_per_run": steps,
        "wall_s": round(time.time() - t0, 3),
    }


def main():
    instances = []
    instances.append(run_instance("lab_backpack_n6", [3, 5, 7, 9, 11, 13]))
    rng12 = random.Random(42)
    instances.append(run_instance("random42_n12", [rng12.randint(10, 99) for _ in range(12)]))
    rng24 = random.Random(7)
    instances.append(run_instance("random7_n24", [rng24.randint(10, 99) for _ in range(24)]))
    large = np.load(ROOT / "utils" / "numbers_large.npy")
    for k in (48, 96, 160):
        instances.append(run_instance(f"numbers_large_prefix_n{k}", large[:k].tolist()))

    payload = {
        "generated": "2026-10-24 (local build phase)",
        "solvers": ["exact_partition (bitset DP)", "greedy_partition (LPT)",
                    "anneal_partition (best of seeds 42/7/123)"],
        "instances": instances,
    }
    (OUT / "baselines-ladder.json").write_text(json.dumps(payload, indent=2))

    lines = ["# Classical baselines — frozen ladder results", "",
             "| Instance | n | Sum | Optimal diff | Greedy diff | Anneal diff (best of 3) |",
             "|---|---|---|---|---|---|"]
    for r in instances:
        lines.append(f"| {r['instance']} | {r['n']} | {r['total_sum']} | {r['diff_optimal']} "
                     f"| {r['diff_greedy']} | {r['diff_anneal_best_of_3']} |")
    lines += ["", "Annealing steps per run: max(20000, 400*n). Exact = bitset DP (provably optimal).",
              "Instances n<=24 frozen above; n=48/96/160 are prefixes of utils/numbers_large.npy."]
    (OUT / "baselines-ladder.md").write_text("\n".join(lines) + "\n")
    for r in instances:
        print(r["instance"], "n=", r["n"], "opt=", r["diff_optimal"],
              "greedy=", r["diff_greedy"], "anneal=", r["diff_anneal_best_of_3"])
    print("saved results/baselines-ladder.json + .md")


if __name__ == "__main__":
    main()
