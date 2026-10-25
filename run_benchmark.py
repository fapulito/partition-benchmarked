"""Trained-QAOA benchmark vs frozen classical baselines (Track 1 entry data).

Method (v2): as Lab 4a — partition -> MaxCut -> Ising -> annotated QAOA
ansatz — trained with COBYLA on Aer EstimatorV2, sampled with Aer SamplerV2
(8192 shots), most-likely bitstring scored with baselines.report().

Training objective uses the Ising Hamiltonian NORMALIZED by its largest
|coefficient|. Number-partition weights give coefficients up to ~10^4, which
makes raw-<H> COBYLA stall (v1 of this script: optimizer declared
convergence after ~30 evals at a near-flat distribution). Normalization does
not change the problem or the sampled distribution at given angles; it only
re-parameterizes the training landscape (standard QAOA practice).

Instances are read from results/baselines-ladder.json (frozen). Results are
written incrementally after every run.
"""
from __future__ import annotations

import json
import time
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
from scipy.optimize import minimize

from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_addon_opt_mapper.applications import Maxcut
from qiskit_addon_opt_mapper.translators import to_ising
from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
from qopt_best_practices.transpilation import UnrollBoxes

import baselines

ROOT = Path(__file__).resolve().parent
SEED = 42
SHOTS = 8192
OUT_PATH = ROOT / "results" / "qaoa-vs-baselines.json"


def build(numbers, reps):
    g = nx.complete_graph(len(numbers))
    for i, j in combinations(range(len(numbers)), 2):
        g.add_edge(i, j, weight=numbers[i] * numbers[j])
    H, offset = to_ising(Maxcut(g).to_optimization_problem())
    circ = annotated_qaoa_ansatz(H, reps=reps)
    pm = generate_preset_pass_manager(optimization_level=3, backend=AerSimulator(),
                                      seed_transpiler=SEED)
    cand = pm.run(UnrollBoxes()(circ))
    return g, H, offset, cand


def train(Hn, cand, inits, maxiter):
    est = AerEstimatorV2()
    best = None
    for x0 in inits:
        vals = []
        def cost(p):
            v = float(est.run([(cand, Hn, p)]).result()[0].data.evs)
            vals.append(v)
            return v
        res = minimize(cost, np.array(x0, dtype=float), method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.3, "tol": 1e-6})
        if best is None or vals[-1] < best[1]:
            best = (res.x, float(vals[-1]), len(vals))
    return best


def sample_mode(numbers, cand, params):
    circ = cand.assign_parameters(params).copy()
    circ.measure_all()
    job = AerSamplerV2().run([(circ,)], shots=SHOTS)
    counts = job.result()[0].data.meas.get_counts()
    mode_bs, mode_count = max(counts.items(), key=lambda kv: kv[1])
    signs = [1 if mode_bs[len(mode_bs) - 1 - i] == "1" else -1 for i in range(len(numbers))]
    _, d_opt = baselines.exact_partition(numbers)
    p_opt = 0.0
    for bs, c in counts.items():
        s = [1 if bs[len(bs) - 1 - i] == "1" else -1 for i in range(len(numbers))]
        if baselines.difference(s, numbers) == d_opt:
            p_opt += c
    return signs, mode_count / SHOTS, p_opt / SHOTS


def exact_dist_stats(numbers, cand, params):
    sv = Statevector.from_instruction(cand.assign_parameters(params))
    probs = sv.probabilities_dict()
    def diff_of(bs):
        s = [1 if bs[len(bs) - 1 - i] == "1" else -1 for i in range(len(numbers))]
        return baselines.difference(s, numbers)
    _, d_opt = baselines.exact_partition(numbers)
    top_bs, top_p = max(probs.items(), key=lambda kv: kv[1])
    return {
        "top_bitstring": top_bs,
        "top_prob": round(float(top_p), 4),
        "top_diff": diff_of(top_bs),
        "expected_diff": round(float(sum(p * diff_of(bs) for bs, p in probs.items())), 3),
        "p_optimal": round(float(sum(p for bs, p in probs.items() if diff_of(bs) == d_opt)), 4),
    }


def main():
    ladder = json.loads((ROOT / "results" / "baselines-ladder.json").read_text())
    inst = {r["instance"]: r for r in ladder["instances"]}
    out = {"shots": SHOTS, "optimizer": "COBYLA (rhobeg 0.3, tol 1e-6), H normalized by max |coeff|",
           "runs": []}
    OUT_PATH.write_text(json.dumps(out, indent=2))

    plan = [
        ("lab_backpack_n6", [1, 2], 100),
        ("random42_n12", [1, 2], 80),
        ("random7_n24", [1], 40),
    ]
    rng = np.random.default_rng(7)
    for name, reps_list, maxiter in plan:
        numbers = inst[name]["numbers"]
        for reps in reps_list:
            t0 = time.time()
            g, H, offset, cand = build(numbers, reps)
            scale = float(np.max(np.abs(H.coeffs))) if len(H.coeffs) else 1.0
            Hn = H / scale
            npar = cand.num_parameters
            inits = []
            if name == "lab_backpack_n6" and reps == 1:
                inits.append(np.load(ROOT / "utils" / "pretrained_parameters_small.npy"))
            inits.append(rng.uniform(-0.8, 0.8, size=npar))
            if len(numbers) <= 12:
                inits.append(rng.uniform(-0.8, 0.8, size=npar))
            params, ncost, iters = train(Hn, cand, inits, maxiter)
            raw_cost = float(AerEstimatorV2().run([(cand, H, params)]).result()[0].data.evs)
            signs, mode_p, p_opt = sample_mode(numbers, cand, params)
            rep = baselines.report(numbers, signs)
            row = {
                "instance": name, "n": len(numbers), "reps": reps,
                "hamiltonian_scale": scale,
                "trained_normalized_cost": round(ncost, 4),
                "trained_raw_cost_expectation": round(raw_cost, 3),
                "ising_offset": float(offset),
                "optimizer_evals_best_run": iters,
                "trained_params": [round(float(x), 4) for x in params],
                "mode_probability": round(mode_p, 4),
                "p_sampled_optimal": round(p_opt, 4),
                "qaoa_signs": signs,
                "benchmark": rep,
                "wall_s": round(time.time() - t0, 1),
            }
            out["runs"].append(row)
            OUT_PATH.write_text(json.dumps(out, indent=2))
            print(f"{name} p={reps}: QAOA diff={rep['diff_qaoa']} (opt {rep['diff_optimal']}, "
                  f"greedy {rep['diff_greedy']}, anneal {rep['diff_anneal']}), "
                  f"P(opt)={p_opt:.3f}, raw<H>={raw_cost:.1f}, {row['wall_s']}s", flush=True)

    numbers = inst["lab_backpack_n6"]["numbers"]
    _, H, offset, cand = build(numbers, 1)
    pre = np.load(ROOT / "utils" / "pretrained_parameters_small.npy")
    out["pretrained_n6_exact"] = exact_dist_stats(numbers, cand, pre)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print("pretrained n=6 exact:", out["pretrained_n6_exact"], flush=True)
    print("saved results/qaoa-vs-baselines.json", flush=True)


if __name__ == "__main__":
    main()
