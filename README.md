# Partition, Benchmarked — QAOA vs Classical Baselines 🧮

**Qiskit Fall Fest 2026 · Warsaw — Track 1: Quantum Optimization**
Author: David Anderson (solo entry)

A benchmark study of QAOA on the **number partition problem**, built on the
Warsaw Lab 4a materials and executed end-to-end on real hardware
(**ibm_fez**, 156-qubit Heron r2). Every quantum result is reported next to
the classical numbers it has to answer to: an exact solver, greedy
largest-first, and simulated annealing. No quantum advantage is claimed —
where QAOA and the quantum pipeline fall short, the tables say so.

**Festival grader: Lab 4a — 15/15 exercises passed ✅**

## 🎯 The problem

Assign each number `a_i` a sign `s_i ∈ {+1, −1}` minimising
`(Σ s_i a_i)²`. Expanding the square gives an Ising model with couplings
`J_ij = 2 a_i a_j` — equivalently a max-cut on the complete graph with edge
weights `a_i · a_j`. Quality is measured by the residual difference
`|Σ s_i a_i|`: zero is a perfect partition.

## 🛠️ What was done

1. **Classical baselines** (`baselines.py`, `run_baselines.py`) — exact
   bitset-DP solver, greedy LPT, simulated annealing, run on a frozen
   instance ladder from n = 6 to n = 160.
2. **QAOA construction** (notebook §1) — partition graph → MaxCut → Ising
   Hamiltonian → annotated QAOA ansatz, trained and sampled on a simulator
   (`run_benchmark.py`).
3. **Error-mitigation comparison on hardware** (notebook §2) — the same
   circuit sampled on ibm_fez four ways: no mitigation, error suppression
   only, M3 readout mitigation only, M3 + suppression.
4. **Scaling via Pauli Correlation Encoding** (notebook §§3–4) — the
   n = 160 instance encoded onto 11 qubits; four estimator scenarios
   (No-EM / TREX / ZNE / PEC) run on ibm_fez with pretrained parameters.

## 📊 Results

### 🏁 Classical baselines (exact optimum vs heuristics)

| Instance | n | Optimal | Greedy | Annealing |
|---|---|---|---|---|
| lab backpack | 6 | 0 | 2 | 0 |
| random (seed 42) | 12 | 1 | 1 | 1 |
| random (seed 7) | 24 | 1 | 3 | 1 |
| numbers_large prefix | 48 | 0 | 0 | 0 |
| numbers_large prefix | 96 | 1 | 1 | 1 |
| numbers_large prefix | 160 | 1 | 1 | 1 |

Simulated annealing matches the exact optimum at **every** size. That is
the bar.

### 🌀 Trained QAOA (simulator, most-likely sampled partition)

| Instance | p | QAOA difference | Optimal | P(optimal) |
|---|---|---|---|---|
| backpack n=6 | 1 | 2 | 0 | 0.060 |
| backpack n=6 | 2 | 2 | 0 | 0.013 |
| random42 n=12 | 1 | 93 | 1 | 0.003 |
| random42 n=12 | 2 | 39 | 1 | 0.005 |

At n = 6 QAOA ties greedy but misses the optimum; at n = 12 it falls well
behind while annealing stays exact.

### 🔬 Error mitigation on ibm_fez (notebook §2, difference of best partition)

| Version | Difference |
|---|---|
| V1 — no mitigation | **0** |
| V2 — suppression only | 4 |
| V3 — M3 only | **0** (best bitstring `000011`) |
| V4 — M3 + suppression | 4 |

A perfect partition was found on real hardware (V1 and V3). Suppression
moved the sampled mode off the optimum in this instance; reported as-is.

### 📈 PCE scaling on ibm_fez (n = 160 → 11 qubits, notebook §4)

| Scenario | Loss | Decoded difference |
|---|---|---|
| No error mitigation | 56,106.51 | 1,365 |
| TREX | −2,108.07 | **351** |
| ZNE | −555.35 | 491 |
| PEC | 14,090.55 | 755 |
| Classical optimum (exact / annealing) | — | **1** |

On an ideal simulator all four scenarios return identical values (the
options are Runtime-only); on hardware the comparison is real: TREX
improves the decoded partition roughly 4× over no mitigation — and still
lands far from the classical optimum.

> 🔀 **Run-to-run variability is part of the result.** The lab's own
> reference run of this section (on ibm_pittsburgh, reproduced in the
> notebook) ranked **PEC first** by decoded difference (≈115), with TREX
> third (≈985) and a TREX loss of ≈ +94,000. On ibm_fez the top of the
> table reversed — **TREX first** (351) with a loss of −2,108. The stable
> conclusion across both devices: every mitigation method beats no
> mitigation by a wide margin, and *which* method wins is device- and
> run-dependent. Raw loss values are not comparable across runs; the
> decoded difference is the checkable metric.

## ⚖️ Verdict

For number partitioning at benchmark sizes, classical methods set a bar
that QAOA on current hardware does not clear — and that is the finding.
What the quantum pipeline demonstrably provides is a complete, classically
verifiable workflow: every sampled partition can be checked by summation,
which is why partition is the right teaching and benchmarking problem.
The quantum result never has to be taken on trust.

## 🗂️ Repository layout

- `lab_4a_track1_working.ipynb` — the Lab 4a notebook with all exercises
  completed and graded (15/15), including the ibm_fez hardware runs.
- `baselines.py` — classical solvers (exact DP, greedy, annealing) and
  scoring helpers.
- `run_baselines.py`, `run_benchmark.py` — reproduce the baseline ladder
  and the trained-QAOA benchmark.
- `results/` — frozen result files: `baselines-ladder.{json,md}`,
  `qaoa-vs-baselines.json`, `hardware-summary.json` (the ibm_fez §2/§4
  numbers quoted above).
- `utils/` — Lab 4a support code and data files (mirrored from the lab
  distribution; see Credits).

## ⚙️ Environment setup

Python 3.12 recommended.

```bash
python -m venv venv && source venv/bin/activate   # or a conda environment
pip install "qiskit[visualization]" qiskit-aer qiskit-ibm-runtime \
    qiskit-addon-opt-mapper mthree networkx rustworkx \
    "qfest-grader>=0.3" \
    "git+https://github.com/qiskit-community/qopt-best-practices.git"
```

## ▶️ Running

- The baselines and the benchmark scripts run anywhere:
  `python run_baselines.py`, `python run_benchmark.py`.
- The notebook runs on an ideal simulator by default. To reproduce the
  hardware sections you need an IBM Quantum account: save it once with
  `QiskitRuntimeService.save_account(token=..., instance=...)` using your
  own API key and instance CRN, then select the least-busy 156-qubit
  backend as shown in the notebook. (Free Open Plan usage was sufficient:
  each §2 job consumed ≈2 s of QPU time.)
- Grader cells use `qfest_grader` with the email you registered with:
  `save_account("your email")`.

## 🙏 Credits and license

Built on Qiskit Fall Fest 2026 Warsaw Lab 4a materials by AI Quantum Tech,
derived from the IBM Qiskit Global Summer School (MIT). The `utils/`
folder is mirrored from the lab distribution under its own license file.
Classical baselines, the benchmark harness, and this write-up are original
work, released under the MIT license in `LICENSE`.
