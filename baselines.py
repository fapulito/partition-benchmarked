"""Classical baselines for the number-partition problem (Warsaw QFF 2026, Track 1).

The hackathon rubric asks for QAOA benchmarked against classical baselines,
scaling behaviour, and an honest account of where QAOA falls short. This module
provides the classical side:

- exact_partition: bitset DP, optimal for n up to a few hundred (values bounded).
- greedy_partition: largest-first heuristic (LPT-style), O(n log n).
- anneal_partition: simulated annealing over sign flips.
- approx_ratio: QAOA-vs-exact quality measure. For partition, the natural
  figure of merit is the residual difference; we report both the raw
  difference and the "energy ratio" E/E_opt on the Ising objective
  E(s) = (sum_i s_i * a_i)^2, where lower is better and 0 is perfect.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def difference(signs: Sequence[int], numbers: Sequence[int]) -> int:
    """|sum(s_i * a_i)| for signs in {+1, -1}."""
    return abs(sum(s * a for s, a in zip(signs, numbers)))


def exact_partition(numbers: Sequence[int]) -> Tuple[List[int], int]:
    """Optimal partition via bitset DP over achievable sums.

    Returns (signs, best_difference). Exact; cost grows with total sum,
    fine for hackathon-scale instances.
    """
    total = sum(numbers)
    # reachable[s] = bitmask over items achieving sum s is overkill; use the
    # classic trick: bitset where bit s set <=> some subset sums to s.
    bits = 1
    for a in numbers:
        bits |= bits << a
    best = min(
        (total - 2 * s) for s in range(total // 2, -1, -1) if (bits >> s) & 1
    )
    target = (total - best) // 2  # subset sum achieving the optimum
    # Reconstruct one optimal subset by walking back.
    subset = []
    s = target
    # Rebuild reachability per prefix to reconstruct.
    reach = [1]
    for a in numbers:
        reach.append(reach[-1] | (reach[-1] << a))
    for i in range(len(numbers), 0, -1):
        a = numbers[i - 1]
        if s >= a and (reach[i - 1] >> (s - a)) & 1:
            subset.append(i - 1)
            s -= a
    signs = [-1] * len(numbers)
    for i in subset:
        signs[i] = 1
    return signs, best


def greedy_partition(numbers: Sequence[int]) -> Tuple[List[int], int]:
    """Largest-first: assign each number (desc) to the currently lighter side."""
    order = sorted(range(len(numbers)), key=lambda i: -numbers[i])
    signs = [0] * len(numbers)
    sums = [0, 0]
    for i in order:
        side = 0 if sums[0] <= sums[1] else 1
        sums[side] += numbers[i]
        signs[i] = 1 if side == 0 else -1
    return signs, abs(sums[0] - sums[1])


def anneal_partition(
    numbers: Sequence[int],
    steps: int = 20000,
    seed: int = 42,
    t0: float | None = None,
) -> Tuple[List[int], int]:
    """Simulated annealing with single-sign flips."""
    rng = random.Random(seed)
    n = len(numbers)
    signs = [rng.choice((1, -1)) for _ in range(n)]
    cur_signed = sum(s * a for s, a in zip(signs, numbers))
    cur = abs(cur_signed)
    best_signs, best = signs[:], cur
    if t0 is None:
        t0 = max(1.0, max(numbers) / 2.0)
    for k in range(steps):
        temp = t0 * (1 - k / steps) + 1e-9
        i = rng.randrange(n)
        delta_signed = -2 * signs[i] * numbers[i]
        new = abs(cur_signed + delta_signed)
        if new <= cur or rng.random() < math.exp(-(new - cur) / temp):
            signs[i] *= -1
            cur_signed += delta_signed
            cur = new
            if cur < best:
                best, best_signs = cur, signs[:]
                if best == 0:
                    break
    return best_signs, best


def energy(signs: Sequence[int], numbers: Sequence[int]) -> int:
    """Ising objective E = (sum s_i a_i)^2 (0 = perfect partition)."""
    d = sum(s * a for s, a in zip(signs, numbers))
    return d * d


def report(numbers: Sequence[int], qaoa_signs: Sequence[int]) -> dict:
    """Benchmark one QAOA result against all classical baselines."""
    _, d_opt = exact_partition(numbers)
    _, d_greedy = greedy_partition(numbers)
    _, d_sa = anneal_partition(numbers)
    d_q = difference(qaoa_signs, numbers)
    return {
        "n": len(numbers),
        "diff_optimal": d_opt,
        "diff_greedy": d_greedy,
        "diff_anneal": d_sa,
        "diff_qaoa": d_q,
        "qaoa_optimal": d_q == d_opt,
        "qaoa_vs_greedy": d_q - d_greedy,
    }


if __name__ == "__main__":
    demo = [3, 5, 7, 9, 11, 13]  # the lab's backpack example (perfect: 24/24)
    print(report(demo, [1, 1, -1, -1, 1, -1]))
    rng = random.Random(7)
    hard = [rng.randint(10, 99) for _ in range(24)]
    s_opt, d_opt = exact_partition(hard)
    s_sa, d_sa = anneal_partition(hard)
    s_g, d_g = greedy_partition(hard)
    print(f"n=24 random: optimal={d_opt} anneal={d_sa} greedy={d_g}")
