# Classical baselines — frozen ladder results

| Instance | n | Sum | Optimal diff | Greedy diff | Anneal diff (best of 3) |
|---|---|---|---|---|---|
| lab_backpack_n6 | 6 | 48 | 0 | 2 | 0 |
| random42_n12 | 12 | 583 | 1 | 1 | 1 |
| random7_n24 | 24 | 1121 | 1 | 3 | 1 |
| numbers_large_prefix_n48 | 48 | 2432 | 0 | 0 | 0 |
| numbers_large_prefix_n96 | 96 | 4819 | 1 | 1 | 1 |
| numbers_large_prefix_n160 | 160 | 7853 | 1 | 1 | 1 |

Annealing steps per run: max(20000, 400*n). Exact = bitset DP (provably optimal).
Instances n<=24 frozen above; n=48/96/160 are prefixes of utils/numbers_large.npy.
