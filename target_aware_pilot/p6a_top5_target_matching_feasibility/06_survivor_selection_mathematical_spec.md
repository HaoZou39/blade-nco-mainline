# Survivor Selection Mathematical Specification

Let candidates `i=1..N`, references `j=1..48`, transformed vectors `x_i,t_j in R^5`, and binary assignment `a_ij`. The ideal discrete transport problem is:

`min sum_ij a_ij ||x_i-t_j||_2^2`

subject to `sum_j a_ij <= 1` for every candidate, `sum_i a_ij = m_j` for every reference, and `a_ij in {0,1}`. Thus exactly 10,000 unique survivors are selected.

Capacities use 32×208 + 16×209. The extra 16 are chosen by ascending `(SHA256(instance_id), instance_id)`, a frozen rule independent of performance and feature geometry.

| instance_id | quota |
|---|---|
| d198 | 208 |
| d493 | 208 |
| d657 | 209 |
| d1291 | 209 |
| fl417 | 209 |
| p654 | 209 |
| pcb1173 | 208 |
| pr107 | 208 |
| pr136 | 208 |
| pr144 | 208 |
| pr152 | 209 |
| pr226 | 208 |
| pr264 | 208 |
| pr439 | 208 |
| pr1002 | 208 |
| u574 | 209 |
| u724 | 208 |
| u1060 | 209 |
| vm1084 | 208 |
| a280 | 208 |
| ch150 | 208 |
| eil51 | 209 |
| eil101 | 208 |
| pcb442 | 208 |
| pr299 | 208 |
| rat99 | 208 |
| rat195 | 208 |
| rat575 | 208 |
| rat783 | 209 |
| ts225 | 209 |
| tsp225 | 208 |
| bier127 | 208 |
| ch130 | 208 |
| gil262 | 208 |
| kroA100 | 208 |
| kroB100 | 208 |
| kroC100 | 209 |
| kroD100 | 208 |
| kroE100 | 209 |
| kroA150 | 208 |
| kroB150 | 209 |
| kroA200 | 208 |
| kroB200 | 209 |
| lin105 | 209 |
| lin318 | 209 |
| rd100 | 208 |
| rd400 | 208 |
| u159 | 208 |

Cost is unweighted squared Euclidean distance. The vectorized matrix is stored as float32 (21.89 MiB at 119,530×48) and promoted to float64 for HiGHS; optimality is therefore with respect to those stored float32 costs. Feature importance is not a metric weight. Candidate order is hash-ascending.

The implemented scalable prototype solves the full 48-reference bipartite assignment through its linear-program relaxation using SciPy/HiGHS dual simplex. The candidate-capacity/reference-equality incidence matrix is totally unimodular and RHS capacities are integers, so an optimal LP vertex is integral; all audited runs returned zero fractional variables and HiGHS optimal status. Thus the selected solution is the global optimum of the stated stored-cost problem, not a greedy claim. The independent-reference bound remains a deliberately loose diagnostic because it illegally reuses candidates across references.
