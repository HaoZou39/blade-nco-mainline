# P2.0 Target Reachability Comparison

## Equal-budget outcome

| Metric | E0 random | E1 structure-guided |
|---|---:|---:|
| Mutation attempts | 110,000 | 110,000 |
| New unique offspring | 109,600 | 109,618 |
| New strict-region unique | 8,645 | 31,707 |
| New strict-region fraction | 7.888% | 28.925% |
| Cumulative strict-region unique | 17,614 | 40,676 |
| Cumulative neighborhood coverage | 45/48 | 47/48 |
| Cumulative JS divergence | 0.4793 | 0.3206 |
| Cumulative largest cell share | 51.793% | 41.232% |
| Cumulative effective cells | 3.859 | 5.210 |
| Child proximity-improved fraction | 50.293% | 28.657% |
| Valid mutation rate | 99.646% | 99.670% |

Both cumulative and new-offspring-only metrics are reported; the comparison does not treat crossing 10,000 cumulative target-region candidates as sufficient by itself.

## Previously uncovered references

- `d198`: baseline 0.279270, E0 0.279270, E1 0.079765.
- `eil51`: baseline 0.256017, E0 0.250353, E1 0.184621.
- `ts225`: baseline 0.097558, E0 0.097312, E1 0.052587.

## Leading operator contributions

By total new offspring retained in the strict region:

- E0 `grid_mutation`: 1035 target-region new unique; 799 entries from outside parents.
- E0 `axis_projection_mutation`: 848 target-region new unique; 750 entries from outside parents.
- E0 `linear_projection_mutation`: 832 target-region new unique; 715 entries from outside parents.
- E0 `rotation_mutation`: 766 target-region new unique; 543 entries from outside parents.
- E1 `normal_mutation`: 5480 target-region new unique; 169 entries from outside parents.
- E1 `implosion_mutation`: 3997 target-region new unique; 213 entries from outside parents.
- E1 `explosion_mutation`: 3646 target-region new unique; 298 entries from outside parents.
- E1 `compression_mutation`: 3326 target-region new unique; 306 entries from outside parents.

By genuine entry from a parent outside the strict region:

- E0 `grid_mutation`: 799 strict-region entries from parents outside the region.
- E0 `axis_projection_mutation`: 750 strict-region entries from parents outside the region.
- E0 `linear_projection_mutation`: 715 strict-region entries from parents outside the region.
- E0 `cluster_mutation`: 673 strict-region entries from parents outside the region.
- E1 `grid_mutation`: 595 strict-region entries from parents outside the region.
- E1 `rotation_mutation`: 530 strict-region entries from parents outside the region.
- E1 `axis_projection_mutation`: 499 strict-region entries from parents outside the region.
- E1 `linear_projection_mutation`: 492 strict-region entries from parents outside the region.

## Auxiliary diagnostic

The unselected FFT Energy / NN-CV diagnostic has coarse target-box JS divergence 0.5210 for E0 and 0.5006 for E1. This diagnostic is not used for parent selection. FFT uses raw-count 32x32 occupancy and is node-count sensitive because all generated candidates have 100 nodes while TSPLIB48 spans multiple sizes; it is therefore interpreted only as a warning against claiming full structural equivalence.
