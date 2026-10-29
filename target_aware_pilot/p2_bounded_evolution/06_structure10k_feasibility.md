# Structure10k Feasibility

- E0: 10,000 unique acquired; JS 0.21029513, SWD 0.366734, strict 3,016 (30.16%), coverage 45/48, effective lineages 563.7, max lineage 1.20%, auxiliary JS 0.493910.
  Uncovered reference neighborhoods: ['d198', 'eil51', 'ts225'].
  Capacity-aware quota fulfillment:
  | Cell | Requested | Capacity | Adjusted | Selected |
  |---:|---:|---:|---:|---:|
  | 0 | 2,292 | 1,451 | 1,451 | 1,451 |
  | 1 | 1,042 | 136 | 136 | 136 |
  | 2 | 0 | 1 | 0 | 0 |
  | 3 | 208 | 7,668 | 1,671 | 1,671 |
  | 4 | 2,083 | 1,543 | 1,543 | 1,543 |
  | 5 | 1,042 | 50 | 50 | 50 |
  | 6 | 833 | 11,965 | 2,296 | 2,296 |
  | 7 | 208 | 5,247 | 1,670 | 1,670 |
  | 8 | 2,292 | 1,183 | 1,183 | 1,183 |
- E1: 10,000 unique acquired; JS 0.06973472, SWD 0.239921, strict 4,120 (41.20%), coverage 45/48, effective lineages 517.3, max lineage 1.28%, auxiliary JS 0.516381.
  Uncovered reference neighborhoods: ['d198', 'eil51', 'ts225'].
  Capacity-aware quota fulfillment:
  | Cell | Requested | Capacity | Adjusted | Selected |
  |---:|---:|---:|---:|---:|
  | 0 | 2,292 | 2,759 | 2,683 | 2,683 |
  | 1 | 1,042 | 289 | 289 | 289 |
  | 2 | 0 | 1 | 0 | 0 |
  | 3 | 208 | 7,433 | 599 | 599 |
  | 4 | 2,083 | 2,617 | 2,475 | 2,475 |
  | 5 | 1,042 | 92 | 92 | 92 |
  | 6 | 833 | 8,080 | 1,224 | 1,224 |
  | 7 | 208 | 6,249 | 599 | 599 |
  | 8 | 2,292 | 2,039 | 2,039 | 2,039 |

Only E1, if complete, is exported under the required `provisional_structure10k*` names. E0 is retained as a control under `support_e0_provisional_structure10k*`. All instances are eligible offspring and no P03 survivor is used merely because it survived.

The exact frozen quotas are infeasible in several bounded-archive cells. Those deficits are explicitly redistributed with frozen target weights and available capacity; they are not hidden by outside-box or duplicate backfill. Consequently, coarse JS is reported alongside SWD, reference coverage, auxiliary features, and lineage metrics rather than treated as a standalone success criterion.
