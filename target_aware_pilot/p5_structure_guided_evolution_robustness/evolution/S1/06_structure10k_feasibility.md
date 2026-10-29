# Structure10k Feasibility

- E0: 10,000 unique acquired; JS 0.20714677, SWD 0.370804, strict 2,954 (29.54%), coverage 46/48, effective lineages 564.9, max lineage 1.21%, auxiliary JS 0.498323.
  Uncovered reference neighborhoods: ['d198', 'eil51'].
  Capacity-aware quota fulfillment:
  | Cell | Requested | Capacity | Adjusted | Selected |
  |---:|---:|---:|---:|---:|
  | 0 | 2,292 | 1,525 | 1,525 | 1,525 |
  | 1 | 1,042 | 146 | 146 | 146 |
  | 2 | 0 | 0 | 0 | 0 |
  | 3 | 208 | 7,621 | 1,660 | 1,660 |
  | 4 | 2,083 | 1,423 | 1,423 | 1,423 |
  | 5 | 1,042 | 46 | 46 | 46 |
  | 6 | 833 | 12,177 | 2,285 | 2,285 |
  | 7 | 208 | 5,087 | 1,659 | 1,659 |
  | 8 | 2,292 | 1,256 | 1,256 | 1,256 |
- E1: 10,000 unique acquired; JS 0.06510613, SWD 0.239681, strict 4,030 (40.30%), coverage 46/48, effective lineages 531.7, max lineage 1.24%, auxiliary JS 0.513315.
  Uncovered reference neighborhoods: ['d198', 'eil51'].
  Capacity-aware quota fulfillment:
  | Cell | Requested | Capacity | Adjusted | Selected |
  |---:|---:|---:|---:|---:|
  | 0 | 2,292 | 2,738 | 2,640 | 2,640 |
  | 1 | 1,042 | 307 | 307 | 307 |
  | 2 | 0 | 1 | 0 | 0 |
  | 3 | 208 | 7,426 | 557 | 557 |
  | 4 | 2,083 | 2,470 | 2,432 | 2,432 |
  | 5 | 1,042 | 90 | 90 | 90 |
  | 6 | 833 | 8,248 | 1,182 | 1,182 |
  | 7 | 208 | 6,037 | 556 | 556 |
  | 8 | 2,292 | 2,236 | 2,236 | 2,236 |

Only E1, if complete, is exported under the required `provisional_structure10k*` names. E0 is retained as a control under `support_e0_provisional_structure10k*`. All instances are eligible offspring and no P03 survivor is used merely because it survived.

The exact frozen quotas are infeasible in several bounded-archive cells. Those deficits are explicitly redistributed with frozen target weights and available capacity; they are not hidden by outside-box or duplicate backfill. Consequently, coarse JS is reported alongside SWD, reference coverage, auxiliary features, and lineage metrics rather than treated as a standalone success criterion.
