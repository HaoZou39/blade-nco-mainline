# Structure10k Feasibility

- E0: 10,000 unique acquired; JS 0.20661169, SWD 0.369847, strict 2,985 (29.85%), coverage 46/48, effective lineages 556.5, max lineage 1.38%, auxiliary JS 0.493918.
  Uncovered reference neighborhoods: ['d198', 'eil51'].
  Capacity-aware quota fulfillment:
  | Cell | Requested | Capacity | Adjusted | Selected |
  |---:|---:|---:|---:|---:|
  | 0 | 2,292 | 1,437 | 1,437 | 1,437 |
  | 1 | 1,042 | 146 | 146 | 146 |
  | 2 | 0 | 0 | 0 | 0 |
  | 3 | 208 | 7,703 | 1,670 | 1,670 |
  | 4 | 2,083 | 1,538 | 1,538 | 1,538 |
  | 5 | 1,042 | 70 | 70 | 70 |
  | 6 | 833 | 12,067 | 2,296 | 2,296 |
  | 7 | 208 | 5,131 | 1,670 | 1,670 |
  | 8 | 2,292 | 1,173 | 1,173 | 1,173 |
- E1: 10,000 unique acquired; JS 0.06807460, SWD 0.242425, strict 4,003 (40.03%), coverage 45/48, effective lineages 510.6, max lineage 1.33%, auxiliary JS 0.515574.
  Uncovered reference neighborhoods: ['d198', 'eil51', 'ts225'].
  Capacity-aware quota fulfillment:
  | Cell | Requested | Capacity | Adjusted | Selected |
  |---:|---:|---:|---:|---:|
  | 0 | 2,292 | 2,616 | 2,616 | 2,616 |
  | 1 | 1,042 | 266 | 266 | 266 |
  | 2 | 0 | 0 | 0 | 0 |
  | 3 | 208 | 7,579 | 604 | 604 |
  | 4 | 2,083 | 2,539 | 2,480 | 2,480 |
  | 5 | 1,042 | 116 | 116 | 116 |
  | 6 | 833 | 8,223 | 1,229 | 1,229 |
  | 7 | 208 | 6,149 | 604 | 604 |
  | 8 | 2,292 | 2,085 | 2,085 | 2,085 |

Only E1, if complete, is exported under the required `provisional_structure10k*` names. E0 is retained as a control under `support_e0_provisional_structure10k*`. All instances are eligible offspring and no P03 survivor is used merely because it survived.

The exact frozen quotas are infeasible in several bounded-archive cells. Those deficits are explicitly redistributed with frozen target weights and available capacity; they are not hidden by outside-box or duplicate backfill. Consequently, coarse JS is reported alongside SWD, reference coverage, auxiliary features, and lineage metrics rather than treated as a standalone success criterion.
