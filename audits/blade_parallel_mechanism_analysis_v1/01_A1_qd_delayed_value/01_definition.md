# A1 definitions

Frozen range: C00–C04.

C_s = unique(P_{s-1} UNION O_s), the stored corrected union `C_union_scored`.

P_s^GP = global GP Top10k(C_s) using stored `gp_predicted_log1p_hardness` and the formal (-score, hash) lexsort.

P_s^FULL = actual stored parents (QD floor + GP fill).

QD_RESCUED_s = P_s^FULL \ P_s^GP
SHARED_GP_s = P_s^FULL ∩ P_s^GP
GP_ONLY_DISPLACED_s = P_s^GP \ P_s^FULL

Descendants of parent x selected at cycle s are mutation children generated from x (or from later selected descendants of x) in cycles s+1..s+k inside the frozen snapshot. The original parent surviving into a later P/T is not counted as a descendant event.

New niche: descendant niche not occupied by any P_t for t ≤ s.

Exact-hard descendant: descendant enters historically labelled T_{s+k} and exact current hardness ≥ T_{s+k} P90. Missing exact labels are marked unavailable; no Concorde calls were made.

Matched control uses SHARED_GP only, so both sides generate lineages. Matching uses GP percentile, age category, and provenance. No outcome entered matching.
