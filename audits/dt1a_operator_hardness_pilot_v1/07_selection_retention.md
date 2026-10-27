# Selection retention

Mutation reachability and T05 retention differ sharply in this pilot.

- 326 valid unique children were generated from 330 attempts.
- 185 children improved raw M5 hardness; 177 of them did not enter T05. Only eight improved children were retained.
- 62 children hit the frozen hard region; 60 did not enter T05. Only two hard hits were retained.
- Fourteen total children entered T05: eight as `quality_qd_elite`, six through `hgb_fill`.
- Retained children had mean `delta_h=0.02694` versus `0.00942` for nonretained valid children, but this is selection-conditioned and not causal.

Implosion generated 10 hard hits and retained one of them; compression generated seven and retained none; normal generated four and retained one. Full per-operator counts, including failures, are in `work/selection_retention_summary.csv`.

The result is evidence that the frozen mutation schedule can reach hard M5 instances, while the frozen HGB/Elite-QD acquisition retains only a small fraction. It does not show that low retention is harmful without a counterfactual selection analysis, and it does not equate selection survival with operator quality.

Identity: T05 membership is an exact-hash join to frozen `C05/T_hard_top10k_membership.csv` (SHA-256 `eef567ccfb243e81f61445cc413f23f9fbdfbf037a4b5efe731ad3c74312567c`); M5 checkpoint SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`; 30 parents/330 attempts.

