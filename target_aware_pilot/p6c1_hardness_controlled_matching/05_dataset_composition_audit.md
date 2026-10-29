# Dataset Composition Audit

Selection used only frozen Top5 cost and, for C1, frozen hardness-bin equalities. The following lineage, source, SWD/energy and operator diagnostics are post-selection descriptions and did not affect optimization. Restoring hardness necessarily changes membership and may alter other F40/provenance properties, so any later C0/C1 difference is the total effect of adding the hardness-distribution constraint, not a pure per-instance hardness effect.

| replicate | arm | membership_sha256 | mean_m4_ce | p90_tail_fraction | p95_tail_fraction | e0_source_fraction | e1_source_fraction | both_source_fraction | top5_swd | top5_energy | effective_lineages | largest_lineage_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | C0 | 1ace440c24b61ecf5795d29a9af7fbb5251fae0f3a97a76da84e62f4fa84c2cd | 0.061685407 | 0.0362 | 0.0149 | 0.0091 | 0.998 | 0.0071 | 0.73088064 | 0.80413494 | 225.60933 | 0.0416 |
| Original | C1 | 6e8725e29ff61baf89ea32de2ed97f2ca468c17eec2ffb5dd0c4be29d198b012 | 0.088040112 | 0.1605 | 0.0819 | 0.2811 | 0.724 | 0.0051 | 0.84161205 | 1.0237249 | 323.8829 | 0.0386 |
| S1 | C0 | f8429e49bf0db1e13a43cdfc46c1c494fda16456d0ae4d373594eede37129b97 | 0.061570402 | 0.0391 | 0.0142 | 0.009 | 0.9973 | 0.0063 | 0.7323203 | 0.82408094 | 216.16777 | 0.0426 |
| S1 | C1 | c9d18de72c5ebeeb00d9802ca8ed7474f5d037af43bf8d4f2719089ac30e0c86 | 0.089402677 | 0.1665 | 0.0844 | 0.3035 | 0.701 | 0.0045 | 0.86085293 | 1.0472745 | 326.20234 | 0.0385 |
| S2 | C0 | ad3acdb50ae497807c6dcda646dc1096df3f0caed04983f475899f261d596826 | 0.061050167 | 0.0344 | 0.0136 | 0.009 | 0.9983 | 0.0073 | 0.73071333 | 0.80561017 | 221.73601 | 0.041 |
| S2 | C1 | a027f23ff8286a763d10333dcd0da2f4342e7f3aada75a99be33f1837cb37bc0 | 0.088356875 | 0.1622 | 0.0873 | 0.2882 | 0.7175 | 0.0057 | 0.84666148 | 1.0138812 | 314.5407 | 0.0378 |

Full selected manifests preserve raw/transformed Top5 and provenance; the P6-C0 frozen Full-F40 cache remains the authority for all 40 transformed inputs. Operator composition is in `logs/operator_composition.csv`. The formal post-gate Full-F40 comparison was not run because the joint feasibility gate failed; these are feasibility-stage diagnostics only.
