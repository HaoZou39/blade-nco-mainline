# Final Research Conclusion

| hypothesis | decision |
|---|---|
| P6E05_POOL_REACHABILITY | PARTIAL |
| P6E05_GENERATION_BOTTLENECK | MIXED |
| P6E05_NICHE_QUOTA_BOTTLENECK | NOT_SUPPORTED |
| P6E05_SELECTION_BOTTLENECK | NOT_SUPPORTED |
| P6E05_CROSS_SIZE_ROBUSTNESS | LIMITED |
| P6E05_REPRESENTATION_ROBUSTNESS | MIXED |
| P6E05_NEXT_STEP | NEXT_STEP_REPRESENTATION_VALIDATION |

| question | answer |
|---|---|
| Full U_s coverage | Original 25/48; S1 24/48; S2 25/48 |
| B/C misses recoverable in pool | Only B-S2 misses one pool-reachable reference; C misses none |
| Frozen-quota 10k maximum | 25, 24, 25; exact certificates |
| Benefit from removing quotas | None: free10k equals quota10k in all seeds |
| Selection headroom | C=0 all seeds; B=0/0/1 reference |
| Matched-size gap | Pools 15/17, 14/17, 15/17; selections essentially at ceilings |
| Representation dependence | Yes; Top5 supports incomplete reachability but different missing IDs |
| EvoReal comparison | EvoReal10k covers more, but FFT_APPROX/size confounding prevents a generation-value claim |
| Unselected candidate hardness | Existing candidates have nonzero M4 CE/tails; no training-value inference |
| Next step | NEXT_STEP_REPRESENTATION_VALIDATION |

The frozen QD quotas and within-niche B/C rules are not the principal FFT–NN reachability bottleneck. The residual pool gap is partly consistent across representations but not reference-identical and remains entangled with FFT approximation and node size. Do not modify FOGA or train a coverage witness yet. Structural reachability is not training value.

`BLADE_P6E05_CANDIDATE_POOL_STRUCTURAL_REACHABILITY_COMPLETE`
