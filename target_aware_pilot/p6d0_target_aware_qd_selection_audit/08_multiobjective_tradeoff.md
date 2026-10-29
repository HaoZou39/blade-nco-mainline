# Multiobjective Trade-off

Nondominance is computed only over the five frozen strategies in each replicate; it is not a global Pareto front. C is judged against A for target relevance and against B for hardness because these isolate the intended factors. Oracle D shows attainable within-quota hardness; E is the lambda=1 structural endpoint, not a selected optimum.

| replicate | selection | mean_true_m4_ce | frozen_p90_tail_fraction | frozen_p95_tail_fraction | top5_ot_mean | top5_swd | qd_target_relevance_mean | effective_qd_niches | effective_lineages | empirically_dominated_within_five_strategies |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | A | 0.08865202 | 0.1605 | 0.0818 | 22.70493 | 1.590281 | 0.0007139888 | 122.8112 | 478.5243 | True |
| Original | B | 0.07664841 | 0.1062 | 0.0517 | 14.47799 | 1.226971 | 0.002907847 | 265.8669 | 461.3892 | False |
| Original | C | 0.08008513 | 0.1256 | 0.0652 | 14.79454 | 1.240011 | 0.002907847 | 265.8669 | 432.3064 | True |
| Original | D | 0.09853585 | 0.198 | 0.0977 | 14.55145 | 1.2302 | 0.002907847 | 265.8669 | 445.9471 | False |
| Original | E | 0.0705258 | 0.0825 | 0.0431 | 10.5837 | 0.9959385 | 0.003645445 | 304.8056 | 344.5292 | False |
| S1 | A | 0.08965873 | 0.1641 | 0.0852 | 22.79884 | 1.59261 | 0.0006915156 | 116.3955 | 477.0706 | True |
| S1 | B | 0.07722042 | 0.1091 | 0.0552 | 14.6 | 1.234385 | 0.002802538 | 263.2092 | 464.5053 | False |
| S1 | C | 0.08091386 | 0.1285 | 0.0672 | 14.89234 | 1.24626 | 0.002802538 | 263.2092 | 436.6398 | True |
| S1 | D | 0.09946528 | 0.2038 | 0.1005 | 14.63967 | 1.235479 | 0.002802538 | 263.2092 | 442.796 | False |
| S1 | E | 0.07125976 | 0.0888 | 0.0448 | 10.54695 | 0.9952805 | 0.003568918 | 306.7391 | 344.6952 | False |
| S2 | A | 0.08867672 | 0.1611 | 0.0864 | 22.93908 | 1.597184 | 0.000666637 | 117.7119 | 478.8685 | True |
| S2 | B | 0.07666363 | 0.1084 | 0.0571 | 14.57723 | 1.231915 | 0.002840938 | 265.5901 | 460.5387 | False |
| S2 | C | 0.07995952 | 0.1254 | 0.0678 | 14.81347 | 1.241117 | 0.002840938 | 265.5901 | 439.6741 | True |
| S2 | D | 0.09877346 | 0.1972 | 0.1014 | 14.5851 | 1.230752 | 0.002840938 | 265.5901 | 439.6319 | False |
| S2 | E | 0.07066555 | 0.0854 | 0.0457 | 10.49233 | 0.9897239 | 0.00358866 | 313.1831 | 344.2845 | False |
