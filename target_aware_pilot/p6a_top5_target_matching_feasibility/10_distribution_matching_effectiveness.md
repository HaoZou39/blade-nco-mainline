# Distribution Matching Effectiveness

| selection | selected_n | unique_n | assignment_total_squared_cost | assignment_mean_squared_cost | sliced_wasserstein_5d | energy_distance_5d | nearest_reference_distance_mean | nearest_reference_distance_p95 | effective_lineages_diagnostic | largest_lineage_share_diagnostic | effective_operators_diagnostic | internal_nn_distance_sample_mean_diagnostic | old2_sliced_wasserstein |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random10k | 10000 | 10000 | 258676 | 25.8676 | 1.7076 | 4.5385 | 2.46969 | 2.77527 | 492.568 | 0.014 | 10.9911 | 0.117812 | 0.635679 |
| TargetMatching10k | 10000 | 10000 | 155665 | 15.5665 | 1.27058 | 2.52701 | 1.89025 | 2.5773 | 426.721 | 0.0181 | 9.81265 | 0.255989 | 0.439596 |

Relative to fixed-seed Random10k from the same union, matching changes mean assigned squared cost from 25.8676 to 15.5665, 5D sliced Wasserstein from 1.7076 to 1.27058, and 5D energy distance from 4.5385 to 2.52701.

The improvement is expected because selection optimizes target proximity. Independent SWD/energy/marginal/nearest-reference diagnostics confirm it is not only a bookkeeping change in the optimized objective. Raw Top5 and legacy two-dimensional metrics are reported diagnostically in `diagnostics/`; no diversity/lineage/operator penalty was added after inspecting them.

This establishes structural-distribution matching on this fixed bank only. It is not evidence that the selected data has greater BQ training value.
