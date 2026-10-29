# P1 Candidate Coverage Report

## Common feature space

TSPLIB48, the C04 FOGA Parent+Child union, and the C04 BLADE Hard10k were evaluated using the same frozen extractor, F40 order, and calibration transform. The selected dimensions are:

- `spn_distance_varcoeff`
- `spn_fraction_shorter_mean_distance`

Both are dimensionless, invariant to uniform coordinate scaling, finite for all 48 development instances, and show negligible association with TSPLIB node count. The analysis therefore treats their differences as structural rather than as a direct node-scale artifact.

## Target distribution

In frozen transformed F40 coordinates, TSPLIB48 has the following distribution:

| Feature | Mean | SD | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| distance variation coefficient | 0.388 | 0.868 | -0.937 | -0.370 | 0.513 | 0.864 | 2.772 |
| fraction shorter than mean distance | 0.559 | 0.888 | -1.631 | 0.050 | 0.709 | 1.226 | 2.392 |

The three non-random EvoReal structural groups occupy visibly different but overlapping portions of this two-dimensional space. Their detailed instance-level values and S1/S2/S3 labels are in `04_joint_distribution_audit.csv` and `03_tsplib48_reference_manifest.csv`.

Relative to this target, the FOGA pool is shifted toward larger distance variation and lower fraction-shorter values (medians 1.027 and -0.535). Hard10k is shifted farther in the same direction (medians 1.279 and -0.881). Thus the current hardness-only acquisition does not reproduce the TSPLIB48 structural mix.

## Coarse and smoothed coverage definitions

Because there are only 48 references, no fine two-dimensional grid was fit.

- The primary target region is a smoothed union of disks after robust target-median/IQR standardization. Its radius, 0.09596, is the median leave-one-out nearest-neighbor distance among the 48 targets.
- A target-derived 3x3 tercile grid and the overall target min/max box are reported as coarse sensitivity views, not as a precise density model.

## Coverage results

| Measure | FOGA union | BLADE Hard10k |
|---|---:|---:|
| Unique instances | 119,530 | 10,000 |
| TSPLIB target points reached by nearest candidate | 45/48 (93.75%) | 45/48 (93.75%) |
| Unique candidates in smoothed target region | 8,969 | 729 |
| Fraction of source in smoothed target region | 7.50% | 7.29% |
| Unique candidates in broad target min/max box | 115,294 | 9,353 |
| Coarse occupied cells | 9 | 8 |
| Coarse effective occupied cells | 3.63 | 2.59 |
| Largest coarse-cell share | 55.0% | 72.5% |
| Jensen-Shannon divergence from target coarse distribution | 0.499 | 0.568 |

The three references not reached under the strict smoothed radius are `d198`, `eil51`, and `ts225`; `ts225` is only marginally outside. The broad box contains many candidates, but it cannot establish adequate density near the 48-point empirical target and must not be interpreted as 10k target-aware capacity.

Hard10k is substantially concentrated: 72.5% lies in one coarse cell, versus 22.9% of TSPLIB48. Only 729 Hard10k instances fall in the strict target region. This supports a real structural-coverage mismatch rather than complete target coverage by the current hardness-only acquisition.

## Candidate provenance and reachability clue

Within the smoothed target region, approximately 8,264 candidates are new offspring and 707 are prior parents (counts can overlap at provenance boundaries). All 11 mutation operators contribute. The largest operator contributions are grid (977), linear projection (821), axis projection (786), explosion (763), compression (744), and uniform (724), with the remaining operators each contributing hundreds.

This shows broad operator reach, but the current single-cycle union provides only 8,969 strict-region unique instances—1,031 short of the later 10,000-instance goal—and still misses three target neighborhoods. Parent-pool and mutation-reachability analysis is therefore required before committing to P2.

## Predicted hardness in the target region

Using the confirmed M4-conditioned HGB40:

- FOGA target-region predicted `log1p_LBQ`: mean 0.07396, P90 0.09234.
- All FOGA candidates: mean 0.07601.
- Hard10k target-region predicted `log1p_LBQ`: mean 0.10439.

The target region is not intrinsically harder on average than the full candidate pool under this surrogate. Hardness-only ranking selects the harder subset within target neighborhoods, but it supplies too few candidates there and overweights a small part of the structural space.

## EvoReal diagnostic features

The corrected repository implementation was used on TSPLIB48 only:

| Group | n | FFT Energy mean / median | NN-CV mean / median |
|---|---:|---:|---:|
| S1 | 19 | 77.74 / 63.61 | 0.913 / 0.719 |
| S2 | 12 | 15.66 / 13.99 | 0.288 / 0.292 |
| S3 | 17 | 14.35 / 13.55 | 0.613 / 0.562 |

FFT Energy captures frequency-domain occupancy structure, while nearest-neighbor CV captures local spacing irregularity. The selected HGB features summarize the global pair-distance distribution. They therefore provide a different diagnostic view, although they should not be claimed to be statistically orthogonal without a larger sample.

## P1 conclusion

FOGA reaches most target neighborhoods and provides a plausible acquisition substrate, but the existing pool is not yet sufficient for a strict 10k target-region acquisition. BLADE Hard10k has a clear distribution mismatch and stronger concentration. The evidence supports reachability analysis and candidate-pool expansion/reshaping before P2, not immediate data generation.
