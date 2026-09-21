# EvoReal S1/S2/S3 structural alignment audit

This audit does not reimplement the EvoReal generator/FFT training pipeline.
It asks whether the frozen LEHD-evaluation splits still carry the paper's two-feature structural meaning.

**Verdict: PARTIAL**

Only subset of categories match EvoReal criteria.

## Section 1. Dataset description

Instances are the frozen external-v2 tensors used in LEHD greedy evaluation.
Coordinates were not regenerated or modified. Instance IDs are preserved.

- EvoReal_S1_100: 1000 instances, frozen SHA-256 `28c8e4c0b8f57f840e6b444e45a21c795993205faae4e27ffc9dcae1df55448f`
- EvoReal_S2_100: 1000 instances, frozen SHA-256 `f393b1be940e09ff3f27c5818fb8b721dd240891d0cda22fedefc10e28c7256c`
- EvoReal_S3_100: 1000 instances, frozen SHA-256 `d1f74c229e6efc609845f3570e58fa199158b5feaafc526792c2b3ee9dbab48f`

Operational features:

- FFT energy: per-instance min-max to [0,1], 128x128 occupancy histogram, 2D FFT with DC removed, `sum|F|^2 / sum|I|^2`.
- NN-ratio: Euclidean `mean min-neighbor distance / mean pairwise distance`.
- These are alignment probes, not a claim of bitwise identity with any unpublished EvoReal FFT code.

Node-count check (scale confound):

Dataset,N_mean,N_std,N_min,N_max,n_instances
S1,100.0,0.0,100,100,1000
S2,100.0,0.0,100,100,1000
S3,100.0,0.0,100,100,1000


All three splits are N=100. Later LEHD gap differences cannot be attributed to an N=100 vs N=500 scale mismatch.

Occupancy-FFT caveat: with N=100 points and a 128x128 grid, numpy Parseval implies
`FFT = 128^2 - N^2 / sum(I^2)`. If every point occupies a unique cell, FFT saturates at 16284.
S2 and S3 medians are exactly that floor. S1 is only slightly above it. The S1>S2,S3 ranking is statistically large in rank-space, but the raw FFT dynamic range is tiny. Treat FFT here as a weak occupancy-collision / peakedness probe, not a rich periodogram.

## Section 2. FFT analysis

Hypothesis: S1 has more repetitive/periodic spatial structure, so FFT(S1) > FFT(S2), FFT(S3).

Dataset,FFT mean,FFT median,NN mean,NN median
S1,16297.917249836077,16291.407407407412,0.05234193367673068,0.03469236005626439
S2,16285.124077786566,16284.000000000005,0.10140535137862251,0.10621333872446491
S3,16285.506922959967,16284.000000000005,0.09583063382595079,0.10120099417325013


comparison,feature,hypothesis,U,p_value,cliffs_delta,cliffs_magnitude,mean_left,mean_right,direction_holds,significant_0.05
S1 vs S2,fft_energy,FFT(S1) > FFT(S2),774235.5,1.1110787497204906e-113,0.548471,large,16297.917249836077,16285.124077786566,True,True
S1 vs S3,fft_energy,FFT(S1) > FFT(S3),740771.5,1.1790038297440774e-83,0.481543,large,16297.917249836077,16285.506922959967,True,True


Do not over-read the large Cliff's delta: S2/S3 pile up at the 16284 floor, so S1 wins a rank test even though absolute FFT values nearly overlap.

## Section 3. NN-ratio analysis

Hypothesis: S2 is more locally regular/space-filling than S3, so NN(S2) < NN(S3).

comparison,feature,hypothesis,U,p_value,cliffs_delta,cliffs_magnitude,mean_left,mean_right,direction_holds,significant_0.05
S2 vs S3,nn_ratio,NN(S2) < NN(S3),535256.0,0.9968359172574043,0.070512,negligible,0.10140535137862251,0.09583063382595079,False,False


NN-ratio directional+significance test: DOES NOT HOLD.

Observed NN means: S1=0.0523, S2=0.1014, S3=0.0958. S1 is much more locally dense. S2 is slightly *higher* than S3, opposite the S2<S3 hypothesis, with negligible effect size.

## Section 4. Feature-space visualization

Figures:

- `fft_distribution.png`
- `nnratio_distribution.png`
- `evoreal_feature_space.png`
- `N_histogram.png`

Two-feature recovery of the dataset labels (5-fold stratified CV on [FFT energy, NN-ratio]):

logistic_regression: accuracy=0.5190, macro F1=0.5178
pred\true | S1 | S2 | S3
S1        |  661 |  171 |  168
S2        |  125 |  571 |  304
S3        |  114 |  561 |  325
(rows = true S1/S2/S3, columns = predicted S1/S2/S3)

random_forest: accuracy=0.6077, macro F1=0.6076
pred\true | S1 | S2 | S3
S1        |  702 |  140 |  158
S2        |  125 |  589 |  286
S3        |  167 |  301 |  532
(rows = true S1/S2/S3, columns = predicted S1/S2/S3)

Random-chance accuracy for three balanced classes is 1/3. High accuracy means the labels are recoverable from the two structure probes. Near-chance accuracy would mean the names are tags without those structural meanings.

## Section 5. Final verdict

**PARTIAL**

Only subset of categories match EvoReal criteria.

Interpretation for the three questions:

1. Do the names match EvoReal FFT/NN categories? Only partly. S1 is the distinctive class (higher FFT rank, much lower NN-ratio). S2 vs S3 are not recovered by NN-ratio in the hypothesized direction and are frequently confused by the classifier (RF accuracy 0.61, well above 0.33 chance but not a clean three-way partition).
2. Did EvoReal-LEHD degrade on the intended structure domain? The S1 drop happened on the split that is structurally the repetitive/local class. The S2 and S3 drops happened on two N=100 generator families that this audit cannot cleanly separate from each other.
3. Adaptation failure vs protocol mismatch? Not a scale artifact (all N=100). The more plausible reading is domain shift: phase-2 EvoReal-LEHD was fine-tuned on real TSPLIB, while these tests are frozen N=100 generator splits. S2/S3 labels should not be treated as a verified NN-ratio contrast.

Classifier best accuracy = 0.6077. FFT tests hold = True. NN tests hold = False.
