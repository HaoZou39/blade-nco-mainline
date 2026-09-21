# EvoReal S1/S2/S3 structural alignment audit (corrected v2)

Previous audit verdict `PARTIAL` is **void**. It used incorrect FFT and NN-ratio definitions.

**Corrected verdict: PARTIAL**

At least one class is highly aligned with the paper rule, but another class shows substantial mixing.

This experiment only answers: do the frozen EvoReal-S1/S2/S3 instances match EvoReal's structural classification rule?
It does not interpret EvoReal-LEHD performance.

## 1. Why the previous NN-ratio was wrong

Previous definition:

`NN_ratio = mean(nearest-neighbor distance) / mean(all-pair distance)`

EvoReal defines NN-ratio as the coefficient of variation of nearest-neighbor distances:

`NN-ratio = std(d_NN) / mean(d_NN)`

with self-exclusion (`k=2` KNN). The old ratio is a dimensionless spacing scale, not CV, so it cannot be compared with the paper threshold 0.5.

Corrected implementation uses `numpy.std(..., ddof=0)` on the N self-excluded nearest-neighbor distances after Algorithm 1 normalization.

## 2. Corrected NN-ratio on frozen S1/S2/S3

- Provided S2 share(NN-ratio < 0.5) = 58.7%; share(>= 0.5) = 41.3%
- Provided S3 share(NN-ratio < 0.5) = 11.9%; share(>= 0.5) = 88.1%
- Provided S1 share(NN-ratio < 0.5) = 28.1%; share(>= 0.5) = 71.9%

S2 mainly < 0.5: False. S3 mainly > 0.5: True.

Author-label check (not frozen generators): S2 100% < 0.5 and S3 100% >= 0.5, so the corrected NN-ratio matches the paper rule on the official validation split.

## 3. Is corrected FFT on the same scale as threshold=35?

FFT calibration status: `CALIBRATED_ON_AUTHOR_TSPLIB_VALIDATION`

Official FFT source code was not found. The statistic is a paper-definition reimplementation calibrated on the author's 48 TSPLIB validation instances.

- Author S1 mostly FFT>35: 78.9% (required for calibration)
- Author non-S1 mostly FFT<=35: 100.0%
- Author S1 FFT median = 63.609; non-S1 median = 13.681
- Values are not concentrated near 16284.

Frozen-set FFT vs 35:

- Provided S1: FFT>35 = 51.3%; FFT<=35 = 48.7%
- Provided S2: FFT>35 = 0.2%; FFT<=35 = 99.8%
- Provided S3: FFT>35 = 0.1%; FFT<=35 = 99.9%

## 4. Rule-based alignment matrix (PRIMARY)

Paper rule: S1 if FFT>35; else S2 if NN-ratio<0.5; else S3.

,predicted_S1,predicted_S2,predicted_S3
provided_S1,513,243,244
provided_S2,2,587,411
provided_S3,1,119,880


- overall rule-based accuracy = 0.6600
- macro F1 = 0.6578
- balanced accuracy = 0.6600
- per-class recall: S1=0.5130, S2=0.5870, S3=0.8800

Region occupancy under the paper rule:

- Provided S1 in S1 region (FFT>35): 51.3%
- Provided S2 in S2 region (FFT<=35 AND NN-ratio<0.5): 58.7%; assigned S1=0.2%; assigned S3=41.1%
- Provided S3 in S3 region (FFT<=35 AND NN-ratio>=0.5): 88.0%; assigned S1=0.1%; assigned S2=11.9%

ML classifiers are secondary only and do not decide PASS/FAIL.

- logistic_regression accuracy=0.7173, macro F1=0.7196
- random_forest accuracy=0.8190, macro F1=0.8195

## 5. Final verdict

**PARTIAL**

At least one class is highly aligned with the paper rule, but another class shows substantial mixing.

## Dataset identity (unchanged)

Coordinates were not regenerated or modified. Frozen SHA-256 values match the previous audit:

- EvoReal_S1_100: 1000 instances, `28c8e4c0b8f57f840e6b444e45a21c795993205faae4e27ffc9dcae1df55448f`
- EvoReal_S2_100: 1000 instances, `f393b1be940e09ff3f27c5818fb8b721dd240891d0cda22fedefc10e28c7256c`
- EvoReal_S3_100: 1000 instances, `d1f74c229e6efc609845f3570e58fa199158b5feaafc526792c2b3ee9dbab48f`

All three splits have N=100 and std(N)=0. Any later performance difference cannot be attributed to S1/S2/S3 problem-size mismatch.

## Descriptive statistics

dataset,n_instances,N_mean,FFT_mean,FFT_median,FFT_std,FFT_q25,FFT_q75,FFT_gt35_fraction,NN_ratio_mean,NN_ratio_median,NN_ratio_std,NN_ratio_q25,NN_ratio_q75,NN_gt05_fraction,NN_lt05_fraction
EvoReal_S1_100,1000,100.0,33.667262675598046,35.48717318727128,13.596131115740732,23.009327497829574,43.37399669849391,0.513,1.1748556754552728,0.843809404210729,0.9313891821072706,0.4685003929264071,1.802288301314918,0.719,0.281
EvoReal_S2_100,1000,100.0,10.082526497867311,10.085231742646886,4.23292085234944,6.393174714612668,12.47989247988562,0.002,0.44389776124787356,0.45320238952546915,0.2431253931157547,0.2595356001877853,0.5706377753637351,0.413,0.587
EvoReal_S3_100,1000,100.0,13.132455509286363,11.711171004540262,5.579291821339196,9.209346153002972,16.015972744860157,0.001,0.6616446585958436,0.6593395136252855,0.14001997363225982,0.5647596654143265,0.7476307252475167,0.881,0.119


Statistical tests are auxiliary only:

comparison,feature,hypothesis,U,p_value,cliffs_delta,cliffs_magnitude,mean_left,mean_right,role
S1 vs S2,fft_energy,FFT(S1) > FFT(S2),953001.0,6.6851028570543705e-270,0.906002,large,33.667262675598046,10.082526497867311,auxiliary_only
S1 vs S3,fft_energy,FFT(S1) > FFT(S3),911571.0,3.24771521930154e-223,0.823142,large,33.667262675598046,13.132455509286363,auxiliary_only
S2 vs S3,nn_ratio,NN(S2) < NN(S3),219205.0,3.867362278139979e-105,-0.56159,large,0.44389776124787356,0.6616446585958436,auxiliary_only


## Figures

- `corrected_fft_distribution.png`
- `corrected_nnratio_distribution.png`
- `corrected_evoreal_feature_space.png`

## Answers to the five required questions

1. Previous NN-ratio used mean NN distance over mean pairwise distance. The paper uses CV of NN distances, std(d_NN)/mean(d_NN).
2. Corrected NN-ratio: provided S2 share<0.5=58.7%; provided S3 share>=0.5=88.1%.
3. Corrected FFT was calibrated on the author's 48 TSPLIB validation instances and is on a scale comparable to threshold=35. Status=`CALIBRATED_ON_AUTHOR_TSPLIB_VALIDATION`.
4. Rule-based alignment matrix is in `rule_based_alignment_matrix.csv`. Overall accuracy=0.6600; recalls S1=0.5130, S2=0.5870, S3=0.8800.
5. Final verdict: PARTIAL

```
EVOREAL_STRUCTURE_ALIGNMENT_V2=PARTIAL
S1_RULE_RECALL=0.513000
S2_RULE_RECALL=0.587000
S3_RULE_RECALL=0.880000
OVERALL_RULE_ACCURACY=0.660000
FFT_CALIBRATION_STATUS=CALIBRATED_ON_AUTHOR_TSPLIB_VALIDATION
NN_IMPLEMENTATION_STATUS=CORRECTED
```
