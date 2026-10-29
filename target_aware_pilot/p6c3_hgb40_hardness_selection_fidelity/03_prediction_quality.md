# Prediction Quality

Raw-space metrics use the frozen inverse transform; no recalibration is applied. Pooled correlations are partly source-shift driven, so E0-only/E1-only and strict-root-safe rows are primary diagnostics. HGB correctly predicts E0 as harder in mean, but underpredicts both groups and has weak within-E1 ranking.

| group | evaluation_mode | n | spearman_rho | spearman_ci_low | spearman_ci_high | pearson_r | mae | mean_signed_error_pred_minus_true | prediction_true_mean_ratio | true_mean | prediction_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original:all | full_replay | 19929 | 0.3888632 | 0.3780763 | 0.3999638 | 0.4191679 | 0.02872459 | -0.01102438 | 0.8533018 | 0.07515004 | 0.06412566 |
| Original:E0_only | full_replay | 9929 | 0.3253674 | 0.3090674 | 0.342227 | 0.3530139 | 0.03240163 | -0.0124021 | 0.8601421 | 0.08867647 | 0.07627437 |
| Original:E1_only | full_replay | 9929 | 0.109393 | 0.09027718 | 0.1280661 | 0.108139 | 0.02507562 | -0.009636193 | 0.8438214 | 0.06169981 | 0.05206362 |
| Original:strict_root_safe | strict_root_safe | 285 | 0.244311 | 0.143197 | 0.3528756 | 0.3221852 | 0.0272114 | -0.005945889 | 0.9029997 | 0.06129764 | 0.05535175 |
| S1:all | full_replay | 19937 | 0.3969785 | 0.3863486 | 0.4087999 | 0.4250459 | 0.02887049 | -0.01169923 | 0.8456357 | 0.07578978 | 0.06409055 |
| S1:E0_only | full_replay | 9937 | 0.3121532 | 0.2935395 | 0.3310703 | 0.343203 | 0.03276007 | -0.01387052 | 0.8460263 | 0.09008369 | 0.07621317 |
| S1:E1_only | full_replay | 9937 | 0.1111575 | 0.09140791 | 0.128051 | 0.1071383 | 0.02498409 | -0.009512597 | 0.8454563 | 0.0615528 | 0.0520402 |
| S1:strict_root_safe | strict_root_safe | 349 | 0.3112002 | 0.2105384 | 0.4024785 | 0.3866227 | 0.02773701 | -0.008951992 | 0.8681682 | 0.06790466 | 0.05895267 |
| S2:all | full_replay | 19927 | 0.3831998 | 0.369912 | 0.394958 | 0.4165623 | 0.02883393 | -0.01118227 | 0.8508848 | 0.07499081 | 0.06380854 |
| S2:E0_only | full_replay | 9927 | 0.3007602 | 0.2829513 | 0.3193613 | 0.3336421 | 0.0329811 | -0.01323889 | 0.8513417 | 0.08905584 | 0.07581695 |
| S2:E1_only | full_replay | 9927 | 0.1004894 | 0.08197883 | 0.1203426 | 0.1024492 | 0.02470049 | -0.009111016 | 0.85065 | 0.06100445 | 0.05189343 |
| S2:strict_root_safe | strict_root_safe | 345 | 0.2691722 | 0.1667393 | 0.3843821 | 0.3152235 | 0.02659245 | -0.00358353 | 0.9391991 | 0.05893881 | 0.05535528 |

Calibration by true decile is in `manifests/calibration_by_true_decile.csv`; tail metrics and bootstrap intervals are separate.
