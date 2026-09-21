# BLADE-NCO M0 External Benchmark Protocol Reconciliation Audit

## Outcome

`GLOBAL_CAUSE=MULTIPLE_BENCHMARK_SPECIFIC_CAUSES`.

- GaussianMixture100: **R5 effective Oracle-reference binding mismatch**. The underlying Oracle file is correct and byte-identical, but current code sorts its IDs independently and positionally zips them with NPZ-order predictions. Cached rebinding restores M0 from 10.114119% to 6.088336%, matching historical 6.088325%.
- OriginalTSPLIB70: **R3 input preprocessing mismatch**. Historical aspect-preserving normalization gives 5.375618%; current per-axis min-max distorts geometry and gives 8.650173% on the same 70 files and same optima.
- Uniform100 and TSPLIB100 are exact controls with no reproducible discrepancy.
- M0 is byte-identical (`6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15`). All four stored benchmark datasets are exact matches.
- The defects are passive external-evaluation only. Training, hardness, QD/GP acquisition and Concorde expert data are unaffected.

## Primary reconciliation table

| Benchmark          |   Hist M0 |   Current M0 | Checkpoint Same   | Dataset Same   | Preprocess Same   | Decoder Same   | Oracle Same          | Cost Metric Same   | Gap Formula Same   | Root Cause                      |
|:-------------------|----------:|-------------:|:------------------|:---------------|:------------------|:---------------|:---------------------|:-------------------|:-------------------|:--------------------------------|
| Uniform100         |  0.345262 |     0.345262 | YES               | YES            | YES               | YES_EFFECTIVE  | YES                  | YES_EFFECTIVE      | YES                | R10_NO_REPRODUCIBLE_DISCREPANCY |
| GaussianMixture100 |  6.08832  |    10.1141   | YES               | YES            | YES               | YES_EFFECTIVE  | NO_EFFECTIVE_BINDING | YES_EFFECTIVE      | YES                | R5_ORACLE_REFERENCE_MISMATCH    |
| TSPLIB100          |  1.22964  |     1.22964  | YES               | YES            | YES               | YES_EFFECTIVE  | YES                  | YES_EFFECTIVE      | YES                | R10_NO_REPRODUCIBLE_DISCREPANCY |
| OriginalTSPLIB70   |  5.37562  |     8.65017  | YES               | YES            | NO                | YES_EFFECTIVE  | YES                  | YES_EFFECTIVE      | YES                | R3_INPUT_PREPROCESSING_MISMATCH |

## Dataset identity

| benchmark          |   historical_instance_count |   current_instance_count | classification      | instance_id_order_equal   | coordinate_order_equal   |
|:-------------------|----------------------------:|-------------------------:|:--------------------|:--------------------------|:-------------------------|
| Uniform100         |                       10000 |                    10000 | EXACT_DATASET_MATCH | True                      | True                     |
| GaussianMixture100 |                       10000 |                    10000 | EXACT_DATASET_MATCH | True                      | True                     |
| TSPLIB100          |                       10000 |                    10000 | EXACT_DATASET_MATCH | True                      | True                     |
| OriginalTSPLIB70   |                          70 |                       70 | EXACT_DATASET_MATCH | True                      | True                     |

## KNN conclusion

For N=100, K=250 is non-binding because the largest closed decoder state contains 101 nodes. Historical K=-1 and current K=250 therefore expose the same full candidate set. Original70 historical 5.375618 and current use K=250. A separate historical K=100 variant exists (7.157152%) and is not the source of the 5.375618 reference.

## Cached reconciliation

Gaussian current cached prediction costs match the historical physical-instance costs on 9,999/10,000 rows. The single difference changes the aggregate by only about 0.000011 percentage points and is non-causal. Original70 Oracle costs match 70/70; predicted costs differ 61/70 after the preprocessing change.

## Replay status

No GPU replay was performed because the active formal runner owns the GPU. Static and cached-output evidence is sufficient for aggregate root-cause isolation. Current formal result files do not store tours, so current tour hashes, first-10-node traces and first-divergence steps are explicitly deferred. No placeholder tour-agreement figure was created.

## Formal-run decision

`CONTINUE_FORMAL_RUN_BUT_REEVALUATE_EXTERNAL_RESULTS`.

Do not retrain. After human review, passively repair/recompute Gaussian metrics by correct ID binding and reevaluate Original70 with aspect-preserving model input for all completed checkpoints and future M checkpoints. Existing formal outputs must be archived, not overwritten.

## Integrity

- Formal audited immutable artifacts unchanged during audit: `TRUE`
- Formal run modified by audit: `NO`
- Model retrained: `NO`
- Concorde called: `NO`
- Oracle labels added: `NO`
- Protocol automatically changed: `NO`
- Audit checks: `54/54`
