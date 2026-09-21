# S1.1 GP4 Oracle ENR ceiling audit

Oracle Top-q sets are an **ORACLE / PERFECT-RANKING CEILING** (analysis-only).
They are not saved as training sets, parents, or archive members.

Source run: `blade_v2_full_seed2026`, offspring-only C00–C14. P0 raw N=1267.
GP ENR was required to reproduce S1 before any ceiling calculation.

## Master ENR10
One row per cycle is in `05_gp_oracle_random_comparison.csv`. Per-q capture and random distributions are in `06_capture_rate_by_cycle.csv` and `03_random_enr_distribution.csv`.

Cycle | Random ENR10 | GP ENR10 | Oracle ENR10 | Capture10
--- | --- | --- | --- | ---
C00 | 0.9990 | 0.9831 | 2.3416 | -0.0118
C01 | 0.9997 | 0.9157 | 2.3229 | -0.0635
C02 | 1.0002 | 1.1132 | 2.3780 | 0.0820
C03 | 0.9997 | 1.0989 | 2.4213 | 0.0698
C04 | 0.9998 | 0.9951 | 2.3702 | -0.0035
C05 | 0.9996 | 1.0585 | 2.4204 | 0.0414
C06 | 1.0001 | 1.0598 | 2.4415 | 0.0415
C07 | 0.9999 | 1.0715 | 2.4377 | 0.0498
C08 | 1.0007 | 1.0704 | 2.5403 | 0.0453
C09 | 1.0004 | 1.0694 | 2.5488 | 0.0446
C10 | 1.0004 | 1.0849 | 2.5405 | 0.0549
C11 | 0.9997 | 1.0919 | 2.5617 | 0.0590
C12 | 1.0003 | 1.0945 | 2.5832 | 0.0595
C13 | 0.9990 | 1.0067 | 2.5636 | 0.0050
C14 | 0.9990 | 1.1457 | 2.6396 | 0.0894

## Phase medians (ENR10 primary)
- EARLY C00–C04: Random median ENR10=0.9997; GP=0.9951; Oracle=2.3702; GP gain=-0.0047; available gain=1.3704; capture=-0.35%; Top10 recall=9.45%; recall capture=-0.61%. Representative cycle: C04.
- MID C05–C09: Random median ENR10=1.0001; GP=1.0694; Oracle=2.4415; GP gain=0.0690; available gain=1.4415; capture=4.46%; Top10 recall=13.49%; recall capture=3.88%. Representative cycle: C09.
- LATE C10–C14: Random median ENR10=0.9997; GP=1.0919; Oracle=2.5636; GP gain=0.0922; available gain=1.5646; capture=5.90%; Top10 recall=14.96%; recall capture=5.51%. Representative cycle: C11.

## Plain-language translations
- Early representative: C04: Random ENR10=0.9998, GP ENR10=0.9951, Oracle ENR10=2.3702. Random sampling gives no systematic enrichment (empirical mean ≈ 0.9998). GP changes selected mean hardness by about -0.5% relative to the random mean. Perfect ranking could change it by about 137.1%. Therefore GP captures -0.35% of the available hard-tail enrichment above random.
- Mid representative: C09: Random ENR10=1.0004, GP ENR10=1.0694, Oracle ENR10=2.5488. Random sampling gives no systematic enrichment (empirical mean ≈ 1.0004). GP changes selected mean hardness by about 6.9% relative to the random mean. Perfect ranking could change it by about 154.8%. Therefore GP captures 4.46% of the available hard-tail enrichment above random.
- Late representative: C11: Random ENR10=0.9997, GP ENR10=1.0919, Oracle ENR10=2.5617. Random sampling gives no systematic enrichment (empirical mean ≈ 0.9997). GP changes selected mean hardness by about 9.2% relative to the random mean. Perfect ranking could change it by about 156.2%. Therefore GP captures 5.90% of the available hard-tail enrichment above random.

## Absolute gain vs fraction of ceiling
Median GP absolute ENR10 gain above random = 0.0697.
Median available Oracle gain = 1.4415.
Median capture rate = 4.53%.
A small absolute GP gain can coexist with a small fraction of a large ceiling, or with most of a small ceiling.

## Oracle hard-tail separation
Median Oracle ENR10 = 2.4415. Panel interpretation: large available hard-tail separation.
See `04_oracle_enr_by_cycle.csv` and `09_hardness_distribution.csv`.

## Descriptive trends vs cycle index
- Oracle_ENR10: Spearman vs cycle = 0.9679
- GP_ENR10: Spearman vs cycle = 0.4214
- CaptureRate10: Spearman vs cycle = 0.4214
- AVAILABLE_GAIN10: Spearman vs cycle = 0.9679
- GP_GAIN10: Spearman vs cycle = 0.4214
Descriptive only: this separates changing panel separability from changing GP capture.

## Pattern: PATTERN C
Headroom: large unused surrogate headroom.

This audit does not prove that HGB, RF, extra features, or an ensemble will close the gap.
It only measures how much hardness-ranking signal exists on sealed P0 and how much historical GP4 captures.
Oracle results are not used for selection or checkpointing.

## Required questions

### Q1. What is the actual Oracle ENR10 ceiling on P0 at each cycle?
See the master table above. Median Oracle ENR10 = 2.4415.

### Q2. How far above random is current GP?
GP ENR10 exceeds empirical random mean in 12/15 cycles. Median absolute gain = 0.0697.

### Q3. How far below Oracle is current GP?
Median Oracle−GP ENR10 gap = 1.4072.

### Q4. What percentage of the available ENR10 gain does GP capture?
Median CaptureRate10 = 4.53%. Per-cycle values are in `06_capture_rate_by_cycle.csv`. Negative captures are not clipped.

### Q5. Does the GP captured fraction improve, worsen, or remain unstable from early to late?
Early median capture=-0.35%; mid=4.46%; late=5.90%. Spearman(CaptureRate10, cycle) = 0.4214.

### Q6. Is GP's modest ENR10 caused mainly by little available tail separation, or failure to capture a large available separation?
Large available hard-tail separation. Median available gain = 1.4415 while median GP gain = 0.0697.

### Q7. Top10 Recall: random vs GP vs perfect ranking?
Random expected ≈ 10%. Early/mid/late median GP recall = 9.45% / 13.49% / 14.96%. Oracle recall = 100%. Median recall capture = -0.61% / 3.88% / 5.51%.

### Q8. Does current evidence suggest substantial room for a stronger surrogate?
large unused surrogate headroom. Room for a later controlled comparison exists if unused ceiling remains large; this file does not implement that comparison.

### Q9. Does this audit justify immediately replacing GP?
No. Ceiling analysis is not a model bake-off. Replacing GP requires a later controlled surrogate comparison on acquisition quality, not Oracle ENR alone.

### Q10. What should the next surrogate experiment measure?
A frozen GP4 vs candidate stronger-surrogate comparison on the same leakage-clean P0: ENR10, capture rate vs this Oracle ceiling, Top10 recall, and operational Top10k concentration. Do not use Oracle labels for training or selection.

STOP FOR HUMAN REVIEW. Do not launch the comparison automatically.
