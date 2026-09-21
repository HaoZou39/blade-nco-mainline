# A0 surrogate bottleneck / panel-Oracle headroom

Executed 2026-09-21T06:35:00.017494+00:00 .. 2026-09-21T06:35:14.484004+00:00

Frozen analysis range: C00-C04, solvers M0-M5, identical to A1-A4 `00_INPUT_SNAPSHOT.json`.
C05+ excluded. Active E4.6 was not modified, paused, restarted, or used for new inference.

All Oracle figures are **PANEL-ORACLE** (independent exact-labelled probe), not full-candidate-pool Oracle.

## Primary verdicts
- Surrogate: **SURROGATE_HEADROOM_INCONCLUSIVE**
- QD-specific: **QD_SURROGATE_LIMITATION_NOT_DIRECTLY_ESTABLISHED**

Only C00 has a valid independent exact-current-H probe. On that PANEL-ORACLE, GP Top10 enrichment is near or below Random. Multi-cycle consistency cannot be shown. C00 is descriptively near-random / very-low headroom use, but the primary label is INCONCLUSIVE because C01-C04 union-solver independent H is CURRENT_H_NOT_AVAILABLE.

median ETA10=-0.0107; mean=-0.0107; min=-0.0107; max=-0.0107
H_GP10>H_RANDOM: 0/1; bootstrap CI strictly above 0: 0/1
median ETA_operational=-0.0096

## Independent probe
Only union C00/M0 has exact teacher-forced H_s on sealed P0. Union C01-C04 solvers differ from
offspring-only S1 solvers; S1 P0 hardness is not reused for those cycles. T_s and B_s are not
independent probes. Missing values are CURRENT_H_NOT_AVAILABLE.

## QD conversion
TIER1 n=0; TIER2 n=1900. TIER1 and TIER2 are never pooled. TIER2 is selection-conditioned diagnostic only.
On TIER2 (already GP-selected T_r descendants), some cycle-horizons are estimable but cannot support an unbiased QD-masking claim. Example: C01 horizon-1 ENR_QD_GP10=1.46 vs PANEL-Oracle 1.98 (ETA_QD10=0.47); C03 horizon-1 GP enrichment is below the TIER2 set mean. These rows are labeled SELECTION_CONDITIONED_DIAGNOSTIC_ONLY.

## Four-stage pipeline
1. QD lineage preservation = SUPPORTED (A1)
2. true hard opportunity = NOT_ESTIMABLE
3. GP recognition = NOT_ESTIMABLE
4. downstream training value = PARTIAL (A3=YES)

## Optional existing-model diagnostic
SKIPPED: no matched independent-panel predictions for HGB4/HGB20/RF exist on the union C00-C04 frozen snapshot. This section is not part of the primary verdict.

## Interpretation
On the only eligible PANEL-ORACLE (C00/M0 sealed P0), GP recovers little of the available independent hard-tail headroom (near-random enrichment). C01-C04 cannot be scored independently under union solvers without new exact H. QD delayed lineage value from A1 remains, but unbiased QD-descendant exact-H is not estimable, so GP is not proven to specifically hide QD opportunities.

## Random baseline implication
A future Random ≈ GP-only result would NOT by itself imply that solver-relative hardness is useless. It may mean the current surrogate is too weak to realize available PANEL-ORACLE hardness-acquisition headroom. A0 did not launch Random/GP-only.

## Human summary

A0 SURROGATE BOTTLENECK / ORACLE HEADROOM AUDIT COMPLETE

SNAPSHOT:
    cycles = C00-C04
    solvers = M0-M5

INDEPENDENT PROBE:
    eligible cycles = C00
    probe source(s) = C00: sealed P0 minus union B0 with S1 exact H_M0; C01-C04: CURRENT_H_NOT_AVAILABLE
    n per cycle = 1259, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE

TOP10 HARDNESS ACQUISITION:
    Random enrichment = 1.000 by definition

    GP ENR10 by cycle =
        [ 0.9856, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE ]

    Panel-Oracle ENR10 by cycle =
        [ 2.3462, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE ]

    ETA10 by cycle =
        [ -0.0107, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE ]

    median ETA10 = -0.0107
    mean ETA10 = -0.0107
    min ETA10 = -0.0107
    max ETA10 = -0.0107

    median Oracle Top10 capture = 0.0800

    H_GP10 > H_RANDOM cycles = 0 / 1
    bootstrap CI(H_GP10-H_RANDOM) strictly > 0 = 0 / 1

OPERATIONAL BUDGET:
    q_operational by cycle =
        [ 0.083486, 0.083539, 0.083677, 0.083728, 0.083644 ]

    ETA_operational =
        [ -0.0096, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE ]

    median ETA_operational = -0.0096

SURROGATE VERDICT:
    SURROGATE_HEADROOM_INCONCLUSIVE

    Only C00 has a valid independent exact-current-H probe. On that PANEL-ORACLE, GP Top10 enrichment is near or below Random. Multi-cycle consistency cannot be shown. C00 is descriptively near-random / very-low headroom use, but the primary label is INCONCLUSIVE because C01-C04 union-solver independent H is CURRENT_H_NOT_AVAILABLE.

QD-DESCENDANT CONVERSION:
    Tier1 n = 0
    Tier2 n = 1900

    true-hard QD opportunity observed =
        NOT_ESTIMABLE

    GP capture of QD Oracle-hard tail =
        NOT_ESTIMABLE

    ETA_QD10 =
        NOT_ESTIMABLE

QD BOTTLENECK VERDICT:
    QD_SURROGATE_LIMITATION_NOT_DIRECTLY_ESTABLISHED

A1 CONNECTION:
    QD delayed lineage value =
        QD_DELAYED_LINEAGE_VALUE_OBSERVED

A3 CONNECTION:
    training value not explained by hardness alone =
        YES

FOUR-STAGE PIPELINE:
    QD lineage preservation = SUPPORTED
    true hard opportunity = NOT_ESTIMABLE
    GP recognition = NOT_ESTIMABLE
    downstream training value = PARTIAL

INTERPRETATION:
    On the only eligible PANEL-ORACLE (C00/M0 sealed P0), GP recovers little of the available independent hard-tail headroom (near-random enrichment). C01-C04 cannot be scored independently under union solvers without new exact H. QD delayed lineage value from A1 remains, but unbiased QD-descendant exact-H is not estimable, so GP is not proven to specifically hide QD opportunities.

IMPLICATION FOR RANDOM BASELINE:
    A future Random ≈ GP-only result would NOT by itself imply that solver-relative hardness is useless. It may mean the current surrogate is too weak to realize available PANEL-ORACLE hardness-acquisition headroom. A0 did not launch Random/GP-only.

FINAL STATUS:

    A0_SURROGATE_BOTTLENECK_AUDIT_COMPLETE
