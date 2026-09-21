# Uniform-rooted Cycle-0 QD bootstrap audit

This structural-only audit starts from the exact canonical 10,000 Uniform roots and applies every frozen FOGA-11 operator exactly once per root. It performs no hardness evaluation, solver inference, training, Concorde call, or external benchmark evaluation.

## Validation

- Nominal children: 110000
- Valid children: 109785
- Invalid children: 215
- Valid unique offspring: 109785
- Duplicate children: 0
- Raw-invalid children exposed before canonical repair: 38093
- Children repaired by the canonical operator adapter: 38093

## Fixed-map coverage

| pool | occupied | fraction | empty |
|---|---:|---:|---:|
| roots only | 51 | 0.0510 | 949 |
| offspring only | 435 | 0.4350 | 565 |
| roots + offspring | 435 | 0.4350 | 565 |

New niches discovered beyond roots: **384**.

Migration fractions among valid unique offspring: same parent niche 0.1389; another root-occupied niche 0.6686; niche absent from roots 0.1925.

## Operator evidence

The full descriptive operator table is `08_operator_expansion_summary.csv`. All 11 operators remain frozen; these diagnostics do not rank, remove, or retune them.

## Provisional GP0 probes

- Offspring-only candidates: 435
- Roots-plus-offspring candidates: 435

Each manifest contains one deterministic centroid-nearest representative per reached niche. These candidates are difficulty-blind, have no Oracle labels, and are not a selected GP0 training source.

## Decision boundary

No coverage pass threshold is applied. The audit stops for human review of acquisition eligibility, probe sufficiency, and whether sparse Cycle-0 coverage should be accepted or expanded before GP0.
