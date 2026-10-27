# DT1-A1 protocol and integrity

## Frozen question and boundary

This pilot asks whether the eleven FOGA-11 operators produce different one-step true-BQ-hardness changes under the same entering C05 checkpoint and matched parents. It uses only the 30 rows predeclared as `pilot30=True` in DT1-A0, retains all eleven original attempts per parent, and does not regenerate mutation.

Allowed computation was limited to missing exact tours and missing fixed-M5 teacher-forced forward losses. No BQ/HGB training, mutation, evolutionary cycling, M0/M6 comparison, SVM, or theorem fitting was run.

## Pre-compute gate

All DT1-A0 deliverables and every source checksum in `integrity_summary.json` were reread and verified before computation. The gate confirmed:

- frozen sample: 300 unique parents and 3,300 attempts;
- full sample outcomes: 3,281 valid unique children, 19 repair failures, zero duplicates;
- pilot: 30 parents, exactly 11 attempts each, 330 scheduled attempts, 326 valid unique children, four `excessive_repair` failures;
- full M5 gap: 214 parent labels plus 3,044 child labels = 3,258;
- pilot M5 gap: 24 parent labels plus 312 child labels = 336;
- 237 preexisting selected-child labels are exact C05/M5 joins; 14 occur in the pilot;
- checkpoint: `M5_E256` entering C05, SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`;
- every pilot parent and valid child coordinate recomputes its recorded SHA-256 identity.

The machine gate is `work/preflight_gate.json`. DT1-A0 sample and attempt manifest hashes are respectively `3070013e8d342070b9295825b22d10c91ac9503957092b06423a909696cdb99f` and `6543330baf04b04afbf5e9e87a86bef60e388d4a35c6d0298f89ef70768bac39`.

## Frozen analysis definitions

`h5(x)=L_BQ(M5,x)` and `f5(x)=log(1+h5(x))`. The CSVs keep raw LBQ and log1p LBQ in distinct columns. For valid child `(i,o)`, `delta_h=h5(child)-h5(parent)` and `delta_f=f5(child)-f5(parent)`.

- positive improvement: `delta_h>0`;
- material improvement: `delta_h>0.01`;
- hard-region hit: `h5(child)>=0.12629708954782198`, the independently frozen C05 calibration P90;
- invalid attempts: no numeric delta, but zero success in scheduled-attempt yields;
- uncertainty: 10,000 parent bootstrap replicates, seed `20261028`; 10,000 matched permutations, seed `20261029`; Holm correction within the 55-pair outcome families.

The 30-parent membership rule and structural fields are preserved in `02_30parent_sample_manifest.csv`; its sample seed is `20261027`. Existing LBQ labels and exact tours were reused by hash. New Concorde used the canonical binary SHA-256 `38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0`. New LBQ used the canonical ground-truth-prefix teacher-forcing implementation, mean expert-action NLL over 99 decisions, target index 1, no rollout, and no parameter update.

