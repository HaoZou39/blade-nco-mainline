# Stage 1.5 corrections

This is a traceable correction pass over Stage 1, not Stage 2 theorem development. The original conditional survival proof, finite counterexamples, conservative novelty position, and stopping decision are preserved.

## Correction log

| # | Correction | Affected source/section | Mathematical or methodological reason | Remaining uncertainty |
|---|---|---|---|---|
| 1 | Replaced the finite-budget finite-state assumption by a measurable solver-visible TSP instance space `(X, 𝓧)`. | `definitions.tex` — Structural states and mutation; `assumption_register.md` A1–A3 | A finite realized run does not make a continuous coordinate space or kernel support finite. Fixed-N coordinates admit a standard Borel model. | A complete implementation-wide measurability proof is not supplied. |
| 2 | Defined QD as a finite measurable partition `N_j = g⁻¹({j})`, not a finite instance space. | `definitions.tex` — Structural states and mutation; A2 | The frozen four-descriptor/centroid map has 1,000 labels even though each cell can contain uncountably many instances. | Exact boundary/tie measurability is assumed for the declared deterministic tie rule. |
| 3 | Distinguished strong lumpability from distribution-dependent aggregation. | `definitions.tex` — Niche aggregation; `counterexamples.tex` — Niche aggregation; A17–A18 | Strong block-transition equality is sufficient for every initial law. A particular-law Markov aggregate can exist without strong lumpability only if its conditional within-block mixture remains history-independent. | Neither condition is verified for a niche-only S6-AE state. |
| 4 | Split BQ cycle `k` from within-cycle parent/operator attempt `(i,o)` and proof opportunity `j`. | `definitions.tex`; `proof_attempts.tex`; `theorem_candidates.tex` | Solver adaptation occurs once per generation, while mutation is a deterministic 10,000×11 batch. Reusing one “proposal cycle” index conflated target motion, cost, and filtration. | A future theorem must choose generation-level, acquisition-level, or serialized-attempt opportunities. |
| 5 | Removed random parent/operator selection as an S6-AE assumption. | `definitions.tex`; Proposition 1 applicability note; proof dependency audit; A5, A6, A11 | The code schedules every stored parent with every registered operator. Content-derived seeds randomize operator parameters/repair; they do not select parents/operators. | The scientifically appropriate ensemble over roots/seeds/runtimes is OPEN. |
| 6 | Declared the probability space and the meaning of conditioning on a frozen run. | `definitions.tex` — Probability space and frozen executions; A6 | A probability statement is undefined until the varying random inputs are named. With all seeds and artifacts fixed, the execution is one realization and many events become deterministic. | Numerical nondeterminism and repeated-run design are not characterized. |
| 7 | Mapped the actual S6-AE generation order and `P=T` rule. | `definitions.tex`; `stage1_5_method_alignment.md`; A4–A6, A16 | Current S6-AE uses fixed parents, 11 mutations per parent, unique parent–child union, HGB40 scoring, one HGB-max elite per occupied niche plus global HGB fill, exact labeling, then BQ adaptation. | The mapping is implementation-specific and does not prove a hazard advantage. |
| 8 | Separated true BQ hardness, HGB40 prediction, generation discovery, and acquisition discovery. | `definitions.tex` — Hardness and moving target; proof dependency audit; A7–A10, A15 | HGB scores the candidate union, but exact current hardness is obtained only after `T_k` is frozen. A generated true-hard candidate can be missed by acquisition. | The hard threshold and generation-to-acquisition retention bound are OPEN. |
| 9 | Corrected the surrogate bridge counterexample. | `counterexamples.tex`; `proof_status.md`; A14 | If two candidates have equal current true hardness but different mutation futures, even a perfect current-hardness predictor ties them. A deterministic tie-break can discard the only useful bridge. | This is an impossibility example, not evidence that S6-AE made such a choice. |
| 10 | Split result status into mathematical validity, S6-AE applicability, and novelty. | `main.tex`; `proof_status.md`; `assumption_register.md`; candidate/proof status tags | A standard proof can be valid while its premises are unverified in code and its scientific novelty is unestablished. | Novelty remains OPEN pending a dedicated literature review and a completed BLADE-specific theorem. |

## Implementation evidence inspected

- `scripts/run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py`
- `scripts/run_s6ac_hgb40_eliteqd1000_pt_10k_hard_only.py`
- `scripts/run_s5b_qd_aware_training.py`
- `scripts/run_s4b_hgb40_union_qd.py`
- `scripts/run_blade_v2.py`
- `src/final_run/science.py`
- `src/qd/roots.py`
- `audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/00_run_config.json`
- S6-AE `STRICT_VERIFICATION.json`, per-cycle mutation manifests, and selection-info artifacts.

The completed S6-AE artifact reports 40/40 PASS. Every C00–C09 mutation manifest contains 110,000 attempted parent–operator pairs, 10,000 distinct parents, 11 operators, and exactly 11 attempts per parent. Inspected C00, C05, and C09 selection artifacts have equal `P` and `T` membership hashes; the strict audit checks this protocol across the run.

## Preserved conclusions

- The history-based survival lemma remains valid and standard.
- Proposition 1 remains a conditional hazard certificate, not a BLADE advantage theorem.
- Universal “memory never hurts,” “diversity always helps,” and “structural mutation always beats HAC” claims remain disproved.
- BLADE-vs-fixed-generator/HAC dominance, implementation-level hazard bounds, and novelty remain OPEN.
- No experiment, Oracle/BQ evaluation, solver training, evolutionary run, or algorithm modification was performed for Stage 1.5.
