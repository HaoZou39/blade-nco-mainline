# Final decision

## A — PILOT_VALID_EXPAND_TO_300

The pilot validates the fixed-record paired design, data joins, exact-tour path, M5 teacher-forced evaluation, cost assumptions, and parent-level analysis. Expansion is recommended for the remaining frozen 270 parents; this recommendation does not depend on pilot statistical significance.

Reasons:

- All 30 parents and 330 original attempts remained exactly paired; 326 valid children received complete M5 labels and four failures remained explicit.
- Checkpoint, coordinates, tours, labels, HGB predictions, QD features, and T05 membership passed exact identity checks.
- New compute completed with 312/312 verified tours and 336/336 finite M5 labels. Successful compute took 33.71 s Concorde wall plus 9.45 s LBQ wall.
- The predeclared analysis ran successfully with parent clustering, scheduled and conditional denominators, design weights, hard threshold, structural displacement, retention, and multiplicity handling.
- Nonsignificant omnibus/pairwise pilot results do not invalidate the design. Marginal implosion/compression signals, large uncertainty, and weighting sensitivity justify completing rather than stopping the frozen panel.

The initial WSL sandbox denial was an infrastructure incident, not a data/solver failure; the permission-corrected identical run completed with zero solver retry. Current mutation source provenance remains the DT1-A0 caveat, but no regeneration is required.

Next step: after human approval, evaluate only the remaining 270 parents already frozen in `audits/dt1a_operator_hardness_preflight_v1/03_parent_sampling_manifest.csv`, reuse all current exact tours/M5 labels including this pilot, and apply the unchanged preregistered analysis to all 300. Do not select operators from pilot results or alter `tau`, `delta_h>0.01`, outcomes, or multiplicity families.

STOP: this audit did not expand to 300, run M0/M6 paired evaluation, train an SVM, derive a Lyapunov theorem, alter formal algorithms, or merge a PR.

Identity: M5 SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`; DT1-A0 parent manifest SHA-256 `3070013e8d342070b9295825b22d10c91ac9503957092b06423a909696cdb99f`; pilot volume 30 parents/330 attempts/326 valid unique children.
