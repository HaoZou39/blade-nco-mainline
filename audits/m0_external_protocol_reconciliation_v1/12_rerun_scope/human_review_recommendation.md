# Human review recommendation

Decision: **CONTINUE_FORMAL_RUN_BUT_REEVALUATE_EXTERNAL_RESULTS**.

Do not retrain or discard checkpoints. After human review, correct only the passive evaluator, preserve the existing files as historical-invalid protocol artifacts, and reevaluate GaussianMixture100 and OriginalTSPLIB70 for every completed checkpoint. Gaussian metrics can be repaired from cached predicted costs; Original70 requires passive inference after GPU isolation is available.
