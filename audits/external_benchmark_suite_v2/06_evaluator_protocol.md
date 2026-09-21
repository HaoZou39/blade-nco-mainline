# Canonical passive evaluator

- Checkpoints: M0 through M20, saved formal checkpoints only; no retraining or selection.
- Decoder: greedy, beam=1, KNN=250, no post-processing.
- KNN=250 is non-binding at N=100 and binding for GaussianMixture500; it is never retuned.
- Synthetic inputs: identity transform (no additional per-axis normalization).
- GaussianMixture100: prediction/reference pairing by stable instance ID and coordinate hash, never row position.
- OriginalTSPLIB70: translate by coordinate minima, divide both axes by one maximum axis span for model input; score the returned tour in original TSPLIB EUC_2D geometry.
- Outputs cannot affect training, GP, QD acquisition, hardness, labels, stopping, or model selection.
