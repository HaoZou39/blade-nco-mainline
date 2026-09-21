# Root-cause evidence

## GaussianMixture100

The checkpoint and all frozen dataset/Oracle files are byte-identical. The NPZ and manifest order is `cdist10,30,50,70,100`; the current evaluator independently sorts Oracle rows lexicographically, producing `cdist100,10,30,50,70`, then zips that vector positionally with predictions. Rebinding the already cached M0 predicted costs to the Oracle by physical manifest ID changes the mean from 10.114118980811% to 6.088335595618%, matching the historical 6.088324622137% within 0.000011 percentage points. Classification: **R5_ORACLE_REFERENCE_MISMATCH** (binding mismatch, not source-cost mismatch).

## OriginalTSPLIB70

All 70 source files, dimensions, instance IDs and known optima match. Historical input is translated and divided by one maximum axis span, preserving aspect ratio. Current input divides x and y by separate spans. Raw-coordinate EUC_2D scoring, KNN=250, greedy start-node-0 decoding and the gap formula otherwise match/effectively match. Cached predicted costs differ on 61/70 instances; historical aspect-preserving evaluation is 5.375617545006% and current per-axis evaluation is 8.650173126554%. Classification: **R3_INPUT_PREPROCESSING_MISMATCH**.

## Controls

Uniform100 and TSPLIB100 have exact stored coordinates, identity preprocessing, valid positional ID order and reproduce their historical aggregate. K=250 is non-binding for N=100 because the largest closed state has 101 nodes. No KNN anomaly is supported.

## Replay boundary

No GPU replay was performed because the formal runner owns the only GPU. Cached outputs are sufficient for the two aggregate causal isolations. Current cached files do not contain tour hashes, so current-vs-historical tour sequence and first-divergence traces are explicitly deferred rather than fabricated.
