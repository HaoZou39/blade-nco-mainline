# Candidate generation audit

The pools are materially different. P5 uses 20,000 mutation attempts per generation, target-aware G1 survivor selection, a target-aware 30k offspring archive, and final cell-quota acquisition. P6-B generation uses 110,000 attempts per generation, pure Top5-OT E1 versus random E0 populations, no archive, and downstream selection from their ~20k union.

Coordinate pool overlap is only **669.0** on average (mean Jaccard **0.0136**); P5-vs-B Selected10k overlap is **187.3** (Jaccard **0.0095**). Roots are largely shared because both descend from P03, but coordinate descendants are not.

Existing P6-E0.5 FFT–NN diagnostics give P5 archive reference coverage 24/48, 24/48, 22/48 and P6-union coverage 25/48, 24/48, 25/48 (Original/S1/S2). This exploratory representation does not show superior P5 pool reachability. P5 historical archives lack aligned full-F40/QD4 and true-M4-CE fields; these were not reconstructed with prohibited new inference. Therefore candidate-generation **difference** is supported, but candidate-generation **superiority** is not established.

Exact coordinate duplicates are zero in every pool. Because the frozen protocols define no coordinate-space near-duplicate threshold, this audit does not invent one after observing outcomes; `local_2d_nn` reports a scale-normalized nearest-neighbor diagnostic in the two predeclared P5 features instead.
