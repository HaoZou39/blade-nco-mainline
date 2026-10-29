# Training Protocol and Log

- Directly reused formal P3/P4-B1 `train`; E000 exactly equals formal M4: **True**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, batch/microbatch 256/64, 256 epochs, 40 updates/epoch, 10,240 updates, seed 2026, all 10,000 instances once/epoch.
- R2 final loss 0.05182892, wall 802.383s, peak GPU bytes 2,137,318,912. Frozen R0 final loss 0.04937416.
- R2 checkpoint SHA256 `c1e569b86221af567b7414d5edc5cb71c128e0df1617063f8aa05442bce61fe4`; membership SHA256 `6f141dcc069fac3c5007af70e5d4adc5b53b3a2ad0372d6269e0eeaa00b8961d`; label bundle SHA256 `8ae216e71b28f8ac5b3b450ea51f651795f0bf42e6383969f2f2753152b53a79`.

E256 was frozen before evaluation; no benchmark selected data, hyperparameters or checkpoint.
