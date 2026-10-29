# Training Protocol and Log

- Directly reused the formal A0/P3 `train` function. E000 state tensors exactly equal formal M4: **True**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, logical batch 256, microbatch 64, 256 epochs, 40 updates/epoch, 10,240 total updates, NumPy seed 2026.
- Every epoch exposes all 10,000 frozen E0 instances exactly once; no anchor mixture. The only P3/E1 versus P4-A/E0 training difference is dataset membership.
- Final loss 0.04359868; training-loop wall 802.570s; peak allocated GPU bytes 2,137,318,912.
- P3 E1 comparison: final loss 0.04256974; wall 799.093s under the identical protocol.
- E0 M5 checkpoint SHA256 `d7d2ffc9ebafa5adb2bca9ccd90f6e8bd0d0487bdf1b4f68cf98dbf425cd4c58`; training membership SHA256 `139ab50f468eb545f378f52372f259d150c84d4e34c94b040e090cbd146d02f7`; labeled bundle SHA256 `3077212b6e98b5f4a860fcfb76f75781f60a638099bc700cd443d87f05bed6d2`.

E256 is frozen before evaluation. No TSPLIB22 result is used for checkpoint or protocol selection.
