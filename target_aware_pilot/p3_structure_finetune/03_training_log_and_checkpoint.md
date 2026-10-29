# Training Log and Checkpoint

- Directly reused formal A0 `train`; initialization M4 SHA256 `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`. P3 E000 state tensors exactly equal M4: **True**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, batch 256, microbatch 64, 256 epochs, 40 updates/epoch, 10,240 total updates, NumPy seed 2026.
- Each epoch exposes every one of the 10,000 Structure10k instances exactly once; no anchor mixture.
- Final loss: 0.04256974; training-loop wall: 799.093s; invocation wall: 800.279s; peak allocated GPU bytes: 2,137,318,912.
- Original C04 comparison: final loss 0.06333430, training-loop wall 828.158s, same 10,240-update protocol.
- Structure M5 checkpoint: `target_aware_pilot/p3_structure_finetune/checkpoints/Structure_M5_E256.tsp.best`; SHA256 `ace34d712ca8d20e730edeed0666362e6cc03008f9b5c0411d53c444856e3233`.
- Training data membership SHA256 `10382659919fbc7f0d4baf8ab681024b7424ff92ae62d3e0fd76623ab07c8a99`; labeled bundle SHA256 `259dab4e317a66e6c67506617dcbb52da52f6e3bb17f417eaac3f3d4474712f6`.

Checkpoint E256 was fixed before any P3 evaluation. No TSPLIB22 result was observed or used during training.
