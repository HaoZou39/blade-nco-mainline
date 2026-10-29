# P6-D1 Frozen Protocol

P6-D1 consumes only the byte-frozen P6-D0 A/B/C memberships. Coordinates and expert tours are restored from the already labelled P6-C1 union; no selection or Concorde call is made. All nine runs start from formal M4 and delegate to the same P6-B/P6-C2 trainer and greedy evaluator.

- M4 SHA256: `c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190`
- Protocol SHA256: `86d6ebd41165c87c07de6870562583e5ff87f84d7bfc11c47397fb95d072d2dc`
- Training: fresh Adam, LR 3.75e-5, no scheduler, logical batch 256, microbatch 64, seed 2026, 256 epochs, 10,240 updates, fixed E256.
- Monitor: one initial check followed by user-systemd oneshot checks every 60 minutes.
