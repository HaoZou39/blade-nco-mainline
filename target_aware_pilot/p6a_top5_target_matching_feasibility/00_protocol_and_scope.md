# P6-A Protocol and Scope

Status: **offline audit only**. No FOGA mutation/evolution, Concorde, BQ training/evaluation, HGB retraining, feature-ranking search, or P6-B execution occurred.

- Frozen Top5 source: P6-0 M4-conditioned HGB40 audit `87dd852d9d309824bba2507ab04a528c19a49b0934117d9e8f97402a6236b183`.
- Target: exactly frozen TSPLIB48 manifest `974df7a019f17d4bdeb4cb29e4f981a243531d94cb056509e96f148d9fa5465e`.
- Held-out22 policy file hash `48741681408da377d994d525e332198a39597e9720d992dd7cec4ddf58d1ed90`; identities/features/performance were not loaded into representation, calibration, matching, or decisions.
- Real offline candidate union: formal C04 `0e8ce149cd6dc93c7cae285483fc67e73bf37460b5ac7e89ab5ccb4256490bb6`, 119,530 unique TSP100 rows. No historical model performance was used.
- Raw F40 values, dimensionless companions, and final conditional scores are all persisted; the final transform does not overwrite raw definitions.
- Only matching objective: empirical TSPLIB48 Top5 distribution matching. Lineage/operator statistics are diagnostic only.
