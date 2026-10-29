# P6-G3 Budget and Reproducibility

## Mutation budgets

| seed | A_conceptual_attempts | B_conceptual_attempts | A_historical_wall_s | B_incremental_wall_s | B_reused_common_attempts | B_new_attempts | A_raw_unique_nonidentity | B_raw_unique_nonidentity |
|---|---|---|---|---|---|---|---|---|
| Original | 110000 | 110000 | 271.87 | 56.376 | 99000 | 11000 | 109548 | 109573 |
| S1 | 110000 | 110000 | 272.25 | 56.226 | 99000 | 11000 | 109523 | 109546 |
| S2 | 110000 | 110000 | 272.44 | 55.999 | 99000 | 11000 | 109553 | 109582 |

Arm A is the exact historical P6-B G1 control, accepted only after deterministic replay of 44 attempts per seed plus full parent/order/profile/namespace/hash gates. Arm B reuses the exact paired 99,000 common-parent attempts and physically evaluates 11,000 supplement-parent attempts per seed. This reduces redundant computation but does not change either arm's conceptual 110,000-attempt budget. The B end-to-end intervention also includes supplement construction and cannot be called equal-cost end to end.

## Replay gates

```json
{
  "Original": {
    "historical_bundle_sha256": "41882b7e47333af34535426ad9b885913fd63ace3deea3c684a5acc6b05d4737",
    "historical_manifest_sha256": "1498dc0e65b639dd790aa902ed8d113eec2a28449b813f4a93d527494a0680b8",
    "historical_raw_n": 109571,
    "manifest_attempts": true,
    "nonidentity_A": {
      "accepted_unique_source_n": 109571,
      "bundle_sha256": "89112331d607b3dd2d89ceb87f11be59e1400d9cf30ec4b19789f4bea93ed27b",
      "identity_removed_n": 23,
      "source_historical_bundle": "target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/Original/shared_G1/mutation/offspring_coordinates.npz",
      "source_sha256": "41882b7e47333af34535426ad9b885913fd63ace3deea3c684a5acc6b05d4737",
      "status": "PASS",
      "valid_nonidentity_unique_n": 109548
    },
    "parent_schedule": true,
    "profile_hash": true,
    "replay_hash": true,
    "replay_seed": true,
    "replay_status": true,
    "replayed_attempt_n": 44,
    "replayed_parent_n": 4,
    "runtime_namespace": true,
    "runtime_parent_membership": true
  },
  "S1": {
    "historical_bundle_sha256": "f3ed1829e20b759df861db9c0c09120f181217ac8f64b20cf84fde2dc5568771",
    "historical_manifest_sha256": "fa3b2cba178a0bf7a3083326c1dc9c6886c8b2f280374b624f2a5bb0ae1805f1",
    "historical_raw_n": 109541,
    "manifest_attempts": true,
    "nonidentity_A": {
      "accepted_unique_source_n": 109541,
      "bundle_sha256": "625dfae84ed6c925fdefd9d6a47612d6feb7eecb7ffd2c3e5c9067928ab00529",
      "identity_removed_n": 18,
      "source_historical_bundle": "target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S1/shared_G1/mutation/offspring_coordinates.npz",
      "source_sha256": "f3ed1829e20b759df861db9c0c09120f181217ac8f64b20cf84fde2dc5568771",
      "status": "PASS",
      "valid_nonidentity_unique_n": 109523
    },
    "parent_schedule": true,
    "profile_hash": true,
    "replay_hash": true,
    "replay_seed": true,
    "replay_status": true,
    "replayed_attempt_n": 44,
    "replayed_parent_n": 4,
    "runtime_namespace": true,
    "runtime_parent_membership": true
  },
  "S2": {
    "historical_bundle_sha256": "a596563ab357b30325ffb1be2f9ef77402c343700ded58560176f6d5181bb871",
    "historical_manifest_sha256": "c13824677d25f1b5c8b4fa1cf22b0a199c7db78efe798f1d022232fbdad72fab",
    "historical_raw_n": 109570,
    "manifest_attempts": true,
    "nonidentity_A": {
      "accepted_unique_source_n": 109570,
      "bundle_sha256": "3eba97a8019044447898fe3a6c4057d36daff9021d6752e9711049072c9023be",
      "identity_removed_n": 17,
      "source_historical_bundle": "target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution/S2/shared_G1/mutation/offspring_coordinates.npz",
      "source_sha256": "a596563ab357b30325ffb1be2f9ef77402c343700ded58560176f6d5181bb871",
      "status": "PASS",
      "valid_nonidentity_unique_n": 109553
    },
    "parent_schedule": true,
    "profile_hash": true,
    "replay_hash": true,
    "replay_seed": true,
    "replay_status": true,
    "replayed_attempt_n": 44,
    "replayed_parent_n": 4,
    "runtime_namespace": true,
    "runtime_parent_membership": true
  }
}
```

- Supplement attempted candidates: 1000; accepted unique: 1,000.
- F40 new extraction: `{"missing_n": 33948, "shards": 136, "wall_seconds": 180.74592993408442, "workers": 12}`.
- Python `3.10.12`; NumPy `2.2.6`; pandas `2.3.3`; source commit `ce86c148e196398560a44b6affa4cc371dffe201`.
- No TSPLIB22, Concorde, BQ/HGB/GP inference/training, survivor selection, or G2 operation was invoked.
