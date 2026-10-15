# BLADE-NCO mainline

This workspace is the clean, frozen mainline for the final dynamic solver-relative QD hard-instance co-evolution experiment. All scientific choices are frozen in [METHOD_FROZEN.md](METHOD_FROZEN.md), [PROTOCOL_STATUS.yaml](PROTOCOL_STATUS.yaml), and [configs/final_20cycle_protocol.yaml](configs/final_20cycle_protocol.yaml).

The formal run is exactly C00–C19 (M0→M20), using fixed QD4/K1000 geometry, exact Uniform10k roots, FOGA-11 once per operator and parent, one centroid-nearest coverage sample per occupied niche, global GP4 fill to hard10k, a cumulative refreshed formal Oracle archive with at most one elite per niche, and frozen 256-epoch BQ adaptation with the Uniform10k anchor.

Validation commands:

```powershell
python scripts/verify_assets.py
python scripts/verify_bootstrap_audit.py
python -m pytest -q
python scripts/run_final_20cycle.py --config configs/final_20cycle_protocol.yaml --run-id blade_final_20cycle_seed2026_v1 --smoke-test
python scripts/verify_final_prelaunch.py
```

Canonical initial launch (exactly once, without `--resume`):

```powershell
python scripts/run_final_20cycle.py --config configs/final_20cycle_protocol.yaml --run-id blade_final_20cycle_seed2026_v1
```

The formal runner creates an exclusive lock and immutable run metadata, evaluates M0, runs all twenty cycles autonomously, performs the final A20/M20 hardness refresh and E20 build, and generates the final report. External benchmarks are passive only. Performance never stops or alters the run.

## S6-AE reduced-budget Hard Only x1

S6-AE tests whether the HGB40 + Elite-QD1000 (P=T) pipeline remains competitive
when each cycle presents 10,000 selected hard instances exactly once per epoch,
with no Uniform anchor or replay. The effective batch size remains 256, giving
40 optimizer updates per epoch and 10,240 updates per 256-epoch cycle. The
user-authorized extended schedule runs C00-C09 (M0-M10) and stops before M11.

Run with the CUDA environment used for the audited experiment:

```powershell
conda activate bq-nco
python scripts/run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py --preflight
python scripts/run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py --smoke-test
python scripts/run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py
```

The prespecified M5 endpoint was a near tie: GAP8 was 1.970390 for x1 versus
1.964619 for Hard Only x2 and 1.948338 for the historical 10k-hard + 10k-Uniform
anchor. The x1 configuration reduced BQ optimizer updates by 49.37%. At M10,
GAP8 was 1.999116 and OriginalTSPLIB70 gap was 2.348372. Across 100,000 selected
slots, 88,767 hard-instance hashes were globally unique. This is a single-seed
descriptive result, not a statistical equivalence claim.

The compact audit bundle is in
[`audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/`](audits/S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1/).
It contains the frozen protocol, GAP summaries, compute/data accounting, final
report, and 40/40 strict verification. Checkpoints, Parquet data, runtime logs,
and generated evaluation work are intentionally excluded from version control.
