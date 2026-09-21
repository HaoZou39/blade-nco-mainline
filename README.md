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
