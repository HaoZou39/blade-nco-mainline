#!/usr/bin/env python
"""E4.6 corrected-union continuation from verified M5 through M15.

This runner intentionally has no watchdog, polling daemon, restart loop, or
background repair agent.  Atomic stage markers and --resume provide the only
recovery mechanism.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_e4_5_corrected_union_sanity as e45
from e4_6_reporting import BENCHMARKS, build_report
from final_run.stage import atomic_json, atomic_stage, sha256_file, utc_now


RUN_ID = e45.RUN_ID
RUN_DIR = e45.RUN_DIR
AUDIT_DIR = ROOT / "audits" / "E4_6_corrected_union_longrun_M15_v1"
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "e4_6_corrected_union_longrun.lock"
STATUS = RUNTIME / "e4_6_status.json"
COMPLETE_MARKER = ROOT / "E4_6_CORRECTED_UNION_LONGRUN_M15_COMPLETE"
E45_AUDIT = ROOT / "audits" / "E4_5_corrected_union_full_sanity_v1"
HISTORICAL_RUN = ROOT / "runs" / "blade_v2_full_seed2026" / "cycles"
HISTORICAL_EXTERNAL = ROOT / "summary" / "full_main_experiment" / "11_external_8set_trajectory.csv"
STORAGE_ROOT = Path(r"C:\NCO_E4_6_corrected_union_longrun_M15_v1")
M5_EXPECTED = "398eef00c2a8f44e07fc099dd4ffaa97a718d0bc9d0107e9e4771ffa93219b11"


class SourceIntegrityError(RuntimeError):
    pass


class InlineStatus:
    """Write status only when the runner reaches a real event; no polling thread."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "arm": "E4_6_CORRECTED_UNION_LONGRUN", "cycle": None,
            "phase": "initializing", "pid": os.getpid(), "latest_error": None,
            "recovery_state": "NONE", "timestamp": utc_now(),
        }
        atomic_json(STATUS, self.state)

    def update(self, **values: Any) -> None:
        changed = any(self.state.get(key) != value for key, value in values.items())
        self.state.update(values)
        if changed:
            self.state["timestamp"] = utc_now()
            atomic_json(STATUS, self.state)

    def close(self) -> None:
        return


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_map(values: dict[str, str]) -> str:
    payload = "\n".join(f"{key}\t{values[key]}" for key in sorted(values)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def source_scientific_paths() -> list[Path]:
    paths: set[Path] = {
        e45.M0,
        e45.CONFIG_DEFAULT,
        Path(e45.__file__).resolve(),
        ROOT / "scripts" / "e4_5_strict_helpers.py",
        E45_AUDIT / "STRICT_VERIFICATION.json",
        E45_AUDIT / "14_external_8set.csv",
        RUN_DIR / "00_protocol" / "RUN_PROTOCOL_FROZEN.json",
        RUN_DIR / "00_protocol" / "HISTORICAL_INPUT_FINGERPRINTS.json",
        HISTORICAL_EXTERNAL,
        ROOT / "assets" / "retention" / "uniform10k" / "fixed_uniform_anchor_10k_coordinates.npz",
        ROOT / "assets" / "retention" / "uniform10k" / "fixed_uniform_anchor_10k_labels.parquet",
        ROOT / "assets" / "qd" / "cvt" / "centroids.csv",
        ROOT / "assets" / "qd" / "normalization" / "normalization_A_P1P99.json",
        ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json",
        ROOT / "src" / "features" / "qd4.py",
        ROOT / "src" / "mutation" / "__init__.py",
    }
    for cycle in range(5):
        directory = RUN_DIR / f"C{cycle:02d}"
        paths.update(directory.rglob("STAGE_DONE.json"))
        paths.update({
            directory / "CYCLE_COMPLETE.json",
            directory / "cycle_summary.json",
            directory / "strict_cycle_verification.json",
            directory / "cycle_report.md",
            directory / "timing_ledger.csv",
            directory / "06_selection" / "P_next.npz",
            directory / "06_selection" / "T_selected.npz",
            directory / "06_selection" / "parent_age_ledger.parquet",
            directory / "07_oracle" / "T" / "labeled_bundle.npz",
            directory / "08_archive" / "A_next.npz",
            directory / "09_bq_training" / f"M{cycle + 1}_E256.tsp.best",
            directory / "10_post_adaptation" / "adaptation_repair.parquet",
        })
    for cycle in range(15):
        directory = HISTORICAL_RUN / f"C{cycle:02d}"
        paths.add(directory / "CYCLE_COMPLETE.json")
        paths.add(directory / "09_bq_training" / f"M{cycle + 1}_E256.tsp.best")
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise SourceIntegrityError(f"missing source scientific paths: {missing[:10]}")
    return sorted(paths)


def current_fingerprints(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in paths}


def verify_storage() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not STORAGE_ROOT.is_dir():
        raise RuntimeError(f"E4.6 storage root missing: {STORAGE_ROOT}")
    for cycle in range(5, 15):
        link = RUN_DIR / f"C{cycle:02d}"
        target = STORAGE_ROOT / f"C{cycle:02d}"
        if not link.is_dir() or not target.is_dir() or link.resolve() != target.resolve():
            raise RuntimeError(f"E4.6 storage junction invalid: {link} -> {target}")
        rows.append({"cycle": f"C{cycle:02d}", "logical_path": str(link),
                     "physical_path": str(target), "verified": True})
    return rows


def verify_historical_comparator() -> dict[str, Any]:
    external = pd.read_csv(HISTORICAL_EXTERNAL)
    if len(external) != 16 * 8 or set(external.model) != {f"M{i}" for i in range(16)}:
        raise SourceIntegrityError("historical comparator external matrix incomplete")
    if not external.passive_only.fillna(False).all() or external.controls_training.fillna(True).any():
        raise SourceIntegrityError("historical comparator evaluation was not passive")
    for index in range(16):
        checkpoint = e45.M0 if index == 0 else HISTORICAL_RUN / f"C{index - 1:02d}" / "09_bq_training" / f"M{index}_E256.tsp.best"
        expected = set(external[external.model.eq(f"M{index}")].checkpoint_sha256.astype(str))
        if expected != {sha256_file(checkpoint)}:
            raise SourceIntegrityError(f"historical comparator checkpoint mismatch M{index}")
        if index > 0:
            marker = read_json(HISTORICAL_RUN / f"C{index - 1:02d}" / "CYCLE_COMPLETE.json")
            if marker.get("status") != "COMPLETE" or marker.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise SourceIntegrityError(f"historical comparator cycle marker mismatch C{index - 1:02d}")
    return {
        "verified": True, "external_rows": len(external),
        "models": sorted(set(external.model), key=lambda x: int(x[1:])),
        "datasets": sorted(set(external.dataset)),
        "evaluator_config_sha256": sorted(set(external.evaluator_config_sha256.astype(str))),
    }


def verify_source_state(write: bool) -> dict[str, Any]:
    config = e45.validate_config(e45.CONFIG_DEFAULT)
    strict = read_json(E45_AUDIT / "STRICT_VERIFICATION.json")
    if strict.get("status") != "PASS" or not all(strict.get("checks", {}).values()):
        raise SourceIntegrityError("E4.5 strict verification is not PASS")
    frozen = read_json(RUN_DIR / "00_protocol" / "RUN_PROTOCOL_FROZEN.json")
    frozen_runner_verified = (
        frozen.get("config_sha256") == sha256_file(e45.CONFIG_DEFAULT)
        and frozen.get("source_runner_sha256") == sha256_file(Path(e45.__file__).resolve())
    )
    if not frozen_runner_verified:
        raise SourceIntegrityError("E4.5 frozen config/runner identity mismatch")
    frozen_inputs = read_json(RUN_DIR / "00_protocol" / "HISTORICAL_INPUT_FINGERPRINTS.json")
    frozen_assets_verified = True
    for relative, expected in frozen_inputs.items():
        path = ROOT / relative
        frozen_assets_verified &= path.is_file() and sha256_file(path) == expected
    if not frozen_assets_verified:
        raise SourceIntegrityError("E4.5 frozen input fingerprint mismatch")
    for cycle in range(5):
        if not e45.verify_cycle(cycle):
            raise SourceIntegrityError(f"E4.5 source C{cycle:02d} incomplete")
    c04 = read_json(RUN_DIR / "C04" / "CYCLE_COMPLETE.json")
    m5 = e45.model_path(5)
    m5_hash = sha256_file(m5)
    if m5_hash != M5_EXPECTED or c04.get("checkpoint_sha256") != m5_hash:
        raise SourceIntegrityError("M5 hash mismatch")
    model, meta = e45.load_model(5, torch.device("cpu"))
    del model
    if meta.get("sha256") != m5_hash:
        raise SourceIntegrityError("M5 load/hash verification failed")
    p4 = RUN_DIR / "C04" / "06_selection" / "P_next.npz"
    a5 = RUN_DIR / "C04" / "08_archive" / "A_next.npz"
    age = RUN_DIR / "C04" / "06_selection" / "parent_age_ledger.parquet"
    if (sha256_file(p4) != c04.get("P_sha256") or sha256_file(a5) != c04.get("A_sha256")
            or sha256_file(age) != c04.get("age_sha256")):
        raise SourceIntegrityError("P4/A5/age source manifest mismatch")
    uniform_keys = [key for key in frozen_inputs if "assets/retention/uniform10k" in key]
    uniform_anchor_verified = len(uniform_keys) == 2 and all(
        sha256_file(ROOT / key) == frozen_inputs[key] for key in uniform_keys
    )
    if not uniform_anchor_verified:
        raise SourceIntegrityError("frozen Uniform anchor mismatch")
    storage = verify_storage()
    historical = verify_historical_comparator()
    paths = source_scientific_paths()
    fingerprint = current_fingerprints(paths)
    result = {
        "status": "PASS", "label": "E4_6_SOURCE_STATE_VERIFIED",
        "verified_utc": utc_now(), "run_id": RUN_ID,
        "M0_sha256": sha256_file(e45.M0), "M5_sha256": m5_hash,
        "M5_expected_prefix": "398eef00", "source_cycles": [f"C{i:02d}" for i in range(5)],
        "P4_sha256": sha256_file(p4), "A5_sha256": sha256_file(a5),
        "P4_age_sha256": sha256_file(age), "uniform_anchor_verified": uniform_anchor_verified,
        "frozen_assets_verified": bool(frozen_assets_verified),
        "frozen_runner_verified": bool(frozen_runner_verified),
        "historical_comparator_verified": bool(historical["verified"]),
        "historical_comparator": historical, "storage": storage,
        "source_fingerprints": fingerprint,
        "source_fingerprint_digest": digest_map(fingerprint),
        "config": config,
    }
    if write:
        atomic_json(AUDIT_DIR / "02_source_state_verification.json", result)
    return result


def source_still_unchanged(initial: dict[str, Any]) -> bool:
    current = current_fingerprints(source_scientific_paths())
    return current == initial["source_fingerprints"] and digest_map(current) == initial["source_fingerprint_digest"]


def write_cycle_completion(cycle: int, summary: dict[str, Any], resume: bool) -> None:
    directory = e45.cycle_dir(cycle)
    report_dir = directory / "12_reporting"
    inputs = {"checkpoint": summary["checkpoint_hash"], "T": summary["T_hash"],
              "P": summary["P_hash"], "A": summary["A_hash"], "age": summary["age_hash"]}
    with atomic_stage(report_dir, "E4_6_cycle_reporting_and_strict_verification", inputs, resume=resume) as outputs:
        if outputs is not None:
            atomic_json(directory / "cycle_summary.json", summary)
            timing = pd.DataFrame(summary["timing"]["phases"])
            timing = pd.concat([timing, pd.DataFrame([{"phase": "TOTAL", "wall_seconds": summary["timing"]["cycle_wall_seconds"]}])], ignore_index=True)
            timing.to_csv(directory / "timing_ledger.csv", index=False)
            parent = e45.read_parquet(directory / "06_selection" / "P_next.parquet")
            ages = parent.candidate_age.to_numpy(int)
            repair = summary["adaptation_repair"]
            losses = pd.read_csv(directory / "09_bq_training" / "training_loss_curve.csv").set_index("epoch")
            milestones = {epoch: float(losses.loc[epoch, "mean_training_loss"]) for epoch in (1, 32, 64, 128, 192, 256)}
            report = f"""# E4.6 Corrected Union Long-Run — C{cycle:02d}

## Identity

- Solver: M{cycle} -> M{cycle + 1}
- Seed: 2026
- Candidate pool: unique(previous actual parents UNION offspring)
- Checkpoint: fixed E256 `{summary['checkpoint_hash']}`

## GP

- Random500 source: {summary['gp']['source']}
- Refresh time: {next(x['wall_seconds'] for x in summary['timing']['phases'] if x['phase'] == 'Random500 hardness refresh'):.3f}s
- GP time: {summary['gp']['total_runtime_s']:.3f}s

## Candidate pool

- Previous parents: {summary['candidate_accounting']['previous_parent_count']}
- Unique offspring: {summary['candidate_accounting']['offspring_unique_count']}
- Union C_s: {summary['candidate_accounting']['union_unique_count']}
- Parent/offspring overlap: {summary['candidate_accounting']['parent_offspring_overlap']}

## Training T

- N=10,000; old-parent share={summary['survival']['old_parent_fraction_T']:.6f}
- Mean/P90/P95 exact H: {summary['hardness']['mean']:.8f} / {summary['hardness']['P90']:.8f} / {summary['hardness']['P95']:.8f}

## Parent P

- N=10,000; old-parent share={summary['survival']['old_parent_fraction_P']:.6f}
- age0/age1/age2/age3+: {np.mean(ages == 0):.6f} / {np.mean(ages == 1):.6f} / {np.mean(ages == 2):.6f} / {np.mean(ages >= 3):.6f}
- max age={int(ages.max())}; turnover={summary['turnover']['P_turnover']:.6f}; N_eff={summary['diversity']['P']['N_eff']:.3f}

## Adaptation

- Pre/post/delta H: {summary['hardness']['mean']:.8f} / {summary['hardness']['mean'] + repair['mean_delta']:.8f} / {repair['mean_delta']:.8f}
- Paired 95% CI: [{repair['ci95_low']:.8f}, {repair['ci95_high']:.8f}]

## BQ

- 10k hard + 10k frozen Uniform; LR=3.75e-5; epochs=256; checkpoint=E256
- Loss E1/E32/E64/E128/E192/E256: {milestones}
- Training time: {summary['bq']['training_loop_wall_s']:.3f}s

## Time

- Union overhead: {summary['timing']['union_overhead_seconds']:.6f}s ({100*summary['timing']['union_overhead_fraction']:.4f}%)
- Cycle wall-clock: {summary['timing']['cycle_wall_seconds']:.3f}s

Status: COMPLETE
"""
            (directory / "cycle_report.md").write_text(report, encoding="utf-8")
            checks = {
                "union_unique": summary["candidate_accounting"]["union_unique_count"] >= 10_000,
                "T_10k": summary["T_n"] == 10_000, "P_10k": summary["P_n"] == 10_000,
                "T_global_GP": summary["selection"]["T_policy"] == "pure_global_GP_Top10000",
                "P_QD_GP": summary["selection"]["P_policy"] == "centroid_floor_plus_global_GP_fill",
                "age_valid": len(ages) == 10_000 and int(ages.max()) <= cycle,
                "archive_unique": summary["archive_unique"],
                "BQ": summary["bq"]["hard"] == 10_000 and summary["bq"]["uniform_anchor"] == 10_000,
                "LR": float(summary["bq"]["learning_rate"]) == 3.75e-5,
                "E256": summary["bq"]["epochs"] == 256,
                "model_finite": summary["bq"]["model_finite"], "repair_finite": summary["finite"],
            }
            strict = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
            atomic_json(directory / "strict_cycle_verification.json", strict)
            if strict["status"] != "PASS":
                raise RuntimeError(f"E4.6 C{cycle:02d} strict cycle verification failed")
            outputs.extend([directory / "cycle_summary.json", directory / "cycle_report.md",
                            directory / "timing_ledger.csv", directory / "strict_cycle_verification.json"])
    required = {
        "checkpoint": directory / "09_bq_training" / f"M{cycle + 1}_E256.tsp.best",
        "T": directory / "07_oracle" / "T" / "labeled_bundle.npz",
        "P": directory / "06_selection" / "P_next.npz",
        "A": directory / "08_archive" / "A_next.npz",
        "age": directory / "06_selection" / "parent_age_ledger.parquet",
        "repair": directory / "10_post_adaptation" / "adaptation_repair.parquet",
        "timing": directory / "timing_ledger.csv", "report": directory / "cycle_report.md",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    marker = directory / "CYCLE_COMPLETE.json"
    if not marker.is_file():
        atomic_json(marker, {"status": "COMPLETE", "run_id": RUN_ID, "cycle": f"C{cycle:02d}",
                            "continuation": "E4_6", "completed_utc": utc_now(),
                            **{f"{key}_sha256": sha256_file(path) for key, path in required.items()}})
    print_cycle_summary(cycle, summary, ages)


def print_cycle_summary(cycle: int, summary: dict[str, Any], ages: np.ndarray) -> None:
    repair = summary["adaptation_repair"]
    print(f"""
CORRECTED UNION LONG-RUN

Cycle: C{cycle:02d}
Solver: M{cycle} -> M{cycle + 1}
GP: Random500; GP time={summary['gp']['total_runtime_s']:.3f}s
Candidate pool: previous=10000; offspring={summary['candidate_accounting']['offspring_unique_count']}; union={summary['candidate_accounting']['union_unique_count']}
Training T: N=10000; old-parent={summary['survival']['old_parent_fraction_T']:.6f}; mean/P90/P95 H={summary['hardness']['mean']:.6f}/{summary['hardness']['P90']:.6f}/{summary['hardness']['P95']:.6f}
Parent P: N=10000; old-parent={summary['survival']['old_parent_fraction_P']:.6f}; age0/1/2/3+={np.mean(ages==0):.4f}/{np.mean(ages==1):.4f}/{np.mean(ages==2):.4f}/{np.mean(ages>=3):.4f}; max={int(ages.max())}; turnover={summary['turnover']['P_turnover']:.4f}; N_eff={summary['diversity']['P']['N_eff']:.3f}
Adaptation: pre={summary['hardness']['mean']:.6f}; post={summary['hardness']['mean']+repair['mean_delta']:.6f}; delta={repair['mean_delta']:.6f}; CI=[{repair['ci95_low']:.6f},{repair['ci95_high']:.6f}]
BQ: 10k hard + 10k Uniform; LR=3.75e-5; epochs=256; E256; time={summary['bq']['training_loop_wall_s']:.1f}s
Union overhead: {summary['timing']['union_overhead_seconds']:.3f}s ({100*summary['timing']['union_overhead_fraction']:.3f}%)
Cycle wall-clock: {summary['timing']['cycle_wall_seconds']:.1f}s
Status: COMPLETE
""", flush=True)


def evaluate_union(device: torch.device, status: InlineStatus) -> pd.DataFrame:
    import external_v2_passive_blade_eval as evaluator
    old = pd.read_csv(E45_AUDIT / "14_external_8set.csv")
    rows: list[dict[str, Any]] = old[
        old.model.isin([f"M{i}" for i in range(6)]) & old.dataset.isin(BENCHMARKS)
    ].to_dict("records")
    work = AUDIT_DIR / "evaluation_work"
    for index in range(6, 16):
        checkpoint = e45.model_path(index)
        checkpoint_hash = sha256_file(checkpoint)
        model, meta = e45.load_model(index, device)
        if meta.get("sha256") != checkpoint_hash:
            raise RuntimeError(f"evaluation checkpoint load mismatch M{index}")
        model.eval()
        for dataset in BENCHMARKS:
            status.update(arm="E4_6_EVALUATION", cycle=f"M{index}", phase=f"passive_eval_{dataset}",
                          latest_checkpoint=str(checkpoint))
            directory = work / f"M{index}" / dataset
            summary_path = directory / "summary.json"
            if summary_path.is_file():
                cached = read_json(summary_path)
                if cached.get("checkpoint_sha256") == checkpoint_hash:
                    rows.append(cached); continue
            directory.mkdir(parents=True, exist_ok=True)
            payload = evaluator.load_fixed_references(dataset)
            if dataset == "OriginalTSPLIB70":
                detail, summary = evaluator.evaluate_original70(model, device, payload, checkpoint_hash)
            else:
                detail, summary = evaluator.evaluate_tensor_dataset(model, device, dataset, payload, checkpoint_hash)
            summary.update({"run_id": RUN_ID, "model": f"M{index}",
                            "checkpoint_sha256": checkpoint_hash, "passive_only": True,
                            "controls_training": False})
            detail.insert(0, "model", f"M{index}"); detail.insert(1, "dataset", dataset)
            detail.to_parquet(directory / "instance_metrics.parquet", index=False)
            atomic_json(summary_path, summary)
            rows.append(summary)
            print(f"E4.6 EVAL M{index}/{dataset} mean={summary['mean_gap_percent']:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    if len(frame) != 16 * 8:
        raise RuntimeError(f"Union external evaluation matrix size mismatch: {len(frame)}")
    return frame


def initialize(resume: bool, source: dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if LOCK.is_file():
        prior = read_json(LOCK)
        command = e45.base._pid_command(int(prior.get("pid", -1)))
        if "run_e4_6_corrected_union_longrun.py" in command:
            raise RuntimeError(f"active E4.6 process already holds lock: {prior['pid']}")
        if not resume and prior.get("state") not in {"COMPLETE", "FAILED"}:
            raise RuntimeError("stale E4.6 lock requires --resume")
    protocol_state = {
        "schema_version": "E4_6_CORRECTED_UNION_LONGRUN_M15_V1",
        "run_id": RUN_ID, "cycles": list(range(5, 15)), "final_model": "M15",
        "e45_config_sha256": sha256_file(e45.CONFIG_DEFAULT),
        "e45_runner_sha256": sha256_file(Path(e45.__file__).resolve()),
        "e46_runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_fingerprint_digest": source["source_fingerprint_digest"],
        "created_utc": utc_now(),
    }
    frozen_path = AUDIT_DIR / "RUN_PROTOCOL_FROZEN.json"
    if frozen_path.is_file():
        old = read_json(frozen_path)
        for key in ("schema_version", "run_id", "cycles", "final_model", "e45_config_sha256",
                    "e45_runner_sha256", "e46_runner_sha256", "source_fingerprint_digest"):
            if old.get(key) != protocol_state.get(key):
                raise RuntimeError(f"E4.6 resume protocol mismatch: {key}")
    else:
        atomic_json(frozen_path, protocol_state)
    atomic_json(LOCK, {"pid": os.getpid(), "host": socket.gethostname(), "state": "ACTIVE",
                       "timestamp": utc_now(), "command": " ".join(sys.argv)})
    protocol = f"""# E4.6 Corrected Union Long-Run M15 Protocol

- Source: verified E4.5 C00-C04, final M5 `{source['M5_sha256']}`
- New cycles: C05-C14 only; final solver M15
- Candidate pool: `unique(previous actual P UNION new O)`
- GP: deterministic Random500(A_s), frozen GP4 and seeds 2026-2030
- T: pure global GP Top10k; P: frozen-centroid QD floor + GP fill
- BQ: 10k hard + frozen 10k Uniform, LR 3.75e-5, 256E, fixed E256
- Age ledger continues from C04; no cap, penalty or offspring quota
- Evaluation: passive M0-M15 frozen eight-set suite; M0-M5 reused after hash verification
- Comparator: verified historical offspring-only Full M0-M15
- Execution: one deterministic runner, atomic stages, manual `--resume`; no watchdog, polling loop or automatic restart
- Storage: D-path C05-C14 junctions backed by `{STORAGE_ROOT}`; logical task paths are unchanged
"""
    (AUDIT_DIR / "01_protocol.md").write_text(protocol, encoding="utf-8")
    atomic_json(AUDIT_DIR / "02_source_state_verification.json", source)


def write_error(exc: BaseException, label: str = "E4_6_EXECUTION_FAILURE") -> None:
    completed = [f"C{i:02d}" for i in range(5, 15) if (RUN_DIR / f"C{i:02d}" / "CYCLE_COMPLETE.json").is_file()]
    text = f"""# E4.6 Error Report

- Label: `{label}`
- Timestamp: `{utc_now()}`
- PID: `{os.getpid()}`
- Last verified continuation cycles: {completed}
- Exception: `{type(exc).__name__}: {exc}`
- Resume command: `python scripts/run_e4_6_corrected_union_longrun.py --resume`

```text
{traceback.format_exc()}
```
"""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "ERROR_REPORT.md").write_text(text, encoding="utf-8")
    if LOCK.is_file():
        state = read_json(LOCK); state.update({"state": "FAILED", "timestamp": utc_now(), "error": repr(exc)}); atomic_json(LOCK, state)


def smoke_test() -> None:
    source = verify_source_state(write=False)
    assert source["M5_sha256"] == M5_EXPECTED
    assert len(source["storage"]) == 10
    assert len(pd.read_csv(HISTORICAL_EXTERNAL)) == 128
    print("E4_6_CORRECTED_UNION_LONGRUN_SMOKE_TEST=PASS")


def print_final(result: dict[str, Any]) -> None:
    survival = result["survival"]; age = result["age"]; turnover = result["turnover"]
    repair = result["repair"]; overhead = result["overhead"]; external = result["union_external"]
    final_age = age[(age.cycle.eq("C14")) & (age.group.eq("all"))].iloc[0]
    c14s = survival[survival.cycle.eq("C14")].iloc[0]
    c14t = turnover[turnover.cycle.eq("C14")].iloc[0]
    ood = external[external.dataset.eq("OOD5")].set_index("model").mean_gap_percent
    endpoint = result["endpoint"]
    late = result["late"][result["late"].dataset.eq("OOD5")].iloc[0]
    cross = result["crossover"][result["crossover"].dataset.eq("OOD5")].iloc[0]
    print("\nE4.6 CORRECTED UNION LONG-RUN COMPLETE\n")
    print("RUN:\n    C00-C14 complete\n    final solver = M15\n\nNEW CONTINUATION:\n    C05-C14 complete")
    print("\nBQ PROTOCOL:\n    10k hard + 10k Uniform\n    LR = 3.75e-5\n    256E\n    E256")
    print("\nUNION:\n    C_s = unique(P_{s-1} UNION O_s)")
    print(f"\nSURVIVAL:\n    median old-parent share T = {survival.old_parent_fraction_T.median():.6f}\n"
          f"    median old-parent share P = {survival.old_parent_fraction_P.median():.6f}\n"
          f"    C14 parent share T/P = {c14s.old_parent_fraction_T:.6f}/{c14s.old_parent_fraction_P:.6f}")
    print(f"\nAGE AT C14:\n    age0/1/2/3+/5+ = {final_age.age0_fraction:.6f}/{final_age.age1_fraction:.6f}/"
          f"{final_age.age2_fraction:.6f}/{final_age.age3plus_fraction:.6f}/{final_age.age5plus_fraction:.6f}\n"
          f"    max age = {int(final_age.max_age)}")
    print(f"\nTURNOVER:\n    median = {turnover.P_turnover.median():.6f}\n    C14 = {c14t.P_turnover:.6f}")
    print(f"\nSTAGNATION:\n    {result['stagnation']}")
    print(f"\nADAPTATION:\n    repair delta C05-C14 = {repair.tail(10).mean_delta.round(6).tolist()}")
    print(f"\nEXTERNAL:\n    Union OOD5 M0/M5/M10/M15 = {ood['M0']:.6f}/{ood['M5']:.6f}/{ood['M10']:.6f}/{ood['M15']:.6f}")
    print("\nMATCHED M15:")
    for row in endpoint.itertuples(): print(f"    {row.dataset}: {row.delta_gap:+.6f} pp")
    print(f"\nLATE M10-M15 OOD5:\n    Union={late.late_M10_M15_union_mean:.6f}\n"
          f"    OffspringOnly={late.late_M10_M15_offspring_mean:.6f}\n    Delta={late.late_delta_gap:+.6f}")
    print(f"\nCROSSOVER:\n    first Union-better OOD5 = {cross.first_union_better_checkpoint}\n"
          f"    remained better afterward = {cross.remained_union_better_after_first}\n    reversals = {int(cross.later_reversals)}")
    print(f"\nTIME:\n    continuation = {overhead.tail(10).cycle_wall_seconds.sum():.1f}s\n"
          f"    corrected C00-C14 = {overhead.cycle_wall_seconds.sum():.1f}s\n"
          f"    mean Union overhead = {overhead.union_overhead_seconds.mean():.3f}s ({100*overhead.overhead_fraction.mean():.3f}%)")
    print(f"\nRUN STATUS:\n    {result['run_status']}\n\nMATCHED OUTCOME:\n    {result['matched_outcome']}\n\nNEXT STEP:\n    STOP for human review.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test(); return 0
    if args.preflight:
        source = verify_source_state(write=False)
        print(json.dumps({key: source[key] for key in ("status", "M5_sha256", "source_fingerprint_digest",
                                                       "uniform_anchor_verified", "frozen_assets_verified",
                                                       "historical_comparator_verified")}, indent=2))
        return 0
    status = InlineStatus()
    try:
        source = verify_source_state(write=False)
        initialize(args.resume, source)
        e45.write_cycle_completion = write_cycle_completion
        device = e45.base._device()
        print(f"E4.6 START device={device} resume={args.resume}; no watchdog/polling loop", flush=True)
        for cycle in range(5, 15):
            status.update(arm="E4_6_CORRECTED_UNION_LONGRUN", cycle=f"C{cycle:02d}", phase="cycle_dispatch")
            if cycle > 5 and not e45.verify_cycle(cycle - 1):
                raise RuntimeError(f"previous cycle C{cycle - 1:02d} not verified")
            e45.run_cycle(cycle, device, args.resume, status)
        status.update(arm="E4_6_EVALUATION", cycle=None, phase="passive_M6_M15_8set")
        external = evaluate_union(device, status)
        status.update(arm="E4_6_REPORTING", cycle=None, phase="matched_audit_and_strict_verification")
        unchanged = source_still_unchanged(source)
        result = build_report(ROOT, RUN_DIR, AUDIT_DIR, e45, source["config"], external, source, unchanged)
        COMPLETE_MARKER.write_text(result["run_status"] + "\n" + result["matched_outcome"] + "\n", encoding="utf-8")
        lock = read_json(LOCK); lock.update({"state": "COMPLETE", "timestamp": utc_now(),
                                            "result": result["run_status"], "matched_outcome": result["matched_outcome"]}); atomic_json(LOCK, lock)
        status.update(arm="COMPLETE", cycle=None, phase="E4_6_CORRECTED_UNION_LONGRUN_M15_COMPLETE",
                      latest_checkpoint=str(e45.model_path(15)), last_valid_artifact=str(COMPLETE_MARKER))
        print_final(result)
        return 0
    except SourceIntegrityError as exc:
        print("E4_6_SOURCE_STATE_INTEGRITY_FAILURE", file=sys.stderr, flush=True)
        status.update(latest_error=repr(exc), recovery_state="STOPPED_SOURCE_INTEGRITY")
        write_error(exc, "E4_6_SOURCE_STATE_INTEGRITY_FAILURE")
        return 2
    except BaseException as exc:
        status.update(latest_error=repr(exc), recovery_state="MANUAL_RESUME_REQUIRED")
        write_error(exc)
        return 1
    finally:
        status.close()


if __name__ == "__main__":
    raise SystemExit(main())
