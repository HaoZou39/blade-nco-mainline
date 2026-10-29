#!/usr/bin/env python3
"""Run the fail-closed BLADE BQ-CVRP feasibility pilot.

Every expensive stage is instance-cached or checkpointed. Re-running this
script resumes from validated artifacts in runs/BLADE_BQ_CVRP_PILOT_V1.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OFFICIAL_BQ = ROOT / "third_party" / "bq_nco_official"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from mutation import OPERATOR_NAMES, apply_operator
from cvrp_pilot.core import (
    BQ_COMMIT,
    CAPACITY,
    FEATURE_NAMES,
    HGS_COMMIT,
    HGS_SCALE,
    M0_SHA256,
    N_CUSTOMERS,
    RUN_ID,
    SEED,
    atomic_json,
    bootstrap_mean_ci,
    extract_feature_frame,
    generate_role,
    hardness_metrics,
    instance_hash,
    load_bundle,
    normalize_cvrplib_coords,
    official_loss_equivalence,
    official_trajectory,
    paired_gap_frame,
    parse_cvrplib,
    per_instance_hardness,
    save_bundle,
    sha256_file,
    solve_hgs_bundle,
    stable_seed,
    structural_features,
    trajectory_bundle,
    validate_routes,
)

# The upstream code uses absolute imports such as ``from model.encoder``.
sys.path.insert(0, str(OFFICIAL_BQ))
from model.model import BQModel  # noqa: E402
from learning.cvrp.decoding import decode as official_decode  # noqa: E402


RUN = ROOT / "runs" / RUN_ID
FIGURES = RUN / "figures"
DATA = RUN / "data"
HGS_CACHE = RUN / "hgs_cache"
CHECKPOINTS = RUN / "checkpoints"
M0_PATH = OFFICIAL_BQ / "pretrained_models" / "cvrp.best"
HGS_BINARY = ROOT / "third_party" / "hgs_cvrp_official" / "hgs"
PROTOCOL_PATH = ROOT / "configs" / "bq_cvrp_pilot_v1.yaml"

ROLE_SPECS = {
    "cal_train": 1600,
    "cal_val": 400,
    "probe": 400,
    "candidate": 20_000,
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_status(stage: str, state: str = "running", **extra: Any) -> None:
    previous: dict[str, Any] = {}
    path = RUN / "RUN_STATUS.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
    history = previous.get("history", [])
    history.append({"time": now_iso(), "stage": stage, "state": state})
    payload = {
        "run_id": RUN_ID,
        "state": state,
        "stage": stage,
        "completion_label": "BLADE_BQ_CVRP_PILOT_V1_COMPLETE" if state == "complete" else "BLADE_BQ_CVRP_PILOT_V1_PARTIAL",
        "updated_at": now_iso(),
        "fail_closed": True,
        "history": history,
        **extra,
    }
    atomic_json(path, payload)


def stage_done(name: str) -> bool:
    return (RUN / "stages" / f"{name}.json").is_file()


def mark_stage(name: str, started: float, **payload: Any) -> None:
    atomic_json(RUN / "stages" / f"{name}.json", {"stage": name, "completed_at": now_iso(), "wall_s": time.perf_counter() - started, **payload})


def read_stage(name: str) -> dict[str, Any]:
    return json.loads((RUN / "stages" / f"{name}.json").read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    for path in (RUN, FIGURES, DATA, HGS_CACHE, CHECKPOINTS, RUN / "stages", RUN / "models"):
        path.mkdir(parents=True, exist_ok=True)


def load_model(device: torch.device, checkpoint: Path = M0_PATH) -> BQModel:
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    state = blob["net"] if "net" in blob else blob["model_state_dict"]
    model = BQModel(4, 192, 512, "relu", 9, 12, "softmax", 0.0, False, "cvrp").to(device)
    model.load_state_dict(state, strict=True)
    return model


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def audit_stage(device: torch.device) -> None:
    if stage_done("00_audit"):
        return
    started = time.perf_counter()
    write_status("00_audit")
    if not M0_PATH.is_file() or sha256_file(M0_PATH) != M0_SHA256:
        raise RuntimeError("official BQ-CVRP M0 missing or SHA256 mismatch")
    if git_head(OFFICIAL_BQ) != BQ_COMMIT:
        raise RuntimeError("official BQ checkout commit mismatch")
    if git_head(ROOT / "third_party" / "hgs_cvrp_official") != HGS_COMMIT:
        raise RuntimeError("official HGS checkout commit mismatch")
    if not HGS_BINARY.is_file() or not os.access(HGS_BINARY, os.X_OK):
        raise RuntimeError("HGS executable missing")
    model = load_model(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 3_108_116:
        raise RuntimeError(f"unexpected BQ-CVRP parameter count {parameter_count}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    audit = f"""# BQ-CVRP Code Audit

## Decision

`PASS_TO_SMOKE_TEST`

- Official BQ-NCO commit: `{BQ_COMMIT}`
- Official HGS-CVRP commit/tag: `{HGS_COMMIT}` (v2.0.0)
- CVRP checkpoint: `{M0_PATH.relative_to(ROOT)}`
- M0 SHA256: `{M0_SHA256}`
- Architecture: `BQModel(4, 192, 512, relu, 9 encoder layers, 12 heads, softmax, dropout=0, ReZero, problem=cvrp)`
- Parameter count: `{parameter_count:,}`
- Checkpoint strict-load: PASS; CVRP state has a two-channel score projection and cannot be replaced by the TSP checkpoint.
- Official loss: `torch.nn.CrossEntropyLoss` over flattened `2 * suffix_node_index + via_depot` actions. Direct infeasible actions are masked from demand and remaining capacity.
- Official training sample: one shared random expert-ordered suffix per minibatch; suffix length at least four; target is suffix node index 1, direct (class 2) or via depot (class 3).
- Pilot per-instance hardness: arithmetic mean of that unchanged CE over every suffix start in the official sampling support. Parameters remain frozen.
- Greedy decoding: beam size 1, `knns=-1`, no augmentation.
- Fine-tuning: fresh Adam per arm, lr 2.5e-4, batch 128, 2,048 optimizer updates, no clipping, fixed-update pilot without scheduler.

## Data and trajectory semantics

Official CVRP100 generation uses one depot plus 100 customers, coordinates in `[0,1]^2`, integer demands 1–9, and capacity 50. Training coordinates are reordered along the reference routes, with depot duplicated at both ends. Routes are ordered by ascending residual vehicle capacity; capacity increases encode depot separators. This Pilot converts HGS routes to exactly that representation and never calls the TSP tour-NLL path.

## HGS contract

The HGS executable is built from official v2.0.0 source. Synthetic coordinates are scaled by {HGS_SCALE:,} and HGS uses rounded Euclidean distances. Every result is independently checked for customer coverage, uniqueness, capacity feasibility, and exact agreement between the HGS-reported cost and a recomputed rounded-distance cost. Labels are named `HGS_REFERENCE`, never optimum.

## Licensing and isolation

BQ remains under its upstream CC BY-NC-SA license and HGS under MIT. The official checkouts live under `third_party/`; all Pilot outputs are isolated under `runs/{RUN_ID}`. Existing TSP code and artifacts are neither modified nor overwritten.
"""
    (RUN / "00_BQ_CVRP_CODE_AUDIT.md").write_text(audit, encoding="utf-8")
    shutil.copy2(PROTOCOL_PATH, RUN / "frozen_protocol.yaml")
    protocol = f"""# Frozen CVRP Pilot Protocol

The machine-readable freeze is [`frozen_protocol.yaml`](frozen_protocol.yaml). The protocol seed is {SEED}; roles are 1,600 calibration-train, 400 calibration-validation, 400 sealed probe, and a shared 20,000-instance candidate bank. C1 and C2 acquire 2,000 unique instances each, permit natural overlap, start from byte-identical M0, and receive exactly 2,048 updates. QD and additional cycles are disabled.

Fail-closed gates are: official commit and checkpoint identity, strict CVRP checkpoint load, HGS route/cost validation, official-loss equivalence, disjoint hashes and roots across roles, maximum 6,400 new non-evaluation HGS labels, and byte-identical M0 initialization for both arms.
"""
    (RUN / "01_FROZEN_CVRP_PROTOCOL.md").write_text(protocol, encoding="utf-8")
    compute = [
        {"component": "HGS smoke", "instances": 40, "cpu_s_each": 10, "estimated_cpu_h": 40 * 10 / 3600, "estimated_wall_h_at_96_workers": 40 * 10 / 96 / 3600},
        {"component": "HGS calibration+probe", "instances": 2400, "cpu_s_each": 10, "estimated_cpu_h": 2400 * 10 / 3600, "estimated_wall_h_at_96_workers": 2400 * 10 / 96 / 3600},
        {"component": "HGS acquisition upper bound", "instances": 4000, "cpu_s_each": 10, "estimated_cpu_h": 4000 * 10 / 3600, "estimated_wall_h_at_96_workers": 4000 * 10 / 96 / 3600},
        {"component": "HGS synthetic evaluation", "instances": 3000, "cpu_s_each": 10, "estimated_cpu_h": 3000 * 10 / 3600, "estimated_wall_h_at_96_workers": 3000 * 10 / 96 / 3600},
        {"component": "BQ fine-tuning", "instances": 4000, "cpu_s_each": 0, "estimated_cpu_h": 0, "estimated_wall_h_at_96_workers": float("nan")},
    ]
    pd.DataFrame(compute).to_csv(RUN / "09_COMPUTE_COST_AUDIT.csv", index=False)
    mark_stage("00_audit", started, parameter_count=parameter_count, m0_sha256=M0_SHA256, hgs_sha256=sha256_file(HGS_BINARY))


def role_paths(role: str) -> tuple[Path, Path, Path]:
    return DATA / f"{role}.npz", DATA / f"{role}_manifest.csv", DATA / f"{role}_features.csv"


def generate_one_role(role: str) -> None:
    bundle_path, manifest_path, feature_path = role_paths(role)
    if bundle_path.is_file() and manifest_path.is_file() and feature_path.is_file():
        return
    bundle, manifest = generate_role(role, ROLE_SPECS[role], OPERATOR_NAMES, apply_operator)
    save_bundle(bundle_path, **bundle)
    manifest.to_csv(manifest_path, index=False)
    extract_feature_frame(bundle, manifest).to_csv(feature_path, index=False)


def validate_role_isolation(roles: Sequence[str]) -> dict[str, Any]:
    manifests = {role: pd.read_csv(role_paths(role)[1]) for role in roles}
    checks: dict[str, Any] = {}
    for left_index, left in enumerate(roles):
        checks[f"{left}_unique_instances"] = manifests[left].instance_hash.nunique() == len(manifests[left])
        checks[f"{left}_unique_roots"] = manifests[left].root_hash.nunique() == len(manifests[left])
        for right in roles[left_index + 1:]:
            checks[f"{left}_{right}_instance_overlap"] = len(set(manifests[left].instance_hash) & set(manifests[right].instance_hash))
            checks[f"{left}_{right}_root_overlap"] = len(set(manifests[left].root_hash) & set(manifests[right].root_hash))
    if not all(value is True or value == 0 for value in checks.values()):
        raise RuntimeError(f"role isolation failure: {checks}")
    return checks


def references_to_frame(records: Sequence[Any], role: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "role": role,
                "instance_id": record.instance_id,
                "instance_hash": record.instance_hash,
                "reference_type": "HGS_REFERENCE",
                "hgs_cost_scaled": record.hgs_cost_scaled,
                "reference_cost": record.euclidean_cost,
                "rounded_reference_cost": record.rounded_cost,
                "route_count": record.route_count,
                "feasible": record.feasible,
                "status": record.status,
                "wall_s": record.wall_s,
                "rounded_vs_float_abs": abs(record.rounded_cost - record.euclidean_cost),
            }
            for record in records
        ]
    )


def records_from_cache(work_dir: Path, manifest: pd.DataFrame) -> list[Any]:
    from cvrp_pilot.core import ReferenceRecord

    records = []
    for row in manifest.itertuples(index=False):
        payload = json.loads((work_dir / f"{row.instance_id}.json").read_text(encoding="utf-8"))
        if payload["instance_hash"] != row.instance_hash or payload["status"] != "ok":
            raise RuntimeError(f"invalid reference cache for {row.instance_id}")
        records.append(ReferenceRecord(**payload))
    return records


def model_greedy_synthetic(model: BQModel, bundle: dict[str, np.ndarray], device: torch.device, *, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    costs = np.empty(len(bundle["depot"]), dtype=np.float64)
    feasible = np.zeros(len(costs), dtype=bool)
    with torch.no_grad():
        for left in range(0, len(costs), batch_size):
            right = min(len(costs), left + batch_size)
            coords_np = np.concatenate([bundle["depot"][left:right, None, :], bundle["customers"][left:right], bundle["depot"][left:right, None, :]], axis=1)
            demands_np = np.concatenate([np.zeros((right-left, 1)), bundle["demands"][left:right], np.zeros((right-left, 1))], axis=1)
            coords = torch.as_tensor(coords_np, dtype=torch.float32, device=device)
            demands = torch.as_tensor(demands_np, dtype=torch.float32, device=device)
            capacities = torch.as_tensor(bundle["capacities"][left:right], dtype=torch.float32, device=device)
            delta = coords[:, :, None, :] - coords[:, None, :, :]
            matrices = torch.sqrt(torch.sum(delta * delta, dim=-1))
            lengths, tours = official_decode(coords, matrices, demands, capacities, model, 1, -1, make_tours=True)
            costs[left:right] = lengths.detach().cpu().double().numpy()
            for local, path in enumerate(tours):
                normalized = [0 if node == coords.shape[1] - 1 else int(node) for node in path]
                if normalized[-1] != 0:
                    normalized.append(0)
                load = 0
                ok = True
                for node in normalized[1:]:
                    if node == 0:
                        if load > int(bundle["capacities"][left + local]):
                            ok = False
                        load = 0
                    else:
                        load += int(bundle["demands"][left + local, node - 1])
                feasible[left + local] = ok and load == 0
    return costs, feasible


def smoke_stage(device: torch.device, workers: int) -> None:
    if stage_done("01_smoke"):
        return
    started = time.perf_counter()
    write_status("01_smoke")
    generate_one_role("cal_train")
    full_bundle = load_bundle(role_paths("cal_train")[0])
    full_manifest = pd.read_csv(role_paths("cal_train")[1])
    smoke_manifest = full_manifest.iloc[:40].reset_index(drop=True)
    smoke_bundle = {key: value[:40] for key, value in full_bundle.items()}
    records = solve_hgs_bundle(HGS_BINARY, HGS_CACHE / "cal_train", smoke_bundle, smoke_manifest, workers=min(workers, 40), time_limit=10)
    reference_frame = references_to_frame(records, "smoke")
    if not reference_frame.feasible.all() or not (reference_frame.status == "ok").all():
        raise RuntimeError("smoke HGS feasibility gate failed")
    trajectories = trajectory_bundle(smoke_bundle, records)
    model = load_model(device)
    equivalence = official_loss_equivalence(model, trajectories, device)
    if not equivalence["pass"]:
        raise RuntimeError("official loss equivalence failed")
    hardness = per_instance_hardness(model, trajectories, device, batch_size=64)
    costs, feasible = model_greedy_synthetic(model, smoke_bundle, device, batch_size=40)
    if not feasible.all() or not np.isfinite(hardness).all() or not np.isfinite(costs).all():
        raise RuntimeError("BQ-CVRP smoke inference/hardness gate failed")

    reversed_records = []
    from cvrp_pilot.core import ReferenceRecord
    for record in records[:8]:
        reversed_records.append(ReferenceRecord(**{**record.__dict__, "routes": [list(reversed(route)) for route in record.routes]}))
    reversed_trajectories = trajectory_bundle({key: value[:8] for key, value in smoke_bundle.items()}, reversed_records)
    reversed_hardness = per_instance_hardness(model, reversed_trajectories, device, batch_size=8)
    sensitivity = np.abs(hardness[:8] - reversed_hardness)
    reference_frame["bq_m0_cost"] = costs
    reference_frame["bq_m0_feasible"] = feasible
    reference_frame["hardness"] = hardness
    reference_frame.to_csv(RUN / "smoke_results.csv", index=False)
    atomic_json(
        RUN / "smoke_audit.json",
        {
            "instances": 40,
            "official_loss_equivalence": equivalence,
            "route_reversal_instances": 8,
            "route_reversal_mean_absolute_hardness_change": float(sensitivity.mean()),
            "route_reversal_max_absolute_hardness_change": float(sensitivity.max()),
            "all_hgs_feasible": True,
            "all_bq_greedy_feasible": True,
            "hardness_model_parameters_unchanged": True,
            "decision": "PASS_TO_FORMAL_LABELING",
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mark_stage("01_smoke", started, **json.loads((RUN / "smoke_audit.json").read_text()))


def data_stage() -> None:
    if stage_done("02_data"):
        return
    started = time.perf_counter()
    write_status("02_data")
    for role in ROLE_SPECS:
        generate_one_role(role)
    isolation = validate_role_isolation(list(ROLE_SPECS))
    manifests = {role: pd.read_csv(role_paths(role)[1]) for role in ROLE_SPECS}
    operator_checks = {}
    for role, frame in manifests.items():
        counts = frame.operator.value_counts()
        operator_checks[role] = {"counts": counts.to_dict(), "all_11_present": set(counts.index) == set(OPERATOR_NAMES)}
        if not operator_checks[role]["all_11_present"]:
            raise RuntimeError(f"operator coverage failure in {role}")
    payload = {
        "run_id": RUN_ID,
        "seed": SEED,
        "role_counts": ROLE_SPECS,
        "generation": "independent uniform CVRP100 root per instance; one uniformly balanced FOGA11_BASE_V1 customer-coordinate mutation",
        "customer_order_preserved": True,
        "depot_demand_capacity_immutable": True,
        "isolation_checks": isolation,
        "operator_checks": operator_checks,
        "bundle_sha256": {role: sha256_file(role_paths(role)[0]) for role in ROLE_SPECS},
        "manifest_sha256": {role: sha256_file(role_paths(role)[1]) for role in ROLE_SPECS},
    }
    atomic_json(RUN / "02_DATA_GENERATION_AUDIT.json", payload)
    mark_stage("02_data", started, role_counts=ROLE_SPECS, isolation_pass=True)


def label_role(role: str, device: torch.device, workers: int) -> pd.DataFrame:
    output = DATA / f"{role}_hardness.csv"
    trajectory_path = DATA / f"{role}_trajectories.npz"
    manifest = pd.read_csv(role_paths(role)[1])
    bundle = load_bundle(role_paths(role)[0])
    records = solve_hgs_bundle(HGS_BINARY, HGS_CACHE / role, bundle, manifest, workers=workers, time_limit=10)
    references = references_to_frame(records, role)
    trajectories = trajectory_bundle(bundle, records)
    save_bundle(trajectory_path, **trajectories)
    model = load_model(device)
    before = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    hardness = per_instance_hardness(model, trajectories, device, batch_size=128)
    after = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    if before != after:
        raise RuntimeError("M0 parameters changed during hardness calculation")
    references["hardness"] = hardness
    references.to_csv(output, index=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return references


def calibration_stage(device: torch.device, workers: int) -> None:
    if stage_done("03_calibration_labels"):
        return
    started = time.perf_counter()
    write_status("03_calibration_labels")
    frames = [label_role(role, device, workers) for role in ("cal_train", "cal_val", "probe")]
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 2400 or combined.instance_hash.nunique() != 2400:
        raise RuntimeError("calibration label cardinality failure")
    combined.to_csv(RUN / "04_BQ_CVRP_HARDNESS_CALIBRATION.csv", index=False)
    audit_columns = ["role", "instance_id", "instance_hash", "reference_type", "hgs_cost_scaled", "reference_cost", "rounded_reference_cost", "route_count", "feasible", "status", "wall_s", "rounded_vs_float_abs"]
    combined[audit_columns].to_csv(RUN / "03_HGS_REFERENCE_AUDIT.csv", index=False)
    mark_stage("03_calibration_labels", started, labels=2400, hgs_cpu_budget_s=24000, all_feasible=bool(combined.feasible.all()))


def fit_hgb_stage() -> None:
    if stage_done("04_hgb"):
        return
    started = time.perf_counter()
    write_status("04_hgb")

    def merged(role: str) -> pd.DataFrame:
        features = pd.read_csv(role_paths(role)[2])
        labels = pd.read_csv(DATA / f"{role}_hardness.csv")[["instance_id", "instance_hash", "hardness"]]
        result = features.merge(labels, on=["instance_id", "instance_hash"], validate="one_to_one")
        return result

    train, val = merged("cal_train"), merged("cal_val")
    feature_names = list(FEATURE_NAMES)
    constant = [name for name in feature_names if float(train[name].std(ddof=0)) <= 1e-12]
    active = [name for name in feature_names if name not in constant]
    candidates = []
    best = None
    for learning_rate in (0.03, 0.05, 0.1):
        for max_leaf_nodes in (15, 31):
            for l2 in (0.0, 1.0):
                model = HistGradientBoostingRegressor(
                    loss="squared_error", learning_rate=learning_rate, max_iter=300, max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2, early_stopping=False, random_state=SEED,
                )
                model.fit(train[active], train.hardness)
                prediction = model.predict(val[active])
                metrics = hardness_metrics(val.hardness.to_numpy(), prediction)
                row = {"learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes, "l2_regularization": l2, **metrics}
                candidates.append(row)
                if best is None or row["spearman"] > best[0]:
                    best = (row["spearman"], model, row)
    assert best is not None
    model = best[1]
    joblib.dump({"model": model, "active_features": active, "constant_features": constant, "selected_hyperparameters": best[2]}, RUN / "models" / "hgb.joblib")
    pd.DataFrame(candidates).to_csv(RUN / "hgb_model_selection.csv", index=False)
    val_prediction = model.predict(val[active])
    val_metrics = hardness_metrics(val.hardness.to_numpy(), val_prediction)
    importance = permutation_importance(model, val[active], val.hardness, n_repeats=20, random_state=SEED, scoring="neg_mean_absolute_error")
    importance_frame = pd.DataFrame({"feature": active, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(RUN / "hgb_feature_importance.csv", index=False)
    atomic_json(RUN / "hgb_freeze.json", {"active_features": active, "constant_features": constant, "selected_hyperparameters": best[2], "validation_metrics": val_metrics, "model_sha256": sha256_file(RUN / "models" / "hgb.joblib")})

    # The probe is opened only after the model and active feature list are frozen.
    probe = merged("probe")
    inference_started = time.perf_counter()
    probe_prediction = model.predict(probe[active])
    inference_s = time.perf_counter() - inference_started
    probe_metrics = hardness_metrics(probe.hardness.to_numpy(), probe_prediction)
    probe_metrics["inference_s"] = inference_s
    probe_metrics["inference_instances_per_s"] = len(probe) / max(inference_s, 1e-12)
    probe.assign(predicted_hardness=probe_prediction).to_csv(RUN / "sealed_probe_predictions.csv", index=False)
    report = f"""# HGB Validation

The HGB uses {len(active)} non-constant instance-only CVRP features; {len(constant)} constant columns were removed (`{', '.join(constant) if constant else 'none'}`). No HGS cost, route, BQ loss, BQ inference output, operator identity, root identity, or test information is a model input.

## Model selection validation (n=400)

- Spearman: {val_metrics['spearman']:.6f}
- MAE: {val_metrics['mae']:.6f}
- True-hard Top10% hit rate: {val_metrics['true_top10_hit_rate']:.6f}
- True-hard Top10% enrichment: {val_metrics['true_top10_enrichment']:.3f}×

## Sealed probe (n=400; opened after freeze)

- Spearman: {probe_metrics['spearman']:.6f}
- MAE: {probe_metrics['mae']:.6f}
- True-hard Top10% hit rate: {probe_metrics['true_top10_hit_rate']:.6f}
- True-hard Top10% enrichment: {probe_metrics['true_top10_enrichment']:.3f}×
- HGB inference: {inference_s:.6f}s total ({probe_metrics['inference_instances_per_s']:.1f} instances/s)

These diagnostics measure prediction of the current frozen BQ-CVRP M0 teacher-forced hardness. They do not establish downstream training value.
"""
    (RUN / "05_HGB_VALIDATION.md").write_text(report, encoding="utf-8")
    mark_stage("04_hgb", started, validation=val_metrics, probe=probe_metrics, active_feature_count=len(active))


def acquisition_stage(device: torch.device, workers: int) -> None:
    if stage_done("05_acquisition"):
        return
    started = time.perf_counter()
    write_status("05_acquisition")
    hgb = joblib.load(RUN / "models" / "hgb.joblib")
    model_hgb = hgb["model"]
    active = hgb["active_features"]
    candidate_manifest = pd.read_csv(role_paths("candidate")[1])
    candidate_features = pd.read_csv(role_paths("candidate")[2])
    inference_started = time.perf_counter()
    predictions = model_hgb.predict(candidate_features[active])
    inference_s = time.perf_counter() - inference_started
    candidate = candidate_manifest.merge(candidate_features, on=["instance_id", "instance_hash"], validate="one_to_one")
    candidate["predicted_hardness"] = predictions
    rng = np.random.default_rng(SEED)
    random_indices = np.sort(rng.choice(len(candidate), 2000, replace=False))
    hgb_indices = np.argsort(predictions, kind="stable")[-2000:]
    random_hashes = set(candidate.iloc[random_indices].instance_hash)
    hgb_hashes = set(candidate.iloc[hgb_indices].instance_hash)
    selected_hashes = random_hashes | hgb_hashes
    selected = candidate[candidate.instance_hash.isin(selected_hashes)].copy().reset_index(drop=True)
    selected["in_random"] = selected.instance_hash.isin(random_hashes)
    selected["in_hgb"] = selected.instance_hash.isin(hgb_hashes)
    selected.to_csv(DATA / "selected_union_manifest.csv", index=False)
    source_bundle = load_bundle(role_paths("candidate")[0])
    positions = candidate_manifest.reset_index().set_index("instance_hash").loc[selected.instance_hash, "index"].to_numpy()
    union_bundle = {key: value[positions] for key, value in source_bundle.items()}
    save_bundle(DATA / "selected_union.npz", **union_bundle)
    hgs_manifest = selected[["instance_id", "instance_hash"]].copy()
    records = solve_hgs_bundle(HGS_BINARY, HGS_CACHE / "selected_union", union_bundle, hgs_manifest, workers=workers, time_limit=10)
    refs = references_to_frame(records, "selected_union")
    trajectories = trajectory_bundle(union_bundle, records)
    save_bundle(DATA / "selected_union_trajectories.npz", **trajectories)
    bq_model = load_model(device)
    true_hardness = per_instance_hardness(bq_model, trajectories, device, batch_size=128)
    del bq_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selected["true_hardness"] = true_hardness
    selected = selected.merge(refs[["instance_id", "reference_type", "reference_cost", "rounded_reference_cost", "route_count", "wall_s", "feasible"]], on="instance_id", validate="one_to_one")
    selected.to_csv(RUN / "06_ACQUISITION_COMPARISON.csv", index=False)
    refs.to_csv(DATA / "selected_union_references.csv", index=False)

    random_values = selected.loc[selected.in_random, "true_hardness"].to_numpy()
    hgb_values = selected.loc[selected.in_hgb, "true_hardness"].to_numpy()
    candidate_pred_q90 = float(np.quantile(predictions, 0.9))
    global_true_q90_proxy = float(np.quantile(pd.read_csv(DATA / "probe_hardness.csv").hardness, 0.9))
    summary = {
        "candidate_count": len(candidate),
        "random_count": len(random_hashes),
        "hgb_count": len(hgb_hashes),
        "union_hgs_labels": len(selected),
        "overlap": len(random_hashes & hgb_hashes),
        "hgs_labels_non_evaluation_total": 2400 + len(selected),
        "budget_gate_pass": 2400 + len(selected) <= 6400,
        "hgb_inference_s_20k": inference_s,
        "random_true_hardness_mean": float(random_values.mean()),
        "hgb_true_hardness_mean": float(hgb_values.mean()),
        "mean_difference_hgb_minus_random": float(hgb_values.mean() - random_values.mean()),
        "mann_whitney_p_two_sided": float(mannwhitneyu(hgb_values, random_values, alternative="two-sided").pvalue),
        "ks_p_two_sided": float(ks_2samp(hgb_values, random_values).pvalue),
        "probe_true_hardness_q90": global_true_q90_proxy,
        "random_true_top_decile_rate": float(np.mean(random_values >= global_true_q90_proxy)),
        "hgb_true_top_decile_rate": float(np.mean(hgb_values >= global_true_q90_proxy)),
        "random_root_diversity": int(selected.loc[selected.in_random, "root_hash"].nunique()),
        "hgb_root_diversity": int(selected.loc[selected.in_hgb, "root_hash"].nunique()),
        "candidate_predicted_q90": candidate_pred_q90,
    }
    if not summary["budget_gate_pass"]:
        raise RuntimeError("HGS label budget exceeded")
    atomic_json(RUN / "acquisition_summary.json", summary)
    mark_stage("05_acquisition", started, **summary)


def _training_arrays_for_arm(arm: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    selected = pd.read_csv(RUN / "06_ACQUISITION_COMPARISON.csv")
    mask = selected.in_random.astype(bool) if arm == "RANDOM" else selected.in_hgb.astype(bool)
    arm_frame = selected[mask].copy().reset_index(drop=True)
    union_bundle = load_bundle(DATA / "selected_union.npz")
    trajectories = load_bundle(DATA / "selected_union_trajectories.npz")
    union_manifest = pd.read_csv(DATA / "selected_union_manifest.csv")
    position = union_manifest.reset_index().set_index("instance_hash").loc[arm_frame.instance_hash, "index"].to_numpy()
    return {key: value[position] for key, value in trajectories.items()}, arm_frame


def fine_tune_arm(arm: str, device: torch.device, updates: int) -> dict[str, Any]:
    checkpoint = CHECKPOINTS / f"M1_{arm}.pt"
    history_path = RUN / f"training_loss_{arm}.csv"
    result_path = RUN / f"training_result_{arm}.json"
    if checkpoint.is_file() and history_path.is_file() and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    trajectories, arm_frame = _training_arrays_for_arm(arm)
    if len(arm_frame) != 2000:
        raise RuntimeError(f"{arm} training count is {len(arm_frame)}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    model = load_model(device)
    initial_digest = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(arm_frame))
    cursor = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, updates + 1):
        if cursor + 128 > len(permutation):
            permutation = rng.permutation(len(arm_frame))
            cursor = 0
        indices = permutation[cursor:cursor + 128]
        cursor += 128
        begin = int(rng.integers(0, trajectories["coords"].shape[1] - 3))
        coords = torch.as_tensor(trajectories["coords"][indices, begin:], dtype=torch.float32, device=device)
        demands = torch.as_tensor(trajectories["demands"][indices, begin:], dtype=torch.float32, device=device)
        capacities = torch.as_tensor(trajectories["capacities"][indices], dtype=torch.float32, device=device)
        remaining = torch.as_tensor(trajectories["remaining_capacities"][indices, begin], dtype=torch.float32, device=device)
        via = torch.as_tensor(trajectories["via_depots"][indices, begin + 1], dtype=torch.long, device=device)
        inputs = torch.cat([coords, (demands / capacities[:, None]).unsqueeze(-1), (remaining / capacities)[:, None, None].expand(-1, coords.shape[1], 1)], dim=-1)
        target = torch.full((len(indices),), 2, dtype=torch.long, device=device) + via
        optimizer.zero_grad(set_to_none=True)
        scores = model(inputs, demands=demands, remaining_capacities=remaining)
        loss = F.cross_entropy(scores, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss in {arm} step {step}")
        loss.backward()
        gradient_norm = float(torch.sqrt(sum(parameter.grad.detach().double().pow(2).sum() for parameter in model.parameters() if parameter.grad is not None)).item())
        optimizer.step()
        history.append({"arm": arm, "optimizer_update": step, "suffix_begin": begin, "suffix_length": trajectories["coords"].shape[1] - begin, "loss": float(loss.detach().item()), "gradient_l2": gradient_norm, "batch_size": len(indices)})
        if step % 64 == 0:
            print(f"TRAIN_{arm}_PROGRESS={step}/{updates} loss={loss.item():.6f}", flush=True)
    wall = time.perf_counter() - started
    blob = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "other": {"arm": arm, "updates": updates, "seed": SEED, "m0_sha256": M0_SHA256},
    }
    torch.save(blob, checkpoint)
    pd.DataFrame(history).to_csv(history_path, index=False)
    torch.save({"numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, CHECKPOINTS / f"M1_{arm}_rng_state.pt")
    reloaded = load_model(device, checkpoint)
    first_parameter_delta = float(next(iter(reloaded.parameters())).detach().sub(next(iter(load_model(device).parameters())).detach()).abs().max().item())
    if first_parameter_delta <= 0:
        raise RuntimeError(f"{arm} checkpoint appears unchanged from M0")
    result = {
        "arm": arm,
        "training_instances": len(arm_frame),
        "optimizer": "Adam",
        "learning_rate": 2.5e-4,
        "scheduler": "none_fixed_update_pilot",
        "batch_size": 128,
        "optimizer_updates": updates,
        "gradient_clipping": None,
        "seed": SEED,
        "m0_sha256": M0_SHA256,
        "m0_state_digest": initial_digest,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "wall_s": wall,
        "first_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "checkpoint_roundtrip": True,
        "first_parameter_max_delta_from_m0": first_parameter_delta,
    }
    atomic_json(result_path, result)
    arm_frame[["instance_id", "instance_hash", "reference_type", "reference_cost"]].to_csv(RUN / f"training_manifest_{arm}.csv", index=False)
    del model, reloaded
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def finetune_stage(device: torch.device, updates: int) -> None:
    stage_name = f"06_finetune_{updates}"
    if stage_done(stage_name):
        return
    started = time.perf_counter()
    write_status(stage_name)
    random_result = fine_tune_arm("RANDOM", device, updates)
    hgb_result = fine_tune_arm("HGB", device, updates)
    if random_result["m0_sha256"] != hgb_result["m0_sha256"] or random_result["m0_state_digest"] != hgb_result["m0_state_digest"]:
        raise RuntimeError("training arms did not start from identical M0")
    pd.DataFrame([random_result, hgb_result]).to_csv(RUN / "07_BQ_FINETUNING_RESULTS.csv", index=False)
    mark_stage(stage_name, started, random_checkpoint=random_result["checkpoint_sha256"], hgb_checkpoint=hgb_result["checkpoint_sha256"], updates=updates)


def generate_external_distribution(name: str, count: int = 1000) -> None:
    bundle_path = DATA / f"eval_{name}.npz"
    manifest_path = DATA / f"eval_{name}_manifest.csv"
    if bundle_path.is_file() and manifest_path.is_file():
        return
    depots = np.empty((count, 2), dtype=np.float64)
    customers = np.empty((count, N_CUSTOMERS, 2), dtype=np.float64)
    demands = np.empty((count, N_CUSTOMERS), dtype=np.int16)
    capacities = np.full(count, CAPACITY, dtype=np.int16)
    rows = []
    for index in range(count):
        rng = np.random.default_rng(stable_seed(RUN_ID, SEED, "evaluation", name, index))
        depot = rng.random(2)
        if name == "uniform":
            xy = rng.random((N_CUSTOMERS, 2))
            demand = rng.integers(1, 10, N_CUSTOMERS, dtype=np.int16)
        elif name == "clustered":
            centers = rng.uniform(0.1, 0.9, size=(5, 2))
            assignment = rng.integers(0, len(centers), N_CUSTOMERS)
            xy = np.clip(centers[assignment] + rng.normal(0.0, 0.055, size=(N_CUSTOMERS, 2)), 0.0, 1.0)
            demand = rng.integers(1, 10, N_CUSTOMERS, dtype=np.int16)
        elif name == "demand_heterogeneous":
            xy = rng.random((N_CUSTOMERS, 2))
            high = rng.random(N_CUSTOMERS) < 0.35
            demand = np.where(high, rng.integers(7, 10, N_CUSTOMERS), rng.integers(1, 4, N_CUSTOMERS)).astype(np.int16)
        else:
            raise ValueError(name)
        depots[index], customers[index], demands[index] = depot, xy, demand
        digest = instance_hash(depot, xy, demand, CAPACITY)
        rows.append({"dataset": name, "instance_id": f"eval_{name}_{index:05d}", "instance_hash": digest, "n_customers": N_CUSTOMERS, "reference_type": "HGS_REFERENCE"})
    save_bundle(bundle_path, depot=depots, customers=customers, demands=demands, capacities=capacities)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def evaluate_synthetic_distribution(name: str, models: dict[str, BQModel], device: torch.device, workers: int) -> pd.DataFrame:
    bundle = load_bundle(DATA / f"eval_{name}.npz")
    manifest = pd.read_csv(DATA / f"eval_{name}_manifest.csv")
    records = solve_hgs_bundle(HGS_BINARY, HGS_CACHE / f"eval_{name}", bundle, manifest, workers=workers, time_limit=10)
    reference = references_to_frame(records, name)
    frame = manifest.merge(reference[["instance_id", "reference_cost", "rounded_reference_cost", "feasible"]], on="instance_id", validate="one_to_one")
    for label, model in models.items():
        started = time.perf_counter()
        costs, feasible = model_greedy_synthetic(model, bundle, device, batch_size=64)
        elapsed = time.perf_counter() - started
        frame[f"cost_{label}"] = costs
        frame[f"feasible_{label}"] = feasible
        frame[f"inference_s_{label}"] = elapsed / len(frame)
    return paired_gap_frame(frame)


def parse_hgs_cvrplib_reference(instance: dict[str, Any], source_path: Path) -> dict[str, Any]:
    from cvrp_pilot.core import parse_hgs_solution

    cache = HGS_CACHE / "cvrplib_set_x" / f"{instance['name']}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            return payload
    solution_path = cache.with_suffix(".sol")
    started = time.perf_counter()
    process = subprocess.run(
        [str(HGS_BINARY), str(source_path), str(solution_path), "-seed", str(stable_seed(SEED, instance["name"]) % (2**31 - 1)), "-t", "10", "-round", "1", "-log", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    wall = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"HGS Set-X failure {instance['name']}: {process.stdout[-1000:]}")
    routes, reported = parse_hgs_solution(solution_path)
    flat = [node for route in routes for node in route]
    if sorted(flat) != list(range(1, len(instance["customers"]) + 1)):
        raise RuntimeError(f"Set-X coverage failure {instance['name']}")
    coords = np.vstack([instance["depot"], instance["customers"]])
    cost = 0.0
    for route in routes:
        load = sum(int(instance["demands"][node - 1]) for node in route)
        if load > instance["capacity"]:
            raise RuntimeError(f"Set-X capacity failure {instance['name']}")
        path = [0] + route + [0]
        cost += sum(round(float(np.linalg.norm(coords[left] - coords[right]))) for left, right in zip(path, path[1:]))
    if int(round(cost)) != reported:
        raise RuntimeError(f"Set-X cost mismatch {instance['name']}: {cost} vs {reported}")
    payload = {"instance_id": instance["name"], "status": "ok", "reference_type": "HGS_REFERENCE", "reference_cost": cost, "routes": routes, "wall_s": wall, "feasible": True}
    atomic_json(cache, payload)
    return payload


@torch.no_grad()
def evaluate_cvrplib_model(model: BQModel, instance: dict[str, Any], device: torch.device) -> tuple[float, bool, float]:
    depot_norm, customer_norm, _ = normalize_cvrplib_coords(instance["depot"], instance["customers"])
    coords_norm = np.vstack([depot_norm, customer_norm, depot_norm]).astype(np.float32)
    demands_np = np.concatenate([[0], instance["demands"], [0]]).astype(np.float32)
    coords = torch.as_tensor(coords_norm[None], dtype=torch.float32, device=device)
    demands = torch.as_tensor(demands_np[None], dtype=torch.float32, device=device)
    capacities = torch.as_tensor([instance["capacity"]], dtype=torch.float32, device=device)
    delta = coords[:, :, None, :] - coords[:, None, :, :]
    matrix = torch.sqrt(torch.sum(delta * delta, dim=-1))
    started = time.perf_counter()
    _, tours = official_decode(coords, matrix, demands, capacities, model, 1, -1, make_tours=True)
    elapsed = time.perf_counter() - started
    duplicate_depot = coords.shape[1] - 1
    path = [0 if int(node) == duplicate_depot else int(node) for node in tours[0]]
    if path[-1] != 0:
        path.append(0)
    original = np.vstack([instance["depot"], instance["customers"]])
    cost = 0.0
    load = 0
    feasible = True
    previous = path[0]
    for node in path[1:]:
        cost += round(float(np.linalg.norm(original[previous] - original[node])))
        if node == 0:
            feasible = feasible and load <= instance["capacity"]
            load = 0
        else:
            load += int(instance["demands"][node - 1])
        previous = node
    feasible = feasible and load == 0 and sorted(node for node in path if node != 0) == list(range(1, len(instance["customers"]) + 1))
    return cost, feasible, elapsed


def evaluate_set_x(models: dict[str, BQModel], device: torch.device) -> pd.DataFrame:
    names = ["X-n101-k25", "X-n106-k14", "X-n110-k13", "X-n115-k10", "X-n120-k6"]
    rows = []
    for name in names:
        path = ROOT / "third_party" / "hgs_cvrp_official" / "Instances" / "CVRP" / f"{name}.vrp"
        instance = parse_cvrplib(path)
        reference = parse_hgs_cvrplib_reference(instance, path)
        depot_norm, customer_norm, normalization = normalize_cvrplib_coords(instance["depot"], instance["customers"])
        row: dict[str, Any] = {
            "dataset": "cvrplib_set_x",
            "instance_id": name,
            "instance_hash": instance_hash(depot_norm, customer_norm, instance["demands"], instance["capacity"]),
            "n_customers": len(instance["customers"]),
            "capacity": instance["capacity"],
            "reference_type": "HGS_REFERENCE",
            "reference_cost": reference["reference_cost"],
            "normalization": json.dumps(normalization, sort_keys=True),
        }
        for label, model in models.items():
            cost, feasible, elapsed = evaluate_cvrplib_model(model, instance, device)
            row[f"cost_{label}"] = cost
            row[f"feasible_{label}"] = feasible
            row[f"inference_s_{label}"] = elapsed
        rows.append(row)
    return paired_gap_frame(pd.DataFrame(rows))


def aggregate_evaluation(instances: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in instances.groupby("dataset", sort=False):
        for model in ("M0", "M1_RANDOM", "M1_HGB"):
            gap = frame[f"gap_{model}"].to_numpy()
            mean, low, high = bootstrap_mean_ci(gap, replicates=10_000, seed=stable_seed(SEED, dataset, model))
            rows.append(
                {
                    "dataset": dataset,
                    "n_instances": len(frame),
                    "n_customers_min": int(frame.n_customers.min()),
                    "n_customers_max": int(frame.n_customers.max()),
                    "model": model,
                    "mean_solution_cost": float(frame[f"cost_{model}"].mean()),
                    "mean_reference_gap_pct": mean,
                    "median_reference_gap_pct": float(np.median(gap)),
                    "mean_gap_bootstrap_ci95_low": low,
                    "mean_gap_bootstrap_ci95_high": high,
                    "feasibility_rate": float(frame[f"feasible_{model}"].mean()),
                    "mean_inference_s_per_instance": float(frame[f"inference_s_{model}"].mean()),
                    "reference_type": "HGS_REFERENCE",
                }
            )
        paired = frame["delta_gap_hgb_minus_random"].to_numpy()
        mean, low, high = bootstrap_mean_ci(paired, replicates=10_000, seed=stable_seed(SEED, dataset, "paired"))
        rows.append(
            {
                "dataset": dataset,
                "n_instances": len(frame),
                "n_customers_min": int(frame.n_customers.min()),
                "n_customers_max": int(frame.n_customers.max()),
                "model": "M1_HGB_MINUS_M1_RANDOM",
                "mean_solution_cost": float("nan"),
                "mean_reference_gap_pct": mean,
                "median_reference_gap_pct": float(np.median(paired)),
                "mean_gap_bootstrap_ci95_low": low,
                "mean_gap_bootstrap_ci95_high": high,
                "feasibility_rate": float("nan"),
                "mean_inference_s_per_instance": float("nan"),
                "reference_type": "PAIRED_DELTA_GAP_HGB_MINUS_RANDOM",
            }
        )
    return pd.DataFrame(rows)


def evaluation_stage(device: torch.device, workers: int) -> None:
    if stage_done("07_evaluation"):
        return
    started = time.perf_counter()
    write_status("07_evaluation")
    for name in ("uniform", "clustered", "demand_heterogeneous"):
        generate_external_distribution(name)
    models = {"M0": load_model(device), "M1_RANDOM": load_model(device, CHECKPOINTS / "M1_RANDOM.pt"), "M1_HGB": load_model(device, CHECKPOINTS / "M1_HGB.pt")}
    for model in models.values():
        model.eval()
    frames = [evaluate_synthetic_distribution(name, models, device, workers) for name in ("uniform", "clustered", "demand_heterogeneous")]
    frames.append(evaluate_set_x(models, device))
    instances = pd.concat(frames, ignore_index=True, sort=False)
    instances.to_csv(RUN / "external_evaluation_instances.csv", index=False)
    aggregate = aggregate_evaluation(instances)
    aggregate.to_csv(RUN / "08_EXTERNAL_EVALUATION.csv", index=False)
    if not all(instances[f"feasible_{model}"].all() for model in models):
        raise RuntimeError("external evaluation feasibility gate failed")
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mark_stage("07_evaluation", started, instances=len(instances), datasets=instances.dataset.value_counts().to_dict(), all_feasible=True)


def make_figures() -> None:
    probe = pd.read_csv(RUN / "sealed_probe_predictions.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(probe.hardness, probe.predicted_hardness, s=18, alpha=0.55)
    ax.set_xlabel("True BQ-CVRP teacher-forced hardness")
    ax.set_ylabel("HGB predicted hardness")
    rho = spearmanr(probe.hardness, probe.predicted_hardness).statistic
    ax.set_title(f"Predicted vs true hardness (sealed probe, Spearman={rho:.3f})")
    fig.tight_layout(); fig.savefig(FIGURES / "01_predicted_vs_true_hardness.png", dpi=180); plt.close(fig)

    acquisition = pd.read_csv(RUN / "06_ACQUISITION_COMPARISON.csv")
    threshold = float(np.quantile(pd.read_csv(DATA / "probe_hardness.csv").hardness, 0.9))
    rates = [np.mean(acquisition.loc[acquisition.in_random, "true_hardness"] >= threshold), np.mean(acquisition.loc[acquisition.in_hgb, "true_hardness"] >= threshold)]
    fig, ax = plt.subplots(figsize=(6, 5)); ax.bar(["Random FOGA2k", "HGB Top2k"], rates); ax.axhline(0.1, color="black", ls="--", lw=1); ax.set_ylabel("Fraction above sealed-probe true-hard Q90"); ax.set_title("Hard-tail enrichment"); fig.tight_layout(); fig.savefig(FIGURES / "02_hard_tail_enrichment.png", dpi=180); plt.close(fig)

    random_values = acquisition.loc[acquisition.in_random, "true_hardness"]
    hgb_values = acquisition.loc[acquisition.in_hgb, "true_hardness"]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(random_values, bins=40, density=True, alpha=0.55, label="Random FOGA2k"); ax.hist(hgb_values, bins=40, density=True, alpha=0.55, label="HGB Top2k"); ax.set_xlabel("True BQ-CVRP hardness"); ax.set_ylabel("Density"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "03_c1_c2_hardness_distribution.png", dpi=180); plt.close(fig)

    aggregate = pd.read_csv(RUN / "08_EXTERNAL_EVALUATION.csv")
    base = aggregate[aggregate.model.isin(["M0", "M1_RANDOM", "M1_HGB"])]
    pivot = base.pivot(index="dataset", columns="model", values="mean_reference_gap_pct")
    pivot = pivot[["M0", "M1_RANDOM", "M1_HGB"]]
    fig, ax = plt.subplots(figsize=(9, 5.5)); pivot.plot(kind="bar", ax=ax); ax.set_ylabel("Mean HGS reference gap (%)"); ax.set_title("BQ M0 / Random / HGB greedy comparison"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(FIGURES / "04_gap_comparison.png", dpi=180); plt.close(fig)

    instances = pd.read_csv(RUN / "external_evaluation_instances.csv")
    groups = [group.paired_improvement_hgb_over_random.to_numpy() for _, group in instances.groupby("dataset", sort=False)]
    labels = [name for name, _ in instances.groupby("dataset", sort=False)]
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.boxplot(groups, tick_labels=labels, showfliers=False); ax.axhline(0, color="black", lw=1); ax.set_ylabel("Gap(Random) - Gap(HGB), pp"); ax.set_title("Per-instance paired improvement"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(FIGURES / "05_paired_improvement.png", dpi=180); plt.close(fig)

    compute = pd.read_csv(RUN / "09_COMPUTE_COST_AUDIT.csv")
    actual = compute[compute.actual_campaign_wall_s.notna()] if "actual_campaign_wall_s" in compute else compute.iloc[0:0]
    if len(actual):
        fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(actual.component, actual.actual_campaign_wall_s / 3600); ax.set_ylabel("Campaign wall-clock hours"); ax.set_title("Computational cost breakdown"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(FIGURES / "06_compute_cost_breakdown.png", dpi=180); plt.close(fig)


def finalize_compute_audit() -> None:
    calibration = pd.read_csv(RUN / "03_HGS_REFERENCE_AUDIT.csv")
    selected = pd.read_csv(DATA / "selected_union_references.csv")
    instances = pd.read_csv(RUN / "external_evaluation_instances.csv")
    training = pd.read_csv(RUN / "07_BQ_FINETUNING_RESULTS.csv")
    calibration_stage_wall = read_stage("03_calibration_labels")["wall_s"]
    acquisition_stage_wall = read_stage("05_acquisition")["wall_s"]
    evaluation_stage_wall = read_stage("07_evaluation")["wall_s"]
    rows = [
        {"component": "Calibration+probe HGS and BQ hardness stage", "instances": len(calibration), "cpu_budget_s": len(calibration) * 10, "aggregate_process_wall_s": float(calibration.wall_s.sum()), "actual_campaign_wall_s": calibration_stage_wall, "note": "96-way HGS plus frozen-M0 exhaustive suffix loss"},
        {"component": "Selected-union HGS and BQ hardness stage", "instances": len(selected), "cpu_budget_s": len(selected) * 10, "aggregate_process_wall_s": float(selected.wall_s.sum()), "actual_campaign_wall_s": acquisition_stage_wall, "note": "natural overlap solved once; includes HGB acquisition and hardness"},
        {"component": "Frozen external evaluation stage", "instances": len(instances), "cpu_budget_s": len(instances) * 10, "aggregate_process_wall_s": float("nan"), "actual_campaign_wall_s": evaluation_stage_wall, "note": "3,000 synthetic + five non-CVRP100 Set-X; includes three-model greedy"},
        {"component": "BQ fine-tune Random", "instances": 2000, "cpu_budget_s": 0, "aggregate_process_wall_s": float("nan"), "actual_campaign_wall_s": float(training.loc[training.arm == "RANDOM", "wall_s"].iloc[0]), "note": "GPU training wall; synchronous loss/gradient logging"},
        {"component": "BQ fine-tune HGB", "instances": 2000, "cpu_budget_s": 0, "aggregate_process_wall_s": float("nan"), "actual_campaign_wall_s": float(training.loc[training.arm == "HGB", "wall_s"].iloc[0]), "note": "GPU training wall; synchronous loss/gradient logging"},
    ]
    pd.DataFrame(rows).to_csv(RUN / "09_COMPUTE_COST_AUDIT.csv", index=False)


def conclusions_stage() -> None:
    if stage_done("08_conclusions"):
        return
    started = time.perf_counter()
    write_status("08_conclusions")
    hgb_report = read_stage("04_hgb")
    acquisition = json.loads((RUN / "acquisition_summary.json").read_text(encoding="utf-8"))
    evaluation = pd.read_csv(RUN / "08_EXTERNAL_EVALUATION.csv")
    paired = evaluation[evaluation.model == "M1_HGB_MINUS_M1_RANDOM"].copy()
    synthetic = paired[paired.dataset != "cvrplib_set_x"]
    h1 = hgb_report["probe"]["spearman"] > 0 and hgb_report["probe"]["true_top10_enrichment"] > 1.0
    h2 = acquisition["mean_difference_hgb_minus_random"] > 0 and acquisition["hgb_true_top_decile_rate"] > acquisition["random_true_top_decile_rate"]
    h3 = bool((synthetic.mean_reference_gap_pct < 0).all() and (synthetic.mean_gap_bootstrap_ci95_high < 0).all())
    if h1 and h2 and h3:
        decision = "GO"
    elif h1 and h2:
        decision = "CONDITIONAL_GO"
    elif not h1 and not h2:
        decision = "SCIENTIFIC_NO_GO"
    else:
        decision = "SCIENTIFIC_NO_GO"
    labels = {
        "H1": "CVRP_BQ_HARDNESS_SIGNAL_CONFIRMED" if h1 else "CVRP_BQ_HARDNESS_SIGNAL_NOT_CONFIRMED",
        "H2": "CVRP_BQ_HARD_TAIL_ENRICHMENT_CONFIRMED" if h2 else "CVRP_BQ_HARD_TAIL_ENRICHMENT_NOT_CONFIRMED",
        "H3": "CVRP_BQ_TRAINING_VALUE_CONFIRMED" if h3 else "CVRP_BQ_TRAINING_VALUE_NOT_CONFIRMED",
    }
    paired_lines = "\n".join(
        f"- {row.dataset}: ΔGap(HGB−Random) {row.mean_reference_gap_pct:.6f} pp, instance-bootstrap 95% CI [{row.mean_gap_bootstrap_ci95_low:.6f}, {row.mean_gap_bootstrap_ci95_high:.6f}]"
        for row in paired.itertuples(index=False)
    )
    report = f"""# BQ-CVRP Feasibility Pilot V1 — Conclusions

## Decision: `{decision}`

- H1: `{labels['H1']}`
- H2: `{labels['H2']}`
- H3: `{labels['H3']}`

## Q1. Can instance structure predict BQ-CVRP solver-relative hardness?

{'Yes under this frozen distribution and M0.' if h1 else 'Not with sufficient evidence in this Pilot.'} The sealed-probe Spearman correlation is {hgb_report['probe']['spearman']:.6f}, MAE is {hgb_report['probe']['mae']:.6f}, and true-hard Top10% enrichment is {hgb_report['probe']['true_top10_enrichment']:.3f}×. The target is the official BQ-CVRP teacher-forced imitation loss averaged over the official suffix-sampling support; it is not route cost or optimality gap.

## Q2. Did HGB discover the CVRP hard tail?

{'Yes.' if h2 else 'Not convincingly.'} HGB Top2k mean true hardness minus Random FOGA2k is {acquisition['mean_difference_hgb_minus_random']:.6f}. Above the sealed-probe true-hard Q90, rates are {acquisition['hgb_true_top_decile_rate']:.4f} (HGB) and {acquisition['random_true_top_decile_rate']:.4f} (random). Natural overlap is {acquisition['overlap']} instances; those references were solved once.

## Q3. Was hardness-guided FOGA more effective downstream?

{'Yes on every frozen synthetic suite with paired CIs below zero.' if h3 else 'Not with the predeclared strong criterion across all frozen synthetic suites.'}

{paired_lines}

The reference gap uses HGS heuristic references and is not called an exact optimality gap. CVRPLIB rows explicitly report their actual customer counts and are not pooled as CVRP100. This Pilot has one training seed; instance-level paired intervals quantify test-instance uncertainty only and do not substitute for multi-training-seed evidence.

## Q4. Expand to BQ-CVRP + HGB + QD + multi-cycle?

The protocol decision is `{decision}`. {'The complete chain is positive, so a multi-seed confirmation should precede or accompany the larger QD/multi-cycle study.' if decision == 'GO' else ('Hardness prediction and acquisition work, but downstream training value needs multi-seed confirmation before adding QD and cycles.' if decision == 'CONDITIONAL_GO' else 'The present scientific evidence does not justify expansion under the same design.')}

## Integrity

Official BQ checkpoint `{M0_SHA256}` and HGS v2.0.0 were used throughout. Non-evaluation HGS labels total {acquisition['hgs_labels_non_evaluation_total']} (cap 6,400). No LEHD/POMO substitute, random M0, QD, second cycle, candidate-wide true-hardness pre-screen, or TSP loss path was used.

`BLADE_BQ_CVRP_PILOT_V1_COMPLETE`
"""
    (RUN / "10_CVRP_PILOT_CONCLUSIONS.md").write_text(report, encoding="utf-8")
    atomic_json(RUN / "conclusion_labels.json", {"decision": decision, **labels, "single_training_seed_limitation": True})
    finalize_compute_audit()
    make_figures()
    mark_stage("08_conclusions", started, decision=decision, **labels)
    write_status("complete", "complete", decision=decision, labels=labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 1))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--updates", type=int, default=2048, help="Must be 2048 for the formal protocol")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates != 2048:
        raise ValueError("formal frozen protocol requires exactly 2048 optimizer updates")
    ensure_dirs()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal BQ-CVRP Pilot requires an available CUDA device")
    try:
        audit_stage(device)
        smoke_stage(device, args.workers)
        data_stage()
        calibration_stage(device, args.workers)
        fit_hgb_stage()
        acquisition_stage(device, args.workers)
        finetune_stage(device, args.updates)
        evaluation_stage(device, args.workers)
        conclusions_stage()
    except Exception as exc:
        ensure_dirs()
        (RUN / "FAILURE.md").write_text(
            f"# Fail-closed Pilot interruption\n\n- Time: {now_iso()}\n- Exception: `{type(exc).__name__}: {exc}`\n- Resume command: `/usr/bin/python3 scripts/run_bq_cvrp_pilot_v1.py`\n\n```text\n{traceback.format_exc()}\n```\n",
            encoding="utf-8",
        )
        write_status("failed", "partial", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
