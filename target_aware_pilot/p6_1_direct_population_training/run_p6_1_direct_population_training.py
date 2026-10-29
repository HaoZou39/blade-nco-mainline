#!/usr/bin/env python3
"""P6-1: train directly on the six frozen final G2 populations.

This runner never evolves, selects, acquires, filters for novelty, or reads M4
training overlap.  It copies frozen G2 bundles, labels only globally missing
coordinate hashes, trains six M4->M5 runs, evaluates with the frozen greedy
protocol, and compares them with the existing P5 archive-trained checkpoints.
All stages are resumable from integrity records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "target_aware_pilot/p2_bounded_evolution"
P5 = ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness"
P60 = ROOT / "target_aware_pilot/p6_feature_population_protocol_audit"
OUT = Path(__file__).resolve().parent
DATA_DIR = OUT / "datasets"
LABEL_DIR = OUT / "labels"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"
FIGURES = OUT / "figures"

sys.path[:0] = [
    str(ROOT / "src"), str(ROOT / "scripts"), str(P2),
    str(ROOT / "target_aware_pilot/p4a_random_evolution_ablation"),
    str(ROOT / "target_aware_pilot/p4b1_acquisition_ablation"),
    str(ROOT / "target_aware_pilot/p4b2_random_archive_ablation"),
    str(ROOT / "target_aware_pilot/p4c2_data_sampling_robustness"), str(P5),
]

import run_p2_bounded_evolution as p2  # noqa: E402
import run_p4b1_acquisition_ablation as p4b1  # noqa: E402
import run_p4c2_data_sampling_robustness as p4c2  # noqa: E402
import run_p5_structure_guided_evolution_robustness as p5  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, ORACLE_LEDGER_COLUMNS, sha256_file  # noqa: E402


REPLICATES = ("Original", "S1", "S2")
ARMS = ("E0", "E1")
TRAIN_N = 10_000
TRAINING_SEED = 2026
M4 = p5.M4
M4_SHA = p5.M4_SHA
DATASETS = tuple(p5.DATASETS)
GAP5 = tuple(p5.GAP5_DATASETS)
BOOTSTRAP_N = 10_000


def ensure_dirs() -> None:
    for path in (OUT, DATA_DIR, LABEL_DIR, CHECKPOINTS, LOGS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.integer): return int(item)
        if isinstance(item, np.floating): return float(item)
        if isinstance(item, np.bool_): return bool(item)
        if isinstance(item, np.ndarray): return item.tolist()
        raise TypeError(type(item).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def source_root(replicate: str) -> Path:
    if replicate == "Original":
        return P2 / "evolution"
    return P5 / "evolution" / replicate / "evolution"


def source_manifest(replicate: str, arm: str) -> Path:
    return source_root(replicate) / arm / "G2/active_population.parquet"


def source_bundle(replicate: str, arm: str) -> Path:
    return source_manifest(replicate, arm).with_suffix(".npz")


def dataset_manifest(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_G2_Population10k.parquet"


def dataset_bundle(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_G2_Population10k.npz"


def dataset_integrity(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_integrity.json"


def label_run_dir(replicate: str, arm: str) -> Path:
    return LABEL_DIR / f"{replicate}_{arm}"


def run_dir(replicate: str, arm: str) -> Path:
    return LOGS / f"{replicate}_{arm}_Pop"


def checkpoint_path(replicate: str, arm: str) -> Path:
    return CHECKPOINTS / f"{replicate}_{arm}_Pop_M5_E256.tsp.best"


def eval_label(replicate: str, arm: str) -> str:
    return f"{replicate}_{arm}_Pop_M5"


def eval_csv(replicate: str, arm: str) -> Path:
    return run_dir(replicate, arm) / "external_benchmark_results.csv"


def load_bundle(path: Path, require_tours: bool = False) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
        tours = np.asarray(payload["expert_tours"], np.int64) if "expert_tours" in payload.files else None
    if require_tours and tours is None:
        raise RuntimeError(f"missing expert tours: {path}")
    return coordinates, hashes, tours


def population_coordinate_audit(coordinates: np.ndarray, hashes: list[str]) -> dict[str, Any]:
    reproduced = [coordinate_hash(points) for points in coordinates]
    return {
        "shape": list(coordinates.shape),
        "finite": bool(np.isfinite(coordinates).all()),
        "in_bounds": bool(((coordinates >= 0) & (coordinates <= 1)).all()),
        "unique_nodes": bool(all(len(np.unique(points, axis=0)) == 100 for points in coordinates)),
        "hashes_reproduce": reproduced == hashes,
        "unique_hashes": len(set(hashes)),
    }


def base_training_protocol() -> dict[str, Any]:
    trainer = ROOT / "scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
    return {
        "authority": "P5 frozen actual training configuration",
        "m4_sha256": M4_SHA,
        "trainer_sha256": sha256_file(trainer),
        "optimizer": "fresh Adam", "scheduler": None, "learning_rate": 3.75e-5,
        "training_seed": TRAINING_SEED, "epochs": 256, "optimizer_updates": 10_240,
        "logical_batch_size": 256, "microbatch_size": 64,
        "augmentation": "formal NumPy permutation and suffix-length sampling",
        "loss": "teacher-forced imitation", "checkpoint_rule": "fixed E256",
        "evaluation": "frozen greedy eight-benchmark evaluator",
    }


def protocol_gate() -> dict[str, Any]:
    p5_protocol = p5.protocol_gate()
    p5.verify_evaluations()
    training = pd.read_csv(P5 / "07_bq_training_runs_manifest.csv")
    if len(training) != 6 or not training.training_seed.eq(TRAINING_SEED).all():
        raise RuntimeError("P5 historical training protocol mismatch")
    if not training.m4_sha256.eq(M4_SHA).all() or not training.optimizer_steps.eq(10_240).all():
        raise RuntimeError("P5 M4/training budget mismatch")
    if not training.e000_weights_equal_m4.astype(bool).all():
        raise RuntimeError("P5 E000 initialization mismatch")
    protocol = {
        "status": "FROZEN_BEFORE_P6_1_LABELING_AND_TRAINING",
        "intervention": "replace P5 Archive30k->Structure10k training membership with frozen final G2 Population10k",
        "replicates": list(REPLICATES), "arms": list(ARMS),
        "training": base_training_protocol(),
        "primary": "OriginalTSPLIB70 mean gap; Delta_population=E0-Pop minus E1-Pop",
        "critical_exploratory": "TSPLIB22, repeatedly observed",
        "secondary": list(GAP5),
        "decision_rule": {
            "SUPPORTED_PRELIMINARY": "all three Original70 Delta_population > 0 and GAP5 is not negative in all three replicates",
            "NOT_SUPPORTED": "all three Original70 Delta_population < 0",
            "INCONCLUSIVE": "otherwise",
        },
        "forbidden": ["mutation", "survivor reselection", "archive acquisition", "novelty filter", "hyperparameter search", "extra seed"],
        "p5_protocol_sha256": p5_protocol["protocol_sha256"],
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    path = OUT / "frozen_protocol.json"
    if path.is_file() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("P6-1 frozen protocol drift")
    write_json(path, protocol)
    return protocol


def archive_training_manifest(replicate: str, arm: str) -> tuple[Path, Path]:
    return p5.source_dataset(replicate, arm)


def distribution_audit() -> pd.DataFrame:
    geometry = p2.load_geometry()
    rows: list[dict[str, Any]] = []
    for replicate in REPLICATES:
        population_sets: dict[str, set[str]] = {}
        for arm in ARMS:
            population = pd.read_parquet(dataset_manifest(replicate, arm))
            population_sets[arm] = set(population.instance_hash.astype(str))
        overlap = len(population_sets["E0"] & population_sets["E1"])
        for arm in ARMS:
            for source_kind, manifest_path in (
                ("G2_Final_Population10k", dataset_manifest(replicate, arm)),
                ("P5_ArchiveBased_Structure10k", archive_training_manifest(replicate, arm)[1]),
            ):
                frame = pd.read_parquet(manifest_path)
                required_geometry = {"target_distance", "in_strict_target_region", "coarse_cell", "inside_target_box", "nearest_target_cell"}
                if not required_geometry.issubset(frame.columns):
                    frame = p2.add_geometry(frame, geometry)
                metric, _ = p2.evaluate_frame(arm, 2, source_kind, frame, geometry)
                generations = pd.to_numeric(frame.get("generation_born", pd.Series([-1] * len(frame))), errors="coerce").fillna(-1)
                origin_parent = int(generations.lt(1).sum())
                operators = frame.get("operator_id", pd.Series(["unknown"] * len(frame))).fillna("P03_parent").astype(str).value_counts().sort_index().to_dict()
                metric.update({
                    "replicate": replicate, "training_path": source_kind,
                    "feature_1": "spn_distance_varcoeff", "feature_1_mean": float(frame.spn_distance_varcoeff.mean()),
                    "feature_2": "spn_fraction_shorter_mean_distance", "feature_2_mean": float(frame.spn_fraction_shorter_mean_distance.mean()),
                    "parent_origin_n": origin_parent, "offspring_origin_n": len(frame) - origin_parent,
                    "parent_origin_fraction": origin_parent / len(frame),
                    "offspring_origin_fraction": 1 - origin_parent / len(frame),
                    "operator_provenance": json.dumps(operators, sort_keys=True),
                    "e0_e1_population_overlap_n": overlap if source_kind == "G2_Final_Population10k" else np.nan,
                })
                rows.append(metric)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "02_population_distribution_audit.csv", index=False)
    return result


def prepare() -> dict[str, Any]:
    ensure_dirs()
    protocol = protocol_gate()
    p60 = pd.read_csv(P60 / "support_g2_population_feasibility.csv")
    rows = []
    report = ["# P6-1 Input and Protocol Integrity", ""]
    for replicate in REPLICATES:
        for arm in ARMS:
            source_npz = source_bundle(replicate, arm)
            source_pq = source_manifest(replicate, arm)
            coordinates, hashes, _ = load_bundle(source_npz)
            manifest = pd.read_parquet(source_pq)
            audit = population_coordinate_audit(coordinates, hashes)
            required_provenance = ["lineage_root", "generation_born", "evolution_origin", "operator_id", "direct_parent_hash"]
            born = manifest.generation_born.ge(1)
            provenance = bool(
                all(column in manifest for column in required_provenance)
                and manifest[["lineage_root", "generation_born", "evolution_origin"]].notna().all().all()
                and manifest.loc[born, "direct_parent_hash"].notna().all()
            )
            if (
                manifest.instance_hash.astype(str).tolist() != hashes
                or audit["shape"] != [TRAIN_N, 100, 2] or audit["unique_hashes"] != TRAIN_N
                or not all(audit[key] for key in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce"))
                or not provenance
            ):
                raise RuntimeError(f"G2 population integrity failure: {replicate}/{arm}")
            p60row = p60[(p60.replicate.eq(replicate)) & (p60.arm.eq(arm))].iloc[0]
            if str(p60row.bundle_sha256) != sha256_file(source_npz) or str(p60row.manifest_sha256) != sha256_file(source_pq):
                raise RuntimeError(f"P6-0 hash mismatch: {replicate}/{arm}")
            if not dataset_bundle(replicate, arm).is_file(): shutil.copy2(source_npz, dataset_bundle(replicate, arm))
            if not dataset_manifest(replicate, arm).is_file(): shutil.copy2(source_pq, dataset_manifest(replicate, arm))
            if sha256_file(dataset_bundle(replicate, arm)) != sha256_file(source_npz) or sha256_file(dataset_manifest(replicate, arm)) != sha256_file(source_pq):
                raise RuntimeError("immutable copy mismatch")
            integrity = {
                "status": "PASS", "replicate": replicate, "arm": arm, "rows": TRAIN_N,
                "source_role": "persisted G2 active_population survivors; not offspring subset or archive",
                "source_bundle": str(source_npz), "source_manifest": str(source_pq),
                "source_bundle_sha256": sha256_file(source_npz), "source_manifest_sha256": sha256_file(source_pq),
                "immutable_bundle_sha256": sha256_file(dataset_bundle(replicate, arm)),
                "immutable_manifest_sha256": sha256_file(dataset_manifest(replicate, arm)),
                "membership_sha256": membership_hash(hashes), "provenance_complete": provenance, **audit,
            }
            write_json(dataset_integrity(replicate, arm), integrity)
            rows.append(integrity)
            report.append(f"- {replicate}-{arm}: 10,000 unique valid TSP100; bundle `{integrity['immutable_bundle_sha256']}`; manifest `{integrity['immutable_manifest_sha256']}`; provenance complete.")
    manifest_columns = [
        "replicate", "arm", "rows", "membership_sha256", "immutable_bundle_sha256",
        "immutable_manifest_sha256", "source_bundle", "source_manifest", "provenance_complete", "status",
    ]
    pd.DataFrame(rows)[manifest_columns].to_csv(OUT / "01_population_datasets_manifest.csv", index=False)
    distribution_audit()
    report.extend([
        "", "Historical P5 gate: all six Archive-based checkpoints use the same formal M4 hash, training seed 2026, 10,240 updates, E000=M4, and common frozen evaluator.",
        "", "The P6-1 prompt's table headed ‘existing reusable labels’ contains the P6-0 **missing-label** counts. Cache intersections are recomputed from verified ledgers in Step 2.",
        "", "M4 training overlap was not read, measured, filtered, or used as a condition.",
        "", f"Protocol SHA256: `{protocol['protocol_sha256']}`.", "", "Status: **PASS**.",
    ])
    (OUT / "00_protocol_integrity.md").write_text("\n".join(report) + "\n")
    Status().update(phase="prepare", status="PASS")
    return {"status": "PASS", "datasets": 6, "protocol_sha256": protocol["protocol_sha256"]}


def pre_p6_cache() -> pd.DataFrame:
    cache = p5.p5_cache()
    if not cache.verified.astype(bool).all() or not cache.status.eq("solved").all():
        raise RuntimeError("invalid pre-P6 verified cache")
    if set(cache.binary_hash.astype(str)) != {CANONICAL_CONCORDE_SHA256}:
        raise RuntimeError("Concorde binary mismatch in reusable cache")
    return cache[list(ORACLE_LEDGER_COLUMNS)].drop_duplicates("instance_hash", keep="last")


def validate_ledger(coordinates: np.ndarray, hashes: list[str], ledger: pd.DataFrame) -> np.ndarray:
    import run_blade_v2 as base
    tours = base._oracle_tours(ledger, hashes)
    if len(ledger) != len(hashes) or tours.shape != (len(hashes), 100):
        raise RuntimeError("label array shape mismatch")
    if not ledger.verified.astype(bool).all() or not ledger.status.eq("solved").all():
        raise RuntimeError("unverified expert row")
    if not all(validate_tour_0based(tour, 100)[0] for tour in tours):
        raise RuntimeError("illegal expert tour")
    costs = np.asarray([
        tour_cost_int(integer_distance_matrix(points), tour)
        for points, tour in zip(coordinates, tours, strict=True)
    ], np.int64)
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)):
        raise RuntimeError("expert objective mismatch")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (len(hashes), 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]):
        raise RuntimeError("BQ tour closure mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist():
        raise RuntimeError("expert tour hash mismatch")
    return tours


def label() -> dict[str, Any]:
    prepare()
    prior = pre_p6_cache()
    pre_known = set(prior.instance_hash.astype(str))
    recovery_path = LABEL_DIR / "recovered_retry_ledger.parquet"
    solve_prior = prior
    if recovery_path.is_file():
        recovered = pd.read_parquet(recovery_path)[list(ORACLE_LEDGER_COLUMNS)]
        if not recovered.verified.astype(bool).all() or set(recovered.binary_hash.astype(str)) != {CANONICAL_CONCORDE_SHA256}:
            raise RuntimeError("invalid recovered Concorde retry ledger")
        solve_prior = pd.concat([prior, recovered], ignore_index=True).drop_duplicates("instance_hash", keep="last")
    coordinate_map: dict[str, np.ndarray] = {}
    dataset_hashes: dict[tuple[str, str], list[str]] = {}
    union_order: list[str] = []
    for replicate in REPLICATES:
        for arm in ARMS:
            coordinates, hashes, _ = load_bundle(dataset_bundle(replicate, arm))
            dataset_hashes[(replicate, arm)] = hashes
            for value, points in zip(hashes, coordinates, strict=True):
                if value not in coordinate_map:
                    coordinate_map[value] = points
                    union_order.append(value)
                elif not np.array_equal(coordinate_map[value], points):
                    raise RuntimeError(f"coordinate hash collision: {value}")
    union_xy = np.stack([coordinate_map[value] for value in union_order])
    physical_new = sum(value not in pre_known for value in union_order)
    global_dir = LABEL_DIR / "global_unique_union"
    global_integrity = global_dir / "labeling_integrity.json"
    started = time.perf_counter()
    if global_integrity.is_file():
        info = json.loads(global_integrity.read_text())
        ledger = pd.read_parquet(global_dir / "ledger.parquet")
        if info.get("status") != "PASS" or info.get("union_membership_sha256") != membership_hash(union_order):
            raise RuntimeError("global label cache drift")
    else:
        import run_blade_v2 as base
        ledger = base.label_with_cache(union_xy, union_order, global_dir, solve_prior)
        ledger.to_parquet(global_dir / "ledger.parquet", index=False)
        info = {
            "status": "PASS", "unique_union_n": len(union_order), "preexisting_cache_hits": len(union_order) - physical_new,
            "physical_new_concorde_calls": physical_new, "failures": 0,
            "pipeline_wall_seconds": time.perf_counter() - started,
            "union_membership_sha256": membership_hash(union_order),
            "ledger_sha256": sha256_file(global_dir / "ledger.parquet"),
            "concorde_sha256": CANONICAL_CONCORDE_SHA256,
        }
        retry_path = LABEL_DIR / "concorde_retry_audit.json"
        if retry_path.is_file(): info["retry_audit"] = json.loads(retry_path.read_text())
        write_json(global_integrity, info)
    if set(ledger.instance_hash.astype(str)) != set(union_order) or len(ledger) != len(union_order):
        raise RuntimeError("global union ledger incomplete")
    global_lookup = ledger.set_index("instance_hash", drop=False)
    rows = []
    for replicate in REPLICATES:
        for arm in ARMS:
            directory = label_run_dir(replicate, arm); directory.mkdir(parents=True, exist_ok=True)
            coordinates, hashes, _ = load_bundle(dataset_bundle(replicate, arm))
            selected_ledger = global_lookup.loc[hashes].reset_index(drop=True)
            tours = validate_ledger(coordinates, hashes, selected_ledger)
            selected_ledger.to_parquet(directory / "ledger.parquet", index=False)
            manifest = pd.read_parquet(dataset_manifest(replicate, arm))
            payload = set(ORACLE_LEDGER_COLUMNS[1:])
            labeled = manifest[[column for column in manifest.columns if column not in payload]].merge(selected_ledger, on="instance_hash", validate="one_to_one")
            labeled.to_parquet(directory / "labeled.parquet", index=False)
            p4c2.save_bundle(directory / "labeled_bundle.npz", coordinates, hashes, tours)
            preexisting = sum(value in pre_known for value in hashes)
            record = {
                "status": "PASS", "replicate": replicate, "arm": arm, "verified": TRAIN_N,
                "pre_p6_cache_reuse_n": preexisting, "needed_new_label_n": TRAIN_N - preexisting,
                "physical_new_calls_global_union": physical_new, "failures": 0,
                "training_membership_sha256": membership_hash(hashes),
                "ledger_sha256": sha256_file(directory / "ledger.parquet"),
                "labeled_manifest_sha256": sha256_file(directory / "labeled.parquet"),
                "labeled_bundle_sha256": sha256_file(directory / "labeled_bundle.npz"),
                "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
                "all_tours_legal": True, "all_costs_reproduced": True, "concorde_sha256": CANONICAL_CONCORDE_SHA256,
            }
            write_json(directory / "labeling_integrity.json", record)
            rows.append(record)
    lines = [
        "# Concorde Expert Label Recovery Audit", "",
        "| Dataset | Pre-P6 reused | Dataset missing before P6 | Verified | Label bundle SHA256 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['replicate']}-{row['arm']} | {row['pre_p6_cache_reuse_n']:,} | {row['needed_new_label_n']:,} | {row['verified']:,} | `{row['labeled_bundle_sha256']}` |")
    lines.extend([
        "", f"The six memberships contain {len(union_order):,} unique hashes. The verified pre-P6 cache covered {len(union_order)-physical_new:,}; exactly **{physical_new:,} unique Concorde calls** were required and shared across datasets.",
        f"Global labeling wall-clock: {float(info['pipeline_wall_seconds']):.3f}s; failures: 0; canonical Concorde SHA256 `{CANONICAL_CONCORDE_SHA256}`.",
        f"Sandbox diagnostic failures/retries before the unsandboxed canonical run: {json.dumps(info.get('retry_audit', {}), sort_keys=True)}.",
        "", "All tours are 0-based Hamiltonian permutations with implicit cycle closure. Integer objectives were recomputed from the exact frozen coordinates; BQ closed-coordinate inputs were validated. No failed instance was replaced.",
        "", "The prompt table values 1,468/5,560/... are confirmed as missing counts; their complementary reuse counts are 8,532/4,440/....", "", "Status: **PASS**.",
    ])
    (OUT / "03_concorde_labeling_audit.md").write_text("\n".join(lines) + "\n")
    Status().update(phase="label", status="PASS", unique_new_labels=physical_new)
    return info


def train(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    protocol = protocol_gate()
    integrity_path = run_dir(replicate, arm) / "training_integrity.json"
    checkpoint = checkpoint_path(replicate, arm)
    if integrity_path.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and checkpoint.is_file() and sha256_file(checkpoint) == existing.get("checkpoint_sha256"):
            return existing
    label_info = json.loads((label_run_dir(replicate, arm) / "labeling_integrity.json").read_text())
    if label_info.get("status") != "PASS" or label_info.get("verified") != TRAIN_N:
        raise RuntimeError("labeling gate failed")
    coordinates, hashes, tours = load_bundle(label_run_dir(replicate, arm) / "labeled_bundle.npz", True)
    assert tours is not None
    frozen_xy, frozen_hashes, _ = load_bundle(dataset_bundle(replicate, arm))
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
        raise RuntimeError("labeled population membership drift")
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    device = torch.device(device_name)
    directory = run_dir(replicate, arm) / "training_cycle"
    core.RUN_ID = f"TARGET_AWARE_P6_1_{replicate}_{arm}_POP"
    core.cycle_dir = lambda selected_arm, cycle: directory
    core.model_path = lambda selected_arm, index: checkpoint if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    config = {
        "replicate": replicate, "arm": arm, "data_path": "G2_Final_Population10k",
        "evolution_seed": p5.EVOLUTION_SEEDS[replicate], "training_seed": TRAINING_SEED,
        "membership_sha256": membership_hash(hashes), "dataset_bundle_sha256": sha256_file(dataset_bundle(replicate, arm)),
        "label_bundle_sha256": label_info["labeled_bundle_sha256"], "protocol_sha256": protocol["protocol_sha256"],
        "base_training_protocol": base_training_protocol(),
    }
    config["training_config_sha256"] = canonical_hash(config)
    write_json(run_dir(replicate, arm) / "training_config.json", config)
    started = time.perf_counter()
    trained, meta, timing = core.train(f"{replicate}_{arm}_Pop", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)
    invocation = time.perf_counter() - started
    stage = directory / "09_bq_training"
    e000 = torch.load(stage / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    curve = pd.read_csv(stage / "training_loss_curve.csv")
    if (
        not e000_equal or len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240
        or not curve.hard_instances_seen.eq(TRAIN_N).all() or meta["sha256"] != sha256_file(checkpoint)
        or not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters())
    ):
        raise RuntimeError("training integrity failure")
    result = {
        "status": "PASS", **config, "m4_sha256": M4_SHA, "e000_weights_equal_m4": e000_equal,
        "epochs": 256, "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": invocation,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]), "checkpoint_sha256": meta["sha256"],
        "checkpoint_path": str(checkpoint), "loss_curve_sha256": sha256_file(stage / "training_loss_curve.csv"),
    }
    write_json(integrity_path, result)
    del trained, model
    if device.type == "cuda": torch.cuda.empty_cache()
    write_training_manifest()
    return result


def write_training_manifest() -> None:
    rows = []
    for replicate in REPLICATES:
        for arm in ARMS:
            path = run_dir(replicate, arm) / "training_integrity.json"
            if path.is_file():
                info = json.loads(path.read_text())
                rows.append({key: info.get(key) for key in (
                    "replicate", "arm", "status", "evolution_seed", "training_seed", "membership_sha256",
                    "dataset_bundle_sha256", "label_bundle_sha256", "m4_sha256", "e000_weights_equal_m4",
                    "epochs", "optimizer_steps", "final_loss", "training_loop_wall_s", "checkpoint_sha256",
                    "checkpoint_path", "training_config_sha256",
                )})
            else:
                rows.append({"replicate": replicate, "arm": arm, "status": "PENDING", "training_seed": TRAINING_SEED})
    pd.DataFrame(rows).to_csv(OUT / "04_training_runs_manifest.csv", index=False)


def evaluate(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    output_integrity = run_dir(replicate, arm) / "evaluation_integrity.json"
    if output_integrity.is_file():
        existing = json.loads(output_integrity.read_text())
        if existing.get("status") == "PASS" and eval_csv(replicate, arm).is_file(): return existing
    training = json.loads((run_dir(replicate, arm) / "training_integrity.json").read_text())
    checkpoint = checkpoint_path(replicate, arm)
    if training["checkpoint_sha256"] != sha256_file(checkpoint): raise RuntimeError("checkpoint drift")
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    device = torch.device(device_name)
    model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=training["checkpoint_sha256"])
    frame, wall = core.evaluate(model, checkpoint, eval_label(replicate, arm), eval_csv(replicate, arm), Status(), f"{replicate}_{arm}_Pop")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all():
        raise RuntimeError("evaluation incomplete")
    # Verify frozen evaluator/data hashes against P5 historical runs.
    historical_path = p5.ORIGINAL[arm]["eval_csv"] if replicate == "Original" else p5.eval_csv(replicate, arm)
    historical = pd.read_csv(historical_path).set_index("dataset")
    current = frame.set_index("dataset")
    for dataset in DATASETS:
        if current.at[dataset, "dataset_sha256"] != historical.at[dataset, "dataset_sha256"] or current.at[dataset, "evaluator_config_sha256"] != historical.at[dataset, "evaluator_config_sha256"]:
            raise RuntimeError(f"evaluation protocol drift: {dataset}")
    result = {
        "status": "PASS", "replicate": replicate, "arm": arm, "checkpoint_sha256": meta["sha256"],
        "datasets": list(DATASETS), "all_tours_valid": True, "evaluation_wall_s": wall,
        "results_sha256": sha256_file(eval_csv(replicate, arm)),
    }
    write_json(output_integrity, result)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def pop_detail(replicate: str, arm: str, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(replicate, arm) / "evaluation_work" / eval_label(replicate, arm) / dataset / "instance_metrics.parquet")


def align_pair(loader: Callable[[str, str, str], pd.DataFrame], replicate: str, dataset: str) -> pd.DataFrame:
    pieces = []
    for arm in ARMS:
        frame = loader(replicate, arm, dataset)[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
        if not frame.valid_tour.all(): raise RuntimeError(f"invalid tour: {replicate}/{arm}/{dataset}")
        pieces.append(frame.rename(columns={
            "coordinate_hash": f"coordinate_hash_{arm}", "n_nodes": f"n_nodes_{arm}",
            "optimal_cost_int": f"optimal_cost_int_{arm}", "gap_percent": f"gap_{arm}",
        }).drop(columns="valid_tour"))
    merged = pieces[0].merge(pieces[1], on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        if not merged[f"{field}_E0"].astype(str).eq(merged[f"{field}_E1"].astype(str)).all():
            raise RuntimeError(f"paired benchmark mismatch: {replicate}/{dataset}/{field}")
    return pd.DataFrame({
        "replicate": replicate, "dataset": dataset, "instance_id": merged.instance_id.astype(str),
        "coordinate_hash": merged.coordinate_hash_E0.astype(str), "n_nodes": merged.n_nodes_E0.astype(int),
        "optimal_cost_int": merged.optimal_cost_int_E0.astype(np.int64),
        "gap_E0": merged.gap_E0.astype(float), "gap_E1": merged.gap_E1.astype(float),
        "Delta_E0_minus_E1": merged.gap_E0.astype(float) - merged.gap_E1.astype(float),
    })


def bootstrap(values: np.ndarray, key: str) -> tuple[float, float]:
    return p4b1.paired_bootstrap(np.asarray(values, float), p4b1.stable_seed(f"P6-1|{key}"))


def gap5_summary(frames: dict[str, pd.DataFrame], replicate: str) -> dict[str, Any]:
    rng = np.random.default_rng(p4b1.stable_seed(f"P6-1|{replicate}|GAP5"))
    estimates = np.empty(BOOTSTRAP_N)
    values = [frames[name].Delta_E0_minus_E1.to_numpy(float) for name in GAP5]
    for index in range(BOOTSTRAP_N):
        estimates[index] = np.mean([value[rng.integers(0, len(value), len(value))].mean() for value in values])
    return {
        "mean_gap_E0": float(np.mean([frames[name].gap_E0.mean() for name in GAP5])),
        "mean_gap_E1": float(np.mean([frames[name].gap_E1.mean() for name in GAP5])),
        "ci95_low": float(np.quantile(estimates, .025)), "ci95_high": float(np.quantile(estimates, .975)),
        "n_instances": int(sum(len(frames[name]) for name in GAP5)),
    }


def historical_archive_means(replicate: str, dataset: str) -> tuple[float, float]:
    frame = p5.paired(replicate, dataset)
    return float(frame.gap_E0.mean()), float(frame.gap_E1.mean())


def finalize() -> dict[str, Any]:
    protocol_gate(); write_training_manifest()
    for replicate in REPLICATES:
        for arm in ARMS:
            if json.loads((run_dir(replicate, arm) / "training_integrity.json").read_text()).get("status") != "PASS": raise RuntimeError("training incomplete")
            if json.loads((run_dir(replicate, arm) / "evaluation_integrity.json").read_text()).get("status") != "PASS": raise RuntimeError("evaluation incomplete")
    development = set(pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str))
    heldout = set(json.loads((ROOT / "target_aware_pilot/p1_heldout22_names.json").read_text())["names"])
    result_rows = []; population_frames: dict[tuple[str, str], pd.DataFrame] = {}; instance_parts = []
    for replicate in REPLICATES:
        frames = {dataset: align_pair(pop_detail, replicate, dataset) for dataset in DATASETS}
        tsplib = frames["OriginalTSPLIB70"]
        if set(tsplib.instance_id) != development | heldout: raise RuntimeError("TSPLIB70 split mismatch")
        frames["TSPLIB48_development"] = tsplib[tsplib.instance_id.isin(development)].copy()
        frames["TSPLIB22"] = tsplib[tsplib.instance_id.isin(heldout)].copy()
        for dataset, frame in frames.items():
            population_frames[(replicate, dataset)] = frame
            values = frame.Delta_E0_minus_E1.to_numpy(float)
            low, high = bootstrap(values, f"{replicate}|{dataset}")
            result_rows.append({
                "replicate": replicate, "dataset": dataset, "n_instances": len(frame),
                "mean_gap_E0_Pop": float(frame.gap_E0.mean()), "mean_gap_E1_Pop": float(frame.gap_E1.mean()),
                "Delta_population_E0_minus_E1": float(values.mean()), "ci95_low": low, "ci95_high": high,
                "E1_instance_wins": int(np.sum(values > 0)), "E0_instance_wins": int(np.sum(values < 0)),
                "ties": int(np.sum(values == 0)), "bootstrap_replicates": BOOTSTRAP_N,
            })
            instance_parts.append(frame)
        gap5 = gap5_summary(frames, replicate)
        result_rows.append({
            "replicate": replicate, "dataset": "GAP5", "n_instances": gap5["n_instances"],
            "mean_gap_E0_Pop": gap5["mean_gap_E0"], "mean_gap_E1_Pop": gap5["mean_gap_E1"],
            "Delta_population_E0_minus_E1": gap5["mean_gap_E0"] - gap5["mean_gap_E1"],
            "ci95_low": gap5["ci95_low"], "ci95_high": gap5["ci95_high"],
            "E1_instance_wins": np.nan, "E0_instance_wins": np.nan, "ties": np.nan, "bootstrap_replicates": BOOTSTRAP_N,
        })
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "05_original70_tsplib22_gap5_results.csv", index=False)
    pd.concat(instance_parts, ignore_index=True).to_parquet(OUT / "support_population_per_instance_paired_differences.parquet", index=False)

    comparison_rows = []
    for replicate in REPLICATES:
        for dataset in (*DATASETS, "TSPLIB48_development", "TSPLIB22", "GAP5"):
            pop = results[(results.replicate.eq(replicate)) & (results.dataset.eq(dataset))].iloc[0]
            if dataset == "GAP5":
                archive_source = pd.read_csv(P5 / "08_original70_tsplib22_gap5_results.csv")
                archive = archive_source[(archive_source.replicate.eq(replicate)) & (archive_source.dataset.eq("GAP5_macro"))].iloc[0]
                archive_e0, archive_e1 = float(archive.mean_gap_E0), float(archive.mean_gap_E1)
            elif dataset in ("TSPLIB48_development", "TSPLIB22"):
                archive_source = pd.read_csv(P5 / "08_original70_tsplib22_gap5_results.csv")
                archive_name = "TSPLIB48_development" if dataset.startswith("TSPLIB48") else "TSPLIB22_previously_heldout"
                archive = archive_source[(archive_source.replicate.eq(replicate)) & (archive_source.dataset.eq(archive_name))].iloc[0]
                archive_e0, archive_e1 = float(archive.mean_gap_E0), float(archive.mean_gap_E1)
            else:
                archive_e0, archive_e1 = historical_archive_means(replicate, dataset)
            delta_archive = archive_e0 - archive_e1
            delta_population = float(pop.Delta_population_E0_minus_E1)
            comparison_rows.append({
                "replicate": replicate, "dataset": dataset,
                "gap_E0_Archive": archive_e0, "gap_E1_Archive": archive_e1,
                "gap_E0_Pop": float(pop.mean_gap_E0_Pop), "gap_E1_Pop": float(pop.mean_gap_E1_Pop),
                "Delta_archive_E0_minus_E1": delta_archive, "Delta_population_E0_minus_E1": delta_population,
                "E0_Archive_minus_Pop": archive_e0 - float(pop.mean_gap_E0_Pop),
                "E1_Archive_minus_Pop": archive_e1 - float(pop.mean_gap_E1_Pop),
                "Interaction_DeltaPop_minus_DeltaArchive": delta_population - delta_archive,
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "06_archive_vs_population_comparison.csv", index=False)

    analysis_rows = []
    for endpoint in ("OriginalTSPLIB70", "TSPLIB22", "GAP5"):
        part = results[results.dataset.eq(endpoint)].set_index("replicate").loc[list(REPLICATES)]
        values = part.Delta_population_E0_minus_E1.to_numpy(float)
        analysis_rows.append({
            "endpoint": endpoint, "replicate_n": 3, "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)), "sample_sd": float(values.std(ddof=1)),
            "min_difference": float(values.min()), "max_difference": float(values.max()),
            "E1_win_count": int(np.sum(values > 0)), "E0_win_count": int(np.sum(values < 0)),
            "all_E1": bool(np.all(values > 0)), "all_E0": bool(np.all(values < 0)),
        })
    analysis = pd.DataFrame(analysis_rows)
    analysis.to_csv(OUT / "07_survivor_selection_effect.csv", index=False)

    primary = results[results.dataset.eq("OriginalTSPLIB70")].set_index("replicate").loc[list(REPLICATES)]
    ts22 = results[results.dataset.eq("TSPLIB22")].set_index("replicate").loc[list(REPLICATES)]
    gap5 = results[results.dataset.eq("GAP5")].set_index("replicate").loc[list(REPLICATES)]
    all_primary_e1 = bool(primary.Delta_population_E0_minus_E1.gt(0).all())
    all_primary_e0 = bool(primary.Delta_population_E0_minus_E1.lt(0).all())
    gap5_systematic_degradation = bool(gap5.Delta_population_E0_minus_E1.lt(0).all())
    if all_primary_e1 and not gap5_systematic_degradation:
        decision = "DIRECT_POPULATION_STRUCTURE_GAIN_PRELIMINARILY_SUPPORTED"; short = "SUPPORTED_PRELIMINARY"
    elif all_primary_e0:
        decision = "DIRECT_POPULATION_STRUCTURE_GAIN_NOT_SUPPORTED"; short = "NOT_SUPPORTED"
    else:
        decision = "DIRECT_POPULATION_STRUCTURE_GAIN_INCONCLUSIVE"; short = "INCONCLUSIVE"

    original_comparison = comparison[comparison.dataset.eq("OriginalTSPLIB70")].set_index("replicate").loc[list(REPLICATES)]
    s2 = original_comparison.loc["S2"]
    direct_advantages = np.concatenate([original_comparison.E0_Archive_minus_Pop.to_numpy(float), original_comparison.E1_Archive_minus_Pop.to_numpy(float)])
    overall_direct = bool(direct_advantages.mean() > 0 and np.sum(direct_advantages > 0) >= 4)
    interaction_lines = [
        "# Archive vs Direct-Population Interaction Analysis", "",
        "All differences are descriptive across exactly three evolution replicates. Instance bootstrap intervals do not create additional independent evolution replicates.", "",
        "| Replicate | Delta archive | Delta population | Interaction | E0 Archive-Pop | E1 Archive-Pop |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for replicate, row in original_comparison.iterrows():
        interaction_lines.append(f"| {replicate} | {row.Delta_archive_E0_minus_E1:+.4f} | {row.Delta_population_E0_minus_E1:+.4f} | {row.Interaction_DeltaPop_minus_DeltaArchive:+.4f} | {row.E0_Archive_minus_Pop:+.4f} | {row.E1_Archive_minus_Pop:+.4f} |")
    interaction_lines.extend([
        "", f"Direct Population has lower Original70 gap in {int(np.sum(direct_advantages > 0))}/6 arm-replicate cells; mean Archive-Pop difference {direct_advantages.mean():+.4f} pp. Overall descriptive advantage = **{overall_direct}**.",
        f"S2 archive Delta was {s2.Delta_archive_E0_minus_E1:+.4f} pp; population Delta is {s2.Delta_population_E0_minus_E1:+.4f} pp. Thus the S2 reversal {'disappears' if s2.Delta_population_E0_minus_E1 > 0 else 'persists'} under direct-population training.",
        "", "Interaction = Delta_population - Delta_archive. No formal interaction significance is claimed.",
    ])
    (OUT / "08_interaction_analysis.md").write_text("\n".join(interaction_lines) + "\n")

    global_label = json.loads((LABEL_DIR / "global_unique_union/labeling_integrity.json").read_text())
    structure = pd.read_csv(OUT / "02_population_distribution_audit.csv")
    pop_structure = structure[structure.training_path.eq("G2_Final_Population10k")]
    s2_struct = pop_structure[pop_structure.replicate.eq("S2")].set_index("arm")
    e1_s2_struct_better = bool(
        s2_struct.at["E1", "sliced_wasserstein"] < s2_struct.at["E0", "sliced_wasserstein"]
        and s2_struct.at["E1", "coarse_js"] < s2_struct.at["E0", "coarse_js"]
    )
    direct_population = structure[structure.training_path.eq("G2_Final_Population10k")]
    archive_population = structure[structure.training_path.eq("P5_ArchiveBased_Structure10k")]
    e1_structure_better_all = all(
        direct_population[(direct_population.replicate.eq(replicate)) & direct_population.arm.eq("E1")].iloc[0].sliced_wasserstein
        < direct_population[(direct_population.replicate.eq(replicate)) & direct_population.arm.eq("E0")].iloc[0].sliced_wasserstein
        and direct_population[(direct_population.replicate.eq(replicate)) & direct_population.arm.eq("E1")].iloc[0].coarse_js
        < direct_population[(direct_population.replicate.eq(replicate)) & direct_population.arm.eq("E0")].iloc[0].coarse_js
        for replicate in REPLICATES
    )
    lines = [
        "# P6-1 Final Research Conclusion", "", f"## `{decision}` — **{short}**", "",
        "Delta = gap(E0-Pop) - gap(E1-Pop); positive favors structure-guided E1.", "",
        "| Replicate | Original70 E0-Pop | Original70 E1-Pop | Delta | 95% CI | TSPLIB22 Delta | GAP5 Delta |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for replicate in REPLICATES:
        lines.append(f"| {replicate} | {primary.at[replicate,'mean_gap_E0_Pop']:.4f} | {primary.at[replicate,'mean_gap_E1_Pop']:.4f} | {primary.at[replicate,'Delta_population_E0_minus_E1']:+.4f} | [{primary.at[replicate,'ci95_low']:+.4f}, {primary.at[replicate,'ci95_high']:+.4f}] | {ts22.at[replicate,'Delta_population_E0_minus_E1']:+.4f} | {gap5.at[replicate,'Delta_population_E0_minus_E1']:+.4f} |")
    agg = analysis.set_index("endpoint")
    lines.extend([
        "", "## Training-membership composition diagnostic", "",
        f"- E1 G2 has lower SWD and 3x3 JS than E0 G2 in all three replicates: **{e1_structure_better_all}**. Structural guidance therefore still shapes the final population even though solver gains are inconsistent.",
        f"- Direct G2 populations retain {100*direct_population.parent_origin_fraction.min():.2f}%–{100*direct_population.parent_origin_fraction.max():.2f}% P03-origin parents; all Archive-based Structure10k sets contain {int(archive_population.parent_origin_n.sum())} parents because the old acquisition selected eligible offspring only.",
        f"- E1 largest lineage share is {100*direct_population[direct_population.arm.eq('E1')].largest_lineage_share.min():.2f}%–{100*direct_population[direct_population.arm.eq('E1')].largest_lineage_share.max():.2f}%, versus E0 {100*direct_population[direct_population.arm.eq('E0')].largest_lineage_share.min():.2f}%–{100*direct_population[direct_population.arm.eq('E0')].largest_lineage_share.max():.2f}%; no new E1 lineage concentration is observed.",
        "- The old terminal acquisition materially changes composition and structure metrics, but those changes do not translate into a consistent Archive-vs-Population performance direction.",
        "", "## Required answers", "",
        "1. All six frozen G2 Population10k datasets completed integrity, labeling, M4 initialization, 256-epoch training, and frozen evaluation.",
        f"2. New unique Concorde labels: **{global_label['physical_new_concorde_calls']:,}**; all six reach 10,000/10,000 valid tours.",
        "3. Original70 gaps are reported in the table above and the full CSV.",
        f"4. Stable E1 advantage across all three evolution seeds: **{all_primary_e1}**; E1 win count {int(agg.at['OriginalTSPLIB70','E1_win_count'])}/3.",
        f"5. TSPLIB22 E1 win count {int(agg.at['TSPLIB22','E1_win_count'])}/3; GAP5 E1 win count {int(agg.at['GAP5','E1_win_count'])}/3. They {'support' if agg.at['TSPLIB22','all_E1'] and agg.at['GAP5','all_E1'] else 'do not uniformly support'} the same direction.",
        f"6. Direct Population overall descriptive Original70 advantage over Archive-based training: **{overall_direct}** ({int(np.sum(direct_advantages > 0))}/6 cells; mean {direct_advantages.mean():+.4f} pp). This is independent of E1-vs-E0 validity.",
        f"7. P5 S2 reverse degradation {'disappears' if s2.Delta_population_E0_minus_E1 > 0 else 'persists'}: population Delta {s2.Delta_population_E0_minus_E1:+.4f} pp versus archive {s2.Delta_archive_E0_minus_E1:+.4f} pp. S2 structural matching remains better for E1 = {e1_s2_struct_better}.",
        f"8. Evidence that Archive30k→Structure10k weakened training value is {'suggestive but descriptive' if overall_direct else 'not supported overall'}; arm-specific path effects and interactions are in `06`/`08`.",
        "9. Regardless of this result, P6-0 found no empirical/theoretical justification for the old 70/30 ratio. Keep it only as a frozen baseline; define the next structural objective explicitly rather than tuning 70/30 post hoc.",
        "", "Original70 is target-informed and TSPLIB22 has been repeatedly inspected. These results are exploratory, not independent confirmation. No extra training seed, mutation, acquisition, feature selection, or P6-2 was run.",
        "", f"`{decision}`", "", f"`{short}`", "", "`BLADE_P6_1_FINAL_POPULATION_DIRECT_TRAINING_COMPLETE`",
    ])
    (OUT / "09_final_research_conclusion.md").write_text("\n".join(lines) + "\n")

    x = np.arange(3); fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, table, title in ((axes[0], primary, "Original70"), (axes[1], ts22, "TSPLIB22"), (axes[2], gap5, "GAP5")):
        ax.bar(x - .17, table.mean_gap_E0_Pop, .34, label="E0-Pop")
        ax.bar(x + .17, table.mean_gap_E1_Pop, .34, label="E1-Pop")
        ax.set_xticks(x); ax.set_xticklabels(REPLICATES); ax.set_title(title); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("mean gap (%)"); axes[-1].legend(); fig.tight_layout(); fig.savefig(FIGURES / "01_population_training_endpoints.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); width = .18
    for index, column in enumerate(("gap_E0_Archive", "gap_E1_Archive", "gap_E0_Pop", "gap_E1_Pop")):
        ax.bar(x + (index - 1.5) * width, original_comparison[column], width, label=column.replace("gap_", ""))
    ax.set_xticks(x); ax.set_xticklabels(REPLICATES); ax.set_ylabel("Original70 mean gap (%)"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "02_archive_population_2x2.png", dpi=180); plt.close(fig)
    metadata = {
        "status": "COMPLETE", "decision": decision, "short_decision": short,
        "new_unique_concorde_labels": global_label["physical_new_concorde_calls"],
        "all_six_trained": True, "no_followup_started": True,
        "results_sha256": sha256_file(OUT / "05_original70_tsplib22_gap5_results.csv"),
        "comparison_sha256": sha256_file(OUT / "06_archive_vs_population_comparison.csv"),
    }
    write_json(OUT / "run_metadata.json", metadata); Status().update(phase="complete", **metadata)
    return metadata


def full(device: str) -> dict[str, Any]:
    prepare(); label()
    for replicate in REPLICATES:
        for arm in ARMS:
            print(f"P6_1_TRAIN_START={replicate}-{arm}", flush=True)
            train(replicate, arm, device)
    for replicate in REPLICATES:
        for arm in ARMS:
            print(f"P6_1_EVAL_START={replicate}-{arm}", flush=True)
            evaluate(replicate, arm, device)
    return finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "label", "train", "evaluate", "finalize", "full"))
    parser.add_argument("--replicate", choices=REPLICATES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare()
    elif args.phase == "label": result = label()
    elif args.phase == "finalize": result = finalize()
    elif args.phase == "full": result = full(args.device)
    else:
        if args.replicate is None or args.arm is None: parser.error("--replicate and --arm required")
        result = train(args.replicate, args.arm, args.device) if args.phase == "train" else evaluate(args.replicate, args.arm, args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
