#!/usr/bin/env python3
"""Formal P6-B paired Top5 target-distribution-guided evolution experiment.

The runner is stage-resumable and intentionally reuses the frozen P6-A
representation/matcher and the P5/P6-1 training/evaluation implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P2 = ROOT / "target_aware_pilot/p2_bounded_evolution"
P5 = ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness"
P6A = ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"
P6AQ = ROOT / "target_aware_pilot/p6aq_coordinate_quantization_sensitivity"
P61 = ROOT / "target_aware_pilot/p6_1_direct_population_training"
EVOLUTION = OUT / "evolution"
POPULATIONS = OUT / "population_manifests"
FEATURES_DIR = OUT / "features"
FEATURE_CACHE = FEATURES_DIR / "cache"
LABELS = OUT / "labels"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"
FIGURES = OUT / "figures"
CONFIGS = OUT / "configs"

sys.path[:0] = [
    str(ROOT / "src"), str(ROOT / "scripts"), str(P2), str(P5), str(P6A), str(P61),
    str(ROOT / "target_aware_pilot/p4b1_acquisition_ablation"),
    str(ROOT / "target_aware_pilot/p4c2_data_sampling_robustness"),
]

import run_p2_bounded_evolution as p2  # noqa: E402
import run_p5_structure_guided_evolution_robustness as p5  # noqa: E402
import run_p6_1_direct_population_training as p61  # noqa: E402
import run_p6a_audit as p6a  # type: ignore  # noqa: E402
import run_p4b1_acquisition_ablation as p4b1  # noqa: E402
import run_p4c2_data_sampling_robustness as p4c2  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import cycle_child_seed, membership_hash  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator_with_config  # noqa: E402
from mutation.parameters import load_named_profile  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, ORACLE_LEDGER_COLUMNS, sha256_file  # noqa: E402

# run_p6a_audit is not importable by module name from its nested directory.
# Replace the placeholder import above if normal resolution did not find it.
if not hasattr(p6a, "exact_transport_lp"):
    raise RuntimeError("P6-A import did not resolve to the frozen audit module")

REPLICATES = ("Original", "S1", "S2")
ARMS = ("E0", "E1")
GENERATIONS = (1, 2)
POPULATION_N = 10_000
ATTEMPTS_PER_GENERATION = 110_000
TRAINING_SEED = 2026
TOP5 = tuple(p6a.TOP5)
PROFILE_NAME = "FOGA11_BASE_V1"
PROFILE_PATH = ROOT / "src/mutation/foga2019_operators.yaml"
P03_BUNDLE = p2.P03_BUNDLE
P03_META = p2.P03_META
M4 = p5.M4
M4_SHA = p5.M4_SHA
DATASETS = tuple(p5.DATASETS)
GAP5 = tuple(p5.GAP5_DATASETS)
BOOTSTRAP_N = 10_000
FEATURE_SHARD_N = 250
FEATURE_WORKERS = 12
SEEDS = dict(p5.EVOLUTION_SEEDS)
NAMESPACES = {name: dict(p5.MUTATION_NAMESPACES[name]) for name in REPLICATES}
QUOTA = p6a.capacities(pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str).tolist())


def ensure_dirs() -> None:
    for path in (OUT, EVOLUTION, POPULATIONS, FEATURES_DIR, FEATURE_CACHE, LABELS, CHECKPOINTS, LOGS, FIGURES, CONFIGS):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.floating): return float(value)
        if isinstance(value, np.bool_): return bool(value)
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Path): return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n")
    os.replace(temporary, path)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str], tours: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    content: dict[str, Any] = {"coordinates": np.asarray(coordinates, np.float64), "instance_hash": np.asarray(hashes)}
    if tours is not None: content["expert_tours"] = np.asarray(tours, np.int64)
    np.savez_compressed(temporary, **content)
    os.replace(temporary, path)


def load_bundle(path: Path, require_tours: bool = False) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
        tours = np.asarray(payload["expert_tours"], np.int64) if "expert_tours" in payload.files else None
    if require_tours and tours is None: raise RuntimeError(f"missing expert tours: {path}")
    return coordinates, hashes, tours


def stage_dir(replicate: str, arm: str, generation: int) -> Path:
    return EVOLUTION / replicate / arm / f"G{generation}"


def shared_g1_dir(replicate: str) -> Path:
    return EVOLUTION / replicate / "shared_G1"


def population_manifest(replicate: str, arm: str) -> Path:
    return POPULATIONS / f"{replicate}_{arm}_P2.parquet"


def population_bundle(replicate: str, arm: str) -> Path:
    return POPULATIONS / f"{replicate}_{arm}_P2.npz"


def label_dir(replicate: str, arm: str) -> Path:
    return LABELS / f"{replicate}_{arm}"


def run_dir(replicate: str, arm: str) -> Path:
    return LOGS / f"{replicate}_{arm}"


def checkpoint_path(replicate: str, arm: str) -> Path:
    return CHECKPOINTS / f"{replicate}_{arm}_M5_E256.tsp.best"


def eval_label(replicate: str, arm: str) -> str:
    return f"P6B_{replicate}_{arm}_M5"


def eval_csv(replicate: str, arm: str) -> Path:
    return run_dir(replicate, arm) / "external_benchmark_results.csv"


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def base_training_protocol() -> dict[str, Any]:
    trainer = ROOT / "scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
    return {
        "authority": "P5/P6-1 frozen actual training configuration",
        "trainer_sha256": sha256_file(trainer), "m4_sha256": M4_SHA,
        "training_seed": TRAINING_SEED, "optimizer": "fresh Adam", "scheduler": None,
        "learning_rate": 3.75e-5, "epochs": 256, "optimizer_updates": 10_240,
        "logical_batch_size": 256, "microbatch_size": 64,
        "loss": "teacher-forced imitation", "checkpoint_rule": "fixed E256",
        "evaluation": "frozen greedy evaluator",
    }


def frozen_protocol() -> dict[str, Any]:
    return {
        "experiment": "BLADE-NCO P6-B",
        "source_commit": git_commit(),
        "initial_population": str(P03_BUNDLE.relative_to(ROOT)),
        "population_n": POPULATION_N, "generations": list(GENERATIONS),
        "attempts_per_parent": len(OPERATOR_NAMES), "attempts_per_generation": ATTEMPTS_PER_GENERATION,
        "operators": list(OPERATOR_NAMES), "profile": PROFILE_NAME,
        "replicates": list(REPLICATES), "evolution_seeds": SEEDS, "mutation_namespaces": NAMESPACES,
        "arms": {"E0": "uniform random without replacement", "E1": "P6-A exact Top5 OT"},
        "top5": list(TOP5), "quota": QUOTA.tolist(), "training_seed": TRAINING_SEED,
        "training": base_training_protocol(), "historical_overlap_allowed": True,
        "archive": False, "additional_diversity_objective": False, "heldout22_used_for_design": False,
        "p03_bundle_sha256": sha256_file(P03_BUNDLE), "p03_manifest_sha256": sha256_file(P03_META),
        "m4_checkpoint_sha256": sha256_file(M4),
        "foga_profile_sha256": sha256_file(PROFILE_PATH),
        "p6a_matching_config_sha256": sha256_file(P6A / "configs/frozen_matching_config.json"),
        "p6a_target_representation_sha256": sha256_file(P6A / "diagnostics/tsplib48_top5_representation.parquet"),
        "p6a_calibration_sha256": sha256_file(P6A / "diagnostics/uniform_reference_top5.parquet"),
        "p6a_implementation_sha256": sha256_file(P6A / "run_p6a_audit.py"),
        "p6aq_config_sha256": sha256_file(P6AQ / "configs/frozen_p6aq_config.json"),
        "scipy": scipy.__version__, "numpy": np.__version__, "python": platform.python_version(),
    }


def protocol_gate() -> dict[str, Any]:
    ensure_dirs()
    config_path = CONFIGS / "frozen_p6b_config.json"
    # Normalize JSON object keys (notably integer generation keys) before the
    # immutable comparison so a write/read round trip cannot create drift.
    current = json.loads(json.dumps(frozen_protocol(), sort_keys=True))
    current["protocol_sha256"] = canonical_hash(current)
    if config_path.is_file():
        saved = json.loads(config_path.read_text())
        if saved != current: raise RuntimeError("frozen P6-B config drift")
    else:
        write_json(config_path, current)
    return current


def coordinate_audit(coordinates: np.ndarray, hashes: list[str]) -> dict[str, Any]:
    reproduced = [coordinate_hash(points) for points in coordinates]
    return {
        "shape": list(coordinates.shape), "finite": bool(np.isfinite(coordinates).all()),
        "in_bounds": bool(((coordinates >= 0) & (coordinates <= 1)).all()),
        "unique_nodes": bool(all(len(np.unique(points, axis=0)) == 100 for points in coordinates)),
        "hashes_reproduce": reproduced == hashes, "unique_hashes": len(set(hashes)),
    }


def audit() -> dict[str, Any]:
    protocol = protocol_gate()
    p03_xy, p03_hashes, _ = load_bundle(P03_BUNDLE)
    p03_meta = pd.read_parquet(P03_META)
    check = coordinate_audit(p03_xy, p03_hashes)
    p5_config = json.loads((P5 / "preregistered_evolution_config.json").read_text())
    p6aq = json.loads((P6AQ / "completion.json").read_text())
    target = pd.read_parquet(P6A / "diagnostics/tsplib48_top5_representation.parquet")
    target_ids = pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str).tolist()
    integrity = {
        "p03_alignment": p03_hashes == p03_meta.instance_hash.astype(str).tolist(),
        "p03_valid": check["shape"] == [10_000, 100, 2] and check["unique_hashes"] == 10_000 and all(check[k] for k in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")),
        "m4_hash_match": sha256_file(M4) == M4_SHA,
        "foga_operator_count": len(OPERATOR_NAMES) == 11,
        "foga_profile_match": sha256_file(PROFILE_PATH) == json.loads(p2.C04_RUNTIME.read_text())["profile_sha256"],
        "seeds_recovered_from_p5": SEEDS == p5_config["new_evolution_seeds"] | {"Original": p5_config["original_evolution_seed"]},
        "namespaces_recovered_from_p5": {name: {str(k): v for k, v in NAMESPACES[name].items()} for name in REPLICATES} == p5_config["mutation_namespace_mapping"],
        "p6aq_robust": p6aq.get("label") == "ROBUST" and p6aq.get("no_p6b_executed") is True,
        "target_membership_match": target.instance_id.astype(str).tolist() == target_ids,
        "target_phi_finite": bool(np.isfinite(target[[f"phi__{f}" for f in TOP5]].to_numpy(float)).all()),
        "quota_match": bool(np.array_equal(QUOTA, p6a.capacities(target_ids))) and int(QUOTA.sum()) == 10_000,
        "matching_tests_pass": bool(pd.read_csv(P6A / "diagnostics/matching_correctness_tests.csv")[["unique_pass", "capacity_pass", "objective_consistent", "deterministic"]].astype(bool).all().all()),
        "m4_training_protocol_match": base_training_protocol()["trainer_sha256"] == p5_config["base_training_protocol"]["trainer_sha256"],
    }
    if not all(integrity.values()): raise RuntimeError(f"P6-B protocol integrity failure: {integrity}")
    report = [
        "# P6-B Frozen Protocol Integrity", "",
        "All frozen gates pass. P6-A Top5 definitions, target representation, Uniform calibration, 208/209 quotas, float32 squared-cost construction, and exact HiGHS TU assignment are reused directly.", "",
        f"- Source commit: `{protocol['source_commit']}`.",
        f"- Protocol SHA256: `{protocol['protocol_sha256']}`.",
        f"- P03 bundle: `{protocol['p03_bundle_sha256']}`; 10,000 unique valid TSP100.",
        f"- M4 checkpoint: `{protocol['m4_checkpoint_sha256']}`.",
        f"- Top5 implementation: `{protocol['p6a_implementation_sha256']}`.",
        f"- Uniform calibration: `{protocol['p6a_calibration_sha256']}`.",
        f"- Matching configuration: `{protocol['p6a_matching_config_sha256']}`.",
        f"- FOGA profile: `{protocol['foga_profile_sha256']}`; operators={len(OPERATOR_NAMES)}.",
        f"- Seeds: `{json.dumps(SEEDS, sort_keys=True)}`; namespaces: `{json.dumps(NAMESPACES, sort_keys=True)}`.",
        "- NNG single-SCC semantic atom, exact objective, HiGHS method, and reference quotas are unchanged.",
        "- TSPLIB22 is absent from evolution, representation fitting, selection, labeling, training, and checkpoint choice.",
        "", "Status: **PASS**.",
    ]
    (OUT / "01_frozen_protocol_integrity.md").write_text("\n".join(report) + "\n")
    (OUT / "00_protocol.md").write_text("# P6-B Protocol\n\nFormal paired E0 random-survivor versus E1 exact Top5-OT evolution. Three frozen evolution seeds, two generations, 110,000 attempts per arm-generation, direct G2 Population10k training, one frozen training seed, and frozen greedy evaluation. No archive, acquisition, novelty filter, hardness, diversity objective, third generation, or held-out-guided choice.\n")
    Status().update(phase="audit", status="PASS", protocol_sha256=protocol["protocol_sha256"])
    return {"status": "PASS", **integrity, "protocol_sha256": protocol["protocol_sha256"]}


def base_population() -> tuple[np.ndarray, pd.DataFrame]:
    coordinates, hashes, _ = load_bundle(P03_BUNDLE)
    source = pd.read_parquet(P03_META)
    if hashes != source.instance_hash.astype(str).tolist(): raise RuntimeError("P03 alignment drift")
    frame = source.copy()
    if "lineage_root" not in frame: frame["lineage_root"] = frame.instance_hash.astype(str)
    frame["direct_parent_hash"] = frame.get("parent_hash", pd.Series([pd.NA] * len(frame))).astype("string")
    frame["generation_born"] = -1
    frame["evolution_origin"] = "P03_formal_parent"
    frame["operator_id"] = frame.get("operator_id", pd.Series([pd.NA] * len(frame))).astype("string")
    return coordinates, frame


def mutation_paths(directory: Path) -> tuple[Path, Path, Path]:
    return directory / "offspring_coordinates.npz", directory / "mutation_manifest.parquet", directory / "runtime.json"


def mutate(replicate: str, arm_scope: str, generation: int, parent_xy: np.ndarray, parents: pd.DataFrame, directory: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    bundle_path, manifest_path, runtime_path = mutation_paths(directory)
    parent_hashes = parents.instance_hash.astype(str).tolist()
    namespace = NAMESPACES[replicate][generation]
    if bundle_path.is_file() and manifest_path.is_file() and runtime_path.is_file():
        coordinates, hashes, _ = load_bundle(bundle_path)
        manifest = pd.read_parquet(manifest_path)
        runtime = json.loads(runtime_path.read_text())
        accepted = manifest.proposal_status.eq("accepted") & manifest.valid_after_repair & ~manifest.within_offspring_duplicate
        if (runtime.get("parent_membership_sha256") == membership_hash(parent_hashes)
                and runtime.get("cycle_seed_namespace") == namespace
                and len(manifest) == ATTEMPTS_PER_GENERATION
                and hashes == manifest.loc[accepted, "child_hash"].astype(str).tolist()):
            return coordinates, manifest, runtime
        raise RuntimeError(f"cached mutation drift: {replicate}/{arm_scope}/G{generation}")
    directory.mkdir(parents=True, exist_ok=True)
    profile = load_named_profile(PROFILE_NAME)
    operators = list(OPERATOR_NAMES)
    if len(operators) != 11 or len(parent_hashes) != POPULATION_N: raise RuntimeError("mutation schedule input drift")
    rows: list[dict[str, Any]] = []
    offspring: list[np.ndarray] = []
    seen: set[str] = set()
    started = time.perf_counter()
    for parent_index, parent_hash in enumerate(parent_hashes):
        for operator_index, operator in enumerate(operators):
            attempt_index = parent_index * len(operators) + operator_index
            seed = cycle_child_seed(namespace, parent_hash, operator)
            result = apply_operator_with_config(operator, parent_xy[parent_index], np.random.default_rng(seed), profile)
            valid = result.proposal_status == "accepted" and result.valid_after_repair
            duplicate = bool(valid and result.final_output_hash in seen)
            rows.append({
                "replicate": replicate, "arm_scope": arm_scope, "generation": generation,
                "attempt_index": attempt_index, "parent_index": parent_index, "parent_hash": parent_hash,
                "operator_id": operator, "operator_index": operator_index, "child_seed": np.uint64(seed),
                "cycle_seed_namespace": namespace, "mutation_profile": PROFILE_NAME,
                "proposal_status": result.proposal_status, "failure_reason": result.failure_reason,
                "valid_before_repair": bool(result.valid_before_repair), "valid_after_repair": bool(result.valid_after_repair),
                "repair_count": int(result.repair_count), "repair_fraction": float(result.repair_fraction),
                "n_affected": int(result.n_affected), "raw_child_hash": result.raw_output_hash,
                "child_hash": result.final_output_hash, "within_offspring_duplicate": duplicate,
                **p2.scalar_trace(operator, dict(result.sampled_parameters)),
                **{key: float(value) for key, value in result.timing.items()},
            })
            if valid and not duplicate:
                seen.add(result.final_output_hash)
                offspring.append(np.asarray(result.final_coordinates, np.float64))
        if (parent_index + 1) % 1000 == 0:
            print(f"[P6-B][{replicate}][{arm_scope}][G{generation}] parents {parent_index+1}/10000", flush=True)
    manifest = pd.DataFrame(rows)
    coordinates = np.stack(offspring)
    accepted = manifest.proposal_status.eq("accepted") & manifest.valid_after_repair & ~manifest.within_offspring_duplicate
    hashes = manifest.loc[accepted, "child_hash"].astype(str).tolist()
    counts = manifest.operator_id.value_counts().sort_index().astype(int)
    if len(manifest) != ATTEMPTS_PER_GENERATION or not counts.eq(10_000).all() or hashes != [coordinate_hash(xy) for xy in coordinates]:
        raise RuntimeError("mutation budget/hash integrity failure")
    runtime = {
        "status": "PASS", "replicate": replicate, "arm_scope": arm_scope, "generation": generation,
        "cycle_seed_namespace": namespace, "nominal_attempts": len(manifest), "operator_attempts": counts.to_dict(),
        "valid_attempts": int((manifest.proposal_status.eq('accepted') & manifest.valid_after_repair).sum()),
        "invalid_attempts": int((~(manifest.proposal_status.eq('accepted') & manifest.valid_after_repair)).sum()),
        "valid_unique_within_generation": len(hashes), "within_offspring_duplicates": int(manifest.within_offspring_duplicate.sum()),
        "parent_membership_sha256": membership_hash(parent_hashes), "offspring_membership_sha256": membership_hash(hashes),
        "profile_sha256": sha256_file(PROFILE_PATH), "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "rng_rule": "cycle_child_seed(namespace,parent_hash,operator_id)",
    }
    save_bundle(bundle_path, coordinates, hashes)
    manifest.to_parquet(manifest_path, index=False)
    write_json(runtime_path, runtime)
    return coordinates, manifest, runtime


def representation_seed_cache() -> pd.DataFrame:
    source = pd.read_parquet(P6A / "diagnostics/c04_union_top5_representation.parquet")
    columns = ["instance_hash", *[f"raw__{f}" for f in TOP5], *[f"theory__{f}" for f in TOP5], *[f"phi__{f}" for f in TOP5]]
    return source[columns].drop_duplicates("instance_hash", keep="first")


def load_extracted_cache() -> pd.DataFrame:
    files = sorted(FEATURE_CACHE.glob("f40_*.parquet"))
    if not files: return pd.DataFrame(columns=["instance_hash", *TOP5])
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).drop_duplicates("instance_hash", keep="first")


def extract_missing_top5(tag: str, coordinates: np.ndarray, hashes: list[str], known_hashes: set[str]) -> pd.DataFrame:
    missing = [index for index, value in enumerate(hashes) if value not in known_hashes]
    if not missing: return load_extracted_cache()
    jobs = []
    parameters = p6a.scaler_parameters()
    rscript = shutil.which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript"
    for start in range(0, len(missing), FEATURE_SHARD_N):
        indices = missing[start:start + FEATURE_SHARD_N]
        block_hashes = [hashes[i] for i in indices]
        token = membership_hash(block_hashes)[:20]
        jobs.append({
            "coordinates": coordinates[np.asarray(indices, int)], "hashes": block_hashes,
            "names": list(TOP5), "out": str(FEATURE_CACHE / f"f40_{token}.parquet"),
            "rscript": rscript, "r_script": str(p2.R_EXTRACTOR), "params": parameters,
        })
    print(f"[P6-B][{tag}] Top5 extraction missing={len(missing)} shards={len(jobs)}", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(FEATURE_WORKERS, len(jobs))) as executor:
        futures = [executor.submit(p2.run_feature_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done == 1 or done % 25 == 0 or done == len(jobs):
                print(f"[P6-B][{tag}] feature shards {done}/{len(jobs)}", flush=True)
    write_json(FEATURES_DIR / f"{tag}_extraction_runtime.json", {
        "tag": tag, "missing_n": len(missing), "shards": len(jobs), "workers": FEATURE_WORKERS,
        "wall_seconds": time.perf_counter() - started, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    return load_extracted_cache()


def top5_representation(tag: str, coordinates: np.ndarray, hashes: list[str]) -> pd.DataFrame:
    output = FEATURES_DIR / f"{tag}_top5.parquet"
    if output.is_file():
        cached = pd.read_parquet(output)
        if cached.instance_hash.astype(str).tolist() == hashes: return cached
    seeded = representation_seed_cache()
    extracted = load_extracted_cache()
    known = set(seeded.instance_hash.astype(str)) | set(extracted.instance_hash.astype(str))
    extracted = extract_missing_top5(tag, coordinates, hashes, known)
    seeded_lookup = seeded.set_index("instance_hash")
    extracted_lookup = extracted.set_index("instance_hash")
    calibration = pd.read_parquet(P6A / "diagnostics/uniform_reference_top5.parquet")
    rows = []
    missing_indices = []
    for index, value in enumerate(hashes):
        if value in seeded_lookup.index:
            row = seeded_lookup.loc[value]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            rows.append({"instance_hash": value, **{column: row[column] for column in seeded.columns if column != "instance_hash"}})
        else:
            rows.append({"instance_hash": value}); missing_indices.append(index)
    frame = pd.DataFrame(rows)
    if missing_indices:
        missing_hashes = [hashes[i] for i in missing_indices]
        z = extracted_lookup.loc[missing_hashes, list(TOP5)].reset_index(drop=True)
        raw = p6a.inverse_f40(z, TOP5)
        diagonal = p6a.bounding_diagonal_batch(coordinates[np.asarray(missing_indices, int)])
        theory = p6a.theoretical_features(raw, np.full(len(raw), 100), diagonal)
        phi = p6a.conditional_percentiles(theory, np.full(len(raw), 100), calibration)
        for local, global_index in enumerate(missing_indices):
            for feature in TOP5:
                frame.at[global_index, f"raw__{feature}"] = raw.at[local, feature]
                frame.at[global_index, f"theory__{feature}"] = theory.at[local, feature]
                frame.at[global_index, f"phi__{feature}"] = phi.at[local, feature]
    phi_values = frame[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    if frame.instance_hash.astype(str).tolist() != hashes or not np.isfinite(phi_values).all():
        raise RuntimeError(f"Top5 representation failure: {tag}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return frame


def build_union(tag: str, generation: int, parent_xy: np.ndarray, parents: pd.DataFrame, offspring_xy: np.ndarray, mutation: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    manifest_path = output_dir / "candidate_union.parquet"
    bundle_path = output_dir / "candidate_union.npz"
    if manifest_path.is_file() and bundle_path.is_file():
        coordinates, hashes, _ = load_bundle(bundle_path)
        frame = pd.read_parquet(manifest_path)
        if hashes == frame.instance_hash.astype(str).tolist(): return coordinates, frame
        raise RuntimeError(f"union cache drift {tag}")
    accepted = mutation.proposal_status.eq("accepted") & mutation.valid_after_repair & ~mutation.within_offspring_duplicate
    child = mutation.loc[accepted].reset_index(drop=True).copy()
    if len(child) != len(offspring_xy): raise RuntimeError("offspring/manifest alignment")
    parent_hashes = set(parents.instance_hash.astype(str))
    child["parent_collision"] = child.child_hash.astype(str).isin(parent_hashes)
    keep = ~child.parent_collision.to_numpy(bool)
    child = child.loc[keep].reset_index(drop=True)
    child_xy = offspring_xy[keep]
    parent_lookup = parents.set_index("instance_hash")
    child["instance_hash"] = child.child_hash.astype(str)
    child["lineage_root"] = child.parent_hash.astype(str).map(parent_lookup.lineage_root.astype(str))
    child["direct_parent_hash"] = child.parent_hash.astype(str)
    child["generation_born"] = generation
    child["evolution_origin"] = f"G{generation}_offspring"
    child["node_count"] = 100
    union = pd.concat([parents, child], ignore_index=True, sort=False).drop_duplicates("instance_hash", keep="first")
    coordinate_map = {value: xy for value, xy in zip(parents.instance_hash.astype(str), parent_xy, strict=True)}
    coordinate_map.update({value: xy for value, xy in zip(child.instance_hash.astype(str), child_xy, strict=True)})
    union = union.sort_values("instance_hash", kind="stable").reset_index(drop=True)
    coordinates = np.stack([coordinate_map[value] for value in union.instance_hash.astype(str)])
    hashes = union.instance_hash.astype(str).tolist()
    representation = top5_representation(tag, coordinates, hashes)
    union = union.drop(columns=[column for column in representation.columns if column != "instance_hash" and column in union], errors="ignore")
    union = union.merge(representation, on="instance_hash", validate="one_to_one", sort=False)
    if hashes != union.instance_hash.astype(str).tolist(): raise RuntimeError("union feature alignment")
    output_dir.mkdir(parents=True, exist_ok=True)
    union.to_parquet(manifest_path, index=False)
    save_bundle(bundle_path, coordinates, hashes)
    write_json(output_dir / "candidate_union_integrity.json", {
        "status": "PASS", "generation": generation, "parent_n": len(parents),
        "accepted_unique_offspring_n": len(offspring_xy), "parent_child_duplicates": int((~keep).sum()),
        "eligible_unique_offspring_n": len(child), "union_unique_n": len(union),
        "parent_membership_sha256": membership_hash(parents.instance_hash.astype(str).tolist()),
        "union_membership_sha256": membership_hash(hashes), "bundle_sha256": sha256_file(bundle_path),
        "manifest_sha256": sha256_file(manifest_path),
    })
    return coordinates, union


def select_population(replicate: str, arm: str, generation: int, union_xy: np.ndarray, union: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    manifest_path = output_dir / "active_population.parquet"
    bundle_path = output_dir / "active_population.npz"
    integrity_path = output_dir / "selection_integrity.json"
    if manifest_path.is_file() and bundle_path.is_file() and integrity_path.is_file():
        coordinates, hashes, _ = load_bundle(bundle_path); frame = pd.read_parquet(manifest_path); info = json.loads(integrity_path.read_text())
        if hashes == frame.instance_hash.astype(str).tolist() and info.get("status") == "PASS": return coordinates, frame, info
        raise RuntimeError("selection cache drift")
    started = time.perf_counter()
    if arm == "E0":
        seed = SEEDS[replicate] + 1000 + generation
        rng = np.random.default_rng(seed)
        selected_indices = rng.choice(len(union), POPULATION_N, replace=False)
        references = np.full(POPULATION_N, -1, int); edge_cost = np.full(POPULATION_N, np.nan)
        solver = "uniform numpy choice without replacement over hash-sorted union"
        solver_status = 0; fractional = 0
    else:
        candidate_phi = union[[f"phi__{f}" for f in TOP5]].to_numpy(float)
        target = pd.read_parquet(P6A / "diagnostics/tsplib48_top5_representation.parquet")
        target_phi = target[[f"phi__{f}" for f in TOP5]].to_numpy(float)
        cost = p6a.squared_distance_matrix(candidate_phi, target_phi)
        result = p6a.exact_transport_lp(cost, QUOTA)
        selected_indices = result["candidate_index"]
        references = result["reference_index"]
        edge_cost = result["edge_cost"]
        solver = result["solver"]; solver_status = result["solver_status"]; fractional = result["fractional_variables"]
        if not np.array_equal(np.bincount(references, minlength=48), QUOTA): raise RuntimeError("E1 quota failure")
        del cost
    selected = union.iloc[selected_indices].copy().reset_index(drop=True)
    selected["selected_reference_index"] = references
    selected["selected_squared_cost"] = edge_cost
    selected["selection_component"] = "uniform_random" if arm == "E0" else "exact_top5_ot"
    coordinates = union_xy[np.asarray(selected_indices, int)]
    hashes = selected.instance_hash.astype(str).tolist()
    if len(hashes) != POPULATION_N or len(set(hashes)) != POPULATION_N: raise RuntimeError("survivor uniqueness failure")
    info = {
        "status": "PASS", "replicate": replicate, "arm": arm, "generation": generation,
        "union_n": len(union), "survivor_n": len(selected), "survivor_unique_n": len(set(hashes)),
        "selection_seed": SEEDS[replicate] + 1000 + generation if arm == "E0" else None,
        "solver": solver, "solver_status": solver_status, "fractional_variables": fractional,
        "quota_valid": True if arm == "E0" else bool(np.array_equal(np.bincount(references, minlength=48), QUOTA)),
        "objective_total": None if arm == "E0" else float(np.sum(edge_cost)),
        "objective_mean": None if arm == "E0" else float(np.mean(edge_cost)),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "union_membership_sha256": membership_hash(union.instance_hash.astype(str).tolist()),
        "survivor_membership_sha256": membership_hash(hashes),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(manifest_path, index=False); save_bundle(bundle_path, coordinates, hashes); write_json(integrity_path, info)
    return coordinates, selected, info


def evolve(replicate: str) -> dict[str, Any]:
    audit()
    if replicate not in REPLICATES: raise ValueError(replicate)
    completion = EVOLUTION / replicate / "completion.json"
    if completion.is_file() and json.loads(completion.read_text()).get("status") == "COMPLETE": return json.loads(completion.read_text())
    started = time.perf_counter()
    p0_xy, p0 = base_population()
    shared = shared_g1_dir(replicate)
    child_xy, mutation, g1_runtime = mutate(replicate, "SHARED", 1, p0_xy, p0, shared / "mutation")
    union_xy, union = build_union(f"{replicate}_shared_G1", 1, p0_xy, p0, child_xy, mutation, shared)
    population: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for arm in ARMS:
        directory = stage_dir(replicate, arm, 1)
        population[arm] = select_population(replicate, arm, 1, union_xy, union, directory)[:2]
    for arm in ARMS:
        parent_xy, parents = population[arm]
        directory = stage_dir(replicate, arm, 2)
        child_xy, mutation, _ = mutate(replicate, arm, 2, parent_xy, parents, directory / "mutation")
        union_xy, union = build_union(f"{replicate}_{arm}_G2", 2, parent_xy, parents, child_xy, mutation, directory)
        population[arm] = select_population(replicate, arm, 2, union_xy, union, directory)[:2]
        final_xy, final_frame = population[arm]
        final_hashes = final_frame.instance_hash.astype(str).tolist()
        final_frame.to_parquet(population_manifest(replicate, arm), index=False)
        save_bundle(population_bundle(replicate, arm), final_xy, final_hashes)
    # G1 is physically generated once but constitutes matched 110k attempts for each arm.
    e0_g1 = json.loads((stage_dir(replicate, "E0", 1) / "selection_integrity.json").read_text())
    e1_g1 = json.loads((stage_dir(replicate, "E1", 1) / "selection_integrity.json").read_text())
    if e0_g1["union_membership_sha256"] != e1_g1["union_membership_sha256"]: raise RuntimeError("G1 union mismatch")
    result = {
        "status": "COMPLETE", "replicate": replicate, "evolution_seed": SEEDS[replicate],
        "mutation_namespaces": NAMESPACES[replicate], "g1_shared_union": True,
        "conceptual_mutation_attempts": 2 * 2 * ATTEMPTS_PER_GENERATION,
        "physical_mutation_attempts": 3 * ATTEMPTS_PER_GENERATION,
        "wall_seconds": time.perf_counter() - started,
        "g1_union_membership_sha256": e0_g1["union_membership_sha256"],
        "final_population_sha256": {arm: sha256_file(population_bundle(replicate, arm)) for arm in ARMS},
    }
    write_json(completion, result)
    Status().update(phase="evolution", status="PASS", replicate=replicate)
    return result


def fixed_swd(source: np.ndarray, target: np.ndarray, projections: int = 256) -> float:
    rng = np.random.default_rng(20261024 + 771)
    directions = rng.normal(size=(projections, source.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(source @ direction, target @ direction) for direction in directions]))


def fixed_energy(source: np.ndarray, target: np.ndarray, sample_n: int = 2000) -> float:
    rng = np.random.default_rng(20261024 + 772)
    sx = source[rng.choice(len(source), min(sample_n, len(source)), replace=False)]
    tx = target[rng.choice(len(target), min(sample_n, len(target)), replace=False)]
    return float(2 * cdist(sx, tx).mean() - cdist(sx, sx).mean() - cdist(tx, tx).mean())


def structural_metric(replicate: str, arm: str, stage: str, object_type: str, frame: pd.DataFrame) -> dict[str, Any]:
    source = frame[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    target_frame = pd.read_parquet(P6A / "diagnostics/tsplib48_top5_representation.parquet")
    target = target_frame[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    target_mass = target[np.repeat(np.arange(48), QUOTA)]
    cost = p6a.squared_distance_matrix(source, target)
    started = time.perf_counter(); match = p6a.exact_transport_lp(cost, QUOTA); matching_wall = time.perf_counter() - started
    nearest = np.sqrt(cost.min(axis=1))
    marginal = [wasserstein_distance(source[:, index], target_mass[:, index]) for index in range(5)]
    lineages = frame.lineage_root.astype(str) if "lineage_root" in frame else frame.instance_hash.astype(str)
    probability = lineages.value_counts(normalize=True).to_numpy(float)
    row = {
        "replicate": replicate, "arm": arm, "stage": stage, "object_type": object_type,
        "n": len(frame), "unique_n": int(frame.instance_hash.astype(str).nunique()),
        "membership_sha256": membership_hash(frame.instance_hash.astype(str).tolist()),
        "exact_ot_total_cost": float(match["edge_cost"].sum()), "exact_ot_mean_cost": float(match["edge_cost"].mean()),
        "solver_status": match["solver_status"], "fractional_variables": match["fractional_variables"],
        "quota_valid": bool(np.array_equal(np.bincount(match["reference_index"], minlength=48), QUOTA)),
        "matching_wall_seconds": matching_wall, "swd_5d": fixed_swd(source, target_mass),
        "energy_distance_5d": fixed_energy(source, target_mass),
        "mean_nearest_reference_distance": float(nearest.mean()), "p90_nearest_reference_distance": float(np.quantile(nearest, .9)),
        "mean_marginal_w1": float(np.mean(marginal)), "max_marginal_w1": float(np.max(marginal)),
        "effective_lineages": float(np.exp(-(probability * np.log(probability)).sum())),
        "maximum_lineage_share": float(probability.max()),
    }
    for index, feature in enumerate(TOP5):
        row[f"marginal_w1__{feature}"] = marginal[index]
        row[f"phi_median__{feature}"] = float(np.median(source[:, index]))
    del cost; gc.collect()
    return row


def collect_evolution_audits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for replicate in REPLICATES:
        complete = json.loads((EVOLUTION / replicate / "completion.json").read_text())
        if complete.get("status") != "COMPLETE": raise RuntimeError(f"evolution incomplete: {replicate}")
    mutation_rows = []; integrity_rows = []; run_rows = []
    for replicate in REPLICATES:
        p0_xy, p0 = base_population(); previous = {arm: p0 for arm in ARMS}
        for generation in GENERATIONS:
            for arm in ARMS:
                directory = shared_g1_dir(replicate) if generation == 1 else stage_dir(replicate, arm, generation)
                mutation_dir = directory / "mutation"
                mutation = pd.read_parquet(mutation_dir / "mutation_manifest.parquet")
                runtime = json.loads((mutation_dir / "runtime.json").read_text())
                union_dir = shared_g1_dir(replicate) if generation == 1 else stage_dir(replicate, arm, generation)
                union = pd.read_parquet(union_dir / "candidate_union.parquet")
                survivors = pd.read_parquet(stage_dir(replicate, arm, generation) / "active_population.parquet")
                selection = json.loads((stage_dir(replicate, arm, generation) / "selection_integrity.json").read_text())
                parent_hashes = set(previous[arm].instance_hash.astype(str))
                accepted = mutation.proposal_status.eq("accepted") & mutation.valid_after_repair
                for operator, part in mutation.groupby("operator_id", sort=True):
                    valid = part.proposal_status.eq("accepted") & part.valid_after_repair
                    unique = part.loc[valid & ~part.within_offspring_duplicate, "child_hash"].astype(str)
                    mutation_rows.append({
                        "replicate": replicate, "arm": arm, "generation": f"G{generation}", "operator_id": operator,
                        "g1_physically_shared": generation == 1, "attempts": len(part), "valid_attempts": int(valid.sum()),
                        "invalid_attempts": int((~valid).sum()), "unique_offspring": int(unique.nunique()),
                        "within_offspring_duplicates": int(part.loc[valid, "within_offspring_duplicate"].sum()),
                        "parent_child_duplicates": int(unique.isin(parent_hashes).sum()),
                        "current_generation_survivor_contribution": int(((survivors.generation_born == generation) & survivors.operator_id.astype(str).eq(str(operator))).sum()),
                    })
                current_hashes = survivors.instance_hash.astype(str).tolist()
                integrity_rows.append({
                    "replicate": replicate, "arm": arm, "generation": f"G{generation}",
                    "attempts": runtime["nominal_attempts"], "valid_offspring": runtime["valid_attempts"],
                    "invalid_offspring": runtime["invalid_attempts"], "unique_offspring": runtime["valid_unique_within_generation"],
                    "union_unique_n": len(union), "survivor_n": len(survivors), "survivor_unique_n": survivors.instance_hash.nunique(),
                    "all_survivors_in_union": set(current_hashes).issubset(set(union.instance_hash.astype(str))),
                    "parent_membership_sha256": runtime["parent_membership_sha256"],
                    "expected_parent_membership_sha256": membership_hash(previous[arm].instance_hash.astype(str).tolist()),
                    "survivor_membership_sha256": membership_hash(current_hashes),
                    "solver_status": selection["solver_status"], "fractional_variables": selection["fractional_variables"],
                    "quota_valid": selection["quota_valid"], "selection_wall_seconds": selection["wall_seconds"],
                    "mutation_wall_seconds": runtime["wall_seconds"],
                })
                previous[arm] = survivors
        g1e0 = pd.read_parquet(stage_dir(replicate, "E0", 1) / "active_population.parquet")
        g1e1 = pd.read_parquet(stage_dir(replicate, "E1", 1) / "active_population.parquet")
        g1union = json.loads((shared_g1_dir(replicate) / "candidate_union_integrity.json").read_text())
        for arm in ARMS:
            xy, hashes, _ = load_bundle(population_bundle(replicate, arm))
            manifest = pd.read_parquet(population_manifest(replicate, arm))
            valid = coordinate_audit(xy, hashes)
            if hashes != manifest.instance_hash.astype(str).tolist() or len(hashes) != 10_000 or valid["unique_hashes"] != 10_000 or not all(valid[k] for k in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")):
                raise RuntimeError(f"final population integrity failure: {replicate}/{arm}")
            run_rows.append({
                "replicate": replicate, "arm": arm, "evolution_seed": SEEDS[replicate],
                "namespace_g1": NAMESPACES[replicate][1], "namespace_g2": NAMESPACES[replicate][2],
                "g1_shared_union_membership_sha256": g1union["union_membership_sha256"],
                "g1_e0_e1_union_identical": True,
                "p1_membership_sha256": membership_hash((g1e0 if arm == "E0" else g1e1).instance_hash.astype(str).tolist()),
                "p2_membership_sha256": membership_hash(hashes), "population_n": len(hashes),
                "bundle_sha256": sha256_file(population_bundle(replicate, arm)),
                "manifest_sha256": sha256_file(population_manifest(replicate, arm)), "status": "PASS",
            })
    mutation_frame = pd.DataFrame(mutation_rows); integrity_frame = pd.DataFrame(integrity_rows); runs = pd.DataFrame(run_rows)
    if not integrity_frame.attempts.eq(ATTEMPTS_PER_GENERATION).all(): raise RuntimeError("attempt budget mismatch")
    if not integrity_frame.parent_membership_sha256.eq(integrity_frame.expected_parent_membership_sha256).all(): raise RuntimeError("G2 parent propagation failure")
    if not integrity_frame[["all_survivors_in_union", "quota_valid"]].astype(bool).all().all(): raise RuntimeError("survivor integrity failure")
    mutation_frame.to_csv(OUT / "02_mutation_operator_statistics.csv", index=False)
    runs.to_csv(OUT / "05_evolution_runs_manifest.csv", index=False)
    return mutation_frame, integrity_frame, runs


def structural_trajectory() -> pd.DataFrame:
    output = OUT / "04_population_structural_trajectory.csv"
    cache_path = FEATURES_DIR / "structural_metric_cache.csv"
    existing = pd.read_csv(cache_path) if cache_path.is_file() else pd.DataFrame()
    rows = [] if existing.empty else existing.to_dict("records")
    done = {(row["replicate"], row["arm"], row["stage"], row["object_type"], row["membership_sha256"]) for row in rows}
    p0_xy, p0 = base_population(); p0_rep = top5_representation("P0", p0_xy, p0.instance_hash.astype(str).tolist())
    p0 = p0.drop(columns=[c for c in p0_rep if c != "instance_hash" and c in p0], errors="ignore").merge(p0_rep, on="instance_hash", validate="one_to_one", sort=False)
    objects: list[tuple[str, str, str, str, pd.DataFrame]] = []
    for replicate in REPLICATES:
        for arm in ARMS: objects.append((replicate, arm, "P0", "population", p0))
        g1_union = pd.read_parquet(shared_g1_dir(replicate) / "candidate_union.parquet")
        for arm in ARMS:
            objects.append((replicate, arm, "G1", "candidate_union", g1_union))
            objects.append((replicate, arm, "P1", "population", pd.read_parquet(stage_dir(replicate, arm, 1) / "active_population.parquet")))
            objects.append((replicate, arm, "G2", "candidate_union", pd.read_parquet(stage_dir(replicate, arm, 2) / "candidate_union.parquet")))
            objects.append((replicate, arm, "P2", "population", pd.read_parquet(stage_dir(replicate, arm, 2) / "active_population.parquet")))
    for replicate, arm, stage, object_type, frame in objects:
        digest = membership_hash(frame.instance_hash.astype(str).tolist())
        key = (replicate, arm, stage, object_type, digest)
        if key in done: continue
        reusable = next((row for row in rows if row["membership_sha256"] == digest and row["object_type"] == object_type), None)
        if reusable is not None:
            cloned = dict(reusable); cloned.update({"replicate": replicate, "arm": arm, "stage": stage})
            rows.append(cloned); done.add(key); pd.DataFrame(rows).to_csv(cache_path, index=False); continue
        print(f"[P6-B][metrics] {replicate}/{arm}/{stage}/{object_type} n={len(frame)}", flush=True)
        rows.append(structural_metric(replicate, arm, stage, object_type, frame))
        pd.DataFrame(rows).to_csv(cache_path, index=False)
    result = pd.DataFrame(rows).drop_duplicates(["replicate", "arm", "stage", "object_type"], keep="last")
    result.to_csv(output, index=False)
    return result


def prepare() -> dict[str, Any]:
    audit(); mutation, integrity, runs = collect_evolution_audits(); trajectory = structural_trajectory()
    conceptual = int(integrity.attempts.sum())
    if conceptual != 1_320_000: raise RuntimeError(f"conceptual attempt total {conceptual}")
    lines = [
        "# Mutation Fairness Audit", "",
        f"All 12 arm-generation cells contain exactly 110,000 attempts (conceptual total {conceptual:,}). Each of the 11 frozen operators is attempted once for every parent.",
        "G1 is physically generated once per replicate and reused read-only by E0/E1; union membership SHA is identical within every pair. G2 uses the arm-specific P1 and the same identity-keyed RNG rule.",
        "No invalid-attempt compensation, archive, lineage cap, novelty exclusion, hardness, or diversity objective is used.", "",
        f"Physical mutation work is 990,000 attempts because each paired G1 is shared. Operator details are in `02_mutation_operator_statistics.csv`.",
    ]
    (OUT / "02_mutation_fairness_audit.md").write_text("\n".join(lines) + "\n")
    integrity.to_csv(FEATURES_DIR / "survivor_selection_integrity.csv", index=False)
    (OUT / "03_survivor_selection_integrity.md").write_text("\n".join([
        "# Survivor Selection Integrity", "",
        "All 12 stages pass union membership, 10,000-unique survivor, previous-P1-as-G2-parent, final-P2 identity, coordinate-hash, and provenance checks.",
        "E0 samples uniformly without replacement from a hash-sorted union with the frozen seed. E1 uses the unmodified P6-A float32 squared-distance matrix and exact HiGHS dual-simplex TU formulation; every E1 solve is optimal, integral, and satisfies all 48 quotas.",
        "Parent records win parent/child hash collisions. No greedy fallback occurred.", "",
        f"Integrity rows: {len(integrity)}; final populations: {len(runs)}; status: **PASS**.",
    ]) + "\n")
    populations = trajectory[(trajectory.object_type == "population") & trajectory.stage.isin(["P0", "P1", "P2"])]
    summary = populations.pivot_table(index=["replicate", "stage"], columns="arm", values=["exact_ot_mean_cost", "swd_5d", "energy_distance_5d"]).reset_index()
    (OUT / "04_population_structural_analysis.md").write_text("# Population Structural Analysis\n\nThe full 10k populations are compared with the same frozen TSPLIB48 mass, Top5 transform, quotas, exact OT cost, SWD, energy distance, and nearest-reference diagnostic. OT—not nearest-reference distance—is the H1 primary structural endpoint.\n\n```text\n" + summary.to_string(index=False) + "\n```\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for replicate, style in zip(REPLICATES, ("-", "--", ":"), strict=True):
        for arm, color in (("E0", "#4c78a8"), ("E1", "#f58518")):
            part = populations[(populations.replicate == replicate) & (populations.arm == arm)].copy()
            part["order"] = part.stage.map({"P0": 0, "P1": 1, "P2": 2}); part = part.sort_values("order")
            for ax, metric in zip(axes, ("exact_ot_mean_cost", "swd_5d", "energy_distance_5d"), strict=True):
                ax.plot(part.stage, part[metric], marker="o", linestyle=style, color=color, label=f"{replicate}-{arm}")
    for ax, title in zip(axes, ("Exact OT mean cost", "5D SWD", "Energy distance"), strict=True): ax.set_title(title); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "01_evolutionary_distribution_alignment.png", dpi=180); plt.close(fig)
    Status().update(phase="prepare", status="PASS")
    return {"status": "PASS", "conceptual_attempts": conceptual, "final_populations": len(runs), "trajectory_rows": len(trajectory)}


def verified_label_cache() -> pd.DataFrame:
    parts = [p5.p5_cache()]
    p61_global = P61 / "labels/global_unique_union/ledger.parquet"
    if p61_global.is_file(): parts.append(pd.read_parquet(p61_global))
    frame = pd.concat([part[list(ORACLE_LEDGER_COLUMNS)] for part in parts], ignore_index=True).drop_duplicates("instance_hash", keep="last")
    if not frame.verified.astype(bool).all() or not frame.status.eq("solved").all(): raise RuntimeError("invalid reusable label cache")
    if set(frame.binary_hash.astype(str)) != {CANONICAL_CONCORDE_SHA256}: raise RuntimeError("Concorde binary cache mismatch")
    return frame


def validate_ledger(coordinates: np.ndarray, hashes: list[str], ledger: pd.DataFrame) -> np.ndarray:
    import run_blade_v2 as base
    tours = base._oracle_tours(ledger, hashes)
    if len(ledger) != len(hashes) or tours.shape != (len(hashes), 100): raise RuntimeError("label alignment failure")
    if not ledger.verified.astype(bool).all() or not ledger.status.eq("solved").all(): raise RuntimeError("unverified label")
    if not all(validate_tour_0based(tour, 100)[0] for tour in tours): raise RuntimeError("illegal expert tour")
    costs = np.asarray([tour_cost_int(integer_distance_matrix(xy), tour) for xy, tour in zip(coordinates, tours, strict=True)], np.int64)
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)): raise RuntimeError("expert objective mismatch")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (len(hashes), 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]): raise RuntimeError("teacher tour closure mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist(): raise RuntimeError("tour hash mismatch")
    return tours


def label() -> dict[str, Any]:
    prepare()
    prior = verified_label_cache(); known = set(prior.instance_hash.astype(str))
    coordinate_map: dict[str, np.ndarray] = {}; union_order: list[str] = []
    for replicate in REPLICATES:
        for arm in ARMS:
            coordinates, hashes, _ = load_bundle(population_bundle(replicate, arm))
            for value, xy in zip(hashes, coordinates, strict=True):
                if value not in coordinate_map: coordinate_map[value] = xy; union_order.append(value)
                elif not np.array_equal(coordinate_map[value], xy): raise RuntimeError(f"coordinate hash collision {value}")
    union_xy = np.stack([coordinate_map[value] for value in union_order])
    missing_n = sum(value not in known for value in union_order)
    global_dir = LABELS / "global_unique_union"; integrity_path = global_dir / "labeling_integrity.json"
    started = time.perf_counter()
    if integrity_path.is_file():
        info = json.loads(integrity_path.read_text()); ledger = pd.read_parquet(global_dir / "ledger.parquet")
        if info.get("status") != "PASS" or info.get("union_membership_sha256") != membership_hash(union_order): raise RuntimeError("global label cache drift")
    else:
        import run_blade_v2 as base
        ledger = base.label_with_cache(union_xy, union_order, global_dir, prior)
        ledger.to_parquet(global_dir / "ledger.parquet", index=False)
        info = {
            "status": "PASS", "total_memberships": 60_000, "unique_union_n": len(union_order),
            "reused_labels": len(union_order) - missing_n, "physical_new_concorde_calls": missing_n,
            "failures": 0, "pipeline_wall_seconds": time.perf_counter() - started,
            "union_membership_sha256": membership_hash(union_order),
            "ledger_sha256": sha256_file(global_dir / "ledger.parquet"), "concorde_sha256": CANONICAL_CONCORDE_SHA256,
        }
        write_json(integrity_path, info)
    if len(ledger) != len(union_order) or set(ledger.instance_hash.astype(str)) != set(union_order): raise RuntimeError("label union incomplete")
    lookup = ledger.set_index("instance_hash", drop=False)
    rows = []
    for replicate in REPLICATES:
        for arm in ARMS:
            directory = label_dir(replicate, arm); directory.mkdir(parents=True, exist_ok=True)
            coordinates, hashes, _ = load_bundle(population_bundle(replicate, arm))
            selected = lookup.loc[hashes].reset_index(drop=True)
            tours = validate_ledger(coordinates, hashes, selected)
            selected.to_parquet(directory / "ledger.parquet", index=False)
            manifest = pd.read_parquet(population_manifest(replicate, arm))
            payload = set(ORACLE_LEDGER_COLUMNS[1:])
            labeled = manifest[[c for c in manifest.columns if c not in payload]].merge(selected, on="instance_hash", validate="one_to_one")
            labeled.to_parquet(directory / "labeled.parquet", index=False)
            save_bundle(directory / "labeled_bundle.npz", coordinates, hashes, tours)
            record = {
                "status": "PASS", "replicate": replicate, "arm": arm, "verified": 10_000,
                "cache_reuse_n": sum(value in known for value in hashes), "missing_before_global_solve_n": sum(value not in known for value in hashes),
                "failures": 0, "training_membership_sha256": membership_hash(hashes),
                "ledger_sha256": sha256_file(directory / "ledger.parquet"),
                "labeled_bundle_sha256": sha256_file(directory / "labeled_bundle.npz"),
                "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
                "all_tours_legal": True, "all_costs_reproduced": True, "concorde_sha256": CANONICAL_CONCORDE_SHA256,
            }
            write_json(directory / "labeling_integrity.json", record); rows.append(record)
    lines = ["# Concorde Label Audit", "", "| Dataset | Reused | Missing before union solve | Verified | Label bundle SHA256 |", "|---|---:|---:|---:|---|"]
    for row in rows: lines.append(f"| {row['replicate']}-{row['arm']} | {row['cache_reuse_n']:,} | {row['missing_before_global_solve_n']:,} | {row['verified']:,} | `{row['labeled_bundle_sha256']}` |")
    lines.extend(["", f"Total memberships 60,000; global unique coordinates {len(union_order):,}; reused labels {len(union_order)-missing_n:,}; newly computed exact Concorde labels {missing_n:,}; failures 0.", f"Global labeling wall-clock {float(info['pipeline_wall_seconds']):.3f}s; Concorde SHA256 `{CANONICAL_CONCORDE_SHA256}`.", "All tours are legal 0-based Hamiltonian permutations with implicit closure; integer objectives and BQ closed-coordinate inputs were independently reproduced. No instance was replaced."])
    (OUT / "06_concorde_label_audit.md").write_text("\n".join(lines) + "\n")
    Status().update(phase="label", status="PASS", new_unique_labels=missing_n)
    return info


def train(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    protocol = protocol_gate(); integrity_path = run_dir(replicate, arm) / "training_integrity.json"; checkpoint = checkpoint_path(replicate, arm)
    if integrity_path.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and checkpoint.is_file() and sha256_file(checkpoint) == existing.get("checkpoint_sha256"): return existing
    label_info = json.loads((label_dir(replicate, arm) / "labeling_integrity.json").read_text())
    if label_info.get("verified") != 10_000: raise RuntimeError("labeling gate failed")
    coordinates, hashes, tours = load_bundle(label_dir(replicate, arm) / "labeled_bundle.npz", True); assert tours is not None
    frozen_xy, frozen_hashes, _ = load_bundle(population_bundle(replicate, arm))
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy): raise RuntimeError("training membership drift")
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    device = torch.device(device_name); directory = run_dir(replicate, arm) / "training_cycle"
    core.RUN_ID = f"TARGET_AWARE_P6B_{replicate}_{arm}"
    core.cycle_dir = lambda selected_arm, cycle: directory
    core.model_path = lambda selected_arm, index: checkpoint if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    config = {
        "replicate": replicate, "arm": arm, "evolution_seed": SEEDS[replicate], "training_seed": TRAINING_SEED,
        "membership_sha256": membership_hash(hashes), "dataset_bundle_sha256": sha256_file(population_bundle(replicate, arm)),
        "label_bundle_sha256": label_info["labeled_bundle_sha256"], "protocol_sha256": protocol["protocol_sha256"],
        "base_training_protocol": base_training_protocol(),
    }
    config["training_config_sha256"] = canonical_hash(config); write_json(run_dir(replicate, arm) / "training_config.json", config)
    started = time.perf_counter()
    trained, meta, timing = core.train(f"P6B_{replicate}_{arm}", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)
    invocation = time.perf_counter() - started; stage = directory / "09_bq_training"
    e000 = torch.load(stage / "M5_E000.tsp.best", map_location="cpu", weights_only=False); m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    curve = pd.read_csv(stage / "training_loss_curve.csv")
    if not e000_equal or len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(10_000).all() or meta["sha256"] != sha256_file(checkpoint) or not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters()):
        raise RuntimeError("training integrity failure")
    result = {
        "status": "PASS", **config, "m4_sha256": M4_SHA, "e000_weights_equal_m4": True,
        "epochs": 256, "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": invocation,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]), "checkpoint_sha256": meta["sha256"],
        "checkpoint_path": str(checkpoint), "loss_curve_sha256": sha256_file(stage / "training_loss_curve.csv"),
    }
    write_json(integrity_path, result); del trained, model
    if device.type == "cuda": torch.cuda.empty_cache()
    write_training_manifest(); return result


def write_training_manifest() -> pd.DataFrame:
    rows = []
    for replicate in REPLICATES:
        for arm in ARMS:
            path = run_dir(replicate, arm) / "training_integrity.json"
            if path.is_file():
                info = json.loads(path.read_text())
                rows.append({key: info.get(key) for key in ("replicate", "arm", "status", "evolution_seed", "training_seed", "membership_sha256", "dataset_bundle_sha256", "label_bundle_sha256", "m4_sha256", "e000_weights_equal_m4", "epochs", "optimizer_steps", "final_loss", "training_loop_wall_s", "gpu_peak_allocated_bytes", "checkpoint_sha256", "checkpoint_path", "training_config_sha256")})
            else: rows.append({"replicate": replicate, "arm": arm, "status": "PENDING", "training_seed": TRAINING_SEED})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "07_bq_training_manifest.csv", index=False)
    passed = frame.status.eq("PASS").sum()
    (OUT / "07_bq_training_integrity.md").write_text(f"# BQ Training Integrity\n\nCompleted frozen M4→M5 runs: {passed}/6. Every completed run checks E000 tensor equality with M4, 256 epochs, 10,240 optimizer updates, finite weights, fixed seed 2026, and fixed final-E256 checkpoint selection.\n")
    return frame


def evaluate(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    integrity_path = run_dir(replicate, arm) / "evaluation_integrity.json"
    if integrity_path.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and eval_csv(replicate, arm).is_file(): return existing
    training = json.loads((run_dir(replicate, arm) / "training_integrity.json").read_text()); checkpoint = checkpoint_path(replicate, arm)
    if training["checkpoint_sha256"] != sha256_file(checkpoint): raise RuntimeError("checkpoint drift")
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    device = torch.device(device_name); model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=training["checkpoint_sha256"])
    frame, wall = core.evaluate(model, checkpoint, eval_label(replicate, arm), eval_csv(replicate, arm), Status(), f"P6B_{replicate}_{arm}")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all(): raise RuntimeError("evaluation incomplete")
    historical_path = p5.ORIGINAL[arm]["eval_csv"] if replicate == "Original" else p5.eval_csv(replicate, arm)
    historical = pd.read_csv(historical_path).set_index("dataset"); current = frame.set_index("dataset")
    for dataset in DATASETS:
        if current.at[dataset, "dataset_sha256"] != historical.at[dataset, "dataset_sha256"] or current.at[dataset, "evaluator_config_sha256"] != historical.at[dataset, "evaluator_config_sha256"]:
            raise RuntimeError(f"evaluation protocol drift: {dataset}")
    result = {"status": "PASS", "replicate": replicate, "arm": arm, "checkpoint_sha256": meta["sha256"], "datasets": list(DATASETS), "all_tours_valid": True, "evaluation_wall_s": wall, "results_sha256": sha256_file(eval_csv(replicate, arm))}
    write_json(integrity_path, result); del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def detail(replicate: str, arm: str, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(replicate, arm) / "evaluation_work" / eval_label(replicate, arm) / dataset / "instance_metrics.parquet")


def paired_frame(replicate: str, dataset: str) -> pd.DataFrame:
    pieces = []
    for arm in ARMS:
        frame = detail(replicate, arm, dataset)[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
        if not frame.valid_tour.all(): raise RuntimeError("invalid evaluation tour")
        pieces.append(frame.rename(columns={"coordinate_hash": f"coordinate_hash_{arm}", "n_nodes": f"n_nodes_{arm}", "optimal_cost_int": f"optimal_cost_int_{arm}", "gap_percent": f"gap_{arm}"}).drop(columns="valid_tour"))
    merged = pieces[0].merge(pieces[1], on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        if not merged[f"{field}_E0"].astype(str).eq(merged[f"{field}_E1"].astype(str)).all(): raise RuntimeError(f"paired dataset drift {replicate}/{dataset}/{field}")
    return pd.DataFrame({
        "replicate": replicate, "dataset": dataset, "instance_id": merged.instance_id.astype(str),
        "coordinate_hash": merged.coordinate_hash_E0.astype(str), "n_nodes": merged.n_nodes_E0.astype(int),
        "optimal_cost_int": merged.optimal_cost_int_E0.astype(np.int64),
        "gap_E0": merged.gap_E0.astype(float), "gap_E1": merged.gap_E1.astype(float),
        "Delta_E0_minus_E1": merged.gap_E0.astype(float) - merged.gap_E1.astype(float),
    })


def bootstrap(values: np.ndarray, key: str) -> tuple[float, float]:
    return p4b1.paired_bootstrap(np.asarray(values, float), p4b1.stable_seed(f"P6B|{key}"))


def gap5_bootstrap(frames: dict[str, pd.DataFrame], replicate: str) -> tuple[float, float]:
    rng = np.random.default_rng(p4b1.stable_seed(f"P6B|{replicate}|GAP5")); estimates = np.empty(BOOTSTRAP_N)
    values = [frames[name].Delta_E0_minus_E1.to_numpy(float) for name in GAP5]
    for index in range(BOOTSTRAP_N): estimates[index] = np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in values])
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def historical_reference_report(results: pd.DataFrame) -> None:
    p5_result = pd.read_csv(P5 / "08_original70_tsplib22_gap5_results.csv")
    p61_result = pd.read_csv(P61 / "05_original70_tsplib22_gap5_results.csv")
    lines = [
        "# Historical Reference Comparison", "",
        "Historical values are descriptive references only; P6-B's causal comparison is its internal paired E0/E1 design.", "",
        "| Replicate | P5 old E1 Original70 | P6-1 old E1 Population Original70 | P6-B E1 Original70 |",
        "|---|---:|---:|---:|",
    ]
    for replicate in REPLICATES:
        old_archive = float(p5_result[(p5_result.replicate == replicate) & (p5_result.dataset == "OriginalTSPLIB70")].mean_gap_E1.iloc[0])
        old_pop = float(p61_result[(p61_result.replicate == replicate) & (p61_result.dataset == "OriginalTSPLIB70")].mean_gap_E1_Pop.iloc[0])
        new = float(results[(results.replicate == replicate) & (results.arm == "E1") & (results.dataset == "OriginalTSPLIB70")].mean_gap.iloc[0])
        lines.append(f"| {replicate} | {old_archive:.6f} | {old_pop:.6f} | {new:.6f} |")
    history = json.loads((ROOT / "target_aware_pilot/p6ae_evoreal50k_top5_alignment/cache/historical_results.json").read_text())
    lines.extend([
        "", f"EvoReal50k historical M5: Original70 {history['evoreal_original70']:.6f}%, GAP5 {history['evoreal_gap5']:.6f}%.",
        "EvoReal uses 50k across a different five-cycle adaptation path and is not a strict budget-matched causal comparator. P5/P6-1 also differ in archive/acquisition and mutation budget, so no single P6-B component is isolated by these historical contrasts.",
    ])
    (OUT / "10_historical_reference_comparison.md").write_text("\n".join(lines) + "\n")


def runtime_report() -> None:
    evolution = [json.loads((EVOLUTION / replicate / "completion.json").read_text()) for replicate in REPLICATES]
    label_info = json.loads((LABELS / "global_unique_union/labeling_integrity.json").read_text())
    training = [json.loads((run_dir(rep, arm) / "training_integrity.json").read_text()) for rep in REPLICATES for arm in ARMS]
    evaluation = [json.loads((run_dir(rep, arm) / "evaluation_integrity.json").read_text()) for rep in REPLICATES for arm in ARMS]
    feature_files = list(FEATURES_DIR.glob("*_extraction_runtime.json")); feature_runtime = [json.loads(path.read_text()) for path in feature_files]
    mutation_runtime = []
    for rep in REPLICATES:
        mutation_runtime.append(json.loads((shared_g1_dir(rep) / "mutation/runtime.json").read_text()))
        for arm in ARMS: mutation_runtime.append(json.loads((stage_dir(rep, arm, 2) / "mutation/runtime.json").read_text()))
    lines = [
        "# Runtime and Compute Audit", "",
        f"- Evolution end-to-end wall (sum of replicate invocations): {sum(x['wall_seconds'] for x in evolution):.1f}s.",
        f"- Physical mutation wall (9 batches, 990k attempts): {sum(x['wall_seconds'] for x in mutation_runtime):.1f}s; conceptual paired budget 1,320,000 attempts.",
        f"- New Top5 extraction wall (parallel shard invocation sums): {sum(x['wall_seconds'] for x in feature_runtime):.1f}s.",
        f"- Concorde labeling wall: {label_info['pipeline_wall_seconds']:.1f}s; new calls {label_info['physical_new_concorde_calls']:,}.",
        f"- BQ training-loop wall sum: {sum(x['training_loop_wall_s'] for x in training):.1f}s; maximum allocated GPU memory {max(x['gpu_peak_allocated_bytes'] for x in training)/(1024**3):.2f} GiB.",
        f"- Greedy evaluation wall sum: {sum(x['evaluation_wall_s'] for x in evaluation):.1f}s.",
        f"- Maximum recorded evolution RSS: {max(x['peak_rss_mb'] for x in mutation_runtime):.1f} MiB.",
        "- Mutation/F40/HiGHS run on CPU; BQ training and greedy evaluation use the declared CUDA devices. All stages are independently resumable.",
    ]
    (OUT / "12_runtime_and_compute_audit.md").write_text("\n".join(lines) + "\n")


def finalize() -> dict[str, Any]:
    protocol_gate(); write_training_manifest()
    for replicate in REPLICATES:
        for arm in ARMS:
            if json.loads((run_dir(replicate, arm) / "training_integrity.json").read_text()).get("status") != "PASS": raise RuntimeError("training incomplete")
            if json.loads((run_dir(replicate, arm) / "evaluation_integrity.json").read_text()).get("status") != "PASS": raise RuntimeError("evaluation incomplete")
    heldout = set(json.loads((ROOT / "target_aware_pilot/p1_heldout22_names.json").read_text())["names"])
    development = set(pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str))
    paired_parts = []; result_rows = []
    for replicate in REPLICATES:
        frames: dict[str, pd.DataFrame] = {}
        for dataset in DATASETS:
            frame = paired_frame(replicate, dataset); frames[dataset] = frame; paired_parts.append(frame)
            low, high = bootstrap(frame.Delta_E0_minus_E1.to_numpy(float), f"{replicate}|{dataset}")
            for arm in ARMS: result_rows.append({"replicate": replicate, "arm": arm, "dataset": dataset, "n_instances": len(frame), "mean_gap": float(frame[f'gap_{arm}'].mean())})
        original = frames["OriginalTSPLIB70"]
        for name, names in (("TSPLIB48", development), ("TSPLIB22", heldout)):
            subset = original[original.instance_id.isin(names)].copy(); subset["dataset"] = name
            if len(subset) != len(names): raise RuntimeError(f"{name} split failure")
            frames[name] = subset; paired_parts.append(subset)
            for arm in ARMS: result_rows.append({"replicate": replicate, "arm": arm, "dataset": name, "n_instances": len(subset), "mean_gap": float(subset[f'gap_{arm}'].mean())})
        gap5_e0 = float(np.mean([frames[name].gap_E0.mean() for name in GAP5])); gap5_e1 = float(np.mean([frames[name].gap_E1.mean() for name in GAP5]))
        result_rows.extend([{"replicate": replicate, "arm": "E0", "dataset": "GAP5", "n_instances": sum(len(frames[n]) for n in GAP5), "mean_gap": gap5_e0}, {"replicate": replicate, "arm": "E1", "dataset": "GAP5", "n_instances": sum(len(frames[n]) for n in GAP5), "mean_gap": gap5_e1}])
    per_instance = pd.concat(paired_parts, ignore_index=True); per_instance.to_csv(OUT / "08_per_instance_results.csv", index=False)
    results = pd.DataFrame(result_rows); results.to_csv(OUT / "08_greedy_evaluation_results.csv", index=False)
    comparison_rows = []
    for replicate in REPLICATES:
        for dataset in (*DATASETS, "TSPLIB48", "TSPLIB22"):
            frame = per_instance[(per_instance.replicate == replicate) & (per_instance.dataset == dataset)]
            values = frame.Delta_E0_minus_E1.to_numpy(float); low, high = bootstrap(values, f"{replicate}|{dataset}")
            comparison_rows.append({"replicate": replicate, "dataset": dataset, "n_instances": len(frame), "mean_gap_E0": frame.gap_E0.mean(), "mean_gap_E1": frame.gap_E1.mean(), "Delta_E0_minus_E1": values.mean(), "paired_ci95_low": low, "paired_ci95_high": high})
        frames = {name: per_instance[(per_instance.replicate == replicate) & (per_instance.dataset == name)] for name in GAP5}
        e0 = np.mean([frame.gap_E0.mean() for frame in frames.values()]); e1 = np.mean([frame.gap_E1.mean() for frame in frames.values()]); low, high = gap5_bootstrap(frames, replicate)
        comparison_rows.append({"replicate": replicate, "dataset": "GAP5", "n_instances": sum(len(x) for x in frames.values()), "mean_gap_E0": e0, "mean_gap_E1": e1, "Delta_E0_minus_E1": e0-e1, "paired_ci95_low": low, "paired_ci95_high": high})
    comparison = pd.DataFrame(comparison_rows)
    aggregates = []
    for dataset, part in comparison.groupby("dataset", sort=False):
        values = part.Delta_E0_minus_E1.to_numpy(float)
        aggregates.append({"replicate": "ALL_EVOLUTION_REPLICATES", "dataset": dataset, "n_instances": np.nan, "mean_gap_E0": part.mean_gap_E0.mean(), "mean_gap_E1": part.mean_gap_E1.mean(), "Delta_E0_minus_E1": values.mean(), "paired_ci95_low": np.nan, "paired_ci95_high": np.nan, "median_delta": np.median(values), "sample_sd_delta": np.std(values, ddof=1), "min_delta": values.min(), "max_delta": values.max(), "e1_win_count": int((values>0).sum()), "e0_win_count": int((values<0).sum())})
    full_comparison = pd.concat([comparison, pd.DataFrame(aggregates)], ignore_index=True); full_comparison.to_csv(OUT / "09_e0_e1_paired_comparison.csv", index=False)
    trajectory = pd.read_csv(OUT / "04_population_structural_trajectory.csv")
    p2 = trajectory[(trajectory.stage == "P2") & (trajectory.object_type == "population")]
    structural = p2.pivot(index="replicate", columns="arm", values=["exact_ot_mean_cost", "swd_5d", "energy_distance_5d"])
    h1_wins = sum(bool(structural.loc[rep, ("exact_ot_mean_cost", "E1")] < structural.loc[rep, ("exact_ot_mean_cost", "E0")]) for rep in REPLICATES)
    h1 = "SUPPORTED" if h1_wins == 3 else "NOT_SUPPORTED" if h1_wins == 0 else "INCONCLUSIVE"
    primary = comparison[comparison.dataset == "OriginalTSPLIB70"].set_index("replicate"); gap5 = comparison[comparison.dataset == "GAP5"].set_index("replicate")
    primary_wins = int((primary.Delta_E0_minus_E1 > 0).sum())
    if primary_wins == 3 and not bool((gap5.Delta_E0_minus_E1 < 0).all()): h2 = "PRELIMINARILY_SUPPORTED"
    elif primary_wins == 0: h2 = "NOT_SUPPORTED"
    else: h2 = "INCONCLUSIVE"
    historical_reference_report(results); runtime_report()
    table = primary[["mean_gap_E0", "mean_gap_E1", "Delta_E0_minus_E1", "paired_ci95_low", "paired_ci95_high"]]
    held = comparison[comparison.dataset == "TSPLIB22"].set_index("replicate"); gap = comparison[comparison.dataset == "GAP5"].set_index("replicate")
    lines = ["# P6-B Final Research Conclusion", "", f"## H1: `{h1}`", f"## H2: `{h2}`", "", "Delta = gap(E0) − gap(E1); positive favors E1.", "", "| Replicate | Original70 E0 | Original70 E1 | Delta | 95% CI | TSPLIB22 Delta | GAP5 Delta |", "|---|---:|---:|---:|---|---:|---:|"]
    for rep in REPLICATES:
        row = table.loc[rep]; lines.append(f"| {rep} | {row.mean_gap_E0:.6f} | {row.mean_gap_E1:.6f} | {row.Delta_E0_minus_E1:+.6f} | [{row.paired_ci95_low:.6f}, {row.paired_ci95_high:.6f}] | {held.at[rep,'Delta_E0_minus_E1']:+.6f} | {gap.at[rep,'Delta_E0_minus_E1']:+.6f} |")
    lines.extend(["", f"H1 is based on all three final P2 exact-OT directions ({h1_wins}/3 favor E1); OT is the optimized objective, while SWD and energy are supporting structural diagnostics.", f"H2 is the core decision: E1 wins Original70 in {primary_wins}/3 independent evolution replicates. Instance-level bootstrap intervals are conditional within replicate and do not create additional evolution replicates; training-seed variability is unmeasured.", "Original70 is target-informed and TSPLIB22 has been repeatedly observed. No checkpoint or protocol was chosen from evaluation results.", "", "`BLADE_P6B_TOP5_DISTRIBUTION_GUIDED_EVOLUTION_COMPLETE`", "", f"`H1: {h1}`", f"`H2: {h2}`"])
    (OUT / "11_final_research_conclusion.md").write_text("\n".join(lines) + "\n")
    x = np.arange(3); fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(x-.18, table.mean_gap_E0, .36, label="E0"); ax.bar(x+.18, table.mean_gap_E1, .36, label="E1"); ax.set_xticks(x); ax.set_xticklabels(REPLICATES); ax.set_ylabel("Original70 mean gap (%)"); ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "02_final_solver_performance.png", dpi=180); plt.close(fig)
    structure_gain = np.asarray([structural.loc[rep, ("exact_ot_mean_cost", "E0")] - structural.loc[rep, ("exact_ot_mean_cost", "E1")] for rep in REPLICATES]); solver_gain = primary.loc[list(REPLICATES), "Delta_E0_minus_E1"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.scatter(structure_gain, solver_gain, s=60); [ax.annotate(rep, (structure_gain[i], solver_gain[i])) for i, rep in enumerate(REPLICATES)]; ax.axhline(0, color="gray", lw=1); ax.set_xlabel("P2 exact OT improvement E0−E1"); ax.set_ylabel("Original70 gap improvement E0−E1 (pp)"); ax.set_title("Three evolution replicates; descriptive only"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIGURES / "03_structure_vs_solver_gain.png", dpi=180); plt.close(fig)
    metadata = {"status": "COMPLETE", "completion_label": "BLADE_P6B_TOP5_DISTRIBUTION_GUIDED_EVOLUTION_COMPLETE", "H1": h1, "H2": h2, "no_followup_started": True, "primary_e1_win_count": primary_wins}
    write_json(OUT / "completion.json", metadata); Status().update(phase="complete", **metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "evolve", "prepare", "label", "train", "evaluate", "finalize"))
    parser.add_argument("--replicate", choices=REPLICATES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "audit": result = audit()
    elif args.phase == "evolve":
        if args.replicate is None: raise SystemExit("--replicate is required for evolve")
        result = evolve(args.replicate)
    elif args.phase == "prepare": result = prepare()
    elif args.phase == "label": result = label()
    elif args.phase == "finalize": result = finalize()
    else:
        if args.replicate is None or args.arm is None: raise SystemExit("--replicate and --arm are required")
        result = train(args.replicate, args.arm, args.device) if args.phase == "train" else evaluate(args.replicate, args.arm, args.device)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
