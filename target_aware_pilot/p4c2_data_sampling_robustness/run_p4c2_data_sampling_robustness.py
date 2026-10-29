#!/usr/bin/env python3
"""P4-C2 data-sampling-seed robustness on one frozen E0 offspring pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

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
P2 = ROOT / "target_aware_pilot" / "p2_bounded_evolution"
P4A = ROOT / "target_aware_pilot" / "p4a_random_evolution_ablation"
P4B1 = ROOT / "target_aware_pilot" / "p4b1_acquisition_ablation"
P4B2 = ROOT / "target_aware_pilot" / "p4b2_random_archive_ablation"
OUT = ROOT / "target_aware_pilot" / "p4c2_data_sampling_robustness"
DATA_DIR = OUT / "datasets"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(P2), str(P4A), str(P4B1), str(P4B2)]

import run_p2_bounded_evolution as p2  # noqa: E402
import run_p4a_random_evolution_ablation as p4a  # noqa: E402
import run_p4b1_acquisition_ablation as p4b1  # noqa: E402
import run_p4b2_random_archive_ablation as p4b2  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, ORACLE_LEDGER_COLUMNS, sha256_file  # noqa: E402


M4 = p4b1.M4
M4_SHA = p4b1.M4_SHA
HARD_M5 = p4b1.HARD_M5
DATASETS = p4b1.DATASETS
ARMS = ("R0", "R2")
NEW_REPLICATES = ("S1", "S2")
ALL_REPLICATES = ("Original", *NEW_REPLICATES)
SEEDS = {
    "Original": {"retention": 20261026, "acquisition": 20261025},
    "S1": {"retention": 260201, "acquisition": 260301},
    "S2": {"retention": 260202, "acquisition": 260302},
}
TRAINING_SEED = 2026
FULL_POOL_N = 39_848
ARCHIVE_N = 30_000
TRAIN_N = 10_000
BOOTSTRAP_REPLICATES = 10_000


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def label_dir(replicate: str, arm: str) -> Path:
    return LOGS / f"{replicate}_{arm}" / "labeling"


def run_dir(replicate: str, arm: str) -> Path:
    return LOGS / f"{replicate}_{arm}"


def checkpoint_path(replicate: str, arm: str) -> Path:
    return CHECKPOINTS / f"{replicate}_{arm}_M5_E256.tsp.best"


def archive_bundle(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_archive30k.npz"


def archive_manifest(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_archive30k_manifest.parquet"


def train_bundle(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_10k.npz"


def train_manifest(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_10k_manifest.parquet"


def train_integrity_path(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_10k_integrity.json"


def eval_label(replicate: str, arm: str) -> str:
    return f"{replicate}_{arm}_M5"


def eval_csv(replicate: str, arm: str) -> Path:
    return run_dir(replicate, arm) / "external_benchmark_results.csv"


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str], tours: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "coordinates": np.asarray(coordinates, dtype=np.float64),
        "instance_hash": np.asarray(hashes, dtype="U64"),
    }
    if tours is not None:
        payload["expert_tours"] = np.asarray(tours, dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_bundle(path: Path, require_tours: bool = False) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
        tours = np.asarray(payload["expert_tours"], dtype=np.int64) if "expert_tours" in payload.files else None
    if require_tours and tours is None:
        raise RuntimeError(f"labeled bundle missing tours: {path}")
    return coordinates, hashes, tours


def base_protocol() -> dict[str, Any]:
    trainer = ROOT / "scripts" / "run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
    return {
        "m4_sha256": M4_SHA,
        "trainer_source": str(trainer),
        "trainer_source_sha256": sha256_file(trainer),
        "training_seed": TRAINING_SEED,
        "optimizer": "fresh Adam",
        "scheduler": None,
        "learning_rate": 3.75e-5,
        "epochs": 256,
        "optimizer_updates": 10_240,
        "logical_batch_size": 256,
        "microbatch_size": 64,
        "augmentation": "formal NumPy permutation and suffix-length sampling",
        "checkpoint_rule": "fixed E256; no validation/test selection",
        "evaluation": "frozen greedy evaluator on the same eight benchmarks",
    }


def ensure_dirs() -> None:
    for path in (OUT, DATA_DIR, FIG, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_frozen_inputs() -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], np.ndarray, pd.DataFrame, dict[str, Any]]:
    full_xy, full, source = p4b2.load_frozen_full_pool()
    if len(full) != FULL_POOL_N or membership_hash(full.instance_hash.astype(str).tolist()) != "2565bd7088d6bdc7b6659add93c5dbabfe848725a99f175328fecda3586d6d0d":
        raise RuntimeError("frozen full-pool membership drift")
    if sha256_file(p4b2.FULL_POOL_BUNDLE) != "327265660c63607ec4c406d4955a60fa072101246ca456d9342443ae12ce6e40":
        raise RuntimeError("persisted full-pool bundle drift")
    r0_xy, r0_hashes = p4b1.load_bundle(p4b2.R0_ARCHIVE_BUNDLE)
    r0 = pd.read_parquet(p4b2.R0_ARCHIVE_MANIFEST)
    if r0.instance_hash.astype(str).tolist() != r0_hashes or sha256_file(p4b2.R0_ARCHIVE_BUNDLE) != p4b2.R0_ARCHIVE_SHA:
        raise RuntimeError("frozen R0 archive drift")
    selection = json.loads(p4b2.R0_ARCHIVE_SELECTION.read_text(encoding="utf-8"))
    return full_xy, full, source, r0_xy, r0, selection


def audit() -> dict[str, Any]:
    ensure_dirs()
    if sha256_file(M4) != M4_SHA:
        raise RuntimeError("formal M4 SHA mismatch")
    full_xy, full, source, r0_xy, r0, selection = load_frozen_inputs()
    geometry = p2.load_geometry()
    reproduced, reproduced_meta = p2.structured_select(
        full, ARCHIVE_N, geometry, p2.TARGET_FRACTION, p2.LINEAGE_CAP_ARCHIVE,
        "common_bounded_archive_retention",
    )
    deterministic_replay = reproduced.instance_hash.astype(str).tolist() == r0.instance_hash.astype(str).tolist()
    metadata_replay = reproduced_meta == selection
    if not deterministic_replay or not metadata_replay:
        raise RuntimeError("deterministic R0 retention replay mismatch")
    full_lookup = dict(zip(full.instance_hash.astype(str), full_xy, strict=True))
    if not all(np.array_equal(points, full_lookup[h]) for h, points in zip(r0.instance_hash.astype(str), r0_xy, strict=True)):
        raise RuntimeError("R0 coordinates do not reproduce from full pool")

    original_config = json.loads(p4b2.R2_CONFIG.read_text(encoding="utf-8"))
    if original_config["retention_seed"] != SEEDS["Original"]["retention"] or original_config["acquisition_seed"] != SEEDS["Original"]["acquisition"]:
        raise RuntimeError("original R2 seed mismatch")
    if json.loads(p4b1.R0_CONFIG.read_text(encoding="utf-8"))["seed"] != SEEDS["Original"]["acquisition"]:
        raise RuntimeError("original R0 acquisition seed mismatch")

    protocol = {
        "status": "PREREGISTERED_BEFORE_NEW_DATA_GENERATION",
        "common_offspring_pool_n": FULL_POOL_N,
        "common_offspring_membership_sha256": source["full_pool_membership_sha256"],
        "r0_retention": {
            "algorithm": "common_bounded_archive_retention",
            "randomness": False,
            "seed_consumed": False,
            "deterministic_replay_exact": deterministic_replay,
            "membership_sha256": membership_hash(r0.instance_hash.astype(str).tolist()),
        },
        "r2_retention": "canonical-hash ordering followed by NumPy uniform sampling without replacement",
        "final_acquisition": "canonical-hash ordering followed by NumPy uniform sampling without replacement",
        "replicates": {
            name: {"retention_seed": values["retention"], "acquisition_seed": values["acquisition"], "training_seed": TRAINING_SEED}
            for name, values in SEEDS.items()
        },
        "important_scope": "R0 archive membership is deterministic and fixed; new seeds vary R0 final acquisition only, while R2 varies retention and acquisition. This is not R0 retention-seed robustness.",
        "base_training_protocol": base_protocol(),
        "delta_definition": "GAP(R2) - GAP(R0); positive favors target-aware R0",
        "gap8_definition": "unweighted arithmetic mean of the same eight frozen benchmark mean gaps",
        "supported_rule": "S1 and S2 GAP8 Delta > 0; R0 wins at least 5/8 benchmark means in each; and no Uniform100, GaussianMixture100, or GaussianMixture500 Delta is below -0.25 pp in either new replicate",
        "not_supported_rule": "S1 and S2 GAP8 Delta < 0 and R2 wins at least 5/8 benchmark means in each",
        "otherwise": "INCONCLUSIVE",
        "bootstrap": "10,000 paired test-instance resamples within each replicate; never pool instances across replicates",
        "tsplib22_role": "repeatedly observed exploratory comparability only",
        "no_third_sampling_seed": True,
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    config_path = OUT / "preregistered_sampling_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("preregistered protocol already exists with different content")
    else:
        write_json(config_path, protocol)

    integrity = f"""# P4-C2 Protocol Integrity

- Frozen E0 pool: {FULL_POOL_N:,} unique eligible TSP100, membership SHA256 `{source['full_pool_membership_sha256']}`; G1/G2 persisted mutation artifacts and provenance reproduce. No FOGA mutation was rerun.
- R0 target-aware retention is **fully deterministic**: the selector has no RNG/seed argument; quota allocation, microbin ordering, lineage caps, backfill, and final instance-hash tie-breaking are deterministic. A fresh replay exactly reproduces all 30,000 ordered members and selection metadata.
- Consequently, S1/S2 cannot vary R0 Archive30k without changing the algorithm. They vary only R0 final Random10k. R2 legitimately varies both uniform retention and final acquisition.
- Original seeds: R2 retention `{SEEDS['Original']['retention']}`; common final acquisition `{SEEDS['Original']['acquisition']}`. R0 has no retention seed.
- New seeds were frozen before generation: S1 `{SEEDS['S1']['retention']}/{SEEDS['S1']['acquisition']}`, S2 `{SEEDS['S2']['retention']}/{SEEDS['S2']['acquisition']}`; all BQ training uses seed `{TRAINING_SEED}`.
- Formal M4 SHA256 `{M4_SHA}` and the P4-B2 trainer/evaluator protocol are frozen.
- Labeling will use the canonical Concorde/cache path without BQ hardness inference.

Status: **PASS**. Scope is acquisition/data-sampling robustness within one frozen E0 pool, not R0 retention-seed robustness or cross-evolution-pool robustness.
"""
    (OUT / "00_protocol_integrity.md").write_text(integrity, encoding="utf-8")
    prereg = f"""# Preregistered Sampling Protocol

| Replicate | Retention seed | Acquisition seed | Training seed |
|---|---:|---:|---:|
| Original | {SEEDS['Original']['retention']} (R2 only; R0 deterministic) | {SEEDS['Original']['acquisition']} | {TRAINING_SEED} |
| S1 | {SEEDS['S1']['retention']} (R2 only; ignored by deterministic R0) | {SEEDS['S1']['acquisition']} | {TRAINING_SEED} |
| S2 | {SEEDS['S2']['retention']} (R2 only; ignored by deterministic R0) | {SEEDS['S2']['acquisition']} | {TRAINING_SEED} |

- Delta = GAP(R2) − GAP(R0); positive favors R0.
- SUPPORTED_PRELIMINARY requires both new GAP8 Deltas positive, at least 5/8 R0 benchmark wins in each new replicate, and no Delta below −0.25 pp on Uniform100/GaussianMixture100/GaussianMixture500.
- NOT_SUPPORTED requires both new GAP8 Deltas negative and at least 5/8 R2 benchmark wins in each. All other outcomes are INCONCLUSIVE.
- Membership is frozen before diagnostics, Concorde labeling, training, or evaluation. No result-dependent seed, resampling, reweighting, or rerun is permitted.

Protocol SHA256: `{protocol['protocol_sha256']}`.
"""
    (OUT / "01_preregistered_sampling_protocol.md").write_text(prereg, encoding="utf-8")
    Status().update(phase="audit", status="PASS", protocol_sha256=protocol["protocol_sha256"])
    return {"status": "PASS", "protocol_sha256": protocol["protocol_sha256"], "r0_retention_random": False, "deterministic_replay": True}


def protocol_gate() -> dict[str, Any]:
    path = OUT / "preregistered_sampling_config.json"
    if not path.is_file():
        raise RuntimeError("run audit before any new sampling")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_NEW_DATA_GENERATION":
        raise RuntimeError("invalid preregistration")
    digest = protocol.pop("protocol_sha256")
    if canonical_hash(protocol) != digest:
        raise RuntimeError("preregistered protocol hash mismatch")
    protocol["protocol_sha256"] = digest
    return protocol


def uniform_sample(frame: pd.DataFrame, coordinates: np.ndarray, n: int, seed: int, stage: str) -> tuple[pd.DataFrame, np.ndarray]:
    hashes = frame.instance_hash.astype(str).tolist()
    order = np.argsort(np.asarray(hashes), kind="stable")
    canonical = frame.iloc[order].reset_index(drop=True)
    canonical_xy = coordinates[order]
    draw = np.random.default_rng(seed).choice(len(canonical), n, replace=False)
    if not np.array_equal(draw, np.random.default_rng(seed).choice(len(canonical), n, replace=False)):
        raise RuntimeError(f"{stage} seed replay failure")
    selected = canonical.iloc[draw].copy().reset_index(drop=True)
    selected[f"{stage}_method"] = "uniform_without_replacement"
    selected[f"{stage}_seed"] = seed
    selected[f"{stage}_draw_order"] = np.arange(n)
    return selected, canonical_xy[draw]


def coordinate_gate(coordinates: np.ndarray, hashes: list[str], n: int) -> dict[str, Any]:
    audit_value = p4b1.coordinate_audit(coordinates, hashes)
    if audit_value["shape"] != [n, 100, 2] or audit_value["unique"] != n:
        raise RuntimeError("coordinate shape/uniqueness failure")
    if not all(audit_value[key] for key in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")):
        raise RuntimeError("coordinate validity failure")
    return audit_value


def metric_row(replicate: str, arm: str, level: str, frame: pd.DataFrame, geometry: dict[str, Any]) -> dict[str, Any]:
    row, _ = p2.evaluate_frame(arm, 2, f"{replicate}_{arm}_{level}", frame, geometry)
    keep = {
        "n": row["n"], "unique_n": row["unique_n"],
        "strict_target_n": row["strict_target_n"], "strict_target_fraction": row["strict_target_fraction"],
        "coarse_js": row["coarse_js"], "sliced_wasserstein": row["sliced_wasserstein"],
        "occupied_cells": row["occupied_cells"], "effective_cells": row["effective_cells"],
        "largest_cell_share": row["largest_cell_share"], "cell_counts": json.dumps(row["cell_counts"]),
        "reference_coverage_n": row["reference_coverage_n"], "effective_lineages": row["effective_lineages"],
        "largest_lineage_share": row["largest_lineage_share"], "fft_median": row["fft_median"],
        "nn_cv_median": row["nn_cv_median"], "aux_coarse_js": row["aux_coarse_js"],
        "aux_effective_cells": row["aux_effective_cells"],
    }
    return {"sampling_replicate": replicate, "arm": arm, "level": level, **keep}


def save_dataset(replicate: str, arm: str, level: str, frame: pd.DataFrame, coordinates: np.ndarray) -> dict[str, Any]:
    hashes = frame.instance_hash.astype(str).tolist()
    n = ARCHIVE_N if level == "archive30k" else TRAIN_N
    audit_value = coordinate_gate(coordinates, hashes, n)
    bundle = archive_bundle(replicate, arm) if level == "archive30k" else train_bundle(replicate, arm)
    manifest = archive_manifest(replicate, arm) if level == "archive30k" else train_manifest(replicate, arm)
    save_bundle(bundle, coordinates, hashes)
    frame.to_parquet(manifest, index=False)
    frame.to_csv(manifest.with_suffix(".csv"), index=False)
    result = {
        "status": "PASS", "sampling_replicate": replicate, "arm": arm, "level": level,
        "bundle_sha256": sha256_file(bundle), "manifest_parquet_sha256": sha256_file(manifest),
        "manifest_csv_sha256": sha256_file(manifest.with_suffix(".csv")),
        "membership_sha256": membership_hash(hashes), **audit_value,
    }
    if level == "training10k":
        write_json(train_integrity_path(replicate, arm), result)
    return result


def original_objects() -> dict[str, tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]]:
    r0a_xy, r0a_hashes = p4b1.load_bundle(p4b2.R0_ARCHIVE_BUNDLE)
    r0a = pd.read_parquet(p4b2.R0_ARCHIVE_MANIFEST)
    r2a_xy, r2a_hashes = p4b1.load_bundle(p4b2.R2_ARCHIVE_BUNDLE)
    r2a = pd.read_parquet(p4b2.R2_ARCHIVE_PARQUET)
    r0t_xy, r0t_hashes = p4b1.load_bundle(p4b2.R0_10K_BUNDLE)
    r0t = pd.read_parquet(p4b2.R0_10K_MANIFEST)
    r2t_xy, r2t_hashes = p4b1.load_bundle(p4b2.R2_10K_BUNDLE)
    r2t = pd.read_parquet(p4b2.R2_10K_PARQUET)
    for frame, hashes in ((r0a, r0a_hashes), (r2a, r2a_hashes), (r0t, r0t_hashes), (r2t, r2t_hashes)):
        if frame.instance_hash.astype(str).tolist() != hashes:
            raise RuntimeError("original manifest/bundle alignment failure")
    return {"R0": (r0a_xy, r0a, r0t_xy, r0t), "R2": (r2a_xy, r2a, r2t_xy, r2t)}


def make_sampling_figures(metrics: pd.DataFrame, frames: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    names = ["strict_target_fraction", "coarse_js", "sliced_wasserstein", "effective_cells", "effective_lineages", "aux_coarse_js"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, names, strict=True):
        for arm, marker in (("R0", "o"), ("R2", "s")):
            part = metrics[(metrics.level == "training10k") & (metrics.arm == arm)].set_index("sampling_replicate").loc[list(ALL_REPLICATES)]
            ax.plot(range(3), part[name], marker=marker, label=arm)
        ax.set_xticks(range(3)); ax.set_xticklabels(ALL_REPLICATES); ax.set_title(name); ax.grid(alpha=.2)
    axes[0, 0].legend(); fig.suptitle("Frozen training10k structure diagnostics"); fig.tight_layout()
    fig.savefig(FIG / "01_training10k_distribution_metrics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, replicate in zip(axes, ALL_REPLICATES, strict=True):
        for arm, color in (("R0", "#4c78a8"), ("R2", "#e45756")):
            frame = frames[(replicate, arm, "training10k")]
            ax.scatter(frame[p2.FEATURES[0]], frame[p2.FEATURES[1]], s=2, alpha=.12, label=arm, color=color)
        ax.set_title(replicate); ax.set_xlabel(p2.FEATURES[0]); ax.grid(alpha=.15)
    axes[0].set_ylabel(p2.FEATURES[1]); axes[-1].legend(markerscale=4); fig.suptitle("R0/R2 frozen training samples")
    fig.tight_layout(); fig.savefig(FIG / "02_training10k_feature_space.png", dpi=180); plt.close(fig)


def sample() -> dict[str, Any]:
    protocol = protocol_gate()
    started = time.perf_counter()
    full_xy, full, source, r0_xy, r0, _ = load_frozen_inputs()
    geometry = p2.load_geometry()
    history, c04 = p4b1.frozen_history()
    original70 = pd.read_parquet(p4b1.CYCLES / "C03" / "evaluation_work" / "M4" / "OriginalTSPLIB70" / "instance_metrics.parquet")
    tsplib_hashes = set(original70.coordinate_hash.astype(str))
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    coordinates: dict[tuple[str, str, str], np.ndarray] = {}
    original = original_objects()
    for arm in ARMS:
        axy, aframe, txy, tframe = original[arm]
        frames[("Original", arm, "archive30k")] = aframe
        frames[("Original", arm, "training10k")] = tframe
        coordinates[("Original", arm, "archive30k")] = axy
        coordinates[("Original", arm, "training10k")] = txy

    sampling_rows: list[dict[str, Any]] = []
    for replicate in NEW_REPLICATES:
        retention_seed = SEEDS[replicate]["retention"]
        acquisition_seed = SEEDS[replicate]["acquisition"]
        r2_archive, r2_archive_xy = uniform_sample(full, full_xy, ARCHIVE_N, retention_seed, "archive_retention")
        archives = {"R0": (r0.copy(), r0_xy.copy()), "R2": (r2_archive, r2_archive_xy)}
        finals: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
        for arm in ARMS:
            archive_frame, archive_xy = archives[arm]
            archive_frame = archive_frame.copy()
            archive_frame["sampling_replicate"] = replicate
            archive_frame["requested_retention_seed"] = retention_seed
            archive_frame["retention_seed_consumed"] = arm == "R2"
            final, final_xy = uniform_sample(archive_frame, archive_xy, TRAIN_N, acquisition_seed, "final_acquisition")
            final["sampling_replicate"] = replicate
            final["arm"] = arm
            finals[arm] = (final, final_xy)
            frames[(replicate, arm, "archive30k")] = archive_frame
            frames[(replicate, arm, "training10k")] = final
            coordinates[(replicate, arm, "archive30k")] = archive_xy
            coordinates[(replicate, arm, "training10k")] = final_xy
            archive_info = save_dataset(replicate, arm, "archive30k", archive_frame, archive_xy)
            final_info = save_dataset(replicate, arm, "training10k", final, final_xy)
            hashes = set(final.instance_hash.astype(str))
            if hashes & history or hashes & tsplib_hashes:
                raise RuntimeError(f"{replicate}/{arm} forbidden overlap")
            sampling_rows.append({
                "sampling_replicate": replicate, "arm": arm,
                "retention_seed_requested": retention_seed,
                "retention_seed_consumed": arm == "R2",
                "acquisition_seed": acquisition_seed, "training_seed": TRAINING_SEED,
                "archive_membership_sha256": archive_info["membership_sha256"],
                "archive_bundle_sha256": archive_info["bundle_sha256"],
                "training_membership_sha256": final_info["membership_sha256"],
                "training_bundle_sha256": final_info["bundle_sha256"],
                "m4_history_overlap_n": len(hashes & history), "c04_overlap_n": len(hashes & c04),
                "direct_tsplib70_overlap_n": len(hashes & tsplib_hashes), "status": "FROZEN_PASS",
            })

    # Add original rows without rewriting prior artifacts.
    original_config = json.loads(p4b2.R2_CONFIG.read_text())
    original_r0_config = json.loads(p4b1.R0_CONFIG.read_text())
    for arm in ARMS:
        aframe = frames[("Original", arm, "archive30k")]
        tframe = frames[("Original", arm, "training10k")]
        bundle = p4b2.R0_10K_BUNDLE if arm == "R0" else p4b2.R2_10K_BUNDLE
        archive_path = p4b2.R0_ARCHIVE_BUNDLE if arm == "R0" else p4b2.R2_ARCHIVE_BUNDLE
        sampling_rows.append({
            "sampling_replicate": "Original", "arm": arm,
            "retention_seed_requested": SEEDS["Original"]["retention"], "retention_seed_consumed": arm == "R2",
            "acquisition_seed": SEEDS["Original"]["acquisition"], "training_seed": TRAINING_SEED,
            "archive_membership_sha256": membership_hash(aframe.instance_hash.astype(str).tolist()),
            "archive_bundle_sha256": sha256_file(archive_path),
            "training_membership_sha256": membership_hash(tframe.instance_hash.astype(str).tolist()),
            "training_bundle_sha256": sha256_file(bundle),
            "m4_history_overlap_n": 0,
            "c04_overlap_n": int(original_r0_config["c04_hard10k_overlap_n"] if arm == "R0" else original_config["c04_hard10k_overlap_n"]),
            "direct_tsplib70_overlap_n": 0, "status": "EXISTING_FROZEN_PASS",
        })

    sampling_manifest = pd.DataFrame(sampling_rows).sort_values(["sampling_replicate", "arm"], key=lambda s: s.map({"Original": 0, "S1": 1, "S2": 2}) if s.name == "sampling_replicate" else s)
    sampling_manifest.to_csv(OUT / "02_sampling_runs_manifest.csv", index=False)

    metric_rows = []
    for replicate in ALL_REPLICATES:
        archive_overlap = len(set(frames[(replicate, "R0", "archive30k")].instance_hash.astype(str)) & set(frames[(replicate, "R2", "archive30k")].instance_hash.astype(str)))
        training_overlap = len(set(frames[(replicate, "R0", "training10k")].instance_hash.astype(str)) & set(frames[(replicate, "R2", "training10k")].instance_hash.astype(str)))
        for arm in ARMS:
            for level in ("archive30k", "training10k"):
                row = metric_row(replicate, arm, level, frames[(replicate, arm, level)], geometry)
                row["r0_r2_archive_overlap_n"] = archive_overlap
                row["r0_r2_training_overlap_n"] = training_overlap
                metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "03_archive_distribution_comparison.csv", index=False)
    make_sampling_figures(metrics, frames)

    lines = ["# Training Dataset Integrity", "", f"- Preregistered protocol SHA256 `{protocol['protocol_sha256']}` was present before S1/S2 generation.", f"- Frozen source pool: {FULL_POOL_N:,} unique TSP100, membership `{source['full_pool_membership_sha256']}`; no mutation was run.", "- Four new datasets are each 10,000 unique, finite, [0,1]-bounded TSP100 with 100 distinct nodes and reproduced coordinate hashes.", "- Every new dataset has zero overlap with C00–C03 M4 history and zero direct OriginalTSPLIB70 coordinate overlap. C04 overlap is reported, not silently filtered, because the frozen requirement only excludes M4 history.", "- R0 archive is identical in Original/S1/S2 because retention is deterministic; R0 final acquisition and both R2 stages change exactly through preregistered uniform seeds.", "", "| Replicate | Arm | Archive overlap R0/R2 | Training overlap R0/R2 | Strict target | 3x3 JS | SWD | Eff. lineages | Max lineage |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for replicate in ALL_REPLICATES:
        for arm in ARMS:
            row = metrics[(metrics.sampling_replicate == replicate) & (metrics.arm == arm) & (metrics.level == "training10k")].iloc[0]
            lines.append(f"| {replicate} | {arm} | {int(row.r0_r2_archive_overlap_n):,} | {int(row.r0_r2_training_overlap_n):,} | {row.strict_target_fraction:.2%} | {row.coarse_js:.6f} | {row.sliced_wasserstein:.6f} | {row.effective_lineages:.1f} | {row.largest_lineage_share:.2%} |")
    lines.extend(["", "Diagnostics were computed only after membership was frozen. They did not control resampling.", "", "Status: **PASS**."])
    (OUT / "04_training_datasets_integrity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "sampling_integrity.json", {
        "status": "PASS", "protocol_sha256": protocol["protocol_sha256"],
        "source_pool_membership_sha256": source["full_pool_membership_sha256"],
        "sampling_manifest_sha256": sha256_file(OUT / "02_sampling_runs_manifest.csv"),
        "distribution_csv_sha256": sha256_file(OUT / "03_archive_distribution_comparison.csv"),
        "sampling_wall_seconds": time.perf_counter() - started,
        "no_evolution": True, "no_bq_hardness": True, "no_resampling_after_diagnostics": True,
    })
    Status().update(phase="sampling", status="PASS")
    return {"status": "PASS", "new_datasets": 4, "wall_seconds": time.perf_counter() - started}


def verified_cache() -> pd.DataFrame:
    paths = [p4b1.SHARED_CACHE, P4A / "labeling" / "ledger.parquet", P4B1 / "labeling" / "ledger.parquet", P4B2 / "labeling" / "ledger.parquet"]
    paths.extend(label_dir(rep, arm) / "ledger.parquet" for rep in NEW_REPLICATES for arm in ARMS)
    pieces = []
    for path in paths:
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        required = set(ORACLE_LEDGER_COLUMNS)
        if not required.issubset(frame.columns):
            continue
        frame = frame[list(ORACLE_LEDGER_COLUMNS)].copy()
        frame = frame[frame.verified.fillna(False) & frame.status.eq("solved")]
        if len(frame) and set(frame.binary_hash.astype(str)) != {CANONICAL_CONCORDE_SHA256}:
            raise RuntimeError(f"noncanonical cache: {path}")
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True).drop_duplicates("instance_hash", keep="last")


def label(replicate: str, arm: str) -> dict[str, Any]:
    protocol_gate()
    if replicate not in NEW_REPLICATES or arm not in ARMS:
        raise ValueError("only S1/S2 and R0/R2 are permitted")
    if not (OUT / "sampling_integrity.json").is_file():
        raise RuntimeError("sampling audit must pass before labeling")
    coordinates, hashes, _ = load_bundle(train_bundle(replicate, arm))
    manifest = pd.read_parquet(train_manifest(replicate, arm))
    if manifest.instance_hash.astype(str).tolist() != hashes:
        raise RuntimeError("training manifest alignment failure")
    directory = label_dir(replicate, arm)
    directory.mkdir(parents=True, exist_ok=True)
    integrity_path = directory / "labeling_integrity.json"
    if integrity_path.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and existing.get("training_membership_sha256") == membership_hash(hashes):
            return existing

    import run_blade_v2 as base

    prior = verified_cache()
    known = set(prior.instance_hash.astype(str))
    started = time.perf_counter()
    ledger = base.label_with_cache(coordinates, hashes, directory, prior)
    tours = base._oracle_tours(ledger, hashes)
    if len(ledger) != TRAIN_N or tours.shape != (TRAIN_N, 100) or not ledger.verified.all():
        raise RuntimeError("incomplete labeling")
    if not all(validate_tour_0based(tour, 100)[0] for tour in tours):
        raise RuntimeError("illegal expert tour")
    costs = np.asarray([tour_cost_int(integer_distance_matrix(points), tour) for points, tour in zip(coordinates, tours, strict=True)], dtype=np.int64)
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)):
        raise RuntimeError("Concorde objective mismatch")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (TRAIN_N, 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]):
        raise RuntimeError("teacher-forced closure mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist():
        raise RuntimeError("tour hash mismatch")
    payload_columns = set(ORACLE_LEDGER_COLUMNS[1:])
    labeled = manifest[[column for column in manifest.columns if column not in payload_columns]].merge(ledger, on="instance_hash", validate="one_to_one")
    ledger.to_parquet(directory / "ledger.parquet", index=False)
    labeled.to_parquet(directory / "labeled.parquet", index=False)
    save_bundle(directory / "labeled_bundle.npz", coordinates, hashes, tours)
    wall = time.perf_counter() - started
    result = {
        "status": "PASS", "sampling_replicate": replicate, "arm": arm,
        "selected": TRAIN_N, "verified": TRAIN_N, "failures": 0,
        "cache_hits": sum(value in known for value in hashes),
        "physical_new_concorde_calls": sum(value not in known for value in hashes),
        "pipeline_wall_seconds": wall, "all_tours_legal": True, "all_costs_reproduced": True,
        "training_membership_sha256": membership_hash(hashes),
        "ledger_sha256": sha256_file(directory / "ledger.parquet"),
        "labeled_bundle_sha256": sha256_file(directory / "labeled_bundle.npz"),
        "labeled_metadata_sha256": sha256_file(directory / "labeled.parquet"),
        "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
        "concorde_sha256": CANONICAL_CONCORDE_SHA256, "bq_hardness_computed": False,
    }
    write_json(integrity_path, result)
    write_label_report()
    Status().update(phase="label", status="PASS", replicate=replicate, arm=arm)
    return result


def write_label_report() -> None:
    rows = []
    for replicate in NEW_REPLICATES:
        for arm in ARMS:
            path = label_dir(replicate, arm) / "labeling_integrity.json"
            if path.is_file():
                rows.append(json.loads(path.read_text()))
    lines = ["# Concorde Labeling Audit", "", "Canonical P4-B2 integer-distance Concorde convention was reused. BQ hardness was not computed.", "", "| Dataset | Verified | Cache hits | New calls | Failures | Wall (s) | Label bundle SHA256 |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['sampling_replicate']}-{row['arm']} | {row['verified']:,} | {row['cache_hits']:,} | {row['physical_new_concorde_calls']:,} | {row['failures']} | {row['pipeline_wall_seconds']:.1f} | `{row['labeled_bundle_sha256']}` |")
    if len(rows) == 4:
        lines.extend(["", f"- Total logical labels: {sum(row['verified'] for row in rows):,}; physical new calls: {sum(row['physical_new_concorde_calls'] for row in rows):,}; cache reuses: {sum(row['cache_hits'] for row in rows):,}.", "- All tours are 0-based Hamiltonian permutations; integer objectives and implicit closure were independently recomputed.", "", "Status: **PASS**."])
    else:
        lines.extend(["", f"Progress: {len(rows)}/4 datasets complete."])
    (OUT / "05_concorde_labeling_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    protocol = protocol_gate()
    if replicate not in NEW_REPLICATES or arm not in ARMS:
        raise ValueError("only S1/S2 and R0/R2 are permitted")
    label_info = json.loads((label_dir(replicate, arm) / "labeling_integrity.json").read_text())
    if label_info.get("status") != "PASS":
        raise RuntimeError("labeling gate failed")
    coordinates, hashes, tours = load_bundle(label_dir(replicate, arm) / "labeled_bundle.npz", require_tours=True)
    assert tours is not None
    frozen_xy, frozen_hashes, _ = load_bundle(train_bundle(replicate, arm))
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
        raise RuntimeError("labeled membership drift")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    directory = run_dir(replicate, arm) / "training_cycle"
    checkpoint = checkpoint_path(replicate, arm)
    core.RUN_ID = f"TARGET_AWARE_P4C2_{replicate}_{arm}"
    core.cycle_dir = lambda selected_arm, cycle: directory
    core.model_path = lambda selected_arm, index: checkpoint if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    config = {
        "sampling_replicate": replicate, "arm": arm, "training_seed": TRAINING_SEED,
        "membership_sha256": membership_hash(hashes), "label_bundle_sha256": label_info["labeled_bundle_sha256"],
        "protocol_sha256": protocol["protocol_sha256"], "base_training_protocol": base_protocol(),
    }
    config["training_config_sha256"] = canonical_hash(config)
    write_json(run_dir(replicate, arm) / "training_config.json", config)
    started = time.perf_counter()
    trained, meta, timing = core.train(f"{replicate}_{arm}", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)
    invocation_wall = time.perf_counter() - started
    stage = directory / "09_bq_training"
    e000 = torch.load(stage / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    curve = pd.read_csv(stage / "training_loss_curve.csv")
    if not e000_equal or len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(TRAIN_N).all():
        raise RuntimeError("training integrity failure")
    if meta["sha256"] != sha256_file(checkpoint) or not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters()):
        raise RuntimeError("checkpoint integrity failure")
    result = {
        "status": "PASS", **config, "m4_sha256": M4_SHA, "e000_weights_equal_m4": e000_equal,
        "epochs": 256, "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": invocation_wall,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]), "checkpoint_sha256": meta["sha256"],
        "loss_curve_sha256": sha256_file(stage / "training_loss_curve.csv"),
    }
    write_json(run_dir(replicate, arm) / "training_integrity.json", result)
    del trained, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_training_manifest()
    return result


def write_training_manifest() -> None:
    rows = []
    for arm, base, checkpoint, label_path, dataset_path in (
        ("R0", P4B1, P4B1 / "checkpoints" / "R0_Random_M5_E256.tsp.best", P4B1 / "labeling" / "labeled_bundle.npz", p4b2.R0_10K_BUNDLE),
        ("R2", P4B2, P4B2 / "checkpoints" / "R2_RandomArchive_Random10k_M5_E256.tsp.best", P4B2 / "labeling" / "labeled_bundle.npz", p4b2.R2_10K_BUNDLE),
    ):
        info = json.loads((base / "training_integrity.json").read_text())
        rows.append({"sampling_replicate": "Original", "arm": arm, "status": "EXISTING_PASS", "training_seed": TRAINING_SEED, "dataset_bundle_sha256": sha256_file(dataset_path), "label_bundle_sha256": sha256_file(label_path), "m4_sha256": M4_SHA, "e000_weights_equal_m4": info["e000_weights_equal_m4"], "optimizer_steps": info["optimizer_steps"], "final_loss": info["final_loss"], "training_wall_s": info["training_loop_wall_s"], "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_path": str(checkpoint)})
    for replicate in NEW_REPLICATES:
        for arm in ARMS:
            path = run_dir(replicate, arm) / "training_integrity.json"
            if path.is_file():
                info = json.loads(path.read_text())
                rows.append({"sampling_replicate": replicate, "arm": arm, "status": info["status"], "training_seed": TRAINING_SEED, "dataset_bundle_sha256": sha256_file(train_bundle(replicate, arm)), "label_bundle_sha256": info["label_bundle_sha256"], "m4_sha256": info["m4_sha256"], "e000_weights_equal_m4": info["e000_weights_equal_m4"], "optimizer_steps": info["optimizer_steps"], "final_loss": info["final_loss"], "training_wall_s": info["training_loop_wall_s"], "checkpoint_sha256": info["checkpoint_sha256"], "checkpoint_path": str(checkpoint_path(replicate, arm)), "training_config_sha256": info["training_config_sha256"]})
            else:
                rows.append({"sampling_replicate": replicate, "arm": arm, "status": "PENDING", "training_seed": TRAINING_SEED, "dataset_bundle_sha256": sha256_file(train_bundle(replicate, arm)) if train_bundle(replicate, arm).is_file() else None})
    pd.DataFrame(rows).to_csv(OUT / "06_training_runs_manifest.csv", index=False)


def evaluate(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    if replicate not in NEW_REPLICATES or arm not in ARMS:
        raise ValueError("only S1/S2 and R0/R2 are permitted")
    info = json.loads((run_dir(replicate, arm) / "training_integrity.json").read_text())
    checkpoint = checkpoint_path(replicate, arm)
    if info["checkpoint_sha256"] != sha256_file(checkpoint):
        raise RuntimeError("checkpoint SHA mismatch")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=info["checkpoint_sha256"])
    frame, wall = core.evaluate(model, checkpoint, eval_label(replicate, arm), eval_csv(replicate, arm), Status(), f"{replicate}_{arm}")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all():
        raise RuntimeError("evaluation incomplete")
    result = {"status": "PASS", "sampling_replicate": replicate, "arm": arm, "checkpoint_sha256": meta["sha256"], "all_tours_valid": True, "datasets": list(DATASETS), "evaluation_wall_s": wall, "results_sha256": sha256_file(eval_csv(replicate, arm))}
    write_json(run_dir(replicate, arm) / "evaluation_integrity.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def detail(replicate: str, arm: str, dataset: str) -> pd.DataFrame:
    if replicate == "Original":
        base = P4B1 if arm == "R0" else P4B2
        label_value = "R0_Random_M5" if arm == "R0" else "R2_RandomArchive_M5"
        path = base / "evaluation_work" / label_value / dataset / "instance_metrics.parquet"
    else:
        path = run_dir(replicate, arm) / "evaluation_work" / eval_label(replicate, arm) / dataset / "instance_metrics.parquet"
    return pd.read_parquet(path)


def hard_detail(dataset: str) -> pd.DataFrame:
    return pd.read_parquet(p4b1.CYCLES / "C04" / "evaluation_work" / "M5" / dataset / "instance_metrics.parquet")


def paired_dataset(replicate: str, dataset: str) -> pd.DataFrame:
    pieces = []
    for arm in ARMS:
        frame = detail(replicate, arm, dataset)[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
        if not frame.valid_tour.all():
            raise RuntimeError(f"invalid tour {replicate}/{arm}/{dataset}")
        frame = frame.rename(columns={"coordinate_hash": f"coordinate_hash_{arm}", "n_nodes": f"n_nodes_{arm}", "optimal_cost_int": f"optimal_cost_int_{arm}", "gap_percent": f"gap_{arm}"}).drop(columns="valid_tour")
        pieces.append(frame)
    merged = pieces[0].merge(pieces[1], on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        if not merged[f"{field}_R0"].astype(str).eq(merged[f"{field}_R2"].astype(str)).all():
            raise RuntimeError(f"evaluation alignment failure {replicate}/{dataset}/{field}")
    return pd.DataFrame({"sampling_replicate": replicate, "dataset": dataset, "instance_id": merged.instance_id.astype(str), "coordinate_hash": merged.coordinate_hash_R0.astype(str), "n_nodes": merged.n_nodes_R0.astype(int), "optimal_cost_int": merged.optimal_cost_int_R0.astype(np.int64), "gap_R0": merged.gap_R0.astype(float), "gap_R2": merged.gap_R2.astype(float), "Delta_R2_minus_R0": merged.gap_R2.astype(float) - merged.gap_R0.astype(float)})


def verify_evaluations() -> None:
    sources = []
    for arm in ARMS:
        source = P4B1 / "r0_m5_external_benchmark_results.csv" if arm == "R0" else P4B2 / "r2_m5_external_benchmark_results.csv"
        sources.append(pd.read_csv(source).set_index("dataset"))
    for replicate in NEW_REPLICATES:
        for arm in ARMS:
            info = json.loads((run_dir(replicate, arm) / "evaluation_integrity.json").read_text())
            if info.get("status") != "PASS":
                raise RuntimeError("missing evaluation")
            frame = pd.read_csv(eval_csv(replicate, arm))
            if set(frame.checkpoint_sha256.astype(str)) != {sha256_file(checkpoint_path(replicate, arm))}:
                raise RuntimeError("evaluation checkpoint mismatch")
            sources.append(frame.set_index("dataset"))
    for dataset in DATASETS:
        if len({str(frame.at[dataset, "dataset_sha256"]) for frame in sources}) != 1 or len({str(frame.at[dataset, "evaluator_config_sha256"]) for frame in sources}) != 1:
            raise RuntimeError(f"evaluation protocol mismatch {dataset}")


def summarize(replicate: str, name: str, frame: pd.DataFrame) -> dict[str, Any]:
    values = frame.Delta_R2_minus_R0.to_numpy(float)
    low, high = p4b1.paired_bootstrap(values, p4b1.stable_seed(f"P4C2|{replicate}|{name}"))
    return {"sampling_replicate": replicate, "dataset": name, "n_instances": len(frame), "mean_gap_R0": float(frame.gap_R0.mean()), "mean_gap_R2": float(frame.gap_R2.mean()), "Delta_R2_minus_R0": float(values.mean()), "Delta_ci95_low": low, "Delta_ci95_high": high, "R0_instance_wins": int(np.sum(values > 0)), "R2_instance_wins": int(np.sum(values < 0)), "ties": int(np.sum(values == 0)), "bootstrap_replicates": BOOTSTRAP_REPLICATES}


def make_final_figures(per_rep: pd.DataFrame) -> None:
    macro = per_rep[per_rep.dataset.eq("GAP8_macro")].set_index("sampling_replicate").loc[list(ALL_REPLICATES)]
    x = np.arange(3); width = .34
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(x-width/2, macro.mean_gap_R0, width, label="R0"); ax.bar(x+width/2, macro.mean_gap_R2, width, label="R2")
    ax.set_xticks(x); ax.set_xticklabels(ALL_REPLICATES); ax.set_ylabel("GAP8 (%)"); ax.set_title("Data-sampling replicate GAP8"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "03_gap8_by_sampling_replicate.png", dpi=180); plt.close(fig)
    base = per_rep[per_rep.dataset.isin(DATASETS)].pivot(index="dataset", columns="sampling_replicate", values="Delta_R2_minus_R0").loc[list(DATASETS), list(ALL_REPLICATES)]
    limit = np.abs(base.to_numpy()).max(); fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(base.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(3)); ax.set_xticklabels(ALL_REPLICATES); ax.set_yticks(range(8)); ax.set_yticklabels(DATASETS); ax.set_title("Delta = R2 - R0 gap (pp)")
    for i in range(8):
        for j in range(3): ax.text(j, i, f"{base.iloc[i,j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(FIG / "04_benchmark_delta_heatmap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, replicate in enumerate(ALL_REPLICATES):
        plot = per_rep[(per_rep.sampling_replicate.eq(replicate)) & per_rep.dataset.isin(DATASETS)].set_index("dataset").loc[list(DATASETS)].reset_index(); y = np.arange(8) + (offset-1)*.18
        value = plot.Delta_R2_minus_R0.to_numpy(); low = plot.Delta_ci95_low.to_numpy(); high = plot.Delta_ci95_high.to_numpy(); ax.errorbar(value, y, xerr=np.vstack([value-low, high-value]), fmt="o", capsize=2, label=replicate)
    ax.axvline(0, color="black", linewidth=.8); ax.set_yticks(range(8)); ax.set_yticklabels(DATASETS); ax.set_xlabel("R2 - R0 gap (pp)"); ax.set_title("Per-replicate paired bootstrap 95% CI"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "05_paired_intervals.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arm in zip(axes, ARMS, strict=True):
        original_base = P4B1 if arm == "R0" else P4B2
        ax.plot(pd.read_csv(original_base / "training_cycle" / "09_bq_training" / "training_loss_curve.csv").mean_training_loss, label="Original")
        for replicate in NEW_REPLICATES:
            ax.plot(pd.read_csv(run_dir(replicate, arm) / "training_cycle" / "09_bq_training" / "training_loss_curve.csv").mean_training_loss, label=replicate)
        ax.set_title(arm); ax.set_xlabel("epoch"); ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel("mean training loss"); fig.suptitle("Fixed-seed training trajectories"); fig.tight_layout(); fig.savefig(FIG / "06_training_loss.png", dpi=180); plt.close(fig)


def finalize() -> dict[str, Any]:
    verify_evaluations(); write_training_manifest(); write_label_report()
    dev = set(pd.read_csv(ROOT / "target_aware_pilot" / "03_tsplib48_reference_manifest.csv").instance_id.astype(str))
    held = set(json.loads((ROOT / "target_aware_pilot" / "p1_heldout22_names.json").read_text())["names"])
    rows = []; instance_parts = []; rep_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for replicate in ALL_REPLICATES:
        for dataset in DATASETS:
            frame = paired_dataset(replicate, dataset); rep_frames[(replicate, dataset)] = frame; instance_parts.append(frame); rows.append(summarize(replicate, dataset, frame))
        tsplib = rep_frames[(replicate, "OriginalTSPLIB70")]
        if set(tsplib.instance_id) != dev | held:
            raise RuntimeError("TSPLIB split mismatch")
        for name, names in (("TSPLIB48_development", dev), ("TSPLIB22_previously_heldout", held)):
            part = tsplib[tsplib.instance_id.isin(names)].copy(); instance_parts.append(part.assign(dataset=name)); rows.append(summarize(replicate, name, part))
    per_rep = pd.DataFrame(rows)
    macros = []
    for replicate in ALL_REPLICATES:
        part = per_rep[(per_rep.sampling_replicate.eq(replicate)) & per_rep.dataset.isin(DATASETS)]
        macros.append({"sampling_replicate": replicate, "dataset": "GAP8_macro", "n_instances": int(part.n_instances.sum()), "mean_gap_R0": float(part.mean_gap_R0.mean()), "mean_gap_R2": float(part.mean_gap_R2.mean()), "Delta_R2_minus_R0": float(part.Delta_R2_minus_R0.mean()), "Delta_ci95_low": np.nan, "Delta_ci95_high": np.nan, "R0_instance_wins": np.nan, "R2_instance_wins": np.nan, "ties": np.nan, "bootstrap_replicates": np.nan})
    per_rep = pd.concat([per_rep, pd.DataFrame(macros)], ignore_index=True)
    per_rep.to_csv(OUT / "07_per_seed_benchmark_gap.csv", index=False)
    pd.concat(instance_parts, ignore_index=True).to_csv(OUT / "support_per_instance_paired_differences.csv", index=False)
    aggregate_rows = []
    for dataset in [*DATASETS, "TSPLIB48_development", "TSPLIB22_previously_heldout", "GAP8_macro"]:
        values = per_rep[per_rep.dataset.eq(dataset)].set_index("sampling_replicate").loc[list(ALL_REPLICATES)].Delta_R2_minus_R0.to_numpy(float)
        aggregate_rows.append({"dataset": dataset, "replicate_n": 3, "mean_Delta": float(values.mean()), "median_Delta": float(np.median(values)), "sample_sd_Delta": float(values.std(ddof=1)), "min_Delta": float(values.min()), "max_Delta": float(values.max()), "R0_winning_replicate_n": int(np.sum(values > 0)), "R2_winning_replicate_n": int(np.sum(values < 0)), "all_R0": bool(np.all(values > 0)), "all_R2": bool(np.all(values < 0))})
    aggregate = pd.DataFrame(aggregate_rows); aggregate.to_csv(OUT / "08_paired_sampling_seed_analysis.csv", index=False)

    benchmark = per_rep[per_rep.dataset.isin(DATASETS)]
    macros_frame = per_rep[per_rep.dataset.eq("GAP8_macro")].set_index("sampling_replicate").loc[list(ALL_REPLICATES)]
    wins = benchmark.assign(r0_win=benchmark.Delta_R2_minus_R0.gt(0)).groupby("sampling_replicate").r0_win.sum().reindex(ALL_REPLICATES)
    new_macro_positive = bool(macros_frame.loc[list(NEW_REPLICATES)].Delta_R2_minus_R0.gt(0).all())
    new_wins = bool(wins.loc[list(NEW_REPLICATES)].ge(5).all())
    guard = benchmark[(benchmark.sampling_replicate.isin(NEW_REPLICATES)) & benchmark.dataset.isin(["Uniform100", "GaussianMixture100", "GaussianMixture500"])]
    no_severe = bool(guard.Delta_R2_minus_R0.ge(-0.25).all())
    new_macro_negative = bool(macros_frame.loc[list(NEW_REPLICATES)].Delta_R2_minus_R0.lt(0).all())
    new_r2_wins = bool(wins.loc[list(NEW_REPLICATES)].le(3).all())
    if new_macro_positive and new_wins and no_severe:
        decision = "TARGET_AWARE_RETENTION_DATA_SEED_ROBUSTNESS_SUPPORTED"; label = "SUPPORTED_PRELIMINARY"
    elif new_macro_negative and new_r2_wins:
        decision = "TARGET_AWARE_RETENTION_DATA_SEED_ROBUSTNESS_NOT_SUPPORTED"; label = "NOT_SUPPORTED"
    else:
        decision = "TARGET_AWARE_RETENTION_DATA_SEED_ROBUSTNESS_INCONCLUSIVE"; label = "INCONCLUSIVE"

    hard_gap8 = float(np.mean([hard_detail(dataset).gap_percent.mean() for dataset in DATASETS]))
    hard_wins = {replicate: bool(macros_frame.loc[replicate].mean_gap_R0 < hard_gap8) for replicate in ALL_REPLICATES}
    gapagg = aggregate[aggregate.dataset.eq("GAP8_macro")].iloc[0]
    dist = pd.read_csv(OUT / "03_archive_distribution_comparison.csv")
    train_dist = dist[dist.level.eq("training10k")]
    labeling = [json.loads((label_dir(rep, arm) / "labeling_integrity.json").read_text()) for rep in NEW_REPLICATES for arm in ARMS]
    training = [json.loads((run_dir(rep, arm) / "training_integrity.json").read_text()) for rep in NEW_REPLICATES for arm in ARMS]
    evaluation = [json.loads((run_dir(rep, arm) / "evaluation_integrity.json").read_text()) for rep in NEW_REPLICATES for arm in ARMS]
    sampling_info = json.loads((OUT / "sampling_integrity.json").read_text())
    make_final_figures(per_rep)

    report = ["# Data-Sampling Robustness Report", "", "Delta = GAP(R2) - GAP(R0); positive favors target-aware R0.", "", "| Replicate | R0 GAP8 | R2 GAP8 | Delta | R0 benchmark wins |", "|---|---:|---:|---:|---:|"]
    for replicate in ALL_REPLICATES:
        row = macros_frame.loc[replicate]
        report.append(f"| {replicate} | {row.mean_gap_R0:.4f} | {row.mean_gap_R2:.4f} | {row.Delta_R2_minus_R0:+.4f} | {int(wins.loc[replicate])}/8 |")
    report.extend(["", f"- Across three sampling replicates: mean GAP8 Delta {gapagg.mean_Delta:+.4f} pp, median {gapagg.median_Delta:+.4f}, sample SD {gapagg.sample_sd_Delta:.4f}, range [{gapagg.min_Delta:+.4f}, {gapagg.max_Delta:+.4f}].", f"- R0 beats Original Hardness M5 GAP8 ({hard_gap8:.4f}%) by replicate: {hard_wins}.", f"- Preregistered guards: both new GAP8 positive = {new_macro_positive}; at least 5/8 R0 wins in both = {new_wins}; no severe Uniform/GM regression = {no_severe}.", "- R0 Archive30k is identical across replicates. Thus observed R0 variation is final-acquisition sampling; R2 variation includes both uniform retention and acquisition.", "- TSPLIB22 remains exploratory and is not pristine held-out evidence.", f"- Costs: sampling {sampling_info['sampling_wall_seconds']:.1f}s; labeling {sum(x['pipeline_wall_seconds'] for x in labeling):.1f}s ({sum(x['physical_new_concorde_calls'] for x in labeling):,} physical calls); training {sum(x['training_loop_wall_s'] for x in training):.1f}s; evaluation {sum(x['evaluation_wall_s'] for x in evaluation):.1f}s.", "- Training seed is fixed at 2026. P4-C1 and P4-C2 cannot be pooled as a fully crossed two-factor experiment, and cross-FOGA-pool robustness remains untested."])
    (OUT / "09_data_sampling_robustness_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_text = f"""# P4-C2 Final Decision

## `{decision}` — **{label}**

- Both new sampling replicates have positive GAP8 Delta: **{new_macro_positive}**.
- R0 wins at least 5/8 benchmark means in both new replicates: **{new_wins}**.
- No preregistered severe Uniform/Gaussian-mixture regression: **{no_severe}**.
- Three-replicate GAP8 Delta: mean {gapagg.mean_Delta:+.4f} pp, sample SD {gapagg.sample_sd_Delta:.4f}, range [{gapagg.min_Delta:+.4f}, {gapagg.max_Delta:+.4f}].
- Scope limitation: R0 retention is deterministic, so this supports data/final-acquisition sampling stability conditional on one fixed target-aware archive and one E0 pool; it does not establish R0 retention-seed or cross-evolution-pool robustness.
- No Diversity-Only, Fresh10k, new evolution, or follow-up experiment was started.

Decision label: `{label}`

`TARGET_AWARE_P4C2_DATA_SAMPLING_ROBUSTNESS_COMPLETE`
"""
    (OUT / "10_final_decision.md").write_text(decision_text, encoding="utf-8")
    write_json(OUT / "run_metadata.json", {"status": "COMPLETE", "decision": decision, "decision_label": label, "gap8_delta_mean": gapagg.mean_Delta, "gap8_delta_sample_sd": gapagg.sample_sd_Delta, "r0_retention_random": False, "p4c2_scope": "fixed R0 archive final-acquisition plus R2 two-stage sampling within fixed E0 pool", "no_followup_started": True, "per_seed_csv_sha256": sha256_file(OUT / "07_per_seed_benchmark_gap.csv"), "aggregate_csv_sha256": sha256_file(OUT / "08_paired_sampling_seed_analysis.csv")})
    Status().update(phase="complete", status="COMPLETE", decision=decision, decision_label=label)
    return {"status": "COMPLETE", "decision": decision, "label": label, "gap8": macros_frame[["mean_gap_R0", "mean_gap_R2", "Delta_R2_minus_R0"]].reset_index().to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "sample", "label", "train", "evaluate", "finalize"))
    parser.add_argument("--replicate", choices=NEW_REPLICATES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "audit": result = audit()
    elif args.phase == "sample": result = sample()
    elif args.phase in ("label", "train", "evaluate"):
        if args.replicate is None or args.arm is None:
            parser.error("--replicate and --arm are required")
        if args.phase == "label": result = label(args.replicate, args.arm)
        elif args.phase == "train": result = train(args.replicate, args.arm, args.device)
        else: result = evaluate(args.replicate, args.arm, args.device)
    else: result = finalize()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
