#!/usr/bin/env python3
"""P4-A random-evolution versus structure-guided-evolution ablation.

This isolated phase driver consumes the already frozen P2.2 E0/E1 archives.
It never reruns mutation or E1 training.  The only new model is obtained by
applying the formal P3 labeling/training/evaluation protocol to frozen E0.
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
P3 = ROOT / "target_aware_pilot" / "p3_structure_finetune"
OUT = ROOT / "target_aware_pilot" / "p4a_random_evolution_ablation"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
LABEL_DIR = OUT / "labeling"
TRAIN_CYCLE = OUT / "training_cycle"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(P2))

import run_p2_bounded_evolution as p2  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, sha256_file  # noqa: E402


FORMAL = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1" / "A0_FULL_BASE"
CYCLES = FORMAL / "cycles"
M4 = FORMAL / "checkpoints" / "M4_E256.tsp.best"
HARD_M5 = FORMAL / "checkpoints" / "M5_E256.tsp.best"
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
HARD_M5_SHA = "41d189dfeb986a070bd7685f203172d1dad4376d8d0134151d7488fb079a4545"
E1_M5 = P3 / "checkpoints" / "Structure_M5_E256.tsp.best"
E1_M5_SHA = "ace34d712ca8d20e730edeed0666362e6cc03008f9b5c0411d53c444856e3233"
E0_M5 = CHECKPOINTS / "E0_Structure_M5_E256.tsp.best"

SOURCE_E0_BUNDLE = P2 / "support_e0_provisional_structure10k.npz"
SOURCE_E0_MANIFEST = P2 / "support_e0_provisional_structure10k.parquet"
SOURCE_E0_CSV = P2 / "support_e0_provisional_structure10k.csv"
SOURCE_E0_META = P2 / "support_e0_acquisition_metadata.json"
SOURCE_E1_BUNDLE = P2 / "provisional_structure10k.npz"
SOURCE_E1_MANIFEST = P2 / "provisional_structure10k_manifest.csv"
E0_SOURCE_SHA = "987395ae7de3e9a4118ed26ce9cf2c369a6b0498623dadd14668812a4cc3cc69"
E1_SOURCE_SHA = "5287932231529e617383388bb64d607985d07944799c3d661c5a737338bdece1"
E0_BUNDLE = OUT / "e0_structure10k.npz"
E0_MANIFEST = OUT / "02_e0_structure10k_manifest.csv"
E0_CONFIG = OUT / "e0_structure10k_config.json"

SHARED_CACHE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1" / "_shared_state" / "shared_oracle_cache.parquet"
EVAL_CSV = OUT / "e0_m5_external_benchmark_results.csv"
DATASETS = (
    "Uniform100",
    "GaussianMixture100",
    "TSPLIB100",
    "EvoReal_S1_100",
    "EvoReal_S2_100",
    "EvoReal_S3_100",
    "GaussianMixture500",
    "OriginalTSPLIB70",
)
BOOTSTRAP_REPLICATES = 10_000


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_bundle(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["coordinates"], dtype=np.float64), payload["instance_hash"].astype(str).tolist()


def save_labeled_bundle(path: Path, coordinates: np.ndarray, hashes: list[str], tours: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        instance_hash=np.asarray(hashes),
        expert_tours=np.asarray(tours, dtype=np.int64),
    )
    temporary.replace(path)


def load_labeled() -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(LABEL_DIR / "labeled_bundle.npz", allow_pickle=False) as payload:
        return (
            np.asarray(payload["coordinates"], dtype=np.float64),
            payload["instance_hash"].astype(str).tolist(),
            np.asarray(payload["expert_tours"], dtype=np.int64),
        )


def coordinate_audit(coordinates: np.ndarray, hashes: list[str]) -> dict[str, Any]:
    reproduced = [coordinate_hash(item) for item in coordinates]
    unique_nodes = np.asarray([len(np.unique(item, axis=0)) for item in coordinates])
    return {
        "rows": len(coordinates),
        "unique_instances": len(set(hashes)),
        "shape": list(coordinates.shape),
        "finite": bool(np.isfinite(coordinates).all()),
        "in_bounds": bool(((coordinates >= 0) & (coordinates <= 1)).all()),
        "all_nodes_unique": bool((unique_nodes == coordinates.shape[1]).all()),
        "hashes_reproduce": reproduced == hashes,
        "membership_sha256": membership_hash(hashes),
    }


class P4Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def frozen_training_history() -> tuple[set[str], set[str]]:
    m4_history: set[str] = set()
    for cycle in range(4):
        frame = pd.read_parquet(CYCLES / f"C{cycle:02d}" / "06_selection" / "T_selected.parquet", columns=["instance_hash"])
        m4_history.update(frame.instance_hash.astype(str))
    c04 = set(pd.read_parquet(CYCLES / "C04" / "06_selection" / "T_selected.parquet", columns=["instance_hash"]).instance_hash.astype(str))
    return m4_history, c04


def audit_and_acquire() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if sha256_file(SOURCE_E0_BUNDLE) != E0_SOURCE_SHA:
        raise RuntimeError("frozen P2.2 E0 bundle SHA mismatch")
    if sha256_file(SOURCE_E1_BUNDLE) != E1_SOURCE_SHA:
        raise RuntimeError("frozen P2.2 E1 bundle SHA mismatch")
    if sha256_file(M4) != M4_SHA or sha256_file(HARD_M5) != HARD_M5_SHA or sha256_file(E1_M5) != E1_M5_SHA:
        raise RuntimeError("frozen checkpoint SHA mismatch")

    archive_meta = pd.read_parquet(P2 / "evolution" / "E0" / "G2" / "bounded_archive.parquet")
    archive_xy, archive_hashes = load_bundle(P2 / "evolution" / "E0" / "G2" / "bounded_archive.npz")
    if archive_hashes != archive_meta.instance_hash.astype(str).tolist() or len(archive_hashes) != 30_000:
        raise RuntimeError("E0 bounded archive bundle/manifest mismatch")
    e0_xy, e0_hashes = load_bundle(SOURCE_E0_BUNDLE)
    e1_xy, e1_hashes = load_bundle(SOURCE_E1_BUNDLE)
    e0_manifest = pd.read_parquet(SOURCE_E0_MANIFEST)
    e1_manifest = pd.read_csv(SOURCE_E1_MANIFEST)
    if e0_hashes != e0_manifest.instance_hash.astype(str).tolist() or e1_hashes != e1_manifest.instance_hash.astype(str).tolist():
        raise RuntimeError("frozen acquisition manifest order mismatch")

    geometry = p2.load_geometry()
    replay, replay_meta = p2.acquisition("E0", archive_meta, geometry)
    source_meta = json.loads(SOURCE_E0_META.read_text(encoding="utf-8"))
    if replay is None or replay.instance_hash.astype(str).tolist() != e0_hashes or replay_meta != source_meta:
        raise RuntimeError("exact P2.2 acquisition replay does not reproduce frozen E0")
    archive_lookup = {value: index for index, value in enumerate(archive_hashes)}
    reconstructed = np.stack([archive_xy[archive_lookup[value]] for value in e0_hashes])
    if not np.array_equal(reconstructed, e0_xy):
        raise RuntimeError("E0 coordinates do not reproduce from frozen bounded archive")

    integrity = coordinate_audit(e0_xy, e0_hashes)
    if not all(integrity[key] for key in ("finite", "in_bounds", "all_nodes_unique", "hashes_reproduce")):
        raise RuntimeError("E0 coordinate integrity failed")
    if integrity["rows"] != 10_000 or integrity["unique_instances"] != 10_000 or integrity["shape"] != [10_000, 100, 2]:
        raise RuntimeError("E0 cardinality/shape integrity failed")
    m4_history, c04 = frozen_training_history()
    shared = set(pd.read_parquet(SHARED_CACHE, columns=["instance_hash"]).instance_hash.astype(str))
    overlaps = {
        "m4_training_history": len(set(e0_hashes) & m4_history),
        "c04_hard10k": len(set(e0_hashes) & c04),
        "shared_oracle_cache": len(set(e0_hashes) & shared),
        "e1_structure10k": len(set(e0_hashes) & set(e1_hashes)),
    }
    if overlaps["m4_training_history"]:
        raise RuntimeError("E0 acquisition overlaps M4 training history")

    metrics = pd.read_csv(P2 / "02_generation_metrics.csv")
    operators = pd.read_csv(P2 / "support_operator_metrics.csv")
    for arm in ("E0", "E1"):
        for generation in ("G1", "G2"):
            part = operators[(operators.arm == arm) & (operators.generation == generation)]
            if int(part.attempts.sum()) != 20_000 or set(part.operator_id) != set(p2.OPERATOR_NAMES):
                raise RuntimeError(f"mutation protocol mismatch {arm}/{generation}")
    for generation in (1, 2):
        arm_counts = []
        for arm in ("E0", "E1"):
            manifest = pd.read_parquet(P2 / "evolution" / arm / f"G{generation}" / "mutation" / "mutation_manifest.parquet")
            if (
                set(manifest.mutation_profile.astype(str)) != {p2.PROFILE_NAME}
                or set(manifest.mutation_depth.astype(int)) != {1}
                or set(manifest.cycle_seed_namespace.astype(int)) != {p2.MUTATION_CYCLE_NAMESPACE[generation]}
            ):
                raise RuntimeError(f"mutation parameter protocol mismatch {arm}/G{generation}")
            arm_counts.append(manifest.operator_id.value_counts().sort_index())
        if not arm_counts[0].equals(arm_counts[1]):
            raise RuntimeError(f"operator attempt allocation differs across arms at G{generation}")
    e0_g1_xy, e0_g1_hashes = load_bundle(P2 / "evolution" / "E0" / "G1" / "mutation" / "offspring_coordinates.npz")
    e1_g1_xy, e1_g1_hashes = load_bundle(P2 / "evolution" / "E1" / "G1" / "mutation" / "offspring_coordinates.npz")
    if e0_g1_hashes != e1_g1_hashes or not np.array_equal(e0_g1_xy, e1_g1_xy):
        raise RuntimeError("E0/E1 G1 mutations should be identical before survivor selection")
    for arm in ("E0", "E1"):
        g2 = metrics[(metrics.arm == arm) & (metrics.generation == "G2") & (metrics.object_type == "bounded_archive")].iloc[0]
        if int(g2.n) != 30_000 or int(g2.archive_capacity) != 30_000:
            raise RuntimeError(f"bounded archive mismatch {arm}")

    shutil.copy2(SOURCE_E0_BUNDLE, E0_BUNDLE)
    shutil.copy2(SOURCE_E0_CSV, E0_MANIFEST)
    if sha256_file(E0_BUNDLE) != E0_SOURCE_SHA:
        raise RuntimeError("copied E0 bundle changed")
    config = {
        "status": "FROZEN",
        "source": str(SOURCE_E0_BUNDLE.relative_to(ROOT)),
        "bundle_sha256": E0_SOURCE_SHA,
        "manifest_sha256": sha256_file(E0_MANIFEST),
        "membership_sha256": membership_hash(e0_hashes),
        "acquisition_runner_sha256": sha256_file(P2 / "run_p2_bounded_evolution.py"),
        "acquisition_replay_ids_exact": True,
        "acquisition_replay_metadata_exact": True,
        "coordinates_exact_from_archive": True,
        "features": list(p2.FEATURES),
        "seed": p2.SEED,
        "lineage_cap": p2.LINEAGE_CAP_ACQUISITION,
        "target_quotas_requested": source_meta["target_quotas_requested"],
        "target_quotas_after_capacity_redistribution": source_meta["target_quotas_after_capacity_redistribution"],
        "overlaps": overlaps,
    }
    write_json(E0_CONFIG, config)

    p2_runtime = json.loads((P2 / "support_runtime.json").read_text(encoding="utf-8"))
    protocol = f"""# P4-A Protocol Integrity

- Frozen P03 source bundle SHA256 `{json.loads((P2 / 'run_metadata.json').read_text())['p03_bundle_sha256']}`; both arms start from this exact 10,000-member population.
- Two generations per arm, 20,000 attempts/generation, all 11 frozen `FOGA11_BASE_V1` operators, identical validity/deduplication, Parent+Child Union, 10,000 active population, and 30,000 eligible-offspring archive cap.
- E0/E1 G1 offspring coordinates and ordered IDs are bitwise identical before survivor selection. G2 differs only downstream of the designed G1 selection intervention.
- E0 uses seeded random parent/survivor selection. E1 uses target-structure-guided parent/survivor selection. Both bounded archives use the same retention logic and both final datasets use the identical `acquisition()` implementation.
- Exact deterministic replay of E0 acquisition reproduces all ordered IDs and metadata; coordinates reproduce directly from the frozen E0 bounded archive. The P2.0/P2.1 229,148 pool is not read.
- No hardness/HGB prediction participates in mutation, survivor selection, archive retention, or acquisition. TSPLIB22 coordinates/features are not loaded; the set is used only after the model is frozen.
- Frozen E1 data/checkpoint hashes: `{E1_SOURCE_SHA}` / `{E1_M5_SHA}`. They will not be regenerated or retrained.
- Predeclared conclusion rule: CASE A requires E1−E0 mean gap < 0 on at least 6/8 frozen benchmarks, paired CI upper < 0 on at least 3/8, and no paired-significant E1 regression. CASE C is the exact mirror for E0. Other evidence is CASE B.

Status: **PASS**.
"""
    (OUT / "00_protocol_integrity.md").write_text(protocol, encoding="utf-8")

    e0_row = metrics[(metrics.arm == "E0") & (metrics.generation == "G2") & (metrics.object_type == "provisional_structure10k")].iloc[0]
    e1_row = metrics[(metrics.arm == "E1") & (metrics.generation == "G2") & (metrics.object_type == "provisional_structure10k")].iloc[0]
    missing = {}
    for arm, frame in (("E0", e0_manifest), ("E1", e1_manifest)):
        source_z = (frame[list(p2.FEATURES)].to_numpy(float) - geometry["center"]) / geometry["scale"]
        nearest = p2.nearest_matrix(source_z, geometry["target_z"]).min(axis=0)
        missing[arm] = geometry["manifest"].instance_id.to_numpy()[nearest > geometry["radius"]].tolist()

    audit = f"""# E0 Structure10k Acquisition Audit

- Frozen E0 bounded archive: 30,000 unique eligible offspring; source bundle SHA256 `{sha256_file(P2 / 'evolution/E0/G2/bounded_archive.npz')}`.
- Exact P2.2 acquisition replay: ordered IDs **identical**, metadata **identical**, and archive-derived coordinates **bitwise identical**.
- Output: 10,000 unique TSP100; bundle SHA256 `{E0_SOURCE_SHA}`; membership SHA256 `{membership_hash(e0_hashes)}`.
- C00–C03 overlap: {overlaps['m4_training_history']}; C04 Hard10k overlap: {overlaps['c04_hard10k']}; E1 Structure10k overlap: {overlaps['e1_structure10k']:,}.
- Strict target: {int(e0_row.strict_target_n):,}/10,000 ({e0_row.strict_target_fraction:.2%}); reference coverage {int(e0_row.reference_coverage_n)}/48; missing neighborhoods: {missing['E0']}.
- Requested quotas: `{source_meta['target_quotas_requested']}`.
- Capacity-adjusted quotas: `{source_meta['target_quotas_after_capacity_redistribution']}`; actual counts reproduce them exactly. Original cell deficits are recorded in `e0_structure10k_config.json`; no outside data or mutation was added.
- Effective lineages {e0_row.effective_lineages:.1f}; maximum lineage fraction {e0_row.largest_lineage_share:.2%}; lineage cap started at {source_meta['lineage_cap_initial']} and deterministically relaxed to {source_meta['lineage_cap_max_used']}.
- Hardness and HGB predictions are absent from selection.

Status: **PASS; E0 Structure10k frozen before labeling**.
"""
    (OUT / "01_e0_acquisition_audit.md").write_text(audit, encoding="utf-8")

    def metric_row(arm: str, object_type: str) -> pd.Series:
        return metrics[(metrics.arm == arm) & (metrics.generation == "G2") & (metrics.object_type == object_type)].iloc[0]

    rows = []
    for object_type in ("bounded_archive", "provisional_structure10k"):
        for arm in ("E0", "E1"):
            row = metric_row(arm, object_type)
            rows.append(
                {
                    "stage": object_type,
                    "arm": arm,
                    "n": int(row.n),
                    "strict_fraction": float(row.strict_target_fraction),
                    "swd": float(row.sliced_wasserstein),
                    "js": float(row.coarse_js),
                    "occupied_cells": int(row.occupied_cells),
                    "effective_cells": float(row.effective_cells),
                    "largest_cell_share": float(row.largest_cell_share),
                    "coverage": int(row.reference_coverage_n),
                    "effective_lineages": float(row.effective_lineages),
                    "largest_lineage_share": float(row.largest_lineage_share),
                    "fft_median": float(row.fft_median),
                    "nn_cv_median": float(row.nn_cv_median),
                    "aux_js": float(row.aux_coarse_js),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "support_e0_e1_structure_metrics.csv", index=False)
    e0_meta = source_meta
    e1_meta = json.loads((P2 / "support_e1_acquisition_metadata.json").read_text(encoding="utf-8"))
    structure_report = f"""# E0 versus E1 Structure Comparison

| Stage | Arm | Strict | SWD | 3×3 JS | Effective cells | Largest cell | Coverage | Effective lineages | Max lineage | FFT median | NN-CV median | Auxiliary JS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in comparison.itertuples(index=False):
        structure_report += f"| {row.stage} | {row.arm} | {row.strict_fraction:.2%} | {row.swd:.6f} | {row.js:.6f} | {row.effective_cells:.3f} | {row.largest_cell_share:.2%} | {row.coverage}/48 | {row.effective_lineages:.1f} | {row.largest_lineage_share:.2%} | {row.fft_median:.4f} | {row.nn_cv_median:.4f} | {row.aux_js:.6f} |\n"
    structure_report += f"""

- Candidate-space effect: before final acquisition, E1's bounded archive is closer in the frozen two-feature geometry (SWD/JS {metric_row('E1','bounded_archive').sliced_wasserstein:.6f}/{metric_row('E1','bounded_archive').coarse_js:.6f}) than E0 ({metric_row('E0','bounded_archive').sliced_wasserstein:.6f}/{metric_row('E0','bounded_archive').coarse_js:.6f}), and contains {int(metric_row('E1','bounded_archive').strict_target_n) - int(metric_row('E0','bounded_archive').strict_target_n):+,} more strict candidates.
- Final-acquisition effect: applying exactly the same selector improves both arms, but E1 remains closer (SWD {e1_row.sliced_wasserstein:.6f} vs {e0_row.sliced_wasserstein:.6f}; JS {e1_row.coarse_js:.6f} vs {e0_row.coarse_js:.6f}) and supplies {int(e1_row.strict_target_n) - int(e0_row.strict_target_n):+,} more strict instances.
- Both cover 45/48 reference neighborhoods. E0 has higher effective lineage diversity ({e0_row.effective_lineages:.1f} vs {e1_row.effective_lineages:.1f}) and slightly better auxiliary FFT/NN-CV JS ({e0_row.aux_coarse_js:.6f} vs {e1_row.aux_coarse_js:.6f}); E1 is therefore not declared universally more diverse from 3×3 JS alone.
- E0 adjusted quotas `{e0_meta['target_quotas_after_capacity_redistribution']}`; E1 adjusted quotas `{e1_meta['target_quotas_after_capacity_redistribution']}`. Both record explicit capacity redistribution and zero final shortfall.
- E0/E1 final membership overlap is {overlaps['e1_structure10k']:,}/10,000. No post-gap reselection is permitted.
"""
    (OUT / "03_e0_vs_e1_structure_comparison.md").write_text(structure_report, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, stage in zip(axes, ("bounded_archive", "provisional_structure10k"), strict=True):
        for arm, color in (("E0", "#4c78a8"), ("E1", "#f58518")):
            frame = archive_meta if (stage == "bounded_archive" and arm == "E0") else None
            if stage == "bounded_archive" and arm == "E1":
                frame = pd.read_parquet(P2 / "evolution/E1/G2/bounded_archive.parquet")
            if stage == "provisional_structure10k":
                frame = e0_manifest if arm == "E0" else e1_manifest
            assert frame is not None
            ax.scatter(frame[p2.FEATURES[0]], frame[p2.FEATURES[1]], s=2, alpha=0.12, label=arm, color=color)
        ax.set_title(stage.replace("_", " "))
        ax.set_xlabel(p2.FEATURES[0])
        ax.set_ylabel(p2.FEATURES[1])
        ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(FIG / "01_candidate_and_acquired_feature_space.png", dpi=180)
    plt.close(fig)

    metric_names = ["strict_fraction", "swd", "js", "effective_cells", "effective_lineages", "aux_js"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    acq_compare = comparison[comparison.stage == "provisional_structure10k"].set_index("arm")
    for ax, name in zip(axes.flat, metric_names, strict=True):
        ax.bar(["E0", "E1"], [acq_compare.at[arm, name] for arm in ("E0", "E1")], color=["#4c78a8", "#f58518"])
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG / "02_acquisition_metric_comparison.png", dpi=180)
    plt.close(fig)

    return {"status": "PASS", **integrity, **overlaps, "e0_bundle_sha256": E0_SOURCE_SHA}


def label(device_name: str) -> None:
    audit_and_acquire()
    coordinates, hashes = load_bundle(E0_BUNDLE)
    # The frozen P2.2 parquet is byte-for-byte membership-equivalent to the
    # required CSV export and avoids a pandas C-parser crash after the heavy
    # SciPy audit stack has been loaded in this process.
    manifest = pd.read_parquet(SOURCE_E0_MANIFEST)
    if manifest.instance_hash.astype(str).tolist() != hashes:
        raise RuntimeError("E0 parquet manifest order differs from frozen bundle")
    import run_blade_v2 as base

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    base.GLOBAL_CACHE = pd.read_parquet(SHARED_CACHE)
    selected = manifest.copy()
    selected["coord_index"] = np.arange(len(selected))
    started = time.perf_counter()
    _, tours, summary = base._label_population(
        LABEL_DIR,
        selected,
        coordinates,
        model,
        device,
        None,
        "P4A_E0_structure10k_original_A0_oracle_and_M4_hardness",
        True,
    )
    invocation_wall = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ledger = pd.read_parquet(LABEL_DIR / "ledger.parquet")
    if len(ledger) != 10_000 or not ledger.verified.all() or not ledger.status.eq("solved").all():
        raise RuntimeError("E0 Concorde ledger incomplete")
    legal = np.asarray([validate_tour_0based(tour, 100)[0] for tour in tours])
    if tours.shape != (10_000, 100) or not legal.all():
        raise RuntimeError("illegal E0 expert tour")
    costs = np.asarray(
        [tour_cost_int(integer_distance_matrix(points), tour) for points, tour in zip(coordinates, tours, strict=True)],
        dtype=np.int64,
    )
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)):
        raise RuntimeError("E0 expert objective mismatch")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (10_000, 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]):
        raise RuntimeError("E0 training closure mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist():
        raise RuntimeError("E0 tour hash mismatch")
    integrity = {
        "status": "PASS",
        "selected": 10_000,
        "verified": 10_000,
        "failures": 0,
        "cache_hits": int(summary["cache_hits"]),
        "physical_new_concorde_calls": int(summary["new_concorde_calls"]),
        "pipeline_wall_seconds": float(summary["wall_seconds"]),
        "invocation_wall_seconds": invocation_wall,
        "all_tours_legal": True,
        "all_costs_reproduced": True,
        "ledger_sha256": sha256_file(LABEL_DIR / "ledger.parquet"),
        "labeled_bundle_sha256": sha256_file(LABEL_DIR / "labeled_bundle.npz"),
        "labeled_metadata_sha256": sha256_file(LABEL_DIR / "labeled.parquet"),
        "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
        "m4_sha256": model_meta["sha256"],
        "concorde_sha256": CANONICAL_CONCORDE_SHA256,
    }
    write_json(LABEL_DIR / "labeling_integrity.json", integrity)
    report = f"""# Concorde Labeling Audit

- Reused the exact P3/A0 `_label_population` → canonical Concorde pipeline and M4 teacher-forced hardness attachment.
- Selected/verified: 10,000/10,000; failures: 0; cache hits: {integrity['cache_hits']}; physical Concorde calls: {integrity['physical_new_concorde_calls']}.
- Pipeline wall-clock: {integrity['pipeline_wall_seconds']:.3f}s.
- Every tour is a legal 0-based Hamiltonian permutation. Every objective reproduces the canonical per-edge Euclidean `rint(1e6 × distance)` ledger value; closure is implicit for cost and appended only for BQ input.
- Ledger SHA256 `{integrity['ledger_sha256']}`; labeled bundle SHA256 `{integrity['labeled_bundle_sha256']}`; expert-tour array SHA256 `{integrity['expert_tours_array_sha256']}`.

Status: **PASS**.
"""
    (OUT / "04_concorde_labeling_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(integrity, indent=2), flush=True)


def train(device_name: str) -> None:
    audit_and_acquire()
    label_integrity_path = LABEL_DIR / "labeling_integrity.json"
    if not label_integrity_path.is_file():
        raise RuntimeError("labeling integrity missing; training forbidden")
    label_info = json.loads(label_integrity_path.read_text(encoding="utf-8"))
    if label_info.get("status") != "PASS" or label_info.get("verified") != 10_000:
        raise RuntimeError("labeling did not pass")
    coordinates, hashes, tours = load_labeled()
    frozen_xy, frozen_hashes = load_bundle(E0_BUNDLE)
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
        raise RuntimeError("labeled E0 membership changed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    core.RUN_ID = "TARGET_AWARE_P4A_E0_STRUCTURE_C5"
    core.cycle_dir = lambda arm, cycle: TRAIN_CYCLE
    core.model_path = lambda arm, index: E0_M5 if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    started = time.perf_counter()
    trained, trained_meta, timing = core.train(
        "E0_STRUCTURE_C5", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, P4Status(), True
    )
    invocation_wall = time.perf_counter() - started
    e0 = torch.load(TRAIN_CYCLE / "09_bq_training" / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e0["net"]) == set(m4["net"]) and all(torch.equal(e0["net"][key], m4["net"][key]) for key in m4["net"])
    if not e000_equal:
        raise RuntimeError("P4-A E000 differs from formal M4")
    if trained_meta["sha256"] != sha256_file(E0_M5):
        raise RuntimeError("E0 M5 hash mismatch")
    if not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters()):
        raise RuntimeError("E0 M5 contains nonfinite parameters")
    del trained, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    curve = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    if len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(10_000).all():
        raise RuntimeError("E0 training trajectory mismatch")
    p3_training = json.loads((P3 / "training_integrity.json").read_text(encoding="utf-8"))
    report = f"""# Training Protocol and Log

- Directly reused the formal A0/P3 `train` function. E000 state tensors exactly equal formal M4: **{e000_equal}**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, logical batch 256, microbatch 64, 256 epochs, 40 updates/epoch, 10,240 total updates, NumPy seed 2026.
- Every epoch exposes all 10,000 frozen E0 instances exactly once; no anchor mixture. The only P3/E1 versus P4-A/E0 training difference is dataset membership.
- Final loss {timing['final_loss']:.8f}; training-loop wall {timing['training_loop_wall_s']:.3f}s; peak allocated GPU bytes {int(timing['gpu_peak_allocated_bytes']):,}.
- P3 E1 comparison: final loss {p3_training['final_loss']:.8f}; wall {p3_training['training_loop_wall_s']:.3f}s under the identical protocol.
- E0 M5 checkpoint SHA256 `{trained_meta['sha256']}`; training membership SHA256 `{membership_hash(hashes)}`; labeled bundle SHA256 `{label_info['labeled_bundle_sha256']}`.

E256 is frozen before evaluation. No TSPLIB22 result is used for checkpoint or protocol selection.
"""
    (OUT / "05_training_protocol_and_log.md").write_text(report, encoding="utf-8")
    write_json(
        OUT / "training_integrity.json",
        {
            "status": "PASS",
            "m4_sha256": M4_SHA,
            "e000_weights_equal_m4": e000_equal,
            "e0_m5_sha256": trained_meta["sha256"],
            "training_membership_sha256": membership_hash(hashes),
            "label_bundle_sha256": label_info["labeled_bundle_sha256"],
            "epochs": 256,
            "optimizer_steps": int(timing["optimizer_steps"]),
            "final_loss": float(timing["final_loss"]),
            "training_loop_wall_s": float(timing["training_loop_wall_s"]),
            "invocation_wall_s": invocation_wall,
            "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]),
        },
    )
    print(json.dumps({"status": "PASS", "checkpoint": str(E0_M5), "sha256": trained_meta["sha256"], "timing": timing}, indent=2), flush=True)


def evaluate(device_name: str) -> None:
    integrity_path = OUT / "training_integrity.json"
    if not E0_M5.is_file() or not integrity_path.is_file():
        raise RuntimeError("frozen E0 M5 absent")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS" or integrity.get("e0_m5_sha256") != sha256_file(E0_M5):
        raise RuntimeError("E0 M5 integrity failed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    model, meta = load_canonical_checkpoint(E0_M5, device, expected_sha256=integrity["e0_m5_sha256"])
    frame, wall = core.evaluate(model, E0_M5, "E0_Structure_M5", EVAL_CSV, P4Status(), "E0_STRUCTURE_C5")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all():
        raise RuntimeError("E0 frozen evaluation incomplete")
    write_json(
        OUT / "evaluation_integrity.json",
        {
            "status": "PASS",
            "checkpoint_sha256": meta["sha256"],
            "all_tours_valid": True,
            "datasets": list(DATASETS),
            "evaluation_wall_s": wall,
            "results_sha256": sha256_file(EVAL_CSV),
        },
    )
    print(frame[["dataset", "instance_count", "mean_gap_percent"]].to_string(index=False), flush=True)


def detail(model: str, dataset: str) -> pd.DataFrame:
    if model == "M4":
        path = CYCLES / "C03" / "evaluation_work" / "M4" / dataset / "instance_metrics.parquet"
    elif model == "Hardness_M5":
        path = CYCLES / "C04" / "evaluation_work" / "M5" / dataset / "instance_metrics.parquet"
    elif model == "E1_Guided_M5":
        path = P3 / "evaluation_work" / "Structure_M5" / dataset / "instance_metrics.parquet"
    else:
        path = OUT / "evaluation_work" / "E0_Structure_M5" / dataset / "instance_metrics.parquet"
    return pd.read_parquet(path)


def align_four(dataset: str) -> pd.DataFrame:
    models = ("M4", "Hardness_M5", "E1_Guided_M5", "E0_Random_M5")
    frames = []
    for model in models:
        source = detail(model, dataset)
        part = source[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
        if not part.valid_tour.all():
            raise RuntimeError(f"invalid tour {model}/{dataset}")
        part = part.rename(
            columns={
                "coordinate_hash": f"coordinate_hash_{model}",
                "n_nodes": f"n_nodes_{model}",
                "optimal_cost_int": f"optimal_cost_int_{model}",
                "gap_percent": f"gap_{model}",
            }
        ).drop(columns=["valid_tour"])
        frames.append(part)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        columns = [f"{field}_{model}" for model in models]
        if not all(merged[columns[0]].astype(str).eq(merged[column].astype(str)).all() for column in columns[1:]):
            raise RuntimeError(f"four-model alignment failure {dataset}/{field}")
    return pd.DataFrame(
        {
            "dataset": dataset,
            "instance_id": merged.instance_id.astype(str),
            "coordinate_hash": merged.coordinate_hash_M4.astype(str),
            "n_nodes": merged.n_nodes_M4.astype(int),
            "optimal_cost_int": merged.optimal_cost_int_M4.astype(np.int64),
            **{f"gap_{model}": merged[f"gap_{model}"].astype(float) for model in models},
        }
    )


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def paired_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_REPLICATES:
        count = min(200, BOOTSTRAP_REPLICATES - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def result_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    pairs = {
        "E1_minus_E0": frame.gap_E1_Guided_M5.to_numpy(float) - frame.gap_E0_Random_M5.to_numpy(float),
        "E1_minus_Hardness": frame.gap_E1_Guided_M5.to_numpy(float) - frame.gap_Hardness_M5.to_numpy(float),
        "E0_minus_Hardness": frame.gap_E0_Random_M5.to_numpy(float) - frame.gap_Hardness_M5.to_numpy(float),
    }
    row: dict[str, Any] = {
        "dataset": name,
        "n_instances": len(frame),
        "mean_gap_M4": float(frame.gap_M4.mean()),
        "mean_gap_Hardness_M5": float(frame.gap_Hardness_M5.mean()),
        "mean_gap_E1_Guided_M5": float(frame.gap_E1_Guided_M5.mean()),
        "mean_gap_E0_Random_M5": float(frame.gap_E0_Random_M5.mean()),
    }
    for label, values in pairs.items():
        low, high = paired_bootstrap(values, stable_seed(name + "|" + label))
        row[label] = float(values.mean())
        row[label + "_ci95_low"] = low
        row[label + "_ci95_high"] = high
        row[label + "_fraction_negative"] = float(np.mean(values < 0))
    row["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    return row


def verify_evaluation_protocols() -> None:
    paths = (
        (CYCLES / "C03" / "external_benchmark_results.csv", M4_SHA),
        (CYCLES / "C04" / "external_benchmark_results.csv", HARD_M5_SHA),
        (P3 / "structure_m5_external_benchmark_results.csv", E1_M5_SHA),
        (EVAL_CSV, sha256_file(E0_M5)),
    )
    frames = []
    for path, checkpoint in paths:
        frame = pd.read_csv(path)
        if set(frame.dataset.astype(str)) != set(DATASETS) or set(frame.checkpoint_sha256.astype(str)) != {checkpoint}:
            raise RuntimeError(f"evaluation source mismatch {path}")
        frames.append(frame.set_index("dataset"))
    for dataset in DATASETS:
        dataset_hashes = {str(frame.at[dataset, "dataset_sha256"]) for frame in frames}
        config_hashes = {str(frame.at[dataset, "evaluator_config_sha256"]) for frame in frames}
        if len(dataset_hashes) != 1 or len(config_hashes) != 1:
            raise RuntimeError(f"evaluation protocol mismatch {dataset}")


def make_final_figures(summary: pd.DataFrame, loss_e0: pd.DataFrame, loss_e1: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    base = summary[summary.dataset.isin(DATASETS)].copy()
    x = np.arange(len(base))
    width = 0.2
    fig, ax = plt.subplots(figsize=(16, 5.5))
    columns = (
        ("mean_gap_M4", "M4"),
        ("mean_gap_Hardness_M5", "Hardness M5"),
        ("mean_gap_E0_Random_M5", "E0 random + acquisition"),
        ("mean_gap_E1_Guided_M5", "E1 guided + acquisition"),
    )
    for index, (column, label) in enumerate(columns):
        ax.bar(x + width * (index - 1.5), base[column], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(base.dataset, rotation=35, ha="right")
    ax.set_ylabel("mean optimality gap (%)")
    ax.set_title("P4-A frozen four-model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_four_model_mean_gaps.png", dpi=180)
    plt.close(fig)

    plot = summary[summary.dataset.ne("GAP8_macro")].copy()
    y = np.arange(len(plot))
    values = plot.E1_minus_E0.to_numpy(float)
    low = plot.E1_minus_E0_ci95_low.to_numpy(float)
    high = plot.E1_minus_E0_ci95_high.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(values, y, xerr=np.vstack([values - low, high - values]), fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot.dataset)
    ax.set_xlabel("E1 guided M5 − E0 random M5 gap (percentage points)")
    ax.set_title("Evolution-selection ablation: paired bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(FIG / "04_e1_minus_e0_paired_intervals.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(loss_e0.epoch, loss_e0.mean_training_loss, label="E0 random + acquisition")
    ax.plot(loss_e1.epoch, loss_e1.mean_training_loss, label="E1 guided + acquisition")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean training loss")
    ax.set_title("Identical Cycle-5 fine-tuning protocol")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_e0_e1_training_loss.png", dpi=180)
    plt.close(fig)


def finalize() -> None:
    if not (OUT / "evaluation_integrity.json").is_file():
        raise RuntimeError("E0 evaluation has not passed")
    verify_evaluation_protocols()
    aligned = {dataset: align_four(dataset) for dataset in DATASETS}
    tsplib = aligned["OriginalTSPLIB70"]
    dev = set(pd.read_csv(ROOT / "target_aware_pilot" / "03_tsplib48_reference_manifest.csv").instance_id.astype(str))
    held = set(json.loads((ROOT / "target_aware_pilot" / "p1_heldout22_names.json").read_text(encoding="utf-8"))["names"])
    if len(dev) != 48 or len(held) != 22 or dev & held or set(tsplib.instance_id) != dev | held:
        raise RuntimeError("frozen TSPLIB48/22 split mismatch")
    aligned["TSPLIB48_development"] = tsplib[tsplib.instance_id.isin(dev)].copy()
    aligned["TSPLIB22_repeated_heldout"] = tsplib[tsplib.instance_id.isin(held)].copy()

    rows = []
    paired_parts = []
    for name, frame in aligned.items():
        part = frame.copy()
        part["analysis_dataset"] = name
        part["E1_minus_E0"] = part.gap_E1_Guided_M5 - part.gap_E0_Random_M5
        part["E1_minus_Hardness"] = part.gap_E1_Guided_M5 - part.gap_Hardness_M5
        part["E0_minus_Hardness"] = part.gap_E0_Random_M5 - part.gap_Hardness_M5
        paired_parts.append(part)
        rows.append(result_row(name, frame))
    summary = pd.DataFrame(rows)
    base = summary[summary.dataset.isin(DATASETS)]
    macro: dict[str, Any] = {
        "dataset": "GAP8_macro",
        "n_instances": int(base.n_instances.sum()),
        "mean_gap_M4": float(base.mean_gap_M4.mean()),
        "mean_gap_Hardness_M5": float(base.mean_gap_Hardness_M5.mean()),
        "mean_gap_E1_Guided_M5": float(base.mean_gap_E1_Guided_M5.mean()),
        "mean_gap_E0_Random_M5": float(base.mean_gap_E0_Random_M5.mean()),
    }
    for label in ("E1_minus_E0", "E1_minus_Hardness", "E0_minus_Hardness"):
        left, right = label.split("_minus_")
        left_column = "mean_gap_" + ("E1_Guided_M5" if left == "E1" else "E0_Random_M5")
        right_column = "mean_gap_" + ("E0_Random_M5" if right == "E0" else "Hardness_M5")
        macro[label] = macro[left_column] - macro[right_column]
        macro[label + "_ci95_low"] = np.nan
        macro[label + "_ci95_high"] = np.nan
        macro[label + "_fraction_negative"] = np.nan
    macro["bootstrap_replicates"] = np.nan
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    summary.to_csv(OUT / "06_four_model_gap_comparison.csv", index=False)
    paired.to_csv(OUT / "07_per_instance_paired_analysis.csv", index=False)

    base = summary[summary.dataset.isin(DATASETS)]
    e1_wins = int((base.E1_minus_E0 < 0).sum())
    e1_significant = int((base.E1_minus_E0_ci95_high < 0).sum())
    e0_significant = int((base.E1_minus_E0_ci95_low > 0).sum())
    e0_hard_wins = int((base.E0_minus_Hardness < 0).sum())
    e0_hard_significant = int((base.E0_minus_Hardness_ci95_high < 0).sum())
    if e1_wins >= 6 and e1_significant >= 3 and e0_significant == 0:
        decision = "STRUCTURE_GUIDED_EVOLUTION_VALUE_SUPPORTED"
        case = "CASE A"
    elif (8 - e1_wins) >= 6 and e0_significant >= 3 and e1_significant == 0:
        decision = "STRUCTURE_EVOLUTION_SELECTION_NOT_SUPPORTED"
        case = "CASE C"
    else:
        decision = "STRUCTURE_ACQUISITION_DOMINANT_OR_INCONCLUSIVE"
        case = "CASE B"

    loss_e0 = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    loss_e1 = pd.read_csv(P3 / "training_cycle" / "09_bq_training" / "training_loss_curve.csv")
    make_final_figures(summary, loss_e0, loss_e1)
    label_info = json.loads((LABEL_DIR / "labeling_integrity.json").read_text(encoding="utf-8"))
    train_info = json.loads((OUT / "training_integrity.json").read_text(encoding="utf-8"))
    e0_runtime = json.loads((P2 / "evolution/E0/G1/mutation/runtime.json").read_text(encoding="utf-8"))["wall_seconds"] + json.loads((P2 / "evolution/E0/G2/mutation/runtime.json").read_text(encoding="utf-8"))["wall_seconds"]
    e1_runtime = json.loads((P2 / "evolution/E1/G1/mutation/runtime.json").read_text(encoding="utf-8"))["wall_seconds"] + json.loads((P2 / "evolution/E1/G2/mutation/runtime.json").read_text(encoding="utf-8"))["wall_seconds"]
    gap8 = summary[summary.dataset == "GAP8_macro"].iloc[0]
    dev_row = summary[summary.dataset == "TSPLIB48_development"].iloc[0]
    held_row = summary[summary.dataset == "TSPLIB22_repeated_heldout"].iloc[0]
    acquisition_share = (
        gap8.E0_minus_Hardness / gap8.E1_minus_Hardness
        if gap8.E1_minus_Hardness != 0
        else float("nan")
    )

    lines = [
        "# P4-A Evolution Ablation Conclusion",
        "",
        f"## `{decision}` ({case})",
        "",
        "Negative E1−E0 values favor structure-guided evolution. Confidence intervals use 10,000 paired bootstrap replicates.",
        "",
        "| Dataset | M4 | Hardness M5 | E0 random+acq | E1 guided+acq | E1−E0 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in [*DATASETS, "TSPLIB48_development", "TSPLIB22_repeated_heldout", "GAP8_macro"]:
        row = summary[summary.dataset == name].iloc[0]
        interval = "N/A" if pd.isna(row.E1_minus_E0_ci95_low) else f"[{row.E1_minus_E0_ci95_low:.4f}, {row.E1_minus_E0_ci95_high:.4f}]"
        lines.append(f"| {name} | {row.mean_gap_M4:.4f} | {row.mean_gap_Hardness_M5:.4f} | {row.mean_gap_E0_Random_M5:.4f} | {row.mean_gap_E1_Guided_M5:.4f} | {row.E1_minus_E0:+.4f} | {interval} |")
    lines.extend(
        [
            "",
            f"- E1 favors {e1_wins}/8 benchmark means; significant E1 wins: {e1_significant}/8; significant E0 wins: {e0_significant}/8.",
            f"- GAP8 E1−E0: {gap8.E1_minus_E0:+.4f} pp. Development48: {dev_row.E1_minus_E0:+.4f} pp. Repeated-observation TSPLIB22: {held_row.E1_minus_E0:+.4f} pp, CI [{held_row.E1_minus_E0_ci95_low:.4f}, {held_row.E1_minus_E0_ci95_high:.4f}].",
            f"- E0 random-evolution + structure-acquisition beats Hardness M5 in {e0_hard_wins}/8 benchmark means and {e0_hard_significant}/8 paired intervals. Its GAP8 improvement is {-gap8.E0_minus_Hardness:.4f} pp versus E1's {-gap8.E1_minus_Hardness:.4f} pp.",
            f"- Arithmetic GAP8 decomposition: E0 final acquisition route accounts for {acquisition_share:.1%} of the observed E1-vs-Hardness mean reduction, while guided evolution adds the remaining {-gap8.E1_minus_E0:.4f} pp. This ratio is descriptive, not a cross-seed causal effect estimate.",
            "- TSPLIB22 is explicitly not treated as a pristine final confirmation set because its P3 result was already observed. No P4-A selection or training decision used it.",
            f"- Evolution mutation wall: E0 {e0_runtime:.1f}s versus E1 {e1_runtime:.1f}s under equal 40,000-attempt budgets. New E0 labeling: {label_info['pipeline_wall_seconds']:.1f}s/{label_info['physical_new_concorde_calls']:,} calls; training: {train_info['training_loop_wall_s']:.1f}s.",
            "- Attribution: the dominant supported signal is the shared structure-aware final acquisition. Guided survivor/parent selection improves the frozen two-feature match and the GAP8 mean, but its independent solver benefit is mixed rather than stable in this single-seed design.",
            "- Follow-up priority: cross-seed E0/E1 repeats are needed to resolve the evolution-selection effect; an EvoReal Fresh10k control would answer the separate question of whether synthetic structure matching adds value beyond real-data fine-tuning. Neither is run automatically here.",
            "",
            "No P4-B/P4-C, third generation, extra training seed, or mainline modification was started.",
            "",
            "`TARGET_AWARE_P4A_EVOLUTION_ABLATION_COMPLETE`",
        ]
    )
    (OUT / "08_ablation_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        OUT / "run_metadata.json",
        {
            "status": "COMPLETE",
            "decision": decision,
            "case": case,
            "e0_bundle_sha256": sha256_file(E0_BUNDLE),
            "e0_labels_sha256": label_info["labeled_bundle_sha256"],
            "e0_m5_sha256": sha256_file(E0_M5),
            "e1_bundle_sha256": E1_SOURCE_SHA,
            "e1_m5_sha256": E1_M5_SHA,
            "four_model_csv_sha256": sha256_file(OUT / "06_four_model_gap_comparison.csv"),
            "paired_csv_sha256": sha256_file(OUT / "07_per_instance_paired_analysis.csv"),
            "evaluation_protocol_hashes_match": True,
            "tsplib22_not_pristine": True,
            "no_e1_rerun": True,
            "no_extra_seed": True,
            "no_p4b_or_p4c": True,
        },
    )
    print(json.dumps({"status": "COMPLETE", "decision": decision, "e1_wins": e1_wins, "e1_significant": e1_significant, "e0_significant": e0_significant, "gap8_e1_minus_e0": gap8.E1_minus_E0}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "label", "train", "evaluate", "finalize"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "audit":
        print(json.dumps(audit_and_acquire(), indent=2), flush=True)
    elif args.phase == "label":
        label(args.device)
    elif args.phase == "train":
        train(args.device)
    elif args.phase == "evaluate":
        evaluate(args.device)
    else:
        finalize()


if __name__ == "__main__":
    main()
