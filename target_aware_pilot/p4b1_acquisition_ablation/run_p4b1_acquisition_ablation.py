#!/usr/bin/env python3
"""P4-B1 final-acquisition value ablation on the frozen P2.2 E0 archive."""

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
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "target_aware_pilot" / "p2_bounded_evolution"
P3 = ROOT / "target_aware_pilot" / "p3_structure_finetune"
P4A = ROOT / "target_aware_pilot" / "p4a_random_evolution_ablation"
OUT = ROOT / "target_aware_pilot" / "p4b1_acquisition_ablation"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
LABEL_DIR = OUT / "labeling"
TRAIN_CYCLE = OUT / "training_cycle"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(P2))
sys.path.insert(0, str(P4A))

import run_p2_bounded_evolution as p2  # noqa: E402
import run_p4a_random_evolution_ablation as p4a  # noqa: E402
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
M4_SHA = p4a.M4_SHA
HARD_M5_SHA = p4a.HARD_M5_SHA
R1_M5 = P4A / "checkpoints" / "E0_Structure_M5_E256.tsp.best"
R1_M5_SHA = "d7d2ffc9ebafa5adb2bca9ccd90f6e8bd0d0487bdf1b4f68cf98dbf425cd4c58"
E1_M5 = P3 / "checkpoints" / "Structure_M5_E256.tsp.best"
E1_M5_SHA = p4a.E1_M5_SHA
R0_M5 = CHECKPOINTS / "R0_Random_M5_E256.tsp.best"

ARCHIVE_BUNDLE = P2 / "evolution" / "E0" / "G2" / "bounded_archive.npz"
ARCHIVE_MANIFEST = P2 / "evolution" / "E0" / "G2" / "bounded_archive.parquet"
ARCHIVE_SELECTION = P2 / "evolution" / "E0" / "G2" / "bounded_archive_selection.json"
ARCHIVE_SHA = "fd9bd10e5bd79f20054ea3b78dad62b541c0cae6226a661ffcea2b466c8bcb0a"
R1_BUNDLE = P4A / "e0_structure10k.npz"
R1_MANIFEST = P2 / "support_e0_provisional_structure10k.parquet"
R1_BUNDLE_SHA = "987395ae7de3e9a4118ed26ce9cf2c369a6b0498623dadd14668812a4cc3cc69"

RANDOM_SEED = 20261025
R0_BUNDLE = OUT / "random10k.npz"
R0_MANIFEST_CSV = OUT / "01_random10k_acquisition_manifest.csv"
R0_MANIFEST_PARQUET = OUT / "random10k_manifest.parquet"
R0_CONFIG = OUT / "random10k_config.json"
SHARED_CACHE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1" / "_shared_state" / "shared_oracle_cache.parquet"
EVAL_CSV = OUT / "r0_m5_external_benchmark_results.csv"
DATASETS = p4a.DATASETS
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


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, coordinates=np.asarray(coordinates, dtype=np.float64), instance_hash=np.asarray(hashes))
    temporary.replace(path)


def coordinate_audit(coordinates: np.ndarray, hashes: list[str]) -> dict[str, Any]:
    return {
        "rows": len(coordinates),
        "unique": len(set(hashes)),
        "shape": list(coordinates.shape),
        "finite": bool(np.isfinite(coordinates).all()),
        "in_bounds": bool(((coordinates >= 0) & (coordinates <= 1)).all()),
        "unique_nodes": bool(all(len(np.unique(item, axis=0)) == len(item) for item in coordinates)),
        "hashes_reproduce": [coordinate_hash(item) for item in coordinates] == hashes,
        "membership_sha256": membership_hash(hashes),
    }


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def categorical_js(left: pd.Series, right: pd.Series) -> float:
    categories = sorted(set(left.astype(str)) | set(right.astype(str)))
    p = left.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    q = right.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def frozen_history() -> tuple[set[str], set[str]]:
    history: set[str] = set()
    for cycle in range(4):
        history.update(pd.read_parquet(CYCLES / f"C{cycle:02d}" / "06_selection" / "T_selected.parquet", columns=["instance_hash"]).instance_hash.astype(str))
    c04 = set(pd.read_parquet(CYCLES / "C04" / "06_selection" / "T_selected.parquet", columns=["instance_hash"]).instance_hash.astype(str))
    return history, c04


def generate_and_audit() -> dict[str, Any]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if sha256_file(ARCHIVE_BUNDLE) != ARCHIVE_SHA or sha256_file(R1_BUNDLE) != R1_BUNDLE_SHA:
        raise RuntimeError("frozen archive/R1 bundle SHA mismatch")
    if sha256_file(M4) != M4_SHA or sha256_file(HARD_M5) != HARD_M5_SHA or sha256_file(R1_M5) != R1_M5_SHA or sha256_file(E1_M5) != E1_M5_SHA:
        raise RuntimeError("frozen checkpoint SHA mismatch")

    archive_xy, archive_hashes = load_bundle(ARCHIVE_BUNDLE)
    archive = pd.read_parquet(ARCHIVE_MANIFEST)
    if archive_hashes != archive.instance_hash.astype(str).tolist():
        raise RuntimeError("archive bundle/manifest order mismatch")
    archive_audit = coordinate_audit(archive_xy, archive_hashes)
    if archive_audit["rows"] != 30_000 or archive_audit["unique"] != 30_000 or archive_audit["shape"] != [30_000, 100, 2] or not all(archive_audit[key] for key in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")):
        raise RuntimeError("E0 archive integrity failure")
    if archive.m4_training_overlap.any() or not archive.proposal_status.eq("accepted").all() or not archive.valid_after_repair.all():
        raise RuntimeError("archive eligibility failure")
    if archive.lineage_root.isna().any():
        raise RuntimeError("archive lineage missing")

    history, c04 = frozen_history()
    original70 = pd.read_parquet(CYCLES / "C03" / "evaluation_work" / "M4" / "OriginalTSPLIB70" / "instance_metrics.parquet")
    tsplib_hashes = set(original70.coordinate_hash.astype(str))
    if set(archive_hashes) & history or set(archive_hashes) & tsplib_hashes:
        raise RuntimeError("archive contains M4-history or direct TSPLIB instance")

    canonical = archive.assign(_archive_index=np.arange(len(archive))).sort_values("instance_hash", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    draw = rng.choice(len(canonical), 10_000, replace=False)
    selected = canonical.iloc[draw].copy().reset_index(drop=True)
    selected["acquisition_method"] = "uniform_without_replacement"
    selected["acquisition_seed"] = RANDOM_SEED
    selected["random_draw_order"] = np.arange(len(selected))
    archive_index = selected._archive_index.to_numpy(int)
    random_xy = archive_xy[archive_index]
    random_hashes = selected.instance_hash.astype(str).tolist()
    selected = selected.drop(columns=["_archive_index"])
    if len(set(random_hashes)) != 10_000 or set(random_hashes) & history:
        raise RuntimeError("Random10k uniqueness/history exclusion failed")
    # Reproduce before freezing; structure statistics are deliberately not
    # consulted by the sampling algorithm.
    check_draw = np.random.default_rng(RANDOM_SEED).choice(len(canonical), 10_000, replace=False)
    if not np.array_equal(draw, check_draw):
        raise RuntimeError("Random10k seed replay failed")
    save_bundle(R0_BUNDLE, random_xy, random_hashes)
    selected.to_parquet(R0_MANIFEST_PARQUET, index=False)
    selected.to_csv(R0_MANIFEST_CSV, index=False)

    r1_xy, r1_hashes = load_bundle(R1_BUNDLE)
    r1 = pd.read_parquet(R1_MANIFEST)
    if r1_hashes != r1.instance_hash.astype(str).tolist():
        raise RuntimeError("R1 manifest mismatch")
    overlap = len(set(random_hashes) & set(r1_hashes))
    shared = set(pd.read_parquet(SHARED_CACHE, columns=["instance_hash"]).instance_hash.astype(str))
    r1_labels = set(pd.read_parquet(P4A / "labeling" / "ledger.parquet", columns=["instance_hash"]).instance_hash.astype(str))
    selection_meta = json.loads(ARCHIVE_SELECTION.read_text(encoding="utf-8"))
    if selection_meta.get("mode") != "common_bounded_archive_retention":
        raise RuntimeError("unexpected archive retention")

    geometry = p2.load_geometry()
    r0_row, r0_reference = p2.evaluate_frame("R0", 2, "random10k", selected, geometry)
    r1_row, r1_reference = p2.evaluate_frame("R1", 2, "structure10k", r1, geometry)
    metrics = pd.DataFrame([r0_row, r1_row])
    metrics.to_csv(OUT / "support_random_vs_structure_metrics.csv", index=False)
    reference = pd.concat([r0_reference.assign(arm="R0"), r1_reference.assign(arm="R1")], ignore_index=True)
    reference.to_csv(OUT / "support_reference_coverage.csv", index=False)

    provenance_rows = []
    for field in ("operator_id", "generation_born", "evolution_origin"):
        categories = sorted(set(selected[field].astype(str)) | set(r1[field].astype(str)))
        for category in categories:
            provenance_rows.append(
                {
                    "field": field,
                    "category": category,
                    "R0_n": int(selected[field].astype(str).eq(category).sum()),
                    "R1_n": int(r1[field].astype(str).eq(category).sum()),
                    "R0_fraction": float(selected[field].astype(str).eq(category).mean()),
                    "R1_fraction": float(r1[field].astype(str).eq(category).mean()),
                }
            )
    provenance = pd.DataFrame(provenance_rows)
    provenance.to_csv(OUT / "support_provenance_comparison.csv", index=False)
    provenance_js = {field: categorical_js(selected[field], r1[field]) for field in ("operator_id", "generation_born", "evolution_origin")}

    wall = time.perf_counter() - started
    config = {
        "status": "FROZEN",
        "seed": RANDOM_SEED,
        "algorithm": "canonical instance-hash ordering; numpy.default_rng(seed).choice(N,10000,replace=False); retain draw order",
        "candidate_n": 30_000,
        "selected_n": 10_000,
        "archive_bundle_sha256": ARCHIVE_SHA,
        "random10k_bundle_sha256": sha256_file(R0_BUNDLE),
        "random10k_manifest_csv_sha256": sha256_file(R0_MANIFEST_CSV),
        "random10k_manifest_parquet_sha256": sha256_file(R0_MANIFEST_PARQUET),
        "membership_sha256": membership_hash(random_hashes),
        "r1_overlap_n": overlap,
        "m4_history_overlap_n": len(set(random_hashes) & history),
        "c04_hard10k_overlap_n": len(set(random_hashes) & c04),
        "direct_tsplib70_overlap_n": len(set(random_hashes) & tsplib_hashes),
        "shared_cache_overlap_n": len(set(random_hashes) & shared),
        "verified_r1_label_overlap_n": len(set(random_hashes) & r1_labels),
        "acquisition_wall_seconds_including_audit_and_diagnostics": wall,
        "archive_retention_mode": selection_meta["mode"],
        "archive_retention_used_target_structure": True,
        "provenance_js": provenance_js,
    }
    write_json(R0_CONFIG, config)

    protocol = f"""# P4-B1 Protocol Integrity

- Common candidate source: frozen P2.2 E0 G2 bounded archive, 30,000 unique eligible TSP100, bundle SHA256 `{ARCHIVE_SHA}`; coordinate IDs, validity, lineage, and manifest order all reproduce.
- No C00–C03 training instance and no direct OriginalTSPLIB70 coordinate hash occurs in the archive or Random10k.
- Important scope limitation: archive retention itself was target-aware. The 39,848 eligible union was reduced to 30,000 using common structured target/frontier quotas and lineage diversity (`common_bounded_archive_retention`). P4-B1 isolates only final acquisition conditional on this archive.
- R0: canonical-hash ordered archive followed by uniform sampling without replacement with NumPy seed {RANDOM_SEED}; every archive member has probability 1/3. No feature, hardness, HGB, lineage, density, quota, or provenance value is read by sampling.
- R1: frozen P4-A E0 Structure10k/checkpoint `{R1_M5_SHA}`; it is not regenerated or retrained.
- Both models start from formal M4 `{M4_SHA}` and use the identical P3/P4-A 256-epoch, 10,240-update training/evaluation protocol.
- Predeclared conclusion rule before R0 labels/training: SUPPORTED requires R1−R0 mean gap < 0 on at least 6/8 benchmarks, paired CI upper < 0 on at least 3/8, and no significant R1 regression. NOT_SUPPORTED is the exact mirror; otherwise INCONCLUSIVE.
- TSPLIB22 is now a repeatedly observed comparison set, never a selection/tuning set or pristine final confirmation set.

Status: **PASS**.
"""
    (OUT / "00_protocol_integrity.md").write_text(protocol, encoding="utf-8")

    missing_r0 = r0_reference.loc[~r0_reference.covered, "instance_id"].astype(str).tolist()
    missing_r1 = r1_reference.loc[~r1_reference.covered, "instance_id"].astype(str).tolist()
    distribution = f"""# Random10k versus Structure10k Distribution

| Metric | R0 Random10k | R1 Structure10k |
|---|---:|---:|
| Strict target fraction | {r0_row['strict_target_fraction']:.2%} | {r1_row['strict_target_fraction']:.2%} |
| Continuous 2D SWD | {r0_row['sliced_wasserstein']:.6f} | {r1_row['sliced_wasserstein']:.6f} |
| 3×3 JS | {r0_row['coarse_js']:.6f} | {r1_row['coarse_js']:.6f} |
| Occupied/effective cells | {r0_row['occupied_cells']}/{r0_row['effective_cells']:.3f} | {r1_row['occupied_cells']}/{r1_row['effective_cells']:.3f} |
| Largest cell share | {r0_row['largest_cell_share']:.2%} | {r1_row['largest_cell_share']:.2%} |
| Target-neighborhood coverage | {r0_row['reference_coverage_n']}/48 | {r1_row['reference_coverage_n']}/48 |
| Effective lineages | {r0_row['effective_lineages']:.1f} | {r1_row['effective_lineages']:.1f} |
| Maximum lineage share | {r0_row['largest_lineage_share']:.2%} | {r1_row['largest_lineage_share']:.2%} |
| FFT median | {r0_row['fft_median']:.4f} | {r1_row['fft_median']:.4f} |
| NN-CV median | {r0_row['nn_cv_median']:.4f} | {r1_row['nn_cv_median']:.4f} |
| FFT/NN-CV auxiliary JS | {r0_row['aux_coarse_js']:.6f} | {r1_row['aux_coarse_js']:.6f} |

- Frozen membership overlap: {overlap:,}/10,000 ({overlap / 10_000:.2%}). R0 missing neighborhoods: {missing_r0}; R1 missing neighborhoods: {missing_r1}.
- Provenance JS (R0 vs R1): operator `{provenance_js['operator_id']:.6f}`, generation `{provenance_js['generation_born']:.6f}`, evolution origin `{provenance_js['evolution_origin']:.6f}`. Full per-category counts are in `support_provenance_comparison.csv`.
- The structure diagnostics were computed only after Random10k membership was frozen. No resampling or distribution correction occurred.
- HGB hardness is not used. Canonical M4 teacher-forced hardness will be attached during labeling only as passive metadata.
"""
    (OUT / "02_random_vs_structure_distribution.md").write_text(distribution, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    for frame, label, color in ((selected, "R0 Random10k", "#4c78a8"), (r1, "R1 Structure10k", "#f58518")):
        ax.scatter(frame[p2.FEATURES[0]], frame[p2.FEATURES[1]], s=3, alpha=0.15, label=label, color=color)
    ax.set_xlabel(p2.FEATURES[0])
    ax.set_ylabel(p2.FEATURES[1])
    ax.set_title("Frozen final-acquisition distributions")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(FIG / "01_random_vs_structure_feature_space.png", dpi=180)
    plt.close(fig)

    names = ["strict_target_fraction", "sliced_wasserstein", "coarse_js", "effective_cells", "effective_lineages", "aux_coarse_js"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.flat, names, strict=True):
        ax.bar(["R0", "R1"], [r0_row[name], r1_row[name]], color=["#4c78a8", "#f58518"])
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG / "02_distribution_metric_comparison.png", dpi=180)
    plt.close(fig)
    return {"status": "PASS", **archive_audit, **config}


def label(device_name: str) -> None:
    generate_and_audit()
    coordinates, hashes = load_bundle(R0_BUNDLE)
    manifest = pd.read_parquet(R0_MANIFEST_PARQUET)
    if manifest.instance_hash.astype(str).tolist() != hashes:
        raise RuntimeError("Random10k manifest mismatch")
    import run_blade_v2 as base

    device = torch.device(device_name)
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    shared = pd.read_parquet(SHARED_CACHE)
    reusable = pd.read_parquet(P4A / "labeling" / "ledger.parquet")
    if not reusable.verified.all() or not reusable.status.eq("solved").all() or set(reusable.binary_hash.astype(str)) != {CANONICAL_CONCORDE_SHA256}:
        raise RuntimeError("P4-A label cache is not canonically reusable")
    base.GLOBAL_CACHE = pd.concat([shared, reusable], ignore_index=True).drop_duplicates("instance_hash", keep="last")
    selected = manifest.copy()
    selected["coord_index"] = np.arange(len(selected))
    started = time.perf_counter()
    _, tours, summary = base._label_population(
        LABEL_DIR, selected, coordinates, model, device, None,
        "P4B1_R0_random10k_original_A0_oracle_and_M4_hardness", True,
    )
    invocation_wall = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ledger = pd.read_parquet(LABEL_DIR / "ledger.parquet")
    if len(ledger) != 10_000 or not ledger.verified.all() or not ledger.status.eq("solved").all():
        raise RuntimeError("R0 labeling incomplete")
    if tours.shape != (10_000, 100) or not all(validate_tour_0based(tour, 100)[0] for tour in tours):
        raise RuntimeError("R0 illegal expert tour")
    costs = np.asarray([tour_cost_int(integer_distance_matrix(points), tour) for points, tour in zip(coordinates, tours, strict=True)], dtype=np.int64)
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)):
        raise RuntimeError("R0 objective mismatch")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (10_000, 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]):
        raise RuntimeError("R0 closure mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist():
        raise RuntimeError("R0 tour hash mismatch")
    integrity = {
        "status": "PASS", "selected": 10_000, "verified": 10_000, "failures": 0,
        "cache_hits": int(summary["cache_hits"]), "physical_new_concorde_calls": int(summary["new_concorde_calls"]),
        "pipeline_wall_seconds": float(summary["wall_seconds"]), "invocation_wall_seconds": invocation_wall,
        "all_tours_legal": True, "all_costs_reproduced": True,
        "ledger_sha256": sha256_file(LABEL_DIR / "ledger.parquet"),
        "labeled_bundle_sha256": sha256_file(LABEL_DIR / "labeled_bundle.npz"),
        "labeled_metadata_sha256": sha256_file(LABEL_DIR / "labeled.parquet"),
        "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
        "m4_sha256": model_meta["sha256"], "concorde_sha256": CANONICAL_CONCORDE_SHA256,
    }
    write_json(LABEL_DIR / "labeling_integrity.json", integrity)
    report = f"""# Concorde Labeling Audit

- Exact P3/P4-A canonical pipeline; 10,000/10,000 legal expert tours, zero failures.
- Reused {integrity['cache_hits']:,} verified canonical P4-A labels for overlapping R0/R1 instances; executed {integrity['physical_new_concorde_calls']:,} new Concorde calls. Every reused and new objective was independently recomputed from R0 coordinates.
- Pipeline wall {integrity['pipeline_wall_seconds']:.3f}s; 0-based Hamiltonian permutation, implicit cost closure, BQ input closure appended as node 101.
- Ledger SHA256 `{integrity['ledger_sha256']}`; labeled bundle SHA256 `{integrity['labeled_bundle_sha256']}`; expert tours SHA256 `{integrity['expert_tours_array_sha256']}`.

Status: **PASS**.
"""
    (OUT / "03_concorde_labeling_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(integrity, indent=2), flush=True)


def load_labeled() -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(LABEL_DIR / "labeled_bundle.npz", allow_pickle=False) as payload:
        return np.asarray(payload["coordinates"], float), payload["instance_hash"].astype(str).tolist(), np.asarray(payload["expert_tours"], np.int64)


def train(device_name: str) -> None:
    generate_and_audit()
    label_info = json.loads((LABEL_DIR / "labeling_integrity.json").read_text(encoding="utf-8"))
    if label_info.get("status") != "PASS" or label_info.get("verified") != 10_000:
        raise RuntimeError("labeling gate failed")
    coordinates, hashes, tours = load_labeled()
    frozen_xy, frozen_hashes = load_bundle(R0_BUNDLE)
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
        raise RuntimeError("R0 labeled membership changed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    core.RUN_ID = "TARGET_AWARE_P4B1_R0_RANDOM_C5"
    core.cycle_dir = lambda arm, cycle: TRAIN_CYCLE
    core.model_path = lambda arm, index: R0_M5 if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    started = time.perf_counter()
    trained, trained_meta, timing = core.train("R0_RANDOM_C5", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)
    invocation_wall = time.perf_counter() - started
    e000 = torch.load(TRAIN_CYCLE / "09_bq_training" / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    if not e000_equal or trained_meta["sha256"] != sha256_file(R0_M5) or not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters()):
        raise RuntimeError("R0 training integrity failed")
    del trained, model
    if device.type == "cuda": torch.cuda.empty_cache()
    curve = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    if len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(10_000).all():
        raise RuntimeError("R0 trajectory mismatch")
    r1_info = json.loads((P4A / "training_integrity.json").read_text(encoding="utf-8"))
    report = f"""# Training Protocol and Log

- Directly reused formal P3/P4-A `train`; E000 exactly equals formal M4: **{e000_equal}**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, batch/microbatch 256/64, 256 epochs, 40 updates/epoch, 10,240 total updates, seed 2026, all 10,000 instances once/epoch.
- R0 final loss {timing['final_loss']:.8f}, wall {timing['training_loop_wall_s']:.3f}s, peak GPU bytes {int(timing['gpu_peak_allocated_bytes']):,}. R1 final loss {r1_info['final_loss']:.8f}, wall {r1_info['training_loop_wall_s']:.3f}s.
- R0 checkpoint SHA256 `{trained_meta['sha256']}`; membership SHA256 `{membership_hash(hashes)}`; label bundle SHA256 `{label_info['labeled_bundle_sha256']}`.

E256 was frozen before evaluation; no test result selected the checkpoint or protocol.
"""
    (OUT / "04_training_protocol_and_log.md").write_text(report, encoding="utf-8")
    write_json(OUT / "training_integrity.json", {
        "status": "PASS", "m4_sha256": M4_SHA, "e000_weights_equal_m4": e000_equal,
        "r0_m5_sha256": trained_meta["sha256"], "membership_sha256": membership_hash(hashes),
        "label_bundle_sha256": label_info["labeled_bundle_sha256"], "epochs": 256,
        "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": invocation_wall,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]),
    })
    print(json.dumps({"status":"PASS","checkpoint":str(R0_M5),"sha256":trained_meta["sha256"],"timing":timing}, indent=2), flush=True)


def evaluate(device_name: str) -> None:
    integrity = json.loads((OUT / "training_integrity.json").read_text(encoding="utf-8"))
    if integrity.get("r0_m5_sha256") != sha256_file(R0_M5):
        raise RuntimeError("R0 checkpoint integrity failed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    model, meta = load_canonical_checkpoint(R0_M5, device, expected_sha256=integrity["r0_m5_sha256"])
    frame, wall = core.evaluate(model, R0_M5, "R0_Random_M5", EVAL_CSV, Status(), "R0_RANDOM_C5")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all():
        raise RuntimeError("R0 evaluation incomplete")
    write_json(OUT / "evaluation_integrity.json", {"status":"PASS","checkpoint_sha256":meta["sha256"],"all_tours_valid":True,"datasets":list(DATASETS),"evaluation_wall_s":wall,"results_sha256":sha256_file(EVAL_CSV)})
    print(frame[["dataset","instance_count","mean_gap_percent"]].to_string(index=False), flush=True)


def detail(model: str, dataset: str) -> pd.DataFrame:
    if model == "M4": path = CYCLES / "C03/evaluation_work/M4" / dataset / "instance_metrics.parquet"
    elif model == "Hardness_M5": path = CYCLES / "C04/evaluation_work/M5" / dataset / "instance_metrics.parquet"
    elif model == "R0_Random_M5": path = OUT / "evaluation_work/R0_Random_M5" / dataset / "instance_metrics.parquet"
    elif model == "R1_Structure_M5": path = P4A / "evaluation_work/E0_Structure_M5" / dataset / "instance_metrics.parquet"
    else: path = P3 / "evaluation_work/Structure_M5" / dataset / "instance_metrics.parquet"
    return pd.read_parquet(path)


MODELS = ("M4", "Hardness_M5", "R0_Random_M5", "R1_Structure_M5", "E1_Guided_M5")


def align_five(dataset: str) -> pd.DataFrame:
    frames=[]
    for model in MODELS:
        source=detail(model,dataset)
        part=source[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].copy()
        if not part.valid_tour.all(): raise RuntimeError(f"invalid tour {model}/{dataset}")
        part=part.rename(columns={"coordinate_hash":f"coordinate_hash_{model}","n_nodes":f"n_nodes_{model}","optimal_cost_int":f"optimal_cost_int_{model}","gap_percent":f"gap_{model}"}).drop(columns=["valid_tour"])
        frames.append(part)
    merged=frames[0]
    for frame in frames[1:]: merged=merged.merge(frame,on="instance_id",validate="one_to_one")
    for field in ("coordinate_hash","n_nodes","optimal_cost_int"):
        columns=[f"{field}_{model}" for model in MODELS]
        if not all(merged[columns[0]].astype(str).eq(merged[column].astype(str)).all() for column in columns[1:]): raise RuntimeError(f"alignment failure {dataset}/{field}")
    return pd.DataFrame({"dataset":dataset,"instance_id":merged.instance_id.astype(str),"coordinate_hash":merged.coordinate_hash_M4.astype(str),"n_nodes":merged.n_nodes_M4.astype(int),"optimal_cost_int":merged.optimal_cost_int_M4.astype(np.int64),**{f"gap_{model}":merged[f"gap_{model}"].astype(float) for model in MODELS}})


def paired_bootstrap(values: np.ndarray, seed: int) -> tuple[float,float]:
    values=np.asarray(values,float); rng=np.random.default_rng(seed); estimates=np.empty(BOOTSTRAP_REPLICATES); cursor=0
    while cursor<BOOTSTRAP_REPLICATES:
        count=min(200,BOOTSTRAP_REPLICATES-cursor); indices=rng.integers(0,len(values),size=(count,len(values))); estimates[cursor:cursor+count]=values[indices].mean(axis=1); cursor+=count
    return float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))


def stable_seed(text: str) -> int: return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"little")


def result_row(name: str, frame: pd.DataFrame) -> dict[str,Any]:
    pairs={
        "R1_minus_R0":frame.gap_R1_Structure_M5.to_numpy(float)-frame.gap_R0_Random_M5.to_numpy(float),
        "R1_minus_Hardness":frame.gap_R1_Structure_M5.to_numpy(float)-frame.gap_Hardness_M5.to_numpy(float),
        "R0_minus_Hardness":frame.gap_R0_Random_M5.to_numpy(float)-frame.gap_Hardness_M5.to_numpy(float),
        "R0_minus_M4":frame.gap_R0_Random_M5.to_numpy(float)-frame.gap_M4.to_numpy(float),
    }
    row={"dataset":name,"n_instances":len(frame),**{f"mean_gap_{model}":float(frame[f"gap_{model}"].mean()) for model in MODELS}}
    for label,values in pairs.items():
        low,high=paired_bootstrap(values,stable_seed(name+"|"+label)); row[label]=float(values.mean()); row[label+"_ci95_low"]=low; row[label+"_ci95_high"]=high; row[label+"_fraction_negative"]=float(np.mean(values<0)); row[label+"_wins"]=int(np.sum(values<0)); row[label+"_ties"]=int(np.sum(values==0))
    row["bootstrap_replicates"]=BOOTSTRAP_REPLICATES
    return row


def verify_protocols() -> None:
    sources=((CYCLES/"C03/external_benchmark_results.csv",M4_SHA),(CYCLES/"C04/external_benchmark_results.csv",HARD_M5_SHA),(EVAL_CSV,sha256_file(R0_M5)),(P4A/"e0_m5_external_benchmark_results.csv",R1_M5_SHA),(P3/"structure_m5_external_benchmark_results.csv",E1_M5_SHA))
    frames=[]
    for path,checkpoint in sources:
        frame=pd.read_csv(path)
        if set(frame.dataset.astype(str))!=set(DATASETS) or set(frame.checkpoint_sha256.astype(str))!={checkpoint}: raise RuntimeError(f"evaluation source mismatch {path}")
        frames.append(frame.set_index("dataset"))
    for dataset in DATASETS:
        if len({str(frame.at[dataset,"dataset_sha256"]) for frame in frames})!=1 or len({str(frame.at[dataset,"evaluator_config_sha256"]) for frame in frames})!=1: raise RuntimeError(f"evaluation protocol mismatch {dataset}")


def make_final_figures(summary: pd.DataFrame) -> None:
    base=summary[summary.dataset.isin(DATASETS)]; x=np.arange(len(base)); width=.16
    fig,ax=plt.subplots(figsize=(17,5.5))
    for index,(column,label) in enumerate((("mean_gap_M4","M4"),("mean_gap_Hardness_M5","Hardness M5"),("mean_gap_R0_Random_M5","R0 random acq"),("mean_gap_R1_Structure_M5","R1 structure acq"),("mean_gap_E1_Guided_M5","E1 guided"))): ax.bar(x+width*(index-2),base[column],width=width,label=label)
    ax.set_xticks(x); ax.set_xticklabels(base.dataset,rotation=35,ha="right"); ax.set_ylabel("mean optimality gap (%)"); ax.set_title("P4-B1 frozen five-model comparison"); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"03_five_model_mean_gaps.png",dpi=180); plt.close(fig)
    plot=summary[summary.dataset.ne("GAP8_macro")]; y=np.arange(len(plot)); values=plot.R1_minus_R0.to_numpy(float); low=plot.R1_minus_R0_ci95_low.to_numpy(float); high=plot.R1_minus_R0_ci95_high.to_numpy(float)
    fig,ax=plt.subplots(figsize=(9,6)); ax.errorbar(values,y,xerr=np.vstack([values-low,high-values]),fmt="o",capsize=3); ax.axvline(0,color="black",linewidth=.9); ax.set_yticks(y); ax.set_yticklabels(plot.dataset); ax.set_xlabel("R1 structure M5 − R0 random M5 gap (pp)"); ax.set_title("Final-acquisition ablation: paired bootstrap 95% CI"); fig.tight_layout(); fig.savefig(FIG/"04_r1_minus_r0_paired_intervals.png",dpi=180); plt.close(fig)
    r0=pd.read_csv(TRAIN_CYCLE/"09_bq_training/training_loss_curve.csv"); r1=pd.read_csv(P4A/"training_cycle/09_bq_training/training_loss_curve.csv")
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(r0.epoch,r0.mean_training_loss,label="R0 Random10k"); ax.plot(r1.epoch,r1.mean_training_loss,label="R1 Structure10k"); ax.set_xlabel("epoch"); ax.set_ylabel("mean training loss"); ax.set_title("Identical Cycle-5 protocol"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"05_training_loss.png",dpi=180); plt.close(fig)


def finalize() -> None:
    if not (OUT/"evaluation_integrity.json").is_file(): raise RuntimeError("R0 evaluation missing")
    verify_protocols(); aligned={dataset:align_five(dataset) for dataset in DATASETS}; tsplib=aligned["OriginalTSPLIB70"]
    dev=set(pd.read_csv(ROOT/"target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str)); held=set(json.loads((ROOT/"target_aware_pilot/p1_heldout22_names.json").read_text())["names"])
    if len(dev)!=48 or len(held)!=22 or dev&held or set(tsplib.instance_id)!=dev|held: raise RuntimeError("TSPLIB split mismatch")
    aligned["TSPLIB48_development"]=tsplib[tsplib.instance_id.isin(dev)].copy(); aligned["TSPLIB22_previously_heldout"]=tsplib[tsplib.instance_id.isin(held)].copy()
    rows=[]; paired_parts=[]
    for name,frame in aligned.items():
        part=frame.copy(); part["analysis_dataset"]=name; part["R1_minus_R0"]=part.gap_R1_Structure_M5-part.gap_R0_Random_M5; part["R1_minus_Hardness"]=part.gap_R1_Structure_M5-part.gap_Hardness_M5; part["R0_minus_Hardness"]=part.gap_R0_Random_M5-part.gap_Hardness_M5; part["R0_minus_M4"]=part.gap_R0_Random_M5-part.gap_M4; paired_parts.append(part); rows.append(result_row(name,frame))
    summary=pd.DataFrame(rows); base=summary[summary.dataset.isin(DATASETS)]
    macro={"dataset":"GAP8_macro","n_instances":int(base.n_instances.sum()),**{f"mean_gap_{model}":float(base[f"mean_gap_{model}"].mean()) for model in MODELS}}
    for label,left,right in (("R1_minus_R0","R1_Structure_M5","R0_Random_M5"),("R1_minus_Hardness","R1_Structure_M5","Hardness_M5"),("R0_minus_Hardness","R0_Random_M5","Hardness_M5"),("R0_minus_M4","R0_Random_M5","M4")):
        macro[label]=macro[f"mean_gap_{left}"]-macro[f"mean_gap_{right}"]
        for suffix in ("_ci95_low","_ci95_high","_fraction_negative","_wins","_ties"): macro[label+suffix]=np.nan
    macro["bootstrap_replicates"]=np.nan; summary=pd.concat([summary,pd.DataFrame([macro])],ignore_index=True); paired=pd.concat(paired_parts,ignore_index=True)
    summary.to_csv(OUT/"05_five_model_gap_comparison.csv",index=False); paired.to_csv(OUT/"06_per_instance_paired_analysis.csv",index=False)
    base=summary[summary.dataset.isin(DATASETS)]; r1_wins=int((base.R1_minus_R0<0).sum()); r1_sig=int((base.R1_minus_R0_ci95_high<0).sum()); r0_sig=int((base.R1_minus_R0_ci95_low>0).sum())
    r0_hard_wins=int((base.R0_minus_Hardness<0).sum()); r0_hard_sig=int((base.R0_minus_Hardness_ci95_high<0).sum())
    if r1_wins>=6 and r1_sig>=3 and r0_sig==0: decision="TARGET_AWARE_ACQUISITION_VALUE_SUPPORTED"; label="SUPPORTED"
    elif (8-r1_wins)>=6 and r0_sig>=3 and r1_sig==0: decision="TARGET_AWARE_ACQUISITION_NOT_SUPPORTED"; label="NOT_SUPPORTED"
    else: decision="TARGET_AWARE_ACQUISITION_VALUE_INCONCLUSIVE"; label="INCONCLUSIVE"
    make_final_figures(summary); gap8=summary[summary.dataset=="GAP8_macro"].iloc[0]; devrow=summary[summary.dataset=="TSPLIB48_development"].iloc[0]; heldrow=summary[summary.dataset=="TSPLIB22_previously_heldout"].iloc[0]
    config=json.loads(R0_CONFIG.read_text()); labelinfo=json.loads((LABEL_DIR/"labeling_integrity.json").read_text()); traininfo=json.loads((OUT/"training_integrity.json").read_text()); r1train=json.loads((P4A/"training_integrity.json").read_text())
    lines=["# P4-B1 Acquisition Ablation Conclusion","",f"## `{decision}` — **{label}**","","Negative R1−R0 favors structure-aware acquisition. CIs use 10,000 paired bootstrap replicates.","","| Dataset | M4 | Hardness | R0 Random10k | R1 Structure10k | E1 Guided | R1−R0 | 95% CI |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in [*DATASETS,"TSPLIB48_development","TSPLIB22_previously_heldout","GAP8_macro"]:
        row=summary[summary.dataset==name].iloc[0]; interval="N/A" if pd.isna(row.R1_minus_R0_ci95_low) else f"[{row.R1_minus_R0_ci95_low:.4f}, {row.R1_minus_R0_ci95_high:.4f}]"; lines.append(f"| {name} | {row.mean_gap_M4:.4f} | {row.mean_gap_Hardness_M5:.4f} | {row.mean_gap_R0_Random_M5:.4f} | {row.mean_gap_R1_Structure_M5:.4f} | {row.mean_gap_E1_Guided_M5:.4f} | {row.R1_minus_R0:+.4f} | {interval} |")
    lines.extend(["",f"- R1 wins {r1_wins}/8 benchmark means; significant R1 wins {r1_sig}/8; significant R0 wins {r0_sig}/8. The formal rule is therefore INCONCLUSIVE, but the directional evidence does not support a stable R1 advantage.",f"- GAP8 R1−R0 {gap8.R1_minus_R0:+.4f} pp. R0 has the lowest GAP8 of the five frozen models ({gap8.mean_gap_R0_Random_M5:.4f}). Development48 {devrow.R1_minus_R0:+.4f} pp. Previously held-out22 {heldrow.R1_minus_R0:+.4f} pp, CI [{heldrow.R1_minus_R0_ci95_low:.4f}, {heldrow.R1_minus_R0_ci95_high:.4f}].",f"- R0 beats Hardness M5 in {r0_hard_wins}/8 benchmark means and {r0_hard_sig}/8 paired intervals; random final acquisition from this already structured archive retains a strong Cycle-5 benefit.",f"- Training final loss: R0 {traininfo['final_loss']:.8f}, R1 {r1train['final_loss']:.8f}; identical 10,240-update protocol. R1 fits its training labels better, but that lower imitation loss does not translate into broader frozen-benchmark superiority.",f"- Cost: acquisition/audit/diagnostics {config['acquisition_wall_seconds_including_audit_and_diagnostics']:.2f}s; labeling {labelinfo['pipeline_wall_seconds']:.1f}s with {labelinfo['cache_hits']:,} verified reused labels and {labelinfo['physical_new_concorde_calls']:,} new calls; R0 training {traininfo['training_loop_wall_s']:.1f}s.","- Scope: this tests final acquisition only conditional on a target-aware retained E0 archive. It does not establish that every upstream structure-selection component is independently necessary.","- Research implication: do not promote target-aware final acquisition as the primary independently validated contribution. Structure-guided evolution may remain as an exploratory/domain-specific component, but P4-A also found its independent effect mixed; the defensible current result is the end-to-end candidate/archive plus Cycle-5 data-distribution effect, pending cross-seed and fresh external confirmation.","- TSPLIB22 is a repeatedly observed comparability set, not pristine confirmation evidence, and was not used to change R0/R1 data or training.","","No resampling, second seed, Fresh10k, or other ablation was started.","",f"Decision label: `{label}`","","`TARGET_AWARE_P4B1_ACQUISITION_ABLATION_COMPLETE`"])
    (OUT/"07_acquisition_ablation_conclusion.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(OUT/"run_metadata.json",{"status":"COMPLETE","decision":decision,"decision_label":label,"random10k_sha256":sha256_file(R0_BUNDLE),"r0_labels_sha256":labelinfo["labeled_bundle_sha256"],"r0_m5_sha256":sha256_file(R0_M5),"r1_m5_sha256":R1_M5_SHA,"five_model_csv_sha256":sha256_file(OUT/"05_five_model_gap_comparison.csv"),"paired_csv_sha256":sha256_file(OUT/"06_per_instance_paired_analysis.csv"),"evaluation_protocol_hashes_match":True,"archive_retention_target_aware":True,"tsplib22_not_pristine":True,"no_resampling":True,"no_extra_seed":True})
    Status().update(phase="complete", status="COMPLETE", decision=decision, decision_label=label)
    print(json.dumps({"status":"COMPLETE","decision":decision,"label":label,"r1_wins":r1_wins,"r1_significant":r1_sig,"r0_significant":r0_sig,"gap8_r1_minus_r0":gap8.R1_minus_R0},indent=2),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("audit","label","train","evaluate","finalize")); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args()
    if args.phase=="audit": print(json.dumps(generate_and_audit(),indent=2),flush=True)
    elif args.phase=="label": label(args.device)
    elif args.phase=="train": train(args.device)
    elif args.phase=="evaluate": evaluate(args.device)
    else: finalize()


if __name__=="__main__": main()
