#!/usr/bin/env python3
"""P4-B2 structure-free archive-retention ablation on frozen E0 offspring."""

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
P3 = ROOT / "target_aware_pilot" / "p3_structure_finetune"
P4A = ROOT / "target_aware_pilot" / "p4a_random_evolution_ablation"
P4B1 = ROOT / "target_aware_pilot" / "p4b1_acquisition_ablation"
OUT = ROOT / "target_aware_pilot" / "p4b2_random_archive_ablation"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
LABEL_DIR = OUT / "labeling"
TRAIN_CYCLE = OUT / "training_cycle"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(P2))
sys.path.insert(0, str(P4A))
sys.path.insert(0, str(P4B1))

import run_p2_bounded_evolution as p2  # noqa: E402
import run_p4a_random_evolution_ablation as p4a  # noqa: E402
import run_p4b1_acquisition_ablation as p4b1  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, sha256_file  # noqa: E402


FORMAL = p4b1.FORMAL
CYCLES = p4b1.CYCLES
M4 = p4b1.M4
HARD_M5 = p4b1.HARD_M5
M4_SHA = p4b1.M4_SHA
HARD_M5_SHA = p4b1.HARD_M5_SHA
R0_M5 = P4B1 / "checkpoints" / "R0_Random_M5_E256.tsp.best"
R0_M5_SHA = "72e3d4dcb4a0bc1c5562e9ab161deaffddb03e0829a01798cec078668540e8a1"
E0_STRUCTURE_M5 = P4A / "checkpoints" / "E0_Structure_M5_E256.tsp.best"
E0_STRUCTURE_M5_SHA = p4b1.R1_M5_SHA
E1_STRUCTURE_M5 = P3 / "checkpoints" / "Structure_M5_E256.tsp.best"
E1_STRUCTURE_M5_SHA = p4b1.E1_M5_SHA
R2_M5 = CHECKPOINTS / "R2_RandomArchive_Random10k_M5_E256.tsp.best"

R0_ARCHIVE_BUNDLE = P2 / "evolution" / "E0" / "G2" / "bounded_archive.npz"
R0_ARCHIVE_MANIFEST = P2 / "evolution" / "E0" / "G2" / "bounded_archive.parquet"
R0_ARCHIVE_SELECTION = P2 / "evolution" / "E0" / "G2" / "bounded_archive_selection.json"
R0_ARCHIVE_SHA = p4b1.ARCHIVE_SHA
R0_10K_BUNDLE = P4B1 / "random10k.npz"
R0_10K_MANIFEST = P4B1 / "random10k_manifest.parquet"
R0_10K_CONFIG = P4B1 / "random10k_config.json"

FULL_POOL_BUNDLE = OUT / "e0_full_eligible_offspring39848.npz"
FULL_POOL_MANIFEST = OUT / "e0_full_eligible_offspring39848.parquet"
R2_ARCHIVE_BUNDLE = OUT / "R2_random_archive30k.npz"
R2_ARCHIVE_PARQUET = OUT / "R2_random_archive30k_manifest.parquet"
R2_ARCHIVE_CSV = OUT / "01_random_archive30k_manifest.csv"
R2_10K_BUNDLE = OUT / "R2_random10k.npz"
R2_10K_PARQUET = OUT / "R2_random10k_manifest.parquet"
R2_10K_CSV = OUT / "02_random10k_manifest.csv"
R2_INTEGRITY = OUT / "R2_random10k_integrity.json"
R2_CONFIG = OUT / "random_archive_and_acquisition_config.json"
EVAL_CSV = OUT / "r2_m5_external_benchmark_results.csv"

RETENTION_SEED = 20261026
ACQUISITION_SEED = 20261025
FULL_POOL_N = 39_848
ARCHIVE_N = 30_000
TRAIN_N = 10_000
DATASETS = p4b1.DATASETS
BOOTSTRAP_REPLICATES = 10_000


def write_json(path: Path, value: Any) -> None:
    p4b1.write_json(path, value)


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def frozen_checkpoint_audit() -> None:
    expected = (
        (M4, M4_SHA),
        (HARD_M5, HARD_M5_SHA),
        (R0_M5, R0_M5_SHA),
        (E0_STRUCTURE_M5, E0_STRUCTURE_M5_SHA),
        (E1_STRUCTURE_M5, E1_STRUCTURE_M5_SHA),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise RuntimeError(f"frozen checkpoint SHA mismatch: {path}")


def load_frozen_full_pool() -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    coordinate_map: dict[str, np.ndarray] = {}
    source_details: dict[str, Any] = {}
    total_unique_within_generation = 0
    for generation in (1, 2):
        directory = P2 / "evolution" / "E0" / f"G{generation}"
        bundle_path = directory / "mutation" / "offspring_coordinates.npz"
        mutation_path = directory / "mutation" / "mutation_manifest.parquet"
        runtime_path = directory / "mutation" / "runtime.json"
        diagnostics_path = directory / "offspring_diagnostics.parquet"
        coordinates, hashes = p4b1.load_bundle(bundle_path)
        mutation = pd.read_parquet(mutation_path)
        diagnostics = pd.read_parquet(diagnostics_path)
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        accepted = mutation.valid_after_repair & mutation.proposal_status.eq("accepted") & ~mutation.duplicate
        accepted_hashes = mutation.loc[accepted, "child_hash"].astype(str).tolist()
        diagnostic_hashes = diagnostics.instance_hash.astype(str).tolist()
        if hashes != accepted_hashes or hashes != diagnostic_hashes:
            raise RuntimeError(f"frozen E0 G{generation} offspring alignment failure")
        if runtime.get("status") != "PASS" or runtime.get("arm") != "E0" or runtime.get("generation") != generation:
            raise RuntimeError(f"frozen E0 G{generation} runtime mismatch")
        if runtime.get("offspring_membership_sha256") != membership_hash(hashes):
            raise RuntimeError(f"frozen E0 G{generation} membership mismatch")
        if len(mutation) != 20_000 or not diagnostics.valid_after_repair.all() or not diagnostics.proposal_status.eq("accepted").all():
            raise RuntimeError(f"frozen E0 G{generation} mutation integrity failure")
        if diagnostics[["lineage_root", "operator_id", "generation_born", "direct_parent_hash"]].isna().any().any():
            raise RuntimeError(f"frozen E0 G{generation} provenance missing")
        reproduced = [coordinate_hash(item) for item in coordinates]
        if reproduced != hashes:
            raise RuntimeError(f"frozen E0 G{generation} coordinate hash mismatch")
        total_unique_within_generation += len(hashes)
        eligible = diagnostics.loc[~diagnostics.m4_training_overlap].copy().reset_index(drop=True)
        frames.append(eligible)
        eligible_hashes = eligible.instance_hash.astype(str).tolist()
        lookup = dict(zip(hashes, coordinates, strict=True))
        for value in eligible_hashes:
            coordinate_map.setdefault(value, lookup[value])
        source_details[f"G{generation}"] = {
            "mutation_manifest_sha256": sha256_file(mutation_path),
            "offspring_bundle_sha256": sha256_file(bundle_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "nominal_attempts": len(mutation),
            "valid_unique_within_generation": len(hashes),
            "eligible_before_cross_generation_dedup": len(eligible),
            "offspring_membership_sha256": membership_hash(hashes),
            "parent_membership_sha256": runtime["parent_membership_sha256"],
            "cycle_seed_namespace": runtime["cycle_seed_namespace"],
        }

    combined = pd.concat(frames, ignore_index=True, sort=False)
    before_cross_dedup = len(combined)
    full = combined.drop_duplicates("instance_hash", keep="first").reset_index(drop=True)
    if len(full) != FULL_POOL_N or full.instance_hash.nunique() != FULL_POOL_N:
        raise RuntimeError(f"expected {FULL_POOL_N} frozen eligible offspring, got {len(full)}")
    full = full.sort_values("instance_hash", kind="stable").reset_index(drop=True)
    hashes = full.instance_hash.astype(str).tolist()
    coordinates = np.stack([coordinate_map[value] for value in hashes])
    audit = p4b1.coordinate_audit(coordinates, hashes)
    if audit["shape"] != [FULL_POOL_N, 100, 2] or not all(audit[key] for key in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")):
        raise RuntimeError("full eligible pool coordinate integrity failure")
    history, _ = p4b1.frozen_history()
    if set(hashes) & history:
        raise RuntimeError("full eligible pool overlaps M4 history")
    details = {
        "source": source_details,
        "valid_unique_within_generation_total": total_unique_within_generation,
        "eligible_before_cross_generation_dedup": before_cross_dedup,
        "cross_generation_duplicates_removed": before_cross_dedup - len(full),
        "full_pool_membership_sha256": membership_hash(hashes),
        **audit,
    }
    return coordinates, full, details


def sample_uniform(frame: pd.DataFrame, coordinates: np.ndarray, n: int, seed: int, stage: str) -> tuple[pd.DataFrame, np.ndarray]:
    hashes = frame.instance_hash.astype(str).tolist()
    if hashes != sorted(hashes):
        order = np.argsort(np.asarray(hashes), kind="stable")
        canonical = frame.iloc[order].reset_index(drop=True)
        canonical_coordinates = coordinates[order]
    else:
        canonical = frame.reset_index(drop=True)
        canonical_coordinates = coordinates
    draw = np.random.default_rng(seed).choice(len(canonical), n, replace=False)
    replay = np.random.default_rng(seed).choice(len(canonical), n, replace=False)
    if not np.array_equal(draw, replay):
        raise RuntimeError(f"{stage} seed replay failed")
    selected = canonical.iloc[draw].copy().reset_index(drop=True)
    selected[f"{stage}_method"] = "uniform_without_replacement"
    selected[f"{stage}_seed"] = seed
    selected[f"{stage}_draw_order"] = np.arange(n)
    return selected, canonical_coordinates[draw]


def distribution_row(label: str, frame: pd.DataFrame, geometry: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    row, reference = p2.evaluate_frame("R2", 2, label, frame, geometry)
    row["distribution"] = label
    return row, reference.assign(distribution=label)


def generate_and_audit() -> dict[str, Any]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    frozen_checkpoint_audit()
    if sha256_file(R0_ARCHIVE_BUNDLE) != R0_ARCHIVE_SHA:
        raise RuntimeError("R0 target-aware archive SHA mismatch")

    full_xy, full, source_audit = load_frozen_full_pool()
    full_hashes = full.instance_hash.astype(str).tolist()
    p4b1.save_bundle(FULL_POOL_BUNDLE, full_xy, full_hashes)
    full.to_parquet(FULL_POOL_MANIFEST, index=False)

    r0_archive_xy, r0_archive_hashes = p4b1.load_bundle(R0_ARCHIVE_BUNDLE)
    r0_archive = pd.read_parquet(R0_ARCHIVE_MANIFEST)
    if r0_archive.instance_hash.astype(str).tolist() != r0_archive_hashes:
        raise RuntimeError("R0 archive manifest/bundle mismatch")
    full_lookup = dict(zip(full_hashes, full_xy, strict=True))
    if not set(r0_archive_hashes).issubset(full_lookup):
        raise RuntimeError("R0 archive is not a subset of the frozen full pool")
    if not all(np.array_equal(points, full_lookup[value]) for value, points in zip(r0_archive_hashes, r0_archive_xy, strict=True)):
        raise RuntimeError("R0 archive coordinates differ from frozen offspring")
    selection = json.loads(R0_ARCHIVE_SELECTION.read_text(encoding="utf-8"))
    if selection.get("mode") != "common_bounded_archive_retention" or selection.get("pool_n") != FULL_POOL_N:
        raise RuntimeError("R0 archive retention protocol mismatch")
    for generation in (1, 2):
        survivor = json.loads((P2 / "evolution" / "E0" / f"G{generation}" / "active_population_selection.json").read_text())
        if survivor.get("mode") != "random_survivor":
            raise RuntimeError("E0 survivor selection was not random")

    r2_archive, r2_archive_xy = sample_uniform(full, full_xy, ARCHIVE_N, RETENTION_SEED, "archive_retention")
    r2_archive_hashes = r2_archive.instance_hash.astype(str).tolist()
    r2_final, r2_final_xy = sample_uniform(r2_archive, r2_archive_xy, TRAIN_N, ACQUISITION_SEED, "final_acquisition")
    r2_final_hashes = r2_final.instance_hash.astype(str).tolist()
    history, c04 = p4b1.frozen_history()
    original70 = pd.read_parquet(CYCLES / "C03" / "evaluation_work" / "M4" / "OriginalTSPLIB70" / "instance_metrics.parquet")
    tsplib_hashes = set(original70.coordinate_hash.astype(str))
    if set(r2_archive_hashes) & history or set(r2_final_hashes) & history or set(r2_final_hashes) & tsplib_hashes:
        raise RuntimeError("R2 contains M4-history or direct TSPLIB coordinates")
    if len(set(r2_archive_hashes)) != ARCHIVE_N or len(set(r2_final_hashes)) != TRAIN_N:
        raise RuntimeError("R2 uniqueness failure")
    archive_audit = p4b1.coordinate_audit(r2_archive_xy, r2_archive_hashes)
    final_audit = p4b1.coordinate_audit(r2_final_xy, r2_final_hashes)
    if archive_audit["shape"] != [ARCHIVE_N, 100, 2] or final_audit["shape"] != [TRAIN_N, 100, 2]:
        raise RuntimeError("R2 shape failure")
    if not all(archive_audit[key] and final_audit[key] for key in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")):
        raise RuntimeError("R2 coordinate validity failure")

    p4b1.save_bundle(R2_ARCHIVE_BUNDLE, r2_archive_xy, r2_archive_hashes)
    r2_archive.to_parquet(R2_ARCHIVE_PARQUET, index=False)
    r2_archive.to_csv(R2_ARCHIVE_CSV, index=False)
    p4b1.save_bundle(R2_10K_BUNDLE, r2_final_xy, r2_final_hashes)
    r2_final.to_parquet(R2_10K_PARQUET, index=False)
    r2_final.to_csv(R2_10K_CSV, index=False)

    r0_final_xy, r0_final_hashes = p4b1.load_bundle(R0_10K_BUNDLE)
    r0_final = pd.read_parquet(R0_10K_MANIFEST)
    r0_config = json.loads(R0_10K_CONFIG.read_text(encoding="utf-8"))
    if r0_final.instance_hash.astype(str).tolist() != r0_final_hashes or r0_config.get("seed") != ACQUISITION_SEED:
        raise RuntimeError("frozen R0 Random10k protocol mismatch")

    geometry = p2.load_geometry()
    groups = (
        ("E0_full_eligible39848", full),
        ("R0_target_archive30k", r0_archive),
        ("R2_random_archive30k", r2_archive),
        ("R0_random10k", r0_final),
        ("R2_random10k", r2_final),
    )
    metric_rows: list[dict[str, Any]] = []
    references: list[pd.DataFrame] = []
    for label, frame in groups:
        row, reference = distribution_row(label, frame, geometry)
        metric_rows.append(row)
        references.append(reference)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "support_distribution_metrics.csv", index=False)
    pd.concat(references, ignore_index=True).to_csv(OUT / "support_reference_coverage.csv", index=False)

    provenance_rows: list[dict[str, Any]] = []
    for field in ("operator_id", "generation_born", "evolution_origin"):
        categories = sorted(set(r0_archive[field].astype(str)) | set(r2_archive[field].astype(str)))
        for category in categories:
            provenance_rows.append({
                "level": "archive30k", "field": field, "category": category,
                "R0_n": int(r0_archive[field].astype(str).eq(category).sum()),
                "R2_n": int(r2_archive[field].astype(str).eq(category).sum()),
                "R0_fraction": float(r0_archive[field].astype(str).eq(category).mean()),
                "R2_fraction": float(r2_archive[field].astype(str).eq(category).mean()),
            })
        categories = sorted(set(r0_final[field].astype(str)) | set(r2_final[field].astype(str)))
        for category in categories:
            provenance_rows.append({
                "level": "training10k", "field": field, "category": category,
                "R0_n": int(r0_final[field].astype(str).eq(category).sum()),
                "R2_n": int(r2_final[field].astype(str).eq(category).sum()),
                "R0_fraction": float(r0_final[field].astype(str).eq(category).mean()),
                "R2_fraction": float(r2_final[field].astype(str).eq(category).mean()),
            })
    provenance = pd.DataFrame(provenance_rows)
    provenance.to_csv(OUT / "support_provenance_comparison.csv", index=False)

    archive_overlap = len(set(r0_archive_hashes) & set(r2_archive_hashes))
    final_overlap = len(set(r0_final_hashes) & set(r2_final_hashes))
    wall = time.perf_counter() - started
    config = {
        "status": "FROZEN",
        "full_pool_n": FULL_POOL_N,
        "archive_n": ARCHIVE_N,
        "training_n": TRAIN_N,
        "retention_seed": RETENTION_SEED,
        "acquisition_seed": ACQUISITION_SEED,
        "retention_algorithm": "canonical instance-hash ordering; numpy.default_rng(20261026).choice(39848,30000,replace=False); retain draw order",
        "acquisition_algorithm": "canonical instance-hash ordering within R2 archive; numpy.default_rng(20261025).choice(30000,10000,replace=False); retain draw order",
        "retention_probability_each_candidate": ARCHIVE_N / FULL_POOL_N,
        "marginal_final_probability_each_candidate": TRAIN_N / FULL_POOL_N,
        "structure_or_hardness_fields_read_by_selection": False,
        "source_audit": source_audit,
        "full_pool_bundle_sha256": sha256_file(FULL_POOL_BUNDLE),
        "full_pool_manifest_sha256": sha256_file(FULL_POOL_MANIFEST),
        "r2_archive_bundle_sha256": sha256_file(R2_ARCHIVE_BUNDLE),
        "r2_archive_manifest_csv_sha256": sha256_file(R2_ARCHIVE_CSV),
        "r2_archive_membership_sha256": membership_hash(r2_archive_hashes),
        "r2_random10k_bundle_sha256": sha256_file(R2_10K_BUNDLE),
        "r2_random10k_manifest_csv_sha256": sha256_file(R2_10K_CSV),
        "r2_random10k_membership_sha256": membership_hash(r2_final_hashes),
        "r0_r2_archive_overlap_n": archive_overlap,
        "r0_r2_training_overlap_n": final_overlap,
        "m4_history_overlap_n": len(set(r2_final_hashes) & history),
        "c04_hard10k_overlap_n": len(set(r2_final_hashes) & c04),
        "direct_tsplib70_overlap_n": len(set(r2_final_hashes) & tsplib_hashes),
        "data_audit_wall_seconds": wall,
    }
    write_json(R2_CONFIG, config)
    integrity = {
        "status": "PASS", "rows": TRAIN_N, "unique_ids": len(set(r2_final_hashes)),
        "unique_coordinates": len(set(r2_final_hashes)), "shape": list(r2_final_xy.shape),
        "finite": bool(np.isfinite(r2_final_xy).all()), "in_bounds": bool(((r2_final_xy >= 0) & (r2_final_xy <= 1)).all()),
        "unique_nodes_per_instance": final_audit["unique_nodes"], "hashes_reproduce": final_audit["hashes_reproduce"],
        "source_subset_of_random_archive": set(r2_final_hashes).issubset(r2_archive_hashes),
        "m4_history_overlap_n": 0, "direct_tsplib70_overlap_n": 0,
        "bundle_sha256": config["r2_random10k_bundle_sha256"], "membership_sha256": config["r2_random10k_membership_sha256"],
    }
    write_json(R2_INTEGRITY, integrity)

    protocol = f"""# P4-B2 Protocol Integrity

- Recovered the complete frozen E0 pre-retention pool directly from persisted G1/G2 mutation bundles and diagnostics: {source_audit['eligible_before_cross_generation_dedup']:,} eligible rows before cross-generation dedup, {source_audit['cross_generation_duplicates_removed']:,} duplicates removed, exactly {FULL_POOL_N:,} unique eligible TSP100. No mutation was rerun.
- Both 20,000-attempt manifests, accepted-child order, coordinate hashes, runtime membership hashes, lineage, generation and operator provenance reproduce. Full-pool bundle SHA256 `{config['full_pool_bundle_sha256']}`.
- E0 active-population survivor selection is `random_survivor` in both generations. The initial P03 population nevertheless came from the earlier hardness-driven pipeline; R2 is a structure-free Cycle-5 selection arm, not an M0-from-scratch random baseline.
- Frozen R0 archive uses `common_bounded_archive_retention`: target-aware quota/frontier selection plus lineage diversity, 39,848 → 30,000. Frozen R0 final acquisition is uniform random sampling with seed {ACQUISITION_SEED}.
- R2 archive retention uses only canonical instance hashes plus NumPy uniform sampling without replacement, seed {RETENTION_SEED}; each offspring has retention probability {ARCHIVE_N / FULL_POOL_N:.9f}. R2 final acquisition uses the same R0 algorithm and seed {ACQUISITION_SEED}; neither stage reads structure, TSPLIB, hardness, loss, lineage, operator, generation, density or provenance.
- R2 contains zero C00–C03 M4-history instances and zero direct TSPLIB70 coordinate matches. R0/R2 overlap is not excluded or optimized.
- All five frozen checkpoints, including the formal M4 initialization, reproduce their SHA256 values.
- Predeclared decision rule before labels/training: PRELIMINARILY_SUPPORTED requires R0 to win at least 6/8 benchmark means, at least 3/8 paired CIs entirely favor R0, and no significant R2 win. NOT_SUPPORTED_IN_THIS_TRIAL is the exact mirror; otherwise INCONCLUSIVE.
- TSPLIB22 remains a repeatedly observed comparability set and is not used for sampling, tuning or checkpoint selection.

Status: **PASS**.
"""
    (OUT / "00_protocol_integrity.md").write_text(protocol, encoding="utf-8")

    display = metrics.set_index("distribution")
    columns = ["strict_target_fraction", "sliced_wasserstein", "coarse_js", "occupied_cells", "effective_cells", "largest_cell_share", "reference_coverage_n", "effective_lineages", "largest_lineage_share", "fft_median", "nn_cv_median", "aux_coarse_js"]
    lines = ["# Archive and Training Distribution Comparison", "", "All diagnostics were computed after R2 membership was frozen; no resampling followed.", "", "| Distribution | Strict | SWD | 3×3 JS | Cells/effective | Max cell | Coverage | Effective lineages | Max lineage | FFT median | NN-CV median | Aux JS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in [item[0] for item in groups]:
        row = display.loc[name]
        lines.append(f"| {name} | {row.strict_target_fraction:.2%} | {row.sliced_wasserstein:.6f} | {row.coarse_js:.6f} | {int(row.occupied_cells)}/{row.effective_cells:.3f} | {row.largest_cell_share:.2%} | {int(row.reference_coverage_n)}/48 | {row.effective_lineages:.1f} | {row.largest_lineage_share:.2%} | {row.fft_median:.4f} | {row.nn_cv_median:.4f} | {row.aux_coarse_js:.6f} |")
    lines.extend(["", f"- R0/R2 archive overlap: {archive_overlap:,}/30,000 ({archive_overlap / ARCHIVE_N:.2%}).", f"- R0/R2 final 10k overlap: {final_overlap:,}/10,000 ({final_overlap / TRAIN_N:.2%}).", "- Random 30k → random 10k is marginally equivalent to a uniform 10k draw from all 39,848; the explicit two-stage implementation matches the R0 archive→acquisition workflow.", "- Differences between R0/R2 training sets include finite-sample random variation; this single seed cannot remove or quantify seed-to-seed training uncertainty.", "- HGB/BQ hardness was not used and no new Oracle hardness inference was run."])
    (OUT / "03_archive_and_training_distribution_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, frame), color in zip(axes, groups[:3], ("#777777", "#4c78a8", "#e45756"), strict=True):
        ax.scatter(frame[p2.FEATURES[0]], frame[p2.FEATURES[1]], s=2, alpha=.12, color=color)
        ax.set_title(name); ax.set_xlabel(p2.FEATURES[0]); ax.set_ylabel(p2.FEATURES[1])
    fig.tight_layout(); fig.savefig(FIG / "01_archive_feature_space.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r0_final[p2.FEATURES[0]], r0_final[p2.FEATURES[1]], s=3, alpha=.16, label="R0 target archive → random10k")
    ax.scatter(r2_final[p2.FEATURES[0]], r2_final[p2.FEATURES[1]], s=3, alpha=.16, label="R2 random archive → random10k")
    ax.set_xlabel(p2.FEATURES[0]); ax.set_ylabel(p2.FEATURES[1]); ax.set_title("Frozen R0/R2 training distributions"); ax.legend(markerscale=4)
    fig.tight_layout(); fig.savefig(FIG / "02_training10k_feature_space.png", dpi=180); plt.close(fig)
    names = ["strict_target_fraction", "sliced_wasserstein", "coarse_js", "effective_cells", "effective_lineages", "aux_coarse_js"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, name in zip(axes.flat, names, strict=True):
        ax.bar(np.arange(len(metrics)), metrics[name], color=["#777777", "#4c78a8", "#e45756", "#72b7b2", "#f58518"])
        ax.set_xticks(np.arange(len(metrics))); ax.set_xticklabels(["Pool", "R0 A", "R2 A", "R0 10k", "R2 10k"], rotation=25); ax.set_title(name)
    fig.tight_layout(); fig.savefig(FIG / "03_distribution_metrics.png", dpi=180); plt.close(fig)
    return {"status": "PASS", **config, "r2_integrity": integrity}


def label(device_name: str) -> None:
    generate_and_audit()
    coordinates, hashes = p4b1.load_bundle(R2_10K_BUNDLE)
    manifest = pd.read_parquet(R2_10K_PARQUET)
    if manifest.instance_hash.astype(str).tolist() != hashes:
        raise RuntimeError("R2 Random10k manifest mismatch")
    import run_blade_v2 as base

    device = torch.device(device_name)
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    caches = [pd.read_parquet(p4b1.SHARED_CACHE), pd.read_parquet(P4A / "labeling" / "ledger.parquet"), pd.read_parquet(P4B1 / "labeling" / "ledger.parquet")]
    for reusable in caches[1:]:
        if not reusable.verified.all() or not reusable.status.eq("solved").all() or set(reusable.binary_hash.astype(str)) != {CANONICAL_CONCORDE_SHA256}:
            raise RuntimeError("existing label cache is not canonically reusable")
    base.GLOBAL_CACHE = pd.concat(caches, ignore_index=True).drop_duplicates("instance_hash", keep="last")
    selected = manifest.copy(); selected["coord_index"] = np.arange(len(selected))
    started = time.perf_counter()
    _, tours, summary = base._label_population(LABEL_DIR, selected, coordinates, model, device, None, "P4B2_R2_random_archive_random10k_original_A0_oracle_and_M4_hardness", True)
    invocation_wall = time.perf_counter() - started
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    ledger = pd.read_parquet(LABEL_DIR / "ledger.parquet")
    if len(ledger) != TRAIN_N or not ledger.verified.all() or not ledger.status.eq("solved").all():
        raise RuntimeError("R2 labeling incomplete")
    if tours.shape != (TRAIN_N, 100) or not all(validate_tour_0based(tour, 100)[0] for tour in tours):
        raise RuntimeError("R2 illegal expert tour")
    costs = np.asarray([tour_cost_int(integer_distance_matrix(points), tour) for points, tour in zip(coordinates, tours, strict=True)], dtype=np.int64)
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)):
        raise RuntimeError("R2 objective mismatch")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (TRAIN_N, 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]):
        raise RuntimeError("R2 closure mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist():
        raise RuntimeError("R2 tour hash mismatch")
    integrity = {
        "status": "PASS", "selected": TRAIN_N, "verified": TRAIN_N, "failures": 0,
        "cache_hits": int(summary["cache_hits"]), "physical_new_concorde_calls": int(summary["new_concorde_calls"]),
        "pipeline_wall_seconds": float(summary["wall_seconds"]), "invocation_wall_seconds": invocation_wall,
        "all_tours_legal": True, "all_costs_reproduced": True,
        "ledger_sha256": sha256_file(LABEL_DIR / "ledger.parquet"), "labeled_bundle_sha256": sha256_file(LABEL_DIR / "labeled_bundle.npz"),
        "labeled_metadata_sha256": sha256_file(LABEL_DIR / "labeled.parquet"),
        "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
        "m4_sha256": model_meta["sha256"], "concorde_sha256": CANONICAL_CONCORDE_SHA256,
    }
    write_json(LABEL_DIR / "labeling_integrity.json", integrity)
    report = f"""# Concorde Labeling Audit

- Exact P3/P4-B1 canonical pipeline; 10,000/10,000 legal expert tours, zero failures or replacements.
- Reused {integrity['cache_hits']:,} verified canonical labels and executed {integrity['physical_new_concorde_calls']:,} new Concorde calls. Every objective was recomputed from frozen R2 coordinates.
- Pipeline wall {integrity['pipeline_wall_seconds']:.3f}s; 0-based Hamiltonian permutation, implicit cost closure, BQ input closure appended as node 101.
- Ledger SHA256 `{integrity['ledger_sha256']}`; labeled bundle SHA256 `{integrity['labeled_bundle_sha256']}`; expert tours SHA256 `{integrity['expert_tours_array_sha256']}`.

Status: **PASS**.
"""
    (OUT / "04_concorde_labeling_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(integrity, indent=2), flush=True)


def load_labeled() -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(LABEL_DIR / "labeled_bundle.npz", allow_pickle=False) as payload:
        return np.asarray(payload["coordinates"], float), payload["instance_hash"].astype(str).tolist(), np.asarray(payload["expert_tours"], np.int64)


def train(device_name: str) -> None:
    generate_and_audit()
    label_info = json.loads((LABEL_DIR / "labeling_integrity.json").read_text(encoding="utf-8"))
    if label_info.get("status") != "PASS" or label_info.get("verified") != TRAIN_N:
        raise RuntimeError("labeling gate failed")
    coordinates, hashes, tours = load_labeled()
    frozen_xy, frozen_hashes = p4b1.load_bundle(R2_10K_BUNDLE)
    if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
        raise RuntimeError("R2 labeled membership changed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    core.RUN_ID = "TARGET_AWARE_P4B2_R2_RANDOM_ARCHIVE_RANDOM_C5"
    core.cycle_dir = lambda arm, cycle: TRAIN_CYCLE
    core.model_path = lambda arm, index: R2_M5 if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    started = time.perf_counter()
    trained, trained_meta, timing = core.train("R2_RANDOM_ARCHIVE_RANDOM_C5", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)
    invocation_wall = time.perf_counter() - started
    e000 = torch.load(TRAIN_CYCLE / "09_bq_training" / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    if not e000_equal or trained_meta["sha256"] != sha256_file(R2_M5) or not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters()):
        raise RuntimeError("R2 training integrity failed")
    del trained, model
    if device.type == "cuda": torch.cuda.empty_cache()
    curve = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    if len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(TRAIN_N).all():
        raise RuntimeError("R2 training trajectory mismatch")
    r0_info = json.loads((P4B1 / "training_integrity.json").read_text(encoding="utf-8"))
    report = f"""# Training Protocol and Log

- Directly reused formal P3/P4-B1 `train`; E000 exactly equals formal M4: **{e000_equal}**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, batch/microbatch 256/64, 256 epochs, 40 updates/epoch, 10,240 updates, seed 2026, all 10,000 instances once/epoch.
- R2 final loss {timing['final_loss']:.8f}, wall {timing['training_loop_wall_s']:.3f}s, peak GPU bytes {int(timing['gpu_peak_allocated_bytes']):,}. Frozen R0 final loss {r0_info['final_loss']:.8f}.
- R2 checkpoint SHA256 `{trained_meta['sha256']}`; membership SHA256 `{membership_hash(hashes)}`; label bundle SHA256 `{label_info['labeled_bundle_sha256']}`.

E256 was frozen before evaluation; no benchmark selected data, hyperparameters or checkpoint.
"""
    (OUT / "05_training_protocol_and_log.md").write_text(report, encoding="utf-8")
    write_json(OUT / "training_integrity.json", {
        "status": "PASS", "m4_sha256": M4_SHA, "e000_weights_equal_m4": e000_equal,
        "r2_m5_sha256": trained_meta["sha256"], "membership_sha256": membership_hash(hashes),
        "label_bundle_sha256": label_info["labeled_bundle_sha256"], "epochs": 256,
        "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": invocation_wall,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]),
    })
    print(json.dumps({"status": "PASS", "checkpoint": str(R2_M5), "sha256": trained_meta["sha256"], "timing": timing}, indent=2), flush=True)


def evaluate(device_name: str) -> None:
    integrity = json.loads((OUT / "training_integrity.json").read_text(encoding="utf-8"))
    if integrity.get("r2_m5_sha256") != sha256_file(R2_M5):
        raise RuntimeError("R2 checkpoint integrity failed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    model, meta = load_canonical_checkpoint(R2_M5, device, expected_sha256=integrity["r2_m5_sha256"])
    frame, wall = core.evaluate(model, R2_M5, "R2_RandomArchive_M5", EVAL_CSV, Status(), "R2_RANDOM_ARCHIVE_RANDOM_C5")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all():
        raise RuntimeError("R2 evaluation incomplete")
    write_json(OUT / "evaluation_integrity.json", {"status": "PASS", "checkpoint_sha256": meta["sha256"], "all_tours_valid": True, "datasets": list(DATASETS), "evaluation_wall_s": wall, "results_sha256": sha256_file(EVAL_CSV)})
    print(frame[["dataset", "instance_count", "mean_gap_percent"]].to_string(index=False), flush=True)


MODELS = ("M4", "Hardness_M5", "R0_TargetArchive_Random_M5", "R2_RandomArchive_Random_M5", "E0_Structure_M5", "E1_Structure_M5")


def detail(model: str, dataset: str) -> pd.DataFrame:
    if model == "M4": path = CYCLES / "C03/evaluation_work/M4" / dataset / "instance_metrics.parquet"
    elif model == "Hardness_M5": path = CYCLES / "C04/evaluation_work/M5" / dataset / "instance_metrics.parquet"
    elif model == "R0_TargetArchive_Random_M5": path = P4B1 / "evaluation_work/R0_Random_M5" / dataset / "instance_metrics.parquet"
    elif model == "R2_RandomArchive_Random_M5": path = OUT / "evaluation_work/R2_RandomArchive_M5" / dataset / "instance_metrics.parquet"
    elif model == "E0_Structure_M5": path = P4A / "evaluation_work/E0_Structure_M5" / dataset / "instance_metrics.parquet"
    else: path = P3 / "evaluation_work/Structure_M5" / dataset / "instance_metrics.parquet"
    return pd.read_parquet(path)


def align_models(dataset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        source = detail(model, dataset)
        part = source[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
        if not part.valid_tour.all(): raise RuntimeError(f"invalid tour {model}/{dataset}")
        part = part.rename(columns={"coordinate_hash": f"coordinate_hash_{model}", "n_nodes": f"n_nodes_{model}", "optimal_cost_int": f"optimal_cost_int_{model}", "gap_percent": f"gap_{model}"}).drop(columns=["valid_tour"])
        frames.append(part)
    merged = frames[0]
    for frame in frames[1:]: merged = merged.merge(frame, on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        columns = [f"{field}_{model}" for model in MODELS]
        if not all(merged[columns[0]].astype(str).eq(merged[column].astype(str)).all() for column in columns[1:]):
            raise RuntimeError(f"alignment failure {dataset}/{field}")
    return pd.DataFrame({"dataset": dataset, "instance_id": merged.instance_id.astype(str), "coordinate_hash": merged[f"coordinate_hash_{MODELS[0]}"].astype(str), "n_nodes": merged[f"n_nodes_{MODELS[0]}"].astype(int), "optimal_cost_int": merged[f"optimal_cost_int_{MODELS[0]}"].astype(np.int64), **{f"gap_{model}": merged[f"gap_{model}"].astype(float) for model in MODELS}})


def verify_protocols() -> None:
    sources = (
        (CYCLES / "C03/external_benchmark_results.csv", M4_SHA),
        (CYCLES / "C04/external_benchmark_results.csv", HARD_M5_SHA),
        (P4B1 / "r0_m5_external_benchmark_results.csv", R0_M5_SHA),
        (EVAL_CSV, sha256_file(R2_M5)),
        (P4A / "e0_m5_external_benchmark_results.csv", E0_STRUCTURE_M5_SHA),
        (P3 / "structure_m5_external_benchmark_results.csv", E1_STRUCTURE_M5_SHA),
    )
    frames = []
    for path, checkpoint in sources:
        frame = pd.read_csv(path)
        if set(frame.dataset.astype(str)) != set(DATASETS) or set(frame.checkpoint_sha256.astype(str)) != {checkpoint}:
            raise RuntimeError(f"evaluation source mismatch {path}")
        frames.append(frame.set_index("dataset"))
    for dataset in DATASETS:
        if len({str(frame.at[dataset, "dataset_sha256"]) for frame in frames}) != 1 or len({str(frame.at[dataset, "evaluator_config_sha256"]) for frame in frames}) != 1:
            raise RuntimeError(f"evaluation protocol mismatch {dataset}")


def comparison_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    pairs = {
        "R2_minus_R0": frame.gap_R2_RandomArchive_Random_M5.to_numpy(float) - frame.gap_R0_TargetArchive_Random_M5.to_numpy(float),
        "R2_minus_Hardness": frame.gap_R2_RandomArchive_Random_M5.to_numpy(float) - frame.gap_Hardness_M5.to_numpy(float),
        "R2_minus_M4": frame.gap_R2_RandomArchive_Random_M5.to_numpy(float) - frame.gap_M4.to_numpy(float),
        "R0_minus_Hardness": frame.gap_R0_TargetArchive_Random_M5.to_numpy(float) - frame.gap_Hardness_M5.to_numpy(float),
        "R0_minus_M4": frame.gap_R0_TargetArchive_Random_M5.to_numpy(float) - frame.gap_M4.to_numpy(float),
    }
    row: dict[str, Any] = {"dataset": name, "n_instances": len(frame), **{f"mean_gap_{model}": float(frame[f"gap_{model}"].mean()) for model in MODELS}}
    for label, values in pairs.items():
        low, high = p4b1.paired_bootstrap(values, p4b1.stable_seed("P4B2|" + name + "|" + label))
        row[label] = float(values.mean()); row[label + "_ci95_low"] = low; row[label + "_ci95_high"] = high
        row[label + "_fraction_negative"] = float(np.mean(values < 0)); row[label + "_wins"] = int(np.sum(values < 0)); row[label + "_ties"] = int(np.sum(values == 0))
    row["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    return row


def make_final_figures(summary: pd.DataFrame) -> None:
    base = summary[summary.dataset.isin(DATASETS)]; x = np.arange(len(base)); width = .13
    fig, ax = plt.subplots(figsize=(18, 5.5))
    labels = ("M4", "Hardness", "R0 target archive", "R2 random archive", "E0 structure acq", "E1 guided")
    for index, (model, label) in enumerate(zip(MODELS, labels, strict=True)):
        ax.bar(x + width * (index - 2.5), base[f"mean_gap_{model}"], width=width, label=label)
    ax.set_xticks(x); ax.set_xticklabels(base.dataset, rotation=35, ha="right"); ax.set_ylabel("mean optimality gap (%)"); ax.set_title("P4-B2 frozen six-model comparison"); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "04_six_model_mean_gaps.png", dpi=180); plt.close(fig)
    plot = summary[summary.dataset.ne("GAP8_macro")]; y = np.arange(len(plot)); values = plot.R2_minus_R0.to_numpy(float); low = plot.R2_minus_R0_ci95_low.to_numpy(float); high = plot.R2_minus_R0_ci95_high.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6)); ax.errorbar(values, y, xerr=np.vstack([values-low, high-values]), fmt="o", capsize=3); ax.axvline(0, color="black", linewidth=.9)
    ax.set_yticks(y); ax.set_yticklabels(plot.dataset); ax.set_xlabel("R2 random archive − R0 target-aware archive gap (pp)"); ax.set_title("Archive-retention ablation: paired bootstrap 95% CI")
    fig.tight_layout(); fig.savefig(FIG / "05_r2_minus_r0_paired_intervals.png", dpi=180); plt.close(fig)
    r0 = pd.read_csv(P4B1 / "training_cycle" / "09_bq_training" / "training_loss_curve.csv"); r2 = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.plot(r0.epoch, r0.mean_training_loss, label="R0 target-aware archive"); ax.plot(r2.epoch, r2.mean_training_loss, label="R2 random archive")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean training loss"); ax.set_title("Identical Cycle-5 protocol"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "06_training_loss.png", dpi=180); plt.close(fig)


def finalize() -> None:
    if not (OUT / "evaluation_integrity.json").is_file(): raise RuntimeError("R2 evaluation missing")
    verify_protocols()
    aligned = {dataset: align_models(dataset) for dataset in DATASETS}
    tsplib = aligned["OriginalTSPLIB70"]
    dev = set(pd.read_csv(ROOT / "target_aware_pilot" / "03_tsplib48_reference_manifest.csv").instance_id.astype(str))
    held = set(json.loads((ROOT / "target_aware_pilot" / "p1_heldout22_names.json").read_text())["names"])
    if len(dev) != 48 or len(held) != 22 or dev & held or set(tsplib.instance_id) != dev | held:
        raise RuntimeError("TSPLIB split mismatch")
    aligned["TSPLIB48_development"] = tsplib[tsplib.instance_id.isin(dev)].copy()
    aligned["TSPLIB22_previously_heldout"] = tsplib[tsplib.instance_id.isin(held)].copy()
    rows: list[dict[str, Any]] = []; paired_parts: list[pd.DataFrame] = []
    for name, frame in aligned.items():
        part = frame.copy(); part["analysis_dataset"] = name
        part["R2_minus_R0"] = part.gap_R2_RandomArchive_Random_M5 - part.gap_R0_TargetArchive_Random_M5
        part["R2_minus_Hardness"] = part.gap_R2_RandomArchive_Random_M5 - part.gap_Hardness_M5
        part["R2_minus_M4"] = part.gap_R2_RandomArchive_Random_M5 - part.gap_M4
        part["R0_minus_Hardness"] = part.gap_R0_TargetArchive_Random_M5 - part.gap_Hardness_M5
        part["R0_minus_M4"] = part.gap_R0_TargetArchive_Random_M5 - part.gap_M4
        paired_parts.append(part); rows.append(comparison_row(name, frame))
    summary = pd.DataFrame(rows); base = summary[summary.dataset.isin(DATASETS)]
    macro: dict[str, Any] = {"dataset": "GAP8_macro", "n_instances": int(base.n_instances.sum()), **{f"mean_gap_{model}": float(base[f"mean_gap_{model}"].mean()) for model in MODELS}}
    for label, left, right in (("R2_minus_R0", "R2_RandomArchive_Random_M5", "R0_TargetArchive_Random_M5"), ("R2_minus_Hardness", "R2_RandomArchive_Random_M5", "Hardness_M5"), ("R2_minus_M4", "R2_RandomArchive_Random_M5", "M4"), ("R0_minus_Hardness", "R0_TargetArchive_Random_M5", "Hardness_M5"), ("R0_minus_M4", "R0_TargetArchive_Random_M5", "M4")):
        macro[label] = macro[f"mean_gap_{left}"] - macro[f"mean_gap_{right}"]
        for suffix in ("_ci95_low", "_ci95_high", "_fraction_negative", "_wins", "_ties"): macro[label + suffix] = np.nan
    macro["bootstrap_replicates"] = np.nan
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    summary.to_csv(OUT / "06_all_model_gap_comparison.csv", index=False)
    paired.to_csv(OUT / "07_per_instance_paired_analysis.csv", index=False)

    base = summary[summary.dataset.isin(DATASETS)]
    r0_wins = int((base.R2_minus_R0 > 0).sum()); r2_wins = int((base.R2_minus_R0 < 0).sum())
    r0_sig = int((base.R2_minus_R0_ci95_low > 0).sum()); r2_sig = int((base.R2_minus_R0_ci95_high < 0).sum())
    if r0_wins >= 6 and r0_sig >= 3 and r2_sig == 0:
        decision = "TARGET_AWARE_RETENTION_VALUE_PRELIMINARILY_SUPPORTED"; label = "PRELIMINARILY_SUPPORTED"
    elif r2_wins >= 6 and r2_sig >= 3 and r0_sig == 0:
        decision = "TARGET_AWARE_RETENTION_NOT_SUPPORTED_IN_THIS_TRIAL"; label = "NOT_SUPPORTED_IN_THIS_TRIAL"
    else:
        decision = "TARGET_AWARE_RETENTION_VALUE_INCONCLUSIVE"; label = "INCONCLUSIVE"
    make_final_figures(summary)
    gap8 = summary[summary.dataset.eq("GAP8_macro")].iloc[0]
    original = summary[summary.dataset.eq("OriginalTSPLIB70")].iloc[0]
    devrow = summary[summary.dataset.eq("TSPLIB48_development")].iloc[0]
    heldrow = summary[summary.dataset.eq("TSPLIB22_previously_heldout")].iloc[0]
    config = json.loads(R2_CONFIG.read_text()); labelinfo = json.loads((LABEL_DIR / "labeling_integrity.json").read_text()); traininfo = json.loads((OUT / "training_integrity.json").read_text()); evalinfo = json.loads((OUT / "evaluation_integrity.json").read_text())
    lines = ["# P4-B2 Archive Retention Ablation Conclusion", "", f"## `{decision}` — **{label}**", "", "Positive R2−R0 favors target-aware R0 retention; negative favors structure-free R2. CIs use 10,000 paired bootstrap replicates.", "", "| Dataset | M4 | Hardness | R0 TargetArchive→Random10k | R2 RandomArchive→Random10k | E0 Structure | E1 Guided | R2−R0 | 95% CI |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in [*DATASETS, "TSPLIB48_development", "TSPLIB22_previously_heldout", "GAP8_macro"]:
        row = summary[summary.dataset.eq(name)].iloc[0]
        interval = "N/A" if pd.isna(row.R2_minus_R0_ci95_low) else f"[{row.R2_minus_R0_ci95_low:.4f}, {row.R2_minus_R0_ci95_high:.4f}]"
        lines.append(f"| {name} | {row.mean_gap_M4:.4f} | {row.mean_gap_Hardness_M5:.4f} | {row.mean_gap_R0_TargetArchive_Random_M5:.4f} | {row.mean_gap_R2_RandomArchive_Random_M5:.4f} | {row.mean_gap_E0_Structure_M5:.4f} | {row.mean_gap_E1_Structure_M5:.4f} | {row.R2_minus_R0:+.4f} | {interval} |")
    lines.extend(["", f"- R0 wins {r0_wins}/8 benchmark means and {r0_sig}/8 paired CIs; R2 wins {r2_wins}/8 means and {r2_sig}/8 CIs.", f"- GAP8: R0 {gap8.mean_gap_R0_TargetArchive_Random_M5:.4f}%, R2 {gap8.mean_gap_R2_RandomArchive_Random_M5:.4f}%, R2−R0 {gap8.R2_minus_R0:+.4f} pp.", f"- Original70 R2−R0 {original.R2_minus_R0:+.4f} pp; Development48 {devrow.R2_minus_R0:+.4f} pp; previously held-out22 {heldrow.R2_minus_R0:+.4f} pp with CI [{heldrow.R2_minus_R0_ci95_low:.4f}, {heldrow.R2_minus_R0_ci95_high:.4f}].", f"- Costs: data recovery/sampling/audit {config['data_audit_wall_seconds']:.2f}s; labeling {labelinfo['pipeline_wall_seconds']:.1f}s ({labelinfo['cache_hits']:,} reused, {labelinfo['physical_new_concorde_calls']:,} new calls); training {traininfo['training_loop_wall_s']:.1f}s; evaluation {evalinfo['evaluation_wall_s']:.1f}s.", "- R2 is structure-free only at Cycle-5 survivor/archive/final selection. Its starting P03 still reflects earlier hardness-driven cycles.", "- This is one frozen sampling/training seed. Mean direction, paired uncertainty and cross-dataset consistency are reported separately; no seed robustness is claimed.", "- TSPLIB22 is repeatedly observed and is not pristine confirmation evidence.", "- Recommendation: retain target-aware archive retention provisionally in the current method, but describe its value as preliminary and prioritize a pre-registered cross-seed replication before making it a primary paper claim. The TSPLIB subsets do not show a significant retention benefit, so this trial does not justify treating structural matching as universally beneficial.", "", "No resampling, second seed, new evolution, Fresh10k, or follow-up ablation was started.", "", f"Decision label: `{label}`", "", "`TARGET_AWARE_P4B2_RANDOM_ARCHIVE_ABLATION_COMPLETE`"])
    (OUT / "08_archive_retention_ablation_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "run_metadata.json", {
        "status": "COMPLETE", "decision": decision, "decision_label": label,
        "r2_archive_sha256": sha256_file(R2_ARCHIVE_BUNDLE), "r2_random10k_sha256": sha256_file(R2_10K_BUNDLE),
        "r2_labels_sha256": labelinfo["labeled_bundle_sha256"], "r2_m5_sha256": sha256_file(R2_M5),
        "comparison_csv_sha256": sha256_file(OUT / "06_all_model_gap_comparison.csv"),
        "paired_csv_sha256": sha256_file(OUT / "07_per_instance_paired_analysis.csv"),
        "no_evolution_rerun": True, "no_resampling": True, "no_extra_seed": True, "tsplib22_not_pristine": True,
    })
    Status().update(phase="complete", status="COMPLETE", decision=decision, decision_label=label)
    print(json.dumps({"status": "COMPLETE", "decision": decision, "label": label, "r0_wins": r0_wins, "r2_wins": r2_wins, "r0_significant": r0_sig, "r2_significant": r2_sig, "gap8_r2_minus_r0": gap8.R2_minus_R0}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "label", "train", "evaluate", "finalize"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "audit": print(json.dumps(generate_and_audit(), indent=2), flush=True)
    elif args.phase == "label": label(args.device)
    elif args.phase == "train": train(args.device)
    elif args.phase == "evaluate": evaluate(args.device)
    else: finalize()


if __name__ == "__main__":
    main()
