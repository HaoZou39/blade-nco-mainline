"""Uniform-rooted, one-sweep FOGA-11 structural bootstrap audit."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact
from mutation import OPERATOR_NAMES, apply_operator
from qd.coordinates import assign_nearest, load_centroids, normalize_qd4
from qd.probes import StructuralCandidate, select_centroid_nearest_probes
from qd.roots import coordinate_hash, recover_uniform_roots


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "audits" / "uniform_rooted_qd_bootstrap_v1"
MASTER_SEED = 2026
RNG_NAMESPACE = "BLADE_UNIFORM_ROOTED_QD_BOOTSTRAP_V1"
ROOT_COUNT = 10_000
K = 1000
N = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def child_seed(root_hash: str, operator_id: str) -> int:
    material = f"{RNG_NAMESPACE}|{MASTER_SEED}|{root_hash}|{operator_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def _coverage(counts: np.ndarray) -> dict[str, Any]:
    values = np.asarray(counts, dtype=np.int64)
    occupied = int(np.count_nonzero(values))
    total = int(values.sum())
    probabilities = values[values > 0] / total if total else np.asarray([], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if total else 0.0
    normalized_entropy = float(entropy / math.log(occupied)) if occupied > 1 else 0.0
    ordered = np.sort(values.astype(np.float64))
    if total:
        indices = np.arange(1, len(ordered) + 1, dtype=np.float64)
        gini = float((2 * np.sum(indices * ordered) / (len(ordered) * total)) - (len(ordered) + 1) / len(ordered))
    else:
        gini = 0.0
    quantiles = np.quantile(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "candidate_count": total,
        "occupied_niches": occupied,
        "occupancy_fraction": occupied / K,
        "empty_niches": K - occupied,
        "candidate_count_per_niche": {
            "min": int(values.min()),
            "P01": float(quantiles[0]),
            "P05": float(quantiles[1]),
            "P10": float(quantiles[2]),
            "P25": float(quantiles[3]),
            "median": float(quantiles[4]),
            "P75": float(quantiles[5]),
            "P90": float(quantiles[6]),
            "P95": float(quantiles[7]),
            "P99": float(quantiles[8]),
            "max": int(values.max()),
        },
        "niches_with_at_least": {str(level): int(np.sum(values >= level)) for level in (1, 2, 5, 10, 20, 50, 100)},
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "gini_concentration": gini,
        "largest_niche_share": float(values.max() / total) if total else 0.0,
    }


def _root_assignments(roots, centroids: np.ndarray) -> pd.DataFrame:
    qd4 = extract_qd4_exact(roots.coordinates)
    normalized = normalize_qd4(qd4)
    niches = assign_nearest(normalized, centroids)
    data: dict[str, Any] = {
        "root_index": roots.root_indices,
        "instance_hash": roots.coordinate_hashes,
        "root_hash": roots.root_hashes,
        "cycle": np.zeros(ROOT_COUNT, dtype=np.int16),
        "niche_id": niches,
        "data_role": ["uniform_root"] * ROOT_COUNT,
    }
    for column, name in enumerate(QD4_FEATURE_NAMES):
        data[name] = qd4[:, column]
        data[f"normalized_{name}"] = normalized[:, column]
    return pd.DataFrame(data)


def _generate_offspring(roots, root_assignments: pd.DataFrame, centroids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    root_niche = dict(zip(root_assignments.instance_hash, root_assignments.niche_id, strict=True))
    root_normalized = {
        row.instance_hash: np.asarray([getattr(row, f"normalized_{name}") for name in QD4_FEATURE_NAMES], dtype=np.float64)
        for row in root_assignments.itertuples(index=False)
    }
    root_hash_set = set(roots.root_hashes)
    seen_children: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    chunk_coordinates: list[np.ndarray] = []
    chunk_metadata: list[dict[str, Any]] = []

    def flush() -> None:
        if not chunk_coordinates:
            return
        coordinates = np.stack(chunk_coordinates)
        qd4 = extract_qd4_exact(coordinates)
        normalized = normalize_qd4(qd4)
        niches = assign_nearest(normalized, centroids)
        rows: list[dict[str, Any]] = []
        for index, meta in enumerate(chunk_metadata):
            parent_norm = root_normalized[meta["parent_hash"]]
            distance = float(np.linalg.norm(normalized[index] - parent_norm))
            row = dict(meta)
            row["parent_niche_id"] = int(root_niche[meta["parent_hash"]])
            row["niche_id"] = int(niches[index])
            row["normalized_qd_displacement"] = distance
            for column, name in enumerate(QD4_FEATURE_NAMES):
                row[name] = float(qd4[index, column])
                row[f"normalized_{name}"] = float(normalized[index, column])
            rows.append(row)
        assignment_frames.append(pd.DataFrame(rows))
        chunk_coordinates.clear()
        chunk_metadata.clear()

    for root_index, (root_hash, coordinates) in enumerate(zip(roots.root_hashes, roots.coordinates, strict=True)):
        for operator_id in OPERATOR_NAMES:
            seed = child_seed(root_hash, operator_id)
            result = apply_operator(operator_id, coordinates, np.random.default_rng(seed))
            child_hash = result.final_output_hash
            accepted = bool(result.valid_after_repair and result.proposal_status == "accepted")
            duplicate = child_hash in seen_children if accepted else False
            if accepted and not duplicate:
                seen_children.add(child_hash)
            manifest_rows.append(
                {
                    "root_index": root_index,
                    "parent_hash": root_hash,
                    "root_hash": root_hash,
                    "operator_id": operator_id,
                    "cycle": 0,
                    "child_seed": np.uint64(seed),
                    "child_hash": child_hash,
                    "raw_child_hash": result.raw_output_hash,
                    "valid_before_repair": bool(result.valid_before_repair),
                    "valid_after_repair": bool(result.valid_after_repair),
                    "proposal_status": result.proposal_status,
                    "failure_reason": result.failure_reason,
                    "repair_count": int(result.repair_count),
                    "duplicate_status": bool(duplicate),
                    "duplicates_uniform_root": child_hash in root_hash_set,
                    "data_role": "candidate_offspring",
                    "expert_tour_available": False,
                    "current_exact_hardness_available": False,
                    "current_surrogate_score": np.nan,
                }
            )
            if accepted and not duplicate:
                chunk_coordinates.append(result.final_coordinates)
                chunk_metadata.append(
                    {
                        "root_index": root_index,
                        "instance_hash": child_hash,
                        "child_hash": child_hash,
                        "parent_hash": root_hash,
                        "root_hash": root_hash,
                        "operator_id": operator_id,
                        "cycle": 0,
                        "data_role": "candidate_offspring",
                    }
                )
        if (root_index + 1) % 250 == 0:
            flush()
        if (root_index + 1) % 1000 == 0:
            print(f"FOGA11_SWEEP_PROGRESS={root_index + 1}/{ROOT_COUNT}", flush=True)
    flush()
    manifest = pd.DataFrame(manifest_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    return manifest, assignments


def _probe_frame(assignments: pd.DataFrame, centroids: np.ndarray) -> pd.DataFrame:
    normalized_columns = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
    candidates = (
        StructuralCandidate(
            instance_hash=row.instance_hash,
            niche_id=int(row.niche_id),
            normalized_qd4=tuple(float(getattr(row, column)) for column in normalized_columns),
        )
        for row in assignments.itertuples(index=False)
    )
    selected = select_centroid_nearest_probes(candidates, centroids)
    selected_hashes = {item.instance_hash: item.centroid_distance for item in selected}
    frame = assignments[assignments.instance_hash.isin(selected_hashes)].copy()
    frame["centroid_distance"] = frame.instance_hash.map(selected_hashes)
    frame["expert_tour_available"] = False
    frame["current_exact_hardness_available"] = False
    frame["oracle_labeled"] = False
    frame["provisional"] = True
    frame["historical_roles"] = frame.data_role.map(lambda role: json.dumps([role, "surrogate_probe"]))
    frame["data_role"] = "surrogate_probe"
    return frame.sort_values("niche_id", kind="stable").reset_index(drop=True)


def _probe_summary(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame.centroid_distance.to_numpy(dtype=np.float64)
    return {
        "probe_count": len(frame),
        "mean_centroid_distance": float(values.mean()),
        "median_centroid_distance": float(np.median(values)),
        "P90_centroid_distance": float(np.quantile(values, 0.90)),
        "P95_centroid_distance": float(np.quantile(values, 0.95)),
        "max_centroid_distance": float(values.max()),
        "oracle_labeled": False,
        "provisional": True,
    }


def _figures(root_counts: np.ndarray, child_counts: np.ndarray, operator: pd.DataFrame, assignments: pd.DataFrame) -> None:
    niches = np.arange(K)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].scatter(niches[root_counts > 0], np.ones(np.count_nonzero(root_counts)), s=5)
    axes[0].set_ylabel("roots occupied")
    axes[1].scatter(niches[child_counts > 0], np.ones(np.count_nonzero(child_counts)), s=5, color="#d95f02")
    axes[1].set_ylabel("offspring occupied"); axes[1].set_xlabel("fixed CVT niche")
    fig.tight_layout(); fig.savefig(AUDIT / "01_root_vs_offspring_niche_occupancy.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.geomspace(1, max(2, int(max(root_counts.max(), child_counts.max()))) + 1, 30)
    ax.hist(root_counts[root_counts > 0], bins=bins, alpha=.65, label="roots")
    ax.hist(child_counts[child_counts > 0], bins=bins, alpha=.65, label="offspring")
    ax.set_xscale("log"); ax.set_xlabel("candidates in occupied niche"); ax.set_ylabel("niches"); ax.legend()
    fig.tight_layout(); fig.savefig(AUDIT / "02_niche_candidate_count_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(operator.operator_id, operator.new_niches_beyond_roots)
    ax.tick_params(axis="x", rotation=55); ax.set_ylabel("new niches beyond roots")
    fig.tight_layout(); fig.savefig(AUDIT / "03_operator_new_niches.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    data = [assignments.loc[assignments.operator_id == name, "normalized_qd_displacement"].to_numpy() for name in OPERATOR_NAMES]
    ax.boxplot(data, tick_labels=OPERATOR_NAMES, showfliers=False, whis=(10, 90))
    ax.tick_params(axis="x", rotation=55); ax.set_ylabel("normalized QD4 displacement")
    fig.tight_layout(); fig.savefig(AUDIT / "04_parent_child_qd_displacement.png", dpi=160); plt.close(fig)

    union_counts = root_counts + child_counts
    labels = ["roots", "offspring", "union"]
    occupied = [np.count_nonzero(root_counts), np.count_nonzero(child_counts), np.count_nonzero(union_counts)]
    fig, ax = plt.subplots(figsize=(7, 5)); bars = ax.bar(labels, occupied, color=["#1b9e77", "#d95f02", "#7570b3"])
    ax.set_ylim(0, K); ax.set_ylabel("occupied K1000 niches"); ax.bar_label(bars)
    fig.tight_layout(); fig.savefig(AUDIT / "05_qd_coverage_expansion_summary.png", dpi=160); plt.close(fig)


def _report(coverage: dict[str, Any], migration: dict[str, Any], operator: pd.DataFrame, probes: dict[str, Any], validation: dict[str, Any]) -> str:
    return f"""# Uniform-rooted Cycle-0 QD bootstrap audit

This structural-only audit starts from the exact canonical 10,000 Uniform roots and applies every frozen FOGA-11 operator exactly once per root. It performs no hardness evaluation, solver inference, training, Concorde call, or external benchmark evaluation.

## Validation

- Nominal children: {validation['nominal_children']}
- Valid children: {validation['valid_children']}
- Invalid children: {validation['invalid_children']}
- Valid unique offspring: {validation['valid_unique_children']}
- Duplicate children: {validation['duplicate_children']}
- Raw-invalid children exposed before canonical repair: {validation['raw_invalid_children']}
- Children repaired by the canonical operator adapter: {validation['children_with_repair']}

## Fixed-map coverage

| pool | occupied | fraction | empty |
|---|---:|---:|---:|
| roots only | {coverage['ROOTS_ONLY']['occupied_niches']} | {coverage['ROOTS_ONLY']['occupancy_fraction']:.4f} | {coverage['ROOTS_ONLY']['empty_niches']} |
| offspring only | {coverage['OFFSPRING_ONLY']['occupied_niches']} | {coverage['OFFSPRING_ONLY']['occupancy_fraction']:.4f} | {coverage['OFFSPRING_ONLY']['empty_niches']} |
| roots + offspring | {coverage['ROOTS_PLUS_OFFSPRING']['occupied_niches']} | {coverage['ROOTS_PLUS_OFFSPRING']['occupancy_fraction']:.4f} | {coverage['ROOTS_PLUS_OFFSPRING']['empty_niches']} |

New niches discovered beyond roots: **{migration['NEW_NICHES_DISCOVERED']}**.

Migration fractions among valid unique offspring: same parent niche {migration['fraction_same_parent_niche']:.4f}; another root-occupied niche {migration['fraction_other_root_occupied_niche']:.4f}; niche absent from roots {migration['fraction_niche_absent_from_roots']:.4f}.

## Operator evidence

The full descriptive operator table is `08_operator_expansion_summary.csv`. All 11 operators remain frozen; these diagnostics do not rank, remove, or retune them.

## Provisional GP0 probes

- Offspring-only candidates: {probes['offspring_only']['probe_count']}
- Roots-plus-offspring candidates: {probes['union']['probe_count']}

Each manifest contains one deterministic centroid-nearest representative per reached niche. These candidates are difficulty-blind, have no Oracle labels, and are not a selected GP0 training source.

## Decision boundary

No coverage pass threshold is applied. The audit stops for human review of acquisition eligibility, probe sufficiency, and whether sparse Cycle-0 coverage should be accepted or expanded before GP0.
"""


def run_uniform_bootstrap_audit() -> dict[str, Any]:
    if AUDIT.exists() and any(AUDIT.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing audit: {AUDIT}")
    AUDIT.mkdir(parents=True, exist_ok=True)
    centroids = load_centroids()
    roots = recover_uniform_roots()
    protocol = {
        "audit": "uniform_rooted_qd_bootstrap_v1",
        "status": "STRUCTURAL_ONLY",
        "created_utc": utc_now(),
        "workspace": str(ROOT),
        "master_seed": MASTER_SEED,
        "rng_namespace": RNG_NAMESPACE,
        "rng_derivation": "SHA256(namespace|master_seed|root_coordinate_hash|operator_id), first 64 little-endian bits",
        "root_rule": "saved_depth == 0 AND ordered canonical root-hash membership",
        "root_count": ROOT_COUNT,
        "operator_ids": list(OPERATOR_NAMES),
        "applications_per_operator_per_root": 1,
        "relay_depth": 1,
        "nominal_offspring": ROOT_COUNT * len(OPERATOR_NAMES),
        "qd4": list(QD4_FEATURE_NAMES),
        "normalization": "frozen deep-reference P1/P99 then clip [0,1]",
        "cvt_k": K,
        "forbidden_operations_executed": {
            "concorde": False,
            "bq_inference": False,
            "teacher_forced_hardness": False,
            "gp_training": False,
            "hgb_training": False,
            "bq_finetuning": False,
            "external_performance_evaluation": False,
            "final_runner": False,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "frozen_inputs": {
            "deep_states": sha256_file(ROOT / "assets/qd/reference_cloud/valid_unique_saved_states.parquet"),
            "root_hashes": sha256_file(ROOT / "assets/qd/roots/root_hashes.txt"),
            "normalization": sha256_file(ROOT / "assets/qd/normalization/normalization_A_P1P99.json"),
            "centroids": sha256_file(ROOT / "assets/qd/cvt/centroids.csv"),
            "operator_config": sha256_file(ROOT / "src/mutation/foga2019_operators.yaml"),
        },
        "old_project_baseline": {
            "git_head": "c1b0f57bc69c952f9092ca704f0f169f6f97322f",
            "tracked_status_entries": 3,
            "tracked_status_sha256": "3372016e4f2d536b62e20aaf7a0ab4cdb99c38b17620bb72bb89e1b47510b8f2",
        },
    }
    write_json(AUDIT / "00_protocol.json", protocol)

    root_frame = _root_assignments(roots, centroids)
    root_frame.to_parquet(AUDIT / "04_root_qd_assignments.parquet", index=False)
    write_json(
        AUDIT / "01_root_manifest.json",
        {
            "count": len(roots.root_hashes),
            "unique_coordinate_hashes": len(set(roots.coordinate_hashes)),
            "N": N,
            "finite_coordinates": bool(np.isfinite(roots.coordinates).all()),
            "unit_square_coordinates": bool(np.all((roots.coordinates >= 0) & (roots.coordinates <= 1))),
            "selection": "saved_depth == 0 AND canonical root-hash membership",
            "regenerated": False,
            "source": "assets/qd/reference_cloud/valid_unique_saved_states.parquet",
            "root_hash_manifest": "assets/qd/roots/root_hashes.txt",
            "root_membership_hash": "edf1719ecc114261ba26b21f7bb8ff113b8247958679abb3d177a858cd7c2c5b",
        },
    )

    offspring, child_qd = _generate_offspring(roots, root_frame, centroids)
    offspring.to_parquet(AUDIT / "02_offspring_manifest.parquet", index=False)
    child_qd.to_parquet(AUDIT / "05_offspring_qd_assignments.parquet", index=False)

    valid = offspring.valid_after_repair & offspring.proposal_status.eq("accepted")
    validation = {
        "nominal_children": len(offspring),
        "valid_children": int(valid.sum()),
        "invalid_children": int((~valid).sum()),
        "valid_unique_children": len(child_qd),
        "duplicate_children": int((valid & offspring.duplicate_status).sum()),
        "children_duplicating_roots": int((valid & offspring.duplicates_uniform_root).sum()),
        "raw_invalid_children": int((~offspring.valid_before_repair).sum()),
        "children_with_repair": int((offspring.repair_count > 0).sum()),
        "unexpected_implementation_errors": 0,
    }
    write_json(AUDIT / "03_validation_summary.json", validation)

    root_counts = np.bincount(root_frame.niche_id, minlength=K)
    child_counts = np.bincount(child_qd.niche_id, minlength=K)
    union_frame = pd.concat([root_frame, child_qd], ignore_index=True, sort=False).drop_duplicates("instance_hash", keep="first")
    union_counts = np.bincount(union_frame.niche_id, minlength=K)
    coverage = {
        "ROOTS_ONLY": _coverage(root_counts),
        "OFFSPRING_ONLY": _coverage(child_counts),
        "ROOTS_PLUS_OFFSPRING": _coverage(union_counts),
        "descriptive_only_no_pass_threshold": True,
    }
    write_json(AUDIT / "06_coverage_summary.json", coverage)
    pd.DataFrame({"niche_id": np.arange(K), "roots_only": root_counts, "offspring_only": child_counts, "roots_plus_offspring": union_counts}).to_csv(AUDIT / "07_niche_count_distribution.csv", index=False)

    root_niches = set(root_frame.niche_id.astype(int))
    child_niches = set(child_qd.niche_id.astype(int))
    same = child_qd.niche_id.eq(child_qd.parent_niche_id)
    other_root = (~same) & child_qd.niche_id.isin(root_niches)
    new = child_qd.niche_id.isin(child_niches - root_niches)
    migration = {
        "root_occupied_niches": len(root_niches),
        "offspring_occupied_niches": len(child_niches),
        "union_occupied_niches": len(root_niches | child_niches),
        "NEW_NICHES_DISCOVERED": len(child_niches - root_niches),
        "valid_unique_offspring_denominator": len(child_qd),
        "same_parent_niche_count": int(same.sum()),
        "other_root_occupied_niche_count": int(other_root.sum()),
        "niche_absent_from_roots_count": int(new.sum()),
        "fraction_same_parent_niche": float(same.mean()),
        "fraction_other_root_occupied_niche": float(other_root.mean()),
        "fraction_niche_absent_from_roots": float(new.mean()),
    }
    write_json(AUDIT / "09_migration_summary.json", migration)

    operator_rows = []
    for operator_id in OPERATOR_NAMES:
        manifest_group = offspring[offspring.operator_id.eq(operator_id)]
        group = child_qd[child_qd.operator_id.eq(operator_id)]
        operator_rows.append(
            {
                "operator_id": operator_id,
                "children_generated": len(manifest_group),
                "valid_unique_children": len(group),
                "occupied_niches": int(group.niche_id.nunique()),
                "new_niches_beyond_roots": len(set(group.niche_id.astype(int)) - root_niches),
                "mean_normalized_qd_displacement": float(group.normalized_qd_displacement.mean()),
                "median_normalized_qd_displacement": float(group.normalized_qd_displacement.median()),
                "P90_normalized_qd_displacement": float(group.normalized_qd_displacement.quantile(.90)),
                "same_niche_rate": float(group.niche_id.eq(group.parent_niche_id).mean()),
            }
        )
    operator_frame = pd.DataFrame(operator_rows)
    operator_frame.to_csv(AUDIT / "08_operator_expansion_summary.csv", index=False)

    child_probe = _probe_frame(child_qd, centroids)
    union_probe = _probe_frame(union_frame, centroids)
    child_probe.to_parquet(AUDIT / "10_gp0_probe_candidates_offspring_only.parquet", index=False)
    union_probe.to_parquet(AUDIT / "11_gp0_probe_candidates_union.parquet", index=False)
    probe_summary = {"offspring_only": _probe_summary(child_probe), "union": _probe_summary(union_probe), "automatic_source_selection": False}
    write_json(AUDIT / "12_gp0_probe_summary.json", probe_summary)

    (AUDIT / "13_human_review_questions.md").write_text(
        "Q1.\nShould Cycle-0 acquisition pool be:\n\nA. offspring only\n\nor\n\nB. Uniform roots + offspring?\n\n"
        "Q2.\nIs one centroid-nearest difficulty-blind probe per occupied niche sufficient for GP0 calibration?\n\n"
        "Q3.\nIf not all K1000 niches are reached in Cycle 0, should the algorithm:\n\nA. accept sparse initial coverage and let evolution expand naturally\n\nor\n\nB. add an explicit pre-GP exploration stage?\n",
        encoding="utf-8",
    )
    _figures(root_counts, child_counts, operator_frame, child_qd)
    (AUDIT / "report.md").write_text(_report(coverage, migration, operator_frame, probe_summary, validation), encoding="utf-8")
    result = {
        "uniform_roots": ROOT_COUNT,
        "nominal_offspring": len(offspring),
        "valid_unique_offspring": len(child_qd),
        "root_occupied_niches": len(root_niches),
        "offspring_occupied_niches": len(child_niches),
        "union_occupied_niches": len(root_niches | child_niches),
        "new_niches_beyond_roots": len(child_niches - root_niches),
        "offspring_only_probe_count": len(child_probe),
        "union_probe_count": len(union_probe),
    }
    print("STRUCTURAL_BOOTSTRAP_AUDIT_OUTPUTS_READY")
    for key, value in result.items():
        print(f"{key.upper()}={value}")
    return result

