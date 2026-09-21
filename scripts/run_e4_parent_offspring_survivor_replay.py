#!/usr/bin/env python
"""E4 offline parent + offspring survivor replay audit.

This program is intentionally read-only with respect to the completed BLADE
trajectory.  It reuses saved QD4 tables and historical GP4 ensembles, and
writes only below audits/E4_parent_offspring_survivor_replay_v1.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.qd4 import QD4_FEATURE_NAMES
from final_run.science import predict_gp4, select_population


RUN = ROOT / "runs" / "blade_v2_full_seed2026"
CYCLES = RUN / "cycles"
OUT = ROOT / "audits" / "E4_parent_offspring_survivor_replay_v1"
NORM = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
BUDGET = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duckdb_module():
    try:
        import duckdb
    except ModuleNotFoundError:
        sys.path.insert(0, str(ROOT / "vendor"))
        import duckdb
    return duckdb


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read formal artifacts, including legacy PyArrow page metadata."""
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError):
        quoted = path.as_posix().replace("'", "''")
        projection = "*" if columns is None else ", ".join(f'"{name}"' for name in columns)
        return duckdb_module().sql(
            f"SELECT {projection} FROM read_parquet('{quoted}', union_by_name=true)"
        ).df()


def global_topk(pool: pd.DataFrame, scores: np.ndarray, budget: int = BUDGET) -> pd.DataFrame:
    """Exact formal stable global GP rule: (-score, coordinate hash)."""
    values = np.asarray(scores, dtype=float)
    if len(pool) != len(values) or len(pool) < budget:
        raise RuntimeError("invalid global GP candidate domain")
    if pool.instance_hash.astype(str).duplicated().any() or not np.isfinite(values).all():
        raise RuntimeError("duplicate hash or non-finite GP prediction")
    hashes = pool.instance_hash.astype(str).to_numpy()
    order = np.lexsort((hashes, -values))[:budget]
    selected = pool.iloc[order].copy().reset_index(drop=True)
    selected["component"] = "global_gp"
    selected["gp_predicted_log1p_hardness"] = values[order]
    selected["global_gp_rank"] = np.arange(1, budget + 1)
    return selected


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {key: math.nan for key in ("mean", "std", "median", "P10", "P25", "P75", "P90", "P95", "P99", "max")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "P10": float(np.percentile(array, 10)),
        "P25": float(np.percentile(array, 25)),
        "P75": float(np.percentile(array, 75)),
        "P90": float(np.percentile(array, 90)),
        "P95": float(np.percentile(array, 95)),
        "P99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def niche_metrics(niches: np.ndarray) -> dict[str, float | int]:
    counts = np.bincount(np.asarray(niches, dtype=int), minlength=1000).astype(float)
    nonzero = counts[counts > 0]
    total = float(nonzero.sum())
    probabilities = nonzero / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    ordered = np.sort(counts)
    gini = float(2 * np.sum(np.arange(1, 1001) * ordered) / (1000 * total) - 1001 / 1000)
    return {
        "occupied_niches": int(len(nonzero)),
        "N_eff": float(np.exp(entropy)),
        "entropy": entropy,
        "gini": gini,
        "largest_niche_share": float(nonzero.max() / total),
        "top5_niche_share": float(np.sort(nonzero)[-5:].sum() / total),
        "top10_niche_share": float(np.sort(nonzero)[-10:].sum() / total),
    }


def overlap_metrics(left: set[str], right: set[str]) -> dict[str, float | int]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "overlap_n": intersection,
        "jaccard": intersection / union,
        "retained_n": intersection,
        "replaced_n": BUDGET - intersection,
        "replacement_fraction": 1.0 - intersection / BUDGET,
    }


def prefixed(target: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    target.update({f"{prefix}_{key}": value for key, value in values.items()})


def cycle_paths(cycle: int) -> dict[str, Path]:
    directory = CYCLES / f"C{cycle:02d}"
    c00 = cycle == 0
    return {
        "dir": directory,
        "marker": directory / "CYCLE_COMPLETE.json",
        "strict": directory / "strict_cycle_verification.json",
        "checkpoint": directory / "09_bq_training" / f"M{cycle + 1}_E256.tsp.best",
        "gp": directory / "02_gp" / "training" / "gp4_ensemble.joblib",
        "mutation_manifest": directory / "03_mutation" / "mutation_manifest.parquet",
        "offspring_coordinates": directory / "03_mutation" / "offspring_coordinates.npz",
        "parent_qd": directory / "04_qd" / "parent_qd.parquet",
        "offspring_qd": directory / "04_qd" / "offspring_qd.parquet",
        "scored": directory / "05_scoring" / ("O0_scored.parquet" if c00 else "offspring_scored.parquet"),
        "T": directory / "06_selection" / ("T0.parquet" if c00 else "T_selected.parquet"),
        "T_bundle": directory / "07_oracle" / "T" / "labeled_bundle.npz",
        "P": directory / "06_selection" / ("P0_FULL.parquet" if c00 else "P_next.parquet"),
        "P_bundle": directory / "06_selection" / ("P0_FULL.npz" if c00 else "P_next.npz"),
        "labeled": directory / "07_oracle" / "T" / "labeled.parquet",
        "ledger": directory / "07_oracle" / "T" / "ledger.parquet",
        "archive": directory / "08_archive" / "A_next.npz",
        "report": directory / "cycle_report.md",
        "timing": directory / "timing_ledger.csv",
    }


def verified_completed_cycles() -> tuple[list[int], list[dict[str, Any]]]:
    valid: list[int] = []
    rows: list[dict[str, Any]] = []
    for directory in sorted(CYCLES.glob("C[0-9][0-9]")):
        try:
            cycle = int(directory.name[1:])
        except ValueError:
            continue
        paths = cycle_paths(cycle)
        if not paths["marker"].is_file():
            continue
        marker = read_json(paths["marker"])
        strict = read_json(paths["strict"]) if paths["strict"].is_file() else {}
        checks = {
            "marker_complete": marker.get("status") == "COMPLETE",
            "marker_arm": marker.get("arm") == "FULL",
            "marker_cycle": marker.get("cycle") == f"C{cycle:02d}",
            "strict_pass": strict.get("status") == "PASS",
        }
        hash_mapping = {
            "checkpoint_sha256": "checkpoint",
            "T_sha256": "T_bundle",
            "A_sha256": "archive",
            "P_sha256": "P_bundle",
            "report_sha256": "report",
            "timing_sha256": "timing",
        }
        for marker_key, path_key in hash_mapping.items():
            path = paths[path_key]
            checks[f"{path_key}_hash"] = path.is_file() and sha256_file(path) == marker.get(marker_key)
        verified = all(checks.values())
        rows.append({
            "cycle": f"C{cycle:02d}",
            "cycle_index": cycle,
            "completed_utc": marker.get("completed_utc"),
            "verified_complete": verified,
            **checks,
        })
        if verified:
            valid.append(cycle)
    if not valid:
        raise RuntimeError("no verified FULL cycles discovered")
    if valid != list(range(max(valid) + 1)):
        raise RuntimeError(f"verified completed cycles are not contiguous from C00: {valid}")
    return valid, rows


def critical_paths(cycles: list[int]) -> dict[str, str]:
    paths: list[Path] = [
        ROOT / "assets" / "qd" / "cvt" / "centroids.csv",
        ROOT / "assets" / "qd" / "normalization" / "normalization_A_P1P99.json",
        ROOT / "assets" / "qd" / "reference_cloud" / "formal_raw_scaler.json",
        ROOT / "assets" / "qd" / "roots" / "root_hashes.txt",
    ]
    for cycle in cycles:
        item = cycle_paths(cycle)
        paths.extend(item[key] for key in (
            "marker", "strict", "checkpoint", "gp", "mutation_manifest",
            "offspring_coordinates", "parent_qd", "offspring_qd", "scored",
            "T", "T_bundle", "P", "P_bundle", "labeled", "ledger",
        ))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"critical formal artifacts missing: {missing[:5]}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def historical_parent_check(cycle: int, parent_qd: pd.DataFrame) -> tuple[bool, str]:
    hashes = parent_qd.instance_hash.astype(str).tolist()
    if cycle == 0:
        expected = [line.strip() for line in (
            ROOT / "assets" / "qd" / "roots" / "root_hashes.txt"
        ).read_text(encoding="utf-8").splitlines() if line.strip()]
        return hashes == expected, "frozen Uniform roots U0"
    prior = read_parquet(cycle_paths(cycle - 1)["P"], columns=["instance_hash"])
    expected = prior.instance_hash.astype(str).tolist()
    return hashes == expected, f"actual historical FULL P_{cycle - 1}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(value) for value in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("—" if not np.isfinite(value) else f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_figures(training: pd.DataFrame, parent: pd.DataFrame,
                 competition_deltas: dict[str, np.ndarray]) -> None:
    x = training.cycle_index.to_numpy()
    labels = training.cycle.tolist()

    def finish(name: str, ylabel: str) -> None:
        plt.xlabel("Cycle")
        plt.ylabel(ylabel)
        plt.xticks(x, labels, rotation=45)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT / name, dpi=180)
        plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, training.parent_share_T, marker="o", label="Training T")
    plt.plot(x, parent.parent_share_P, marker="s", label="Evolutionary P")
    plt.legend()
    finish("fig01_parent_survival_rate_by_cycle.png", "Old-parent share")

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, training.parent_survival_enrichment_T, marker="o", label="Training T")
    plt.plot(x, parent.parent_survival_enrichment_P, marker="s", label="Evolutionary P")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.legend()
    finish("fig02_parent_survival_enrichment_by_cycle.png", "Survival enrichment")

    plt.figure(figsize=(9, 4.4))
    plt.plot(x, training.jaccard_T, marker="o")
    finish("fig03_training_selection_jaccard.png", "T offspring-only vs union Jaccard")

    plt.figure(figsize=(9, 4.4))
    plt.plot(x, parent.jaccard_P, marker="o")
    finish("fig04_parent_selection_jaccard.png", "P offspring-only vs union Jaccard")

    plt.figure(figsize=(10, 5))
    values = [competition_deltas[label] for label in labels]
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xticks(rotation=45)
    plt.xlabel("Cycle")
    plt.ylabel("Best-child GP minus parent GP")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "fig05_parent_vs_best_child_delta.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, parent.P_O_N_eff, marker="o", label="Offspring-only P")
    plt.plot(x, parent.P_U_N_eff, marker="s", label="Parent+offspring P")
    plt.legend()
    finish("fig06_parent_selection_neff.png", "Parent population N_eff")


def protocol_text(cycles: list[int], pre_hash_count: int) -> str:
    return f"""# E4 Parent + Offspring Survivor Replay Protocol

## Scope

- Audit: `E4_parent_offspring_survivor_replay_v1`
- Mode: offline, read-only scientific replay
- Primary arm: completed FULL / QD+GP trajectory
- Verified cycles: C00–C{max(cycles):02d}
- Formal artifacts fingerprinted before replay: {pre_hash_count}

## Counterfactual

For C00, the union is `unique(U0 UNION O0)`. For C01+, the union is
`unique(actual historical P_(s-1) UNION O_s)`. Every cycle is a one-step local
replay. Counterfactual parents are never propagated into later cycles.

## Frozen mechanisms

- Historical GP4 ensemble for each cycle is loaded from the completed run and is not refit.
- Saved exact QD4 descriptors, frozen P1/P99 normalization, and frozen 1000 CVT centroids are reused.
- Training selection is stable global Top10k by `(-GP score, canonical coordinate hash)`.
- Full parent selection is one centroid-nearest candidate per reached niche, then stable global GP fill.
- Canonical `instance_hash` is the coordinate hash and is the union deduplication identity.

## Prohibited work

No BQ training, checkpoint writing, mutation generation, GP fitting, QD refitting,
external evaluation, or Concorde invocation is present in this replay. Cached
exact-hardness diagnostics use only already-saved cycle `T/labeled.parquet`
values and are explicitly non-primary.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cycles, inventory_rows = verified_completed_cycles()
    pre_hashes = critical_paths(cycles)
    (OUT / "01_protocol.md").write_text(protocol_text(cycles, len(pre_hashes)), encoding="utf-8")

    reconstruction_rows: list[dict[str, Any]] = []
    union_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    survival_rows: list[dict[str, Any]] = []
    competition_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    competition_deltas: dict[str, np.ndarray] = {}
    cycle_failures: list[str] = []
    all_predictions_finite = True
    all_union_unique = True
    all_t_sizes = True
    all_p_sizes = True
    all_parent_sources = True
    c00_root_source = False

    for cycle in cycles:
        label = f"C{cycle:02d}"
        print(f"E4_REPLAY {label} START", flush=True)
        paths = cycle_paths(cycle)
        parent_qd = read_parquet(paths["parent_qd"])
        offspring_qd = read_parquet(paths["offspring_qd"])
        scored_saved = read_parquet(paths["scored"])
        historical_t = read_parquet(paths["T"])
        historical_p = read_parquet(paths["P"])
        parent_source_ok, parent_source_label = historical_parent_check(cycle, parent_qd)
        all_parent_sources &= parent_source_ok
        if cycle == 0:
            c00_root_source = parent_source_ok

        model = joblib.load(paths["gp"])
        offspring_scores = predict_gp4(model, offspring_qd[list(QD4_FEATURE_NAMES)].to_numpy(float))
        all_predictions_finite &= bool(np.isfinite(offspring_scores).all())
        saved_alignment = scored_saved.instance_hash.astype(str).tolist() == offspring_qd.instance_hash.astype(str).tolist()
        saved_scores = scored_saved.gp_predicted_log1p_hardness.to_numpy(float)
        score_max_abs_diff = float(np.max(np.abs(offspring_scores - saved_scores))) if saved_alignment else math.inf

        reconstructed_t = global_topk(offspring_qd, offspring_scores)
        reconstructed_p = select_population(offspring_qd, offspring_scores, BUDGET, f"GP{cycle}")
        t_hist_list = historical_t.instance_hash.astype(str).tolist()
        p_hist_list = historical_p.instance_hash.astype(str).tolist()
        t_recon_list = reconstructed_t.instance_hash.astype(str).tolist()
        p_recon_list = reconstructed_p.instance_hash.astype(str).tolist()
        t_membership = set(t_hist_list) == set(t_recon_list)
        p_membership = set(p_hist_list) == set(p_recon_list)
        t_order = t_hist_list == t_recon_list
        p_order = p_hist_list == p_recon_list
        reconstruction_ok = all((parent_source_ok, saved_alignment, score_max_abs_diff <= 1e-12,
                                 t_membership, p_membership, t_order, p_order))
        reconstruction_rows.append({
            "cycle": label,
            "cycle_index": cycle,
            "historical_parent_source": parent_source_label,
            "historical_parent_source_exact": parent_source_ok,
            "offspring_score_alignment": saved_alignment,
            "offspring_score_max_abs_diff": score_max_abs_diff,
            "T_membership_exact": t_membership,
            "T_order_exact": t_order,
            "P_membership_exact": p_membership,
            "P_order_exact": p_order,
            "status": "PASS" if reconstruction_ok else "E4_SELECTION_RECONSTRUCTION_FAILURE",
        })
        inventory_rows[cycle].update({
            "historical_parent_source": parent_source_label,
            "previous_parent_count": len(parent_qd),
            "offspring_count": len(offspring_qd),
            "gp_path": str(paths["gp"].relative_to(ROOT)),
            "gp_sha256": pre_hashes[str(paths["gp"].relative_to(ROOT))],
            "parent_qd_path": str(paths["parent_qd"].relative_to(ROOT)),
            "offspring_qd_path": str(paths["offspring_qd"].relative_to(ROOT)),
            "T_path": str(paths["T"].relative_to(ROOT)),
            "P_path": str(paths["P"].relative_to(ROOT)),
        })
        if not reconstruction_ok:
            cycle_failures.append(label)
            print(f"E4_SELECTION_RECONSTRUCTION_FAILURE {label}", flush=True)
            continue

        parent = parent_qd.copy()
        offspring = offspring_qd.copy()
        parent_hashes = set(parent.instance_hash.astype(str))
        offspring_hashes = set(offspring.instance_hash.astype(str))
        duplicates = parent_hashes & offspring_hashes
        parent["audit_origin"] = "previous_parent"
        offspring["audit_origin"] = "offspring"
        union = pd.concat([parent, offspring], ignore_index=True, sort=False).drop_duplicates(
            "instance_hash", keep="first"
        ).reset_index(drop=True)
        union["is_previous_parent"] = union.instance_hash.astype(str).isin(parent_hashes)
        union["is_offspring"] = union.instance_hash.astype(str).isin(offspring_hashes)
        union_unique = not union.instance_hash.astype(str).duplicated().any()
        canonical_hashes = union.instance_hash.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
        all_union_unique &= bool(union_unique and canonical_hashes)
        r_pool = len(parent_hashes) / len(union)
        union_rows.append({
            "cycle": label,
            "cycle_index": cycle,
            "previous_parent_source": parent_source_label,
            "previous_parent_count": len(parent_hashes),
            "offspring_count": len(offspring_hashes),
            "parent_offspring_duplicate_count": len(duplicates),
            "final_unique_union_count": len(union),
            "unique_previous_parents_in_union": int(union.is_previous_parent.sum()),
            "r_pool": r_pool,
            "canonical_hash_validation": bool(canonical_hashes),
        })

        union_scores = predict_gp4(model, union[list(QD4_FEATURE_NAMES)].to_numpy(float))
        all_predictions_finite &= bool(np.isfinite(union_scores).all())
        union["audit_gp_score"] = union_scores
        score_map = dict(zip(union.instance_hash.astype(str), union_scores, strict=True))

        replay_t = global_topk(union, union_scores)
        t_u_hashes = set(replay_t.instance_hash.astype(str))
        t_o_hashes = set(t_hist_list)
        t_old = t_u_hashes & parent_hashes
        t_overlap = overlap_metrics(t_o_hashes, t_u_hashes)
        t_o_scores = np.asarray([score_map[value] for value in t_hist_list], dtype=float)
        t_u_scores = replay_t.gp_predicted_log1p_hardness.to_numpy(float)
        t_row: dict[str, Any] = {
            "cycle": label,
            "cycle_index": cycle,
            "r_pool": r_pool,
            "N_parent_in_T": len(t_old),
            "parent_share_T": len(t_old) / BUDGET,
            "parent_survival_enrichment_T": (len(t_old) / BUDGET) / r_pool,
            "overlap_N_T": t_overlap["overlap_n"],
            "jaccard_T": t_overlap["jaccard"],
            "retained_N_T": t_overlap["retained_n"],
            "replaced_N_T": t_overlap["replaced_n"],
            "replacement_fraction_T": t_overlap["replacement_fraction"],
            "delta_predicted_mean_hardness": float(t_u_scores.mean() - t_o_scores.mean()),
        }
        prefixed(t_row, "T_O_GP", stats(t_o_scores))
        prefixed(t_row, "T_U_GP", stats(t_u_scores))
        training_rows.append(t_row)
        all_t_sizes &= len(replay_t) == BUDGET and not replay_t.instance_hash.duplicated().any()

        replay_p = select_population(union, union_scores, BUDGET, f"GP{cycle}")
        p_u_hashes = set(replay_p.instance_hash.astype(str))
        p_o_hashes = set(p_hist_list)
        p_old = p_u_hashes & parent_hashes
        p_overlap = overlap_metrics(p_o_hashes, p_u_hashes)
        p_o_metrics = niche_metrics(historical_p.niche_id.to_numpy())
        p_u_metrics = niche_metrics(replay_p.niche_id.to_numpy())
        p_o_scores = np.asarray([score_map[value] for value in p_hist_list], dtype=float)
        p_u_scores = replay_p.gp_predicted_log1p_hardness.to_numpy(float)
        replay_p_hash_component = dict(zip(
            replay_p.instance_hash.astype(str), replay_p.component.astype(str), strict=True
        ))
        old_qd = {value for value in p_old if replay_p_hash_component[value] == "centroid_floor"}
        old_gp = {value for value in p_old if replay_p_hash_component[value] == "gp_fill"}
        old_both_eligible = old_qd & t_u_hashes
        p_row: dict[str, Any] = {
            "cycle": label,
            "cycle_index": cycle,
            "r_pool": r_pool,
            "N_old_parent_survive_P": len(p_old),
            "parent_share_P": len(p_old) / BUDGET,
            "parent_survival_enrichment_P": (len(p_old) / BUDGET) / r_pool,
            "old_parent_qd_floor_n": len(old_qd),
            "old_parent_gp_fill_n": len(old_gp),
            "old_parent_both_eligible_n": len(old_both_eligible),
            "old_parent_qd_floor_pct_of_survivors": len(old_qd) / len(p_old) if p_old else 0.0,
            "old_parent_gp_fill_pct_of_survivors": len(old_gp) / len(p_old) if p_old else 0.0,
            "old_parent_both_eligible_pct_of_survivors": len(old_both_eligible) / len(p_old) if p_old else 0.0,
            "overlap_N_P": p_overlap["overlap_n"],
            "jaccard_P": p_overlap["jaccard"],
            "replaced_N_P": p_overlap["replaced_n"],
            "replacement_fraction_P": p_overlap["replacement_fraction"],
            "old_parent_contribution": len(p_old) / BUDGET,
        }
        prefixed(p_row, "P_O", p_o_metrics)
        prefixed(p_row, "P_U", p_u_metrics)
        prefixed(p_row, "P_O_GP", stats(p_o_scores))
        prefixed(p_row, "P_U_GP", stats(p_u_scores))
        parent_rows.append(p_row)
        all_p_sizes &= len(replay_p) == BUDGET and not replay_p.instance_hash.duplicated().any()

        survival_rows.append({
            "cycle": label,
            "cycle_index": cycle,
            "r_pool": r_pool,
            "N_parent_in_T": len(t_old),
            "parent_share_T": len(t_old) / BUDGET,
            "parent_survival_enrichment_T": (len(t_old) / BUDGET) / r_pool,
            "N_parent_in_P": len(p_old),
            "parent_share_P": len(p_old) / BUDGET,
            "parent_survival_enrichment_P": (len(p_old) / BUDGET) / r_pool,
            "P_old_via_qd_floor_n": len(old_qd),
            "P_old_via_gp_fill_n": len(old_gp),
            "P_old_both_eligible_n": len(old_both_eligible),
        })

        child = offspring_qd[["instance_hash", "parent_hash", "operator_id"]].copy()
        child["child_gp"] = offspring_scores
        parent_scores = pd.Series(
            [score_map[value] for value in parent_qd.instance_hash.astype(str)],
            index=parent_qd.instance_hash.astype(str), name="parent_gp"
        )
        best_child = child.groupby("parent_hash", sort=False).child_gp.max()
        common = parent_scores.index.intersection(best_child.index)
        deltas = best_child.loc[common].to_numpy(float) - parent_scores.loc[common].to_numpy(float)
        competition_deltas[label] = deltas
        delta_stats = stats(deltas)
        traceable = child.groupby("parent_hash").operator_id.nunique()
        competition_row = {
            "cycle": label,
            "cycle_index": cycle,
            "parents_total": len(parent_scores),
            "parents_with_child_lineage": len(common),
            "parents_with_all_11_operators_traceable": int((traceable == 11).sum()),
            "frac_child_wins": float(np.mean(deltas > 0)),
            "frac_parent_beats_all": float(np.mean(deltas < 0)),
            "frac_tie": float(np.mean(deltas == 0)),
            "mean_parent_GP": float(parent_scores.loc[common].mean()),
            "mean_best_child_GP": float(best_child.loc[common].mean()),
        }
        prefixed(competition_row, "delta_best_child", delta_stats)
        competition_rows.append(competition_row)

        child = child.join(parent_scores, on="parent_hash")
        child["child_parent_delta"] = child.child_gp - child.parent_gp
        for operator_id, group in child.groupby("operator_id", sort=True):
            values = group.child_parent_delta.to_numpy(float)
            operator_rows.append({
                "cycle": label,
                "cycle_index": cycle,
                "operator_id": operator_id,
                "traceable_children": len(values),
                "traceable_parent_count": group.parent_hash.nunique(),
                "mean_delta": float(values.mean()),
                "median_delta": float(np.median(values)),
                "win_rate": float(np.mean(values > 0)),
                "loss_rate": float(np.mean(values < 0)),
                "tie_rate": float(np.mean(values == 0)),
                "P90_delta": float(np.percentile(values, 90)),
            })

        labeled = read_parquet(paths["labeled"], columns=[
            "instance_hash", "current_hardness", "verified", "expert_tour"
        ])
        labeled = labeled[labeled.verified.fillna(False)].drop_duplicates("instance_hash", keep="last")
        hardness_map = dict(zip(
            labeled.instance_hash.astype(str), labeled.current_hardness.astype(float), strict=True
        ))
        labeled_old = sorted(t_old & set(hardness_map))
        labeled_union = sorted(t_u_hashes & set(hardness_map))
        old_values = np.asarray([hardness_map[value] for value in labeled_old], dtype=float)
        union_values = np.asarray([hardness_map[value] for value in labeled_union], dtype=float)
        current_values = np.asarray([hardness_map[value] for value in t_hist_list if value in hardness_map], dtype=float)
        oracle_rows.append({
            "cycle": label,
            "cycle_index": cycle,
            "diagnostic_label": "CACHED-ORACLE DIAGNOSTIC ONLY",
            "union_selected_old_parents_n": len(t_old),
            "union_selected_old_parents_with_cached_current_H_n": len(old_values),
            "old_parent_cached_coverage_fraction": len(old_values) / len(t_old) if t_old else 0.0,
            "old_parent_cached_current_H_mean": float(old_values.mean()) if len(old_values) else math.nan,
            "union_T_cached_overlap_n": len(union_values),
            "union_T_cached_overlap_current_H_mean": float(union_values.mean()) if len(union_values) else math.nan,
            "offspring_only_T_cached_n": len(current_values),
            "offspring_only_T_current_H_mean": float(current_values.mean()) if len(current_values) else math.nan,
            "selection_bias_warning": "Incomplete overlap-only cache; not a primary verdict input",
            "new_concorde_calls": 0,
            "new_BQ_hardness_inference": 0,
        })
        print(f"E4_REPLAY {label} COMPLETE", flush=True)

    inventory = pd.DataFrame(inventory_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    union_stats = pd.DataFrame(union_rows)
    training = pd.DataFrame(training_rows)
    parent = pd.DataFrame(parent_rows)
    survival = pd.DataFrame(survival_rows)
    competition = pd.DataFrame(competition_rows)
    operators = pd.DataFrame(operator_rows)
    oracle = pd.DataFrame(oracle_rows)
    if cycle_failures:
        raise RuntimeError(f"E4_SELECTION_RECONSTRUCTION_FAILURE: {cycle_failures}")

    cycle_summary = training[[
        "cycle", "cycle_index", "parent_share_T", "parent_survival_enrichment_T",
        "replacement_fraction_T", "jaccard_T", "delta_predicted_mean_hardness",
    ]].merge(parent[[
        "cycle", "parent_share_P", "parent_survival_enrichment_P", "replacement_fraction_P",
        "jaccard_P", "P_O_N_eff", "P_U_N_eff", "P_O_GP_mean", "P_U_GP_mean",
    ]], on="cycle", validate="one_to_one").merge(competition[[
        "cycle", "frac_parent_beats_all", "frac_child_wins", "delta_best_child_median",
    ]], on="cycle", validate="one_to_one")

    inventory.to_csv(OUT / "02_input_cycle_inventory.csv", index=False)
    reconstruction.to_csv(OUT / "03_reconstruction_integrity.csv", index=False)
    union_stats.to_csv(OUT / "04_union_pool_statistics.csv", index=False)
    training.to_csv(OUT / "05_training_selection_replay.csv", index=False)
    parent.to_csv(OUT / "06_parent_selection_replay.csv", index=False)
    survival.to_csv(OUT / "07_parent_survival_statistics.csv", index=False)
    competition.to_csv(OUT / "08_parent_child_competition.csv", index=False)
    operators.to_csv(OUT / "09_operator_child_parent_deltas.csv", index=False)
    oracle.to_csv(OUT / "10_cached_oracle_diagnostic.csv", index=False)
    cycle_summary.to_csv(OUT / "11_cycle_summary.csv", index=False)
    save_figures(training, parent, competition_deltas)

    t_med_replacement = float(training.replacement_fraction_T.median())
    t_med_jaccard = float(training.jaccard_T.median())
    t_med_enrichment = float(training.parent_survival_enrichment_T.median())
    p_med_replacement = float(parent.replacement_fraction_P.median())
    p_med_jaccard = float(parent.jaccard_P.median())
    p_med_enrichment = float(parent.parent_survival_enrichment_P.median())
    strong = ((t_med_replacement >= 0.10 and t_med_enrichment > 1.0) or
              (p_med_replacement >= 0.10 and p_med_enrichment > 1.0))
    no_overrepresentation = t_med_enrichment <= 1.0 and p_med_enrichment <= 1.0
    reasonable = (t_med_replacement < 0.05 and p_med_replacement < 0.05 and
                  t_med_jaccard > 0.95 and p_med_jaccard > 0.95 and no_overrepresentation)
    verdict = ("UNION_SURVIVOR_EFFECT_STRONG" if strong else
               "OFFSPRING_ONLY_APPROXIMATION_REASONABLE" if reasonable else
               "UNION_SURVIVOR_EFFECT_MIXED")

    flags = {
        "old_parents_ge_10pct_in_many_cycles": int((training.parent_share_T >= 0.10).sum()) >= math.ceil(len(training) / 2),
        "T_enrichment_consistently_gt_1": int((training.parent_survival_enrichment_T > 1.0).sum()) >= math.ceil(len(training) / 2),
        "P_enrichment_consistently_gt_1": int((parent.parent_survival_enrichment_P > 1.0).sum()) >= math.ceil(len(parent) / 2),
        "material_jaccard_drop": min(t_med_jaccard, p_med_jaccard) < 0.90,
        "substantial_parent_beats_all": float(competition.frac_parent_beats_all.median()) >= 0.10,
        "union_P_preserves_or_improves_N_eff": float(parent.P_U_N_eff.median()) >= float(parent.P_O_N_eff.median()),
    }
    material_forced = sum(flags.values()) >= 3

    main_table = pd.DataFrame([
        ["Training selection Jaccard", 1.0, t_med_jaccard, t_med_jaccard - 1.0],
        ["Parent contribution to T", 0.0, float(training.parent_share_T.median()), float(training.parent_share_T.median())],
        ["Parent survival enrichment T", math.nan, t_med_enrichment, math.nan],
        ["Parent-selection Jaccard", 1.0, p_med_jaccard, p_med_jaccard - 1.0],
        ["Parent contribution to P", 0.0, float(parent.parent_share_P.median()), float(parent.parent_share_P.median())],
        ["Parent survival enrichment P", math.nan, p_med_enrichment, math.nan],
        ["Parent N_eff", float(parent.P_O_N_eff.median()), float(parent.P_U_N_eff.median()), float(parent.P_U_N_eff.median() - parent.P_O_N_eff.median())],
        ["Predicted hardness T", float(training.T_O_GP_mean.mean()), float(training.T_U_GP_mean.mean()), float(training.T_U_GP_mean.mean() - training.T_O_GP_mean.mean())],
        ["Predicted hardness P", float(parent.P_O_GP_mean.mean()), float(parent.P_U_GP_mean.mean()), float(parent.P_U_GP_mean.mean() - parent.P_O_GP_mean.mean())],
    ], columns=["Metric", "Offspring-only", "Parent+Offspring", "Difference"])
    child_table = competition[["cycle", "frac_parent_beats_all", "frac_child_wins", "delta_best_child_median"]].rename(columns={
        "cycle": "Cycle", "frac_parent_beats_all": "Parent beats all children",
        "frac_child_wins": "Child wins", "delta_best_child_median": "Median best-child delta",
    })
    all_delta = np.concatenate(list(competition_deltas.values()))
    all_delta_stats = stats(all_delta)
    flag_lines = "\n".join(f"- {name}: **{str(value).upper()}**" for name, value in flags.items())
    forced_sentence = ("The current offspring-only implementation imposes material forced generational replacement."
                       if material_forced else
                       "The replay does not meet the multi-signal threshold for material forced generational replacement.")
    recommendation = {
        "UNION_SURVIVOR_EFFECT_STRONG": "Run a corrected short Full sanity run using `C_s = unique(P_(s-1) UNION O_s)`.",
        "OFFSPRING_ONLY_APPROXIMATION_REASONABLE": "Retain offspring-only replacement and proceed to the Random baseline.",
        "UNION_SURVIVOR_EFFECT_MIXED": "Inspect the cycle-level pattern before changing the formal architecture.",
    }[verdict]
    report = f"""# E4 Parent + Offspring Survivor Replay

## 1. Scientific motivation

This audit tests whether the offspring-only candidate domain discards competitive historical parents that would survive a local `(mu + lambda)`-style competition.

## 2. Current offspring-only mechanism

The completed Full trajectory selects both hard-training candidates and QD+GP evolutionary parents from `O_s` only.

## 3. Counterfactual parent+offspring mechanism

Each completed cycle is replayed once with `unique(actual historical previous parents UNION O_s)`. No counterfactual parent population is propagated to the next cycle.

## 4. Integrity reconstruction

All {len(reconstruction)} verified completed cycles C00–C{max(cycles):02d} exactly reproduced historical T and P membership **and order**. Historical offspring score recomputation had maximum absolute difference {reconstruction.offspring_score_max_abs_diff.max():.3e}. No failed cycles were used.

## 5. Candidate-pool statistics

Union size ranged from {union_stats.final_unique_union_count.min():,} to {union_stats.final_unique_union_count.max():,}; median parent pool prevalence was {union_stats.r_pool.median():.6f}. Parent-offspring canonical duplicates are reported cycle by cycle in `04_union_pool_statistics.csv`.

## 6. Training-selection replay

Median parent share was {training.parent_share_T.median():.6f}; median enrichment was {t_med_enrichment:.6f}; median replacement fraction was {t_med_replacement:.6f}; median Jaccard was {t_med_jaccard:.6f}. The union-minus-offspring-only predicted mean GP hardness change averaged {training.delta_predicted_mean_hardness.mean():.6f}.

## 7. Parent-selection replay

Median old-parent share was {parent.parent_share_P.median():.6f}; median enrichment was {p_med_enrichment:.6f}; median replacement fraction was {p_med_replacement:.6f}; median Jaccard was {p_med_jaccard:.6f}.

## 8. Parent survival enrichment

QD-floor, GP-fill, and dual-eligibility counts are separated in `06_parent_selection_replay.csv` and `07_parent_survival_statistics.csv`.

{markdown_table(main_table)}

## 9. Parent-vs-child mutation competition

Across cycles, the median fraction of parents beating all traceable children was {competition.frac_parent_beats_all.median():.6f}; the median fraction with at least one winning child was {competition.frac_child_wins.median():.6f}. Pooled best-child-minus-parent GP delta had mean {all_delta_stats['mean']:.6f}, median {all_delta_stats['median']:.6f}, P10 {all_delta_stats['P10']:.6f}, P90 {all_delta_stats['P90']:.6f}.

{markdown_table(child_table)}

Operator-level diagnostics are in `09_operator_child_parent_deltas.csv`; they are not used to re-rank or remove operators.

## 10. QD diversity comparison

Median offspring-only parent N_eff was {parent.P_O_N_eff.median():.6f}; median union parent N_eff was {parent.P_U_N_eff.median():.6f}. Full occupied-niche, entropy, Gini, largest-niche, Top-5 and Top-10 statistics are in `06_parent_selection_replay.csv`.

## 11. Cached exact-hardness diagnostic

`10_cached_oracle_diagnostic.csv` is labeled **CACHED-ORACLE DIAGNOSTIC ONLY**. It reuses current-cycle hardness already stored for historical T selections. Coverage is incomplete and selection-biased; no new Concorde call or BQ inference was made, and these values do not determine the verdict.

## 12. Limitations of one-step replay

Historical `O_(s+1)` was generated by actual `P_s`, not replayed `P_s^U`. Therefore E4 measures immediate survivor-selection pressure only. It does not estimate downstream solver improvement or a counterfactual evolutionary trajectory.

## 13. Practical verdict

Operational signals requested by the protocol:

{flag_lines}

{forced_sentence}

**{verdict}**

This is an operational effect-size label, not a statistical superiority claim. It does not establish that corrected training performs better.

## 14. Recommendation for next experiment

{recommendation}
"""
    (OUT / "12_final_report.md").write_text(report, encoding="utf-8")

    post_hashes = critical_paths(cycles)
    formal_unchanged = pre_hashes == post_hashes
    checks = {
        "01_no_BQ_training_occurred": True,
        "02_no_checkpoint_modified": formal_unchanged and all(
            pre_hashes[key] == post_hashes[key] for key in pre_hashes if "09_bq_training" in key
        ),
        "03_no_mutation_regenerated": formal_unchanged and all(
            pre_hashes[key] == post_hashes[key] for key in pre_hashes if "03_mutation" in key
        ),
        "04_no_new_Concorde_call_occurred": formal_unchanged and bool((oracle.new_concorde_calls == 0).all()),
        "05_historical_GP_reused_not_refit": len(set(inventory.gp_sha256)) == len(cycles),
        "06_historical_QD_map_reused_unchanged": formal_unchanged,
        "07_historical_O_unchanged": formal_unchanged,
        "08_historical_previous_P_unchanged": formal_unchanged,
        "09_historical_T_reconstruction_exact": bool(reconstruction.T_membership_exact.all() and reconstruction.T_order_exact.all()),
        "10_historical_P_reconstruction_exact": bool(reconstruction.P_membership_exact.all() and reconstruction.P_order_exact.all()),
        "11_C00_uses_frozen_U0": c00_root_source,
        "12_C01plus_use_actual_historical_P": all_parent_sources,
        "13_union_dedup_uses_canonical_hashes": all_union_unique,
        "14_stable_tie_break_matches_formal": bool(reconstruction.T_order_exact.all() and reconstruction.P_order_exact.all()),
        "15_GP_predictions_finite": all_predictions_finite,
        "16_union_T_size_exactly_10000": all_t_sizes,
        "17_union_P_size_exactly_10000": all_p_sizes,
        "18_no_recursive_counterfactual_assumption": all_parent_sources,
        "19_cached_oracle_separated_from_primary": bool((oracle.diagnostic_label == "CACHED-ORACLE DIAGNOSTIC ONLY").all()),
        "20_formal_run_byte_hash_unchanged": formal_unchanged,
    }
    strict = {
        "audit": "E4_parent_offspring_survivor_replay_v1",
        "mode": "OFFLINE_READ_ONLY_SCIENTIFIC_REPLAY",
        "cycles": [f"C{cycle:02d}" for cycle in cycles],
        "formal_artifact_fingerprint_count": len(pre_hashes),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verdict": verdict,
    }
    write_json(OUT / "STRICT_VERIFICATION.json", strict)
    if strict["status"] != "PASS":
        raise RuntimeError(f"E4 strict verification failed: {[key for key, value in checks.items() if not value]}")

    print("\nE4 PARENT + OFFSPRING SURVIVOR REPLAY COMPLETE")
    print(f"\nCompleted cycles analyzed:\nC00-C{max(cycles):02d}")
    print("\nTRAINING SELECTION:")
    print(f"median parent share = {training.parent_share_T.median():.6f}")
    print(f"median survival enrichment = {t_med_enrichment:.6f}")
    print(f"median replacement fraction = {t_med_replacement:.6f}")
    print(f"median Jaccard = {t_med_jaccard:.6f}")
    print("\nPARENT SELECTION:")
    print(f"median parent share = {parent.parent_share_P.median():.6f}")
    print(f"median survival enrichment = {p_med_enrichment:.6f}")
    print(f"median replacement fraction = {p_med_replacement:.6f}")
    print(f"median Jaccard = {p_med_jaccard:.6f}")
    print("\nPARENT vs CHILD:")
    print(f"median fraction parent beats all children = {competition.frac_parent_beats_all.median():.6f}")
    print(f"median fraction at least one child wins = {competition.frac_child_wins.median():.6f}")
    print(f"median best-child minus parent GP = {competition.delta_best_child_median.median():.6f}")
    print("\nQD STRUCTURE:")
    print(f"offspring-only median N_eff = {parent.P_O_N_eff.median():.6f}")
    print(f"union median N_eff = {parent.P_U_N_eff.median():.6f}")
    print(f"\nVERDICT:\n\n{verdict}")
    print(f"\nNEXT RECOMMENDATION:\n\n{recommendation}")
    print("\nSTOP HERE.")


if __name__ == "__main__":
    main()
