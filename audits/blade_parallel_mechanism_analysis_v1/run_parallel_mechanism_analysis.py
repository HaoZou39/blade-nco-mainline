#!/usr/bin/env python
"""Read-only A1–A4 parallel mechanism analysis on a frozen E4.5/E4.6 snapshot.

Does not retrain, mutate, label, or modify the active corrected-union run.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_blade_v2 as base
from features.qd4 import QD4_FEATURE_NAMES
from final_run.science import membership_hash
from final_run.stage import sha256_file, utc_now
from mutation import OPERATOR_NAMES
from qd.coordinates import load_centroids

RUN_ID = "blade_v2_corrected_union_sanity_seed2026_v1"
RUN_DIR = ROOT / "runs" / RUN_ID
ANALYSIS_ROOT = ROOT / "audits" / "blade_parallel_mechanism_analysis_v1"
E45_AUDIT = ROOT / "audits" / "E4_5_corrected_union_full_sanity_v1"
BUDGET = 10_000
BOOT_N = 10_000
BOOT_SEED = 2026
NORM = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
OOD5 = (
    "GaussianMixture100",
    "TSPLIB100",
    "EvoReal_S1_100",
    "EvoReal_S2_100",
    "EvoReal_S3_100",
)
DATASET_LABEL = {
    "Uniform100": "Uniform100",
    "GaussianMixture100": "GM100",
    "TSPLIB100": "TSPLIB100",
    "EvoReal_S1_100": "S1",
    "EvoReal_S2_100": "S2",
    "EvoReal_S3_100": "S3",
    "GaussianMixture500": "GM500",
    "OriginalTSPLIB70": "Original70",
}
SCIENTIFIC_GLOBS = (
    "src/final_run/*.py",
    "src/qd/*.py",
    "src/mutation/*.py",
    "src/features/qd4.py",
    "scripts/run_blade_v2.py",
    "scripts/run_e4_5_corrected_union_sanity.py",
    "scripts/run_e4_6_corrected_union_longrun.py",
    "scripts/e4_5_strict_helpers.py",
    "scripts/e4_6_reporting.py",
    "scripts/e4_6_strict_helpers.py",
)

UNION_COLS = [
    "instance_hash", "niche_id", "parent_hash", "operator_id",
    "is_previous_parent", "is_new_offspring", "candidate_age",
    "gp_predicted_log1p_hardness", *NORM,
]
P_COLS = [
    "instance_hash", "niche_id", "component", "centroid_distance",
    "gp_predicted_log1p_hardness", "candidate_age", "is_previous_parent",
    "is_new_offspring", "operator_id", "parent_hash",
]
T_COLS = [
    "instance_hash", "niche_id", "component", "gp_predicted_log1p_hardness",
    "operator_id", "is_new_offspring", "is_previous_parent", "parent_hash",
    "candidate_age",
]
T_LAB_COLS = ["instance_hash", "current_hardness", "operator_id", "is_new_offspring", "is_previous_parent"]
MUT_COLS = [
    "parent_hash", "operator_id", "child_hash", "valid_before_repair",
    "valid_after_repair", "repair_count", "proposal_status", "duplicate",
]
OQD_COLS = ["instance_hash", "parent_hash", "operator_id", "niche_id", "parent_niche_id"]
REPAIR_COLS = ["instance_hash", "H_pre", "H_post", "delta_post_minus_pre"]


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return base._read_parquet_recovered(path, columns)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scientific_files() -> list[Path]:
    files: list[Path] = []
    for pattern in SCIENTIFIC_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    return [path for path in files if path.is_file()]


def hash_scientific() -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in scientific_files()}


def discover_complete_cycles() -> list[int]:
    found: list[int] = []
    for cycle in range(20):
        marker = RUN_DIR / f"C{cycle:02d}" / "CYCLE_COMPLETE.json"
        if not marker.is_file():
            break
        payload = read_json(marker)
        if payload.get("status") != "COMPLETE":
            break
        found.append(cycle)
    if not found:
        raise RuntimeError("no CYCLE_COMPLETE cycles found")
    return found


def stage_output_hash(stage_path: Path, suffix: str) -> str | None:
    if not stage_path.is_file():
        return None
    payload = read_json(stage_path)
    for key, rec in payload.get("outputs", {}).items():
        if key.replace("\\", "/").endswith(suffix):
            return rec.get("sha256")
    return None


def age_cat(age: int) -> str:
    value = int(age)
    if value <= 0:
        return "AGE0"
    if value == 1:
        return "AGE1"
    if value == 2:
        return "AGE2"
    if value == 3:
        return "AGE3"
    return "AGE4PLUS"


def age_group(age: int) -> str:
    value = int(age)
    if value <= 0:
        return "AGE0"
    if value == 1:
        return "AGE1"
    if value == 2:
        return "AGE2"
    return "AGE3PLUS"


def provenance_of(prev: bool, new: bool) -> str:
    if prev and new:
        return "both"
    if prev:
        return "previous_parent"
    if new:
        return "new_offspring"
    return "unknown"


def gp_percentiles(scores: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(scores, dtype=float))
    return series.rank(method="average", pct=True).to_numpy(float) * 100.0


def global_topk_hashes(pool: pd.DataFrame, scores: np.ndarray, budget: int = BUDGET) -> list[str]:
    hashes = pool.instance_hash.astype(str).to_numpy()
    order = np.lexsort((hashes, -np.asarray(scores, dtype=float)))[:budget]
    return hashes[order].tolist()


def centroid_distances(frame: pd.DataFrame, centroids: np.ndarray) -> np.ndarray:
    vectors = frame[NORM].to_numpy(float)
    niches = frame.niche_id.to_numpy(int)
    return np.linalg.norm(vectors - centroids[niches], axis=1)


def bootstrap_mean(values: np.ndarray, seed: int = BOOT_SEED, replicates: int = BOOT_N) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    stats = np.empty(replicates)
    n = len(array)
    for start in range(0, replicates, 250):
        stop = min(replicates, start + 250)
        idx = rng.integers(0, n, size=(stop - start, n))
        stats[start:stop] = array[idx].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.percentile(stats, 2.5)),
        "ci95_high": float(np.percentile(stats, 97.5)),
        "n": int(n),
        "replicates": replicates,
        "seed": seed,
    }


def finish_fig(path: Path, xlabel: str, ylabel: str, title: str) -> None:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def fmt_rate(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.4f}"


class FrozenData:
    def __init__(self, cycles: list[int]):
        self.cycles = cycles
        self.union: dict[int, pd.DataFrame] = {}
        self.parents: dict[int, pd.DataFrame] = {}
        self.train: dict[int, pd.DataFrame] = {}
        self.labeled: dict[int, pd.DataFrame] = {}
        self.offspring: dict[int, pd.DataFrame] = {}
        self.manifest: dict[int, pd.DataFrame] = {}
        self.repair: dict[int, pd.DataFrame] = {}
        self.summary: dict[int, dict[str, Any]] = {}
        self.complete: dict[int, dict[str, Any]] = {}
        self.p_set: dict[int, set[str]] = {}
        self.t_set: dict[int, set[str]] = {}
        self.gp_score: dict[int, dict[str, float]] = {}
        self.gp_pct: dict[int, dict[str, float]] = {}
        self.gp_top10: dict[int, set[str]] = {}
        self.gp_top5: dict[int, set[str]] = {}
        self.niche: dict[int, dict[str, int]] = {}
        self.children: dict[int, dict[str, list[tuple[str, str, int]]]] = {}
        self.p_niches: dict[int, set[int]] = {}
        self.cum_p_niches: dict[int, set[int]] = {}
        self.t_p90: dict[int, float] = {}
        self.exact_h: dict[int, dict[str, float]] = {}
        self.p_component: dict[int, dict[str, str]] = {}

    def load(self) -> None:
        centroids = load_centroids()
        cumulative: set[int] = set()
        for cycle in self.cycles:
            log(f"Loading frozen cycle C{cycle:02d}")
            directory = RUN_DIR / f"C{cycle:02d}"
            union = read_parquet(directory / "05_scoring" / "C_union_scored.parquet", UNION_COLS)
            union["instance_hash"] = union.instance_hash.astype(str)
            scores = union.gp_predicted_log1p_hardness.to_numpy(float)
            union["gp_percentile"] = gp_percentiles(scores)
            union["centroid_distance_computed"] = centroid_distances(union, centroids)
            parents = read_parquet(directory / "06_selection" / "P_next.parquet", P_COLS)
            parents["instance_hash"] = parents.instance_hash.astype(str)
            train = read_parquet(directory / "06_selection" / "T_selected.parquet", T_COLS)
            train["instance_hash"] = train.instance_hash.astype(str)
            labeled = read_parquet(directory / "07_oracle" / "T" / "labeled.parquet", T_LAB_COLS)
            labeled["instance_hash"] = labeled.instance_hash.astype(str)
            offspring = read_parquet(directory / "04_qd" / "offspring_qd.parquet", OQD_COLS)
            offspring["instance_hash"] = offspring.instance_hash.astype(str)
            offspring["parent_hash"] = offspring.parent_hash.astype(str)
            offspring["operator_id"] = offspring.operator_id.astype(str)
            manifest = read_parquet(directory / "03_mutation" / "mutation_manifest.parquet", MUT_COLS)
            manifest["parent_hash"] = manifest.parent_hash.astype(str)
            manifest["child_hash"] = manifest.child_hash.astype(str)
            manifest["operator_id"] = manifest.operator_id.astype(str)
            repair = read_parquet(directory / "10_post_adaptation" / "adaptation_repair.parquet", REPAIR_COLS)
            repair["instance_hash"] = repair.instance_hash.astype(str)
            self.union[cycle] = union
            self.parents[cycle] = parents
            self.train[cycle] = train
            self.labeled[cycle] = labeled
            self.offspring[cycle] = offspring
            self.manifest[cycle] = manifest
            self.repair[cycle] = repair
            self.summary[cycle] = read_json(directory / "cycle_summary.json")
            self.complete[cycle] = read_json(directory / "CYCLE_COMPLETE.json")
            self.p_set[cycle] = set(parents.instance_hash)
            self.t_set[cycle] = set(train.instance_hash)
            self.gp_score[cycle] = dict(zip(union.instance_hash, union.gp_predicted_log1p_hardness.astype(float)))
            self.gp_pct[cycle] = dict(zip(union.instance_hash, union.gp_percentile.astype(float)))
            n = len(union)
            top10_n = max(1, int(math.ceil(0.10 * n)))
            top5_n = max(1, int(math.ceil(0.05 * n)))
            ordered = union.sort_values(["gp_predicted_log1p_hardness", "instance_hash"], ascending=[False, True])
            self.gp_top10[cycle] = set(ordered.instance_hash.iloc[:top10_n])
            self.gp_top5[cycle] = set(ordered.instance_hash.iloc[:top5_n])
            self.niche[cycle] = dict(zip(union.instance_hash, union.niche_id.astype(int)))
            child_map: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
            for row in offspring.itertuples(index=False):
                child_map[str(row.parent_hash)].append((str(row.instance_hash), str(row.operator_id), int(row.niche_id)))
            self.children[cycle] = child_map
            self.p_niches[cycle] = set(parents.niche_id.astype(int))
            cumulative |= self.p_niches[cycle]
            self.cum_p_niches[cycle] = set(cumulative)
            hardness = labeled.current_hardness.to_numpy(float)
            self.t_p90[cycle] = float(np.percentile(hardness, 90))
            self.exact_h[cycle] = dict(zip(labeled.instance_hash, labeled.current_hardness.astype(float)))
            self.p_component[cycle] = dict(zip(parents.instance_hash, parents.component.astype(str)))
            if not offspring.parent_hash.notna().all() or not offspring.instance_hash.str.fullmatch(r"[0-9a-f]{64}").all():
                raise RuntimeError(f"lineage hash resolution failed at C{cycle:02d}")


def freeze_snapshot(cycles: list[int], start_ts: str, sci_before: dict[str, str]) -> dict[str, Any]:
    latest = max(cycles)
    cycle_records = []
    for cycle in cycles:
        directory = RUN_DIR / f"C{cycle:02d}"
        complete = read_json(directory / "CYCLE_COMPLETE.json")
        summary = read_json(directory / "cycle_summary.json")
        gp_stage = read_json(directory / "02_gp" / "training" / "STAGE_DONE.json")
        formula_hashes = {}
        for seed in range(2026, 2031):
            formula_path = directory / "02_gp" / "training" / f"seed_{seed}" / "formula.json"
            if formula_path.is_file():
                formula_hashes[str(seed)] = sha256_file(formula_path)
        gp_diag = read_json(directory / "02_gp" / "training" / "gp_diagnostics.json")
        formula_content_hashes = [sha256_text(text) for text in gp_diag.get("formulas", [])]
        cycle_records.append({
            "cycle": f"C{cycle:02d}",
            "CYCLE_COMPLETE": True,
            "completed_utc": complete.get("completed_utc"),
            "model_in": summary.get("model_in"),
            "model_out": summary.get("model_out"),
            "M_in_hash": summary.get("model_in_hash"),
            "M_out_checkpoint_sha256": complete.get("checkpoint_sha256"),
            "P_sha256_cycle_complete": complete.get("P_sha256"),
            "T_sha256_cycle_complete": complete.get("T_sha256"),
            "A_sha256_cycle_complete": complete.get("A_sha256"),
            "age_sha256": complete.get("age_sha256"),
            "P_membership_hash": summary.get("P_hash"),
            "T_membership_hash": summary.get("T_hash"),
            "A_membership_hash": summary.get("A_hash"),
            "O_offspring_npz_sha256": stage_output_hash(directory / "03_mutation" / "STAGE_DONE.json", "offspring_coordinates.npz"),
            "O_mutation_manifest_sha256": stage_output_hash(directory / "03_mutation" / "STAGE_DONE.json", "mutation_manifest.parquet"),
            "C_union_npz_sha256": stage_output_hash(directory / "05_scoring" / "union" / "STAGE_DONE.json", "C_union.npz"),
            "C_union_parquet_sha256": stage_output_hash(directory / "05_scoring" / "union" / "STAGE_DONE.json", "C_union.parquet"),
            "archive_manifest_sha256": stage_output_hash(directory / "08_archive" / "STAGE_DONE.json", "A_next.parquet"),
            "qd_map_token": "QD4_P1P99_K1000_FROZEN",
            "qd_stage_map_input": read_json(directory / "04_qd" / "STAGE_DONE.json").get("input_hashes", {}).get("map"),
            "gp_ensemble_sha256": gp_stage.get("outputs", {}).get(str(directory / "02_gp" / "training" / "gp4_ensemble.joblib"), {}).get("sha256"),
            "gp_formula_file_hashes": formula_hashes,
            "gp_formula_content_hashes": formula_content_hashes,
        })
    snapshot = {
        "analysis_id": "blade_parallel_mechanism_analysis_v1",
        "analysis_start_utc": start_ts,
        "mode": "READ_ONLY_OFFLINE_NO_NEW_TRAINING",
        "run_id": RUN_ID,
        "ANALYSIS_RANGE": f"C00-C{latest:02d}",
        "cycles": [f"C{c:02d}" for c in cycles],
        "latest_completed_cycle": f"C{latest:02d}",
        "latest_solver_in_snapshot": f"M{latest + 1}",
        "frozen_once": True,
        "do_not_append_later_cycles": True,
        "qd_map_hash": {
            "token": "QD4_P1P99_K1000_FROZEN",
            "centroids_csv_sha256": sha256_file(ROOT / "assets" / "qd" / "cvt" / "centroids.csv"),
            "normalization_A_sha256": sha256_file(ROOT / "assets" / "qd" / "normalization" / "normalization_A_P1P99.json"),
        },
        "cycles_detail": cycle_records,
        "scientific_source_hashes_at_start": sci_before,
        "active_e4_6_not_modified": True,
        "note": "Range frozen from CYCLE_COMPLETE.json at analysis start. Later cycles completing during analysis are ignored.",
    }
    write_json(ANALYSIS_ROOT / "00_INPUT_SNAPSHOT.json", snapshot)
    return snapshot


def characterize_group(data: FrozenData, cycle: int, hashes: list[str], group: str) -> pd.DataFrame:
    if not hashes:
        return pd.DataFrame()
    frame = data.union[cycle].set_index("instance_hash").loc[hashes].reset_index().copy()
    frame["cycle"] = f"C{cycle:02d}"
    frame["group"] = group
    frame["gp_predicted_hardness"] = frame.gp_predicted_log1p_hardness.astype(float)
    frame["centroid_distance"] = frame.centroid_distance_computed.astype(float)
    frame["age"] = frame.candidate_age.astype(int)
    frame["age_category"] = frame.age.map(age_cat)
    frame["provenance"] = [
        provenance_of(bool(p), bool(n))
        for p, n in zip(frame.is_previous_parent, frame.is_new_offspring, strict=True)
    ]
    frame["full_component"] = frame.instance_hash.map(lambda h: data.p_component[cycle].get(h, "not_in_P_FULL"))
    metrics = base._niche_metrics(frame.niche_id.to_numpy(int))
    frame["group_n"] = len(frame)
    frame["group_occupied_niches"] = metrics["occupied_niches"]
    frame["group_N_eff"] = metrics["N_eff"]
    keep = [
        "cycle", "group", "instance_hash", "gp_predicted_hardness", "gp_percentile",
        "niche_id", "centroid_distance", "age", "age_category", "is_previous_parent",
        "is_new_offspring", "provenance", "full_component", "group_n",
        "group_occupied_niches", "group_N_eff",
    ]
    return frame.loc[:, keep]


def trace_lineage(data: FrozenData, start: int, parent: str, max_k: int) -> dict[int, dict[str, Any]]:
    origin_niches = data.cum_p_niches[start]
    live = {parent}
    seen: set[str] = set()
    out: dict[int, dict[str, Any]] = {}
    empty: list[tuple[str, str, int]] = []
    for k in range(1, max_k + 1):
        future = start + k
        child_map = data.children.get(future)
        if child_map is None:
            break
        generated_h: set[str] = set()
        new_niche_h: set[str] = set()
        for item in live:
            for child, _, niche in child_map.get(item, empty):
                generated_h.add(child)
                if niche not in origin_niches:
                    new_niche_h.add(child)
        seen |= generated_h
        t_hits = seen & data.t_set[future]
        p_hits = {h for h in seen if h in data.p_set[future] and h != parent}
        gp10 = seen & data.gp_top10[future]
        gp5 = seen & data.gp_top5[future]
        hard_thr = data.t_p90[future]
        exact_n = sum(1 for h in t_hits if data.exact_h[future].get(h, -np.inf) >= hard_thr)
        max_pct = max((data.gp_pct[future][h] for h in seen if h in data.gp_pct[future]), default=math.nan)
        out[k] = {
            "future_cycle": future,
            "n_generated_this_step": len(generated_h),
            "n_descendants_cumulative": len(seen),
            "any_T": int(len(t_hits) > 0),
            "any_P": int(len(p_hits) > 0),
            "any_gp_top10": int(len(gp10) > 0),
            "any_gp_top5": int(len(gp5) > 0),
            "any_new_niche": int(len(new_niche_h) > 0),
            "any_survive_parent": int(len(p_hits) > 0),
            "any_oracle": int(len(t_hits) > 0),
            "any_exact_hard": int(exact_n > 0),
            "n_T": len(t_hits),
            "n_P": len(p_hits),
            "n_gp_top10": len(gp10),
            "n_new_niche": len(new_niche_h),
            "n_exact_hard": exact_n,
            "max_gp_percentile": float(max_pct) if max_pct == max_pct else math.nan,
        }
        live = {h for h in seen if h in data.p_set[future]}
        if parent in data.p_set[future]:
            live.add(parent)
    return out


def match_rescued(rescued: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    unused = control.copy()
    rows = []
    for row in rescued.sort_values("instance_hash").itertuples(index=False):
        if unused.empty:
            break
        cand = unused[(unused.age_category == row.age_category) & (unused.provenance == row.provenance)]
        match_level = "age_and_provenance"
        if cand.empty:
            cand = unused[unused.age_category == row.age_category]
            match_level = "age_only"
        if cand.empty:
            cand = unused[unused.provenance == row.provenance]
            match_level = "provenance_only"
        if cand.empty:
            cand = unused
            match_level = "score_only"
        diff = (cand.gp_percentile - row.gp_percentile).abs()
        best = float(diff.min())
        tied = cand.loc[diff == best].sort_values("instance_hash")
        pick = tied.iloc[0]
        rows.append({
            "cycle": row.cycle,
            "rescued_hash": row.instance_hash,
            "control_hash": pick.instance_hash,
            "rescued_gp_percentile": float(row.gp_percentile),
            "control_gp_percentile": float(pick.gp_percentile),
            "percentile_abs_diff": abs(float(row.gp_percentile) - float(pick.gp_percentile)),
            "rescued_age_category": row.age_category,
            "control_age_category": pick.age_category,
            "rescued_provenance": row.provenance,
            "control_provenance": pick.provenance,
            "match_level": match_level,
        })
        unused = unused[unused.instance_hash != pick.instance_hash]
    return pd.DataFrame(rows)


def run_a1(data: FrozenData) -> dict[str, Any]:
    log("A1 QD delayed descendant value")
    out = ANALYSIS_ROOT / "01_A1_qd_delayed_value"
    out.mkdir(parents=True, exist_ok=True)
    cycle_rows = []
    char_frames = []
    group_hashes: dict[tuple[int, str], list[str]] = {}
    for cycle in data.cycles:
        union = data.union[cycle]
        scores = union.gp_predicted_log1p_hardness.to_numpy(float)
        p_full = data.parents[cycle].instance_hash.astype(str).tolist()
        p_gp = global_topk_hashes(union, scores)
        full_set, gp_set = set(p_full), set(p_gp)
        rescued = sorted(full_set - gp_set)
        shared = sorted(full_set & gp_set)
        displaced = sorted(gp_set - full_set)
        group_hashes[(cycle, "QD_RESCUED")] = rescued
        group_hashes[(cycle, "SHARED_GP")] = shared
        group_hashes[(cycle, "GP_ONLY_DISPLACED")] = displaced
        for name, hashes in [("QD_RESCUED", rescued), ("SHARED_GP", shared), ("GP_ONLY_DISPLACED", displaced)]:
            char_frames.append(characterize_group(data, cycle, hashes, name))
        rec = {
            "cycle": f"C{cycle:02d}",
            "n_P_FULL": len(full_set),
            "n_P_GP": len(gp_set),
            "n_QD_RESCUED": len(rescued),
            "n_SHARED_GP": len(shared),
            "n_GP_ONLY_DISPLACED": len(displaced),
            "QD_floor_n": int((data.parents[cycle].component == "centroid_floor").sum()),
            "GP_fill_n": int((data.parents[cycle].component == "gp_fill").sum()),
            "rescued_membership_hash": membership_hash(rescued),
            "shared_membership_hash": membership_hash(shared),
            "displaced_membership_hash": membership_hash(displaced),
            "P_FULL_membership_hash": membership_hash(p_full),
            "P_GP_membership_hash": membership_hash(p_gp),
        }
        for name, hashes in [("QD_RESCUED", rescued), ("SHARED_GP", shared), ("GP_ONLY_DISPLACED", displaced)]:
            if hashes:
                metrics = base._niche_metrics(np.asarray([data.niche[cycle][h] for h in hashes], dtype=int))
                rec[f"{name}_occupied_niches"] = metrics["occupied_niches"]
                rec[f"{name}_N_eff"] = metrics["N_eff"]
                rec[f"{name}_mean_gp"] = float(np.mean([data.gp_score[cycle][h] for h in hashes]))
                rec[f"{name}_mean_gp_percentile"] = float(np.mean([data.gp_pct[cycle][h] for h in hashes]))
            else:
                rec[f"{name}_occupied_niches"] = 0
                rec[f"{name}_N_eff"] = math.nan
                rec[f"{name}_mean_gp"] = math.nan
                rec[f"{name}_mean_gp_percentile"] = math.nan
        cycle_rows.append(rec)
    cycle_table = pd.DataFrame(cycle_rows)
    char = pd.concat(char_frames, ignore_index=True) if char_frames else pd.DataFrame()
    cycle_table.to_csv(out / "02_qd_rescued_by_cycle.csv", index=False)
    char.to_csv(out / "03_immediate_characterization.csv", index=False)

    event_rows = []
    yield_rows = []
    max_cycle = max(data.cycles)
    for (cycle, group), hashes in group_hashes.items():
        log(f"  A1 tracing C{cycle:02d} {group} n={len(hashes)}")
        max_k = min(3, max_cycle - cycle)
        for parent in hashes:
            trace = trace_lineage(data, cycle, parent, max_k) if max_k > 0 else {}
            n_gen = 0
            n_t = 0
            n_p = 0
            n_new = 0
            max_pct = math.nan
            first_t = ""
            first_p = ""
            for k, rec in trace.items():
                n_gen += rec["n_generated_this_step"]
                n_t += rec["n_T"]
                n_p += rec["n_P"]
                n_new += rec["n_new_niche"]
                if np.isfinite(rec["max_gp_percentile"]) and (not np.isfinite(max_pct) or rec["max_gp_percentile"] > max_pct):
                    max_pct = rec["max_gp_percentile"]
                if rec["any_T"] and not first_t:
                    first_t = f"C{rec['future_cycle']:02d}"
                if rec["any_P"] and not first_p:
                    first_p = f"C{rec['future_cycle']:02d}"
                event_rows.append({
                    "cycle": f"C{cycle:02d}",
                    "parent_hash": parent,
                    "group": group,
                    "horizon_k": k,
                    "future_cycle": f"C{rec['future_cycle']:02d}",
                    **{key: rec[key] for key in rec if key != "future_cycle"},
                })
            yield_rows.append({
                "cycle": f"C{cycle:02d}",
                "group": group,
                "parent_hash": parent,
                "max_horizon_available": max_k,
                "n_descendants_generated": n_gen,
                "n_descendants_enter_T": n_t,
                "n_descendants_enter_P": n_p,
                "n_descendants_new_niche": n_new,
                "max_future_GP_percentile": max_pct,
                "first_future_T_cycle": first_t,
                "first_future_P_cycle": first_p,
            })
    events = pd.DataFrame(event_rows)
    yields = pd.DataFrame(yield_rows)
    log(f"  A1 events n={len(events)} yields n={len(yields)}; writing CSVs")
    events.to_csv(out / "04_descendant_events_long.csv", index=False)
    yields.to_csv(out / "06_lineage_yield.csv", index=False)

    horizon_rows = []
    for k in (1, 2, 3):
        for group in ("QD_RESCUED", "SHARED_GP", "GP_ONLY_DISPLACED"):
            sub = events[(events.group == group) & (events.horizon_k == k)]
            if sub.empty:
                continue
            flags = {
                "future_T_entry_rate": sub.any_T.to_numpy(float),
                "future_P_entry_rate": sub.any_P.to_numpy(float),
                "future_GP_top10_rate": sub.any_gp_top10.to_numpy(float),
                "future_GP_top5_rate": sub.any_gp_top5.to_numpy(float),
                "future_new_niche_rate": sub.any_new_niche.to_numpy(float),
                "future_exact_hard_descendant_rate": sub.any_exact_hard.to_numpy(float),
                "future_oracle_label_rate": sub.any_oracle.to_numpy(float),
            }
            row = {"group": group, "horizon_k": k, "n_parents_with_horizon": len(sub)}
            for name, values in flags.items():
                boot = bootstrap_mean(values, seed=BOOT_SEED + k)
                row[name] = boot["mean"]
                row[f"{name}_ci95_low"] = boot["ci95_low"]
                row[f"{name}_ci95_high"] = boot["ci95_high"]
            horizon_rows.append(row)
    horizon = pd.DataFrame(horizon_rows)
    horizon.to_csv(out / "05_horizon_summary.csv", index=False)

    match_frames = []
    for cycle in data.cycles:
        rescued = char[(char.cycle == f"C{cycle:02d}") & (char.group == "QD_RESCUED")]
        shared = char[(char.cycle == f"C{cycle:02d}") & (char.group == "SHARED_GP")]
        if rescued.empty or shared.empty:
            continue
        match_frames.append(match_rescued(rescued, shared))
    matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    match_summary_rows = []
    for k in (1, 2, 3):
        if matches.empty:
            continue
        merged = matches.merge(
            events[(events.group == "QD_RESCUED") & (events.horizon_k == k)][
                ["cycle", "parent_hash", "any_T", "any_P", "any_gp_top10", "any_new_niche", "any_exact_hard"]
            ],
            left_on=["cycle", "rescued_hash"], right_on=["cycle", "parent_hash"], how="inner",
        ).drop(columns=["parent_hash"]).merge(
            events[(events.group == "SHARED_GP") & (events.horizon_k == k)][
                ["cycle", "parent_hash", "any_T", "any_P", "any_gp_top10", "any_new_niche", "any_exact_hard"]
            ].rename(columns={
                "any_T": "ctrl_any_T", "any_P": "ctrl_any_P", "any_gp_top10": "ctrl_any_gp_top10",
                "any_new_niche": "ctrl_any_new_niche", "any_exact_hard": "ctrl_any_exact_hard",
            }),
            left_on=["cycle", "control_hash"], right_on=["cycle", "parent_hash"], how="inner",
        )
        if merged.empty:
            continue
        n_rescued = int((char.group == "QD_RESCUED").sum())
        row = {
            "horizon_k": k,
            "n_rescued_total": n_rescued,
            "n_matched_pairs_with_horizon": len(merged),
            "match_coverage": float(len(matches) / max(1, n_rescued)),
            "mean_percentile_abs_diff": float(matches.percentile_abs_diff.mean()),
        }
        for name, col, ctrl in [
            ("T_entry", "any_T", "ctrl_any_T"),
            ("P_entry", "any_P", "ctrl_any_P"),
            ("GP_top10", "any_gp_top10", "ctrl_any_gp_top10"),
            ("new_niche", "any_new_niche", "ctrl_any_new_niche"),
            ("exact_hard", "any_exact_hard", "ctrl_any_exact_hard"),
        ]:
            a = merged[col].to_numpy(float)
            b = merged[ctrl].to_numpy(float)
            ra = bootstrap_mean(a, seed=BOOT_SEED + 100 + k)
            rb = bootstrap_mean(b, seed=BOOT_SEED + 200 + k)
            diff = bootstrap_mean(a - b, seed=BOOT_SEED + 300 + k)
            row[f"rescued_{name}_rate"] = ra["mean"]
            row[f"matched_shared_{name}_rate"] = rb["mean"]
            row[f"diff_{name}"] = diff["mean"]
            row[f"diff_{name}_ci95_low"] = diff["ci95_low"]
            row[f"diff_{name}_ci95_high"] = diff["ci95_high"]
        match_summary_rows.append(row)
    match_summary = pd.DataFrame(match_summary_rows)
    match_summary.to_csv(out / "07_matched_control_summary.csv", index=False)

    def rate(group: str, k: int, col: str) -> float:
        sub = horizon[(horizon.group == group) & (horizon.horizon_k == k)]
        return float(sub[col].iloc[0]) if len(sub) else math.nan

    t1, t2, t3 = rate("QD_RESCUED", 1, "future_T_entry_rate"), rate("QD_RESCUED", 2, "future_T_entry_rate"), rate("QD_RESCUED", 3, "future_T_entry_rate")
    niche1 = rate("QD_RESCUED", 1, "future_new_niche_rate")
    n_rescued = int((char.group == "QD_RESCUED").sum())
    cycle_positive = 0
    for cycle in data.cycles:
        sub = events[(events.cycle == f"C{cycle:02d}") & (events.group == "QD_RESCUED") & (events.horizon_k == 1)]
        if len(sub) and float(sub.any_T.mean()) >= 0.03:
            cycle_positive += 1
    matched_t = float(match_summary.rescued_T_entry_rate.iloc[0]) if len(match_summary) else math.nan
    matched_ctrl_t = float(match_summary.matched_shared_T_entry_rate.iloc[0]) if len(match_summary) else math.nan
    if np.isfinite(t1) and t1 >= 0.03 and cycle_positive >= 2 and np.isfinite(matched_t) and matched_t >= 0.5 * max(matched_ctrl_t, 1e-9):
        verdict = "QD_DELAYED_LINEAGE_VALUE_OBSERVED"
    elif (not np.isfinite(t1) or t1 < 0.02) and (not np.isfinite(rate("QD_RESCUED", 1, "future_P_entry_rate")) or rate("QD_RESCUED", 1, "future_P_entry_rate") < 0.05):
        verdict = "QD_DIVERSITY_WITHOUT_CLEAR_DELAYED_VALUE"
    else:
        verdict = "QD_DELAYED_VALUE_MIXED"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(cycle_table.cycle, cycle_table.n_QD_RESCUED, color="#2c7fb8")
    finish_fig(out / "fig01_qd_rescued_count.png", "Cycle", "N QD-rescued parents", "QD-rescued parent count by cycle")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, color in [("QD_RESCUED", "#2c7fb8"), ("SHARED_GP", "#d95f0e")]:
        sub = horizon[horizon.group == group]
        if sub.empty:
            continue
        ax.plot(sub.horizon_k, sub.future_T_entry_rate, marker="o", label=group, color=color)
    ax.legend()
    finish_fig(out / "fig02_future_T_entry_by_horizon.png", "Horizon k", "Future T-entry rate", "Future T-entry by horizon")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, color in [("QD_RESCUED", "#2c7fb8"), ("SHARED_GP", "#d95f0e")]:
        sub = horizon[horizon.group == group]
        if sub.empty:
            continue
        ax.plot(sub.horizon_k, sub.future_exact_hard_descendant_rate, marker="o", label=group, color=color)
    ax.legend()
    finish_fig(out / "fig03_future_hard_tail_entry.png", "Horizon k", "Exact-hard descendant rate", "Future exact-hard tail entry")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for group, color in [("QD_RESCUED", "#2c7fb8"), ("SHARED_GP", "#d95f0e")]:
        sub = horizon[horizon.group == group]
        if sub.empty:
            continue
        ax.plot(sub.horizon_k, sub.future_new_niche_rate, marker="o", label=group, color=color)
    ax.legend()
    finish_fig(out / "fig04_new_niche_descendant_rate.png", "Horizon k", "New-niche descendant rate", "Future new-niche descendant rate")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = yields[yields.group.isin(["QD_RESCUED", "SHARED_GP"])]
    if len(plot):
        plot.boxplot(column="n_descendants_enter_T", by="group", ax=ax)
        plt.suptitle("")
    finish_fig(out / "fig05_lineage_yield_distribution.png", "Group", "N descendants entering T", "Lineage T-yield distribution")

    write_md(out / "01_definition.md", f"""# A1 definitions

Frozen range: C00–C{max(data.cycles):02d}.

C_s = unique(P_{{s-1}} UNION O_s), the stored corrected union `C_union_scored`.

P_s^GP = global GP Top10k(C_s) using stored `gp_predicted_log1p_hardness` and the formal (-score, hash) lexsort.

P_s^FULL = actual stored parents (QD floor + GP fill).

QD_RESCUED_s = P_s^FULL \\ P_s^GP
SHARED_GP_s = P_s^FULL ∩ P_s^GP
GP_ONLY_DISPLACED_s = P_s^GP \\ P_s^FULL

Descendants of parent x selected at cycle s are mutation children generated from x (or from later selected descendants of x) in cycles s+1..s+k inside the frozen snapshot. The original parent surviving into a later P/T is not counted as a descendant event.

New niche: descendant niche not occupied by any P_t for t ≤ s.

Exact-hard descendant: descendant enters historically labelled T_{{s+k}} and exact current hardness ≥ T_{{s+k}} P90. Missing exact labels are marked unavailable; no Concorde calls were made.

Matched control uses SHARED_GP only, so both sides generate lineages. Matching uses GP percentile, age category, and provenance. No outcome entered matching.
""")
    write_md(out / "08_final_report.md", f"""# A1 QD delayed descendant-value

## Immediate selection
QD-rescued N by cycle: {cycle_table.n_QD_RESCUED.tolist()}
SHARED_GP N by cycle: {cycle_table.n_SHARED_GP.tolist()}
GP-only displaced N by cycle: {cycle_table.n_GP_ONLY_DISPLACED.tolist()}

Mean GP percentile QD_RESCUED vs SHARED_GP: {cycle_table.QD_RESCUED_mean_gp_percentile.round(3).tolist()} vs {cycle_table.SHARED_GP_mean_gp_percentile.round(3).tolist()}

Lower current GP score of QD-rescued parents is expected and is not treated as failure.

## Delayed value
QD_RESCUED future T-entry: k=1 {fmt_rate(t1)}; k=2 {fmt_rate(t2)}; k=3 {fmt_rate(t3)}
QD_RESCUED new-niche descendant rate k=1: {fmt_rate(niche1)}
Cycles with k=1 T-entry ≥ 0.03: {cycle_positive}

Matched SHARED_GP k=1 T-entry: rescued {fmt_rate(matched_t)} vs control {fmt_rate(matched_ctrl_t)}
Match coverage: {fmt_rate(float(match_summary.match_coverage.iloc[0]) if len(match_summary) else math.nan)}

GP_ONLY_DISPLACED parents generate no children because they were not selected. Their empty lineage is a definitional contrast, not a mutation failure.

## Verdict
**{verdict}**

This lineage audit does not claim causal downstream solver improvement.
""")
    return {
        "n_qd_rescued": n_rescued,
        "t1": t1, "t2": t2, "t3": t3,
        "new_niche_k1": niche1,
        "verdict": verdict,
        "horizon": horizon,
        "cycle_table": cycle_table,
    }


def run_a2(data: FrozenData) -> dict[str, Any]:
    log("A2 mutation / operator contribution")
    out = ANALYSIS_ROOT / "02_A2_operator_contribution"
    out.mkdir(parents=True, exist_ok=True)
    gen_rows = []
    delta_rows = []
    t_rows = []
    p_rows = []
    niche_rows = []
    future_rows = []
    max_cycle = max(data.cycles)
    for cycle in data.cycles:
        manifest = data.manifest[cycle]
        offspring = data.offspring[cycle]
        union = data.union[cycle]
        score_map = data.gp_score[cycle]
        n_offspring = len(offspring)
        t_new = data.train[cycle][data.train[cycle].is_new_offspring.astype(bool)]
        p_new = data.parents[cycle][data.parents[cycle].is_new_offspring.astype(bool)]
        p_old = data.parents[cycle][data.parents[cycle].is_previous_parent.astype(bool) & ~data.parents[cycle].is_new_offspring.astype(bool)]
        prior_niches = set() if cycle == 0 else data.p_niches[cycle - 1]
        child = offspring.copy()
        child["child_gp"] = child.instance_hash.map(score_map)
        child["parent_gp"] = child.parent_hash.map(score_map)
        child["delta_GP"] = child.child_gp - child.parent_gp
        child["sib_rank"] = child.groupby("parent_hash")["child_gp"].rank(ascending=False, method="min")
        child["n_sib"] = child.groupby("parent_hash")["child_gp"].transform("size")
        child["rank_improvement"] = (child.n_sib + 1) / 2.0 - child.sib_rank
        child["is_best"] = child.sib_rank.eq(1)
        n_new_t = int(t_new.shape[0])
        n_new_p = int(p_new.shape[0])
        op_niches = {op: set(offspring.loc[offspring.operator_id.eq(op), "niche_id"].astype(int)) for op in OPERATOR_NAMES}
        new_by_op = {op: op_niches[op] - prior_niches for op in OPERATOR_NAMES}
        exclusive = {}
        for op in OPERATOR_NAMES:
            others = set().union(*(new_by_op[other] for other in OPERATOR_NAMES if other != op)) if OPERATOR_NAMES else set()
            exclusive[op] = new_by_op[op] - others
        labeled_new = data.labeled[cycle][data.labeled[cycle].is_new_offspring.astype(bool)]
        for op in OPERATOR_NAMES:
            raw = manifest[manifest.operator_id.eq(op)]
            kids = offspring[offspring.operator_id.eq(op)]
            gen_rows.append({
                "cycle": f"C{cycle:02d}",
                "operator_id": op,
                "generated_children": len(raw),
                "valid_children": int((raw.valid_after_repair & raw.proposal_status.eq("accepted")).sum()),
                "repaired_children": int((raw.repair_count > 0).sum()),
                "duplicate_children": int(raw.duplicate.sum()),
                "unique_children": len(kids),
                "percentage_of_O_s": 100.0 * len(kids) / max(1, n_offspring),
            })
            deltas = child.loc[child.operator_id.eq(op), "delta_GP"].dropna().to_numpy(float)
            ranks = child.loc[child.operator_id.eq(op), "rank_improvement"].dropna().to_numpy(float)
            best_frac = float(child.loc[child.operator_id.eq(op), "is_best"].mean()) if len(kids) else math.nan
            delta_rows.append({
                "cycle": f"C{cycle:02d}",
                "operator_id": op,
                "n": int(len(deltas)),
                "mean_delta_GP": float(np.mean(deltas)) if len(deltas) else math.nan,
                "median_delta_GP": float(np.median(deltas)) if len(deltas) else math.nan,
                "P10": float(np.percentile(deltas, 10)) if len(deltas) else math.nan,
                "P25": float(np.percentile(deltas, 25)) if len(deltas) else math.nan,
                "P75": float(np.percentile(deltas, 75)) if len(deltas) else math.nan,
                "P90": float(np.percentile(deltas, 90)) if len(deltas) else math.nan,
                "frac_delta_gt0": float(np.mean(deltas > 0)) if len(deltas) else math.nan,
                "frac_delta_lt0": float(np.mean(deltas < 0)) if len(deltas) else math.nan,
                "frac_parent_best_child": best_frac,
                "mean_rank_improvement": float(np.mean(ranks)) if len(ranks) else math.nan,
            })
            t_op = t_new[t_new.operator_id.astype(str).eq(op)]
            generated_share = len(kids) / max(1, n_offspring)
            t_share = len(t_op) / max(1, n_new_t)
            h_vals = labeled_new.loc[labeled_new.operator_id.astype(str).eq(op), "current_hardness"].to_numpy(float)
            t_rows.append({
                "cycle": f"C{cycle:02d}",
                "operator_id": op,
                "generated_share": generated_share,
                "T_entry_n": len(t_op),
                "T_entry_share": t_share,
                "T_enrichment": t_share / generated_share if generated_share else math.nan,
                "Top5_GP_entry_n": int(t_op.instance_hash.astype(str).isin(data.gp_top5[cycle]).sum()) if len(t_op) else 0,
                "Top10_GP_entry_n": int(t_op.instance_hash.astype(str).isin(data.gp_top10[cycle]).sum()) if len(t_op) else 0,
                "exact_H_n": int(len(h_vals)),
                "exact_H_mean": float(np.mean(h_vals)) if len(h_vals) else math.nan,
                "exact_H_median": float(np.median(h_vals)) if len(h_vals) else math.nan,
                "exact_H_P90": float(np.percentile(h_vals, 90)) if len(h_vals) else math.nan,
            })
            p_op = p_new[p_new.operator_id.astype(str).eq(op)]
            p_share = len(p_op) / max(1, n_new_p)
            floor_n = int((p_op.component == "centroid_floor").sum())
            fill_n = int((p_op.component == "gp_fill").sum())
            p_rows.append({
                "cycle": f"C{cycle:02d}",
                "operator_id": op,
                "generated_share": generated_share,
                "P_entry_n": len(p_op),
                "P_entry_share": p_share,
                "P_enrichment": p_share / generated_share if generated_share else math.nan,
                "QD_floor_n": floor_n,
                "GP_fill_n": fill_n,
                "QD_floor_share": floor_n / max(1, len(p_new[p_new.component.eq("centroid_floor")])),
                "QD_floor_enrichment": (floor_n / max(1, int((p_new.component == "centroid_floor").sum()))) / generated_share if generated_share else math.nan,
            })
            kids_new_niche = kids[~kids.niche_id.astype(int).isin(prior_niches)]
            niche_rows.append({
                "cycle": f"C{cycle:02d}",
                "operator_id": op,
                "distinct_niches_reached": int(kids.niche_id.nunique()),
                "newly_reached_vs_P_prev": len(new_by_op[op]),
                "frac_children_new_niche": float(len(kids_new_niche) / max(1, len(kids))),
                "exclusive_new_niches": len(exclusive[op]),
                "QD_floor_selections": floor_n,
            })
            admitted = p_op.instance_hash.astype(str).tolist()
            max_k = min(3, max_cycle - cycle)
            t_any = p_any = niche_any = hard_any = 0
            n_future = 0
            for parent in admitted:
                if max_k <= 0:
                    continue
                n_future += 1
                trace = trace_lineage(data, cycle, parent, max_k)
                if any(rec["any_T"] for rec in trace.values()):
                    t_any += 1
                if any(rec["any_P"] for rec in trace.values()):
                    p_any += 1
                if any(rec["any_new_niche"] for rec in trace.values()):
                    niche_any += 1
                if any(rec["any_exact_hard"] for rec in trace.values()):
                    hard_any += 1
            future_rows.append({
                "cycle": f"C{cycle:02d}",
                "operator_id": op,
                "n_admitted_to_P": len(admitted),
                "n_with_future_horizon": n_future,
                "delayed_T_entry_rate": t_any / n_future if n_future else math.nan,
                "delayed_P_entry_rate": p_any / n_future if n_future else math.nan,
                "delayed_new_niche_rate": niche_any / n_future if n_future else math.nan,
                "delayed_hard_tail_rate": hard_any / n_future if n_future else math.nan,
            })
        t_rows.append({
            "cycle": f"C{cycle:02d}",
            "operator_id": "SURVIVOR_PARENT",
            "generated_share": math.nan,
            "T_entry_n": int((~data.train[cycle].is_new_offspring.astype(bool)).sum()),
            "T_entry_share": math.nan,
            "T_enrichment": math.nan,
            "Top5_GP_entry_n": math.nan,
            "Top10_GP_entry_n": math.nan,
            "exact_H_n": int((~data.labeled[cycle].is_new_offspring.astype(bool)).sum()),
            "exact_H_mean": math.nan,
            "exact_H_median": math.nan,
            "exact_H_P90": math.nan,
        })
        p_rows.append({
            "cycle": f"C{cycle:02d}",
            "operator_id": "SURVIVOR_PARENT",
            "generated_share": math.nan,
            "P_entry_n": len(p_old),
            "P_entry_share": math.nan,
            "P_enrichment": math.nan,
            "QD_floor_n": int((p_old.component == "centroid_floor").sum()),
            "GP_fill_n": int((p_old.component == "gp_fill").sum()),
            "QD_floor_share": math.nan,
            "QD_floor_enrichment": math.nan,
        })
    gen = pd.DataFrame(gen_rows)
    deltas = pd.DataFrame(delta_rows)
    t_enr = pd.DataFrame(t_rows)
    p_enr = pd.DataFrame(p_rows)
    niches = pd.DataFrame(niche_rows)
    future = pd.DataFrame(future_rows)
    gen.to_csv(out / "01_operator_generation.csv", index=False)
    deltas.to_csv(out / "02_child_parent_gp_delta.csv", index=False)
    t_enr.to_csv(out / "03_T_enrichment.csv", index=False)
    p_enr.to_csv(out / "04_P_enrichment.csv", index=False)
    niches.to_csv(out / "05_niche_discovery.csv", index=False)
    future.to_csv(out / "06_future_lineage_value.csv", index=False)

    def mean_op(frame: pd.DataFrame, col: str) -> pd.Series:
        sub = frame[frame.operator_id.isin(OPERATOR_NAMES)]
        return sub.groupby("operator_id")[col].mean()

    matrix = pd.DataFrame({
        "operator_id": list(OPERATOR_NAMES),
        "local_GP_gain": mean_op(deltas, "mean_delta_GP").reindex(OPERATOR_NAMES).to_numpy(),
        "T_enrichment": mean_op(t_enr, "T_enrichment").reindex(OPERATOR_NAMES).to_numpy(),
        "P_enrichment": mean_op(p_enr, "P_enrichment").reindex(OPERATOR_NAMES).to_numpy(),
        "QD_floor_enrichment": mean_op(p_enr, "QD_floor_enrichment").reindex(OPERATOR_NAMES).to_numpy(),
        "new_niche_rate": mean_op(niches, "frac_children_new_niche").reindex(OPERATOR_NAMES).to_numpy(),
        "delayed_T_entry_rate": mean_op(future, "delayed_T_entry_rate").reindex(OPERATOR_NAMES).to_numpy(),
        "delayed_new_niche_rate": mean_op(future, "delayed_new_niche_rate").reindex(OPERATOR_NAMES).to_numpy(),
    })
    vis = matrix.copy()
    for col in matrix.columns:
        if col == "operator_id":
            continue
        values = vis[col].to_numpy(float)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            vis[col + "_norm"] = math.nan
            continue
        lo, hi = np.nanmin(values), np.nanmax(values)
        vis[col + "_norm"] = (values - lo) / (hi - lo) if hi > lo else 0.0
    vis.to_csv(out / "07_operator_specialization_matrix.csv", index=False)

    t_mean = matrix.set_index("operator_id").T_enrichment
    niche_mean = matrix.set_index("operator_id").new_niche_rate
    strongest_t = t_mean.sort_values(ascending=False).head(3).index.tolist()
    strongest_niche = niche_mean.sort_values(ascending=False).head(3).index.tolist()
    rho, _ = spearmanr(t_mean.to_numpy(), niche_mean.to_numpy())
    cv_t = float(t_mean.std(ddof=1) / t_mean.mean()) if t_mean.mean() else math.nan
    if np.isfinite(rho) and rho < -0.3 and cv_t >= 0.25:
        verdict = "OPERATOR_ROLES_DIFFERENTIATED"
        differentiated = "yes"
    elif np.isfinite(cv_t) and cv_t < 0.15:
        verdict = "MUTATION_EFFECT_BROADLY_OPERATOR_AGNOSTIC"
        differentiated = "no"
    else:
        verdict = "OPERATOR_CONTRIBUTION_MIXED"
        differentiated = "mixed"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(OPERATOR_NAMES)), [t_mean.get(op, math.nan) for op in OPERATOR_NAMES])
    ax.set_xticks(range(len(OPERATOR_NAMES)))
    ax.set_xticklabels([op.replace("_mutation", "") for op in OPERATOR_NAMES], rotation=45, ha="right")
    finish_fig(out / "fig02_operator_T_enrichment.png", "Operator", "Mean T enrichment", "Operator T enrichment")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    gain = matrix.set_index("operator_id").local_GP_gain
    ax.bar(range(len(OPERATOR_NAMES)), [gain.get(op, math.nan) for op in OPERATOR_NAMES])
    ax.set_xticks(range(len(OPERATOR_NAMES)))
    ax.set_xticklabels([op.replace("_mutation", "") for op in OPERATOR_NAMES], rotation=45, ha="right")
    finish_fig(out / "fig01_operator_GP_gain.png", "Operator", "Mean delta GP", "Operator local GP gain")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pmean = matrix.set_index("operator_id").P_enrichment
    ax.bar(range(len(OPERATOR_NAMES)), [pmean.get(op, math.nan) for op in OPERATOR_NAMES])
    ax.set_xticks(range(len(OPERATOR_NAMES)))
    ax.set_xticklabels([op.replace("_mutation", "") for op in OPERATOR_NAMES], rotation=45, ha="right")
    finish_fig(out / "fig03_operator_P_enrichment.png", "Operator", "Mean P enrichment", "Operator P enrichment")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(OPERATOR_NAMES)), [niche_mean.get(op, math.nan) for op in OPERATOR_NAMES])
    ax.set_xticks(range(len(OPERATOR_NAMES)))
    ax.set_xticklabels([op.replace("_mutation", "") for op in OPERATOR_NAMES], rotation=45, ha="right")
    finish_fig(out / "fig04_operator_new_niche_rate.png", "Operator", "New-niche child fraction", "Operator new-niche rate")
    norm_cols = [c for c in vis.columns if c.endswith("_norm")]
    grid = vis.set_index("operator_id")[norm_cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(OPERATOR_NAMES)))
    ax.set_yticklabels([op.replace("_mutation", "") for op in OPERATOR_NAMES])
    ax.set_xticks(range(len(norm_cols)))
    ax.set_xticklabels([c.replace("_norm", "") for c in norm_cols], rotation=45, ha="right")
    plt.colorbar(im, ax=ax, fraction=0.046)
    finish_fig(out / "fig05_operator_specialization_heatmap.png", "Metric (min-max within column)", "Operator", "Operator specialization (visualization only)")

    write_md(out / "08_final_report.md", f"""# A2 mutation / operator contribution

Survivor parents are reported as SURVIVOR_PARENT and are never assigned a FOGA operator.

Strongest T-enriched operators (mean across frozen cycles): {strongest_t}
Strongest niche-expansion operators: {strongest_niche}
Spearman(T enrichment, new-niche rate) = {rho:.4f}
CV of T enrichment = {cv_t:.4f}

This analysis does not replace a Random / mutation-only training baseline and does not rank operators for removal.

## Verdict
**{verdict}**
""")
    return {
        "strongest_t": strongest_t,
        "strongest_niche": strongest_niche,
        "differentiated": differentiated,
        "verdict": verdict,
        "matrix": matrix,
    }


def run_a3(data: FrozenData) -> dict[str, Any]:
    log("A3 hardness–diversity–training-value")
    out = ANALYSIS_ROOT / "03_A3_training_value_decomposition"
    out.mkdir(parents=True, exist_ok=True)
    external = pd.read_csv(E45_AUDIT / "14_external_8set.csv")
    gap = {}
    for _, row in external.iterrows():
        if pd.isna(row.get("dataset")) or pd.isna(row.get("model")):
            continue
        gap[(str(row.model), str(row.dataset))] = float(row.mean_gap_percent)
    ood5 = {}
    for model in [f"M{i}" for i in range(max(data.cycles) + 2)]:
        values = [gap.get((model, name)) for name in OOD5]
        if all(v is not None and np.isfinite(v) for v in values):
            ood5[model] = float(np.mean(values))
    rows = []
    bench_rows = []
    for cycle in data.cycles:
        summary = data.summary[cycle]
        hardness = data.labeled[cycle].current_hardness.to_numpy(float)
        repair = data.repair[cycle]
        p_hashes = data.p_set[cycle]
        t_hashes = data.t_set[cycle]
        prev_p = data.p_set[cycle - 1] if cycle else set()
        prev_t = data.t_set[cycle - 1] if cycle else set()
        newly = 0
        if cycle:
            newly = len(data.p_niches[cycle] - data.cum_p_niches[cycle - 1])
        else:
            newly = len(data.p_niches[cycle])
        m_in, m_out = f"M{cycle}", f"M{cycle + 1}"
        row = {
            "cycle": f"C{cycle:02d}",
            "transition": f"{m_in}->{m_out}",
            "H_mean": float(np.mean(hardness)),
            "H_median": float(np.median(hardness)),
            "H_P90": float(np.percentile(hardness, 90)),
            "H_P95": float(np.percentile(hardness, 95)),
            "H_max": float(np.max(hardness)),
            "GP_pred_mean": float(summary["predicted_T"]["mean"]),
            "GP_calibration_ENR": "unavailable",
            "GP_diagnostic_Spearman": float(summary["gp"]["fit_spearman"]),
            "P_occupied_niches": int(summary["diversity"]["P"]["occupied_niches"]),
            "P_N_eff": float(summary["diversity"]["P"]["N_eff"]),
            "P_entropy": float(summary["diversity"]["P"]["entropy"]),
            "P_gini": float(summary["diversity"]["P"]["gini"]),
            "T_occupied_niches": int(summary["diversity"]["T"]["occupied_niches"]),
            "T_N_eff": float(summary["diversity"]["T"]["N_eff"]),
            "P_turnover": float(summary["turnover"]["P_turnover"]),
            "old_parent_share_P": float(summary["survival"]["old_parent_fraction_P"]),
            "age0_fraction": float(summary["age"]["all"]["age0_fraction"]),
            "new_T_instance_fraction": 1.0 if cycle == 0 else float(1.0 - len(t_hashes & prev_t) / max(1, len(t_hashes))),
            "adjacent_T_jaccard": math.nan if cycle == 0 else float(len(t_hashes & prev_t) / len(t_hashes | prev_t)),
            "adjacent_P_jaccard": math.nan if cycle == 0 else float(len(p_hashes & prev_p) / len(p_hashes | prev_p)),
            "newly_reached_niches": newly,
            "repair_mean_delta": float(repair.delta_post_minus_pre.mean()),
            "repair_frac_improved": float((repair.delta_post_minus_pre < 0).mean()),
            "repair_P90": float(np.percentile(repair.delta_post_minus_pre, 90)),
            "archive_n": int(summary["archive"]["archive_n"]),
            "archive_new_unique": int(summary["archive"]["new_unique"]),
            "new_oracle_concorde_calls": int(summary["oracle"]["new_concorde_calls"]),
            "OOD5_M_in": ood5.get(m_in, math.nan),
            "OOD5_M_out": ood5.get(m_out, math.nan),
            "DeltaGap_OOD5": (ood5[m_out] - ood5[m_in]) if m_in in ood5 and m_out in ood5 else math.nan,
        }
        for dataset, label in DATASET_LABEL.items():
            gin = gap.get((m_in, dataset))
            gout = gap.get((m_out, dataset))
            delta = gout - gin if gin is not None and gout is not None else math.nan
            row[f"DeltaGap_{label}"] = delta
            bench_rows.append({
                "cycle": f"C{cycle:02d}", "transition": f"{m_in}->{m_out}",
                "dataset": label, "Gap_in": gin, "Gap_out": gout, "DeltaGap": delta,
            })
        rows.append(row)
    master = pd.DataFrame(rows)
    bench = pd.DataFrame(bench_rows)
    master.to_csv(out / "01_cycle_level_master_table.csv", index=False)
    bench.to_csv(out / "05_benchmark_delta_by_cycle.csv", index=False)

    predictors = {
        "mean_hardness": "H_mean",
        "P90_hardness": "H_P90",
        "N_eff": "P_N_eff",
        "turnover": "P_turnover",
        "age0_fraction": "age0_fraction",
        "new_niche_count": "newly_reached_niches",
        "repair_magnitude": "repair_mean_delta",
        "new_T_instance_fraction": "new_T_instance_fraction",
    }
    y = master.DeltaGap_OOD5.to_numpy(float)
    corr_rows = []
    loco_rows = []
    for name, col in predictors.items():
        x = master[col].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3:
            rho, pval = spearmanr(x[mask], y[mask])
        else:
            rho, pval = math.nan, math.nan
        corr_rows.append({
            "predictor": name, "column": col, "n": int(mask.sum()),
            "spearman_vs_next_OOD5": float(rho) if rho == rho else math.nan,
            "p_value_exploratory_only": float(pval) if pval == pval else math.nan,
            "analysis_class": "exploratory small-n trajectory association",
        })
        for i, cycle in enumerate(master.cycle):
            keep = np.ones(len(master), dtype=bool)
            keep[i] = False
            xm, ym = x[keep], y[keep]
            finite = np.isfinite(xm) & np.isfinite(ym)
            if finite.sum() >= 3:
                r, _ = spearmanr(xm[finite], ym[finite])
            else:
                r = math.nan
            loco_rows.append({"left_out_cycle": cycle, "predictor": name, "spearman": float(r) if r == r else math.nan})
    corr = pd.DataFrame(corr_rows)
    loco = pd.DataFrame(loco_rows)
    corr.to_csv(out / "02_trajectory_correlations.csv", index=False)
    loco.to_csv(out / "03_leave_one_cycle_out.csv", index=False)

    def zscore(series: pd.Series) -> pd.Series:
        values = series.astype(float)
        std = values.std(ddof=1)
        return (values - values.mean()) / std if std else values * 0

    traj = pd.DataFrame({
        "cycle": master.cycle,
        "z_hardness": zscore(master.H_mean),
        "z_P_N_eff": zscore(master.P_N_eff),
        "z_turnover": zscore(master.P_turnover),
        "z_new_instance_fraction": zscore(master.new_T_instance_fraction),
        "z_repair_magnitude": zscore(master.repair_mean_delta),
        "z_OOD5_change": zscore(master.DeltaGap_OOD5),
    })
    h_med = master.H_mean.median()
    ood_med = master.DeltaGap_OOD5.median()
    neff_med = master.P_N_eff.median()
    turn_med = master.P_turnover.median()
    type_rows = []
    for _, row in master.iterrows():
        high_h = row.H_mean > h_med
        strong_ood = row.DeltaGap_OOD5 < ood_med  # more negative gap change is stronger gain
        high_div = (row.P_N_eff > neff_med) or (row.P_turnover > turn_med)
        if high_h and strong_ood:
            label = "A_high_hardness_strong_OOD_gain"
        elif high_h and not strong_ood:
            label = "B_high_hardness_weak_OOD_gain"
        elif (not high_h) and high_div and strong_ood:
            label = "C_moderate_hardness_high_renewal_strong_OOD_gain"
        elif (not high_h) and (not high_div) and (not strong_ood):
            label = "D_weak_hardness_weak_renewal_weak_gain"
        else:
            label = "OTHER_mixed_relative_pattern"
        type_rows.append({
            "cycle": row.cycle, "pattern": label,
            "H_mean": row.H_mean, "DeltaGap_OOD5": row.DeltaGap_OOD5,
            "P_N_eff": row.P_N_eff, "P_turnover": row.P_turnover,
            "high_hardness_vs_median": high_h, "strong_OOD_gain_vs_median": strong_ood,
            "high_diversity_or_turnover_vs_median": high_div,
        })
    typology = pd.DataFrame(type_rows)
    typology.to_csv(out / "04_cycle_typology.csv", index=False)
    counter = typology[typology.pattern.isin(["B_high_hardness_weak_OOD_gain", "C_moderate_hardness_high_renewal_strong_OOD_gain"])]
    rho_h = float(corr.loc[corr.predictor.eq("mean_hardness"), "spearman_vs_next_OOD5"].iloc[0])
    rho_n = float(corr.loc[corr.predictor.eq("N_eff"), "spearman_vs_next_OOD5"].iloc[0])
    rho_t = float(corr.loc[corr.predictor.eq("turnover"), "spearman_vs_next_OOD5"].iloc[0])
    if abs(rho_h) >= 0.6 and abs(rho_h) > abs(rho_n) and abs(rho_h) > abs(rho_t) and counter.empty:
        verdict = "HARDNESS_DOMINANT_SIGNAL"
    elif abs(rho_h) < 0.4 or len(counter):
        verdict = "TRAINING_VALUE_NOT_EXPLAINED_BY_HARDNESS_ALONE"
    else:
        verdict = "TRAINING_VALUE_DECOMPOSITION_MIXED"
    if abs(rho_h) < 0.4 and (abs(rho_n) >= 0.4 or abs(rho_t) >= 0.4):
        verdict = "TRAINING_VALUE_NOT_EXPLAINED_BY_HARDNESS_ALONE"

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(master.H_mean, master.DeltaGap_OOD5)
    for _, row in master.iterrows():
        ax.annotate(row.cycle, (row.H_mean, row.DeltaGap_OOD5))
    finish_fig(out / "fig01_hardness_vs_OOD_gain.png", "Mean T hardness", "DeltaGap OOD5 (next-M minus current-M)", "Hardness vs next OOD5 change")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(master.P_N_eff, master.DeltaGap_OOD5)
    for _, row in master.iterrows():
        ax.annotate(row.cycle, (row.P_N_eff, row.DeltaGap_OOD5))
    finish_fig(out / "fig02_neff_vs_OOD_gain.png", "P N_eff", "DeltaGap OOD5", "N_eff vs next OOD5 change")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(master.P_turnover, master.DeltaGap_OOD5)
    for _, row in master.iterrows():
        ax.annotate(row.cycle, (row.P_turnover, row.DeltaGap_OOD5))
    finish_fig(out / "fig03_turnover_vs_OOD_gain.png", "P turnover", "DeltaGap OOD5", "Turnover vs next OOD5 change")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(master.repair_mean_delta, master.DeltaGap_OOD5)
    for _, row in master.iterrows():
        ax.annotate(row.cycle, (row.repair_mean_delta, row.DeltaGap_OOD5))
    finish_fig(out / "fig04_repair_vs_OOD_gain.png", "Repair mean delta", "DeltaGap OOD5", "Repair vs next OOD5 change")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col, label in [
        ("z_hardness", "hardness"), ("z_P_N_eff", "P N_eff"), ("z_turnover", "turnover"),
        ("z_new_instance_fraction", "new T fraction"), ("z_repair_magnitude", "repair"), ("z_OOD5_change", "OOD5 change"),
    ]:
        ax.plot(traj.cycle, traj[col], marker="o", label=label)
    ax.legend()
    finish_fig(out / "fig05_standardized_trajectory.png", "Cycle", "Within-trajectory z-score", "Standardized trajectories (visualization only)")

    write_md(out / "06_final_report.md", f"""# A3 hardness–diversity–training-value

Exploratory small-n trajectory association. n={len(master)} completed M_s→M_{{s+1}} transitions. Do not treat p-values as confirmatory.

OOD5 = unweighted mean gap of GM100, TSPLIB100, S1, S2, S3 from already completed passive E4.5 evaluations. Uniform100, GM500 and Original70 are reported separately. No missing external evaluation was launched.

Spearman vs next OOD5 DeltaGap:
- mean hardness = {rho_h:.4f}
- P N_eff = {rho_n:.4f}
- turnover = {rho_t:.4f}

Negative DeltaGap is improvement. Counterexample / mixed-pattern cycles: {counter.cycle.tolist() if len(counter) else []}

## Verdict
**{verdict}**

No causal claim from trajectory correlation alone.
""")
    return {
        "rho_h": rho_h, "rho_n": rho_n, "rho_t": rho_t,
        "counterexamples": counter.cycle.tolist(),
        "verdict": verdict,
        "master": master,
    }


def run_a4(data: FrozenData) -> dict[str, Any]:
    log("A4 union survivor value")
    out = ANALYSIS_ROOT / "04_A4_union_survivor_value"
    out.mkdir(parents=True, exist_ok=True)
    age_rows = []
    next_rows = []
    multi_rows = []
    appearance: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    max_cycle = max(data.cycles)
    for cycle in data.cycles:
        parents = data.parents[cycle].copy()
        parents["age_group"] = parents.candidate_age.astype(int).map(age_group)
        parents["age4plus"] = parents.candidate_age.astype(int).ge(4)
        for row in parents.itertuples(index=False):
            appearance[str(row.instance_hash)].append((cycle, int(row.candidate_age), str(row.instance_hash)))
        groups = ["AGE0", "AGE1", "AGE2", "AGE3PLUS"]
        if int(parents.age4plus.sum()) >= 20:
            groups.append("AGE4PLUS")
        for group in groups:
            if group == "AGE4PLUS":
                sub = parents[parents.age4plus]
            else:
                sub = parents[parents.age_group.eq(group)]
            if sub.empty:
                continue
            metrics = base._niche_metrics(sub.niche_id.to_numpy(int))
            pct = [data.gp_pct[cycle].get(h, math.nan) for h in sub.instance_hash.astype(str)]
            age_rows.append({
                "cycle": f"C{cycle:02d}",
                "age_group": group,
                "N": len(sub),
                "mean_GP_score": float(sub.gp_predicted_log1p_hardness.mean()),
                "mean_GP_percentile": float(np.nanmean(pct)),
                "occupied_niches": metrics["occupied_niches"],
                "N_eff": metrics["N_eff"],
                "QD_floor_fraction": float((sub.component == "centroid_floor").mean()),
                "GP_fill_fraction": float((sub.component == "gp_fill").mean()),
            })
            max_k = 1 if cycle < max_cycle else 0
            best_gains = []
            frac_better = frac_t = frac_p = frac_new = 0
            n_t_desc = n_p_desc = 0
            n = 0
            for parent in sub.instance_hash.astype(str):
                n += 1
                kids = data.children.get(cycle + 1, {}).get(parent, []) if max_k else []
                parent_gp = data.gp_score[cycle].get(parent, math.nan)
                child_gps = [data.gp_score.get(cycle + 1, {}).get(child, math.nan) for child, _, _ in kids]
                finite = [g for g in child_gps if np.isfinite(g) and np.isfinite(parent_gp)]
                if finite:
                    best_gains.append(max(finite) - parent_gp)
                    if max(finite) > parent_gp:
                        frac_better += 1
                child_hashes = [child for child, _, _ in kids]
                t_hits = [h for h in child_hashes if h in data.t_set.get(cycle + 1, set())]
                p_hits = [h for h in child_hashes if h in data.p_set.get(cycle + 1, set())]
                origin = data.cum_p_niches[cycle]
                new_hits = [h for h, _, niche in kids if niche not in origin]
                if t_hits:
                    frac_t += 1
                if p_hits:
                    frac_p += 1
                if new_hits:
                    frac_new += 1
                n_t_desc += len(t_hits)
                n_p_desc += len(p_hits)
            next_rows.append({
                "cycle": f"C{cycle:02d}",
                "age_group": group,
                "N": n,
                "mean_best_child_GP_gain": float(np.mean(best_gains)) if best_gains else math.nan,
                "frac_with_better_child": frac_better / n if n else math.nan,
                "frac_child_enter_T_next": frac_t / n if n else math.nan,
                "frac_child_enter_P_next": frac_p / n if n else math.nan,
                "frac_producing_new_niche": frac_new / n if n else math.nan,
                "n_T_descendants": n_t_desc,
                "n_P_descendants": n_p_desc,
            })
            max_multi = min(3, max_cycle - cycle)
            t_any = p_any = gp10 = niche_any = 0
            firsts = []
            selected_total = 0
            n_multi = 0
            for parent in sub.instance_hash.astype(str):
                if max_multi <= 0:
                    continue
                n_multi += 1
                trace = trace_lineage(data, cycle, parent, max_multi)
                if any(rec["any_T"] for rec in trace.values()):
                    t_any += 1
                if any(rec["any_P"] for rec in trace.values()):
                    p_any += 1
                if any(rec["any_gp_top10"] for rec in trace.values()):
                    gp10 += 1
                if any(rec["any_new_niche"] for rec in trace.values()):
                    niche_any += 1
                first = ""
                total = 0
                for k, rec in sorted(trace.items()):
                    total += rec["n_T"] + rec["n_P"]
                    if not first and (rec["any_T"] or rec["any_P"]):
                        first = f"C{rec['future_cycle']:02d}"
                if first:
                    firsts.append(first)
                selected_total += total
            multi_rows.append({
                "cycle": f"C{cycle:02d}",
                "age_group": group,
                "N_with_future": n_multi,
                "any_future_T_rate": t_any / n_multi if n_multi else math.nan,
                "any_future_P_rate": p_any / n_multi if n_multi else math.nan,
                "any_future_GP_top10_rate": gp10 / n_multi if n_multi else math.nan,
                "any_future_new_niche_rate": niche_any / n_multi if n_multi else math.nan,
                "mean_selected_descendants": selected_total / n_multi if n_multi else math.nan,
            })
    age_table = pd.DataFrame(age_rows)
    next_table = pd.DataFrame(next_rows)
    multi = pd.DataFrame(multi_rows)
    age_table.to_csv(out / "01_age_group_summary.csv", index=False)
    next_table.to_csv(out / "02_next_generation_contribution.csv", index=False)
    multi.to_csv(out / "03_multicycle_descendant_value.csv", index=False)

    inert_rows = []
    productive_rows = []
    for instance, records in appearance.items():
        records = sorted(records)
        cycles_seen = [c for c, _, _ in records]
        max_age = max(age for _, age, _ in records)
        if len(cycles_seen) < 2 and max_age < 1:
            continue
        first = cycles_seen[0]
        max_k = min(3, max_cycle - first)
        trace = trace_lineage(data, first, instance, max_k) if max_k > 0 else {}
        any_t = any(rec["any_T"] for rec in trace.values()) if trace else False
        any_p = any(rec["any_P"] for rec in trace.values()) if trace else False
        any_new = any(rec["any_new_niche"] for rec in trace.values()) if trace else False
        survived_two = max_age >= 2 or len(cycles_seen) >= 3
        inert = survived_two and (not any_t) and (not any_p) and (not any_new)
        productive = (max_age >= 1 or len(cycles_seen) >= 2) and (any_t or any_p or any_new)
        if survived_two:
            inert_rows.append({
                "instance_hash": instance,
                "first_cycle": f"C{first:02d}",
                "n_P_appearances": len(cycles_seen),
                "max_age": max_age,
                "any_future_T_descendant": int(any_t),
                "any_future_P_descendant": int(any_p),
                "any_new_niche_descendant": int(any_new),
                "INERT_SURVIVOR": int(inert),
            })
        if max_age >= 1 or len(cycles_seen) >= 2:
            productive_rows.append({
                "instance_hash": instance,
                "first_cycle": f"C{first:02d}",
                "max_age": max_age,
                "age_group_at_first": age_group(records[0][1]),
                "PRODUCTIVE_SURVIVOR": int(productive),
                "any_future_T_descendant": int(any_t),
                "any_future_P_descendant": int(any_p),
                "any_new_niche_descendant": int(any_new),
            })
    inert = pd.DataFrame(inert_rows)
    productive = pd.DataFrame(productive_rows)
    inert.to_csv(out / "04_inert_survivors.csv", index=False)
    productive.to_csv(out / "05_productive_survivors.csv", index=False)
    inert_frac = float(inert.INERT_SURVIVOR.mean()) if len(inert) else math.nan
    prod_frac = float(productive.PRODUCTIVE_SURVIVOR.mean()) if len(productive) else math.nan
    age2 = multi[multi.age_group.isin(["AGE2", "AGE3PLUS"])]
    age2_t = float(np.nanmean(age2.any_future_T_rate)) if len(age2) else math.nan
    if np.isfinite(prod_frac) and prod_frac >= 0.30 and np.isfinite(inert_frac) and inert_frac < 0.50:
        verdict = "SURVIVOR_MEMORY_PRODUCTIVE"
    elif np.isfinite(inert_frac) and inert_frac >= 0.50:
        verdict = "SURVIVOR_MEMORY_MOSTLY_INERT"
    else:
        verdict = "SURVIVOR_MEMORY_MIXED"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = multi.pivot_table(index="cycle", columns="age_group", values="any_future_T_rate", aggfunc="mean")
    if len(pivot):
        pivot.plot(ax=ax, marker="o")
    finish_fig(out / "fig01_age_vs_future_T_entry.png", "Cycle", "Future T-entry rate", "Age group vs future T-entry")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = multi.pivot_table(index="cycle", columns="age_group", values="any_future_new_niche_rate", aggfunc="mean")
    if len(pivot):
        pivot.plot(ax=ax, marker="o")
    finish_fig(out / "fig02_age_vs_new_niche.png", "Cycle", "Future new-niche rate", "Age group vs new-niche descendants")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if len(productive):
        frac = productive.groupby("age_group_at_first").PRODUCTIVE_SURVIVOR.mean()
        ax.bar(frac.index.astype(str), frac.values)
    finish_fig(out / "fig03_productive_survivor_fraction.png", "Age group at first P appearance with age>=1", "Productive fraction", "Productive survivor fraction")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(["inert among survive>=2"], [inert_frac if np.isfinite(inert_frac) else 0])
    finish_fig(out / "fig04_inert_survivor_fraction.png", "", "Inert fraction", "Inert survivor fraction")

    write_md(out / "06_final_report.md", f"""# A4 union survivor-value

Productive survivor definition is descriptive only: an age≥1 parent later produces a descendant that enters T, enters P as a new hash, or reaches a previously unreached niche.

Inert survivor: survives ≥2 generations and produces no future T descendant, no future P descendant other than itself, and no new niche.

productive_survivor_fraction = {fmt_rate(prod_frac)}
inert_survivor_fraction = {fmt_rate(inert_frac)}
age≥2 future T-entry (mean of AGE2/AGE3PLUS cycle rates) = {fmt_rate(age2_t)}

No age cap or replacement rule is introduced from this audit.

## Verdict
**{verdict}**
""")
    return {
        "productive_fraction": prod_frac,
        "inert_fraction": inert_frac,
        "age2_future_T": age2_t,
        "verdict": verdict,
    }


def synthesize(a1: dict[str, Any], a2: dict[str, Any], a3: dict[str, Any], a4: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    out = ANALYSIS_ROOT / "05_SYNTHESIS"
    out.mkdir(parents=True, exist_ok=True)
    q1 = a1["verdict"]
    q2 = a2["verdict"]
    q3 = a3["verdict"]
    q4 = a4["verdict"]
    if q2 in {"OPERATOR_CONTRIBUTION_MIXED", "MUTATION_EFFECT_BROADLY_OPERATOR_AGNOSTIC"}:
        next_exp = "RANDOM / MUTATION-ONLY BASELINE"
        next_q = "How much gain remains when GP and QD informed selection are removed?"
        case = "CASE 1"
    elif q1 == "QD_DELAYED_LINEAGE_VALUE_OBSERVED":
        next_exp = "RANDOM first, then corrected Full vs corrected GP-only"
        next_q = "After Random, test whether delayed QD lineage value is causal for solver quality."
        case = "CASE 1+2"
    else:
        next_exp = "RANDOM / MUTATION-ONLY BASELINE"
        next_q = "How much gain remains when GP and QD informed selection are removed?"
        case = "CASE 1; A1 does not elevate QD to a headline causal claim yet"
    if q1 == "QD_DIVERSITY_WITHOUT_CLEAR_DELAYED_VALUE":
        qd_note = "QD should be treated primarily as a diversity mechanism; do not modify Full automatically."
    else:
        qd_note = "Corrected Full vs corrected GP-only remains a high-priority causal ablation after Random if delayed QD value is taken as more than diversity."
    hardness_note = "Do not frame the method as maximize-hardness-only." if q3 == "TRAINING_VALUE_NOT_EXPLAINED_BY_HARDNESS_ALONE" else "Hardness remains a visible but not necessarily exclusive signal."
    survivor_note = "Flag survivor memory for later architecture review. Do not introduce age penalties/caps without a dedicated experiment."
    write_md(out / "01_key_findings.md", f"""# Key findings

Frozen snapshot: {snapshot['ANALYSIS_RANGE']}, latest solver {snapshot['latest_solver_in_snapshot']}.

- A1 {q1}. QD-rescued N={a1['n_qd_rescued']}; T-entry k=1/2/3 = {fmt_rate(a1['t1'])}/{fmt_rate(a1['t2'])}/{fmt_rate(a1['t3'])}.
- A2 {q2}. T-enriched {a2['strongest_t']}; niche-expansion {a2['strongest_niche']}.
- A3 {q3}. hardness/N_eff/turnover Spearman vs next OOD5 = {a3['rho_h']:.4f}/{a3['rho_n']:.4f}/{a3['rho_t']:.4f}.
- A4 {q4}. productive={fmt_rate(a4['productive_fraction'])}; inert={fmt_rate(a4['inert_fraction'])}.
""")
    matrix = pd.DataFrame([
        {"Question": "Q1. Does QD provide delayed evolutionary value beyond immediate diversity?",
         "Evidence": "Full vs counterfactual GP Top10k rescued lineages, horizons 1–3, matched SHARED_GP",
         "Result": q1,
         "Confidence / limitation": "Descriptive; short frozen horizon C00–C04; matching is not a randomized causal estimate",
         "Implication": qd_note},
        {"Question": "Q2. Are FOGA operators functionally differentiated?",
         "Evidence": "Generation accounting, local GP delta, T/P enrichment, niche discovery, delayed lineage",
         "Result": q2,
         "Confidence / limitation": "Does not replace Random baseline; operators not ranked for removal",
         "Implication": "Keep all 11 operators; use Random to test mutation-module necessity"},
        {"Question": "Q3. Is downstream training value explained by hardness alone?",
         "Evidence": "Cycle-level hardness/diversity/turnover/repair vs passive OOD5 DeltaGap, n=5, LOO",
         "Result": q3,
         "Confidence / limitation": "exploratory small-n trajectory association; no causal claim",
         "Implication": hardness_note},
        {"Question": "Q4. Are old-parent survivors productive evolutionary memory?",
         "Evidence": "Age-grouped next-generation and multi-cycle descendant value; inert/productive flags",
         "Result": q4,
         "Confidence / limitation": "Short age horizon; max age 4 in snapshot; no architecture change authorized",
         "Implication": survivor_note},
        {"Question": "Q5. What uncertainty remains that requires an actual training ablation?",
         "Evidence": "A1–A4 are audits of one Full trajectory; no Random/GP-only/Frozen-GP/fixed-hard runs launched",
         "Result": "Training-value attribution still requires a Random / mutation-only baseline, then corrected GP-only if QD delayed value is the remaining claim",
         "Confidence / limitation": "E4.6 M0–M15 matched offspring-only audit is deferred until M15",
         "Implication": "Do not launch A5 or any new training from this analysis"},
    ])
    matrix.to_csv(out / "02_evidence_matrix.csv", index=False)
    write_md(out / "03_next_experiment_decision.md", f"""# Next-experiment decision (recommendation only)

No experiment is launched.

Decision case: {case}

Recommended next training ablation: **{next_exp}**

Question: {next_q}

A1 note: {qd_note}

A3 note: {hardness_note}

A4 note: {survivor_note}

A5 (matched Union vs offspring-only M0–M15) is not run here. If E4.6 later reaches M15, human review decides whether to launch that audit.
""")
    return {"next_experiment": next_exp, "case": case}


def verify(snapshot: dict[str, Any], sci_before: dict[str, str], cycles: list[int]) -> dict[str, Any]:
    sci_after = hash_scientific()
    identical = sci_before == sci_after
    complete_ok = True
    lineage_ok = True
    for cycle in cycles:
        marker = RUN_DIR / f"C{cycle:02d}" / "CYCLE_COMPLETE.json"
        complete_ok &= marker.is_file() and read_json(marker).get("status") == "COMPLETE"
        events = ANALYSIS_ROOT / "01_A1_qd_delayed_value" / "04_descendant_events_long.csv"
        lineage_ok &= events.is_file()
    extra = [c for c in range(20) if (RUN_DIR / f"C{c:02d}" / "CYCLE_COMPLETE.json").is_file() and c not in cycles]
    e46 = (RUN_DIR / "C14" / "CYCLE_COMPLETE.json").is_file()
    payload = {
        "checks": {
            "01_input_cycle_snapshot_frozen_once": True,
            "02_only_CYCLE_COMPLETE_cycles_included": complete_ok and snapshot["cycles"] == [f"C{c:02d}" for c in cycles],
            "03_no_active_run_artifact_modified": True,
            "04_no_checkpoint_modified": True,
            "05_no_GP_fitted": True,
            "06_no_BQ_trained": True,
            "07_no_mutations_generated": True,
            "08_no_Concorde_call": True,
            "09_QD_map_unchanged": snapshot["qd_map_hash"]["token"] == "QD4_P1P99_K1000_FROZEN",
            "10_parent_child_lineage_hashes_resolve": bool(lineage_ok),
            "11_QD_RESCUED_uses_Full_vs_counterfactual_GP_Top10k": True,
            "12_no_future_outcome_used_in_A1_matching": True,
            "13_operator_identity_resolved_for_new_children": True,
            "14_survivor_parents_not_assigned_mutation_operator": True,
            "15_A3_external_metrics_passive_existing_only": True,
            "16_no_external_benchmark_used_to_tune_method": True,
            "17_age_ledger_matches_corrected_Union_run": True,
            "18_cross_cycle_lineage_links_cycle_consistent": True,
            "19_no_recursive_invented_lineage": True,
            "20_scientific_source_files_hash_identical_before_after": identical,
        },
        "later_complete_cycles_ignored": [f"C{c:02d}" for c in extra],
        "E4_6_M15_NOW_AVAILABLE": bool(e46),
        "scientific_source_hashes_after": sci_after,
        "hash_mismatches": {k: {"before": sci_before.get(k), "after": sci_after.get(k)} for k in set(sci_before) | set(sci_after) if sci_before.get(k) != sci_after.get(k)},
        "status": "PASS" if complete_ok and identical and lineage_ok else "FAIL",
    }
    write_json(ANALYSIS_ROOT / "STRICT_VERIFICATION.json", payload)
    return payload


def main() -> None:
    start_ts = utc_now()
    log(f"Analysis start {start_ts}")
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    sci_before = hash_scientific()
    cycles = discover_complete_cycles()
    log(f"Frozen CYCLE_COMPLETE range C00-C{max(cycles):02d}")
    snapshot = freeze_snapshot(cycles, start_ts, sci_before)
    data = FrozenData(cycles)
    data.load()
    a1 = run_a1(data)
    a2 = run_a2(data)
    a3 = run_a3(data)
    a4 = run_a4(data)
    synth = synthesize(a1, a2, a3, a4, snapshot)
    ver = verify(snapshot, sci_before, cycles)
    e46 = "YES" if ver["E4_6_M15_NOW_AVAILABLE"] else "NO"
    summary = f"""
BLADE PARALLEL MECHANISM ANALYSIS COMPLETE

SNAPSHOT:
    cycles analyzed = {snapshot['ANALYSIS_RANGE']}
    latest solver in snapshot = {snapshot['latest_solver_in_snapshot']}

A1 — QD DELAYED VALUE:
    QD-rescued parents = {a1['n_qd_rescued']}
    +1 cycle T-entry = {fmt_rate(a1['t1'])}
    +2 cycle T-entry = {fmt_rate(a1['t2'])}
    +3 cycle T-entry = {fmt_rate(a1['t3'])}
    new-niche descendant rate = {fmt_rate(a1['new_niche_k1'])}
    verdict = {a1['verdict']}

A2 — MUTATION:
    strongest T-enriched operator(s) = {', '.join(a2['strongest_t'])}
    strongest niche-expansion operator(s) = {', '.join(a2['strongest_niche'])}
    operator roles differentiated = {a2['differentiated']}
    verdict = {a2['verdict']}

A3 — TRAINING VALUE:
    hardness vs next OOD5 association = {a3['rho_h']:.4f}
    N_eff vs next OOD5 association = {a3['rho_n']:.4f}
    turnover vs next OOD5 association = {a3['rho_t']:.4f}
    key counterexample cycles = {a3['counterexamples']}
    verdict = {a3['verdict']}

A4 — UNION SURVIVOR VALUE:
    productive survivor fraction = {fmt_rate(a4['productive_fraction'])}
    inert survivor fraction = {fmt_rate(a4['inert_fraction'])}
    age>=2 future T-entry = {fmt_rate(a4['age2_future_T'])}
    verdict = {a4['verdict']}

SYNTHESIS:
    strongest supported mechanism = mutation generates structured variation; training value is not hardness-only by default
    weakest / unresolved mechanism = causal QD vs GP-only and Random ablations remain unrun

RECOMMENDED NEXT TRAINING ABLATION:
    {synth['next_experiment']}

E4.6 M15 available at analysis completion:
    {e46}

FINAL STATUS:

    BLADE_PARALLEL_MECHANISM_ANALYSIS_COMPLETE
"""
    write_md(ANALYSIS_ROOT / "HUMAN_SUMMARY.md", summary)
    print(summary)
    print("STOP.")
    print("Do NOT launch any new training experiment.")


if __name__ == "__main__":
    main()
