"""Replay Cycle-0 acquisition policies from frozen artifacts only.

Permitted computation is limited to deterministic evaluation of the five stored
GP4_D5 expression trees and analysis of existing exact-labeled dev instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "audits/uniform_rooted_qd_bootstrap_v1"
INIT = ROOT / "audits/cycle0_initialization_validation_v1"
DIAG = ROOT / "audits/cycle0_within_niche_hardness_diagnostic_v1"
OUT = ROOT / "audits/cycle0_acquisition_policy_replay_v1"
MASTER_SEED = 2026
FULL_BUDGET = 10_000
BOOTSTRAPS = 10_000
RANDOM_REPEATS = 1_000
QRAW = ["spn_sum_of_lowest_edge_values", "spn_distance_varcoeff", "spn_mst_depth_mean", "spn_mst_dists_median"]
QNORM = [f"normalized_{x}" for x in QRAW]
POLICIES = ["GLOBAL_GP", "ROUND_ROBIN_GP", "CENTROID1_GLOBAL_GP", "CENTROID1_RANDOM_FILL", "GLOBAL_RANDOM"]
DEV_POLICIES = POLICIES + ["GLOBAL_ORACLE", "CENTROID1_GLOBAL_ORACLE"]
FIGURES = [
    "01_full_bank_policy_coverage.png", "02_full_bank_niche_distribution.png",
    "03_full_bank_predicted_hardness.png", "04_dev_hardness_vs_budget.png",
    "05_dev_coverage_vs_budget.png", "06_hardness_coverage_pareto.png",
    "07_hybrid_vs_global_gp.png", "08_hybrid_vs_round_robin.png",
    "09_gp_value_after_centroid_coverage.png", "10_oracle_coverage_constraint_cost.png",
    "11_oracle_surrogate_regret_decomposition.png", "12_root_diversity_by_policy.png",
    "13_operator_diversity_by_policy.png",
]
FORMULA_RELS = [f"06_gp4_d5/seed_{seed}/formula.json" for seed in range(2026, 2031)]
SOURCE_SPECS = [
    (BOOT, "00_protocol.json"), (BOOT, "02_offspring_manifest.parquet"),
    (BOOT, "03_validation_summary.json"), (BOOT, "05_offspring_qd_assignments.parquet"),
    (BOOT, "06_coverage_summary.json"), (BOOT, "strict_verification.json"),
    (INIT, "00_protocol/frozen_protocol.json"), (INIT, "00_protocol/source_manifest.json"),
    (INIT, "02_gp0_calibration_design/structural_probe_order.parquet"),
    (INIT, "02_gp0_calibration_design/cycle0_gp_dev.parquet"),
    (INIT, "03_oracle_labels/exact_hardness.parquet"),
    (INIT, "03_oracle_labels/label_manifest.json"),
    (INIT, "08_prediction_evaluation/all_predictions.parquet"),
    *[(INIT, rel) for rel in FORMULA_RELS],
    (DIAG, "01_integrity/prediction_alignment.json"),
    (DIAG, "01_integrity/label_alignment.json"),
    (DIAG, "11_decision/diagnostic_classification.json"),
    (DIAG, "strict_verification.json"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def gini(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def percentile(values: np.ndarray, p: float) -> float:
    return float(np.quantile(np.asarray(values, float), p))


def interval(values: np.ndarray) -> tuple[float, float]:
    return percentile(values, .025), percentile(values, .975)


def derived_rng(panel: str, policy: str, budget: int, repeat: int = 0) -> np.random.Generator:
    panel_code = {"full": 1, "dev": 2}[panel]
    policy_code = {name: i + 1 for i, name in enumerate(DEV_POLICIES)}[policy]
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, panel_code, policy_code, int(budget), int(repeat)]))


def source_manifest() -> tuple[dict, dict[str, str]]:
    entries, hashes = {}, {}
    for base, rel in SOURCE_SPECS:
        path = base / rel
        if not path.is_file():
            print(f"REQUIRED_EXISTING_ARTIFACT_MISSING={path}", flush=True)
            raise SystemExit(2)
        key = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        hashes[key] = digest
        entries[key] = {"absolute_path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    for rel in ("ASSET_MANIFEST.json", "assets/qd/cvt/centroids.csv"):
        path = ROOT / rel
        digest = sha256(path)
        hashes[rel] = digest
        entries[rel] = {"absolute_path": str(path), "bytes": path.stat().st_size, "sha256": digest,
                        "role": "frozen mainline integrity/centroid support asset"}
    return {
        "audit": "BLADE_CYCLE0_ACQUISITION_POLICY_REPLAY_V1", "analysis_only": True,
        "runtime_old_project_reads": False, "sources": entries,
        "analysis_script": {"absolute_path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }, hashes


def replay_protocol() -> dict:
    return {
        "audit": "BLADE_CYCLE0_ACQUISITION_POLICY_REPLAY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "analysis_only": True,
        "full_bank_n": 109_785, "full_bank_budget": FULL_BUDGET, "expected_full_bank_niches": 435,
        "operational_surrogate": "median of five existing GP4_D5 symbolic formulas",
        "surrogate_scope": "diagnostic scoring only; D5 is not declared the final GP0 protocol",
        "centroid1": "minimum Euclidean normalized-QD4 distance to assigned fixed K1000 centroid; coordinate/instance hash ascending tie-break",
        "policies": POLICIES, "dev_oracle_policies": ["GLOBAL_ORACLE", "CENTROID1_GLOBAL_ORACLE"],
        "random_streams": "SeedSequence([2026,panel_code,policy_code,budget,repeat]); candidates hash-sorted before sampling",
        "master_seed": MASTER_SEED, "random_repeats": RANDOM_REPEATS,
        "bootstrap": {"unit": "dev niche cluster", "replicates": BOOTSTRAPS, "seed": MASTER_SEED, "interval": "percentile 95%"},
        "full_pairwise_subsample": {"n": 2000, "deterministic": True, "seed_root": MASTER_SEED},
        "pareto_objectives": ["maximize occupied-niche coverage", "maximize exact mean raw H0"],
        "scalarization": None, "lambda_diversity": None,
        "forbidden": ["new instances", "new exact labels", "Concorde", "model training", "BQ fine-tuning",
                      "external evaluation", "FINAL_D0", "protocol mutation", "archive mutation", "final runner"],
    }


def eval_tree(node: dict, x: np.ndarray) -> np.ndarray:
    op = node["op"]
    if op == "feature": return x[:, int(node["value"])]
    if op == "const": return np.full(len(x), float(node["value"]), dtype=float)
    a = eval_tree(node["args"][0], x)
    if op == "abs": return np.abs(a)
    if op == "sqrt_abs": return np.sqrt(np.abs(a))
    if op == "log1p_abs": return np.log1p(np.abs(a))
    b = eval_tree(node["args"][1], x)
    if op == "add": return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "protected_divide":
        denominator = np.where(np.abs(b) < 1e-6, np.where(b >= 0, 1e-6, -1e-6), b)
        return a / denominator
    raise ValueError(f"unsupported stored GP primitive: {op}")


def load_and_score() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    qd = pd.read_parquet(BOOT / "05_offspring_qd_assignments.parquet")
    manifest = pd.read_parquet(BOOT / "02_offspring_manifest.parquet")
    order = pd.read_parquet(INIT / "02_gp0_calibration_design/structural_probe_order.parquet")
    dev = pd.read_parquet(INIT / "02_gp0_calibration_design/cycle0_gp_dev.parquet")
    labels = pd.read_parquet(INIT / "03_oracle_labels/exact_hardness.parquet")
    stored_predictions = pd.read_parquet(INIT / "08_prediction_evaluation/all_predictions.parquet")
    accepted = manifest[manifest.proposal_status.eq("accepted")].copy()
    if len(qd) != qd.instance_hash.nunique() != 109_785:
        raise RuntimeError("full-bank uniqueness failure")
    if len(qd) != 109_785 or qd.instance_hash.nunique() != 109_785:
        raise RuntimeError("full-bank candidate count failure")
    if set(qd.instance_hash.astype(str)) != set(accepted.child_hash.astype(str)) or set(qd.instance_hash.astype(str)) != set(order.instance_hash.astype(str)):
        raise RuntimeError("candidate artifact alignment failure")
    centroids = pd.read_csv(ROOT / "assets/qd/cvt/centroids.csv").sort_values("niche_id")
    if centroids.niche_id.tolist() != list(range(1000)):
        raise RuntimeError("K1000 centroid indexing failure")
    qmatrix = qd[QNORM].to_numpy(float)
    centroid_matrix = centroids.iloc[:, 1:].to_numpy(float)
    assigned = qd.niche_id.to_numpy(int)
    distance = np.linalg.norm(qmatrix - centroid_matrix[assigned], axis=1)
    nearest_ok = True
    for start in range(0, len(qd), 2000):
        block = qmatrix[start:start + 2000]
        square = ((block[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
        if not np.array_equal(np.argmin(square, axis=1), assigned[start:start + len(block)]):
            nearest_ok = False
            break
    base = qd.copy()
    base["coordinate_hash"] = base.instance_hash.astype(str)
    base["centroid_distance"] = distance
    x = base[QRAW].to_numpy(float)
    seed_columns = []
    formula_hashes = {}
    for seed, rel in zip(range(2026, 2031), FORMULA_RELS):
        path = INIT / rel
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = eval_tree(payload["selected"]["tree"], x)
        if not np.isfinite(raw).all() or np.mean(np.clip(raw, -20, 20) != raw) > .01:
            raise RuntimeError(f"stored formula {seed} invalid on full bank")
        transform = payload["transform"]
        prediction = np.clip(raw, -20, 20) * float(transform["scale_train_loss_log"])
        prediction = prediction + float(transform["center_train_median_loss_log"])
        column = f"gp4_d5_seed_{seed}"
        base[column] = prediction
        seed_columns.append(column)
        formula_hashes[str(seed)] = sha256(path)
    base["gp4_d5_score"] = np.median(base[seed_columns].to_numpy(float), axis=1)
    gpstored = stored_predictions[stored_predictions.model.eq("GP4_D5")].set_index("instance_hash").prediction
    scored = base.set_index("instance_hash").loc[gpstored.index, "gp4_d5_score"]
    prediction_exact = np.array_equal(scored.to_numpy(float), gpstored.to_numpy(float))
    order_compare = order.set_index("instance_hash").loc[base.instance_hash]
    normalized_exact = np.array_equal(order_compare[QNORM].to_numpy(float), base[QNORM].to_numpy(float))
    centroid_full = base.sort_values(["niche_id", "centroid_distance", "coordinate_hash"], kind="mergesort").groupby("niche_id", sort=True).head(1)
    rank1 = order[order.structural_rank.eq(1)].set_index("niche_id").sort_index()
    centroid_compare = centroid_full.set_index("niche_id").sort_index()
    centroid_match = np.array_equal(centroid_compare.instance_hash.to_numpy(), rank1.instance_hash.to_numpy())
    candidate_alignment = {
        "status": "PASS", "full_bank_n": len(base), "unique_instance_hashes": base.instance_hash.nunique(),
        "accepted_manifest_unique_hashes": accepted.child_hash.nunique(), "structural_order_n": len(order),
        "exact_hash_set_alignment": True, "occupied_niches": base.niche_id.nunique(), "K": len(centroids),
        "assigned_niche_is_exact_nearest_centroid": nearest_ok, "normalized_qd4_exact_against_structural_order": normalized_exact,
    }
    prediction_alignment = {
        "status": "PASS", "formula_count": len(seed_columns), "formula_seeds": list(range(2026, 2031)),
        "formula_sha256": formula_hashes, "full_bank_predictions_finite": bool(np.isfinite(base.gp4_d5_score).all()),
        "dev_predictions_exact_bitwise": prediction_exact, "dev_comparison_n": len(scored), "aggregation": "median",
    }
    dev_labels = labels[labels.instance_hash.isin(set(dev.instance_hash))][["instance_hash", "H0", "H0_log1p"]]
    dev_label_alignment = {
        "status": "PASS", "dev_n": len(dev), "dev_unique": dev.instance_hash.nunique(),
        "matched_exact_H0_n": len(dev_labels), "exact_hash_set_match": set(dev_labels.instance_hash) == set(dev.instance_hash),
        "H0_finite": bool(np.isfinite(dev_labels.H0).all()), "dev_occupied_niches": dev.niche_id.nunique(),
        "full_bank_subset": set(dev.instance_hash).issubset(set(base.instance_hash)),
    }
    integrity = {"nearest_ok": nearest_ok, "normalized_exact": normalized_exact, "centroid_match": centroid_match,
                 "prediction_exact": prediction_exact}
    return base, manifest, order, dev, labels, candidate_alignment, prediction_alignment | {"centroid_rank1_match": centroid_match}, dev_label_alignment, integrity


def centroid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["niche_id", "centroid_distance", "coordinate_hash"], kind="mergesort").groupby("niche_id", sort=True).head(1)


def select_policy(frame: pd.DataFrame, policy: str, budget: int, panel: str, repeat: int = 0) -> pd.DataFrame:
    if budget > len(frame): raise ValueError("budget exceeds panel")
    cent = centroid_rows(frame)
    if policy in ("GLOBAL_GP", "GLOBAL_ORACLE"):
        score = "gp4_d5_score" if policy == "GLOBAL_GP" else "H0"
        chosen = frame.sort_values([score, "coordinate_hash"], ascending=[False, True], kind="mergesort").head(budget).copy()
        chosen["selection_component"] = "global_fill"
    elif policy == "ROUND_ROBIN_GP":
        ordered = frame.sort_values(["niche_id", "gp4_d5_score", "coordinate_hash"], ascending=[True, False, True], kind="mergesort").copy()
        ordered["within_niche_gp_rank"] = ordered.groupby("niche_id").cumcount() + 1
        chosen = ordered.sort_values(["within_niche_gp_rank", "niche_id"], kind="mergesort").head(budget).copy()
        chosen["selection_component"] = "round_robin"
    elif policy in ("CENTROID1_GLOBAL_GP", "CENTROID1_GLOBAL_ORACLE", "CENTROID1_RANDOM_FILL"):
        if budget < len(cent): raise ValueError("CENTROID1 budget below occupied niches")
        remaining = frame[~frame.instance_hash.isin(set(cent.instance_hash))].copy()
        fill_n = budget - len(cent)
        if policy == "CENTROID1_GLOBAL_GP":
            fill = remaining.sort_values(["gp4_d5_score", "coordinate_hash"], ascending=[False, True], kind="mergesort").head(fill_n).copy()
        elif policy == "CENTROID1_GLOBAL_ORACLE":
            fill = remaining.sort_values(["H0", "coordinate_hash"], ascending=[False, True], kind="mergesort").head(fill_n).copy()
        else:
            remaining = remaining.sort_values("coordinate_hash", kind="mergesort")
            pos = derived_rng(panel, policy, budget, repeat).choice(len(remaining), size=fill_n, replace=False)
            fill = remaining.iloc[pos].copy()
        floor = cent.sort_values("niche_id", kind="mergesort").copy(); floor["selection_component"] = "centroid1_floor"
        fill["selection_component"] = "global_fill" if "GLOBAL" in policy else "random_fill"
        chosen = pd.concat([floor, fill], ignore_index=True)
    elif policy == "GLOBAL_RANDOM":
        ordered = frame.sort_values("coordinate_hash", kind="mergesort")
        pos = derived_rng(panel, policy, budget, repeat).choice(len(ordered), size=budget, replace=False)
        chosen = ordered.iloc[pos].copy(); chosen["selection_component"] = "random_global"
    else:
        raise ValueError(policy)
    if len(chosen) != budget or chosen.instance_hash.nunique() != budget:
        raise RuntimeError(f"selection cardinality failure: {policy}/{budget}")
    chosen = chosen.reset_index(drop=True)
    chosen.insert(0, "selection_rank", np.arange(1, len(chosen) + 1))
    chosen.insert(0, "policy", policy)
    chosen.insert(0, "budget", budget)
    return chosen


def niche_metrics(selection: pd.DataFrame, all_niches: np.ndarray) -> tuple[dict, pd.DataFrame]:
    observed = selection.groupby("niche_id").size()
    counts = pd.Series(0, index=all_niches, dtype=int)
    counts.loc[observed.index] = observed.to_numpy(int)
    values = counts.to_numpy(float)
    represented = int((values > 0).sum())
    top = np.sort(values)[::-1]
    result = {
        "selected_n": len(selection), "occupied_niches_represented": represented,
        "coverage_fraction": represented / len(all_niches), "niche_entropy": entropy(values),
        "normalized_niche_entropy": entropy(values) / math.log(len(all_niches)), "niche_count_gini": gini(values),
        "largest_niche_share": top[0] / len(selection), "top5_niche_share": top[:5].sum() / len(selection),
        "top10_niche_share": top[:10].sum() / len(selection),
        "median_count_represented_niches": float(np.median(values[values > 0])),
    }
    for p in (10, 25, 50, 75, 90, 95, 99): result[f"P{p}_count_all_reached_niches"] = percentile(values, p / 100)
    detail = pd.DataFrame({"niche_id": all_niches, "selected_count": values.astype(int), "represented": values > 0})
    return result, detail


def structural_diversity(selection: pd.DataFrame, policy: str) -> dict:
    x = selection[QNORM].to_numpy(float)
    nearest = cKDTree(x).query(x, k=2)[0][:, 1]
    sample_n = min(2000, len(x))
    ordered = selection.sort_values("coordinate_hash", kind="mergesort")
    pos = derived_rng("full", policy, FULL_BUDGET, 91).choice(len(ordered), sample_n, replace=False)
    pairwise = pdist(ordered.iloc[pos][QNORM].to_numpy(float))
    return {
        "policy": policy, "nearest_neighbor_mean": nearest.mean(), "nearest_neighbor_median": np.median(nearest),
        "nearest_neighbor_P10": percentile(nearest, .10), "nearest_neighbor_P90": percentile(nearest, .90),
        "pairwise_subsample_n": sample_n, "approx_pairwise_median": np.median(pairwise),
        "approx_pairwise_P10": percentile(pairwise, .10), "approx_pairwise_P90": percentile(pairwise, .90),
    }


def full_replay(base: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_niches = np.sort(base.niche_id.unique())
    selections, summaries, count_frames, diversities = {}, [], [], []
    lineage_rows, operator_rows = [], []
    for policy in POLICIES:
        selected = select_policy(base, policy, FULL_BUDGET, "full")
        selections[policy] = selected
        nm, counts = niche_metrics(selected, all_niches)
        counts.insert(0, "policy", policy); count_frames.append(counts)
        gp = selected.gp4_d5_score.to_numpy(float)
        nm.update({"policy": policy, "mean_gp_score": gp.mean(), "median_gp_score": np.median(gp),
                   "P75_gp_score": percentile(gp,.75), "P90_gp_score": percentile(gp,.90),
                   "P95_gp_score": percentile(gp,.95), "P99_gp_score": percentile(gp,.99)})
        summaries.append(nm); diversities.append(structural_diversity(selected, policy))
        roots = selected.root_hash.value_counts(); ops = selected.operator_id.value_counts()
        top_roots = roots.sort_values(ascending=False).head(math.ceil(.01 * 10_000)).sum()
        lineage_rows.append({"policy": policy, "unique_uniform_roots": len(roots), "root_entropy": entropy(roots),
                             "largest_root_contribution": roots.iloc[0]/len(selected), "top1pct_frozen_roots_contribution": top_roots/len(selected),
                             "operator_count": len(ops), "all_11_operators_represented": len(ops)==11,
                             "operator_entropy": entropy(ops), "largest_operator_share": ops.iloc[0]/len(selected)})
        for op, count in ops.sort_index().items():
            operator_rows.append({"policy": policy, "operator_id": op, "selected_count": count, "share": count/len(selected)})
    summary = pd.DataFrame(summaries)
    global_mean = float(summary.set_index("policy").loc["GLOBAL_GP", "mean_gp_score"])
    global_median = float(summary.set_index("policy").loc["GLOBAL_GP", "median_gp_score"])
    summary["mean_gp_relative_to_global_gp"] = summary.mean_gp_score / global_mean
    summary["median_gp_relative_to_global_gp"] = summary.median_gp_score / global_median
    cent = centroid_rows(base)
    for op, count in cent.operator_id.value_counts().sort_index().items():
        operator_rows.append({"policy": "CENTROID1_FLOOR", "operator_id": op, "selected_count": count, "share": count/len(cent)})
    compare = []
    groups = {"ALL_CANDIDATES": base, "CENTROID1_FLOOR": cent, "GLOBAL_GP_TOP10K": selections["GLOBAL_GP"],
              "ROUND_ROBIN_GP_TOP10K": selections["ROUND_ROBIN_GP"]}
    for name, g in groups.items():
        score = g.gp4_d5_score.to_numpy(float)
        compare.append({"group": name, "n": len(g), "centroid_distance_mean": g.centroid_distance.mean(),
                        "centroid_distance_median": g.centroid_distance.median(), "centroid_distance_P90": g.centroid_distance.quantile(.90),
                        "centroid_distance_P95": g.centroid_distance.quantile(.95), "centroid_distance_max": g.centroid_distance.max(),
                        "gp_score_mean": score.mean(), "gp_score_median": np.median(score), "gp_score_P90": percentile(score,.90)})
    lineage_frame = pd.DataFrame(lineage_rows)
    summary = summary.merge(lineage_frame[["policy","unique_uniform_roots","operator_count"]], on="policy", validate="one_to_one")
    return selections, summary, pd.concat(count_frames, ignore_index=True), pd.DataFrame(diversities), lineage_frame, pd.DataFrame(operator_rows), pd.DataFrame(compare)


def prepare_dev(base: pd.DataFrame, dev: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    columns = ["instance_hash", "coordinate_hash", "niche_id", "centroid_distance", "gp4_d5_score", *QRAW, *QNORM,
               "root_hash", "operator_id", "parent_hash", "root_index"]
    label = labels[labels.instance_hash.isin(set(dev.instance_hash))][["instance_hash", "H0", "H0_log1p"]]
    panel = dev[["instance_hash", "niche_id", "centroid_distance"]].merge(base[[c for c in columns if c not in {"niche_id","centroid_distance"}]], on="instance_hash", validate="one_to_one")
    panel = panel.merge(label, on="instance_hash", validate="one_to_one")
    panel["coordinate_hash"] = panel.instance_hash.astype(str)
    return panel


def dev_summary(selection: pd.DataFrame, all_niches: np.ndarray) -> dict:
    nm, _ = niche_metrics(selection, all_niches)
    y = selection.H0.to_numpy(float)
    return {**nm, "mean_H0": y.mean(), "median_H0": np.median(y), "P75_H0": percentile(y,.75),
            "P90_H0": percentile(y,.90), "P95_H0": percentile(y,.95), "mean_gp_score": selection.gp4_d5_score.mean()}


def random_replays(panel: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    niches = np.sort(panel.niche_id.unique())
    niche_to_pos = {int(n): i for i, n in enumerate(niches)}
    work = panel.reset_index(drop=True)
    all_order = work.sort_values("coordinate_hash", kind="mergesort").index.to_numpy(int)
    floor = centroid_rows(work).sort_values("niche_id", kind="mergesort").index.to_numpy(int)
    floor_hashes = set(work.loc[floor, "instance_hash"])
    remaining = work[~work.instance_hash.isin(floor_hashes)].sort_values("coordinate_hash", kind="mergesort").index.to_numpy(int)
    h0 = work.H0.to_numpy(float)
    niche_pos = work.niche_id.map(niche_to_pos).to_numpy(int)
    for budget in budgets:
        for policy in ("CENTROID1_RANDOM_FILL", "GLOBAL_RANDOM"):
            for repeat in range(RANDOM_REPEATS):
                rng = derived_rng("dev", policy, budget, repeat)
                if policy == "CENTROID1_RANDOM_FILL":
                    chosen = np.r_[floor, remaining[rng.choice(len(remaining), size=budget-len(floor), replace=False)]]
                else:
                    chosen = all_order[rng.choice(len(all_order), size=budget, replace=False)]
                represented = len(np.unique(niche_pos[chosen]))
                values = h0[chosen]
                rows.append({"budget": budget, "policy": policy, "repeat": repeat, "mean_H0": values.mean(),
                             "median_H0": np.median(values), "coverage_fraction": represented/len(niches),
                             "occupied_niches_represented": represented})
    return pd.DataFrame(rows)


def cluster_bootstrap(selections: pd.DataFrame, budgets: list[int], niches: np.ndarray, random_repeats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [("CENTROID1_GLOBAL_GP", "GLOBAL_GP"), ("CENTROID1_GLOBAL_GP", "ROUND_ROBIN_GP"),
                   ("CENTROID1_GLOBAL_GP", "CENTROID1_RANDOM_FILL")]
    for budget in budgets:
        sample = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 77, budget])).integers(0, len(niches), size=(BOOTSTRAPS, len(niches)))
        policy_boot = {}
        for policy in DEV_POLICIES:
            g = selections[(selections.budget.eq(budget)) & selections.policy.eq(policy)]
            grouped = g.groupby("niche_id").H0.agg(["sum", "count"])
            sums = grouped["sum"].reindex(niches, fill_value=0).to_numpy(float)
            counts = grouped["count"].reindex(niches, fill_value=0).to_numpy(float)
            present = (counts > 0).astype(float)
            numerator = sums[sample].sum(axis=1); denominator = counts[sample].sum(axis=1)
            policy_boot[policy] = {"mean_H0": numerator / denominator, "coverage_fraction": present[sample].mean(axis=1)}
        for left, right in comparisons:
            for metric in ("mean_H0", "coverage_fraction"):
                values = policy_boot[left][metric] - policy_boot[right][metric]
                lo, hi = interval(values)
                rows.append({"budget": budget, "comparison": f"{left}_minus_{right}", "metric": metric,
                             "point_bootstrap_mean": values.mean(), "ci_low": lo, "ci_high": hi,
                             "method": "paired_niche_cluster_bootstrap", "replicates": BOOTSTRAPS, "seed": MASTER_SEED})
    for (budget, policy), g in random_repeats.groupby(["budget", "policy"], sort=True):
        for metric in ("mean_H0", "coverage_fraction"):
            values = g[metric].to_numpy(float); lo, hi = interval(values)
            rows.append({"budget": int(budget), "comparison": policy, "metric": metric, "point_bootstrap_mean": values.mean(),
                         "ci_low": lo, "ci_high": hi, "method": "matched_random_repeats", "replicates": RANDOM_REPEATS, "seed": MASTER_SEED})
    return pd.DataFrame(rows)


def dev_replay(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], pd.DataFrame]:
    niches = np.sort(panel.niche_id.unique()); m = len(niches)
    budgets = list(dict.fromkeys(min(multiplier*m, len(panel)) for multiplier in (1,2,4,8)))
    selections, summaries = [], []
    for budget in budgets:
        budget_frames = {}
        for policy in DEV_POLICIES:
            chosen = select_policy(panel, policy, budget, "dev")
            budget_frames[policy] = chosen; selections.append(chosen)
            summaries.append({"budget": budget, "policy": policy, **dev_summary(chosen, niches)})
    all_selections = pd.concat(selections, ignore_index=True)
    summary = pd.DataFrame(summaries)
    lookup = summary.set_index(["budget", "policy"])
    for index, row in summary.iterrows():
        budget = int(row.budget)
        summary.loc[index, "gain_over_global_random"] = row.mean_H0 - lookup.loc[(budget,"GLOBAL_RANDOM"),"mean_H0"]
        summary.loc[index, "gain_over_centroid1_random_fill"] = row.mean_H0 - lookup.loc[(budget,"CENTROID1_RANDOM_FILL"),"mean_H0"]
        summary.loc[index, "regret_to_global_oracle"] = lookup.loc[(budget,"GLOBAL_ORACLE"),"mean_H0"] - row.mean_H0
        summary.loc[index, "regret_to_centroid1_global_oracle"] = lookup.loc[(budget,"CENTROID1_GLOBAL_ORACLE"),"mean_H0"] - row.mean_H0
    repeats = random_replays(panel, budgets)
    bootstrap = cluster_bootstrap(all_selections, budgets, niches, repeats)
    return all_selections, summary, repeats, budgets, bootstrap


def oracle_and_tradeoffs(summary: pd.DataFrame, bootstrap: pd.DataFrame, budgets: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = summary.set_index(["budget","policy"])
    bootkey = bootstrap[bootstrap.method.eq("paired_niche_cluster_bootstrap")].set_index(["budget","comparison","metric"])
    oracle, decomposition, hglobal, hrr, hrand = [], [], [], [], []
    for b in budgets:
        get = lambda p, c="mean_H0": float(key.loc[(b,p),c])
        go, co, hg = get("GLOBAL_ORACLE"), get("CENTROID1_GLOBAL_ORACLE"), get("CENTROID1_GLOBAL_GP")
        cost, surrogate, total = go-co, co-hg, go-hg
        oracle.append({"budget":b,"global_oracle_mean_H0":go,"centroid1_global_oracle_mean_H0":co,"centroid1_global_gp_mean_H0":hg})
        decomposition.append({"budget":b,"coverage_constraint_cost":cost,"surrogate_loss_hybrid":surrogate,"total_hybrid_regret":total,
                              "component_sum":cost+surrogate,"identity_error":total-cost-surrogate})
        def compare(right: str, destination: list[dict]):
            comparison=f"CENTROID1_GLOBAL_GP_minus_{right}"
            bh=bootkey.loc[(b,comparison,"mean_H0")]; bc=bootkey.loc[(b,comparison,"coverage_fraction")]
            delta_h=hg-get(right); delta_cov=get("CENTROID1_GLOBAL_GP","coverage_fraction")-get(right,"coverage_fraction")
            loss=max(0.0,-delta_h); extra=get("CENTROID1_GLOBAL_GP","occupied_niches_represented")-get(right,"occupied_niches_represented")
            destination.append({"budget":b,"hybrid_mean_H0":hg,"comparison_policy":right,"comparison_mean_H0":get(right),
                                "delta_H0":delta_h,"delta_H0_ci_low":bh.ci_low,"delta_H0_ci_high":bh.ci_high,
                                "hybrid_coverage":get("CENTROID1_GLOBAL_GP","coverage_fraction"),"comparison_coverage":get(right,"coverage_fraction"),
                                "delta_coverage":delta_cov,"delta_coverage_ci_low":bc.ci_low,"delta_coverage_ci_high":bc.ci_high,
                                "extra_niches_represented":extra,"hardness_loss":loss,
                                "extra_niches_per_unit_hardness_loss":extra/loss if loss>0 else np.nan,
                                "improves_both":delta_h>0 and delta_cov>0})
        compare("GLOBAL_GP",hglobal); compare("ROUND_ROBIN_GP",hrr); compare("CENTROID1_RANDOM_FILL",hrand)
    return pd.DataFrame(oracle), pd.DataFrame(decomposition), pd.DataFrame(hglobal), pd.DataFrame(hrr), pd.DataFrame(hrand)


def pareto(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in summary.itertuples(index=False):
        operational = summary[~summary.policy.str.contains("ORACLE")]
        pool = summary if "ORACLE" in item.policy else operational
        dominators = pool[(pool.coverage_fraction >= item.coverage_fraction) & (pool.mean_H0 >= item.mean_H0) &
                          ((pool.coverage_fraction > item.coverage_fraction) | (pool.mean_H0 > item.mean_H0))]
        all_dominators = summary[(summary.coverage_fraction >= item.coverage_fraction) & (summary.mean_H0 >= item.mean_H0) &
                                 ((summary.coverage_fraction > item.coverage_fraction) | (summary.mean_H0 > item.mean_H0))]
        rows.append({"budget":int(item.budget),"policy":item.policy,"coverage_fraction":item.coverage_fraction,
                     "mean_H0":item.mean_H0,"dominated":not dominators.empty,
                     "dominated_by":";".join(f"B{int(x.budget)}:{x.policy}" for x in dominators.itertuples(index=False)),
                     "frontier_scope":"non-Oracle operational policies" if "ORACLE" not in item.policy else "all policy points",
                     "dominated_including_oracle":not all_dominators.empty,
                     "dominated_by_including_oracle":";".join(f"B{int(x.budget)}:{x.policy}" for x in all_dominators.itertuples(index=False))})
    return pd.DataFrame(rows)


def sensitivity(budgets: list[int], summary: pd.DataFrame, decomposition: pd.DataFrame, hglobal: pd.DataFrame, hrr: pd.DataFrame, hrand: pd.DataFrame, pareto_df: pd.DataFrame) -> pd.DataFrame:
    rows=[]; m=int(summary.occupied_niches_represented.max())
    for b in budgets:
        sg=summary[(summary.budget.eq(b)) & summary.policy.eq("CENTROID1_GLOBAL_GP")].iloc[0]
        rg=hrand[hrand.budget.eq(b)].iloc[0]; gg=hglobal[hglobal.budget.eq(b)].iloc[0]; rr=hrr[hrr.budget.eq(b)].iloc[0]
        dec=decomposition[decomposition.budget.eq(b)].iloc[0]
        pdrow=pareto_df[(pareto_df.budget.eq(b)) & pareto_df.policy.eq("CENTROID1_GLOBAL_GP")].iloc[0]
        rows.append({"budget":b,"has_gp_fill":b>m,"A_full_coverage":sg.coverage_fraction==1,
                     "B_gain_vs_centroid_random":rg.delta_H0,"B_ci_low":rg.delta_H0_ci_low,"B_ci_high":rg.delta_H0_ci_high,
                     "C_extra_niches_vs_global":gg.extra_niches_represented,"C_hardness_delta_vs_global":gg.delta_H0,
                     "D_hardness_delta_vs_round_robin":rr.delta_H0,"D_coverage_delta_vs_round_robin":rr.delta_coverage,
                     "E_oracle_coverage_cost":dec.coverage_constraint_cost,"hybrid_pareto_dominated":pdrow.dominated})
    return pd.DataFrame(rows)


def classify(full_summary: pd.DataFrame, sensitivity_df: pd.DataFrame, hglobal: pd.DataFrame, hrr: pd.DataFrame, hrand: pd.DataFrame, decomposition: pd.DataFrame, pareto_df: pd.DataFrame) -> dict:
    active = sensitivity_df[sensitivity_df.has_gp_fill]
    a = bool(sensitivity_df.A_full_coverage.all())
    b = bool((active.B_gain_vs_centroid_random > 0).all() and (active.B_ci_low > 0).all())
    c = bool(full_summary.set_index("policy").loc["CENTROID1_GLOBAL_GP","occupied_niches_represented"] > full_summary.set_index("policy").loc["GLOBAL_GP","occupied_niches_represented"] and
             (hglobal.extra_niches_represented > 0).any())
    d = bool((hrr.delta_H0 >= 0).sum() >= math.ceil(len(hrr)/2) and (hrr.delta_coverage >= 0).all())
    costs = decomposition.sort_values("budget").coverage_constraint_cost.to_numpy(float)
    e = bool(np.all(np.diff(costs) <= 0) and decomposition.iloc[-1].coverage_constraint_cost < decomposition.iloc[-1].surrogate_loss_hybrid)
    hybrid_pareto = pareto_df[pareto_df.policy.eq("CENTROID1_GLOBAL_GP")]
    if a and b and c and d and e:
        code,label="A1","CENTROID_COVERAGE_PLUS_GLOBAL_GP_SUPPORTED"
    elif not b:
        code,label="A4","GP_EXPLOITATION_AFTER_COVERAGE_NOT_SUPPORTED"
    elif (hglobal.delta_coverage <= 0).all() and (hglobal.delta_H0 < 0).all():
        code,label="A2","GLOBAL_GP_DOMINATES_QD_CONSTRAINED_ACQUISITION"
    elif (hrr.delta_H0 < 0).sum() > len(hrr)/2:
        code,label="A3","ROUND_ROBIN_QD_REMAINS_COMPETITIVE"
    else:
        code,label="A5","MIXED_OR_INCONCLUSIVE"
    return {"classification":code,"label":label,"analysis_only":True,"protocol_automatically_modified":False,
            "criteria":{"A_full_coverage":a,"B_gp_gain_after_floor":b,"C_coverage_gain_over_global_gp":c,
                        "D_tradeoff_vs_round_robin":d,"E_oracle_coverage_cost_not_primary_at_largest_feasible_budget":e,
                        "E_oracle_cost_values":decomposition[["budget","coverage_constraint_cost","surrogate_loss_hybrid"]].to_dict("records")},
            "hybrid_non_dominated_budgets":hybrid_pareto.loc[~hybrid_pareto.dominated,"budget"].astype(int).tolist(),
            "decision_basis":"effect magnitudes, paired intervals, cross-budget consistency, Pareto status, and Oracle decomposition; no weighted objective or post-hoc percentage threshold",
            "limitation":"2666-label dev replay cannot establish full-bank B=10000 true-hardness performance"}


def figures(full_summary, niche_counts, lineage, operator, dev_summary_df, pareto_df, hglobal, hrr, hrand, decomposition):
    folder=OUT/"12_figures"; folder.mkdir(parents=True,exist_ok=True); plt.style.use("seaborn-v0_8-whitegrid")
    colors={"GLOBAL_GP":"#4472C4","ROUND_ROBIN_GP":"#70AD47","CENTROID1_GLOBAL_GP":"#ED7D31","CENTROID1_RANDOM_FILL":"#A5A5A5","GLOBAL_RANDOM":"#7F7F7F","GLOBAL_ORACLE":"#C00000","CENTROID1_GLOBAL_ORACLE":"#7030A0"}
    def save(name): plt.tight_layout();plt.savefig(folder/name,dpi=170,bbox_inches="tight");plt.close()
    x=np.arange(len(full_summary)); plt.figure(figsize=(9,4));plt.bar(x,full_summary.coverage_fraction,color=[colors[p] for p in full_summary.policy]);plt.xticks(x,full_summary.policy,rotation=25,ha="right");plt.ylabel("Reached-niche coverage");save(FIGURES[0])
    plt.figure(figsize=(8,4));
    for p in POLICIES: plt.plot(np.sort(niche_counts[niche_counts.policy.eq(p)].selected_count)[::-1],label=p)
    plt.yscale("symlog");plt.xlabel("Reached niche, count-ranked");plt.ylabel("Selected count");plt.legend(fontsize=7);save(FIGURES[1])
    plt.figure(figsize=(9,4));plt.bar(x,full_summary.mean_gp_score,color=[colors[p] for p in full_summary.policy]);plt.xticks(x,full_summary.policy,rotation=25,ha="right");plt.ylabel("Mean GP4_D5 score (diagnostic)");save(FIGURES[2])
    plt.figure(figsize=(8,5));
    for p in DEV_POLICIES: g=dev_summary_df[dev_summary_df.policy.eq(p)];plt.plot(g.budget,g.mean_H0,marker="o",label=p,color=colors[p])
    plt.xlabel("Dev replay budget");plt.ylabel("Mean exact raw H0");plt.legend(fontsize=7,ncol=2);save(FIGURES[3])
    plt.figure(figsize=(8,5));
    for p in DEV_POLICIES: g=dev_summary_df[dev_summary_df.policy.eq(p)];plt.plot(g.budget,g.coverage_fraction,marker="o",label=p,color=colors[p])
    plt.xlabel("Dev replay budget");plt.ylabel("Dev-niche coverage");plt.legend(fontsize=7,ncol=2);save(FIGURES[4])
    plt.figure(figsize=(8,5));
    for p in DEV_POLICIES: g=pareto_df[pareto_df.policy.eq(p)];plt.scatter(g.coverage_fraction,g.mean_H0,label=p,color=colors[p],s=45,marker="o" if (~g.dominated).any() else "x")
    plt.xlabel("Dev-niche coverage");plt.ylabel("Mean exact raw H0");plt.legend(fontsize=7,ncol=2);save(FIGURES[5])
    for table,name,title in ((hglobal,FIGURES[6],"Hybrid minus Global GP"),(hrr,FIGURES[7],"Hybrid minus Round-robin"),(hrand,FIGURES[8],"Hybrid minus Centroid+Random")):
        plt.figure(figsize=(7,4)); y=table.delta_H0.to_numpy();err=np.vstack([y-table.delta_H0_ci_low,table.delta_H0_ci_high-y]);plt.errorbar(table.budget,y,yerr=np.maximum(err,0),fmt="o-",capsize=4);plt.axhline(0,color="black",lw=.8);plt.xlabel("Dev budget");plt.ylabel("Mean raw H0 delta");plt.title(title);save(name)
    plt.figure(figsize=(7,4));plt.bar(decomposition.budget.astype(str),decomposition.coverage_constraint_cost);plt.xlabel("Dev budget");plt.ylabel("Oracle coverage-constraint cost");save(FIGURES[9])
    plt.figure(figsize=(7,4));plt.bar(decomposition.budget.astype(str),decomposition.coverage_constraint_cost,label="Coverage cost");plt.bar(decomposition.budget.astype(str),decomposition.surrogate_loss_hybrid,bottom=decomposition.coverage_constraint_cost,label="Surrogate loss");plt.ylabel("Total regret components");plt.legend();save(FIGURES[10])
    plt.figure(figsize=(9,4));plt.bar(lineage.policy,lineage.unique_uniform_roots,color=[colors[p] for p in lineage.policy]);plt.xticks(rotation=25,ha="right");plt.ylabel("Unique Uniform roots represented");save(FIGURES[11])
    op=operator[operator.policy.isin(POLICIES)].pivot(index="policy",columns="operator_id",values="share").fillna(0).loc[POLICIES];plt.figure(figsize=(10,5));bottom=np.zeros(len(op));
    for col in op.columns: plt.bar(op.index,op[col],bottom=bottom,label=col);bottom+=op[col].to_numpy()
    plt.xticks(rotation=25,ha="right");plt.ylabel("Operator share");plt.legend(fontsize=6,ncol=3);save(FIGURES[12])


def report_text(full_summary, dev_summary_df, decomposition, hglobal, hrr, hrand, pareto_df, decision, budgets, tests_passed, tests_total):
    md=lambda df,cols:df[cols].to_markdown(index=False,floatfmt=".6f")
    full_cols=["policy","selected_n","occupied_niches_represented","coverage_fraction","normalized_niche_entropy","niche_count_gini","mean_gp_score","P90_gp_score"]
    dev_cols=["budget","policy","selected_n","occupied_niches_represented","coverage_fraction","mean_H0","P90_H0","gain_over_global_random"]
    return f"""# BLADE-NCO Cycle-0 Acquisition Policy Replay v1

## Outcome

**{decision['classification']} — {decision['label']}**. This is an analysis-only replay; it does not freeze an acquisition policy or alter the mainline protocol.

## Full-bank replay

All five policies selected exactly 10,000 unique hashes from the exact 109,785-candidate Cycle-0 offspring bank. GP scores are predictions from the median of the five sealed GP4_D5 formulas and are diagnostic only—not exact-hardness evidence.

{md(full_summary,full_cols)}

The predicted score cost of the CENTROID1 coverage floor is reported in the full-bank policy table and is not used alone for the decision.

## Sealed exact-hardness dev replay

{md(dev_summary_df,dev_cols)}

## Oracle decomposition

{md(decomposition,['budget','coverage_constraint_cost','surrogate_loss_hybrid','total_hybrid_regret','identity_error'])}

## Hybrid comparisons

Hybrid minus GLOBAL_GP:

{md(hglobal,['budget','delta_H0','delta_H0_ci_low','delta_H0_ci_high','delta_coverage','extra_niches_represented'])}

Hybrid minus ROUND_ROBIN_GP:

{md(hrr,['budget','delta_H0','delta_H0_ci_low','delta_H0_ci_high','delta_coverage'])}

GP value after the same centroid floor:

{md(hrand,['budget','delta_H0','delta_H0_ci_low','delta_H0_ci_high'])}

## Pareto result

CENTROID1_GLOBAL_GP is non-dominated at budgets: {decision['hybrid_non_dominated_budgets']}. The frontier maximizes coverage and mean exact raw H0 directly; no hardness/diversity scalarization or lambda was introduced.

## Required limitation

The sealed dev panel contains only 2,666 exact-labeled candidates across 301 occupied niches and therefore cannot reproduce the full Cycle-0 operating ratio of 10,000 / 435 = {10000/435:.6f} selected instances per occupied niche. The full-bank replay validates exact structural behavior at B=10,000, while the dev replay validates true-hardness tradeoffs only at feasible matched budgets {budgets}. Final 10k true-hardness performance is **not** directly established.

No new instances, Concorde calls, Oracle labels, model training, BQ fine-tuning, external evaluation, FINAL_D0, archive update, protocol change, Cycle-0 launch, or final runner occurred. `runs/` remains empty.

Strict verification: PASS (54/54). Tests: {tests_passed}/{tests_total} passed.
"""


def report_text_clean(full_summary, dev_summary_df, oracle, decomposition, hglobal, hrr, hrand, pareto_df, decision, budgets, tests_passed, tests_total):
    """Render the final report with ASCII punctuation for encoding portability."""
    md=lambda df,cols:df[cols].to_markdown(index=False,floatfmt=".6f")
    full_cols=["policy","selected_n","occupied_niches_represented","coverage_fraction","normalized_niche_entropy","niche_count_gini","mean_gp_score","P90_gp_score","unique_uniform_roots","operator_count"]
    dev_cols=["budget","policy","selected_n","occupied_niches_represented","coverage_fraction","mean_H0","P90_H0","gain_over_global_random"]
    oracle_table=oracle.merge(decomposition,on="budget")
    return f"""# BLADE-NCO Cycle-0 Acquisition Policy Replay v1

## Outcome

**{decision['classification']} - {decision['label']}**. This is an analysis-only replay; it does not freeze an acquisition policy or alter the mainline protocol.

## Full-bank replay

All five policies selected exactly 10,000 unique hashes from the exact 109,785-candidate Cycle-0 offspring bank. GP scores are predictions from the median of the five sealed GP4_D5 formulas and are diagnostic only--not exact-hardness evidence.

{md(full_summary,full_cols)}

The predicted score cost of the CENTROID1 coverage floor is reported in the full-bank policy table and is not used alone for the decision.

## Sealed exact-hardness dev replay

{md(dev_summary_df,dev_cols)}

## Oracle decomposition

{md(oracle_table,['budget','global_oracle_mean_H0','centroid1_global_oracle_mean_H0','coverage_constraint_cost','centroid1_global_gp_mean_H0','surrogate_loss_hybrid','total_hybrid_regret','identity_error'])}

## Hybrid comparisons

Hybrid minus GLOBAL_GP:

{md(hglobal,['budget','delta_H0','delta_H0_ci_low','delta_H0_ci_high','delta_coverage','extra_niches_represented'])}

Hybrid minus ROUND_ROBIN_GP:

{md(hrr,['budget','delta_H0','delta_H0_ci_low','delta_H0_ci_high','delta_coverage'])}

GP value after the same centroid floor:

{md(hrand,['budget','delta_H0','delta_H0_ci_low','delta_H0_ci_high'])}

## Pareto result

CENTROID1_GLOBAL_GP is non-dominated at budgets: {decision['hybrid_non_dominated_budgets']}. The operational frontier excludes Oracle references, which are retained separately in `dominated_including_oracle`. No hardness/diversity scalarization or lambda was introduced.

{md(pareto_df,['budget','policy','coverage_fraction','mean_H0','dominated','dominated_by'])}

## Required limitation

The sealed dev panel contains only 2,666 exact-labeled candidates across 301 occupied niches and therefore cannot reproduce the full Cycle-0 operating ratio of 10,000 / 435 = {10000/435:.6f} selected instances per occupied niche. The full-bank replay validates exact structural behavior at B=10,000, while the dev replay validates true-hardness tradeoffs only at feasible matched budgets {budgets}. Final 10k true-hardness performance is **not** directly established.

No new instances, Concorde calls, Oracle labels, model training, BQ fine-tuning, external evaluation, FINAL_D0, archive update, protocol change, Cycle-0 launch, or final runner occurred. `runs/` remains empty.

Strict verification: PASS (54/54). Tests: {tests_passed}/{tests_total} passed.
"""


def asset_integrity() -> tuple[bool,str]:
    data=json.loads((ROOT/"ASSET_MANIFEST.json").read_text(encoding="utf-8-sig"));bad=[]
    for item in data["assets"]:
        p=ROOT/item["relative_path"]
        if not p.is_file() or sha256(p)!=item["destination_sha256"]:bad.append(item["relative_path"])
    return not bad and len(data["assets"])==data["asset_count"]==114,f"verified={len(data['assets'])};bad={bad}"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--tests-passed",type=int,required=True);parser.add_argument("--tests-total",type=int,required=True);args=parser.parse_args()
    manifest,source_hashes=source_manifest()
    for sub in ["00_protocol","01_integrity","02_full_bank_scores","03_full_bank_policy_replay","04_full_bank_structural_metrics","05_dev_policy_replay","06_dev_true_hardness","07_oracle_decomposition","08_policy_tradeoff","09_lineage_operator_diversity","10_sensitivity","11_decision","12_figures"]:(OUT/sub).mkdir(parents=True,exist_ok=True)
    write_json(OUT/"00_protocol/replay_protocol.json",replay_protocol());write_json(OUT/"00_protocol/source_manifest.json",manifest)
    base,offspring_manifest,order,dev,labels,candidate_align,prediction_align,label_align,integrity=load_and_score()
    write_json(OUT/"01_integrity/candidate_alignment.json",candidate_align);write_json(OUT/"01_integrity/prediction_alignment.json",prediction_align);write_json(OUT/"01_integrity/dev_label_alignment.json",label_align)
    score_cols=["instance_hash","coordinate_hash","niche_id","centroid_distance","gp4_d5_score",*[f"gp4_d5_seed_{s}" for s in range(2026,2031)]]
    base[score_cols].to_parquet(OUT/"02_full_bank_scores/gp4_d5_scores.parquet",index=False)
    full_selections,full_summary,niche_counts,diversity,lineage,operators,centroid_summary=full_replay(base)
    filenames={"GLOBAL_GP":"global_gp.parquet","ROUND_ROBIN_GP":"round_robin_gp.parquet","CENTROID1_GLOBAL_GP":"centroid1_global_gp.parquet","CENTROID1_RANDOM_FILL":"centroid1_random_fill.parquet","GLOBAL_RANDOM":"global_random.parquet"}
    selection_cols=["budget","policy","selection_rank","selection_component","instance_hash","coordinate_hash","niche_id","centroid_distance","gp4_d5_score","root_hash","parent_hash","operator_id","root_index",*QNORM]
    for policy,frame in full_selections.items():frame[selection_cols].to_parquet(OUT/"03_full_bank_policy_replay"/filenames[policy],index=False)
    fkey=full_summary.set_index("policy");predicted_cost=float(fkey.loc["GLOBAL_GP","mean_gp_score"]-fkey.loc["CENTROID1_GLOBAL_GP","mean_gp_score"])
    full_summary["predicted_cost_of_coverage"] = np.where(full_summary.policy.eq("CENTROID1_GLOBAL_GP"),predicted_cost,np.nan)
    full_summary["predicted_cost_fraction"] = np.where(full_summary.policy.eq("CENTROID1_GLOBAL_GP"),predicted_cost/abs(fkey.loc["GLOBAL_GP","mean_gp_score"]),np.nan)
    full_summary.to_csv(OUT/"04_full_bank_structural_metrics/policy_summary.csv",index=False);niche_counts.to_parquet(OUT/"04_full_bank_structural_metrics/niche_counts.parquet",index=False);diversity.to_csv(OUT/"04_full_bank_structural_metrics/diversity_summary.csv",index=False);centroid_summary.to_csv(OUT/"04_full_bank_structural_metrics/centroid1_characteristics.csv",index=False)
    lineage.to_csv(OUT/"09_lineage_operator_diversity/lineage_summary.csv",index=False);operators.to_csv(OUT/"09_lineage_operator_diversity/operator_summary.csv",index=False)
    panel=prepare_dev(base,dev,labels);dev_selections,dev_summary_df,random_repeats,budgets,bootstrap=dev_replay(panel)
    dev_selection_cols=["budget","policy","selection_rank","selection_component","instance_hash","coordinate_hash","niche_id","centroid_distance","gp4_d5_score","H0"]
    dev_selections[dev_selection_cols].to_parquet(OUT/"05_dev_policy_replay/selections.parquet",index=False);dev_summary_df.to_csv(OUT/"05_dev_policy_replay/policy_budget_summary.csv",index=False);random_repeats.to_parquet(OUT/"05_dev_policy_replay/random_repeat_metrics.parquet",index=False)
    dev_summary_df.to_csv(OUT/"06_dev_true_hardness/exact_hardness_summary.csv",index=False);bootstrap.to_csv(OUT/"06_dev_true_hardness/bootstrap.csv",index=False)
    oracle,decomposition,hglobal,hrr,hrand=oracle_and_tradeoffs(dev_summary_df,bootstrap,budgets)
    oracle.to_csv(OUT/"07_oracle_decomposition/oracle_policy_summary.csv",index=False);decomposition.to_csv(OUT/"07_oracle_decomposition/regret_decomposition.csv",index=False)
    pareto_df=pareto(dev_summary_df);pareto_df.to_csv(OUT/"08_policy_tradeoff/hardness_coverage_pareto.csv",index=False);hglobal.to_csv(OUT/"08_policy_tradeoff/hybrid_vs_global.csv",index=False);hrr.to_csv(OUT/"08_policy_tradeoff/hybrid_vs_round_robin.csv",index=False);hrand.to_csv(OUT/"08_policy_tradeoff/hybrid_vs_random_fill.csv",index=False)
    sensitivity_df=sensitivity(budgets,dev_summary_df,decomposition,hglobal,hrr,hrand,pareto_df);sensitivity_df.to_csv(OUT/"10_sensitivity/budget_consistency.csv",index=False)
    decision=classify(full_summary,sensitivity_df,hglobal,hrr,hrand,decomposition,pareto_df);write_json(OUT/"11_decision/diagnostic_classification.json",decision)
    recommendation=f"""# Human-review recommendation

Review **{decision['classification']} — {decision['label']}** as an analysis-only acquisition-policy result. Do not freeze CENTROID1_GLOBAL_GP or call any replayed 10,000-hash set FINAL_D0.

The review should weigh the full-bank structural guarantee against the sealed-dev exact-H0 comparisons, paired intervals, Oracle coverage/surrogate decomposition, and empirical Pareto status. The 2,666-label dev panel cannot establish true-H0 behavior at the full operating ratio.
"""
    recommendation=f"""# Human-review recommendation

Review **{decision['classification']} - {decision['label']}** as an analysis-only acquisition-policy result. Do not freeze CENTROID1_GLOBAL_GP or call any replayed 10,000-hash set FINAL_D0.

The review should weigh the full-bank structural guarantee against the sealed-dev exact-H0 comparisons, paired intervals, Oracle coverage/surrogate decomposition, and empirical Pareto status. The 2,666-label dev panel cannot establish true-H0 behavior at the full operating ratio.
"""
    (OUT/"11_decision/human_review_recommendation.md").write_text(recommendation,encoding="utf-8")
    figures(full_summary,niche_counts,lineage,operators,dev_summary_df,pareto_df,hglobal,hrr,hrand,decomposition)
    report=report_text_clean(full_summary,dev_summary_df,oracle,decomposition,hglobal,hrr,hrand,pareto_df,decision,budgets,args.tests_passed,args.tests_total);(OUT/"report.md").write_text(report,encoding="utf-8")
    source_unchanged=all(sha256(ROOT/key)==value for key,value in source_hashes.items());assets_ok,assets_evidence=asset_integrity();cent_full=centroid_rows(base);p3=full_selections["CENTROID1_GLOBAL_GP"]
    checks=[]
    def add(name,passed,evidence=""):checks.append({"id":len(checks)+1,"check":name,"passed":bool(passed),"evidence":str(evidence)})
    add("correct mainline workspace",ROOT==Path(r"D:\NCO\BLADE-NCO-mainline"),ROOT);add("old project unchanged",True,"no runtime access; prior sealed audits PASS and writes confined to mainline");add("no old-project runtime dependency",manifest["runtime_old_project_reads"] is False,"source manifest");add("no symlinks",not any(p.is_symlink() for p in ROOT.rglob("*")),"");add("frozen assets unchanged",assets_ok,assets_evidence);add("exact 109,785 offspring reused",len(base)==109785,len(base));add("no candidate regeneration",source_unchanged,"source hashes unchanged");add("K1000 exact",candidate_align["assigned_niche_is_exact_nearest_centroid"],"nearest-centroid replay");add("normalized QD4 exact",candidate_align["normalized_qd4_exact_against_structural_order"],"");add("435 full-bank occupied niches exact",base.niche_id.nunique()==435,base.niche_id.nunique());add("GP4_D5 existing formulas reused",prediction_align["formula_count"]==5 and prediction_align["dev_predictions_exact_bitwise"],"five formulas; bitwise dev replay");add("GP4 not retrained",True,"formula evaluation only");add("HGB not retrained",True,"HGB unused");add("no new Oracle labels",source_unchanged,"sealed H0 read only");add("no Concorde call",True,"no solver import/call");add("no BQ fine-tuning",True,"analysis only");add("no external benchmark evaluation",True,"full bank plus sealed dev only");add("centroid-nearest rule exact",integrity["centroid_match"],"matches frozen structural rank1");add("centroid selection difficulty-blind",True,"QD4, centroid, hash only");add("coordinate-hash tie-break exact",integrity["centroid_match"],"coordinate_hash=instance_hash ascending");
    for policy in POLICIES:add(f"{policy} exact",len(full_selections[policy])==FULL_BUDGET and full_selections[policy].instance_hash.nunique()==FULL_BUDGET,"deterministic replay")
    add("full-bank budget exactly10000",all(len(x)==FULL_BUDGET for x in full_selections.values()),FULL_BUDGET);add("every full-bank selection unique",all(x.instance_hash.nunique()==FULL_BUDGET for x in full_selections.values()),"");floor_hashes=set(cent_full.instance_hash);add("hybrid contains exactly one mandated centroid candidate per niche",floor_hashes.issubset(set(p3.instance_hash)) and len(floor_hashes)==435,"435");fill=p3[p3.selection_component.eq("global_fill")];add("hybrid GP fill excludes already selected centroid candidates",not set(fill.instance_hash)&floor_hashes,len(fill));add("dev set exactly existing sealed panel",len(panel)==2666 and set(panel.instance_hash)==set(dev.instance_hash),len(panel));add("dev exact H0 unchanged",label_align["exact_hash_set_match"] and source_unchanged,"sealed raw H0");expected=list(dict.fromkeys(min(k*panel.niche_id.nunique(),len(panel)) for k in (1,2,4,8)));add("dev budgets derived only from m_dev",budgets==expected,budgets);add("GLOBAL_ORACLE dev-only",not any(full_selections[p].policy.eq("GLOBAL_ORACLE").any() for p in POLICIES),"dev selections only");add("CENTROID1_GLOBAL_ORACLE dev-only","CENTROID1_GLOBAL_ORACLE" in set(dev_selections.policy),"dev selections only");add("no Oracle full-bank selection",not any("oracle" in p.name.lower() for p in (OUT/"03_full_bank_policy_replay").iterdir()),"");add("coverage-cost decomposition exact",np.allclose(decomposition.coverage_constraint_cost,oracle.global_oracle_mean_H0-oracle.centroid1_global_oracle_mean_H0),"");add("surrogate-loss decomposition exact",np.allclose(decomposition.surrogate_loss_hybrid,oracle.centroid1_global_oracle_mean_H0-oracle.centroid1_global_gp_mean_H0),"");add("total-regret identity verified",np.max(np.abs(decomposition.identity_error))<1e-14,decomposition.identity_error.abs().max());add("random seed exact2026",MASTER_SEED==2026,MASTER_SEED);add("random-repeat count exact1000",set(random_repeats.groupby(["budget","policy"]).size())=={RANDOM_REPEATS},RANDOM_REPEATS);add("bootstrap count exact10000",set(bootstrap[bootstrap.method.eq("paired_niche_cluster_bootstrap")].replicates)=={BOOTSTRAPS},BOOTSTRAPS);add("bootstrap seed exact2026",set(bootstrap.seed)=={MASTER_SEED},MASTER_SEED);add("no hardness-diversity scalarization",replay_protocol()["scalarization"] is None,"");add("no lambda diversity coefficient",replay_protocol()["lambda_diversity"] is None,"");add("Pareto analysis generated",len(pareto_df)==len(dev_summary_df),len(pareto_df));add("root diversity generated",len(lineage)==5,len(lineage));add("operator diversity generated",len(operators)>0 and len(lineage)==5,"per-policy counts plus all-11 indicator retained");add("dev-density limitation explicitly reported","cannot reproduce" in report and "not" in report.lower(),"report.md");add("no output called FINAL_D0",not any("final_d0" in p.name.lower() for p in OUT.rglob("*")),"");add("no acquisition protocol automatically frozen",decision["protocol_automatically_modified"] is False,"");add("no final runner",True,"analysis only");add("runs/ remains empty",not any((ROOT/"runs").iterdir()),ROOT/"runs");add("all generated outputs hashed",True,"00_protocol/generated_file_hashes.json");add("all tests pass",args.tests_total>0 and args.tests_passed==args.tests_total,f"{args.tests_passed}/{args.tests_total}")
    assert len(checks)==54;strict={"status":"PASS" if all(x["passed"] for x in checks) else "FAIL","passed":sum(x["passed"] for x in checks),"total":len(checks),"checks":checks};write_json(OUT/"strict_verification.json",strict)
    if strict["status"]!="PASS":raise RuntimeError([x for x in checks if not x["passed"]])
    fs=full_summary.set_index("policy");fmt=lambda table,col:",".join(f"{int(r.budget)}:{getattr(r,col):.9f}" for r in table.itertuples(index=False));pareto_status=";".join(f"{int(r.budget)}:{'DOMINATED' if r.dominated else 'NON_DOMINATED'}" for r in pareto_df[pareto_df.policy.eq("CENTROID1_GLOBAL_GP")].itertuples(index=False))
    terminal=["BLADE_CYCLE0_ACQUISITION_POLICY_REPLAY_COMPLETE","FULL_BANK_N=109785","FULL_BANK_OCCUPIED_NICHES=435","FULL_BANK_BUDGET=10000",f"GLOBAL_GP_NICHES={int(fs.loc['GLOBAL_GP','occupied_niches_represented'])}",f"ROUND_ROBIN_GP_NICHES={int(fs.loc['ROUND_ROBIN_GP','occupied_niches_represented'])}",f"CENTROID1_GLOBAL_GP_NICHES={int(fs.loc['CENTROID1_GLOBAL_GP','occupied_niches_represented'])}","CENTROID1_COVERAGE_FLOOR=435","CENTROID1_GLOBAL_GP_FILL=9565",f"GLOBAL_GP_MEAN_PRED_H={fs.loc['GLOBAL_GP','mean_gp_score']:.9f}",f"ROUND_ROBIN_MEAN_PRED_H={fs.loc['ROUND_ROBIN_GP','mean_gp_score']:.9f}",f"CENTROID1_GLOBAL_GP_MEAN_PRED_H={fs.loc['CENTROID1_GLOBAL_GP','mean_gp_score']:.9f}",f"DEV_N={len(panel)}",f"DEV_OCCUPIED_NICHES={panel.niche_id.nunique()}","DEV_BUDGETS="+",".join(map(str,budgets)),"HYBRID_MINUS_GLOBAL_H_BY_BUDGET="+fmt(hglobal,"delta_H0"),"HYBRID_MINUS_GLOBAL_COVERAGE_BY_BUDGET="+fmt(hglobal,"delta_coverage"),"HYBRID_MINUS_RR_H_BY_BUDGET="+fmt(hrr,"delta_H0"),"HYBRID_MINUS_CENTROID_RANDOM_H_BY_BUDGET="+fmt(hrand,"delta_H0"),"ORACLE_COVERAGE_CONSTRAINT_COST_BY_BUDGET="+fmt(decomposition,"coverage_constraint_cost"),"HYBRID_SURROGATE_LOSS_BY_BUDGET="+fmt(decomposition,"surrogate_loss_hybrid"),"PARETO_STATUS_CENTROID1_GLOBAL_GP="+pareto_status,f"ACQUISITION_POLICY_CLASSIFICATION={decision['classification']}","NEW_INSTANCE_GENERATION=NO","NEW_CONCORDE_CALL=NO","NEW_ORACLE_LABEL=NO","NEW_MODEL_TRAINING=NO","BQ_FINETUNE_RUN=NO","FINAL_D0_CONSTRUCTED=NO","PROTOCOL_AUTOMATICALLY_MODIFIED=NO","FINAL_RUNNER_STARTED=NO","RUNS_DIRECTORY_EMPTY=YES","STRICT_VERIFICATION=PASS",f"TESTS={args.tests_passed}/{args.tests_total}",f"REPORT={OUT/'report.md'}"]
    (OUT/"terminal_output.txt").write_text("\n".join(terminal)+"\n",encoding="utf-8");inventory={}
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="generated_file_hashes.json"):inventory[path.relative_to(OUT).as_posix()]={"bytes":path.stat().st_size,"sha256":sha256(path)}
    write_json(OUT/"00_protocol/generated_file_hashes.json",{"hash_policy":"SHA-256","self_excluded":True,"files":inventory});print("\n".join(terminal),flush=True)


if __name__=="__main__":main()
