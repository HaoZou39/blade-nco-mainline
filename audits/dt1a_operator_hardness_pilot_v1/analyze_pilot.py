#!/usr/bin/env python
"""Preregistered descriptive and matched-parent analysis for DT1-A1."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WORK = OUT / "work"
A0 = ROOT / "audits" / "dt1a_operator_hardness_preflight_v1"
RUN = ROOT / "audits" / "S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1"
C05 = RUN / "cycles" / "C05"
TAU = 0.12629708954782198
MATERIAL_DELTA = 0.01
BOOT_REPS = 10_000
STRUCT_BOOT_REPS = 2_000
BOOT_SEED = 20261028
PERM_SEED = 20261029
QD = [
    "normalized_spn_sum_of_lowest_edge_values",
    "normalized_spn_distance_varcoeff",
    "normalized_spn_mst_depth_mean",
    "normalized_spn_mst_dists_median",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}|{BOOT_SEED}".encode()).digest()[:8], "little")


def quantile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite): return np.nan, np.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def bootstrap_operator(frame: pd.DataFrame, operator: str) -> dict[str, float]:
    part = frame[frame.operator_id == operator].sort_values("sample_order").reset_index(drop=True)
    if len(part) != 30: raise RuntimeError(f"{operator} does not have 30 scheduled attempts")
    rng = np.random.default_rng(stable_seed(operator))
    draws = rng.integers(0, len(part), size=(BOOT_REPS, len(part)))
    valid = part.evaluable_child.to_numpy(bool)
    delta = part.delta_h.to_numpy(float)
    positive = part.unconditional_positive.to_numpy(float)
    hard = part.unconditional_hard_hit.to_numpy(float)
    weight = part.analysis_weight.to_numpy(float)
    mean_boot = np.empty(BOOT_REPS); median_boot = np.empty(BOOT_REPS)
    weighted_mean_boot = np.empty(BOOT_REPS); weighted_positive_boot = np.empty(BOOT_REPS); weighted_hard_boot = np.empty(BOOT_REPS)
    positive_boot = positive[draws].mean(axis=1); hard_boot = hard[draws].mean(axis=1)
    for i, index in enumerate(draws):
        values = delta[index][valid[index]]
        mean_boot[i] = values.mean() if len(values) else np.nan
        median_boot[i] = np.median(values) if len(values) else np.nan
        valid_index = index[valid[index]]
        weighted_mean_boot[i] = np.average(delta[valid_index], weights=weight[valid_index]) if len(valid_index) else np.nan
        weighted_positive_boot[i] = np.average(positive[index], weights=weight[index])
        weighted_hard_boot[i] = np.average(hard[index], weights=weight[index])
    result: dict[str, float] = {}
    for name, values in (("mean_delta_h", mean_boot), ("median_delta_h", median_boot),
                         ("unconditional_positive_yield", positive_boot), ("unconditional_hard_hit_rate", hard_boot)):
        low, high = quantile_ci(values); result[f"{name}_ci_low"] = low; result[f"{name}_ci_high"] = high
    for name, values in (("weighted_mean_delta_h", weighted_mean_boot),
                         ("weighted_unconditional_positive_yield", weighted_positive_boot),
                         ("weighted_unconditional_hard_hit_rate", weighted_hard_boot)):
        low, high = quantile_ci(values); result[f"{name}_ci_low"] = low; result[f"{name}_ci_high"] = high
    return result


def holm(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    adjusted = np.empty(len(array), dtype=float)
    running = 0.0
    m = len(array)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * array[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_boot_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(BOOT_REPS, len(values)))
    return quantile_ci(values[index].mean(axis=1))


def sign_flip_p(values: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(BOOT_REPS, len(values)))
    hits = int(np.sum(np.abs(np.mean(signs * values[None, :], axis=1)) >= observed - 1e-15))
    return (hits + 1.0) / (BOOT_REPS + 1.0)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    discord = int(np.sum(a != b))
    if discord == 0: return 1.0
    wins = int(np.sum((a == 1) & (b == 0)))
    return float(binomtest(wins, discord, 0.5, alternative="two-sided").pvalue)


def omnibus_permutation(matrix: np.ndarray, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    observed_means = matrix.mean(axis=0)
    observed = float(np.sum((observed_means - observed_means.mean()) ** 2))
    hits = 0
    for _ in range(BOOT_REPS):
        permuted = np.vstack([row[rng.permutation(matrix.shape[1])] for row in matrix])
        means = permuted.mean(axis=0)
        statistic = float(np.sum((means - means.mean()) ** 2))
        hits += statistic >= observed - 1e-15
    return {"n_parents": int(matrix.shape[0]), "operators": int(matrix.shape[1]),
            "statistic_sum_squared_operator_means": observed,
            "permutations": BOOT_REPS, "p_value": (hits + 1.0) / (BOOT_REPS + 1.0)}


def cluster_fe_slope(frame: pd.DataFrame) -> dict[str, float | int]:
    data = frame[frame.evaluable_child].copy()
    operators = sorted(data.operator_id.unique())
    parents = sorted(data.parent_hash.unique())
    columns = [np.ones(len(data)), data.delta_phi.to_numpy(float)]
    columns.extend((data.operator_id == value).to_numpy(float) for value in operators[1:])
    columns.extend((data.parent_hash == value).to_numpy(float) for value in parents[1:])
    x = np.column_stack(columns); y = data.delta_h.to_numpy(float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]; residual = y - x @ beta
    bread = np.linalg.pinv(x.T @ x); meat = np.zeros((x.shape[1], x.shape[1]))
    for parent in parents:
        mask = data.parent_hash.eq(parent).to_numpy()
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    correction = len(parents) / (len(parents) - 1) * (len(data) - 1) / (len(data) - x.shape[1])
    covariance = correction * bread @ meat @ bread
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    estimate = float(beta[1])
    return {"n": len(data), "parent_clusters": len(parents), "operator_levels": len(operators),
            "coefficient_delta_phi": estimate, "cluster_robust_se": se,
            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            "model": "OLS delta_h ~ delta_phi + operator_FE + parent_FE; parent-cluster sandwich SE"}


def main() -> None:
    evals = pd.read_csv(OUT / "03_true_lbq_evaluation_manifest.csv")
    parents = pd.read_csv(A0 / "03_parent_sampling_manifest.csv").query("pilot30").copy()
    attempts = pd.read_csv(A0 / "04_operator_attempt_manifest.csv").query("pilot30").copy()
    if len(evals) != 360 or len(parents) != 30 or len(attempts) != 330:
        raise RuntimeError("pilot input count mismatch")
    parent_eval = evals[evals.entity_role == "parent"][["entity_hash", "M5_true_LBQ", "M5_log1p_LBQ"]].rename(
        columns={"entity_hash": "parent_hash", "M5_true_LBQ": "parent_h5", "M5_log1p_LBQ": "parent_f5"})
    child_eval = evals[evals.entity_role == "child_attempt"][["sample_order", "parent_hash", "operator_id", "entity_hash", "M5_true_LBQ", "M5_log1p_LBQ"]].rename(
        columns={"entity_hash": "child_hash_eval", "M5_true_LBQ": "child_h5", "M5_log1p_LBQ": "child_f5"})
    paired = attempts.merge(parent_eval, on="parent_hash", validate="many_to_one")
    paired = paired.merge(child_eval, on=["sample_order", "parent_hash", "operator_id"], validate="one_to_one")
    if not bool((paired.child_hash.astype(str) == paired.child_hash_eval.astype(str)).all()):
        raise RuntimeError("child evaluation join mismatch")
    parent_features = parents[["instance_hash", "niche_id", "qd_region", "analysis_weight", *QD]].rename(
        columns={"instance_hash": "parent_hash", "niche_id": "parent_niche_id", **{name: f"parent_{name}" for name in QD}})
    paired = paired.merge(parent_features, on="parent_hash", validate="many_to_one")
    child_hash_list = paired.loc[paired.evaluable_child, "child_hash"].astype(str).tolist()
    child_meta = duckdb.execute(
        "select instance_hash," + ",".join(QD) + " from read_csv_auto(?) where instance_hash in (select unnest(?))",
        [str(C05 / "candidate_manifest.csv"), child_hash_list],
    ).df().rename(columns={"instance_hash": "child_hash", **{name: f"child_{name}" for name in QD}})
    paired = paired.merge(child_meta, on="child_hash", how="left", validate="one_to_one")
    paired["delta_h"] = paired.child_h5 - paired.parent_h5
    paired["delta_f"] = paired.child_f5 - paired.parent_f5
    feature_deltas = []
    for name in QD:
        delta_name = "delta_" + name.removeprefix("normalized_")
        paired[delta_name] = paired[f"child_{name}"] - paired[f"parent_{name}"]
        feature_deltas.append(delta_name)
    paired["delta_phi"] = np.sqrt(np.square(paired[feature_deltas]).sum(axis=1))
    paired["positive_delta_h"] = paired.evaluable_child & paired.delta_h.gt(0)
    paired["material_delta_h"] = paired.evaluable_child & paired.delta_h.gt(MATERIAL_DELTA)
    paired["hard_region_hit"] = paired.evaluable_child & paired.child_h5.ge(TAU)
    paired["unconditional_positive"] = paired.positive_delta_h.astype(int)
    paired["unconditional_hard_hit"] = paired.hard_region_hit.astype(int)
    paired["hgb_residual_log1p_pred_minus_true"] = paired.hgb_predicted_log1p_hardness - paired.child_f5
    paired["niche_migration"] = paired.evaluable_child & paired.child_niche_id.ne(paired.parent_niche_id)
    paired["repaired"] = paired.repair_count.gt(0)
    paired["hard_not_retained"] = paired.hard_region_hit & ~paired.entered_T05
    paired["positive_not_retained"] = paired.positive_delta_h & ~paired.entered_T05
    paired.to_csv(WORK / "paired_attempts.csv", index=False)

    rows: list[dict[str, Any]] = []
    for operator, part in paired.groupby("operator_id", sort=True):
        valid = part[part.evaluable_child]
        valid_weights = valid.analysis_weight.to_numpy(float)
        all_weights = part.analysis_weight.to_numpy(float)
        row: dict[str, Any] = {
            "operator_id": operator, "scheduled_attempts": len(part),
            "accepted_attempts": int(part.proposal_status.eq("accepted").sum()),
            "unique_valid_offspring": int(part.evaluable_child.sum()),
            "rejected_attempts": int((~part.evaluable_child).sum()), "repair_count_sum": int(part.repair_count.sum()),
            "mean_delta_h": float(valid.delta_h.mean()), "median_delta_h": float(valid.delta_h.median()),
            "weighted_mean_delta_h": float(np.average(valid.delta_h, weights=valid_weights)),
            "std_delta_h": float(valid.delta_h.std(ddof=1)), "mean_delta_f": float(valid.delta_f.mean()),
            "median_delta_f": float(valid.delta_f.median()),
            "positive_fraction_conditional_valid": float(valid.positive_delta_h.mean()),
            "unconditional_positive_yield": float(part.unconditional_positive.mean()),
            "weighted_positive_fraction_conditional_valid": float(np.average(valid.positive_delta_h, weights=valid_weights)),
            "weighted_unconditional_positive_yield": float(np.average(part.unconditional_positive, weights=all_weights)),
            "material_fraction_conditional_valid": float(valid.material_delta_h.mean()),
            "unconditional_material_yield": float(part.material_delta_h.mean()),
            "hard_hit_rate_conditional_valid": float(valid.hard_region_hit.mean()),
            "unconditional_hard_hit_rate": float(part.unconditional_hard_hit.mean()),
            "weighted_hard_hit_rate_conditional_valid": float(np.average(valid.hard_region_hit, weights=valid_weights)),
            "weighted_unconditional_hard_hit_rate": float(np.average(part.unconditional_hard_hit, weights=all_weights)),
            "hgb_residual_mean": float(valid.hgb_residual_log1p_pred_minus_true.mean()),
            "hgb_residual_median": float(valid.hgb_residual_log1p_pred_minus_true.median()),
            "hgb_residual_mae": float(valid.hgb_residual_log1p_pred_minus_true.abs().mean()),
            "elite_qd_selected_fraction_scheduled": float(part.entered_T05.mean()),
            "elite_qd_selected_fraction_valid": float(valid.entered_T05.mean()),
            "weighted_elite_qd_selected_fraction_scheduled": float(np.average(part.entered_T05, weights=all_weights)),
            "hard_not_retained_count": int(part.hard_not_retained.sum()),
            "positive_not_retained_count": int(part.positive_not_retained.sum()),
            "mean_delta_phi": float(valid.delta_phi.mean()),
            "niche_migration_fraction_valid": float(valid.niche_migration.mean()),
        }
        row.update(bootstrap_operator(paired, operator)); rows.append(row)
    summary = pd.DataFrame(rows).sort_values("operator_id").reset_index(drop=True)
    summary.to_csv(OUT / "04_operator_paired_results.csv", index=False)

    pairwise: list[dict[str, Any]] = []
    operators = sorted(paired.operator_id.unique())
    for a, b in itertools.combinations(operators, 2):
        left = paired[paired.operator_id == a].set_index("parent_hash")
        right = paired[paired.operator_id == b].set_index("parent_hash")
        common = left.index.intersection(right.index)
        both = left.loc[common].evaluable_child.to_numpy(bool) & right.loc[common].evaluable_child.to_numpy(bool)
        delta_diff = left.loc[common].delta_h.to_numpy(float)[both] - right.loc[common].delta_h.to_numpy(float)[both]
        seed = stable_seed(f"pair|{a}|{b}")
        low, high = paired_boot_ci(delta_diff, seed)
        positive_a = left.loc[common].unconditional_positive.to_numpy(int)
        positive_b = right.loc[common].unconditional_positive.to_numpy(int)
        hard_a = left.loc[common].unconditional_hard_hit.to_numpy(int)
        hard_b = right.loc[common].unconditional_hard_hit.to_numpy(int)
        pairwise.append({
            "operator_a": a, "operator_b": b, "common_valid_parents": int(both.sum()),
            "mean_delta_h_difference_a_minus_b": float(delta_diff.mean()),
            "mean_delta_h_difference_ci_low": low, "mean_delta_h_difference_ci_high": high,
            "delta_h_signflip_p": sign_flip_p(delta_diff, seed + 1),
            "positive_yield_difference_a_minus_b": float((positive_a - positive_b).mean()),
            "positive_mcnemar_p": mcnemar_exact(positive_a, positive_b),
            "hard_hit_difference_a_minus_b": float((hard_a - hard_b).mean()),
            "hard_hit_mcnemar_p": mcnemar_exact(hard_a, hard_b),
        })
    pairwise_frame = pd.DataFrame(pairwise)
    pairwise_frame["delta_h_holm_p"] = holm(pairwise_frame.delta_h_signflip_p.tolist())
    pairwise_frame["positive_holm_p"] = holm(pairwise_frame.positive_mcnemar_p.tolist())
    pairwise_frame["hard_hit_holm_p"] = holm(pairwise_frame.hard_hit_mcnemar_p.tolist())
    pairwise_frame.to_csv(WORK / "pairwise_operator_contrasts.csv", index=False)

    pivot_delta = paired.pivot(index="parent_hash", columns="operator_id", values="delta_h").dropna()
    pivot_positive = paired.pivot(index="parent_hash", columns="operator_id", values="unconditional_positive")
    pivot_hard = paired.pivot(index="parent_hash", columns="operator_id", values="unconditional_hard_hit")
    omnibus = {
        "delta_h_complete_case": omnibus_permutation(pivot_delta.to_numpy(float), PERM_SEED),
        "positive_yield_all_scheduled": omnibus_permutation(pivot_positive.to_numpy(float), PERM_SEED + 1),
        "hard_hit_all_scheduled": omnibus_permutation(pivot_hard.to_numpy(float), PERM_SEED + 2),
    }
    write_json(WORK / "omnibus_tests.json", omnibus)

    structural_rows: list[dict[str, Any]] = []
    for operator, part in paired[paired.evaluable_child].groupby("operator_id", sort=True):
        rho = float(spearmanr(part.delta_phi, part.delta_h).statistic)
        rng = np.random.default_rng(stable_seed("struct|" + operator))
        boot = np.empty(STRUCT_BOOT_REPS)
        values = part.reset_index(drop=True)
        for i in range(STRUCT_BOOT_REPS):
            take = rng.integers(0, len(values), len(values))
            boot[i] = spearmanr(values.delta_phi.iloc[take], values.delta_h.iloc[take]).statistic
        low, high = quantile_ci(boot)
        structural_rows.append({"operator_id": operator, "n": len(part), "spearman_delta_phi_delta_h": rho,
                                "ci_low": low, "ci_high": high})
    structural = pd.DataFrame(structural_rows)
    structural.to_csv(WORK / "structural_correlations.csv", index=False)
    fe = cluster_fe_slope(paired)
    write_json(WORK / "structural_fixed_effect_model.json", fe)

    region = paired[paired.evaluable_child].groupby(["qd_region", "operator_id"], as_index=False).agg(
        n=("delta_h", "size"), mean_delta_h=("delta_h", "mean"), median_delta_h=("delta_h", "median"),
        positive_fraction=("positive_delta_h", "mean"), hard_hit_fraction=("hard_region_hit", "mean"),
    )
    region.to_csv(WORK / "qd_region_descriptive.csv", index=False)
    retention = paired.groupby("operator_id", as_index=False).agg(
        scheduled=("operator_id", "size"), valid=("evaluable_child", "sum"), retained=("entered_T05", "sum"),
        positive=("positive_delta_h", "sum"), positive_not_retained=("positive_not_retained", "sum"),
        hard_hits=("hard_region_hit", "sum"), hard_not_retained=("hard_not_retained", "sum"),
        failures=("evaluable_child", lambda x: int((~x).sum())),
    )
    retention.to_csv(WORK / "selection_retention_summary.csv", index=False)

    best = summary.sort_values("mean_delta_h", ascending=False).iloc[0]
    worst = summary.sort_values("mean_delta_h").iloc[0]
    result = {
        "status": "PASS", "parents": 30, "scheduled_attempts": 330,
        "valid_unique_children": int(paired.evaluable_child.sum()), "failures": int((~paired.evaluable_child).sum()),
        "tau": TAU, "material_delta_h": MATERIAL_DELTA, "bootstrap_replicates": BOOT_REPS,
        "structural_bootstrap_replicates": STRUCT_BOOT_REPS,
        "best_mean_delta_operator": best.operator_id, "best_mean_delta_h": float(best.mean_delta_h),
        "worst_mean_delta_operator": worst.operator_id, "worst_mean_delta_h": float(worst.mean_delta_h),
        "operators_with_positive_mean_delta_h": int(summary.mean_delta_h.gt(0).sum()),
        "operators_with_ci_excluding_zero_positive": int((summary.mean_delta_h_ci_low > 0).sum()),
        "holm_significant_delta_pairs": int(pairwise_frame.delta_h_holm_p.lt(0.05).sum()),
        "holm_significant_positive_pairs": int(pairwise_frame.positive_holm_p.lt(0.05).sum()),
        "holm_significant_hard_pairs": int(pairwise_frame.hard_hit_holm_p.lt(0.05).sum()),
        "omnibus": omnibus, "structural_fixed_effect": fe,
        "total_hard_hits": int(paired.hard_region_hit.sum()),
        "hard_hits_not_retained": int(paired.hard_not_retained.sum()),
        "total_positive_children": int(paired.positive_delta_h.sum()),
        "positive_children_not_retained": int(paired.positive_not_retained.sum()),
        "retained_children": int(paired.entered_T05.sum()),
    }
    write_json(WORK / "analysis_summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
