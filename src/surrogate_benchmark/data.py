"""S2-A data access: historical B_s, P0_clean_s, O_s, S1.1 references."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
FULL = ROOT / "runs" / "blade_v2_full_seed2026"
S1 = ROOT / "audits" / "S1_gp4_surrogate_behavior_M0_M15_v1"
S1_1 = ROOT / "audits" / "S1_1_gp4_oracle_enrichment_ceiling_v1"
FEATURES = [
    "spn_sum_of_lowest_edge_values",
    "spn_distance_varcoeff",
    "spn_mst_depth_mean",
    "spn_mst_dists_median",
]
CYCLES = list(range(15))
SEEDS = [2026, 2027, 2028, 2029, 2030]
P0_MEMBER = ROOT / "runs" / "audits" / "gp4_calibration_sample_size_v1" / "P0_MEMBERSHIP_AND_FEATURES.csv"
P0_HARDNESS = S1 / "06_p0_current_hardness.parquet"
RF_NOT_RUN = "NOT_RUN_RF_CONFIG_NOT_FROZEN"
PHASES = {
    "EARLY_C00-C04": list(range(0, 5)),
    "MID_C05-C09": list(range(5, 10)),
    "LATE_C10-C14": list(range(10, 15)),
}


def membership_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def read_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError, Exception):
        try:
            import duckdb
        except ModuleNotFoundError:
            import sys
            sys.path.insert(0, str(ROOT / "vendor"))
            import duckdb
        quoted = path.as_posix().replace("'", "''")
        projection = "*" if columns is None else ", ".join(f'"{name}"' for name in columns)
        return duckdb.sql(f"SELECT {projection} FROM read_parquet('{quoted}', union_by_name=true)").df()


def cycle_dir(s: int) -> Path:
    return FULL / "cycles" / f"C{s:02d}"


def b_meta_path(s: int) -> Path:
    if s == 0:
        return cycle_dir(0) / "02_gp" / "B0_selection" / "B0.parquet"
    return cycle_dir(s) / "02_gp" / "B_selection_and_refresh" / "B_s.parquet"


def b_labeled_path(s: int) -> Path:
    if s == 0:
        return cycle_dir(0) / "07_oracle" / "B0" / "labeled.parquet"
    return cycle_dir(s) / "02_gp" / "B_selection_and_refresh" / "B_s.parquet"


def gp_joblib_path(s: int) -> Path:
    return cycle_dir(s) / "02_gp" / "training" / "gp4_ensemble.joblib"


def offspring_path(s: int) -> Path:
    if s == 0:
        return cycle_dir(0) / "05_scoring" / "O0_scored.parquet"
    return cycle_dir(s) / "05_scoring" / "offspring_scored.parquet"


def load_b_s(s: int) -> pd.DataFrame:
    labeled = read_frame(b_labeled_path(s))
    labeled["instance_hash"] = labeled["instance_hash"].astype(str)
    missing = [c for c in FEATURES + ["current_hardness", "instance_hash"] if c not in labeled.columns]
    if missing:
        raise RuntimeError(f"C{s:02d} B_s missing columns {missing}")
    if labeled[FEATURES].isna().any().any() or not np.isfinite(labeled[FEATURES].to_numpy(float)).all():
        raise RuntimeError(f"C{s:02d} B_s has non-finite features")
    if labeled["current_hardness"].isna().any() or not np.isfinite(labeled["current_hardness"].to_numpy(float)).all():
        raise RuntimeError(f"C{s:02d} B_s has non-finite current_hardness")
    if len(labeled) != 500 or labeled["instance_hash"].nunique() != 500:
        raise RuntimeError(f"C{s:02d} B_s n={len(labeled)} unique={labeled['instance_hash'].nunique()}")
    return labeled.reset_index(drop=True)


def load_p0_clean_hashes_from_s1() -> dict[int, set[str]]:
    leak = pd.read_csv(S1 / "05_p0_leakage_by_cycle.csv")
    member = pd.read_csv(ROOT / "runs" / "audits" / "gp4_calibration_sample_size_v1" / "P0_MEMBERSHIP_AND_FEATURES.csv")
    p0 = [str(h) for h in member["instance_hash"].tolist()]
    out = {}
    for s in CYCLES:
        b = load_b_s(s)
        calib = set(b["instance_hash"].tolist())
        clean = [h for h in p0 if h not in calib]
        expected = int(leak.loc[leak["cycle"] == f"C{s:02d}", "p0_clean_n"].iloc[0])
        if len(clean) != expected:
            raise RuntimeError(f"C{s:02d} P0_clean n={len(clean)} != S1 {expected}")
        overlap = set(clean) & calib
        if overlap:
            raise RuntimeError(f"C{s:02d} P0 leakage {len(overlap)}")
        out[s] = set(clean)
    return out


def load_p0_member() -> pd.DataFrame:
    member = pd.read_csv(P0_MEMBER)
    member["instance_hash"] = member["instance_hash"].astype(str)
    missing = [c for c in FEATURES if c not in member.columns]
    if missing:
        raise RuntimeError(f"P0 membership missing features {missing}")
    feats = member[FEATURES].to_numpy(float)
    if not np.isfinite(feats).all():
        raise RuntimeError("P0 membership has non-finite GP4 features")
    return member.reset_index(drop=True)


def load_p0_hardness() -> pd.DataFrame:
    hardness = read_frame(P0_HARDNESS)
    hardness["instance_hash"] = hardness["instance_hash"].astype(str)
    return hardness


def load_offspring(s: int) -> pd.DataFrame:
    frame = read_frame(offspring_path(s))
    frame["instance_hash"] = frame["instance_hash"].astype(str)
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise RuntimeError(f"C{s:02d} O_s missing features {missing}")
    if "gp_predicted_log1p_hardness" not in frame.columns:
        raise RuntimeError(f"C{s:02d} O_s missing cached GP scores")
    if "niche_id" not in frame.columns:
        raise RuntimeError(f"C{s:02d} O_s missing niche_id")
    feats = frame[FEATURES].to_numpy(float)
    if not np.isfinite(feats).all():
        raise RuntimeError(f"C{s:02d} O_s has non-finite GP4 features")
    return frame.reset_index(drop=True)


def load_s11_references() -> dict[str, pd.DataFrame]:
    return {
        "comparison": pd.read_csv(S1_1 / "05_gp_oracle_random_comparison.csv"),
        "random": pd.read_csv(S1_1 / "03_random_enr_distribution.csv"),
        "oracle": pd.read_csv(S1_1 / "04_oracle_enr_by_cycle.csv"),
        "capture": pd.read_csv(S1_1 / "06_capture_rate_by_cycle.csv"),
        "recall": pd.read_csv(S1_1 / "07_recall_ceiling_analysis.csv"),
        "s1_metrics": pd.read_csv(S1 / "07_p0_independent_metrics.csv"),
        "s1_tail": pd.read_csv(S1 / "09_p0_tail_ranking.csv"),
        "s1_recall": pd.read_csv(S1 / "08_p0_top_tail_retrieval.csv"),
        "s1_conc": pd.read_csv(S1 / "15_selection_concentration.csv"),
    }
