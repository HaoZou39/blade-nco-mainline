#!/usr/bin/env python3
"""P6-C3 frozen HGB40 cross-distribution prediction/fidelity audit.

This is a read-only audit of existing model/data artifacts. It performs no
mutation, Concorde labeling, BQ training, or HGB fitting/recalibration.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6C0 = ROOT / "target_aware_pilot/p6c0_training_value_mechanism_audit"
P6C1 = ROOT / "target_aware_pilot/p6c1_hardness_controlled_matching"
P6C2 = ROOT / "target_aware_pilot/p6c2_oracle_hardness_restoration_training_value"
P6A = ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"
C04 = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C04"
M4 = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/checkpoints/M4_E256.tsp.best"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
REPLICATES = ("Original", "S1", "S2")
SEEDS = tuple(range(2026, 2031))
K = 10_000
BOOT_N = 300
BOOT_SEED = 20261031
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=conv) + "\n")


def md_table(frame: pd.DataFrame) -> str:
    def cell(v: Any) -> str:
        if pd.isna(v): return "NA"
        if isinstance(v, (float, np.floating)): return f"{float(v):.7g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def safe_corr(fun, y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0: return float("nan")
    value = fun(y, p)
    return float(value.statistic if hasattr(value, "statistic") else value[0])


def bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, np.asarray(values, float), side="right")


def bootstrap_prediction(y: np.ndarray, p: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rho, mae = [], []
    for _ in range(BOOT_N):
        ix = rng.integers(0, len(y), len(y))
        rho.append(safe_corr(stats.spearmanr, y[ix], p[ix]))
        mae.append(float(np.mean(np.abs(y[ix] - p[ix]))))
    return {
        "spearman_ci_low": float(np.nanquantile(rho, .025)),
        "spearman_ci_high": float(np.nanquantile(rho, .975)),
        "mae_ci_low": float(np.quantile(mae, .025)),
        "mae_ci_high": float(np.quantile(mae, .975)),
    }


def prediction_metrics(frame: pd.DataFrame, label: str, mode: str, seed: int) -> dict[str, Any]:
    y = frame.true_m4_ce.to_numpy(float); p = frame.predicted_m4_ce.to_numpy(float)
    row: dict[str, Any] = {
        "group": label, "evaluation_mode": mode, "n": len(frame),
        "unique_n": frame.instance_hash.nunique(),
        "spearman_rho": safe_corr(stats.spearmanr, y, p),
        "pearson_r": safe_corr(stats.pearsonr, y, p),
        "mae": float(np.mean(np.abs(y-p))), "rmse": float(np.sqrt(np.mean((y-p)**2))),
        "mean_signed_error_pred_minus_true": float(np.mean(p-y)),
        "prediction_true_mean_ratio": float(np.mean(p)/np.mean(y)),
        "true_mean": float(np.mean(y)), "prediction_mean": float(np.mean(p)),
    }
    for name, q in (("p10", .10), ("p50", .50), ("p90", .90), ("p95", .95)):
        row[f"true_{name}"] = float(np.quantile(y, q)); row[f"prediction_{name}"] = float(np.quantile(p, q))
    if len(frame) >= 30: row.update(bootstrap_prediction(y, p, seed))
    return row


def tail_metrics(frame: pd.DataFrame, label: str, mode: str, tail: str, threshold: float, seed: int) -> dict[str, Any]:
    y = frame.true_m4_ce.to_numpy(float); score = frame.predicted_m4_ce.to_numpy(float)
    actual = y >= threshold; pred = score >= threshold
    tp = int(np.sum(actual & pred)); fp = int(np.sum(~actual & pred)); fn = int(np.sum(actual & ~pred)); tn = int(np.sum(~actual & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    prevalence = float(np.mean(actual)); predprev = float(np.mean(pred))
    ap = float(average_precision_score(actual, score)) if actual.any() and (~actual).any() else float("nan")
    auc = float(roc_auc_score(actual, score)) if actual.any() and (~actual).any() else float("nan")
    enrichment = (tp/(tp+fp))/prevalence if tp+fp and prevalence else 0.0
    rng = np.random.default_rng(seed); recalls=[]; precisions=[]
    if len(frame) >= 30:
        for _ in range(BOOT_N):
            ix = rng.integers(0, len(y), len(y)); a=actual[ix]; z=pred[ix]
            t=int(np.sum(a&z)); f=int(np.sum(~a&z)); n=int(np.sum(a&~z))
            recalls.append(t/(t+n) if t+n else np.nan); precisions.append(t/(t+f) if t+f else np.nan)
    def finite_quantile(values: list[float], q: float) -> float:
        finite=np.asarray(values,float); finite=finite[np.isfinite(finite)]
        return float(np.quantile(finite,q)) if len(finite) else float("nan")
    return {
        "group": label, "evaluation_mode": mode, "tail": tail, "threshold": threshold, "n": len(frame),
        "true_tail_n": int(actual.sum()), "predicted_tail_n": int(pred.sum()), "tp":tp,"fp":fp,"fn":fn,"tn":tn,
        "precision":precision,"recall":recall,"f1":f1,"true_prevalence":prevalence,"predicted_prevalence":predprev,
        "tail_enrichment":enrichment,"false_negative_fraction":fn/(tp+fn) if tp+fn else np.nan,
        "false_positive_fraction":fp/(fp+tn) if fp+tn else np.nan,"average_precision":ap,"auroc":auc,
        "precision_ci_low":finite_quantile(precisions,.025),
        "precision_ci_high":finite_quantile(precisions,.975),
        "recall_ci_low":finite_quantile(recalls,.025),
        "recall_ci_high":finite_quantile(recalls,.975),
    }


def main() -> None:
    started = time.perf_counter()
    for d in (OUT, OUT/"configs", OUT/"manifests", OUT/"selected_hashes", OUT/"figures"):
        d.mkdir(parents=True, exist_ok=True)
    (OUT/"00_protocol.md").write_text(
        "# P6-C3 Frozen Protocol\n\nThis offline audit replays the formal C04 M4-conditioned five-seed HGB40 ensemble without fitting, calibration, mutation, Concorde, or BQ training. Raw predictions are `expm1(median(seed log1p predictions))`. P6-C1 thresholds, E0 decile quotas, target references, Top5 geometry, and candidate unions are immutable. If predicted-bin supply is insufficient, exact selection stops without threshold changes or backfill.\n"
    )

    # Model/protocol gate.
    model_manifest = json.loads((C04/"HGB_model_manifest.json").read_text())
    cycle_summary = json.loads((C04/"cycle_summary.json").read_text())
    cal_info = json.loads((C04/"_calibration_info.json").read_text())
    f40 = json.loads(F40_MANIFEST.read_text())["F40"]
    if sha256_file(M4) != M4_SHA or cycle_summary["model_before"] != "M4" or cycle_summary["input_checkpoint_sha256"] != M4_SHA:
        raise RuntimeError("formal M4 provenance mismatch")
    if model_manifest["features"] != f40 or model_manifest["target"] != "log1p_LBQ" or model_manifest["n_train"] != 10_000:
        raise RuntimeError("HGB target/F40/training-size provenance mismatch")
    payloads = [joblib.load(C04/"HGB_seed_models"/f"HGB40_seed_{s}.joblib") for s in SEEDS]
    for s, payload in zip(SEEDS, payloads):
        if payload["features"] != f40 or payload["target"] != "log1p_LBQ" or payload["seed"] != s:
            raise RuntimeError("HGB payload drift")
    models = [p["model"] for p in payloads]

    train_members = pd.read_csv(C04/"B_calibration_membership.csv")
    archive = pd.read_parquet(C04/"01_inputs/archive_in.parquet")
    train = train_members.merge(archive[["instance_hash","lineage_root","parent_hash","operator_id","archive_provenance","first_cycle_observed"]], on="instance_hash", validate="one_to_one")
    train_hashes = set(train.instance_hash.astype(str)); train_roots = set(train.lineage_root.astype(str))
    postfit_validation_hashes = set(pd.read_parquet(C04/"07_oracle/T/labeled.parquet").instance_hash.astype(str))

    cache_files = sorted((P6C0/"full_f40_cache").glob("f40_*.parquet"))
    cache = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)
    if len(cache) != 59678 or cache.instance_hash.nunique() != len(cache) or cache[f40].isna().any().any():
        raise RuntimeError("P6-C0 F40 cache integrity failure")
    cache = cache.set_index("instance_hash")
    edges = pd.read_csv(P6C1/"logs/decile_boundaries.csv").boundary.to_numpy(float)
    quotas = pd.read_csv(P6C1/"02_decile_quotas.csv")
    hstats = pd.read_csv(P6C0/"02_hardness_statistics.csv")
    p90 = float(hstats.pooled_p90_threshold.dropna().iloc[0]); p95 = float(hstats.pooled_p95_threshold.dropna().iloc[0])

    model_hash_rows=[]
    for s in SEEDS:
        p=C04/"HGB_seed_models"/f"HGB40_seed_{s}.joblib"
        model_hash_rows.append({"artifact":p.name,"sha256":sha256_file(p),"role":"frozen HGB40 seed model"})
    source_files = [C04/"HGB_model_manifest.json",C04/"B_calibration_membership.csv",C04/"_state_B_features.parquet",C04/"B_current_hardness.parquet",M4,P6C1/"02_decile_quotas.csv",P6C1/"logs/decile_boundaries.csv"]
    model_hash_rows += [{"artifact":str(p.relative_to(ROOT)),"sha256":sha256_file(p),"role":"frozen source"} for p in source_files]
    for rep in REPLICATES:
        for suffix,role in (("union.parquet","frozen evaluation union manifest"),("union.npz","frozen evaluation union bundle"),("C0.parquet","frozen C0 manifest"),("C1.parquet","frozen Oracle-C1 manifest")):
            p=P6C1/"dataset_manifests"/f"{rep}_{suffix}"
            model_hash_rows.append({"artifact":str(p.relative_to(ROOT)),"sha256":sha256_file(p),"role":role})
    cache_digest=hashlib.sha256("".join(sha256_file(p) for p in cache_files).encode()).hexdigest()
    model_hash_rows.append({"artifact":"P6-C0 full_f40_cache ordered shard-hash digest","sha256":cache_digest,"role":f"frozen F40 cache ({len(cache_files)} shards)"})
    pd.DataFrame(model_hash_rows).to_csv(OUT/"manifests/source_artifact_hashes.csv",index=False)

    # Build one row per replicate/hash; C0/C1 remain membership flags only.
    all_frames=[]; overlap_rows=[]
    for rep in REPLICATES:
        u = pd.read_parquet(P6C1/"dataset_manifests"/f"{rep}_union.parquet").copy()
        c0=set(pd.read_parquet(P6C1/"dataset_manifests"/f"{rep}_C0.parquet").instance_hash.astype(str))
        c1=set(pd.read_parquet(P6C1/"dataset_manifests"/f"{rep}_C1.parquet").instance_hash.astype(str))
        if u.instance_hash.nunique()!=len(u): raise RuntimeError(f"union duplicate {rep}")
        u["in_C0"]=u.instance_hash.astype(str).isin(c0); u["in_Oracle_C1"]=u.instance_hash.astype(str).isin(c1)
        u["exact_hgb_train_overlap"]=u.instance_hash.astype(str).isin(train_hashes)
        u["hgb_train_root_overlap"]=u.lineage_root.astype(str).isin(train_roots)
        u["postfit_validation_overlap"]=u.instance_hash.astype(str).isin(postfit_validation_hashes)
        u["exact_hash_safe"]=~u.exact_hgb_train_overlap
        u["strict_root_safe"]=(~u.exact_hgb_train_overlap)&(~u.hgb_train_root_overlap)
        x=cache.loc[u.instance_hash.astype(str),f40].to_numpy(float)
        seed_pred=np.vstack([m.predict(x) for m in models])
        u["predicted_log1p_m4_ce"]=np.median(seed_pred,axis=0)
        u["predicted_m4_ce"]=np.expm1(u.predicted_log1p_m4_ce.to_numpy(float))
        u["true_m4_ce"]=u.m4_ce.astype(float)
        u["true_decile"]=bins(u.true_m4_ce.to_numpy(),edges)+1
        u["predicted_decile"]=bins(u.predicted_m4_ce.to_numpy(),edges)+1
        u["prediction_error"]=u.predicted_m4_ce-u.true_m4_ce
        tr_x=pd.read_parquet(C04/"_state_B_features.parquet")[f40]
        lo=tr_x.min().to_numpy(); hi=tr_x.max().to_numpy()
        u["f40_out_of_train_range_count"]=np.sum((x<lo)|(x>hi),axis=1)
        all_frames.append(u)
        overlap_rows.append({"record_type":"replicate","replicate":rep,"evaluation_n":len(u),"exact_coordinate_overlap_n":int(u.exact_hgb_train_overlap.sum()),"exact_coordinate_overlap_fraction":u.exact_hgb_train_overlap.mean(),"training_lineage_root_overlap_n":int(u.hgb_train_root_overlap.sum()),"training_lineage_root_overlap_fraction":u.hgb_train_root_overlap.mean(),"strict_root_safe_n":int(u.strict_root_safe.sum()),"postfit_validation_coordinate_overlap_n":int(u.postfit_validation_overlap.sum())})
    full=pd.concat(all_frames,ignore_index=True)
    global_unique=full.sort_values(["replicate","instance_hash"],kind="stable").drop_duplicates("instance_hash").copy()
    overlap_rows.append({"record_type":"global_unique","replicate":"GlobalUnique","evaluation_n":len(global_unique),"exact_coordinate_overlap_n":int(global_unique.exact_hgb_train_overlap.sum()),"exact_coordinate_overlap_fraction":global_unique.exact_hgb_train_overlap.mean(),"training_lineage_root_overlap_n":int(global_unique.hgb_train_root_overlap.sum()),"training_lineage_root_overlap_fraction":global_unique.hgb_train_root_overlap.mean(),"strict_root_safe_n":int(global_unique.strict_root_safe.sum()),"postfit_validation_coordinate_overlap_n":int(global_unique.postfit_validation_overlap.sum())})
    for i,a in enumerate(REPLICATES):
        ah=set(full[full.replicate.eq(a)].instance_hash)
        for b in REPLICATES[i+1:]:
            bh=set(full[full.replicate.eq(b)].instance_hash)
            overlap_rows.append({"record_type":"cross_replicate","replicate":f"{a}__{b}","evaluation_n":np.nan,"exact_coordinate_overlap_n":len(ah&bh),"exact_coordinate_overlap_fraction":len(ah&bh)/min(len(ah),len(bh)),"training_lineage_root_overlap_n":np.nan,"training_lineage_root_overlap_fraction":np.nan,"strict_root_safe_n":np.nan,"postfit_validation_coordinate_overlap_n":np.nan})
    overlap=pd.DataFrame(overlap_rows); overlap.to_csv(OUT/"01_model_training_overlap.csv",index=False)

    manifest_cols=["replicate","instance_hash","source","in_E0","in_E1","in_C0","in_Oracle_C1","true_m4_ce","predicted_m4_ce","predicted_log1p_m4_ce","true_decile","predicted_decile","prediction_error","lineage_root","operator_id","generation_born","evolution_origin","exact_hgb_train_overlap","hgb_train_root_overlap","exact_hash_safe","strict_root_safe","postfit_validation_overlap","f40_out_of_train_range_count",*[f"phi__{x}" for x in json.loads((P6C1/"configs/frozen_p6c1_config.json").read_text())["top5"]]]
    full[manifest_cols].to_csv(OUT/"02_instance_manifest.csv",index=False)

    # Prediction, calibration, tail, and dependence-sensitive metrics.
    metric_rows=[]; tail_rows=[]; confusion_rows=[]; cal_rows=[]; group_frames=[]
    for rep in REPLICATES:
        d=full[full.replicate.eq(rep)]
        group_frames += [(f"{rep}:all",d,"full_replay"),(f"{rep}:E0_only",d[d.source.eq("E0")],"full_replay"),(f"{rep}:E1_only",d[d.source.eq("E1")],"full_replay"),(f"{rep}:shared",d[d.source.eq("both")],"full_replay"),(f"{rep}:exact_hash_safe",d[d.exact_hash_safe],"exact_hash_safe"),(f"{rep}:strict_root_safe",d[d.strict_root_safe],"strict_root_safe")]
    group_frames += [("GlobalUnique:all",global_unique,"full_replay"),("GlobalUnique:E0_only",global_unique[global_unique.source.eq("E0")],"full_replay"),("GlobalUnique:E1_only",global_unique[global_unique.source.eq("E1")],"full_replay"),("GlobalUnique:shared",global_unique[global_unique.source.eq("both")],"full_replay"),("GlobalUnique:exact_hash_safe",global_unique[global_unique.exact_hash_safe],"exact_hash_safe"),("GlobalUnique:strict_root_safe",global_unique[global_unique.strict_root_safe],"strict_root_safe")]
    for gi,(name,d,mode) in enumerate(group_frames):
        if len(d)<3: continue
        metric_rows.append(prediction_metrics(d,name,mode,BOOT_SEED+gi))
        for ti,(tail,threshold) in enumerate((("P90",p90),("P95",p95))): tail_rows.append(tail_metrics(d,name,mode,tail,threshold,BOOT_SEED+100+2*gi+ti))
        cm=confusion_matrix(d.true_decile,d.predicted_decile,labels=np.arange(1,11))
        for a in range(10):
            for b in range(10): confusion_rows.append({"group":name,"evaluation_mode":mode,"true_decile":a+1,"predicted_decile":b+1,"count":int(cm[a,b])})
        for dec in range(1,11):
            z=d[d.true_decile.eq(dec)]
            if len(z): cal_rows.append({"group":name,"evaluation_mode":mode,"true_decile":dec,"n":len(z),"true_mean_ce":z.true_m4_ce.mean(),"predicted_mean_ce":z.predicted_m4_ce.mean(),"mean_bias":z.prediction_error.mean()})
    metrics=pd.DataFrame(metric_rows); tails=pd.DataFrame(tail_rows); confusion=pd.DataFrame(confusion_rows); calibration=pd.DataFrame(cal_rows)
    metrics.to_csv(OUT/"manifests/prediction_metrics.csv",index=False); tails.to_csv(OUT/"03_hard_tail_metrics.csv",index=False); confusion.to_csv(OUT/"03_decile_confusion.csv",index=False); calibration.to_csv(OUT/"manifests/calibration_by_true_decile.csv",index=False)
    full[["replicate","instance_hash","source","true_m4_ce","predicted_m4_ce","prediction_error","true_decile","predicted_decile","exact_hgb_train_overlap","hgb_train_root_overlap","strict_root_safe","lineage_root","operator_id","f40_out_of_train_range_count"]].to_csv(OUT/"03_prediction_per_instance.csv",index=False)

    # Lineage/operator/OOD sensitivity.
    dependency=[]
    for rep in REPLICATES:
        d=full[full.replicate.eq(rep)]
        for unit,col in (("lineage_root","lineage_root"),("operator","operator_id")):
            for key,z in d.groupby(col):
                dependency.append({"replicate":rep,"unit_type":unit,"unit":str(key),"n":len(z),"true_mean":z.true_m4_ce.mean(),"prediction_mean":z.predicted_m4_ce.mean(),"bias":z.prediction_error.mean(),"mae":np.mean(np.abs(z.prediction_error)),"p90_true_fraction":np.mean(z.true_m4_ce>=p90),"p90_pred_fraction":np.mean(z.predicted_m4_ce>=p90)})
    dep=pd.DataFrame(dependency); dep.to_csv(OUT/"manifests/lineage_operator_sensitivity.csv",index=False)
    lineage_summary=[]
    for rep in REPLICATES:
        z=dep[(dep.replicate.eq(rep))&(dep.unit_type.eq("lineage_root"))]
        lineage_summary.append({"replicate":rep,"lineages":len(z),"lineage_mean_spearman":safe_corr(stats.spearmanr,z.true_mean.to_numpy(),z.prediction_mean.to_numpy()),"median_lineage_bias":z.bias.median(),"p90_true_vs_pred_lineage_spearman":safe_corr(stats.spearmanr,z.p90_true_fraction.to_numpy(),z.p90_pred_fraction.to_numpy())})
    lineage_summary=pd.DataFrame(lineage_summary); lineage_summary.to_csv(OUT/"manifests/lineage_level_summary.csv",index=False)

    # Frozen predicted-bin supply check. This is decisive before any LP solve.
    shortage=[]
    for rep in REPLICATES:
        d=full[full.replicate.eq(rep)]; available=np.bincount(d.predicted_decile.to_numpy(int)-1,minlength=10)
        target=quotas[quotas.replicate.eq(rep)].sort_values("decile").quota.to_numpy(int)
        for dec,(a,q) in enumerate(zip(available,target),1): shortage.append({"replicate":rep,"predicted_decile":dec,"available_candidates":int(a),"required_quota":int(q),"surplus":int(a-q),"shortage":int(max(q-a,0)),"bin_feasible":bool(a>=q)})
    shortage=pd.DataFrame(shortage); shortage.to_csv(OUT/"manifests/predicted_decile_supply.csv",index=False)
    feasible_by_rep=shortage.groupby("replicate").bin_feasible.all()
    if feasible_by_rep.any(): raise RuntimeError("unexpected partial feasibility: runner intentionally audits the observed all-replicate supply failure")
    integrity=pd.DataFrame([{"replicate":rep,"union_n":len(full[full.replicate.eq(rep)]),"unique_n":full[full.replicate.eq(rep)].instance_hash.nunique(),"reference_quota_sum":K,"predicted_decile_quota_sum":int(quotas[quotas.replicate.eq(rep)].quota.sum()),"infeasible_decile_count":int((~shortage[shortage.replicate.eq(rep)].bin_feasible).sum()),"total_shortage_if_bins_independent":int(shortage[shortage.replicate.eq(rep)].shortage.sum()),"strict_selection_feasible":False,"lp_started":False,"reason":"candidate supply below required predicted-decile quota"} for rep in REPLICATES])
    integrity.to_csv(OUT/"manifests/surrogate_selection_integrity.csv",index=False)

    # Existing immutable C0/Oracle-C1 diagnostics; no surrogate row is fabricated.
    old_metrics=pd.read_csv(P6C1/"logs/selection_metrics.csv")
    fidelity=[]
    for rep in REPLICATES:
        d=full[full.replicate.eq(rep)]; target=quotas[quotas.replicate.eq(rep)].sort_values("decile").quota.to_numpy(int)/K
        for arm,col in (("C0","in_C0"),("Oracle-C1","in_Oracle_C1")):
            z=d[d[col]]; counts=np.bincount(z.true_decile.to_numpy(int)-1,minlength=10); prob=counts/K
            roots=z.lineage_root.astype(str).value_counts().to_numpy(float); rp=roots/roots.sum()
            f40z=cache.loc[z.instance_hash.astype(str),f40].to_numpy(float); nn=cKDTree(f40z).query(f40z,k=2,workers=-1)[0][:,1]
            oldarm="C1" if arm=="Oracle-C1" else arm; old=old_metrics[(old_metrics.replicate.eq(rep))&(old_metrics.arm.eq(oldarm))].iloc[0]
            fidelity.append({"replicate":rep,"selection":arm,"status":"AVAILABLE_FROZEN","n":len(z),"unique_n":z.instance_hash.nunique(),"mean_true_m4_ce":z.true_m4_ce.mean(),"median_true_m4_ce":z.true_m4_ce.median(),"p90_tail_fraction":np.mean(z.true_m4_ce>=p90),"p95_tail_fraction":np.mean(z.true_m4_ce>=p95),"true_decile_counts":" ".join(map(str,counts)),"true_decile_tvd_to_E0_target":.5*np.abs(prob-target).sum(),"true_decile_ordinal_emd":stats.wasserstein_distance(np.arange(1,11),np.arange(1,11),u_weights=prob,v_weights=target),"true_ce_wasserstein_to_E0":stats.wasserstein_distance(z.true_m4_ce,d[d.in_E0].true_m4_ce),"ot_total_cost":old.ot_total_cost,"ot_mean_cost":old.ot_mean_cost,"top5_swd":old.top5_swd,"top5_energy":old.top5_energy,"reference_quota_exact":bool(old.reference_quota_exact),"true_decile_quota_exact":bool(old.decile_quota_exact) if arm=="Oracle-C1" else False,"e0_source_fraction":z.in_E0.mean(),"e1_source_fraction":z.in_E1.mean(),"effective_lineages":np.exp(-(rp*np.log(rp)).sum()),"largest_lineage_share":rp.max(),"f40_nn_median":np.median(nn),"overlap_with_oracle_c1":np.mean(z.instance_hash.isin(set(d[d.in_Oracle_C1].instance_hash))),"overlap_with_c0":np.mean(z.instance_hash.isin(set(d[d.in_C0].instance_hash)))})
        fidelity.append({"replicate":rep,"selection":"Surrogate-C1","status":"INFEASIBLE_NOT_SELECTED","n":0,"unique_n":0})
    fidelity=pd.DataFrame(fidelity); fidelity.to_csv(OUT/"05_selection_fidelity.csv",index=False)
    selected_ops=[]
    for rep in REPLICATES:
        d=full[full.replicate.eq(rep)]
        for arm,col in (("C0","in_C0"),("Oracle-C1","in_Oracle_C1")):
            z=d[d[col]]
            for op,n in z.operator_id.astype(str).value_counts().items(): selected_ops.append({"replicate":rep,"selection":arm,"operator_id":op,"count":int(n),"share":float(n/len(z))})
        selected_ops.append({"replicate":rep,"selection":"Surrogate-C1","operator_id":"NOT_AVAILABLE_INFEASIBLE","count":0,"share":np.nan})
    pd.DataFrame(selected_ops).to_csv(OUT/"manifests/selection_operator_composition.csv",index=False)
    reference_rows=[]
    for rep in REPLICATES:
        for arm,file_arm in (("C0","C0"),("Oracle-C1","C1")):
            z=pd.read_parquet(P6C1/"dataset_manifests"/f"{rep}_{file_arm}.parquet")
            counts=z.selected_reference_index.value_counts().reindex(range(48),fill_value=0)
            for ref,n in counts.items(): reference_rows.append({"replicate":rep,"selection":arm,"reference_index":int(ref),"count":int(n),"status":"FROZEN_AVAILABLE"})
        for ref in range(48): reference_rows.append({"replicate":rep,"selection":"Surrogate-C1","reference_index":ref,"count":np.nan,"status":"NOT_AVAILABLE_INFEASIBLE"})
    pd.DataFrame(reference_rows).to_csv(OUT/"manifests/reference_quota_audit.csv",index=False)
    pd.DataFrame(columns=["replicate","instance_hash","reason"]).to_csv(OUT/"manifests/surrogate_c1_selection_manifest.csv",index=False)
    for rep in REPLICATES: (OUT/"selected_hashes"/f"{rep}_Surrogate_C1_NOT_CREATED.txt").write_text("INFEASIBLE: frozen predicted-decile candidate supply below quota; no selection created.\n")

    # Plots.
    colors={"E0":"#4477AA","E1":"#CC6677","both":"#777777"}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        d=full[full.replicate.eq(rep)]
        for source in ("E0","E1","both"):
            z=d[d.source.eq(source)]; ax.scatter(z.true_m4_ce,z.predicted_m4_ce,s=3,alpha=.18,color=colors[source],label=source)
        lim=max(d.true_m4_ce.max(),d.predicted_m4_ce.max()); ax.plot([0,lim],[0,lim],"k--",lw=1); ax.axhline(p90,color="orange",lw=.8); ax.axvline(p90,color="orange",lw=.8); ax.set_title(rep); ax.set_xlabel("True M4 CE"); ax.grid(alpha=.15)
    axes[0].set_ylabel("HGB40 predicted raw CE"); axes[-1].legend(markerscale=3); fig.tight_layout(); fig.savefig(OUT/"figures/01_true_vs_predicted_hardness.png",dpi=180); plt.close(fig)
    cm=confusion_matrix(global_unique.true_decile,global_unique.predicted_decile,labels=np.arange(1,11)); cmn=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cmn,cmap="magma",vmin=0,vmax=1); ax.set_xticks(range(10),range(1,11)); ax.set_yticks(range(10),range(1,11)); ax.set_xlabel("Predicted decile"); ax.set_ylabel("True decile"); ax.set_title("Global unique row-normalized decile confusion"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/"figures/02_decile_confusion_matrix.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(tail,threshold) in zip(axes,(("P90",p90),("P95",p95))):
        for rep in REPLICATES:
            d=full[full.replicate.eq(rep)]; precision,recall,_=precision_recall_curve(d.true_m4_ce>=threshold,d.predicted_m4_ce); ax.plot(recall,precision,label=rep)
        ax.axhline(float(np.mean(global_unique.true_m4_ce>=threshold)),color="k",ls="--",lw=1,label="global prevalence"); ax.set(xlabel="Recall",ylabel="Precision",title=f"{tail} continuous-score PR"); ax.grid(alpha=.2)
    axes[1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/03_tail_precision_recall.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        d=full[full.replicate.eq(rep)]; x=np.arange(1,11); width=.25
        for off,(name,col) in zip((-.25,0),(('C0','in_C0'),('Oracle-C1','in_Oracle_C1'))):
            counts=np.bincount(d[d[col]].true_decile.to_numpy(int)-1,minlength=10); ax.bar(x+off,counts,width,label=name)
        ax.text(.5,.92,"Surrogate-C1: infeasible",transform=ax.transAxes,ha="center",color="crimson"); ax.set_title(rep); ax.set_xlabel("True hardness decile"); ax.set_xticks(x); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Selected count"); axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/04_selected_true_decile_distribution.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for rep,marker in zip(REPLICATES,("o","s","^")):
        z=fidelity[(fidelity.replicate.eq(rep))&fidelity.selection.isin(["C0","Oracle-C1"])]
        ax.plot(z.mean_true_m4_ce,z.ot_mean_cost,marker=marker,label=rep)
        for _,r in z.iterrows(): ax.annotate(r.selection,(r.mean_true_m4_ce,r.ot_mean_cost),fontsize=8)
    ax.text(.98,.05,"Surrogate-C1 absent: exact quotas infeasible",transform=ax.transAxes,ha="right",color="crimson"); ax.set(xlabel="Mean true M4 CE",ylabel="Mean exact Top5 OT cost",title="Structure cost vs true hardness restoration"); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figures/05_ot_vs_true_hardness_recovery.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,rep in zip(axes,REPLICATES):
        d=full[full.replicate.eq(rep)]; c0=set(d[d.in_C0].instance_hash); c1=set(d[d.in_Oracle_C1].instance_hash)
        comp=[len(c0-c1),len(c0&c1),len(c1-c0)]; ax.bar(["C0 only","shared","Oracle only"],comp,color=["#4477AA","#999999","#CC6677"]); ax.set_title(f"{rep}\nSurrogate absent"); ax.tick_params(axis="x",rotation=20); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Membership count"); fig.tight_layout(); fig.savefig(OUT/"figures/06_membership_lineage_composition.png",dpi=180); plt.close(fig)

    # Human-readable reports.
    prov=overlap[overlap.record_type.isin(["replicate","global_unique"])]
    (OUT/"01_hgb40_provenance_audit.md").write_text(
        "# HGB40 Provenance Audit\n\n**Gate: PASS for frozen inference.** The recovered formal C04 ensemble is conditioned on formal M4 SHA256 `"+M4_SHA+"`. C04 refreshed 10,000 archive examples with M4 teacher-forced LBQ/CE and fitted five `HistGradientBoostingRegressor` models (seeds 2026–2030) to `log1p_LBQ`; deployment aggregation is the median and this audit applies `expm1` exactly once. There is no output calibrator. The payload F40 names/order equal the frozen manifest byte-for-byte. Parameters were fixed, `early_stopping=false`, and no P6-B/C0/C1/C2 labels were used for fitting or model selection.\n\nThe historical model has no internal held-out validation split: its manifest reports training fit only. The later C04 Hard10k labels were obtained after deployment and were not used to refit or choose a seed. Exact coordinate leakage into P6-B unions is small, but root overlap is very high because both experiments descend from the same evolutionary ecosystem. Consequently full replay and exact-hash-safe results are descriptive; only the small no-training-root subset is called strict root-safe. No geometric near-duplicate threshold was pre-registered, so near-duplicate leakage is not asserted from feature distance.\n\n"+md_table(prov)+"\n\nCross-replicate reuse and all hashes are in the CSV/manifests.\n"
    )
    pool_rows=[]
    for rep in REPLICATES:
        d=full[full.replicate.eq(rep)]; pool_rows.append({"replicate":rep,"memberships_E0_plus_E1":20000,"unique_union_n":len(d),"E0_only":int(d.source.eq("E0").sum()),"E1_only":int(d.source.eq("E1").sum()),"shared":int(d.source.eq("both").sum()),"strict_root_safe_n":int(d.strict_root_safe.sum()),"C0_membership":int(d.in_C0.sum()),"Oracle_C1_membership":int(d.in_Oracle_C1.sum()),"F40_complete":bool(cache.loc[d.instance_hash,f40].notna().all().all())})
    pool_rows=pd.DataFrame(pool_rows)
    (OUT/"02_frozen_evaluation_pool.md").write_text("# Frozen Evaluation Pool\n\nEach row in the CSV is a unique coordinate within one replicate. C0/C1 are boolean memberships, not extra observations. Global deduplication leaves "+f"{len(global_unique):,}"+" coordinate hashes from "+f"{len(full):,}"+" replicate-level unique rows.\n\n"+md_table(pool_rows)+"\n")
    core=metrics[metrics.group.isin([f"{r}:all" for r in REPLICATES]+[f"{r}:E0_only" for r in REPLICATES]+[f"{r}:E1_only" for r in REPLICATES]+[f"{r}:strict_root_safe" for r in REPLICATES])]
    (OUT/"03_prediction_quality.md").write_text(
        "# Prediction Quality\n\nRaw-space metrics use the frozen inverse transform; no recalibration is applied. Pooled correlations are partly source-shift driven, so E0-only/E1-only and strict-root-safe rows are primary diagnostics. HGB correctly predicts E0 as harder in mean, but underpredicts both groups and has weak within-E1 ranking.\n\n"+md_table(core[["group","evaluation_mode","n","spearman_rho","spearman_ci_low","spearman_ci_high","pearson_r","mae","mean_signed_error_pred_minus_true","prediction_true_mean_ratio","true_mean","prediction_mean"]])+"\n\nCalibration by true decile is in `manifests/calibration_by_true_decile.csv`; tail metrics and bootstrap intervals are separate.\n"
    )
    (OUT/"04_surrogate_selection_protocol.md").write_text("# Surrogate-C1 Selection Protocol\n\nThe intended solve is the byte-identical P6-C1 totally-unimodular transport LP: candidate capacity ≤1, exactly 10,000 mass, exact 48 frozen reference quotas, and exact replicate-specific E0 decile quotas. The only intended substitution is candidate bin membership from frozen HGB raw predictions. Candidate true CE is hidden from selection. Before constructing the LP, necessary per-bin supply is checked. Thresholds, quota counts, objective, and `side='right'` assignment are immutable; no true-CE repair, threshold shift, greedy backfill, or quota relaxation is permitted.\n")
    (OUT/"04_surrogate_selection_integrity.md").write_text("# Surrogate-C1 Selection Integrity\n\n**Strict selection is infeasible in all three replicates.** Necessary candidate supply fails in five bins (D1 and D7–D10) per replicate. In particular D10 has only 21/8/20 predicted candidates for Original/S1/S2 versus quotas 980/1003/1017. The LP was therefore not started and no Surrogate-C1 manifest, bundle, membership hash, OT value, or recovery statistic was fabricated.\n\n"+md_table(shortage)+"\n\n`P6C3_SURROGATE_SELECTION_FEASIBILITY_FAILURE`\n")
    (OUT/"05_true_hardness_restoration.md").write_text("# True Hardness Restoration\n\nFrozen C0 and Oracle-C1 values are reproduced below. Surrogate-C1 is absent because its frozen predicted-decile constraints fail the necessary supply condition. Consequently mean-hardness recovery, P90/P95 recovery, true-decile TV/EMD, membership overlap, source fraction, lineage diversity, operator composition, and Full-F40 clustering for Surrogate-C1 are **not estimable**. Treating them as zero would incorrectly conflate infeasibility with an observed selected population.\n\n"+md_table(fidelity[["replicate","selection","status","n","mean_true_m4_ce","median_true_m4_ce","p90_tail_fraction","p95_tail_fraction","true_decile_tvd_to_E0_target","true_decile_ordinal_emd","ot_mean_cost","effective_lineages","largest_lineage_share","f40_nn_median"]])+"\n")
    (OUT/"06_structural_matching_cost.md").write_text("# Structural Matching Cost\n\nC0 and Oracle-C1 exact costs/SWD/energy are immutable P6-C1 measurements. No Surrogate/Oracle or Surrogate/C0 ratio exists because strict Surrogate-C1 selection is infeasible. The P6-C1 1.20× gate is not applied here.\n\n"+md_table(fidelity[["replicate","selection","status","ot_total_cost","ot_mean_cost","top5_swd","top5_energy","mean_true_m4_ce","p90_tail_fraction","p95_tail_fraction"]])+"\n")
    op_summary=full.groupby(["replicate","source"],as_index=False).agg(n=("instance_hash","size"),bias=("prediction_error","mean"),mae=("prediction_error",lambda x:np.mean(np.abs(x))),ood_any_fraction=("f40_out_of_train_range_count",lambda x:np.mean(x>0)))
    source_means=full.groupby(["replicate","source"],as_index=False).agg(true_mean=("true_m4_ce","mean"),pred_mean=("predicted_m4_ce","mean"))
    (OUT/"07_failure_mode_analysis.md").write_text(
        "# Failure-Mode Analysis\n\nThe decisive failure is output compression/negative bias, not an optimization artifact: frozen high-decile thresholds sit far above most predictions, so quota supply fails before the selection LP. HGB preserves the E0>E1 mean direction, but within-E1 rho is only about 0.10–0.11 and no E1-only instance crosses either frozen P90/P95 threshold. Thus pooled ranking overstates candidate-level tail utility. True high-hardness examples are systematically assigned to lower predicted deciles. F40 extrapolation, operator and lineage summaries are diagnostic only and were not used to alter the model.\n\n## Source means\n\n"+md_table(source_means)+"\n\n## Bias and F40 range\n\n"+md_table(op_summary)+"\n\n## Lineage sensitivity\n\n"+md_table(lineage_summary)+"\n"
    )

    # Conclusions: consistent modest strict-root-safe ranking, but no calibrated tail/selection deployment.
    (OUT/"08_final_research_conclusion.md").write_text(
        "# Final Research Conclusion\n\n1. **Model identity:** yes—formal C04 HGB40 predicts `log1p` of formal-M4 teacher-forced CE; this audit applies the frozen inverse transform.\n2. **Leakage:** exact training-coordinate overlap is 27–34 per replicate (68 global-unique), while training-root overlap is 98.25%–98.57%; strict root-safe subsets contain 285/349/345 examples.\n3. **Strict evaluation:** possible but small and restricted to this evolutionary ecosystem. It shows consistent modest ranking, not deployment-grade calibration.\n4. **Source behavior:** full-pool rho is about 0.38–0.40; E0-only about 0.30–0.33; E1-only about 0.10–0.11. HGB correctly orders E0 mean above E1 mean, but underpredicts both.\n5. **Tail behavior:** raw-threshold recall is only about 3.3%–3.7% at P90 and 0.1%–0.9% at P95 on full unions; E1-only and strict-root-safe recall is zero because no prediction crosses the thresholds. Continuous AP/AUROC are better than threshold recall but do not repair calibration.\n6. **Selection:** no replicate has enough predicted-bin candidates for the frozen quotas, so exact 10k/48-reference/predicted-decile Surrogate-C1 is infeasible and no selection is created.\n7. **Restoration/structure:** true-hardness recovery, tail recovery, Oracle overlap, and Surrogate OT are undefined—not zero—because no lawful Surrogate-C1 exists.\n8. **Concentration:** existing C0/Oracle lineage and Full-F40 diagnostics are reported; no new Surrogate concentration can occur or be assessed.\n9. **Mechanism:** P6-C2 established preliminary value for restoring *true* hardness. P6-C3 shows that the frozen HGB40 cannot reproduce that intervention under the frozen raw-decile constraints. These are distinct conclusions.\n10. **Minimal next experiment:** build a time-separated, root-disjoint M4-labeled validation set and audit a leakage-safe, tail-aware/calibrated surrogate offline. Because retraining is outside P6-C3, none is launched.\n\n`P6C3_PREDICTION_GENERALIZATION_SUPPORTED` (modest ranking only; strict subsets are small)\n\n`P6C3_HARD_TAIL_IDENTIFICATION_NOT_SUPPORTED`\n\n`P6C3_SURROGATE_SELECTION_FIDELITY_NOT_SUPPORTED`\n\n`P6C3_DEPLOYMENT_READINESS: NOT_SUPPORTED`\n\n`BLADE_P6C3_HGB40_HARDNESS_SELECTION_FIDELITY_AUDIT_COMPLETE`\n"
    )
    runtime={"status":"COMPLETE","wall_s":time.perf_counter()-started,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"cpu_only":True,"gpu_used":False,"new_hgb_training":False,"new_concorde":False,"new_mutation_or_evolution":False,"new_bq_training":False,"selection_lp_started":False,"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}
    write_json(OUT/"configs/frozen_p6c3_config.json",{"model_sha256s":{str(s):sha256_file(C04/"HGB_seed_models"/f"HGB40_seed_{s}.joblib") for s in SEEDS},"model_manifest_sha256":sha256_file(C04/"HGB_model_manifest.json"),"training_membership_sha256":cal_info["membership_sha256"],"hgb_params":model_manifest["params"],"m4_sha256":M4_SHA,"f40":f40,"target":"log1p_LBQ","ensemble":"median then expm1","decile_edges":edges,"p90_threshold":p90,"p95_threshold":p95,"replicates":list(REPLICATES),"bootstrap_n":BOOT_N,"bootstrap_seed":BOOT_SEED,"prohibitions":["HGB fit/recalibration","Concorde","mutation/evolution","BQ training","quota/threshold changes","true-CE repair"]})
    write_json(OUT/"completion.json",{"status":"COMPLETE","H1":"P6C3_PREDICTION_GENERALIZATION_SUPPORTED","H2":"P6C3_HARD_TAIL_IDENTIFICATION_NOT_SUPPORTED","H3":"P6C3_SURROGATE_SELECTION_FIDELITY_NOT_SUPPORTED","H4":"NOT_SUPPORTED","selection_feasible":False,"final_label":"BLADE_P6C3_HGB40_HARDNESS_SELECTION_FIDELITY_AUDIT_COMPLETE"})
    (OUT/"09_runtime_and_resource_audit.md").write_text("# Runtime and Resource Audit\n\n"+md_table(pd.DataFrame([runtime]))+"\n\nAll work was CPU-only frozen inference and offline analysis. No GPU, Concorde, mutation, evolution, model training, or external benchmark evaluation was used.\n")
    output_files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"output_artifact_hashes.csv"} and "__pycache__" not in p.parts)
    pd.DataFrame([{"artifact":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in output_files]).to_csv(OUT/"manifests/output_artifact_hashes.csv",index=False)
    print(json.dumps(runtime,indent=2))


if __name__ == "__main__":
    main()
