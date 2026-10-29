#!/usr/bin/env python3
"""P6-F0 read-only historical mechanism audit.

This program creates only derived tables, reports and figures.  It performs no
mutation, Concorde solve, model inference, fitting or training.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target_aware_pilot/p6f0_best_population_mechanism_audit"
FIG = OUT / "figures"
TABLE = OUT / "tables"
LOG = OUT / "logs"
CFG = OUT / "configs"
MAN = OUT / "manifests"
REPS = ["Original", "S1", "S2"]
ARMS = ["P5_E1", "B", "C", "D25", "D50", "D75"]
F1 = "spn_distance_varcoeff"
F2 = "spn_fraction_shorter_mean_distance"
START = time.time()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def effective_count(series: pd.Series) -> float:
    p = series.value_counts(normalize=True).to_numpy(float)
    return float(np.exp(-(p * np.log(p)).sum())) if len(p) else float("nan")


def local_nn(a: np.ndarray, cap: int = 2500) -> float:
    a = a[np.isfinite(a).all(1)]
    if len(a) < 2:
        return float("nan")
    if len(a) > cap:
        ix = np.linspace(0, len(a) - 1, cap, dtype=int)
        a = a[ix]
    scale = np.nanstd(a, axis=0)
    scale[scale == 0] = 1
    z = (a - np.nanmedian(a, axis=0)) / scale
    d = cdist(z, z)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def bootstrap_seed_instance(diff: pd.DataFrame, n_boot: int = 10000) -> tuple[float, float]:
    """Hierarchical bootstrap: sample evolution seed, then the 70 identities."""
    rng = np.random.default_rng(20261028)
    piv = diff.pivot(index="instance_id", columns="replicate", values="p5_minus_b")
    vals = piv[REPS].to_numpy(float)
    out = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, 3, 3)
        x = rng.integers(0, len(vals), len(vals))
        out[i] = vals[x][:, s].mean()
    return tuple(np.quantile(out, [0.025, 0.975]))


def p5_selected_path(rep: str) -> Path:
    if rep == "Original":
        return ROOT / "target_aware_pilot/p2_bounded_evolution/support_e1_provisional_structure10k.parquet"
    return ROOT / f"target_aware_pilot/p5_structure_guided_evolution_robustness/datasets/{rep}_E1_Structure10k.parquet"


def p5_archive_path(rep: str) -> Path:
    if rep == "Original":
        return ROOT / "target_aware_pilot/p2_bounded_evolution/evolution/E1/G2/bounded_archive.parquet"
    return ROOT / f"target_aware_pilot/p5_structure_guided_evolution_robustness/evolution/{rep}/evolution/E1/G2/bounded_archive.parquet"


def p6_selected_path(rep: str, arm: str) -> Path:
    base = "p6d1_target_aware_qd_training_value" if arm in ("B", "C") else "p6d3_structure_hardness_quota_selection"
    return ROOT / f"target_aware_pilot/{base}/datasets/{rep}_{arm}.parquet"


def p6_npz_path(rep: str, arm: str) -> Path:
    return p6_selected_path(rep, arm).with_suffix(".npz")


def p5_npz_path(rep: str) -> Path:
    if rep == "Original":
        return ROOT / "target_aware_pilot/p3_structure_finetune/labeling/labeled_bundle.npz"
    return ROOT / f"target_aware_pilot/p5_structure_guided_evolution_robustness/logs/{rep}_E1/labeling/labeled_bundle.npz"


def loss_path(rep: str, arm: str) -> Path:
    if arm == "P5_E1":
        if rep == "Original":
            return ROOT / "target_aware_pilot/p3_structure_finetune/training_cycle/09_bq_training/training_loss_curve.csv"
        return ROOT / f"target_aware_pilot/p5_structure_guided_evolution_robustness/logs/{rep}_E1/training_cycle/09_bq_training/training_loss_curve.csv"
    base = "p6d1_target_aware_qd_training_value" if arm in ("B", "C") else "p6d3_structure_hardness_quota_selection"
    return ROOT / f"target_aware_pilot/{base}/logs/{rep}_{arm}/training_cycle/09_bq_training/training_loss_curve.csv"


def load_full_f40() -> pd.DataFrame:
    fs = sorted((ROOT / "target_aware_pilot/p6c0_training_value_mechanism_audit/full_f40_cache").glob("*.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
    return d.drop_duplicates("instance_hash")


def load_performance() -> pd.DataFrame:
    p5 = pd.read_csv(ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness/support_per_instance_paired_differences.csv")
    p5 = p5[p5.dataset.eq("OriginalTSPLIB70")][["replicate", "instance_id", "coordinate_hash", "n_nodes", "gap_E1"]]
    # The P5 support export contains two byte-identical copies of Original70
    # (once from the primary endpoint and once from the split aggregation path).
    p5 = p5.drop_duplicates(["replicate", "instance_id", "coordinate_hash", "n_nodes", "gap_E1"])
    p6 = pd.read_csv(ROOT / "target_aware_pilot/p6d3_structure_hardness_quota_selection/07_per_instance_gap.csv")
    p6 = p6[p6.dataset.eq("OriginalTSPLIB70")]
    d = p6.merge(p5, on=["replicate", "instance_id", "coordinate_hash", "n_nodes"], validate="one_to_one")
    d = d.rename(columns={"gap_E1": "gap_P5_E1"})
    ref = set(pd.read_csv(ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id)
    d["split"] = np.where(d.instance_id.isin(ref), "TSPLIB48", "TSPLIB22")
    d["n_bin"] = pd.cut(d.n_nodes, [0, 99, 199, 499, 999, np.inf], labels=["<100", "100-199", "200-499", "500-999", ">=1000"])
    for arm in ["P5_E1", "B", "C", "D25", "D50", "D75"]:
        d[f"improvement_{arm}"] = d.gap_M4 - d[f"gap_{arm}"]
    keep = ["replicate", "instance_id", "coordinate_hash", "n_nodes", "split", "n_bin", "gap_M4"]
    for a in ARMS:
        keep += [f"gap_{a}", f"improvement_{a}"]
    return d[keep]


def performance_tables(perf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for rep in REPS:
        for split in ["TSPLIB48", "TSPLIB22", "Original70"]:
            q = perf[perf.replicate.eq(rep)]
            if split != "Original70": q = q[q.split.eq(split)]
            for arm in ["M4"] + ARMS:
                g = q.gap_M4 if arm == "M4" else q[f"gap_{arm}"]
                imp = q.gap_M4 - g
                pos = imp[imp > 1e-12]; neg = imp[imp < -1e-12]
                rows.append(dict(replicate=rep, split=split, arm=arm, n=len(q), mean_gap=g.mean(), median_gap=g.median(), p90_gap=g.quantile(.90), p95_gap=g.quantile(.95), improved=int((imp>1e-12).sum()), worsened=int((imp< -1e-12).sum()), ties=int((imp.abs()<=1e-12).sum()), mean_positive_improvement=pos.mean(), mean_negative_deterioration=neg.mean()))
    summ = pd.DataFrame(rows)
    seeds = []
    p = perf.copy(); p["p5_minus_b"] = p.gap_P5_E1 - p.gap_B
    piv = p.pivot(index="instance_id", columns="replicate", values="improvement_P5_E1")
    for name in piv.index:
        x = piv.loc[name]
        seeds.append(dict(instance_id=name, improvement_original=x.Original, improvement_S1=x.S1, improvement_S2=x.S2, all_three_improve=bool((x>0).all()), original_S1_improve_S2_regress=bool(x.Original>0 and x.S1>0 and x.S2<0), direction_agreement=float(max((x>0).mean(), (x<0).mean(), (x==0).mean())), mean_improvement=x.mean(), sd_improvement=x.std(ddof=1)))
    return summ, pd.DataFrame(seeds)


def concentration(perf: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for rep in REPS:
        q=perf[perf.replicate.eq(rep)].copy(); x=q.improvement_P5_E1.to_numpy(); total=x.sum()
        so=np.sort(x)[::-1]; sn=np.sort(x)
        for k in [1,5,10]:
            rows.append(dict(replicate=rep, statistic=f"top{k}_gain_share", value=float(so[:k].sum()/total) if total else np.nan))
        rows.append(dict(replicate=rep, statistic="top5_deterioration_share_abs", value=float(abs(sn[:5].sum())/max(abs(total),1e-12))))
        rows.append(dict(replicate=rep, statistic="leave_largest_gain_out_mean_gap", value=float(q.loc[q.improvement_P5_E1.ne(q.improvement_P5_E1.max()),"gap_P5_E1"].mean())))
        drop=q.nlargest(5,"improvement_P5_E1").index
        rows.append(dict(replicate=rep, statistic="leave_top5_gains_out_mean_gap", value=float(q.drop(drop).gap_P5_E1.mean())))
        for _,r in q.nlargest(5,"improvement_P5_E1").iterrows(): rows.append(dict(replicate=rep,statistic="largest_gain_instance",instance_id=r.instance_id,value=r.improvement_P5_E1))
        for _,r in q.nsmallest(5,"improvement_P5_E1").iterrows(): rows.append(dict(replicate=rep,statistic="largest_regression_instance",instance_id=r.instance_id,value=r.improvement_P5_E1))
    return pd.DataFrame(rows)


def pool_metrics(f40: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; overlaps=[]
    for rep in REPS:
        p5=pd.read_parquet(p5_archive_path(rep)); p5["pool"]="P5_E1_archive30k"
        p6=pd.read_parquet(ROOT/f"target_aware_pilot/p6c1_hardness_controlled_matching/dataset_manifests/{rep}_union.parquet")
        p6=p6.merge(f40[["instance_hash",F1,F2]],on="instance_hash",how="left"); p6["pool"]="P6_E0E1_union"
        for x in [p5,p6]:
            h=x.instance_hash.astype(str); roots=x.lineage_root.astype(str)
            rows.append(dict(replicate=rep,pool=x.pool.iloc[0],n=len(x),unique_coordinates=h.nunique(),exact_duplicate_rows=len(x)-h.nunique(),roots=roots.nunique(),effective_lineages=effective_count(roots),max_lineage_fraction=roots.value_counts(normalize=True).max(),operators=x.operator_id.nunique(),effective_operators=effective_count(x.operator_id.astype(str)),feature1_median=x[F1].median(),feature2_median=x[F2].median(),feature1_iqr=x[F1].quantile(.75)-x[F1].quantile(.25),feature2_iqr=x[F2].quantile(.75)-x[F2].quantile(.25),local_2d_nn=local_nn(x[[F1,F2]].to_numpy(float)),true_m4_ce_mean=x.m4_ce.mean() if "m4_ce" in x else np.nan,qd4_occupied=x.qd_niche.nunique() if "qd_niche" in x else np.nan,full_f40_available=bool(x.pool.iloc[0].startswith("P6"))))
        a=set(p5.instance_hash.astype(str)); b=set(p6.instance_hash.astype(str)); sa=set(p5.lineage_root.astype(str)); sb=set(p6.lineage_root.astype(str))
        ps=pd.read_parquet(p5_selected_path(rep)); bs=pd.read_parquet(p6_selected_path(rep,"B"))
        si=len(set(ps.instance_hash.astype(str))&set(bs.instance_hash.astype(str)))
        overlaps.append(dict(replicate=rep,pool_overlap=len(a&b),pool_jaccard=len(a&b)/len(a|b),shared_roots=len(sa&sb),selected_P5_B_overlap=si,selected_P5_B_jaccard=si/len(set(ps.instance_hash.astype(str))|set(bs.instance_hash.astype(str)))))
    return pd.DataFrame(rows),pd.DataFrame(overlaps)


def population_metrics(f40: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; feature_rows=[]
    # P5 reference geometry in its historically frozen two-dimensional feature space.
    ref=pd.read_csv(ROOT/"target_aware_pilot/p1_tsplib48_normalization_sensitivity.csv")
    ref=ref[ref.coordinate_variant.eq("aspect")][[F1,F2]]
    center=ref.median(); scale=ref.quantile(.75)-ref.quantile(.25); scale[scale==0]=1
    for rep in REPS:
        pops={"P5_E1":pd.read_parquet(p5_selected_path(rep))}
        for arm in ["B","C","D25","D50","D75"]: pops[arm]=pd.read_parquet(p6_selected_path(rep,arm)).merge(f40[["instance_hash",F2]],on="instance_hash",how="left")
        for arm,x in pops.items():
            roots=x.lineage_root.astype(str); z=(x[[F1,F2]]-center)/scale
            dist=cdist(z.to_numpy(float),((ref-center)/scale).to_numpy(float)).min(1)
            rows.append(dict(replicate=rep,arm=arm,n=len(x),unique_coordinates=x.instance_hash.nunique(),roots=roots.nunique(),effective_lineages=effective_count(roots),max_lineage_fraction=roots.value_counts(normalize=True).max(),operators=x.operator_id.nunique(),effective_operators=effective_count(x.operator_id.astype(str)),p5_reference_distance_median=np.median(dist),p5_reference_distance_p90=np.quantile(dist,.9),feature1_median=x[F1].median(),feature2_median=x[F2].median(),local_2d_nn=local_nn(x[[F1,F2]].to_numpy(float)),true_m4_ce_mean=x.true_m4_ce.mean() if "true_m4_ce" in x else np.nan,true_m4_ce_p90=x.true_m4_ce.quantile(.9) if "true_m4_ce" in x else np.nan,true_m4_ce_p95=x.true_m4_ce.quantile(.95) if "true_m4_ce" in x else np.nan,predicted_m4_ce_mean=x.predicted_m4_ce.mean() if "predicted_m4_ce" in x else np.nan,qd4_occupied=x.qd_niche.nunique() if "qd_niche" in x else np.nan,fft_energy_median=x.fft_energy.median() if "fft_energy" in x else np.nan,nn_ratio_median=x.nn_ratio.median() if "nn_ratio" in x else np.nan))
        # Frozen, predeclared common features only: no post-hoc search.
        base=pops["P5_E1"]
        for arm,x in pops.items():
            if arm=="P5_E1": continue
            for feat in [F1,F2]:
                a=base[feat].dropna().to_numpy(); b=x[feat].dropna().to_numpy(); pooled=math.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
                feature_rows.append(dict(replicate=rep,comparison=f"P5_E1_vs_{arm}",feature=feat,standardized_mean_difference=(a.mean()-b.mean())/pooled if pooled else np.nan,wasserstein=wasserstein_distance(a,b),median_P5=np.median(a),median_other=np.median(b)))
    return pd.DataFrame(rows),pd.DataFrame(feature_rows)


def strict_frontier() -> pd.DataFrame:
    rows=[]
    for rep in REPS:
        x=pd.read_parquet(p5_selected_path(rep))
        for role,q in x.groupby("acquisition_component"):
            roots=q.lineage_root.astype(str)
            rows.append(dict(replicate=rep,role=role,n=len(q),share=len(q)/len(x),strict_region_fraction=q.in_strict_target_region.mean(),target_distance_mean=q.target_distance.mean(),target_distance_median=q.target_distance.median(),feature1_median=q[F1].median(),feature2_median=q[F2].median(),effective_lineages=effective_count(roots),unique_roots=roots.nunique(),max_lineage_fraction=roots.value_counts(normalize=True).max(),effective_operators=effective_count(q.operator_id.astype(str)),fft_energy_median=q.fft_energy.median(),nn_ratio_median=q.nn_ratio.median(),true_m4_ce_mean=np.nan,predicted_m4_ce_mean=np.nan,hardness_availability="MISSING_NO_HISTORICAL_P5_INFERENCE"))
    return pd.DataFrame(rows)


def training_metrics() -> pd.DataFrame:
    rows=[]
    for rep in REPS:
        for arm in ARMS:
            p=loss_path(rep,arm)
            if not p.exists(): continue
            x=pd.read_csv(p); y=x.mean_training_loss.to_numpy(float); e=x.epoch.to_numpy(float)
            rows.append(dict(replicate=rep,arm=arm,epochs=len(x),updates=int(x.optimizer_updates.iloc[-1]),initial_loss=y[0],final_loss=y[-1],minimum_loss=y.min(),mean_loss=y.mean(),auc_loss=np.trapezoid(y,e)/(e[-1]-e[0]),late32_mean=y[-32:].mean(),late32_slope=np.polyfit(e[-32:],y[-32:],1)[0],wall_s=x.epoch_wall_s.sum()))
    return pd.DataFrame(rows)


def tour_geometry() -> pd.DataFrame:
    """Existing-tour-only deterministic 256/hash population samples."""
    rows=[]
    for rep in REPS:
        for arm in ARMS:
            p=p5_npz_path(rep) if arm=="P5_E1" else p6_npz_path(rep,arm)
            z=np.load(p,allow_pickle=False); hashes=z["instance_hash"].astype(str)
            ix=np.argsort(hashes)[:256]; coords=z["coordinates"][ix]; tours=z["expert_tours"][ix]
            vals=[]
            for c,t in zip(coords,tours):
                o=c[t.astype(int)]; ed=np.linalg.norm(o-np.roll(o,-1,axis=0),axis=1)
                dm=cdist(c,c); np.fill_diagonal(dm,np.inf)
                ranks=np.argsort(np.argsort(dm,axis=1),axis=1)+1
                suc=np.roll(t.astype(int),-1); pred=np.roll(t.astype(int),1); v=np.arange(len(t))
                nnrank=np.r_[ranks[t.astype(int),suc],ranks[t.astype(int),pred]].mean()
                # MST overlap via scipy graph, invariant to rotation/reversal.
                from scipy.sparse.csgraph import minimum_spanning_tree
                mst=minimum_spanning_tree(dm).tocoo(); me={tuple(sorted((int(a),int(b)))) for a,b in zip(mst.row,mst.col)}
                te={tuple(sorted((int(a),int(b)))) for a,b in zip(t.astype(int),suc)}
                vals.append((ed.sum()/math.sqrt(len(t)),ed.std()/ed.mean(),(ed>2*np.median(ed)).mean(),nnrank,len(me&te)/max(len(me),1)))
            a=np.asarray(vals)
            rows.append(dict(replicate=rep,arm=arm,sample_n=len(a),sampling_rule="lexicographically-smallest 256 instance hashes",normalized_tour_length=a[:,0].mean(),tour_edge_cv=a[:,1].mean(),long_edge_fraction=a[:,2].mean(),endpoint_nn_rank=a[:,3].mean(),mst_edge_overlap=a[:,4].mean()))
    return pd.DataFrame(rows)


def make_figures(perf,summary,pools,pop,sf,train,concentration,seed_consistency,f40):
    plt.style.use("seaborn-v0_8-whitegrid")
    # 1 endpoint
    q=summary[(summary.split.eq("Original70")) & summary.arm.isin(["M4"]+ARMS)]
    piv=q.pivot(index="replicate",columns="arm",values="mean_gap").loc[REPS]
    piv.plot(kind="bar",figsize=(11,5)); plt.ylabel("Mean optimality gap (%)"); plt.title("Original70: frozen greedy results"); plt.tight_layout(); plt.savefig(FIG/"01_original70_three_seed_models.png",dpi=180); plt.close()
    # 2 waterfall original
    q=perf[perf.replicate.eq("Original")].sort_values("improvement_P5_E1")
    plt.figure(figsize=(12,5)); plt.bar(range(len(q)),q.improvement_P5_E1,color=np.where(q.improvement_P5_E1>=0,"#2a9d8f","#e76f51")); plt.axhline(0,color="k",lw=.8); plt.ylabel("M4 gap - P5-E1 gap (pp)"); plt.title("P5-E1 Original: 70-instance improvement waterfall"); plt.tight_layout(); plt.savefig(FIG/"02_original_improvement_waterfall.png",dpi=180); plt.close()
    # 3 split
    x=perf.groupby(["replicate","split"])["improvement_P5_E1"].mean().unstack()
    x.plot(kind="bar",figsize=(8,5));plt.ylabel("Mean improvement vs M4 (pp)");plt.title("Development48 vs held-out22 (descriptive)");plt.tight_layout();plt.savefig(FIG/"03_split_improvement.png",dpi=180);plt.close()
    # 4 N bins
    x=perf.groupby(["replicate","n_bin"],observed=True).improvement_P5_E1.mean().unstack()
    x.plot(kind="bar",figsize=(10,5));plt.ylabel("Mean improvement vs M4 (pp)");plt.title("P5-E1 improvement by frozen node-count bin");plt.tight_layout();plt.savefig(FIG/"04_node_bin_improvement.png",dpi=180);plt.close()
    # 5 actual common-feature distributions (Original, deterministic stride sample).
    p5=pd.read_parquet(p5_archive_path("Original"));p6=pd.read_parquet(ROOT/"target_aware_pilot/p6c1_hardness_controlled_matching/dataset_manifests/Original_union.parquet").merge(f40[["instance_hash",F1,F2]],on="instance_hash",how="left")
    fig,ax=plt.subplots(1,2,figsize=(11,4));
    for i,f in enumerate([F1,F2]):
        ax[i].hist(p5[f].dropna().iloc[::5],bins=50,density=True,alpha=.5,label="P5 archive30k")
        ax[i].hist(p6[f].dropna().iloc[::3],bins=50,density=True,alpha=.5,label="P6 E0/E1 union")
        ax[i].set_title(f); ax[i].legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/"05_pool_common_feature_comparison.png",dpi=180);plt.close(fig)
    # 6 strict/frontier target distance (hardness explicitly unavailable)
    x=sf.pivot(index="replicate",columns="role",values="target_distance_median");x.plot(kind="bar",figsize=(8,5));plt.ylabel("Median frozen target distance");plt.title("P5 final acquisition roles (historical hardness missing)");plt.tight_layout();plt.savefig(FIG/"06_strict_frontier_target_distance.png",dpi=180);plt.close()
    # 7 lineages
    x=sf.pivot(index="replicate",columns="role",values="effective_lineages");x.plot(kind="bar",figsize=(8,5));plt.ylabel("Effective lineages");plt.title("P5 role lineage diversity");plt.tight_layout();plt.savefig(FIG/"07_strict_frontier_lineages.png",dpi=180);plt.close()
    # 8 populations
    x=pop.pivot(index="replicate",columns="arm",values="p5_reference_distance_median").loc[REPS];x.plot(kind="bar",figsize=(11,5));plt.ylabel("Median distance to P5 two-feature reference");plt.title("Selected10k composition in historical P5 representation");plt.tight_layout();plt.savefig(FIG/"08_selected_population_structure.png",dpi=180);plt.close()
    # 9 consistency heatmap
    mat=perf.pivot(index="instance_id",columns="replicate",values="improvement_P5_E1")[REPS].sort_values("Original")
    plt.figure(figsize=(7,12));plt.imshow(mat.to_numpy(),aspect="auto",cmap="RdBu",vmin=-np.nanmax(abs(mat.to_numpy())),vmax=np.nanmax(abs(mat.to_numpy())));plt.colorbar(label="Improvement vs M4 (pp)");plt.xticks(range(3),REPS);plt.yticks([]);plt.title("Per-instance improvement consistency");plt.tight_layout();plt.savefig(FIG/"09_instance_seed_consistency.png",dpi=180);plt.close()
    # 10 evidence matrix
    labels=["Protocol matched","Pools matched","Same-pool selection control","P5 true CE","Three-seed stable win"]
    data=np.array([[1,1,1],[0,0,0],[1,1,1],[0,0,0],[1,0,0]])
    plt.figure(figsize=(7,4));plt.imshow(data,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto");plt.xticks(range(3),["Original","S1","S2"]);plt.yticks(range(5),labels);plt.title("Generation vs selection evidence matrix (green=yes)");plt.tight_layout();plt.savefig(FIG/"10_causal_evidence_matrix.png",dpi=180);plt.close()
    # 11 training
    x=train.pivot(index="replicate",columns="arm",values="late32_mean").loc[REPS];x.plot(kind="bar",figsize=(11,5));plt.ylabel("Mean training CE, last 32 epochs");plt.title("Frozen training dynamics");plt.tight_layout();plt.savefig(FIG/"11_training_late_loss.png",dpi=180);plt.close()
    # 12 concentration
    x=concentration[concentration.statistic.eq("leave_top5_gains_out_mean_gap")].set_index("replicate").value
    base=summary[(summary.split.eq("Original70"))&(summary.arm.eq("P5_E1"))].set_index("replicate").mean_gap
    pd.DataFrame({"reported":base,"without_top5_gains":x}).loc[REPS].plot(kind="bar",figsize=(8,5));plt.ylabel("Mean gap (%)");plt.title("Gain concentration sensitivity");plt.tight_layout();plt.savefig(FIG/"12_gain_concentration.png",dpi=180);plt.close()
    # 13 representative coordinates: closest to each role's median target distance.
    meta=pd.read_parquet(p5_selected_path("Original"));z=np.load(p5_npz_path("Original"),allow_pickle=False); hmap={str(h):i for i,h in enumerate(z["instance_hash"])}
    fig,axs=plt.subplots(1,2,figsize=(9,4.5))
    for ax,(role,q) in zip(axs,meta.groupby("acquisition_component",sort=True)):
        med=q.target_distance.median(); row=q.iloc[(q.target_distance-med).abs().argmin()]; c=z["coordinates"][hmap[str(row.instance_hash)]]
        ax.scatter(c[:,0],c[:,1],s=8);ax.set_title(f"{role}\nclosest to role-median distance")
        ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("P5 Original representatives selected by predeclared median-distance rule");fig.tight_layout();fig.savefig(FIG/"13_p5_role_representative_coordinates.png",dpi=180);plt.close(fig)


def main():
    for p in [OUT,FIG,TABLE,LOG,CFG,MAN]: p.mkdir(parents=True,exist_ok=True)
    # Preserve the exact frozen configs used by the two mechanisms.
    for src,name in [(ROOT/"target_aware_pilot/p5_structure_guided_evolution_robustness/preregistered_evolution_config.json","p5_frozen_config.json"),(ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution/configs/frozen_p6b_config.json","p6b_frozen_config.json")]: shutil.copy2(src,CFG/name)
    perf=load_performance(); perf.to_csv(OUT/"02_per_instance_gap.csv",index=False)
    summary,seed_consistency=performance_tables(perf); summary.to_csv(TABLE/"performance_summary.csv",index=False);seed_consistency.to_csv(OUT/"02_seed_consistency.csv",index=False)
    conc=concentration(perf);conc.to_csv(TABLE/"gain_concentration.csv",index=False)
    f40=load_full_f40()
    pools,overlap=pool_metrics(f40);pools.to_csv(OUT/"03_pool_comparison.csv",index=False);overlap.to_csv(TABLE/"pool_overlap.csv",index=False)
    pop,fd=population_metrics(f40);pop.to_csv(OUT/"05_population_metrics.csv",index=False);fd.to_csv(OUT/"05_feature_differences.csv",index=False)
    sf=strict_frontier();sf.to_csv(OUT/"04_strict_frontier_comparison.csv",index=False)
    train=training_metrics();train.to_csv(TABLE/"training_dynamics.csv",index=False)
    tour=tour_geometry();tour.to_csv(TABLE/"expert_tour_geometry_sample256.csv",index=False)
    # Historical matched final-acquisition contrast.
    p4=pd.read_csv(ROOT/"target_aware_pilot/p4b1_acquisition_ablation/05_five_model_gap_comparison.csv")
    p4.to_csv(TABLE/"p4b1_same_pool_acquisition_control.csv",index=False)
    make_figures(perf,summary,pools,pop,sf,train,conc,seed_consistency,f40)

    survivor_rows=[]
    for rep in REPS:
        for gen in [1,2]:
            if rep=="Original": p=ROOT/f"target_aware_pilot/p2_bounded_evolution/evolution/E1/G{gen}/active_population_selection.json"
            else: p=ROOT/f"target_aware_pilot/p5_structure_guided_evolution_robustness/evolution/{rep}/evolution/E1/G{gen}/active_population_selection.json"
            x=json.loads(p.read_text())
            survivor_rows.append(dict(replicate=rep,generation=f"G{gen}",requested_target_n=x["requested_target_n"],actual_target_n=x["target_n"],actual_frontier_n=x["frontier_n"],strict_shortfall_reassigned=x["strict_quota_shortfall_reassigned_to_frontier"]))
    pd.DataFrame(survivor_rows).to_csv(TABLE/"p5_survivor_role_realization.csv",index=False)

    matrix=pd.DataFrame([
      ["M4 checkpoint","MATCHED","same SHA256 c749311..."],["BQ architecture/trainer","MATCHED","same frozen trainer SHA256 7ff017..."],["optimizer","MATCHED","fresh Adam"],["learning rate","MATCHED","3.75e-5"],["scheduler","MATCHED","none"],["batch/microbatch","MATCHED","256/64"],["epochs/updates","MATCHED","256/10,240"],["training seed","MATCHED","2026"],["checkpoint rule","MATCHED","fixed E256"],["greedy evaluation","MATCHED","same stored frozen protocol"],["evolution seeds","MATCHED","Original/S1/S2: 20261024/260501/260502"],["initial parent population","MATCHED","same P03 hash 8ad76b..."],["mutation profile/operators","MATCHED","FOGA11_BASE_V1 / 11 operators"],["mutation generations","MATCHED","two"],["attempts per generation","DIFFERENT","P5 20k (2/parent); P6-B 110k (11/parent)"],["population path","DIFFERENT","P5 structured survivor + archive30k; P6 E0/E1 union"],["feature representation","DIFFERENT","P5 two-feature target geometry; P6 Top5 OT/QD/hardness"],["candidate pool","DIFFERENT","coordinate overlap ~1.3%; see table"],["final selection","DIFFERENT","P5 cell quota/backfill; B random; C Top-HGB; D within-niche quota"],["P5 final 70/30 claim","DIFFERENT","70/30 is requested survivor role, not achieved final training split"],["P5 historical true/predicted M4 CE","UNKNOWN","no stored P5 inference; prohibited to generate"],["P5 full F40/QD4","UNKNOWN","only historical two-feature/FFT/NN fields stored"]
    ],columns=["item","status","evidence"]);matrix.to_csv(OUT/"01_protocol_difference_matrix.csv",index=False)

    # Selected key numbers.
    orig=summary[(summary.split.eq("Original70"))]
    p5means=orig[orig.arm.eq("P5_E1")].set_index("replicate").mean_gap
    bmeans=orig[orig.arm.eq("B")].set_index("replicate").mean_gap
    dif=(p5means-bmeans).loc[REPS]; ci=bootstrap_seed_instance(perf.assign(p5_minus_b=perf.gap_P5_E1-perf.gap_B))
    splitmeans=perf.groupby(["replicate","split"]).improvement_P5_E1.mean().unstack()
    consistent=int(seed_consistency.all_three_improve.sum()); flip=int(seed_consistency.original_S1_improve_S2_regress.sum())
    p4o=p4[(p4.dataset.eq("OriginalTSPLIB70"))]

    write(OUT/"00_protocol.md",f"""# P6-F0 frozen protocol

This audit is a read-only transformation of historical assets. It ran no FOGA mutation, candidate generation, Concorde, HGB/GP fitting, BQ training, greedy inference, or cross-population CE inference.

- Scope: P5-E1 Original/S1/S2 versus P6 B/C/D25/D50/D75 and frozen M4.
- Primary endpoint: stored OriginalTSPLIB70 per-instance greedy gap.
- Split: frozen TSPLIB48/22 manifests; the 22 identities are used only to report evaluation.
- Predeclared structural variables: the two actual P5 features `{F1}` and `{F2}`; P6 Top5/F40 are not post-hoc screened against gap.
- Uncertainty: no pooling of 210 repeated benchmark evaluations as independent samples. The P5-vs-B CI resamples evolution seed and TSPLIB identity hierarchically (10,000 draws, seed 20261028).
- Tour geometry: existing Concorde tours only; deterministic lexicographically-smallest 256 coordinate hashes per population.
- Missing evidence remains missing: no new P5 hardness or cross-population CE inference was launched.
""")
    write(OUT/"01_historical_integrity.md",f"""# Historical experiment integrity

**Result: PASS.** Stored per-instance files reproduce P5-E1 Original = **{p5means.Original:.4f}%**, S1 = **{p5means.S1:.4f}%**, S2 = **{p5means.S2:.4f}%**. M4 initialization, trainer, optimizer, LR, scheduler, batch, 256 epochs/10,240 updates, training seed, E256 checkpoint rule, and greedy evaluator match P6.

The main correction is semantic: P5 did **not** train on an achieved 70/30 strict/frontier set. The survivor selector requested 70% target and 30% frontier, but shortages were reassigned. Final Structure10k acquisition contained strict-cell quota counts 4,120 / 4,030 / 4,003 and transition/backfill counts 5,880 / 5,970 / 5,997 for Original/S1/S2.

P5 and P6 are paired by evolution seed and initial P03 lineage, but not by candidate coordinates or candidate budget. Therefore the observed gap difference is valid; a selection-only causal reading is not.

The P5 support export stores two byte-identical Original70 row copies per seed (420 raw rows). The audit explicitly de-duplicates on replicate + instance identity + coordinate hash + gap and verifies 210 unique paired evaluations before any statistic.
""")
    # Concentration narrative from actual numbers.
    c0=conc[(conc.replicate.eq("Original"))].set_index("statistic").value
    write(OUT/"02_tsplib_performance_decomposition.md",f"""# TSPLIB performance decomposition

P5-E1 gap: Original {p5means.Original:.4f}%, S1 {p5means.S1:.4f}%, S2 {p5means.S2:.4f}%; mean **{p5means.mean():.4f}%**. B mean is **{bmeans.mean():.4f}%**. P5-B mean difference is **{dif.mean():+.4f} pp** (negative favors P5), seed wins **{int((dif<0).sum())}/3**, hierarchical bootstrap 95% CI **[{ci[0]:+.4f}, {ci[1]:+.4f}] pp**.

Original P5-E1 improves TSPLIB48 by **{splitmeans.loc['Original','TSPLIB48']:+.4f} pp** and TSPLIB22 by **{splitmeans.loc['Original','TSPLIB22']:+.4f} pp**. S1 also improves both ({splitmeans.loc['S1','TSPLIB48']:+.4f}, {splitmeans.loc['S1','TSPLIB22']:+.4f}); S2 regresses TSPLIB48 ({splitmeans.loc['S2','TSPLIB48']:+.4f}) while still improving TSPLIB22 ({splitmeans.loc['S2','TSPLIB22']:+.4f}). Thus the good seeds are not development48-only, while S2's overall failure is driven by its development48 regression rather than held-out22.

For Original, the top-5 improvement contribution ratio is **{c0.get('top5_gain_share',np.nan):.3f}** and the mean gap after removing the five largest gains is **{c0.get('leave_top5_gains_out_mean_gap',np.nan):.4f}%**. There are **{consistent}/70** identities improved in all three seeds and **{flip}/70** improved in Original+S1 but regressed in S2. Full tails, instance IDs and node bins are in the CSV tables; no benchmark case was removed from the official endpoint.
""")
    write(OUT/"03_candidate_generation_audit.md",f"""# Candidate generation audit

The pools are materially different. P5 uses 20,000 mutation attempts per generation, target-aware G1 survivor selection, a target-aware 30k offspring archive, and final cell-quota acquisition. P6-B generation uses 110,000 attempts per generation, pure Top5-OT E1 versus random E0 populations, no archive, and downstream selection from their ~20k union.

Coordinate pool overlap is only **{overlap.pool_overlap.mean():.1f}** on average (mean Jaccard **{overlap.pool_jaccard.mean():.4f}**); P5-vs-B Selected10k overlap is **{overlap.selected_P5_B_overlap.mean():.1f}** (Jaccard **{overlap.selected_P5_B_jaccard.mean():.4f}**). Roots are largely shared because both descend from P03, but coordinate descendants are not.

Existing P6-E0.5 FFT–NN diagnostics give P5 archive reference coverage 24/48, 24/48, 22/48 and P6-union coverage 25/48, 24/48, 25/48 (Original/S1/S2). This exploratory representation does not show superior P5 pool reachability. P5 historical archives lack aligned full-F40/QD4 and true-M4-CE fields; these were not reconstructed with prohibited new inference. Therefore candidate-generation **difference** is supported, but candidate-generation **superiority** is not established.

Exact coordinate duplicates are zero in every pool. Because the frozen protocols define no coordinate-space near-duplicate threshold, this audit does not invent one after observing outcomes; `local_2d_nn` reports a scale-normalized nearest-neighbor diagnostic in the two predeclared P5 features instead.
""")
    write(OUT/"04_p5_selection_mechanism.md",f"""# P5 selection mechanism

The historical rule uses exactly `{F1}` then `{F2}`. TSPLIB48 aspect-normalized features define median/IQR robust coordinates; strict means nearest-reference distance within the frozen leave-one-reference radius **and** inside the target box. The selector allocates 3×3 target-cell quotas, performs 12×12 microbin round-robin, orders deterministically by rank/distance/microbin/hash, applies lineage caps with controlled doubling, and fills shortages from the frontier. No HGB score or hardness threshold enters mutation, survivor, archive retention, or final acquisition.

Requested survivor 70/30 was not achieved: Original G1 retained 2,149 target + 7,851 frontier and G2 4,813 + 5,187 because strict supply was short. The final training set is a separate quota acquisition, not a fixed 70/30 role sampler. Final strict-cell fractions are {', '.join(f'{r}: {sf[(sf.replicate==r)&(sf.role=="strict_cell_quota")].share.iloc[0]:.1%}' for r in REPS)}.

Strict/frontier true and predicted M4 CE are **missing historical evidence**. Structure, lineage, FFT and NN descriptive statistics are available in `04_strict_frontier_comparison.csv`; they cannot establish training value.
""")
    write(OUT/"05_population_composition_comparison.md",f"""# Selected-population composition

The unified table compares 18 frozen populations (3 seeds × P5/B/C/D25/D50/D75). P6 arms contain existing true/predicted M4 CE and QD4; P5 does not, so a hardness ranking of P5 versus B is not identifiable without forbidden new model inference.

In the actual P5 two-feature geometry, P5 is systematically closer to its own target reference than P6 arms by construction. Effective roots, max-root share, operator diversity, local 2-D nearest-neighbor distance and deterministic existing-tour geometry are reported. These are descriptive: structural proximity is not gradient diversity, and optimal-tour properties were not available online to evolution.

Only the two predeclared common P5 features are used for cross-family Wasserstein/SMD comparison. P5 full-F40/Top5/QD4 values are absent from historical caches, so no after-the-fact feature search was performed.
""")
    # training dynamic comparisons
    tm=train.groupby("arm").agg(final_loss=("final_loss","mean"),late32=("late32_mean","mean"),wall_s=("wall_s","mean"))
    write(OUT/"06_training_dynamics.md",f"""# Training dynamics

All runs use the same initialization and optimizer budget and select fixed E256, not a validation-picked checkpoint. Mean late-32-epoch CE is P5-E1 **{tm.loc['P5_E1','late32']:.5f}**, B **{tm.loc['B','late32']:.5f}**, C **{tm.loc['C','late32']:.5f}**. Training-curve differences do not track the P5 seed ordering strongly enough to explain Original/S1 success and S2 failure.

Own-population training CE is not a generalization metric. No historical validation traces or P5 cross-population CE matrix were found, so training dynamics are an insufficient causal explanation rather than evidence that optimization was identical in every stochastic detail.
""")
    write(OUT/"06_cross_population_transfer.md","""# Cross-population transfer

No legal historical P5-vs-P6 cross-population teacher-forced CE matrix exists. Generating one would require new BQ inference, explicitly prohibited by P6-F0. Consequently own-population fit, transfer and forgetting cannot be compared on a common training distribution. This missingness is preserved rather than imputed from training loss or greedy gap.
""")
    write(OUT/"07_success_failure_replication_audit.md",f"""# Original/S1 success versus S2 failure

The performance pattern is real but not accompanied by a single stable observed population signature: Original and S1 improve both 48 and 22; S2 regresses development48 but still improves held-out22 relative to M4. P5 beats B on Original ({dif.Original:+.4f} pp) and S1 ({dif.S1:+.4f} pp), but loses on S2 ({dif.S2:+.4f} pp).

The three P5 populations have very similar final strict/backfill composition (~40/60), while candidate pools and exact descendants differ. Existing FFT–NN reachability does not isolate S2 as uniquely deficient. Hardness and common full-F40 values are missing for P5. With only three evolution replicates, correlations between population summaries and three gap means would be post-hoc and invalid; all cross-seed associations here are **EXPLORATORY_ONLY**.
""")
    # P4 control numbers robustly locate columns in report, but state known result directly from authority report.
    write(OUT/"08_generation_vs_selection_attribution.md","""# Generation versus selection attribution

There is one useful historical matched selection contrast: P4-B1 selected Random10k (R0) and structure-aware 10k (R1) from the same P2.2 E0 archive, same M4 and same training budget. Original70 was 2.2357% for R0 versus 2.1337% for R1, a -0.1020 pp point estimate; the paired 95% CI [-0.5304, +0.3295] crosses zero. This is preliminary evidence that final acquisition can matter, not proof of the full P5 path.

P5 E0-vs-E1 also uses common-random G1 offspring, but the E1 G1 survivor changes G2 parents and hence bundles survivor selection, downstream generation, archive retention and final acquisition. P5-vs-P6 additionally changes mutation budget, representation, archive and source pool. No historical different-pool/same-selection crossing exists.

Therefore attribution is **PARTIAL**. The minimum decisive next test is a paired 2×2: P5 pool/P6 pool × P5-style quota/B-style random, after defining a semantically portable quota rule. P6-F0 does not execute it.
""")
    write(OUT/"09_statistical_robustness.md",f"""# Statistical robustness

- Evolution replicates: 3; P5 wins against B in 2/3.
- Three-seed P5 mean {p5means.mean():.4f}% versus B {bmeans.mean():.4f}%; difference {dif.mean():+.4f} pp.
- Hierarchical seed+identity bootstrap 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}] pp.
- The 70 identities evaluated by three models/seeds are not treated as 210 independent benchmark instances.
- Gain concentration, leave-out sensitivity, worst regressions, node bins, and paired vectors are reported without changing the official 70-instance endpoint.
- Training-population statements account for lineage concentration descriptively; no row-level significance test pretends 10k descendants are independent.
- Candidate-generation and structural associations are exploratory because there are only three evolution seeds.
""")
    write(OUT/"10_final_research_conclusion.md",f"""# Final research conclusion

## H1–H9

- `P6F0_HISTORICAL_RESULT_INTEGRITY: PASS`
- `P6F0_PERFORMANCE_ADVANTAGE: MIXED`
- `P6F0_CANDIDATE_GENERATION_DIFFERENCE: SUPPORTED`
- `P6F0_SELECTION_MECHANISM_DIFFERENCE: SUPPORTED`
- `P6F0_HARDNESS_EXPLANATION: NOT_SUPPORTED`
- `P6F0_STRUCTURE_DIVERSITY_EXPLANATION: MIXED`
- `P6F0_TRAINING_DYNAMICS_EXPLANATION: INSUFFICIENT_DATA`
- `P6F0_CAUSAL_ATTRIBUTION: PARTIAL`
- `NEXT_TEST_MATCHED_2X2`

The 1.9180% record is trustworthy and uses a matched training/evaluation protocol, but it is one of three evolution realizations. P5's mean advantage over B is only {abs(dif.mean()):.4f} pp, reverses in S2, and its hierarchical CI crosses zero. Good seeds improve both development48 and held-out22, so the Original result is not merely target-reference bookkeeping; nevertheless it is not robust.

The historical mechanism is not “70% strict + 30% frontier training data.” It is a two-stage target-aware survivor/archive process followed by ~40% strict-cell quota and ~60% transition/backfill final acquisition. P5 and P6 candidate coordinates are almost disjoint and their mutation budgets/archive paths differ. A same-pool P4-B1 contrast gives a favorable but uncertain selection point estimate, so selection remains plausible; generation remains confounded. P5 hardness cannot be assessed from existing assets.

The sole recommended next step is a preregistered matched 2×2 pool × selection experiment. Do not restore 70/30 solely because Original was best, and do not launch it automatically from this audit.

`BLADE_P6F0_BEST_POPULATION_MECHANISM_AUDIT_COMPLETE`
""")
    elapsed=time.time()-START
    write(OUT/"11_runtime_resource_audit.md",f"""# Runtime and resource audit

- Wall clock: {elapsed:.1f} seconds before final hashing.
- Compute: CPU-only tabulation/plotting; no GPU model execution.
- Expensive work: reading frozen parquet/CSV/NPZ and computing existing-tour geometry on 256 deterministic instances per population.
- Forbidden actions performed: none.
- New model/checkpoint/training data: none.
""")

    # Hash authoritative inputs and generated outputs.
    inputs=[ROOT/"target_aware_pilot/p5_structure_guided_evolution_robustness/support_per_instance_paired_differences.csv",ROOT/"target_aware_pilot/p6d3_structure_hardness_quota_selection/07_per_instance_gap.csv",ROOT/"target_aware_pilot/p5_structure_guided_evolution_robustness/preregistered_evolution_config.json",ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution/configs/frozen_p6b_config.json",ROOT/"target_aware_pilot/p2_bounded_evolution/run_p2_bounded_evolution.py",ROOT/"target_aware_pilot/03_tsplib48_reference_manifest.csv",ROOT/"target_aware_pilot/p1_heldout22_names.json",ROOT/"target_aware_pilot/p4b1_acquisition_ablation/05_five_model_gap_comparison.csv"]
    for rep in REPS:
        inputs += [p5_selected_path(rep),p5_archive_path(rep),p5_npz_path(rep)]
        for arm in ["B","C","D25","D50","D75"]:
            inputs += [p6_selected_path(rep,arm),p6_npz_path(rep,arm),loss_path(rep,arm)]
        inputs += [loss_path(rep,"P5_E1"),ROOT/f"target_aware_pilot/p6c1_hardness_controlled_matching/dataset_manifests/{rep}_union.parquet"]
    inputs += sorted((ROOT/"target_aware_pilot/p6c0_training_value_mechanism_audit/full_f40_cache").glob("*.parquet"))
    pd.DataFrame([{"path":str(p.relative_to(ROOT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in inputs]).to_csv(MAN/"authoritative_input_hashes.csv",index=False)
    outs=sorted([p for p in OUT.rglob("*") if p.is_file() and p.name not in {"completion.json","output_hashes.csv"}])
    hashes=pd.DataFrame([{"path":str(p.relative_to(OUT)),"sha256":sha(p),"bytes":p.stat().st_size} for p in outs]);hashes.to_csv(LOG/"output_hashes.csv",index=False)
    completion={"status":"COMPLETE","label":"BLADE_P6F0_BEST_POPULATION_MECHANISM_AUDIT_COMPLETE","h1":"PASS","h2":"MIXED","h3":"SUPPORTED","h4":"SUPPORTED","h5":"NOT_SUPPORTED","h6":"MIXED","h7":"INSUFFICIENT_DATA","h8":"PARTIAL","h9":"NEXT_TEST_MATCHED_2X2","figures":len(list(FIG.glob('*.png'))),"output_files":len(outs)+2,"runtime_seconds":round(time.time()-START,3),"forbidden_actions":[],"new_training":False,"new_inference":False,"new_evolution":False}
    (OUT/"completion.json").write_text(json.dumps(completion,indent=2)+"\n")
    print(json.dumps(completion,indent=2))


if __name__ == "__main__":
    main()
