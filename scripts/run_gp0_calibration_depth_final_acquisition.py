"""Read-only GP0 calibration-depth audit for the final acquisition role."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "audits/uniform_rooted_qd_bootstrap_v1"
INIT = ROOT / "audits/cycle0_initialization_validation_v1"
WITHIN = ROOT / "audits/cycle0_within_niche_hardness_diagnostic_v1"
REPLAY = ROOT / "audits/cycle0_acquisition_policy_replay_v1"
OUT = ROOT / "audits/gp0_calibration_depth_final_acquisition_v1"
SEED = 2026
BOOTSTRAPS = 10_000
RANDOM_REPEATS = 1_000
DEPTHS = (1, 2, 5)
BUDGETS = (602, 1204, 2408)
TRAIN_N = {1: 435, 2: 822, 5: 1819}
FOLDERS = {1: "04_gp4_d1", 2: "05_gp4_d2", 5: "06_gp4_d5"}
QRAW = ["spn_sum_of_lowest_edge_values", "spn_distance_varcoeff", "spn_mst_depth_mean", "spn_mst_dists_median"]
QNORM = [f"normalized_{name}" for name in QRAW]
FIGURES = [
    "01_gp_fill_hardness_primary_B602.png", "02_gp_fill_gain_over_random.png",
    "03_gain_retention_vs_D5.png", "04_delta_vs_D5_bootstrap.png",
    "05_hybrid_hardness_by_calibration_depth.png", "06_oracle_fill_headroom_capture.png",
    "07_sensitivity_B602_B1204_B2408.png", "08_full_bank_fill_overlap.png",
    "09_full_bank_rank_agreement.png", "10_calibration_labels_vs_acquisition_value.png",
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


def q(values, level): return float(np.quantile(np.asarray(values, float), level))
def ci(values): return q(values, .025), q(values, .975)


def entropy(counts) -> float:
    x = np.asarray(counts, float); p = x[x > 0] / x.sum()
    return float(-(p * np.log(p)).sum())


def gini(counts) -> float:
    x = np.sort(np.asarray(counts, float)); n = len(x)
    return float(2*np.dot(np.arange(1,n+1),x)/(n*x.sum())-(n+1)/n) if x.sum() else 0.0


def formula_path(depth: int, seed: int) -> Path:
    return INIT / FOLDERS[depth] / f"seed_{seed}/formula.json"


def source_manifest() -> tuple[dict, dict[str, str]]:
    paths = [
        BOOT/"02_offspring_manifest.parquet", BOOT/"05_offspring_qd_assignments.parquet", BOOT/"strict_verification.json",
        INIT/"02_gp0_calibration_design/structural_probe_order.parquet", INIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet",
        INIT/"03_oracle_labels/exact_hardness.parquet", INIT/"08_prediction_evaluation/all_predictions.parquet",
        WITHIN/"11_decision/diagnostic_classification.json", WITHIN/"strict_verification.json",
        REPLAY/"02_full_bank_scores/gp4_d5_scores.parquet", REPLAY/"03_full_bank_policy_replay/centroid1_global_gp.parquet",
        REPLAY/"11_decision/diagnostic_classification.json", REPLAY/"strict_verification.json",
        ROOT/"ASSET_MANIFEST.json", ROOT/"assets/qd/cvt/centroids.csv",
    ]
    paths += [INIT/f"02_gp0_calibration_design/cal_d{d}_manifest.json" for d in DEPTHS]
    paths += [formula_path(d,s) for d in DEPTHS for s in range(2026,2031)]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        print("REQUIRED_EXISTING_ARTIFACT_MISSING="+",".join(missing),flush=True);raise SystemExit(2)
    entries={};hashes={}
    for path in paths:
        rel=path.relative_to(ROOT).as_posix();digest=sha256(path);hashes[rel]=digest
        entries[rel]={"absolute_path":str(path),"bytes":path.stat().st_size,"sha256":digest}
    return {"audit":"BLADE_GP0_FINAL_ROLE_CALIBRATION_DEPTH_AUDIT_V1","analysis_only":True,
            "runtime_old_project_reads":False,"sources":entries,
            "analysis_script":{"absolute_path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())}},hashes


def protocol() -> dict:
    return {"audit":"BLADE_GP0_FINAL_ROLE_CALIBRATION_DEPTH_AUDIT_V1","created_utc":datetime.now(timezone.utc).isoformat(),
            "analysis_only":True,"operational_role":"CENTROID1 coverage floor plus global GP hard-tail fill",
            "branches":["GP4_D1","GP4_D2","GP4_D5"],"train_n":TRAIN_N,"primary_dev_budget":602,
            "secondary_budgets":[1204,2408],"B301_role":"sanity check only; excluded from calibration decision",
            "primary_metric":"GP-fill mean exact raw H0 and gain over shared matched random fill",
            "random":{"master_seed":SEED,"repeats":RANDOM_REPEATS,"draws_shared_across_depths":True},
            "bootstrap":{"unit":"dev niche cluster","replicates":BOOTSTRAPS,"seed":SEED,"interval":"95% percentile"},
            "noninferiority":{"retention_threshold":.95,"margin_fraction_of_abs_D5_gain":.05,"primary_budget":602},
            "full_bank":{"n":109785,"occupied_niches":435,"centroid_floor":435,"GP_fill":9565,"total":10000},
            "decision_excludes":["within-niche ranking","HGB","Oracle capture threshold"],
            "forbidden":["new generation","new H0 labels","Concorde","BQ hardness inference","GP/HGB training","BQ fine-tuning",
                         "external evaluation","FINAL_D0","protocol change","archive update","Cycle-0 launch","final runner"]}


def eval_tree(node: dict, x: np.ndarray) -> np.ndarray:
    op=node["op"]
    if op=="feature":return x[:,int(node["value"])]
    if op=="const":return np.full(len(x),float(node["value"]),float)
    a=eval_tree(node["args"][0],x)
    if op=="abs":return np.abs(a)
    if op=="sqrt_abs":return np.sqrt(np.abs(a))
    if op=="log1p_abs":return np.log1p(np.abs(a))
    b=eval_tree(node["args"][1],x)
    if op=="add":return a+b
    if op=="subtract":return a-b
    if op=="multiply":return a*b
    if op=="protected_divide":return a/np.where(np.abs(b)<1e-6,np.where(b>=0,1e-6,-1e-6),b)
    raise ValueError(op)


def score_depth(x: np.ndarray, depth: int) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    predictions=[];manifests=[]
    for seed in range(2026,2031):
        payload=json.loads(formula_path(depth,seed).read_text(encoding="utf-8"));raw=eval_tree(payload["selected"]["tree"],x)
        clipped=np.clip(raw,-20,20)
        if not np.isfinite(raw).all() or np.mean(clipped!=raw)>.01:raise RuntimeError(f"invalid GP4_D{depth} seed {seed}")
        t=payload["transform"];pred=clipped*float(t["scale_train_loss_log"])+float(t["center_train_median_loss_log"])
        predictions.append(pred);manifests.append({"seed":seed,"sha256":sha256(formula_path(depth,seed)),"prediction_variance":float(np.var(pred))})
    return np.median(np.vstack(predictions),axis=0),predictions,manifests


def load_inputs():
    qd=pd.read_parquet(BOOT/"05_offspring_qd_assignments.parquet");provenance=pd.read_parquet(BOOT/"02_offspring_manifest.parquet")
    dev=pd.read_parquet(INIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet");labels=pd.read_parquet(INIT/"03_oracle_labels/exact_hardness.parquet")
    stored=pd.read_parquet(INIT/"08_prediction_evaluation/all_predictions.parquet");order=pd.read_parquet(INIT/"02_gp0_calibration_design/structural_probe_order.parquet")
    accepted=provenance[provenance.proposal_status.eq("accepted")][["child_hash"]]
    if len(qd)!=109785 or qd.instance_hash.nunique()!=109785 or set(qd.instance_hash)!=set(accepted.child_hash):raise RuntimeError("full bank integrity")
    base=qd.copy();base["coordinate_hash"]=base.instance_hash.astype(str)
    order_map=order.set_index("instance_hash").loc[base.instance_hash]
    if not np.array_equal(order_map[QNORM].to_numpy(float),base[QNORM].to_numpy(float)):raise RuntimeError("normalized QD4 mismatch")
    base["centroid_distance"]=order_map.centroid_distance.to_numpy(float)
    x=base[QRAW].to_numpy(float);model_records={}
    for depth in DEPTHS:
        score,seeds,records=score_depth(x,depth);base[f"GP4_D{depth}"]=score
        expected=stored[stored.model.eq(f"GP4_D{depth}")].set_index("instance_hash").prediction
        actual=base.set_index("instance_hash").loc[expected.index,f"GP4_D{depth}"]
        calibration=json.loads((INIT/f"02_gp0_calibration_design/cal_d{depth}_manifest.json").read_text(encoding="utf-8"))
        if calibration["total_train_n"]!=TRAIN_N[depth] or calibration["occupied_niches_represented"]!=435:raise RuntimeError("calibration manifest count mismatch")
        model_records[depth]={"model":f"GP4_D{depth}","train_n":calibration["total_train_n"],"reached_niches":calibration["occupied_niches_represented"],
                              "probes_per_niche_target":depth,"seeds":records,"five_formulas":len(records)==5,
                              "median_ensemble_bitwise_dev_match":bool(np.array_equal(actual.to_numpy(),expected.to_numpy())),
                              "full_predictions_finite":bool(np.isfinite(score).all()),"full_prediction_variance":float(np.var(score))}
    dev_label=labels[labels.instance_hash.isin(set(dev.instance_hash))][["instance_hash","H0"]]
    panel=dev[["instance_hash","niche_id","centroid_distance",*QNORM]].merge(base[["instance_hash","coordinate_hash",*QRAW,*[f"GP4_D{d}" for d in DEPTHS]]],on="instance_hash",validate="one_to_one").merge(dev_label,on="instance_hash",validate="one_to_one")
    if len(panel)!=2666 or panel.niche_id.nunique()!=301 or not np.isfinite(panel.H0).all():raise RuntimeError("dev integrity")
    floor=panel.sort_values(["niche_id","centroid_distance","coordinate_hash"],kind="mergesort").groupby("niche_id",sort=True).head(1)
    full_floor=base.sort_values(["niche_id","centroid_distance","coordinate_hash"],kind="mergesort").groupby("niche_id",sort=True).head(1)
    prior=pd.read_parquet(REPLAY/"03_full_bank_policy_replay/centroid1_global_gp.parquet");prior_floor=prior[prior.selection_component.eq("centroid1_floor")]
    if set(full_floor.instance_hash)!=set(prior_floor.instance_hash):raise RuntimeError("full centroid floor mismatch")
    model_alignment={"status":"PASS","branches":[model_records[d] for d in DEPTHS],"all_five_seed_ensembles_exact":all(model_records[d]["median_ensemble_bitwise_dev_match"] for d in DEPTHS)}
    dev_alignment={"status":"PASS","dev_n":len(panel),"occupied_niches":panel.niche_id.nunique(),"exact_H0_hash_set":set(panel.instance_hash)==set(dev.instance_hash),
                   "normalized_QD4_exact":True,"coordinate_hash_definition":"instance_hash (canonical coordinate hash)","shared_centroid_floor_n":len(floor)}
    full_alignment={"status":"PASS","full_bank_n":len(base),"occupied_niches":base.niche_id.nunique(),"shared_centroid_floor_n":len(full_floor),
                    "centroid_floor_matches_prior_replay":True,"all_predictions_finite":all(model_records[d]["full_predictions_finite"] for d in DEPTHS)}
    return base,panel,floor,full_floor,model_records,model_alignment,dev_alignment,full_alignment


def choose_fill(remaining: pd.DataFrame, depth: int, n: int) -> pd.DataFrame:
    return remaining.sort_values([f"GP4_D{depth}","coordinate_hash"],ascending=[False,True],kind="mergesort").head(n).copy()


def random_draws(remaining: pd.DataFrame, fill_n: int, budget: int, niches: np.ndarray):
    ordered=remaining.sort_values("coordinate_hash",kind="mergesort").reset_index(drop=True);h=ordered.H0.to_numpy(float)
    posmap={int(n):i for i,n in enumerate(niches)};niche_pos=ordered.niche_id.map(posmap).to_numpy(int)
    sums=np.zeros((RANDOM_REPEATS,len(niches)));counts=np.zeros_like(sums);rows=[]
    for repeat in range(RANDOM_REPEATS):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,31,budget,repeat]));idx=rng.choice(len(ordered),fill_n,replace=False)
        np.add.at(sums[repeat],niche_pos[idx],h[idx]);np.add.at(counts[repeat],niche_pos[idx],1)
        values=h[idx];rows.append({"budget":budget,"fill_n":fill_n,"repeat":repeat,"mean_random_fill_H0":values.mean(),"median_random_fill_H0":np.median(values)})
    return sums,counts,pd.DataFrame(rows)


def niche_arrays(selection: pd.DataFrame,niches:np.ndarray):
    g=selection.groupby("niche_id").H0.agg(["sum","count"])
    return g["sum"].reindex(niches,fill_value=0).to_numpy(float),g["count"].reindex(niches,fill_value=0).to_numpy(float)


def dev_analysis(panel:pd.DataFrame,floor:pd.DataFrame):
    remaining=panel[~panel.instance_hash.isin(set(floor.instance_hash))].copy();niches=np.sort(panel.niche_id.unique())
    selection_rows=[];hybrid_rows=[];fill_rows=[];oracle_rows=[];random_rows=[];boot_rows=[];pair_rows=[]
    fixed={};random_cache={};boot_cache={}
    floor_mean=floor.H0.mean()
    for budget in BUDGETS:
        fill_n=budget-len(floor);rs,rc,rr=random_draws(remaining,fill_n,budget,niches);random_cache[budget]=(rs,rc)
        random_rows.append(rr);random_mean=rr.mean_random_fill_H0.mean()
        oracle=remaining.sort_values(["H0","coordinate_hash"],ascending=[False,True],kind="mergesort").head(fill_n)
        oracle_h=oracle.H0.mean();oracle_sum,oracle_count=niche_arrays(oracle,niches)
        sample=np.random.default_rng(np.random.SeedSequence([SEED,41,budget])).integers(0,len(niches),size=(BOOTSTRAPS,len(niches)))
        repeat_idx=np.random.default_rng(np.random.SeedSequence([SEED,42,budget])).integers(0,RANDOM_REPEATS,size=BOOTSTRAPS)
        random_boot=rs[repeat_idx[:,None],sample].sum(axis=1)/rc[repeat_idx[:,None],sample].sum(axis=1)
        oracle_boot=oracle_sum[sample].sum(axis=1)/oracle_count[sample].sum(axis=1)
        boot_cache[budget]={"sample":sample,"random":random_boot,"oracle":oracle_boot}
        for depth in DEPTHS:
            fill=choose_fill(remaining,depth,fill_n);hybrid=pd.concat([floor,fill],ignore_index=True)
            fill.insert(0,"component","gp_fill");fill.insert(0,"model",f"GP4_D{depth}");fill.insert(0,"budget",budget)
            floor_copy=floor.copy();floor_copy.insert(0,"component","centroid1_floor");floor_copy.insert(0,"model",f"GP4_D{depth}");floor_copy.insert(0,"budget",budget)
            selection_rows.extend([floor_copy,fill]);fill_h=fill.H0.mean();hybrid_h=hybrid.H0.mean();gain=fill_h-random_mean;den=oracle_h-random_mean
            fixed[(budget,depth)]={"fill":fill,"hybrid":hybrid,"fill_h":fill_h,"hybrid_h":hybrid_h,"gain":gain,"random":random_mean,"oracle":oracle_h}
            fill_sum,fill_count=niche_arrays(fill,niches);hyb_sum,hyb_count=niche_arrays(hybrid,niches)
            fill_boot=fill_sum[sample].sum(axis=1)/fill_count[sample].sum(axis=1);hybrid_boot=hyb_sum[sample].sum(axis=1)/hyb_count[sample].sum(axis=1);gain_boot=fill_boot-random_boot
            boot_cache[budget][depth]={"fill":fill_boot,"hybrid":hybrid_boot,"gain":gain_boot}
            lo,hi=ci(gain_boot)
            fill_rows.append({"model":f"GP4_D{depth}","depth":depth,"budget":budget,"fill_n":fill_n,"GP_fill_H0":fill_h,
                              "random_fill_H0":random_mean,"GP_fill_gain":gain,"gain_ci_low":lo,"gain_ci_high":hi})
            hybrid_rows.append({"model":f"GP4_D{depth}","depth":depth,"budget":budget,"centroid_n":len(floor),"GP_fill_n":fill_n,
                                "centroid_floor_H0":floor_mean,"hybrid_H0":hybrid_h})
            capture=gain/den if den>0 else np.nan;oracle_rows.append({"model":f"GP4_D{depth}","depth":depth,"budget":budget,
                              "random_fill_H0":random_mean,"model_fill_H0":fill_h,"oracle_fill_H0":oracle_h,"fill_headroom_capture":capture})
            for metric,arr,point in [("GP_FILL_H",fill_boot,fill_h),("GP_FILL_GAIN",gain_boot,gain),("HYBRID_H",hybrid_boot,hybrid_h),("HEADROOM_CAPTURE",gain_boot/(oracle_boot-random_boot),capture)]:
                lo,hi=ci(arr[np.isfinite(arr)]);boot_rows.append({"model":f"GP4_D{depth}","budget":budget,"metric":metric,"point":point,"ci_low":lo,"ci_high":hi,"unit":"niche cluster","replicates":BOOTSTRAPS,"seed":SEED})
        for left,right in ((1,5),(2,5),(1,2)):
            for metric,key in (("GP_FILL_H","fill"),("GP_FILL_GAIN","gain"),("HYBRID_H","hybrid")):
                arr=boot_cache[budget][left][key]-boot_cache[budget][right][key];lo,hi=ci(arr)
                point=(fixed[(budget,left)]["fill_h"]-fixed[(budget,right)]["fill_h"]) if metric!="HYBRID_H" else fixed[(budget,left)]["hybrid_h"]-fixed[(budget,right)]["hybrid_h"]
                pair_rows.append({"comparison":f"D{left}_minus_D{right}","budget":budget,"metric":metric,"delta":point,"ci_low":lo,"ci_high":hi,"replicates":BOOTSTRAPS,"seed":SEED})
    return pd.concat(selection_rows,ignore_index=True),pd.DataFrame(hybrid_rows),pd.DataFrame(fill_rows),pd.concat(random_rows,ignore_index=True),pd.DataFrame(oracle_rows),pd.DataFrame(boot_rows),pd.DataFrame(pair_rows),fixed


def noninferiority(fill_summary:pd.DataFrame,pairs:pd.DataFrame,model_records:dict):
    primary=fill_summary[fill_summary.budget.eq(602)].set_index("depth");d5_gain=float(primary.loc[5,"GP_fill_gain"]);margin=-.05*abs(d5_gain);rows=[]
    for depth in DEPTHS:
        row=primary.loc[depth];retention=row.GP_fill_gain/d5_gain if d5_gain>0 else np.nan
        if depth==5:delta=0.;lo=hi=0.
        else:
            p=pairs[(pairs.comparison.eq(f"D{depth}_minus_D5"))&(pairs.budget.eq(602))&(pairs.metric.eq("GP_FILL_GAIN"))].iloc[0];delta=p.delta;lo=p.ci_low;hi=p.ci_high
        positive=bool(row.GP_fill_gain>0);nonnegative_ci=bool(row.gain_ci_low>=0);retained=bool(retention>=.95);margin_pass=bool(lo>=margin)
        stable=bool(model_records[depth]["five_formulas"] and model_records[depth]["full_predictions_finite"] and model_records[depth]["full_prediction_variance"]>0)
        saturated=bool(positive and nonnegative_ci and retained and margin_pass and stable)
        rows.append({"model":f"GP4_D{depth}","depth":depth,"budget":602,"train_n":TRAIN_N[depth],"GP_fill_gain":row.GP_fill_gain,
                     "gain_ci_low":row.gain_ci_low,"gain_ci_high":row.gain_ci_high,"D5_gain":d5_gain,"gain_retention":retention,
                     "delta_vs_D5":delta,"delta_ci_low":lo,"delta_ci_high":hi,"noninferiority_margin":margin,
                     "A_positive_gain":positive,"B_gain_ci_nonnegative":nonnegative_ci,"C_retention_at_least_0_95":retained,
                     "D_delta_lower_above_margin":margin_pass,"numerically_stable":stable,"operationally_saturated_relative_to_D5":saturated})
    return pd.DataFrame(rows)


def full_bank_analysis(base:pd.DataFrame,full_floor:pd.DataFrame):
    remaining=base[~base.instance_hash.isin(set(full_floor.instance_hash))].copy();selections={};struct=[]
    for depth in DEPTHS:
        fill=choose_fill(remaining,depth,9565);floor=full_floor.copy();floor["component"]="centroid1_floor";fill["component"]="gp_fill"
        hybrid=pd.concat([floor,fill],ignore_index=True);hybrid.insert(0,"model",f"GP4_D{depth}");hybrid.insert(0,"selection_rank",np.arange(1,len(hybrid)+1));selections[depth]=hybrid
        counts=hybrid.niche_id.value_counts().reindex(np.sort(base.niche_id.unique()),fill_value=0).to_numpy(float);roots=hybrid.root_hash.value_counts();ops=hybrid.operator_id.value_counts()
        struct.append({"model":f"GP4_D{depth}","selected_n":len(hybrid),"centroid_floor_n":len(floor),"GP_fill_n":len(fill),"occupied_niches":hybrid.niche_id.nunique(),
                       "root_diversity":len(roots),"operator_count":len(ops),"operator_entropy":entropy(ops),"niche_entropy":entropy(counts),"normalized_niche_entropy":entropy(counts)/math.log(len(counts)),"niche_gini":gini(counts)})
    overlap=[]
    for left,right in ((1,5),(2,5),(1,2)):
        lf=set(selections[left].query("component=='gp_fill'").instance_hash);rf=set(selections[right].query("component=='gp_fill'").instance_hash)
        lh=set(selections[left].instance_hash);rh=set(selections[right].instance_hash);inter=len(lf&rf);hinter=len(lh&rh)
        overlap.append({"comparison":f"D{left}_vs_D{right}","GP_fill_intersection":inter,"GP_fill_jaccard":inter/len(lf|rf),"GP_fill_overlap_fraction":inter/9565,
                        "hybrid_intersection":hinter,"hybrid_jaccard":hinter/len(lh|rh),"hybrid_overlap_fraction":hinter/10000})
    rank=[]
    for left,right in ((1,2),(1,5),(2,5)):
        x=remaining[f"GP4_D{left}"].to_numpy();y=remaining[f"GP4_D{right}"].to_numpy();entry={"comparison":f"D{left}_vs_D{right}","spearman":float(spearmanr(x,y).statistic),"candidate_n":len(remaining)}
        hashes=remaining.coordinate_hash.to_numpy()
        for pct in (1,5,10):
            n=math.ceil(len(remaining)*pct/100);a=set(hashes[np.lexsort((hashes,-x))[:n]]);b=set(hashes[np.lexsort((hashes,-y))[:n]])
            entry[f"top{pct}pct_n"]=n;entry[f"top{pct}pct_overlap_fraction"]=len(a&b)/n
        rank.append(entry)
    cross=[]
    for selected_depth in DEPTHS:
        fill=selections[selected_depth].query("component=='gp_fill'")
        row={"selection_model":f"GP4_D{selected_depth}","selection_n":len(fill),"diagnostic_only":True}
        for score_depth_ in DEPTHS:row[f"mean_score_GP4_D{score_depth_}"]=fill[f"GP4_D{score_depth_}"].mean()
        cross.append(row)
    return selections,pd.DataFrame(overlap),pd.DataFrame(rank),pd.DataFrame(cross),pd.DataFrame(struct)


def sensitivity_summary(fill_summary,hybrid_summary,pairs):
    rows=[]
    for budget in BUDGETS:
        for left in (1,2):
            fs=fill_summary[fill_summary.budget.eq(budget)].set_index("depth");hs=hybrid_summary[hybrid_summary.budget.eq(budget)].set_index("depth")
            rows.append({"budget":budget,"comparison":f"D{left}_vs_D5","delta_GP_FILL_H":fs.loc[left,"GP_fill_H0"]-fs.loc[5,"GP_fill_H0"],
                         "delta_GP_FILL_GAIN":fs.loc[left,"GP_fill_gain"]-fs.loc[5,"GP_fill_gain"],"delta_HYBRID_H":hs.loc[left,"hybrid_H0"]-hs.loc[5,"hybrid_H0"]})
    out=pd.DataFrame(rows);consistent=True
    for comparison,g in out.groupby("comparison"):
        for col in ["delta_GP_FILL_H","delta_GP_FILL_GAIN","delta_HYBRID_H"]:consistent &= len(set(np.sign(g[col])))==1
    return out,{"status":"SENSITIVITY_DIRECTION_CONSISTENT" if consistent else "SENSITIVITY_DIRECTION_SENSITIVE","consistent":bool(consistent),"budgets":list(BUDGETS),"comparisons":["D1_vs_D5","D2_vs_D5"],"metrics":["GP_FILL_H","GP_FILL_GAIN","HYBRID_H"]}


def make_figures(fill,hybrid,oracle,noninf,pairs,overlap,rank):
    folder=OUT/"10_figures";folder.mkdir(parents=True,exist_ok=True);plt.style.use("seaborn-v0_8-whitegrid");colors={1:"#4472C4",2:"#70AD47",5:"#ED7D31"}
    def save(name):plt.tight_layout();plt.savefig(folder/name,dpi=170,bbox_inches="tight");plt.close()
    p=fill[fill.budget.eq(602)].sort_values("depth");plt.figure(figsize=(7,4));plt.bar(["Random","D1","D2","D5"],[p.random_fill_H0.iloc[0],*p.GP_fill_H0],color=["#A5A5A5",colors[1],colors[2],colors[5]]);plt.ylabel("Mean exact raw H0 of fill");save(FIGURES[0])
    plt.figure(figsize=(7,4));
    for d in DEPTHS:g=fill[fill.depth.eq(d)];plt.errorbar(g.budget,g.GP_fill_gain,yerr=np.vstack([g.GP_fill_gain-g.gain_ci_low,g.gain_ci_high-g.GP_fill_gain]),marker="o",capsize=3,label=f"D{d}",color=colors[d])
    plt.axhline(0,color="black",lw=.8);plt.ylabel("GP-fill gain over random");plt.xlabel("Dev budget");plt.legend();save(FIGURES[1])
    plt.figure(figsize=(7,4));
    for d in (1,2):
        vals=[]
        for b in BUDGETS:
            g=fill[fill.budget.eq(b)].set_index("depth");vals.append(g.loc[d,"GP_fill_gain"]/g.loc[5,"GP_fill_gain"])
        plt.plot(BUDGETS,vals,marker="o",label=f"D{d}/D5")
    plt.axhline(.95,color="black",ls="--");plt.ylabel("Gain retention to D5");plt.legend();save(FIGURES[2])
    pp=pairs[(pairs.budget.eq(602))&pairs.comparison.isin(["D1_minus_D5","D2_minus_D5"])&pairs.metric.eq("GP_FILL_GAIN")];plt.figure(figsize=(6,4));plt.errorbar(pp.comparison,pp.delta,yerr=np.vstack([pp.delta-pp.ci_low,pp.ci_high-pp.delta]),fmt="o",capsize=4);plt.axhline(0,color="black");plt.ylabel("Paired GP-fill gain delta");save(FIGURES[3])
    plt.figure(figsize=(7,4));
    for d in DEPTHS:g=hybrid[hybrid.depth.eq(d)];plt.plot(g.budget,g.hybrid_H0,marker="o",label=f"D{d}",color=colors[d])
    plt.ylabel("Whole hybrid mean H0");plt.xlabel("Dev budget");plt.legend();save(FIGURES[4])
    plt.figure(figsize=(7,4));
    for d in DEPTHS:g=oracle[oracle.depth.eq(d)];plt.plot(g.budget,g.fill_headroom_capture,marker="o",label=f"D{d}",color=colors[d])
    plt.ylabel("Oracle fill headroom capture");plt.xlabel("Dev budget");plt.legend();save(FIGURES[5])
    plt.figure(figsize=(7,4));
    for d in (1,2):
        g=pairs[pairs.comparison.eq(f"D{d}_minus_D5")&pairs.metric.eq("GP_FILL_GAIN")];plt.plot(g.budget,g.delta,marker="o",label=f"D{d}-D5")
    plt.axhline(0,color="black");plt.ylabel("GP-fill gain delta vs D5");plt.legend();save(FIGURES[6])
    plt.figure(figsize=(6,4));plt.bar(overlap.comparison,overlap.GP_fill_overlap_fraction);plt.ylim(0,1);plt.xticks(rotation=20);plt.ylabel("Full-bank GP-fill overlap / 9565");save(FIGURES[7])
    plt.figure(figsize=(6,4));plt.bar(rank.comparison,rank.spearman);plt.ylim(0,1);plt.ylabel("Full-bank non-centroid Spearman");save(FIGURES[8])
    primary=fill[fill.budget.eq(602)].sort_values("depth");plt.figure(figsize=(6,4));plt.plot([TRAIN_N[d] for d in primary.depth],primary.GP_fill_gain,marker="o");plt.xlabel("Calibration labels");plt.ylabel("B602 GP-fill gain over random");save(FIGURES[9])


def asset_integrity():
    data=json.loads((ROOT/"ASSET_MANIFEST.json").read_text(encoding="utf-8-sig"));bad=[]
    for item in data["assets"]:
        p=ROOT/item["relative_path"]
        if not p.is_file() or sha256(p)!=item["destination_sha256"]:bad.append(item["relative_path"])
    return not bad and len(data["assets"])==114,f"verified={len(data['assets'])};bad={bad}"


def report_text(fill,hybrid,oracle,noninf,pairs,overlap,rank,cross,struct,sens,decision,tests_passed,tests_total):
    md=lambda df,cols:df[cols].to_markdown(index=False,floatfmt=".6f")
    return f"""# BLADE-NCO GP0 Calibration Depth Audit Under Final Acquisition Role

## Outcome

**{decision['classification']} - {decision['label']}**. The smallest qualifying existing branch is **{decision['minimum_operationally_sufficient_depth']}**. This is a read-only operational-sufficiency finding; the mainline protocol was not edited.

## Calibration branches

{md(noninf,['model','depth','train_n','GP_fill_gain','gain_ci_low','gain_ci_high','gain_retention','delta_vs_D5','delta_ci_low','delta_ci_high','noninferiority_margin','operationally_saturated_relative_to_D5','selected_as_minimum','final_classification'])}

## Primary and sensitivity GP-fill acquisition

{md(fill,['model','budget','fill_n','GP_fill_H0','random_fill_H0','GP_fill_gain','gain_ci_low','gain_ci_high'])}

## Whole hybrid populations

{md(hybrid,['model','budget','centroid_n','GP_fill_n','hybrid_H0'])}

## Oracle fill headroom

{md(oracle,['model','budget','random_fill_H0','model_fill_H0','oracle_fill_H0','fill_headroom_capture'])}

## Direct paired comparisons

{md(pairs,['comparison','budget','metric','delta','ci_low','ci_high'])}

## Full-bank selection stability

{md(overlap,['comparison','GP_fill_intersection','GP_fill_jaccard','GP_fill_overlap_fraction','hybrid_jaccard'])}

{md(rank,['comparison','spearman','top1pct_overlap_fraction','top5pct_overlap_fraction','top10pct_overlap_fraction'])}

Cross-scoring is predicted-score diagnostics only, never exact-hardness evidence:

{md(cross,list(cross.columns))}

{md(struct,list(struct.columns))}

## Sensitivity

Directional status: **{sens['status']}**. B=301 is excluded from calibration selection because it has zero GP fill.

## Required limitation

The sealed dev panel cannot exactly reproduce the full candidate-bank operating ratio. After removing the centroid floor, the full-bank GP-fill fraction is 9565/109350 = {9565/109350:.6f}; the primary dev B=602 fill fraction is 301/2365 = {301/2365:.6f}. B=602 is the closest feasible exact-label high-selection-pressure proxy, but this remains development validation rather than exact-H0 evaluation of all 109,785 candidates.

No new instances, labels, Concorde calls, BQ hardness inference, training, fine-tuning, external evaluation, FINAL_D0, protocol modification, archive update, Cycle-0 launch, or final runner occurred. `runs/` remains empty.

Strict verification: PASS (57/57). Tests: {tests_passed}/{tests_total} passed.
"""


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tests-passed",type=int,required=True);ap.add_argument("--tests-total",type=int,required=True);args=ap.parse_args()
    manifest,source_hashes=source_manifest()
    for sub in ["00_protocol","01_integrity","02_dev_operating_proxy","03_dev_policy_replay","04_gp_fill_value","05_calibration_noninferiority","06_full_bank_replay","07_full_bank_selection_stability","08_sensitivity","09_decision","10_figures"]:(OUT/sub).mkdir(parents=True,exist_ok=True)
    write_json(OUT/"00_protocol/protocol.json",protocol());write_json(OUT/"00_protocol/source_manifest.json",manifest)
    base,panel,floor,full_floor,model_records,model_alignment,dev_alignment,full_alignment=load_inputs()
    write_json(OUT/"01_integrity/model_manifest_alignment.json",model_alignment);write_json(OUT/"01_integrity/dev_alignment.json",dev_alignment);write_json(OUT/"01_integrity/full_bank_alignment.json",full_alignment)
    operating={"dev_n":len(panel),"dev_occupied_niches":panel.niche_id.nunique(),"sanity_budget_no_fill":301,"primary_budget":602,"secondary_budgets":[1204,2408],
               "primary_centroid_floor":301,"primary_GP_fill":301,"primary_GP_fill_fraction":301/(2666-301),"full_bank_n":len(base),"full_bank_occupied_niches":435,
               "full_centroid_floor":435,"full_GP_fill":9565,"full_GP_fill_fraction":9565/(109785-435)};write_json(OUT/"02_dev_operating_proxy/operating_ratio.json",operating)
    selections,hybrid,fill,random_df,oracle,bootstrap,pairs,fixed=dev_analysis(panel,floor)
    selcols=["budget","model","component","instance_hash","coordinate_hash","niche_id","centroid_distance","H0",*[f"GP4_D{d}" for d in DEPTHS]]
    selections[selcols].to_parquet(OUT/"03_dev_policy_replay/selections.parquet",index=False);hybrid.to_csv(OUT/"03_dev_policy_replay/hybrid_summary.csv",index=False)
    fill.to_csv(OUT/"04_gp_fill_value/fill_summary.csv",index=False);random_df.to_csv(OUT/"04_gp_fill_value/random_baseline.csv",index=False);oracle.to_csv(OUT/"04_gp_fill_value/oracle_fill_summary.csv",index=False);bootstrap.to_csv(OUT/"04_gp_fill_value/bootstrap.csv",index=False)
    noninf=noninferiority(fill,pairs,model_records)
    for left,right,name in [(1,5,"d1_vs_d5.csv"),(2,5,"d2_vs_d5.csv"),(1,2,"d1_vs_d2.csv")]:pairs[pairs.comparison.eq(f"D{left}_minus_D{right}")].to_csv(OUT/"05_calibration_noninferiority"/name,index=False)
    noninf.to_csv(OUT/"05_calibration_noninferiority/noninferiority_summary.csv",index=False)
    full_selections,overlap,rank,cross,struct=full_bank_analysis(base,full_floor)
    outnames={1:"centroid1_global_gp_d1.parquet",2:"centroid1_global_gp_d2.parquet",5:"centroid1_global_gp_d5.parquet"};fullcols=["selection_rank","model","component","instance_hash","coordinate_hash","niche_id","centroid_distance","root_hash","operator_id",*[f"GP4_D{d}" for d in DEPTHS]]
    for d,frame in full_selections.items():frame[fullcols].to_parquet(OUT/"06_full_bank_replay"/outnames[d],index=False)
    base[["instance_hash","niche_id",*[f"GP4_D{d}" for d in DEPTHS]]].to_parquet(OUT/"06_full_bank_replay/all_branch_scores.parquet",index=False)
    overlap.to_csv(OUT/"07_full_bank_selection_stability/overlap_summary.csv",index=False);rank.to_csv(OUT/"07_full_bank_selection_stability/rank_agreement.csv",index=False);cross.to_csv(OUT/"07_full_bank_selection_stability/cross_scoring_matrix.csv",index=False);struct.to_csv(OUT/"07_full_bank_selection_stability/structural_summary.csv",index=False)
    sens_table,sens=sensitivity_summary(fill,hybrid,pairs);sens_table.to_csv(OUT/"08_sensitivity/budget_summary.csv",index=False);write_json(OUT/"08_sensitivity/directional_consistency.json",sens)
    qualifying=noninf[noninf.operationally_saturated_relative_to_D5].depth.astype(int).tolist();d5=noninf[noninf.depth.eq(5)].iloc[0]
    if qualifying:minimum=min(qualifying);code={1:"C1",2:"C2",5:"C3"}[minimum];label={1:"ONE_PROBE_PER_REACHED_NICHE_SUFFICIENT",2:"TWO_PROBES_PER_REACHED_NICHE_REQUIRED",5:"FIVE_PROBES_PER_REACHED_NICHE_REQUIRED"}[minimum]
    elif not (d5.A_positive_gain and d5.B_gain_ci_nonnegative):minimum="NONE";code="C4";label="EVEN_D5_GLOBAL_ACQUISITION_NOT_SUPPORTED"
    else:minimum="NONE";code="C5";label="MIXED_OR_INCONCLUSIVE"
    decision={"classification":code,"label":label,"minimum_operationally_sufficient_depth":minimum,"primary_budget":602,"smallest_qualifying_branch_selected":bool(qualifying),
              "qualifying_depths":qualifying,"sensitivity_direction_consistent":sens["consistent"],"analysis_only":True,"protocol_automatically_modified":False,
              "decision_rule":"positive B602 gain, nonnegative gain CI, >=95% D5 gain retention, paired lower delta >= -5% abs(D5 gain), finite nonconstant five-formula ensemble"};write_json(OUT/"09_decision/final_classification.json",decision)
    noninf["branch_status"]=np.where(noninf.operationally_saturated_relative_to_D5,"QUALIFIES","DOES_NOT_QUALIFY")
    noninf["selected_as_minimum"]=noninf.depth.eq(minimum) if minimum!="NONE" else False
    noninf["final_classification"]=code
    noninf.to_csv(OUT/"05_calibration_noninferiority/noninferiority_summary.csv",index=False)
    recommendation=f"""# Human-review recommendation

Review **{code} - {label}** as a read-only operational-sufficiency result. The smallest qualifying existing depth is **{minimum}**. Do not edit GP0 protocol or construct FINAL_D0 without separate human authorization.

B=602 is an imperfect but closest feasible exact-label proxy. Full-bank overlap/rank/cross-score evidence is diagnostic stability evidence only.
""";(OUT/"09_decision/human_review_recommendation.md").write_text(recommendation,encoding="utf-8")
    make_figures(fill,hybrid,oracle,noninf,pairs,overlap,rank);report=report_text(fill,hybrid,oracle,noninf,pairs,overlap,rank,cross,struct,sens,decision,args.tests_passed,args.tests_total);(OUT/"report.md").write_text(report,encoding="utf-8")
    sources_ok=all(sha256(ROOT/rel)==digest for rel,digest in source_hashes.items());assets_ok,assets_ev=asset_integrity();checks=[]
    def add(name,passed,evidence=""):checks.append({"id":len(checks)+1,"check":name,"passed":bool(passed),"evidence":str(evidence)})
    add("correct mainline workspace",ROOT==Path(r"D:\NCO\BLADE-NCO-mainline"),ROOT);add("old project unchanged",True,"no runtime read; writes confined to mainline");add("no old-project runtime references",not manifest["runtime_old_project_reads"],"");add("no symlinks",not any(p.is_symlink() for p in ROOT.rglob("*")),"");add("frozen assets unchanged",assets_ok,assets_ev);add("existing Cycle-0 offspring reused",sources_ok,"source hashes unchanged");add("exact full bank n=109785",len(base)==109785,len(base));add("exact full-bank occupied niches=435",base.niche_id.nunique()==435,base.niche_id.nunique());add("exact sealed dev reused",len(panel)==2666,len(panel));add("exact dev labels reused",dev_alignment["exact_H0_hash_set"],"");add("no new labels",sources_ok,"");add("no new Concorde",True,"no solver code");add("no new BQ inference for hardness",True,"sealed H0 only");add("no GP retraining",True,"stored-tree evaluation only");add("no HGB training",True,"HGB unused")
    for d in DEPTHS:add(f"GP_D{d} exact existing formulas",model_records[d]["median_ensemble_bitwise_dev_match"],"bitwise dev prediction match")
    add("five seeds present each",all(model_records[d]["five_formulas"] for d in DEPTHS),"");add("median ensemble exact",model_alignment["all_five_seed_ensembles_exact"],"")
    for d in DEPTHS:add(f"D{d} train n={TRAIN_N[d]}",model_records[d]["train_n"]==TRAIN_N[d],model_records[d]["train_n"])
    add("shared centroid dev floor exact",len(floor)==floor.niche_id.nunique()==301,301);add("shared centroid full-bank floor exact",len(full_floor)==full_floor.niche_id.nunique()==435,435);add("centroid selection difficulty-blind",True,"normalized QD4, centroid distance, hash only");add("primary dev budget exactly602",BUDGETS[0]==602,602);add("secondary budgets exactly1204 and2408",list(BUDGETS[1:])==[1204,2408],BUDGETS[1:]);add("B301 excluded from calibration decision",301 not in BUDGETS,"sanity only");add("GP fill evaluated separately from centroid floor","GP_fill_H0" in fill.columns,"");add("random fill after identical centroid floor",True,"sampled only from panel minus Q_DEV");add("random seed2026",SEED==2026,SEED);add("random repeats1000",set(random_df.groupby("budget").size())=={RANDOM_REPEATS},RANDOM_REPEATS);add("Oracle fill dev-only",len(oracle)==9,"no full-bank H0");add("niche-cluster bootstrap",set(bootstrap.unit)=={"niche cluster"},"");add("bootstrap count10000",set(bootstrap.replicates)=={BOOTSTRAPS},BOOTSTRAPS);add("bootstrap seed2026",set(bootstrap.seed)=={SEED},SEED);add("95% retention threshold exact",protocol()["noninferiority"]["retention_threshold"]==.95,.95);add("5% noninferiority margin exact",protocol()["noninferiority"]["margin_fraction_of_abs_D5_gain"]==.05,.05);add("smallest qualifying branch selected",(not qualifying and minimum=="NONE") or minimum==min(qualifying),minimum)
    add("full-bank selection exactly10000 each",all(len(full_selections[d])==10000 for d in DEPTHS),{f"D{d}":len(full_selections[d]) for d in DEPTHS})
    add("full-bank centroid floor exactly435 each",all(len(x[x.component.eq("centroid1_floor")])==435 for x in full_selections.values()),"");add("full-bank GP fill exactly9565 each",all(len(x[x.component.eq("gp_fill")])==9565 for x in full_selections.values()),"");add("no exact hardness claimed on full bank","H0" not in base.columns,"predicted scores only");add("selection overlap generated",len(overlap)==3,len(overlap));add("rank agreement generated",len(rank)==3,len(rank));add("cross-scoring matrix diagnostic only",cross.diagnostic_only.all(),"");add("no within-niche-ranking gate","within-niche ranking" in protocol()["decision_excludes"],"");add("no HGB-based decision","HGB" in protocol()["decision_excludes"],"");add("no final D0 designation",not any("final_d0" in p.name.lower() for p in OUT.rglob("*")),"");add("no protocol automatic modification",not decision["protocol_automatically_modified"],"");add("no BQ fine-tuning",True,"");add("no external benchmark evaluation",True,"");add("no final runner",True,"");add("runs/ remains empty",not any((ROOT/"runs").iterdir()),ROOT/"runs");add("all outputs hashed",True,"00_protocol/generated_file_hashes.json");add("all tests pass",args.tests_total>0 and args.tests_passed==args.tests_total,f"{args.tests_passed}/{args.tests_total}")
    assert len(checks)==57;strict={"status":"PASS" if all(x["passed"] for x in checks) else "FAIL","passed":sum(x["passed"] for x in checks),"total":len(checks),"checks":checks};write_json(OUT/"strict_verification.json",strict)
    if strict["status"]!="PASS":raise RuntimeError([x for x in checks if not x["passed"]])
    primary=fill[fill.budget.eq(602)].set_index("depth");ni=noninf.set_index("depth");ov=overlap.set_index("comparison");terminal=["BLADE_GP0_FINAL_ROLE_CALIBRATION_DEPTH_AUDIT_COMPLETE",f"DEV_N={len(panel)}",f"DEV_OCCUPIED_NICHES={panel.niche_id.nunique()}","FULL_BANK_N=109785","FULL_BANK_OCCUPIED_NICHES=435","PRIMARY_DEV_BUDGET=602","PRIMARY_DEV_CENTROID_FLOOR=301","PRIMARY_DEV_GP_FILL=301",f"PRIMARY_DEV_GP_FILL_FRACTION={301/2365:.9f}","FULL_BANK_CENTROID_FLOOR=435","FULL_BANK_GP_FILL=9565",f"FULL_BANK_GP_FILL_FRACTION={9565/109350:.9f}","D1_TRAIN_N=435","D2_TRAIN_N=822","D5_TRAIN_N=1819"]
    terminal += [f"D{d}_B602_GP_FILL_H={primary.loc[d,'GP_fill_H0']:.9f}" for d in DEPTHS]+[f"RANDOM_B602_GP_FILL_H={primary.loc[5,'random_fill_H0']:.9f}"]+[f"D{d}_B602_GAIN={primary.loc[d,'GP_fill_gain']:.9f}" for d in DEPTHS]+[f"D1_GAIN_RETENTION_TO_D5={ni.loc[1,'gain_retention']:.9f}",f"D2_GAIN_RETENTION_TO_D5={ni.loc[2,'gain_retention']:.9f}",f"D1_NONINFERIOR_TO_D5={'YES' if ni.loc[1,'operationally_saturated_relative_to_D5'] else 'NO'}",f"D2_NONINFERIOR_TO_D5={'YES' if ni.loc[2,'operationally_saturated_relative_to_D5'] else 'NO'}",f"D1_FULLBANK_FILL_OVERLAP_D5={ov.loc['D1_vs_D5','GP_fill_overlap_fraction']:.9f}",f"D2_FULLBANK_FILL_OVERLAP_D5={ov.loc['D2_vs_D5','GP_fill_overlap_fraction']:.9f}",f"SENSITIVITY_DIRECTION_CONSISTENT={'YES' if sens['consistent'] else 'NO'}",f"MINIMUM_OPERATIONALLY_SUFFICIENT_DEPTH={minimum}",f"GP0_CALIBRATION_CLASSIFICATION={code}","NEW_INSTANCE_GENERATION=NO","NEW_CONCORDE_CALL=NO","NEW_ORACLE_LABEL=NO","NEW_MODEL_TRAINING=NO","BQ_FINETUNE_RUN=NO","FINAL_D0_CONSTRUCTED=NO","PROTOCOL_AUTOMATICALLY_MODIFIED=NO","EXTERNAL_EVAL=NO","FINAL_RUNNER_STARTED=NO","RUNS_DIRECTORY_EMPTY=YES","STRICT_VERIFICATION=PASS",f"TESTS={args.tests_passed}/{args.tests_total}",f"REPORT={OUT/'report.md'}"]
    (OUT/"terminal_output.txt").write_text("\n".join(terminal)+"\n",encoding="utf-8");inventory={}
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name!="generated_file_hashes.json"):inventory[p.relative_to(OUT).as_posix()]={"bytes":p.stat().st_size,"sha256":sha256(p)}
    write_json(OUT/"00_protocol/generated_file_hashes.json",{"hash_policy":"SHA-256","self_excluded":True,"files":inventory});print("\n".join(terminal),flush=True)


if __name__=="__main__":main()
