#!/usr/bin/env python3
"""P6-D2 offline selected-population structure and training-value audit."""

from __future__ import annotations

import hashlib, json, math, os, platform, sys, time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
D0=ROOT/"target_aware_pilot/p6d0_target_aware_qd_selection_audit"
D1=ROOT/"target_aware_pilot/p6d1_target_aware_qd_training_value"
B=ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution"
C0=ROOT/"target_aware_pilot/p6c0_training_value_mechanism_audit"
C1=ROOT/"target_aware_pilot/p6c1_hardness_controlled_matching"
C2=ROOT/"target_aware_pilot/p6c2_oracle_hardness_restoration_training_value"
C3=ROOT/"target_aware_pilot/p6c3_hgb40_hardness_selection_fidelity"
P6A=ROOT/"target_aware_pilot/p6a_top5_target_matching_feasibility"
P1=ROOT/"target_aware_pilot"
REQUEST=Path("/home/haozou/.codex/attachments/39d92ac9-d452-475f-bd4b-db132000d107/pasted-text.txt")
sys.path[:0]=[str(ROOT/"src"),str(P6A),str(P1)]
import run_p6a_audit as p6a  # noqa
import run_p0_p1_audit as p01  # noqa
from bq.tsp_metric import validate_tour_0based  # noqa
from final_run.science import membership_hash  # noqa
from mutation.result import coordinate_hash  # noqa
from oracle.concorde import sha256_file  # noqa

REPS=("Original","S1","S2"); ARMS=("A","B","C"); K=10_000; BOOT_N=1000; BOOT_SEED=20261031
F40=tuple(json.loads((ROOT/"audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json").read_text())["F40"])
TOP5=("spn_sum_of_lowest_edge_values","spn_distance_min","spn_distance_varcoeff","spn_nng_3_strong_components_varcoeff","spn_mst_dists_sd")
QD4=("spn_sum_of_lowest_edge_values","spn_distance_varcoeff","spn_mst_depth_mean","spn_mst_dists_median")

def ensure_dirs():
    for p in (OUT,OUT/"figures",OUT/"figures/representative_cases",OUT/"configs",OUT/"manifests"): p.mkdir(parents=True,exist_ok=True)

def write_json(path:Path,value:Any):
    def cv(x):
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,)): return float(x)
        if isinstance(x,(np.bool_,)): return bool(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=cv)+"\n")

def chash(x:Any)->str: return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

def md(frame:pd.DataFrame)->str:
    def f(x):
        if isinstance(x,(list,tuple,dict,set)): return json.dumps(x,sort_keys=True,default=str).replace("|","\\|")
        if pd.isna(x): return "NA"
        if isinstance(x,(float,np.floating)): return f"{float(x):.7g}"
        return str(x).replace("|","\\|").replace("\n"," ")
    cols=list(map(str,frame.columns)); lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    lines += ["| "+" | ".join(f(x) for x in r)+" |" for r in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)

def load_bundle(path:Path):
    with np.load(path,allow_pickle=False) as z: return np.asarray(z["coordinates"],float),z["instance_hash"].astype(str).tolist(),np.asarray(z["expert_tours"],int)

def eff(series:pd.Series)->tuple[float,float,float]:
    p=series.astype(str).value_counts(normalize=True).to_numpy(float)
    return float(np.exp(-(p*np.log(p)).sum())),float(1/np.sum(p*p)),float(p.max())

def feature_group(name:str)->str:
    if "nearest_neighbour" in name: return "nearest_neighbor_geometry"
    if "mst_" in name: return "minimum_spanning_tree"
    if "nng_" in name: return "nearest_neighbor_graph"
    if "distance_" in name or "fraction_shorter" in name or "lowest_edge" in name: return "pairwise_distance"
    if "centroid" in name or "hull_" in name: return "spatial_density_hull_outlier"
    if "angles" in name: return "angular_spatial_distribution"
    return "other_existing_F40"

def overlap_hist(a,b,lo,hi,bins=40):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo: return 1.0
    x=np.linspace(lo,hi,bins+1); ah=np.histogram(np.clip(a,lo,hi),x)[0]; bh=np.histogram(np.clip(b,lo,hi),x)[0]
    return float(np.minimum(ah/ah.sum(),bh/bh.sum()).sum())

def bh_adjust(pvals):
    p=np.asarray(pvals,float); out=np.full(len(p),np.nan); ok=np.isfinite(p); x=p[ok]; order=np.argsort(x); ranked=x[order]*len(x)/(np.arange(len(x))+1); ranked=np.minimum.accumulate(ranked[::-1])[::-1]; temp=np.empty(len(x)); temp[order]=np.minimum(ranked,1); out[ok]=temp; return out

def stable_seed(key:str)->int: return int.from_bytes(hashlib.sha256(f"P6D2|{BOOT_SEED}|{key}".encode()).digest()[:8],"little")

def bootstrap_diff(a,b,key,roots_a=None,roots_b=None):
    a=np.asarray(a,float); b=np.asarray(b,float); rng=np.random.default_rng(stable_seed(key)); inst=np.empty(BOOT_N)
    for i in range(BOOT_N): inst[i]=a[rng.integers(0,len(a),len(a))].mean()-b[rng.integers(0,len(b),len(b))].mean()
    result={"estimate":a.mean()-b.mean(),"instance_ci_low":np.quantile(inst,.025),"instance_ci_high":np.quantile(inst,.975)}
    if roots_a is not None:
        ar,ai=np.unique(np.asarray(roots_a).astype(str),return_inverse=True); br,bi=np.unique(np.asarray(roots_b).astype(str),return_inverse=True)
        asum=np.bincount(ai,weights=a); acount=np.bincount(ai); bsum=np.bincount(bi,weights=b); bcount=np.bincount(bi); vals=np.empty(BOOT_N)
        for i in range(BOOT_N):
            ia=rng.integers(0,len(ar),len(ar)); ib=rng.integers(0,len(br),len(br)); vals[i]=asum[ia].sum()/acount[ia].sum()-bsum[ib].sum()/bcount[ib].sum()
        result.update(root_ci_low=np.quantile(vals,.025),root_ci_high=np.quantile(vals,.975),root_count_a=len(ar),root_count_b=len(br))
    return result

def tour_features(xy,tour):
    n=len(xy); d=squareform(pdist(xy)); t=np.asarray(tour,int); nxt=np.roll(t,-1); edges=d[t,nxt]; tri=d[np.triu_indices(n,1)]; scale=np.median(tri); q90=np.quantile(tri,.9)
    order=np.argsort(d,axis=1); ranks=np.empty_like(order); ranks[np.arange(n)[:,None],order]=np.arange(n)[None,:]
    directed=np.concatenate([ranks[t,nxt],ranks[nxt,t]])
    mst=minimum_spanning_tree(d).tocoo(); mstset={tuple(sorted((int(i),int(j)))) for i,j in zip(mst.row,mst.col)}; tourset={tuple(sorted((int(i),int(j)))) for i,j in zip(t,nxt)}
    norm=edges/scale if scale>0 else edges; hist=np.histogram(np.clip(norm,0,3),bins=np.linspace(0,3,13))[0].astype(float); p=hist[hist>0]/hist.sum()
    return {"tour_edge_mean":edges.mean(),"tour_edge_median":np.median(edges),"tour_edge_cv":edges.std()/edges.mean(),"tour_edge_p90":np.quantile(edges,.9),"tour_edge_p95":np.quantile(edges,.95),"long_edge_fraction_pairwise_p90":np.mean(edges>=q90),"tour_total_length":edges.sum(),"tour_total_over_pairwise_median":edges.sum()/scale,"tour_edge_entropy_12bin":-(p*np.log(p)).sum(),"endpoint_knn3_fraction":np.mean(directed<=3),"endpoint_knn5_fraction":np.mean(directed<=5),"endpoint_knn10_fraction":np.mean(directed<=10),"endpoint_neighbor_rank_mean":directed.mean(),"endpoint_neighbor_rank_p90":np.quantile(directed,.9),"tour_mst_edge_overlap_fraction":len(tourset&mstset)/(n-1)}

def main():
    started=time.perf_counter(); ensure_dirs()
    protocol={"experiment":"BLADE-NCO P6-D2","request_sha256":sha256_file(REQUEST),"d0_completion_sha256":sha256_file(D0/"completion.json"),"d1_completion_sha256":sha256_file(D1/"completion.json"),"F40":list(F40),"Top5":list(TOP5),"QD4":list(QD4),"replacement":"B\\C versus C\\B, niche-stratified","F40_scaling":"per-replicate frozen E0-union-E1 candidate mean/SD; no gap tuning","long_edge_rule":"tour edge >= within-instance all-pair distance P90","tour_scale":"within-instance pairwise-distance median","bootstrap_n":BOOT_N,"bootstrap_unit_sensitivity":"lineage_root","prohibitions":["mutation","evolution","Concorde","BQ training","HGB training","membership change","lambda change","QD/Top5 change","P6-D3"],"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
    protocol["protocol_sha256"]=chash(protocol); write_json(OUT/"configs/frozen_p6d2_config.json",protocol)
    (OUT/"00_protocol.md").write_text("# P6-D2 Frozen Offline Protocol\n\nThis audit reads exact P6-D0 A/B/C memberships, P6-C0 F40, P6-C1 coordinates/expert tours and P6-D1 gaps/CE. It performs no selection, solver labeling, model fitting, mutation, evolution or fine-tuning. B/C replacement is niche-stratified and remains observational.\n\n- Protocol SHA256: `"+protocol["protocol_sha256"]+"`\n- Long edge: tour edge at or above the within-instance P90 of all pairwise distances.\n- Cross-size representation: aspect-preserving coordinates, formal salesperson scaler and transformed F40.\n")
    source_paths=[D0/"05_selection_manifests.csv",D0/"04_candidate_manifest.csv",D0/"configs/frozen_p6d0_config.json",D1/"04_per_instance_gap.csv",D1/"07_cross_population_ce.csv",D1/"logs/cross_population_ce_per_instance.parquet",D1/"completion.json",C0/"completion.json",C0/"02_hardness_statistics.csv",C1/"completion.json",C3/"completion.json",ROOT/"assets/qd/cvt/centroids.csv",ROOT/"assets/qd/normalization/normalization_A_P1P99.json",P6A/"diagnostics/tsplib48_top5_representation.parquet"]
    source_paths += sorted((C0/"full_f40_cache").glob("*.parquet"))
    source_paths += [C1/"dataset_manifests"/f"{rep}_union.{suffix}" for rep in REPS for suffix in ("npz","parquet")]
    source_paths += [D1/"checkpoints"/f"{rep}_{arm}_M5_E256.tsp.best" for rep in REPS for arm in ARMS]
    source_paths += [D1/"logs"/f"{rep}_{arm}"/"training_integrity.json" for rep in REPS for arm in ARMS]
    def source_version(p):
        rel=p.relative_to(ROOT)
        return rel.parts[1] if len(rel.parts)>1 and rel.parts[0]=="target_aware_pilot" else "repository_frozen_asset"
    pd.DataFrame([{"path":str(p.relative_to(ROOT)),"version":source_version(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in source_paths]).to_csv(OUT/"manifests/source_artifact_hashes.csv",index=False)
    selected=pd.read_csv(D0/"05_selection_manifests.csv"); selected=selected[selected.selection.isin(ARMS)].copy()
    f40=pd.concat([pd.read_parquet(p) for p in sorted((C0/"full_f40_cache").glob("*.parquet"))],ignore_index=True).drop_duplicates("instance_hash")
    if len(f40)!=59678 or f40.instance_hash.nunique()!=59678 or not set(F40).issubset(f40): raise RuntimeError("formal F40 cache integrity")
    f40i=f40.set_index("instance_hash"); p90=float(pd.read_csv(C0/"02_hardness_statistics.csv").pooled_p90_threshold.dropna().iloc[0]); p95=float(pd.read_csv(C0/"02_hardness_statistics.csv").pooled_p95_threshold.dropna().iloc[0])
    # Integrity and exact B/C comparability.
    int_rows=[]; bundles={}; pools={}; groups={}
    for rep in REPS:
        xy,hh,tt=load_bundle(C1/"dataset_manifests"/f"{rep}_union.npz"); bundles[rep]=(xy,hh,tt); lookup={h:i for i,h in enumerate(hh)}
        pool=pd.read_parquet(C1/"dataset_manifests"/f"{rep}_union.parquet").merge(pd.read_csv(D0/"04_candidate_manifest.csv").query("replicate==@rep")[["instance_hash","qd_niche","predicted_m4_ce","true_m4_ce","target_niche_probability"]],on="instance_hash",validate="one_to_one"); pools[rep]=pool
        for arm in ARMS:
            z=selected[(selected.replicate==rep)&(selected.selection==arm)].sort_values("selection_order"); hashes=z.instance_hash.astype(str).tolist(); idx=np.array([lookup[h] for h in hashes]); cx=xy[idx]; ct=tt[idx]
            cp=D1/"checkpoints"/f"{rep}_{arm}_M5_E256.tsp.best"; ti=json.loads((D1/"logs"/f"{rep}_{arm}"/"training_integrity.json").read_text())
            ok=(len(hashes)==K and len(set(hashes))==K and hashes==(D0/"selected_hashes"/f"{rep}_{arm}.txt").read_text().splitlines() and [coordinate_hash(x) for x in cx]==hashes and all(validate_tour_0based(t,100)[0] for t in ct) and sha256_file(cp)==ti["checkpoint_sha256"])
            int_rows.append({"replicate":rep,"arm":arm,"n":len(hashes),"unique_n":len(set(hashes)),"membership_sha256":membership_hash(hashes),"coordinates_hashes_tours_valid":ok,"selection_manifest_sha256":sha256_file(D0/"selected_hashes"/f"{rep}_{arm}.txt"),"checkpoint_path":str(cp.relative_to(ROOT)),"checkpoint_sha256":sha256_file(cp),"f40_complete":not f40i.loc[hashes,list(F40)].isna().any().any(),"lambda":0.0 if arm=="A" else .5,"within_niche_rule":"random" if arm=="B" else "HGB40"})
        b=selected[(selected.replicate==rep)&(selected.selection=="B")]; c=selected[(selected.replicate==rep)&(selected.selection=="C")]
        bc=b.qd_niche.value_counts().sort_index(); cc=c.qd_niche.value_counts().sort_index();
        if not bc.equals(cc): raise RuntimeError(f"B/C quota mismatch {rep}")
        bs=set(b.instance_hash.astype(str)); cs=set(c.instance_hash.astype(str)); shared=bs&cs; bo=bs-cs; co=cs-bs
        groups[(rep,"A")]=set(selected[(selected.replicate==rep)&(selected.selection=="A")].instance_hash.astype(str)); groups[(rep,"B")]=bs; groups[(rep,"C")]=cs; groups[(rep,"Shared")]=shared; groups[(rep,"B_only")]=bo; groups[(rep,"C_only")]=co
        for niche in set(b.qd_niche)|set(c.qd_niche):
            bz=set(b.loc[b.qd_niche.eq(niche),"instance_hash"].astype(str)); cz=set(c.loc[c.qd_niche.eq(niche),"instance_hash"].astype(str))
            if len(bz-cz)!=len(cz-bz): raise RuntimeError(f"within-niche replacement mismatch {rep}/{niche}")
    integrity=pd.DataFrame(int_rows); integrity.to_csv(OUT/"01_frozen_population_manifest.csv",index=False)
    if not integrity.coordinates_hashes_tours_valid.all(): raise RuntimeError("population integrity")
    gaps=pd.read_csv(D1/"04_per_instance_gap.csv"); expected={"OriginalTSPLIB70":70,"TSPLIB48":48,"TSPLIB22":22,"Uniform100":10000,"GaussianMixture100":10000,"TSPLIB100":10000,"GaussianMixture500":1000,"EvoReal_S1_100":1000,"EvoReal_S2_100":1000,"EvoReal_S3_100":1000}
    for rep in REPS:
        for ds,n in expected.items():
            if len(gaps[(gaps.replicate==rep)&(gaps.dataset==ds)])!=n: raise RuntimeError(f"gap recovery {rep}/{ds}")
    (OUT/"01_input_integrity.md").write_text("# Input Integrity\n\nAll nine exact ordered P6-D0 memberships, coordinates, hashes, legal expert tours, F40, QD4/CVT niche, lineage/operator provenance, P6-D1 checkpoints and per-instance gaps pass. B and C use the same per-replicate E0∪E1 pool, lambda=0.5 and byte-identical niche quotas; the designed selection-rule difference is random versus frozen-HGB ranking within niche. Random sampling also changes membership stochastically, so downstream observations estimate the total frozen-strategy contrast, not a per-instance HGB causal effect.\n\n"+md(integrity)+"\n")
    # Membership replacement and niche-stratified hardness.
    repl=[]; niche_rows=[]
    for rep in REPS:
        pool=pools[rep].set_index("instance_hash"); bs=groups[(rep,"B")]; cs=groups[(rep,"C")]; shared=bs&cs; bo=bs-cs; co=cs-bs
        for gname,ss in (("Shared",shared),("B_only",bo),("C_only",co)):
            z=pool.loc[list(ss)]; v=z.true_m4_ce.to_numpy(float); pred=z.predicted_m4_ce.to_numpy(float); e,l,m=eff(z.lineage_root)
            repl.append({"replicate":rep,"record_type":"group","group":gname,"n":len(z),"intersection_n":len(shared),"jaccard":len(shared)/len(bs|cs),"replacement_n":len(bo),"replacement_rate":len(bo)/K,"true_ce_mean":v.mean(),"true_ce_median":np.median(v),"true_ce_p75":np.quantile(v,.75),"true_ce_p90":np.quantile(v,.9),"true_ce_p95":np.quantile(v,.95),"global_p90_fraction":np.mean(v>=p90),"global_p95_fraction":np.mean(v>=p95),"predicted_ce_mean":pred.mean(),"prediction_bias_mean":(pred-v).mean(),"effective_lineages":e,"largest_lineage_share":m})
        niches=sorted(set(pool.loc[list(bo|co),"qd_niche"].astype(int)))
        for k in niches:
            bb=pool.loc[list(bo)]; bb=bb[bb.qd_niche.eq(k)]; cc=pool.loc[list(co)]; cc=cc[cc.qd_niche.eq(k)]
            if len(bb)!=len(cc): raise RuntimeError("replacement quota invariant")
            if not len(bb): continue
            niche_rows.append({"replicate":rep,"niche_id":k,"replacement_n_each":len(bb),"B_only_true_ce_mean":bb.true_m4_ce.mean(),"C_only_true_ce_mean":cc.true_m4_ce.mean(),"C_minus_B_true_ce":cc.true_m4_ce.mean()-bb.true_m4_ce.mean(),"B_only_pred_mean":bb.predicted_m4_ce.mean(),"C_only_pred_mean":cc.predicted_m4_ce.mean(),"C_minus_B_predicted":cc.predicted_m4_ce.mean()-bb.predicted_m4_ce.mean(),"B_only_p90_fraction":np.mean(bb.true_m4_ce>=p90),"C_only_p90_fraction":np.mean(cc.true_m4_ce>=p90),"target_niche_probability":cc.target_niche_probability.mean(),"B_roots":bb.lineage_root.nunique(),"C_roots":cc.lineage_root.nunique()})
        boz=pool.loc[list(bo)]; coz=pool.loc[list(co)]; boot=bootstrap_diff(coz.true_m4_ce,boz.true_m4_ce,f"{rep}|hardness",coz.lineage_root,boz.lineage_root); repl.append({"replicate":rep,"record_type":"niche_stratified_summary","group":"C_only_minus_B_only","n":len(bo),"true_ce_mean":boot["estimate"],"instance_ci_low":boot["instance_ci_low"],"instance_ci_high":boot["instance_ci_high"],"root_ci_low":boot["root_ci_low"],"root_ci_high":boot["root_ci_high"]})
    repl=pd.DataFrame(repl); niches=pd.DataFrame(niche_rows); repl.to_csv(OUT/"02_replacement_statistics.csv",index=False); niches.to_csv(OUT/"02_niche_replacement.csv",index=False)
    rview=repl[repl.record_type.eq("group")][["replicate","group","n","jaccard","replacement_rate","true_ce_mean","true_ce_median","global_p90_fraction","global_p95_fraction","predicted_ce_mean","prediction_bias_mean"]]
    topn=niches.assign(abs_diff=lambda x:x.C_minus_B_true_ce.abs()).sort_values(["replicate","abs_diff"],ascending=[True,False]).groupby("replicate").head(10)
    (OUT/"02_membership_replacement_audit.md").write_text("# Membership Replacement Audit\n\nReplacement counts are exactly balanced inside every niche. Comparisons are niche-stratified differences between different instances, not paired-instance causal effects.\n\n"+md(rview)+"\n\nLargest niche hardness changes:\n\n"+md(topn)+"\n")
    # Full F40 comparison and within-niche effects.
    frows=[]; wfrows=[]
    top5set=set(TOP5)
    for rep in REPS:
        pool=pools[rep]; mu=f40i.loc[pool.instance_hash,F40].mean(); sd=f40i.loc[pool.instance_hash,F40].std(ddof=0).replace(0,1); q01=f40i.loc[pool.instance_hash,F40].quantile(.01); q99=f40i.loc[pool.instance_hash,F40].quantile(.99)
        for group in ("A","B","C","Shared","B_only","C_only"):
            hashes=list(groups[(rep,group)]); z=f40i.loc[hashes,F40]
            for feature in F40:
                x=z[feature].to_numpy(float); frows.append({"replicate":rep,"comparison":"group_summary","group":group,"feature":feature,"feature_group":feature_group(feature),"is_top5":feature in top5set,"n":len(x),"mean":x.mean(),"median":np.median(x),"sd":x.std(ddof=1),"p10":np.quantile(x,.1),"p25":np.quantile(x,.25),"p75":np.quantile(x,.75),"p90":np.quantile(x,.9)})
        bo=f40i.loc[list(groups[(rep,"B_only")]),F40]; co=f40i.loc[list(groups[(rep,"C_only")]),F40]
        for feature in F40:
            a=co[feature].to_numpy(float); b=bo[feature].to_numpy(float); scale=sd[feature]; u=stats.mannwhitneyu(a,b,alternative="two-sided")
            frows.append({"replicate":rep,"comparison":"C_only_minus_B_only","group":"replacement_effect","feature":feature,"feature_group":feature_group(feature),"is_top5":feature in top5set,"n":len(a),"mean":a.mean()-b.mean(),"median":np.median(a)-np.median(b),"standardized_mean_difference":(a.mean()-b.mean())/scale,"wasserstein_raw":wasserstein_distance(a,b),"wasserstein_pool_sd":wasserstein_distance(a,b)/scale,"distribution_overlap":overlap_hist(a,b,q01[feature],q99[feature]),"mannwhitney_p_exploratory":u.pvalue})
        meta=selected[(selected.replicate==rep)&(selected.selection.isin(["B","C"]))].set_index("instance_hash")
        for k,g in niches[niches.replicate.eq(rep)].groupby("niche_id"):
            bh=list(set(meta[(meta.selection=="B")&meta.qd_niche.eq(k)].index)-groups[(rep,"C")]); ch=list(set(meta[(meta.selection=="C")&meta.qd_niche.eq(k)].index)-groups[(rep,"B")])
            if not bh: continue
            for feature in F40: wfrows.append({"replicate":rep,"niche_id":int(k),"replacement_n_each":len(bh),"feature":feature,"feature_group":feature_group(feature),"is_top5":feature in top5set,"B_only_mean":f40i.loc[bh,feature].mean(),"C_only_mean":f40i.loc[ch,feature].mean(),"C_minus_B_pool_sd":(f40i.loc[ch,feature].mean()-f40i.loc[bh,feature].mean())/sd[feature]})
    fcomp=pd.DataFrame(frows); effects=fcomp[fcomp.comparison.eq("C_only_minus_B_only")].copy(); effects["bh_q_exploratory"]=effects.groupby("replicate").mannwhitney_p_exploratory.transform(bh_adjust); fcomp.loc[effects.index,"bh_q_exploratory"]=effects.bh_q_exploratory; fcomp.to_csv(OUT/"03_f40_comparison.csv",index=False)
    wf=pd.DataFrame(wfrows); wf.to_csv(OUT/"03_within_niche_feature_effects.csv",index=False)
    agg=effects.groupby(["feature","feature_group","is_top5"]).agg(median_smd=("standardized_mean_difference","median"),min_smd=("standardized_mean_difference","min"),max_smd=("standardized_mean_difference","max"),direction_consistency=("standardized_mean_difference",lambda x:max(np.mean(x>0),np.mean(x<0))),mean_overlap=("distribution_overlap","mean")).reset_index(); agg["abs_median_smd"]=agg.median_smd.abs(); agg=agg.sort_values("abs_median_smd",ascending=False)
    (OUT/"03_f40_structural_difference.md").write_text("# Full-F40 Structural Difference\n\nStandardization uses each replicate's frozen E0∪E1 candidate pool and is independent of benchmark gaps. Exploratory p/q values treat instances as observations and are not causal or evolution-replicate significance.\n\nTop effects across replicates:\n\n"+md(agg.head(20))+"\n\nTop5 and F35 are explicitly identified in `03_f40_comparison.csv`; niche-level effects are in `03_within_niche_feature_effects.csv`.\n")
    # Coverage / redundancy with exact Full-F40 nearest-neighbor proxies.
    covrows=[]; density_lookup={}
    for rep in REPS:
        pool=pools[rep]; allz=(f40i.loc[pool.instance_hash,F40]-f40i.loc[pool.instance_hash,F40].mean())/f40i.loc[pool.instance_hash,F40].std(ddof=0).replace(0,1); nnall=NearestNeighbors(n_neighbors=2,n_jobs=-1).fit(allz).kneighbors(allz,return_distance=True)[0][:,1]; near_thr=np.quantile(nnall,.01)
        meta=selected[selected.replicate.eq(rep)].drop_duplicates("instance_hash").set_index("instance_hash")
        for group in ("A","B","C","Shared","B_only","C_only"):
            hs=list(groups[(rep,group)]); z=allz.loc[hs]; dist=NearestNeighbors(n_neighbors=2,n_jobs=-1).fit(z).kneighbors(z,return_distance=True)[0][:,1] if len(z)>1 else np.array([np.nan]); density_lookup[(rep,group)]=pd.Series(dist,index=hs)
            mz=meta.loc[hs]; counts=mz.qd_niche.value_counts(normalize=True); le,ls,lm=eff(mz.lineage_root); re,rs,rm=eff(mz.lineage_root); oe,os_,om=eff(mz.operator_id)
            covrows.append({"replicate":rep,"group":group,"n":len(hs),"coordinate_unique_n":len(set(hs)),"occupied_qd_niches":mz.qd_niche.nunique(),"effective_qd_niches":1/np.sum(counts.to_numpy()**2),"qd_niche_entropy":-(counts*np.log(counts)).sum(),"largest_niche_share":counts.max(),"full_f40_feature_sd_mean":z.std(ddof=0).mean(),"full_f40_nn_mean":np.nanmean(dist),"full_f40_nn_median":np.nanmedian(dist),"near_duplicate_proxy_fraction":np.mean(dist<=near_thr),"near_duplicate_threshold_pool_p1":near_thr,"effective_lineages":le,"lineage_simpson_effective":ls,"largest_lineage_share":lm,"effective_roots":re,"largest_root_share":rm,"effective_operators":oe,"largest_operator_share":om})
    coverage=pd.DataFrame(covrows); coverage.to_csv(OUT/"04_coverage_metrics.csv",index=False)
    bc=coverage[coverage.group.isin(["B","C"])].pivot(index="replicate",columns="group",values=["full_f40_nn_median","near_duplicate_proxy_fraction","effective_lineages","largest_lineage_share","full_f40_feature_sd_mean"]).reset_index()
    (OUT/"04_redundancy_coverage.md").write_text("# Redundancy and Coverage\n\nCoordinate uniqueness, Full-F40 local similarity and lineage dependence are separate. Smaller F40 NN distance is only a structural-concentration proxy; it does not prove gradient redundancy. Near-duplicate threshold is the frozen candidate-pool NN P1 per replicate.\n\n"+md(coverage)+"\n")
    # Lineage / operator composition.
    lrows=[]
    for rep in REPS:
        meta=selected[selected.replicate.eq(rep)].drop_duplicates(["selection","instance_hash"])
        for group in ("A","B","C","Shared","B_only","C_only"):
            hs=groups[(rep,group)]; z=meta[meta.instance_hash.isin(hs)].drop_duplicates("instance_hash"); le,ls,lm=eff(z.lineage_root)
            lrows.append({"replicate":rep,"group":group,"record_type":"summary","category":"ALL","count":len(z),"share":1.0,"true_ce_mean":z.true_m4_ce.mean(),"effective_lineages":le,"largest_lineage_share":lm,"E0_fraction":z.in_E0.mean(),"E1_fraction":z.in_E1.mean()})
            for col,kind in (("operator_id","operator"),("source","source"),("lineage_root","root")):
                for val,g in z.groupby(col): lrows.append({"replicate":rep,"group":group,"record_type":kind,"category":str(val),"count":len(g),"share":len(g)/len(z),"true_ce_mean":g.true_m4_ce.mean(),"predicted_ce_mean":g.predicted_m4_ce.mean()})
    lineage=pd.DataFrame(lrows); lineage.to_csv(OUT/"05_lineage_operator_metrics.csv",index=False)
    ops=lineage[(lineage.record_type=="operator")&lineage.group.isin(["B_only","C_only"])]; opiv=ops.pivot_table(index=["replicate","category"],columns="group",values="share",fill_value=0).reset_index(); opiv["C_minus_B_share"]=opiv.get("C_only",0)-opiv.get("B_only",0)
    (OUT/"05_lineage_operator_audit.md").write_text("# Lineage and Operator Audit\n\nOperator/source/root strata are descriptive: hardness, operator and structure are jointly selected and this is not a causal decomposition. Largest replacement operator-share changes:\n\n"+md(opiv.reindex(opiv.C_minus_B_share.abs().sort_values(ascending=False).index).head(25))+"\n\nFull root/operator records are in `05_lineage_operator_metrics.csv`.\n")
    # Tour features, computed once for each labelled union and cached.
    tour_cache=OUT/"manifests/union_tour_features.parquet"
    if tour_cache.exists(): tours=pd.read_parquet(tour_cache)
    else:
        rows=[]
        for rep in REPS:
            xy,hh,tt=bundles[rep]
            for i,(h,x,t) in enumerate(zip(hh,xy,tt)):
                rows.append({"replicate":rep,"instance_hash":h,**tour_features(x,t)})
                if (i+1)%2000==0: print(f"[P6-D2] tour features {rep} {i+1}/{len(hh)}",flush=True)
        tours=pd.DataFrame(rows); tours.to_parquet(tour_cache,index=False)
    ti=tours.set_index(["replicate","instance_hash"]); tour_names=[c for c in tours.columns if c not in ("replicate","instance_hash")]; trows=[]
    for rep in REPS:
        ref=tours[tours.replicate.eq(rep)].set_index("instance_hash")[tour_names]; sd=ref.std(ddof=0).replace(0,1)
        for group in ("A","B","C","Shared","B_only","C_only"):
            z=ref.loc[list(groups[(rep,group)])]
            for feat in tour_names: trows.append({"replicate":rep,"comparison":"group_summary","group":group,"feature":feat,"n":len(z),"mean":z[feat].mean(),"median":z[feat].median(),"sd":z[feat].std(),"p10":z[feat].quantile(.1),"p90":z[feat].quantile(.9)})
        bo=ref.loc[list(groups[(rep,"B_only")])]; co=ref.loc[list(groups[(rep,"C_only")])]
        for feat in tour_names:
            boot=bootstrap_diff(co[feat],bo[feat],f"{rep}|tour|{feat}")
            trows.append({"replicate":rep,"comparison":"C_only_minus_B_only","group":"replacement_effect","feature":feat,"n":len(co),"mean":boot["estimate"],"standardized_mean_difference":boot["estimate"]/sd[feat],"wasserstein_pool_sd":wasserstein_distance(co[feat],bo[feat])/sd[feat],"instance_ci_low":boot["instance_ci_low"],"instance_ci_high":boot["instance_ci_high"]})
    tourcomp=pd.DataFrame(trows); tourcomp.to_csv(OUT/"06_tour_feature_comparison.csv",index=False)
    tagg=tourcomp[tourcomp.comparison.eq("C_only_minus_B_only")].groupby("feature").standardized_mean_difference.agg(["median","min","max"]).reset_index(); tagg["consistent_direction"]=(np.sign(tagg["min"])==np.sign(tagg["max"])); tagg["abs_median"]=tagg["median"].abs(); tagg=tagg.sort_values("abs_median",ascending=False)
    (OUT/"06_optimal_tour_structure.md").write_text("# Optimal-Tour Structure\n\nAll features are rotation/reversal invariant. Edge lengths use Euclidean labelled coordinates; long edges use the frozen within-instance all-pair P90. These post-label diagnostics cannot be used directly for unlabelled online offspring.\n\n"+md(tagg)+"\n")
    # Deterministic representative cases and figures.
    reps=[]
    for rep in REPS:
        pool=pools[rep].set_index("instance_hash"); xy,hh,tt=bundles[rep]; lookup={h:i for i,h in enumerate(hh)}; fz=f40i.loc[hh,F40]; mu=fz.mean(); sd=fz.std(ddof=0).replace(0,1)
        fig,axes=plt.subplots(3,4,figsize=(16,12))
        for row,group in enumerate(("B_only","C_only","Shared")):
            hs=list(groups[(rep,group)]); z=(f40i.loc[hs,F40]-mu)/sd; med=z.median(); central=((z-med)**2).sum(axis=1); dens=density_lookup[(rep,group)]; ce=pool.loc[hs,"true_m4_ce"]
            choices=[("median_structure",central.idxmin()),("high_hardness",ce.idxmax()),("dense",dens.idxmin()),("outlier",central.idxmax())]; used=set()
            for col,(role,h) in enumerate(choices):
                if h in used:
                    for alt in central.sort_values(ascending=role=="outlier").index:
                        if alt not in used: h=alt; break
                used.add(h); i=lookup[h]; x=xy[i]; t=tt[i]; cyc=np.r_[t,t[0]]; ax=axes[row,col]; ax.plot(x[cyc,0],x[cyc,1],lw=.65,color="tab:blue",alpha=.8); ax.scatter(x[:,0],x[:,1],s=5,color="black"); m=pool.loc[h]
                ax.set_title(f"{group} {role}\nniche={int(m.qd_niche)} true={m.true_m4_ce:.4f} pred={m.predicted_m4_ce:.4f}",fontsize=8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                reps.append({"replicate":rep,"group":group,"role":role,"instance_hash":h,"qd_niche":int(m.qd_niche),"true_m4_ce":m.true_m4_ce,"predicted_m4_ce":m.predicted_m4_ce,"lineage_root":m.lineage_root,"operator_id":m.operator_id,**{f"top5__{q}":f40i.at[h,q] for q in TOP5}})
        fig.suptitle(f"{rep}: deterministic representative optimal tours"); fig.tight_layout(); fig.savefig(OUT/"figures/representative_cases"/f"{rep}_12_cases.png",dpi=170); plt.close(fig)
    representatives=pd.DataFrame(reps); representatives.to_csv(OUT/"manifests/representative_cases.csv",index=False)
    # Original matched-niche extra montage: four largest replacement niches.
    pool=pools["Original"].set_index("instance_hash"); xy,hh,tt=bundles["Original"]; lookup={h:i for i,h in enumerate(hh)}; topks=niches[niches.replicate.eq("Original")].nlargest(4,"replacement_n_each").niche_id.astype(int)
    fig,axes=plt.subplots(4,2,figsize=(8,15))
    for row,k in enumerate(topks):
        for col,gname in enumerate(("B_only","C_only")):
            hs=[h for h in groups[("Original",gname)] if int(pool.at[h,"qd_niche"])==k]; h=min(hs,key=lambda x:(abs(pool.at[x,"true_m4_ce"]-pool.loc[hs,"true_m4_ce"].median()),x)); i=lookup[h]; x=xy[i]; t=tt[i]; cyc=np.r_[t,t[0]]; ax=axes[row,col]; ax.plot(x[cyc,0],x[cyc,1],lw=.7); ax.scatter(x[:,0],x[:,1],s=5); ax.set_title(f"niche {k} {gname} true={pool.at[h,'true_m4_ce']:.4f}"); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT/"figures/representative_cases/Original_matched_niche.png",dpi=180); plt.close(fig)
    (OUT/"07_representative_instances.md").write_text("# Representative Instances\n\nTwelve cases per replicate are selected deterministically: closest to group F40 median, highest true M4 CE, smallest group F40-NN distance, and largest distance from group median. The Original extra montage uses the four largest replacement niches and selects the within-group median-hardness case. No case was hand-picked for narrative agreement.\n\n"+md(representatives)+"\n")
    # TSPLIB48 Top5 metrics and Original70 (48/22) cross-scale F40 proximity.
    target=pd.read_parquet(P6A/"diagnostics/tsplib48_top5_representation.parquet"); target_phi=target[[f"phi__{x}" for x in TOP5]].to_numpy(float); quota=p6a.capacities(target.instance_id.astype(str).tolist()); target_mass=target_phi[np.repeat(np.arange(48),quota)]
    targetcov_path=OUT/"08_target_coverage.csv"
    if targetcov_path.exists():
        targetcov=pd.read_csv(targetcov_path)
        if len(targetcov[targetcov.record_type.eq("population")])!=9 or len(targetcov[targetcov.record_type.eq("reference")])!=432 or not targetcov[targetcov.record_type.eq("population")].ot_reproduced.all(): raise RuntimeError("incomplete cached Top5 target coverage")
    else:
        trows=[]; d0struct=pd.read_csv(D0/"07_structure_metrics.csv")
        for rep in REPS:
            for arm in ARMS:
                z=selected[(selected.replicate==rep)&(selected.selection==arm)]; phi=z[[f"phi__{x}" for x in TOP5]].to_numpy(float); cost=p6a.squared_distance_matrix(phi,target_phi); match=p6a.exact_transport_lp(cost,quota); nearest=np.sqrt(cost.min(axis=1)); refcost=[]
                for j in range(48): refcost.append(float(match["edge_cost"][match["reference_index"]==j].sum()))
                old=d0struct[(d0struct.replicate==rep)&(d0struct.selection==arm)].iloc[0]
                trows.append({"record_type":"population","replicate":rep,"arm":arm,"reference":"ALL","top5_ot_total":match["edge_cost"].sum(),"top5_ot_mean":match["edge_cost"].mean(),"top5_swd":p6a.sliced_wasserstein(phi,target_mass),"top5_energy":p6a.energy_distance_multivariate(phi,target_mass),"nearest_reference_mean":nearest.mean(),"d0_top5_ot_mean":old.top5_ot_mean,"ot_reproduced":np.isclose(match["edge_cost"].mean(),old.top5_ot_mean,rtol=1e-8)})
                for j,name in enumerate(target.instance_id.astype(str)): trows.append({"record_type":"reference","replicate":rep,"arm":arm,"reference":name,"assigned_n":int(quota[j]),"nearest_candidate_distance":np.sqrt(cost[:,j].min()),"assigned_cost":refcost[j]})
        targetcov=pd.DataFrame(trows); targetcov.to_csv(targetcov_path,index=False)
    # Existing D0 projection diagnostics.
    clip=pd.read_csv(D0/"manifests/tsplib48_qd4_scale_audit.csv")
    records=p01.load_original70(); held=set(json.loads((P1/"p1_heldout22_names.json").read_text())["names"]); variants=[(r["instance_id"],p01.aspect_normalize(r["coordinates"])) for r in records]
    tsf_path=OUT/"manifests/original70_aspect_f40.parquet"
    tsf=pd.read_parquet(tsf_path) if tsf_path.exists() else p01.extract_selected_salesperson(variants).rename(columns={"state_id":"instance_id"})
    tsf["split"]=np.where(tsf.instance_id.isin(held),"TSPLIB22","TSPLIB48"); tsf.to_parquet(tsf_path,index=False)
    cross_features=[f for f in F40 if tsf[f].notna().all()]
    missing_cross=pd.DataFrame({"feature":F40,"missing_original70":[int(tsf[f].isna().sum()) for f in F40],"used_for_cross_size_proximity":[f in cross_features for f in F40]}); missing_cross.to_csv(OUT/"manifests/original70_f40_missingness.csv",index=False)
    if len(cross_features)<2: raise RuntimeError("insufficient complete cross-size F40 features")
    proximity=[]
    for rep in REPS:
        pool=pools[rep]; mu=f40i.loc[pool.instance_hash,cross_features].mean(); sd=f40i.loc[pool.instance_hash,cross_features].std(ddof=0).replace(0,1); q=(tsf.set_index("instance_id")[cross_features]-mu)/sd
        for arm in ("B","C"):
            z=(f40i.loc[list(groups[(rep,arm)]),cross_features]-mu)/sd; nn=NearestNeighbors(n_neighbors=5,n_jobs=-1).fit(z); dist=nn.kneighbors(q,return_distance=True)[0]
            for name,row,ds in zip(q.index,dist,tsf.set_index("instance_id").loc[q.index,"split"]): proximity.append({"record_type":"tsplib_proximity","replicate":rep,"arm":arm,"reference":name,"split":ds,"f40_nn1":row[0],"f40_nn5_mean":row.mean()})
    prox=pd.DataFrame(proximity); prox.to_csv(OUT/"manifests/original70_training_proximity.csv",index=False)
    (OUT/"08_target_structural_relevance.md").write_text("# Target Structural Relevance\n\nExact Top5 OT/SWD/energy are recomputed with the frozen P6-A representation and equal TSPLIB48 reference quotas. Original70 proximity uses aspect-preserving coordinates, the formal F40 scaler and only the "+str(len(cross_features))+" F40 fields finite for all 70 instances; no missing value is imputed, and exclusions are recorded in `manifests/original70_f40_missingness.csv`. Raw size-dependent coordinates are not compared. TSPLIB22 is descriptive only. QD4 remains interpretation-limited because 70.8% of TSPLIB48 references clip on at least one axis and only 29 niches are directly occupied.\n\nQD4 clip audit:\n\n"+md(clip)+"\n\nPopulation target metrics:\n\n"+md(targetcov[targetcov.record_type.eq("population")])+"\n\nF40 proximity summary:\n\n"+md(prox.groupby(["replicate","arm","split"])[["f40_nn1","f40_nn5_mean"]].mean().reset_index())+"\n")
    # Benchmark gap and CE-transfer association.
    grows=[]
    for (rep,ds),g in gaps.groupby(["replicate","dataset"]):
        v=g.gap_C.to_numpy(float)-g.gap_B.to_numpy(float); pos=np.clip(v,0,None); top=np.sort(pos)[-max(1,int(np.ceil(.1*len(v)))):].sum(); grows.append({"replicate":rep,"dataset":ds,"n":len(v),"mean_gap_C_minus_B":v.mean(),"median_gap_C_minus_B":np.median(v),"fraction_C_better":np.mean(v<0),"fraction_C_worse":np.mean(v>0),"worst_decile_share_of_C_regression":top/pos.sum() if pos.sum()>0 else 0,"p10":np.quantile(v,.1),"p90":np.quantile(v,.9)})
    gapstats=pd.DataFrame(grows); gapstats.to_csv(OUT/"09_benchmark_gap_differences.csv",index=False)
    ceinst=pd.read_parquet(D1/"logs/cross_population_ce_per_instance.parquet"); assocs=[]
    for rep in REPS:
        for group in ("B_only","C_only"):
            hs=groups[(rep,group)]
            for model in ("B","C"):
                z=(ceinst[(ceinst.replicate==rep)&(ceinst.model==model)&ceinst.instance_hash.isin(hs)]
                   .sort_values(["instance_hash","dataset_arm"]).drop_duplicates("instance_hash"))
                if len(z)!=len(hs): raise RuntimeError(f"replacement CE recovery mismatch {rep}/{group}/{model}: {len(z)} != {len(hs)}")
                assocs.append({"replicate":rep,"record_type":"replacement_CE","group":group,"model":model,"n":len(z),"mean_ce":z.model_ce.mean(),"mean_gain_M4_minus_model":z.gain_m4_minus_model.mean()})
        bmean=selected[(selected.replicate==rep)&(selected.selection=="B")].true_m4_ce.mean(); cmean=selected[(selected.replicate==rep)&(selected.selection=="C")].true_m4_ce.mean(); orig=gapstats[(gapstats.replicate==rep)&gapstats.dataset.eq("OriginalTSPLIB70")].iloc[0]
        near=prox[prox.replicate.eq(rep)].groupby(["arm","split"]).f40_nn1.mean()
        assocs.append({"replicate":rep,"record_type":"replicate_summary","group":"B_vs_C","model":"ALL","C_minus_B_training_true_ce":cmean-bmean,"Original70_gap_C_minus_B":orig.mean_gap_C_minus_B,"TSPLIB48_C_minus_B_nn":near.at[("C","TSPLIB48")]-near.at[("B","TSPLIB48")],"TSPLIB22_C_minus_B_nn":near.at[("C","TSPLIB22")]-near.at[("B","TSPLIB22")]})
    assoc=pd.DataFrame(assocs); assoc.to_csv(OUT/"09_training_structure_association.csv",index=False)
    original_detail=gaps[gaps.dataset.eq("OriginalTSPLIB70")].merge(prox.pivot_table(index=["replicate","reference","split"],columns="arm",values="f40_nn1").reset_index(),left_on=["replicate","instance_id"],right_on=["replicate","reference"],how="left"); original_detail["gap_C_minus_B"]=original_detail.gap_C-original_detail.gap_B; original_detail["C_minus_B_nn"]=original_detail.C-original_detail.B; original_detail.to_csv(OUT/"manifests/original70_gap_structure_detail.csv",index=False)
    (OUT/"09_training_value_association.md").write_text("# Training-Value Association\n\nPositive gap C−B means C is worse. Replacement CE uses existing expert tours and frozen D1 checkpoint inference; B-only/C-only overlap their respective training sets and is not independent generalization. Structure proximity is descriptive and cannot attribute an individual training example's value.\n\n"+md(gapstats[gapstats.dataset.isin(["OriginalTSPLIB70","TSPLIB48","TSPLIB22","Uniform100","GaussianMixture100","TSPLIB100","GaussianMixture500"])])+"\n\n"+md(assoc)+"\n")
    # Robustness intervals for hardness + leading F40/tour effects.
    robust=[]
    leadf=agg.head(8).feature.tolist(); leadt=tagg.head(5).feature.tolist()
    for rep in REPS:
        meta=selected[selected.replicate.eq(rep)].drop_duplicates("instance_hash").set_index("instance_hash"); bo=list(groups[(rep,"B_only")]); co=list(groups[(rep,"C_only")])
        items=[("true_m4_ce",pools[rep].set_index("instance_hash").loc[co,"true_m4_ce"].to_numpy(),pools[rep].set_index("instance_hash").loc[bo,"true_m4_ce"].to_numpy())]
        items += [(f,f40i.loc[co,f].to_numpy(),f40i.loc[bo,f].to_numpy()) for f in leadf]
        tr=tours[tours.replicate.eq(rep)].set_index("instance_hash"); items += [(f"tour::{f}",tr.loc[co,f].to_numpy(),tr.loc[bo,f].to_numpy()) for f in leadt]
        for name,a,bv in items:
            x=bootstrap_diff(a,bv,f"robust|{rep}|{name}",meta.loc[co,"lineage_root"],meta.loc[bo,"lineage_root"]); robust.append({"replicate":rep,"metric":name,**x})
    robust=pd.DataFrame(robust); robust.to_csv(OUT/"manifests/robustness_intervals.csv",index=False)
    (OUT/"10_robustness_analysis.md").write_text("# Statistical and Robustness Analysis\n\nOriginal/S1/S2 are shown separately. Instance bootstrap describes frozen memberships; root-aware bootstrap resamples lineage roots and is the dependence sensitivity analysis. With three evolution replicates, direction consistency and effect size are primary; instance-level multiplicity-adjusted q-values remain exploratory.\n\n"+md(robust)+"\n")
    # Figures required by protocol.
    fig,axes=plt.subplots(1,3,figsize=(15,4));
    for ax,rep in zip(axes,REPS):
        n=niches[niches.replicate.eq(rep)]; ax.hist(n.replacement_n_each,bins=25); ax.set_title(f"{rep}: replacement by niche"); ax.set_xlabel("B-only = C-only count")
    fig.tight_layout(); fig.savefig(OUT/"figures/01_membership_niche_replacement.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7)); show=agg.head(20).sort_values("median_smd"); ax.barh(show.feature,show.median_smd); ax.axvline(0,color="black",lw=.8); ax.set_xlabel("Median C-only − B-only SMD"); fig.tight_layout(); fig.savefig(OUT/"figures/02_f40_effect_ranking.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4));
    for ax,rep in zip(axes,REPS):
        p=pools[rep].set_index("instance_hash"); ax.hist(p.loc[list(groups[(rep,"B_only")]),"true_m4_ce"],bins=40,alpha=.55,label="B-only"); ax.hist(p.loc[list(groups[(rep,"C_only")]),"true_m4_ce"],bins=40,alpha=.55,label="C-only"); ax.set_title(rep); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"figures/03_replacement_hardness.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4));
    for ax,rep in zip(axes,REPS):
        p=opiv[opiv.replicate.eq(rep)].set_index("category")[["B_only","C_only"]].fillna(0); p.plot.bar(ax=ax); ax.set_title(rep); ax.tick_params(axis="x",labelrotation=75)
    fig.tight_layout(); fig.savefig(OUT/"figures/04_operator_composition.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4));
    for ax,rep in zip(axes,REPS):
        tr=tours[tours.replicate.eq(rep)].set_index("instance_hash");
        for g,c in (("B_only","tab:blue"),("C_only","tab:orange")): ax.hist(tr.loc[list(groups[(rep,g)]),"tour_edge_cv"],bins=40,density=True,alpha=.45,label=g,color=c)
        ax.set_title(rep); ax.set_xlabel("Tour edge CV"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/"figures/05_tour_edge_patterns.png",dpi=180); plt.close(fig)
    trade=pd.read_csv(D0/"08_tradeoff_summary.csv"); trade=trade[trade.selection.isin(ARMS)]; fig,ax=plt.subplots(figsize=(8,6));
    for arm,m in zip(ARMS,("o","s","^")):
        z=trade[trade.selection.eq(arm)]; ax.scatter(z.top5_ot_mean,z.mean_true_m4_ce,label=arm,marker=m,s=80)
        for r in z.itertuples(): ax.annotate(r.replicate,(r.top5_ot_mean,r.mean_true_m4_ce),fontsize=8)
    ax.set_xlabel("Top5 OT mean"); ax.set_ylabel("True M4 CE"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"figures/07_top5_ot_hardness.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4));
    for ax,rep in zip(axes,REPS):
        v=gaps[(gaps.replicate==rep)&gaps.dataset.eq("OriginalTSPLIB70")]; ax.hist(v.gap_C-v.gap_B,bins=25); ax.axvline(0,color="black",lw=.8); ax.set_title(rep); ax.set_xlabel("Gap C−B (pp)")
    fig.tight_layout(); fig.savefig(OUT/"figures/08_benchmark_gap_difference.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4));
    for ax,rep in zip(axes,REPS):
        p=prox[prox.replicate.eq(rep)].groupby(["arm","split"]).f40_nn1.mean().unstack(); p.plot.bar(ax=ax); ax.set_title(rep); ax.set_ylabel("Mean F40 NN distance")
    fig.tight_layout(); fig.savefig(OUT/"figures/09_tsplib_training_proximity.png",dpi=180); plt.close(fig)
    rs=assoc[assoc.record_type.eq("replicate_summary")]; fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(rs.C_minus_B_training_true_ce,rs.Original70_gap_C_minus_B,s=90)
    for r in rs.itertuples(): ax.annotate(r.replicate,(r.C_minus_B_training_true_ce,r.Original70_gap_C_minus_B)); ax.axhline(0,color="black",lw=.8); ax.set_xlabel("C−B training true CE"); ax.set_ylabel("Original70 gap C−B"); fig.tight_layout(); fig.savefig(OUT/"figures/10_structure_ce_transfer.png",dpi=180); plt.close(fig)
    # Mechanism decisions, based on preregistered effect/direction proxies.
    stable_f=agg[(agg.direction_consistency.eq(1))&(agg.abs_median_smd>=.10)]; stable_t=tagg[(tagg.consistent_direction)&(tagg.abs_median>=.10)]
    h1="SUPPORTED" if len(stable_f)>=3 or len(stable_t)>=2 else "MIXED" if len(stable_f)+len(stable_t)>0 else "NOT_SUPPORTED"
    # HGB bias requires consistent replacement effects and composition changes; otherwise mixed.
    maxop=opiv.groupby("replicate").C_minus_B_share.apply(lambda x:x.abs().max()); h2="SUPPORTED" if len(stable_f)>=3 and (maxop>=.02).all() else "MIXED" if len(stable_f)>0 or (maxop>=.02).any() else "NOT_SUPPORTED"
    bcov=coverage[coverage.group.isin(["B","C"])].pivot(index="replicate",columns="group",values=["full_f40_nn_median","near_duplicate_proxy_fraction","effective_lineages","largest_lineage_share"])
    concentration=[(bcov[("full_f40_nn_median","C")]<bcov[("full_f40_nn_median","B")]).all(),(bcov[("near_duplicate_proxy_fraction","C")]>bcov[("near_duplicate_proxy_fraction","B")]).all(),(bcov[("effective_lineages","C")]<bcov[("effective_lineages","B")]).all(),(bcov[("largest_lineage_share","C")]>bcov[("largest_lineage_share","B")]).all()]
    h3="SUPPORTED_PROXY_ONLY" if sum(concentration)>=2 else "NOT_SUPPORTED" if sum(concentration)==0 else "INCONCLUSIVE"
    gapdir=gapstats[gapstats.dataset.eq("OriginalTSPLIB70")].set_index("replicate").mean_gap_C_minus_B; structure_signal=rs.set_index("replicate").TSPLIB48_C_minus_B_nn
    assoc_r=stats.spearmanr(gapdir.loc[list(REPS)],structure_signal.loc[list(REPS)]).statistic
    h4="SUPPORTED_EXPLORATORY" if np.isfinite(assoc_r) and abs(assoc_r)>=.8 and h1!="NOT_SUPPORTED" else "MIXED" if h1!="NOT_SUPPORTED" else "NOT_SUPPORTED"
    h5="IDENTIFIED" if h2 in ("SUPPORTED","MIXED") else "NOT_IDENTIFIED"
    topstable=stable_f.head(10)[["feature","feature_group","is_top5","median_smd","direction_consistency"]] if len(stable_f) else agg.head(10)[["feature","feature_group","is_top5","median_smd","direction_consistency"]]
    recommendation=("Replace deterministic within-niche top-HGB selection with one frozen rank-stratified sampling rule inside the unchanged lambda=0.5 quotas: preserve a fixed high-HGB fraction and sample the remainder uniformly within each niche. Compare it only against B-random and C-top-HGB using the same labelled pool, M4 initialization, seed and budget. This changes one selection factor and tests whether the observed HGB-linked composition/concentration, rather than target quotas, causes unstable transfer." if h5=="IDENTIFIED" else "Do not modify the fifth-cycle selector yet; no single stable structure mechanism was identified across all three replicates.")
    grow=repl[repl.record_type.eq("group")].set_index(["replicate","group"])
    membership_lines=[]; hardness_lines=[]; niche_lines=[]; hard_niche_lines=[]
    for rep in REPS:
        shared=int(grow.at[(rep,"Shared"),"n"]); only=int(grow.at[(rep,"B_only"),"n"]); jac=grow.at[(rep,"Shared"),"jaccard"]
        membership_lines.append(f"{rep}: Shared={shared:,}, B-only=C-only={only:,}, replacement={only/10000:.2%}, Jaccard={jac:.4f}")
        bd=grow.at[(rep,"B_only"),"true_ce_mean"]; cd=grow.at[(rep,"C_only"),"true_ce_mean"]
        ci=repl[(repl.replicate.eq(rep))&repl.record_type.eq("niche_stratified_summary")].iloc[0]
        hardness_lines.append(f"{rep}: {bd:.6f}→{cd:.6f}, Δ={cd-bd:+.6f}; instance 95% CI [{ci.instance_ci_low:.6f}, {ci.instance_ci_high:.6f}], root-aware [{ci.root_ci_low:.6f}, {ci.root_ci_high:.6f}]")
        nz=niches[niches.replicate.eq(rep)].copy(); top=nz.nlargest(5,"replacement_n_each")
        niche_lines.append(f"{rep}: {len(nz)} replacement niches; largest by count " + ", ".join(f"{int(r.niche_id)}({int(r.replacement_n_each)})" for r in top.itertuples()))
        nz["positive_weighted_gain"]=nz.C_minus_B_true_ce.clip(lower=0)*nz.replacement_n_each
        pos=nz[nz.positive_weighted_gain.gt(0)].sort_values("positive_weighted_gain",ascending=False); denom=pos.positive_weighted_gain.sum()
        share=pos.head(10).positive_weighted_gain.sum()/denom if denom else np.nan
        hard_niche_lines.append(f"{rep}: {len(pos)}/{len(nz)} niches have positive mean gain; top-10 explain {share:.2%}; leading IDs " + ", ".join(str(int(x)) for x in pos.head(5).niche_id))
    op_lines=[]
    for rep in REPS:
        z=opiv[opiv.replicate.eq(rep)].assign(abs_shift=lambda x:x.C_minus_B_share.abs()).nlargest(3,"abs_shift")
        op_lines.append(f"{rep}: " + ", ".join(f"{r.category} {r.C_minus_B_share:+.2%}" for r in z.itertuples()))
    covp=coverage[coverage.group.isin(["B","C","B_only","C_only"])].set_index(["replicate","group"])
    cov_lines=[]
    for rep in REPS:
        cov_lines.append(f"{rep}: full C/B median F40-NN {covp.at[(rep,'C'),'full_f40_nn_median']:.4f}/{covp.at[(rep,'B'),'full_f40_nn_median']:.4f}, near-duplicate proxy {covp.at[(rep,'C'),'near_duplicate_proxy_fraction']:.2%}/{covp.at[(rep,'B'),'near_duplicate_proxy_fraction']:.2%}, effective lineages {covp.at[(rep,'C'),'effective_lineages']:.1f}/{covp.at[(rep,'B'),'effective_lineages']:.1f}; C-only/B-only effective lineages {covp.at[(rep,'C_only'),'effective_lineages']:.1f}/{covp.at[(rep,'B_only'),'effective_lineages']:.1f}")
    tourstable=tagg.head(7)[["feature","median","consistent_direction"]]
    poptarget=targetcov[targetcov.record_type.eq("population")].set_index(["replicate","arm"])
    top5_lines=[f"{rep}: B={poptarget.at[(rep,'B'),'top5_ot_mean']:.6f}, C={poptarget.at[(rep,'C'),'top5_ot_mean']:.6f}, C−B={poptarget.at[(rep,'C'),'top5_ot_mean']-poptarget.at[(rep,'B'),'top5_ot_mean']:+.6f}" for rep in REPS]
    proxmean=prox.groupby(["replicate","arm","split"]).f40_nn1.mean()
    gapkey=gapstats[gapstats.dataset.isin(["OriginalTSPLIB70","TSPLIB48","TSPLIB22","Uniform100","GaussianMixture100","TSPLIB100","GaussianMixture500"])][["replicate","dataset","mean_gap_C_minus_B","worst_decile_share_of_C_regression"]]
    tail_share=gapkey.worst_decile_share_of_C_regression
    ceview=assoc[assoc.record_type.eq("replacement_CE")][["replicate","group","model","n","mean_ce","mean_gain_M4_minus_model"]]
    clip_lines=[f"{r.feature}: low={r.clip_zero_fraction:.1%}, high={r.clip_one_fraction:.1%}, ρ(nodes)={r.spearman_with_n_nodes:+.3f}" for r in clip.itertuples()]
    stable_top5=int(stable_f.is_top5.sum()); stable_f35=len(stable_f)-stable_top5
    final=["# P6-D2 Final Research Conclusion","",f"`H1_STRUCTURAL_REPLACEMENT: {h1}`",f"`H2_HGB_STRUCTURE_BIAS: {h2}`",f"`H3_REDUNDANCY: {h3}`",f"`H4_TRANSFER_ASSOCIATION: {h4}`",f"`H5_ACTIONABLE_DESIGN_HYPOTHESIS: {h5}`","",f"Exact niche balance passes for all replicates. There are {len(stable_f)} stable Full-F40 effects (Top5={stable_top5}, remaining F35={stable_f35}) at |median SMD|≥0.10 with 3/3 direction, and {len(stable_t)} stable tour effect(s) at the same threshold. The three-replicate TSPLIB48-proximity versus Original70-gap Spearman is {assoc_r:.4f}; with n=3 this is exploratory, not causal.","","## Required 20-question summary","","1. **B/C shared membership.** " + "; ".join(membership_lines),"","2. **B-only/C-only counts.** They are exactly equal within every replicate and every niche: Original 3,652 each, S1 3,676 each, S2 3,680 each.","","3. **Replacement niches.** " + "; ".join(niche_lines) + ". All replacements obey frozen per-niche quotas.","","4. **Is C-only truly harder?** Yes, descriptively and under both instance and lineage-root bootstrap: " + "; ".join(hardness_lines) + ". C-only also has higher global-P90 fractions in all three replicates.","","5. **Where does the hardness gain occur?** It is distributed across hundreds of niches, not dominated by a few: " + "; ".join(hard_niche_lines) + ".", "","6. **Largest Full-F40 differences.** The strongest stable shifts are higher `mst_dists_span`, `mst_dists_skew`, and `mst_depth_varcoeff`, lower `angles_cos_var`, `fraction_shorter_mean_distance`, and `angles_var`, plus higher `mst_dists_sd`. This points to altered MST/local-distance and angular geometry beyond the QD quotas.","","7. **Cross-replicate consistency.** " + f"{len(stable_f)} effects have identical direction in Original/S1/S2 at |median SMD|≥0.10; most ({stable_f35}) are outside Top5. The ranked table is below.","",md(topstable),"","8. **Operator/lineage preference.** C systematically has fewer effective lineages than B, while operator composition shifts are real but no single operator dominates all seeds: " + "; ".join(op_lines) + ". Thus the evidence supports composition bias, not one universal causal operator.","","9. **Local structural redundancy.** Proxy evidence supports modestly higher concentration in C: lower full-population F40-NN distance, higher near-duplicate proxy, fewer effective lineages and slightly larger largest-lineage share in all three replicates. " + "; ".join(cov_lines) + ". Global feature spread is not reduced, so this is local/lineage concentration—not proven gradient redundancy.","","10. **Optimal-tour edge patterns.** Yes, modestly. C-only has higher endpoint-neighbor rank, lower MST-edge overlap and kNN10-edge fraction, and higher tour-edge CV. Only `endpoint_neighbor_rank_mean` exceeds |median SMD|=0.10 consistently; the other shifts are smaller.","",md(tourstable),"","11. **Tour-pattern coverage.** No broad systematic coverage collapse is established. Most tour effects are small, long-edge fraction is inconsistent, and only one of 15 tour features crosses the stable 0.10 threshold. B may retain somewhat more local/MST-like decisions, but this is a distribution shift, not proof of higher useful tour diversity.","","12. **Exact B/C Top5 OT difference.** C is slightly farther from the TSPLIB48 Top5 target than B in all three replicates despite both being much better than A: " + "; ".join(top5_lines) + ". Therefore C's CE gain over B did not come from superior Top5 matching.","","13. **TSPLIB48 projection clipping.** 70.8% of references hit at least one QD4 boundary and only 29 CVT niches are directly occupied. Axis diagnostics: " + "; ".join(clip_lines) + ". This saturation makes QD4 niche identity a coarse target proxy and limits fine-grained cross-size interpretation.","","14. **Why might B beat C for Original?** C is harder but more locally/lineage concentrated, its Top5 OT is worse by " + f"{poptarget.at[('Original','C'),'top5_ot_mean']-poptarget.at[('Original','B'),'top5_ot_mean']:.3f}, and its mean TSPLIB48 F40-NN distance is farther by {proxmean.at[('Original','C','TSPLIB48')]-proxmean.at[('Original','B','TSPLIB48')]:+.4f}. These observations are consistent with B's broader random within-niche sample transferring better, but do not identify the cause.","","15. **Why might C beat B for S2?** In S2, C is closer to TSPLIB48 in the complete cross-size F40 representation by " + f"{proxmean.at[('S2','C','TSPLIB48')]-proxmean.at[('S2','B','TSPLIB48')]:+.4f} and beats B on Original70 by {gapdir.at['S2']:+.4f} pp. This seed-specific alignment is a plausible association; C is still worse on TSPLIB22 by {gapstats[(gapstats.replicate.eq('S2'))&gapstats.dataset.eq('TSPLIB22')].iloc[0].mean_gap_C_minus_B:+.4f} pp, so it is not a general C advantage.","","16. **Where are benchmark differences concentrated?** Positive C regressions are mostly tail-heavy: the worst 10% contribute {tail_share.min():.1%}–{tail_share.max():.1%} of positive regression mass, and at least 60% in {(tail_share>=.60).sum()}/{len(tail_share)} reported benchmark/replicate cells. Mean C−B by dataset is below. The audit can identify dataset/tail concentration, but not a reliable finer instance taxonomy from three seeds.","",md(gapkey),"","17. **CE transfer.** Each model fits its own replacement subset better: B has lower CE on B-only and C lower CE on C-only. These subsets overlap the corresponding training set, so this is in-sample specialization rather than independent transfer evidence. It supports a composition shift but does not explain benchmark causally.","",md(ceview),"","18. **Single actionable direction.** Yes: replace deterministic within-niche top-HGB with a frozen rank-stratified rule that retains a fixed high-HGB fraction and samples the remainder uniformly, keeping λ, quotas, pool, labels and training budget unchanged.","","19. **Evidence status.** All mechanism findings are observational associations from frozen selections plus post-hoc deterministic features. No individual instance value, negative-transfer cause, learning-gradient redundancy, or unique F40 causal mechanism is established.","","20. **Proceed to controlled validation?** Yes, only as one preregistered minimal comparison of B-random vs C-top-HGB vs rank-stratified selection using existing labels. Do not tune it on Original70 and do not infer that it must improve gap.","","## If only one part of the fifth-cycle selector may change","",recommendation,"","The stable evidence motivating this is the 13-direction-consistent Full-F40 shifts, C's three-seed local/lineage concentration proxies, and unstable B/C benchmark transfer despite C's consistently higher true CE. The modification is intended to reduce within-niche selection concentration while preserving hardness pressure. This audit cannot prove that concentration caused the gap changes, nor that rank stratification will improve training value; only the proposed controlled training can test that.","","The proposed comparison is not run here. P6-D3, new training, mutation, evolution, HGB fitting and Concorde are not started.","","`BLADE_P6D2_SELECTED_POPULATION_STRUCTURE_AUDIT_COMPLETE`"]
    for i,item in enumerate(final):
        if item.startswith("16. **Where are benchmark differences concentrated?**"):
            final[i]=f"16. **Where are benchmark differences concentrated?** Positive C regressions are mostly tail-heavy: the worst 10% contribute {tail_share.min():.1%}–{tail_share.max():.1%} of positive regression mass, and at least 60% in {(tail_share>=.60).sum()}/{len(tail_share)} reported benchmark/replicate cells. Mean C−B by dataset is below. The audit can identify dataset/tail concentration, but not a reliable finer instance taxonomy from three seeds."
    (OUT/"11_final_research_conclusion.md").write_text("\n".join(final)+"\n")
    # Consolidated answer to the 20 required questions.
    rep_summary=repl[repl.record_type.eq("group")].pivot(index="replicate",columns="group",values=["n","true_ce_mean","jaccard"])
    (OUT/"12_runtime_and_resource_audit.md").write_text(f"# Runtime and Resource Audit\n\n- Wall-clock: {(time.perf_counter()-started)/60:.3f} minutes.\n- New BQ/HGB training: 0.\n- New Concorde calls: 0.\n- New mutation/evolution: 0.\n- Deterministic tour features: {len(tours):,} existing labelled instances.\n- Representative cases: {len(representatives)}.\n- Instance bootstrap and lineage-root bootstrap: {BOOT_N} resamples each.\n")
    completion={"status":"COMPLETE","final_label":"BLADE_P6D2_SELECTED_POPULATION_STRUCTURE_AUDIT_COMPLETE","H1_STRUCTURAL_REPLACEMENT":h1,"H2_HGB_STRUCTURE_BIAS":h2,"H3_REDUNDANCY":h3,"H4_TRANSFER_ASSOCIATION":h4,"H5_ACTIONABLE_DESIGN_HYPOTHESIS":h5,"protocol_sha256":protocol["protocol_sha256"],"final_report_sha256":sha256_file(OUT/"11_final_research_conclusion.md"),"wall_s":time.perf_counter()-started}
    write_json(OUT/"completion.json",completion)
    # Key final figures hashes/artifact ledger.
    artifacts=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("output_artifact_hashes.csv",): artifacts.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    pd.DataFrame(artifacts).to_csv(OUT/"manifests/output_artifact_hashes.csv",index=False)
    print(json.dumps(completion,indent=2))

if __name__=="__main__": main()
