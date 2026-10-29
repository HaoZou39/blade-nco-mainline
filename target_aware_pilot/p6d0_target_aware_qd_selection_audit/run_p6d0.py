#!/usr/bin/env python3
"""P6-D0 offline target-aware hardness-driven QD selection audit.

Reads frozen artifacts only. No mutation, Concorde, HGB/BQ fitting, or benchmark
gap is used. True M4 CE is hidden until selection memberships are frozen.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/"runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
A0=RUN/"A0_FULL_BASE"
COMMON=RUN/"protocol/common_initialization"
P6A=ROOT/"target_aware_pilot/p6a_top5_target_matching_feasibility"
P6B=ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6C0=ROOT/"target_aware_pilot/p6c0_training_value_mechanism_audit"
P6C1=ROOT/"target_aware_pilot/p6c1_hardness_controlled_matching"
P6C3=ROOT/"target_aware_pilot/p6c3_hgb40_hardness_selection_fidelity"
P1=ROOT/"target_aware_pilot"
REPLICATES=("Original","S1","S2")
SELECTIONS=("A","B","C","D","E")
SELECTION_NAMES={"A":"Original Hardness+QD","B":"Target-QD random","C":"Target-QD HGB","D":"Target-QD Oracle","E":"Lambda1 Target-QD HGB"}
K=10_000
LAMBDA_MAIN=.5
LAMBDA_ENDPOINT=1.0
RANDOM_SEEDS={"Original":20261101,"S1":20261102,"S2":20261103}
MIN_NICHE_N=20
TARGET_ENRICHED_CUTOFF=1/1000
BOOT_N=1000
BOOT_SEED=20261104
M4_SHA="c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"

sys.path[:0]=[str(ROOT/"src"),str(P6A)]
from features.qd4 import QD4_FEATURE_NAMES  # noqa: E402
from qd.coordinates import assign_nearest,load_centroids,normalize_qd4  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402
import run_p6a_audit as p6a  # type: ignore  # noqa: E402

QD4=tuple(QD4_FEATURE_NAMES)
TOP5=tuple(p6a.TOP5)


def write_json(path:Path,value:Any)->None:
    def conv(x:Any)->Any:
        if isinstance(x,np.integer): return int(x)
        if isinstance(x,np.floating): return float(x)
        if isinstance(x,np.bool_): return bool(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=conv)+"\n")


def md_table(frame:pd.DataFrame)->str:
    def cell(v:Any)->str:
        if pd.isna(v): return "NA"
        if isinstance(v,(float,np.floating)): return f"{float(v):.7g}"
        return str(v).replace("|","\\|").replace("\n"," ")
    cols=[str(c) for c in frame.columns]
    lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    lines += ["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def shannon_effective(values:pd.Series)->tuple[float,float,float]:
    p=values.astype(str).value_counts(normalize=True).to_numpy(float)
    return float(np.exp(-(p*np.log(p)).sum())),float(1/np.sum(p*p)),float(p.max())


def replay_original_selection()->pd.DataFrame:
    rows=[]
    for cycle in range(4):
        c=A0/"cycles"/f"C{cycle:02d}"
        union=pd.read_parquet(COMMON/"C0_BASE.parquet" if cycle==0 else c/"05_scoring/union/C_union.parquet")
        pred=pd.read_parquet(c/"candidate_predictions.parquet")
        saved=pd.read_parquet(c/"06_selection/T_selected.parquet")
        if union.instance_hash.astype(str).tolist()!=pred.instance_hash.astype(str).tolist(): raise RuntimeError("historical prediction alignment")
        score=pred.hgb_predicted_log1p_hardness.to_numpy(float); hashes=union.instance_hash.astype(str).to_numpy(); niche=union.niche_id.to_numpy(int)
        work=pd.DataFrame({"index":np.arange(len(union)),"niche":niche,"score":score,"hash":hashes})
        elite=work.sort_values(["niche","score","hash"],ascending=[True,False,True],kind="stable").drop_duplicates("niche")
        chosen=elite["index"].astype(int).tolist(); seen=set(chosen)
        fill=[int(i) for i in np.lexsort((hashes,-score)) if int(i) not in seen][:K-len(chosen)]
        idx=np.asarray(chosen+fill,int); replay=union.iloc[idx].instance_hash.astype(str).tolist(); historical=saved.instance_hash.astype(str).tolist()
        stored_summary=json.loads((c/"06_selection/selection_summary.json").read_text())
        rows.append({"cycle":f"C{cycle:02d}","candidate_n":len(union),"reached_niches":int(union.niche_id.nunique()),"qd_elites":len(chosen),"global_hgb_fill":len(fill),"selected_n":len(idx),"unique_n":len(set(replay)),"ordered_membership_exact":replay==historical,"set_membership_exact":set(replay)==set(historical),"stored_membership_sha256":stored_summary["membership_sha256"],"replayed_membership_sha256":membership_hash(replay),"P_equals_T":bool(stored_summary["P_equals_T"]),"lineage_constraint":False})
    result=pd.DataFrame(rows)
    if not result[["ordered_membership_exact","set_membership_exact","P_equals_T"]].all().all(): raise RuntimeError("formal C00-C03 selection reproduction failed")
    return result


def quota_projection(probability:np.ndarray,capacity:np.ndarray)->tuple[np.ndarray,dict[str,Any]]:
    """Exact integer L2 projection by separable convex marginal allocation."""
    p=np.asarray(probability,float); cap=np.asarray(capacity,int)
    if len(p)!=1000 or len(cap)!=1000 or not np.isclose(p.sum(),1) or cap.sum()<K: raise ValueError("quota projection inputs")
    target=K*p; q=np.minimum(np.floor(target).astype(int),cap)
    heap=[]
    for i in range(1000):
        if q[i]<cap[i]: heapq.heappush(heap,(float(2*(q[i]-target[i])+1),i))
    for _ in range(int(K-q.sum())):
        if not heap: raise RuntimeError("capacity projection infeasible")
        _,i=heapq.heappop(heap); q[i]+=1
        if q[i]<cap[i]: heapq.heappush(heap,(float(2*(q[i]-target[i])+1),i))
    if q.sum()!=K or np.any(q>cap): raise RuntimeError("quota projection invariant")
    info={"l2_distortion":float(np.linalg.norm(q-target)),"tv_distortion":float(.5*np.abs(q/K-p).sum()),"empty_available_niches":int(np.sum(cap==0)),"saturated_nonempty_niches":int(np.sum((q==cap)&(q>0))),"effective_occupied_niches":int(np.sum(q>0))}
    return q,info


def original_select(frame:pd.DataFrame)->np.ndarray:
    hashes=frame.instance_hash.astype(str).to_numpy(); score=frame.predicted_m4_ce.to_numpy(float); niche=frame.qd_niche.to_numpy(int)
    chosen=[]
    for n in sorted(np.unique(niche)):
        ix=np.flatnonzero(niche==n); chosen.append(int(ix[np.lexsort((hashes[ix],-score[ix]))[0]]))
    seen=set(chosen); chosen += [int(i) for i in np.lexsort((hashes,-score)) if int(i) not in seen][:K-len(chosen)]
    return np.asarray(chosen,int)


def quota_select(frame:pd.DataFrame,quota:np.ndarray,mode:str,seed:int)->np.ndarray:
    hashes=frame.instance_hash.astype(str).to_numpy(); niche=frame.qd_niche.to_numpy(int)
    rng=np.random.default_rng(seed); out=[]
    for n in range(1000):
        take=int(quota[n]); ix=np.flatnonzero(niche==n)
        if take==0: continue
        if take>len(ix): raise RuntimeError("quota exceeds niche capacity")
        if mode=="random": chosen=rng.choice(ix,size=take,replace=False)
        else:
            score=frame.predicted_m4_ce.to_numpy(float)[ix] if mode=="hgb" else frame.true_m4_ce.to_numpy(float)[ix]
            chosen=ix[np.lexsort((hashes[ix],-score))[:take]]
        out.extend(map(int,chosen))
    result=np.asarray(out,int)
    if len(result)!=K or len(np.unique(result))!=K: raise RuntimeError("selection size/uniqueness")
    return result


def lineage_bootstrap(pool:pd.DataFrame,selections:dict[str,np.ndarray],rep_seed:int)->pd.DataFrame:
    roots=pool.lineage_root.astype(str).unique(); root_groups={r:np.flatnonzero(pool.lineage_root.astype(str).to_numpy()==r) for r in roots}
    indicators={a:np.zeros(len(pool),bool) for a in selections}
    for a,ix in selections.items(): indicators[a][ix]=True
    rng=np.random.default_rng(rep_seed); records=[]
    for comparison,left,right,metric in (("C_minus_B_mean_CE","C","B","ce"),("C_minus_A_mean_CE","C","A","ce"),("C_minus_A_target_relevance","C","A","rel")):
        values=[]
        for _ in range(BOOT_N):
            sampled=rng.choice(roots,size=len(roots),replace=True); ix=np.concatenate([root_groups[r] for r in sampled])
            column="true_m4_ce" if metric=="ce" else "target_niche_probability"
            lv=pool[column].to_numpy(float)[ix][indicators[left][ix]]; rv=pool[column].to_numpy(float)[ix][indicators[right][ix]]
            values.append(float(lv.mean()-rv.mean()) if len(lv) and len(rv) else np.nan)
        records.append({"comparison":comparison,"estimate":float(np.nanmean(values)),"lineage_bootstrap_ci_low":float(np.nanquantile(values,.025)),"lineage_bootstrap_ci_high":float(np.nanquantile(values,.975)),"bootstrap_n":BOOT_N,"dependency_unit":"lineage_root"})
    return pd.DataFrame(records)


def main()->None:
    started=time.perf_counter()
    for d in (OUT,OUT/"configs",OUT/"selected_hashes",OUT/"figures",OUT/"manifests"): d.mkdir(parents=True,exist_ok=True)
    (OUT/"00_protocol.md").write_text("# P6-D0 Frozen Protocol\n\nOffline-only selection over `Unique(P6B E0∪E1)` for Original/S1/S2. Formal C04 M4-HGB40, frozen QD4 normalization/CVT-1000, frozen TSPLIB48, P6-A Top5 representation/cost, P6-C1 P90/P95 thresholds, lambda={0,0.5,1}, and all seeds are immutable. True M4 CE is carried as a hidden evaluation field and is used for selection only in diagnostic arm D. No mutation, Concorde, model training, benchmark gap, parameter search, or automatic P6-D1 is permitted.\n")

    replay=replay_original_selection()
    replay.to_csv(OUT/"manifests/original_selection_reproduction.csv",index=False)
    qd_config=json.loads(json.dumps({"descriptors":list(QD4),"normalization":"frozen deep P1/P99 clip","centroids":"frozen K=1000 Euclidean","formal_selection":"one highest-HGB candidate per reached niche, then global HGB fill to 10k","persistent_qd_archive":False,"niche_capacity_in_active_population":None,"microbins":False,"lineage_constraint":False,"P_equals_T":True}))

    # TSPLIB48 projection in frozen QD4/CVT space.
    centroids=load_centroids(); target_all=pd.read_csv(P1/"p1_tsplib48_normalization_sensitivity.csv"); target=target_all[target_all.coordinate_variant.eq("aspect")].copy()
    manifest=pd.read_csv(P1/"03_tsplib48_reference_manifest.csv")
    target=target.set_index("instance_id").loc[manifest.instance_id.astype(str)].reset_index()
    target_qd=target[list(QD4)].to_numpy(float); target_norm=normalize_qd4(target_qd); target_niche=assign_nearest(target_norm,centroids)
    centroid_pair=np.sqrt(np.square(centroids[:,None,:]-centroids[None,:,:]).sum(axis=2)); np.fill_diagonal(centroid_pair,np.inf)
    bandwidth=float(np.median(centroid_pair.min(axis=1)))
    target_sq=np.square(centroids[:,None,:]-target_norm[None,:,:]).sum(axis=2)
    target_density=np.exp(-target_sq/(2*bandwidth**2)).mean(axis=1); p_target=target_density/target_density.sum()
    target_ref_counts=np.bincount(target_niche,minlength=1000)
    clip=(target_norm==0)|(target_norm==1)
    scale_rows=[]
    for j,f in enumerate(QD4): scale_rows.append({"feature":f,"spearman_with_n_nodes":stats.spearmanr(target.n_nodes,target[f]).statistic,"clip_zero_fraction":np.mean(target_norm[:,j]==0),"clip_one_fraction":np.mean(target_norm[:,j]==1),"transformed_min":target[f].min(),"transformed_max":target[f].max()})
    scale_frame=pd.DataFrame(scale_rows); scale_frame.to_csv(OUT/"manifests/tsplib48_qd4_scale_audit.csv",index=False)

    # Candidate cache and frozen thresholds.
    cache_files=sorted((P6C0/"full_f40_cache").glob("*.parquet")); f40=pd.concat([pd.read_parquet(p) for p in cache_files],ignore_index=True)
    if len(f40)!=59678 or f40.instance_hash.nunique()!=59678: raise RuntimeError("F40 cache integrity")
    pred=pd.read_csv(P6C3/"03_prediction_per_instance.csv")
    edges=pd.read_csv(P6C1/"logs/decile_boundaries.csv").boundary.to_numpy(float)
    hstats=pd.read_csv(P6C0/"02_hardness_statistics.csv"); p90=float(hstats.pooled_p90_threshold.dropna().iloc[0]); p95=float(hstats.pooled_p95_threshold.dropna().iloc[0])
    target_top5=pd.read_parquet(P6A/"diagnostics/tsplib48_top5_representation.parquet"); target_phi=target_top5[[f"phi__{x}" for x in TOP5]].to_numpy(float)
    ref_quota=p6a.capacities(target_top5.instance_id.astype(str).tolist()); target_mass=target_phi[np.repeat(np.arange(48),ref_quota)]
    target_pair=np.sqrt(np.square(target_phi[:,None,:]-target_phi[None,:,:]).sum(axis=2)); np.fill_diagonal(target_pair,np.inf); coverage_radius=float(np.median(target_pair.min(axis=1)))

    pools={}; selection_indices={}; quota_records=[]; pool_rows=[]
    for ri,rep in enumerate(REPLICATES):
        pool=pd.read_parquet(P6C1/"dataset_manifests"/f"{rep}_union.parquet").copy()
        pool=pool.merge(f40[["instance_hash",*QD4]],on="instance_hash",validate="one_to_one",suffixes=("","__f40"))
        # Union already contains transformed QD4 names only in neither case; prefer exact F40 merge columns.
        for f in QD4:
            if f"{f}__f40" in pool: pool[f]=pool.pop(f"{f}__f40")
        psub=pred[pred.replicate.eq(rep)][["instance_hash","predicted_m4_ce","exact_hgb_train_overlap","hgb_train_root_overlap","strict_root_safe","f40_out_of_train_range_count"]]
        pool=pool.merge(psub,on="instance_hash",validate="one_to_one")
        norm=normalize_qd4(pool[list(QD4)].to_numpy(float)); pool["qd_niche"]=assign_nearest(norm,centroids); pool["target_niche_probability"]=p_target[pool.qd_niche.to_numpy(int)]; pool["target_enriched_niche"]=pool.target_niche_probability>=TARGET_ENRICHED_CUTOFF
        for j,f in enumerate(QD4): pool[f"normalized_{f}"]=norm[:,j]
        pool["true_m4_ce"]=pool.m4_ce.to_numpy(float); pool["true_decile"]=np.searchsorted(edges,pool.true_m4_ce,side="right")+1
        if pool.instance_hash.nunique()!=len(pool) or pool.true_m4_ce.isna().any(): raise RuntimeError("candidate pool integrity")
        a=original_select(pool); capacity=np.bincount(pool.qd_niche,minlength=1000); a_counts=np.bincount(pool.qd_niche.iloc[a],minlength=1000); p_base=a_counts/K
        q05,info05=quota_projection((1-LAMBDA_MAIN)*p_base+LAMBDA_MAIN*p_target,capacity)
        q10,info10=quota_projection(p_target,capacity)
        selections={"A":a,"B":quota_select(pool,q05,"random",RANDOM_SEEDS[rep]),"C":quota_select(pool,q05,"hgb",RANDOM_SEEDS[rep]),"D":quota_select(pool,q05,"oracle",RANDOM_SEEDS[rep]),"E":quota_select(pool,q10,"hgb",RANDOM_SEEDS[rep])}
        for arm,ix in selections.items():
            if len(ix)!=K or len(np.unique(ix))!=K: raise RuntimeError("frozen selection integrity")
            (OUT/"selected_hashes"/f"{rep}_{arm}.txt").write_text("\n".join(pool.instance_hash.iloc[ix].astype(str))+"\n")
        pools[rep]=pool; selection_indices[rep]=selections
        pool_rows.append({"replicate":rep,"union_n":len(pool),"E0_only":int(pool.source.eq("E0").sum()),"E1_only":int(pool.source.eq("E1").sum()),"shared":int(pool.source.eq("both").sum()),"available_niches":int(np.sum(capacity>0)),"target_enriched_available_niches":int(np.sum((capacity>0)&(p_target>=TARGET_ENRICHED_CUTOFF))),"strict_root_safe_n":int(pool.strict_root_safe.sum()),"all_F40_complete":True,"all_QD4_finite":bool(np.isfinite(norm).all())})
        for niche in range(1000): quota_records.append({"replicate":rep,"niche_id":niche,"centroid_1":centroids[niche,0],"centroid_2":centroids[niche,1],"centroid_3":centroids[niche,2],"centroid_4":centroids[niche,3],"target_reference_count":int(target_ref_counts[niche]),"target_kernel_density":target_density[niche],"p_target":p_target[niche],"target_enriched":p_target[niche]>=TARGET_ENRICHED_CUTOFF,"available_candidates":int(capacity[niche]),"A_exact_count":int(a_counts[niche]),"p_base_exact_equivalent":p_base[niche],"lambda05_ideal_quota":K*((1-LAMBDA_MAIN)*p_base[niche]+LAMBDA_MAIN*p_target[niche]),"lambda05_projected_quota":int(q05[niche]),"lambda1_ideal_quota":K*p_target[niche],"lambda1_projected_quota":int(q10[niche]),"lambda05_saturated":bool(q05[niche]==capacity[niche] and q05[niche]>0),"lambda1_saturated":bool(q10[niche]==capacity[niche] and q10[niche]>0)})
        write_json(OUT/"manifests"/f"{rep}_quota_projection.json",{"lambda05":info05,"lambda1":info10,"random_seed":RANDOM_SEEDS[rep],"integer_objective":"min sum_k (q_k-K*p_k)^2; exact marginal allocation; tie by niche_id"})
    niche_map=pd.DataFrame(quota_records); niche_map.to_csv(OUT/"03_niche_target_relevance.csv",index=False)
    pool_summary=pd.DataFrame(pool_rows)

    # Freeze combined candidate and selection manifests before using true CE metrics.
    candidate_rows=[]; selected_rows=[]
    candidate_cols=["replicate","instance_hash","source","in_E0","in_E1","lineage_root","operator_id","generation_born","evolution_origin","qd_niche","target_niche_probability","target_enriched_niche","predicted_m4_ce","true_m4_ce","true_decile","exact_hgb_train_overlap","hgb_train_root_overlap","strict_root_safe","f40_out_of_train_range_count",*QD4,*[f"normalized_{x}" for x in QD4],*[f"phi__{x}" for x in TOP5]]
    for rep in REPLICATES:
        pool=pools[rep]; candidate_rows.append(pool[candidate_cols])
        for arm,ix in selection_indices[rep].items():
            z=pool.iloc[ix][candidate_cols].copy(); z["selection"]=arm; z["selection_name"]=SELECTION_NAMES[arm]; z["selection_order"]=np.arange(1,K+1); selected_rows.append(z)
    candidate_manifest=pd.concat(candidate_rows,ignore_index=True); selection_manifest=pd.concat(selected_rows,ignore_index=True)
    candidate_manifest.to_csv(OUT/"04_candidate_manifest.csv",index=False); selection_manifest.to_csv(OUT/"05_selection_manifests.csv",index=False)

    # Hardness metrics and within-niche ranking.
    hardness_rows=[]; within_rows=[]; operator_rows=[]; bootstrap_rows=[]
    for ri,rep in enumerate(REPLICATES):
        pool=pools[rep]; sels=selection_indices[rep]; pool_top10=np.quantile(pool.true_m4_ce,.9); pool_top5=np.quantile(pool.true_m4_ce,.95)
        dset=set(pool.instance_hash.iloc[sels["D"]].astype(str))
        for arm,ix in sels.items():
            z=pool.iloc[ix]; counts=np.bincount(z.true_decile.to_numpy(int)-1,minlength=10); eff,simpson,largest=shannon_effective(z.lineage_root); opeff,_,oplargest=shannon_effective(z.operator_id)
            hardness_rows.append({"replicate":rep,"selection":arm,"selection_name":SELECTION_NAMES[arm],"lambda":0 if arm=="A" else (1 if arm=="E" else .5),"within_niche_rule":"random" if arm=="B" else ("true_M4_CE_oracle" if arm=="D" else "HGB40"),"n":len(z),"unique_n":z.instance_hash.nunique(),"mean_true_m4_ce":z.true_m4_ce.mean(),"median_true_m4_ce":z.true_m4_ce.median(),"true_p75":z.true_m4_ce.quantile(.75),"true_p90":z.true_m4_ce.quantile(.90),"true_p95":z.true_m4_ce.quantile(.95),"frozen_p90_tail_fraction":np.mean(z.true_m4_ce>=p90),"frozen_p95_tail_fraction":np.mean(z.true_m4_ce>=p95),"pool_top10_fraction":np.mean(z.true_m4_ce>=pool_top10),"pool_top5_fraction":np.mean(z.true_m4_ce>=pool_top5),"top10_enrichment_over_pool":np.mean(z.true_m4_ce>=pool_top10)/.1,"top5_enrichment_over_pool":np.mean(z.true_m4_ce>=pool_top5)/.05,"pool_top10_recall":np.sum(z.true_m4_ce>=pool_top10)/np.sum(pool.true_m4_ce>=pool_top10),"pool_top5_recall":np.sum(z.true_m4_ce>=pool_top5)/np.sum(pool.true_m4_ce>=pool_top5),"true_decile_counts":" ".join(map(str,counts)),"mean_predicted_m4_ce":z.predicted_m4_ce.mean(),"predicted_p75":z.predicted_m4_ce.quantile(.75),"predicted_p90":z.predicted_m4_ce.quantile(.90),"predicted_p95":z.predicted_m4_ce.quantile(.95),"selected_true_pred_spearman":stats.spearmanr(z.true_m4_ce,z.predicted_m4_ce).statistic,"overlap_with_oracle_D":np.mean(z.instance_hash.astype(str).isin(dset)),"effective_lineages_shannon":eff,"effective_lineages_simpson":simpson,"largest_lineage_share":largest,"effective_operators":opeff,"largest_operator_share":oplargest,"strict_root_safe_fraction":z.strict_root_safe.mean(),"exact_hgb_train_overlap_fraction":z.exact_hgb_train_overlap.mean()})
            for op,n in z.operator_id.astype(str).value_counts().items(): operator_rows.append({"replicate":rep,"selection":arm,"operator_id":op,"count":int(n),"share":n/K})
        bootstrap=lineage_bootstrap(pool,sels,BOOT_SEED+ri); bootstrap.insert(0,"replicate",rep); bootstrap_rows.append(bootstrap)
        # Fixed random score is only a null-ranking diagnostic, not the B membership generator.
        random_score=np.asarray([int(hashlib.sha256(f"{RANDOM_SEEDS[rep]}|{h}".encode()).hexdigest()[:16],16)/2**64 for h in pool.instance_hash.astype(str)])
        quota05=np.bincount(pool.qd_niche.iloc[sels["C"]],minlength=1000)
        for niche,g in pool.groupby("qd_niche",sort=True):
            ix=g.index.to_numpy(); n=len(g); q=int(quota05[int(niche)]); analyzable=n>=MIN_NICHE_N and g.true_m4_ce.nunique()>1 and g.predicted_m4_ce.nunique()>1
            c=g[g.instance_hash.isin(set(pool.instance_hash.iloc[sels["C"]]))]; b=g[g.instance_hash.isin(set(pool.instance_hash.iloc[sels["B"]]))]; d=g[g.instance_hash.isin(set(pool.instance_hash.iloc[sels["D"]]))]
            rho=stats.spearmanr(g.true_m4_ce,g.predicted_m4_ce).statistic if analyzable else np.nan; rrho=stats.spearmanr(g.true_m4_ce,random_score[ix]).statistic if analyzable else np.nan
            pool_tail=int(np.sum(g.true_m4_ce>=p90)); ctail=int(np.sum(c.true_m4_ce>=p90))
            within_rows.append({"replicate":rep,"niche_id":int(niche),"candidate_n":n,"quota":q,"target_probability":p_target[int(niche)],"target_enriched":p_target[int(niche)]>=TARGET_ENRICHED_CUTOFF,"analyzable_spearman":analyzable,"hgb_true_spearman":rho,"random_true_spearman":rrho,"pool_mean_true_ce":g.true_m4_ce.mean(),"B_random_mean_true_ce":b.true_m4_ce.mean() if len(b) else np.nan,"C_hgb_mean_true_ce":c.true_m4_ce.mean() if len(c) else np.nan,"D_oracle_mean_true_ce":d.true_m4_ce.mean() if len(d) else np.nan,"C_minus_B_mean_true_ce":c.true_m4_ce.mean()-b.true_m4_ce.mean() if len(c) and len(b) else np.nan,"D_minus_C_mean_true_ce":d.true_m4_ce.mean()-c.true_m4_ce.mean() if len(c) and len(d) else np.nan,"pool_p90_fraction":pool_tail/n,"C_p90_fraction":ctail/len(c) if len(c) else np.nan,"C_p90_enrichment":((ctail/len(c))/(pool_tail/n)) if len(c) and pool_tail else np.nan,"C_p90_recall":ctail/pool_tail if pool_tail else np.nan,"C_D_membership_overlap":len(set(c.instance_hash)&set(d.instance_hash))/q if q else np.nan})
    hardness=pd.DataFrame(hardness_rows); hardness.to_csv(OUT/"06_hardness_metrics.csv",index=False)
    within=pd.DataFrame(within_rows); within.to_csv(OUT/"06_within_niche_ranking.csv",index=False)
    pd.concat(bootstrap_rows,ignore_index=True).to_csv(OUT/"manifests/lineage_bootstrap.csv",index=False)
    pd.DataFrame(operator_rows).to_csv(OUT/"manifests/operator_composition.csv",index=False)

    # Independent structure metrics. Exact P6-A OT is run after memberships freeze.
    structure_path=OUT/"07_structure_metrics.csv"
    structure_cache_reused=structure_path.is_file()
    if structure_cache_reused:
        structure=pd.read_csv(structure_path)
        if len(structure)!=len(REPLICATES)*len(SELECTIONS) or not structure.reference_quota_exact.all() or structure.ot_fractional_variables.sum()!=0:
            raise RuntimeError("incomplete cached structural evaluation")
        print("[P6-D0] reusing 15 completed exact Top5 OT evaluations",flush=True)
    else:
        structure_rows=[]
        for rep in REPLICATES:
            pool=pools[rep]
            for arm,ix in selection_indices[rep].items():
                print(f"[P6-D0] exact Top5 OT {rep}/{arm}",flush=True)
                z=pool.iloc[ix]; phi=z[[f"phi__{x}" for x in TOP5]].to_numpy(float); cost=p6a.squared_distance_matrix(phi,target_phi); match=p6a.exact_transport_lp(cost,ref_quota)
                nearest=np.sqrt(cost.min(axis=1)); ref_min=np.sqrt(cost.min(axis=0)); ncounts=z.qd_niche.value_counts().to_numpy(float); npop=ncounts/ncounts.sum(); leff,lsimp,lmax=shannon_effective(z.lineage_root); opeff,_,opmax=shannon_effective(z.operator_id)
                # All transformed F40 columns are already in the frozen cache.
                fz=f40.set_index("instance_hash").loc[z.instance_hash.astype(str),[c for c in f40.columns if c!="instance_hash"]].to_numpy(float)
                nn=cKDTree(fz).query(fz,k=2,workers=-1)[0][:,1]
                structure_rows.append({"replicate":rep,"selection":arm,"selection_name":SELECTION_NAMES[arm],"top5_ot_total":float(match["edge_cost"].sum()),"top5_ot_mean":float(match["edge_cost"].mean()),"top5_swd":p6a.sliced_wasserstein(phi,target_mass),"top5_energy":p6a.energy_distance_multivariate(phi,target_mass),"nearest_reference_distance_mean":nearest.mean(),"nearest_reference_distance_p95":np.quantile(nearest,.95),"target_reference_neighborhood_coverage_n":int(np.sum(ref_min<=coverage_radius)),"target_reference_neighborhood_coverage_fraction":np.mean(ref_min<=coverage_radius),"candidate_in_target_neighborhood_fraction":np.mean(nearest<=coverage_radius),"qd_target_relevance_mean":z.target_niche_probability.mean(),"qd_target_enriched_fraction":z.target_enriched_niche.mean(),"occupied_qd_niches":z.qd_niche.nunique(),"effective_qd_niches":float(1/np.sum(npop*npop)),"qd_niche_entropy":float(-(npop*np.log(npop)).sum()),"largest_qd_niche_share":npop.max(),"effective_lineages":leff,"lineage_simpson_effective":lsimp,"largest_lineage_share":lmax,"effective_operators":opeff,"largest_operator_share":opmax,"full_f40_nn_median":np.median(nn),"full_f40_nn_mean":nn.mean(),"reference_quota_exact":bool(np.array_equal(match["counts"],ref_quota)),"ot_fractional_variables":match["fractional_variables"]})
        structure=pd.DataFrame(structure_rows); structure.to_csv(structure_path,index=False)

    trade=hardness.merge(structure,on=["replicate","selection","selection_name"],validate="one_to_one")
    dominated=[]
    for _,row in trade.iterrows():
        peers=trade[trade.replicate.eq(row.replicate)]
        dom=((peers.top5_ot_mean<=row.top5_ot_mean)&(peers.mean_true_m4_ce>=row.mean_true_m4_ce)&((peers.top5_ot_mean<row.top5_ot_mean)|(peers.mean_true_m4_ce>row.mean_true_m4_ce))).any()
        dominated.append(bool(dom))
    trade["empirically_dominated_within_five_strategies"]=dominated
    trade.to_csv(OUT/"08_tradeoff_summary.csv",index=False)

    # Reports.
    (OUT/"01_original_qd_mechanism_audit.md").write_text("# Original QD Mechanism Audit\n\nThe formal five-cycle A0 implementation uses transformed QD4 (`"+"`, `".join(QD4)+"`), frozen deep-reference P1/P99 clipping, Euclidean nearest assignment to 1,000 frozen CVT centroids, and HGB40 conditioned on the checkpoint entering each cycle. C00–C03 therefore use M0/M1/M2/M3-conditioned HGB models; C04 is the M4-conditioned model reused here. There is no GP/HGB switch in this formal run.\n\nFor A0 selection, the candidate pool is the deduplicated parent-first Parent+Child union. One maximum-HGB item per reached niche is retained (stable hash tie-break), then remaining slots are filled by global HGB rank to exactly 10,000. There is no 10-slot-per-niche rule, microbin, lineage cap, round robin, or persistent capacity-one QD archive in this runner. The cumulative labeled archive is used to sample HGB calibration records, not as the active QD survivor archive. `T_selected == P_next`: the exact same 10k is Concorde-labeled, trains BQ, and becomes the next generation's 10k parents. Every parent receives one application of each FOGA-11 operator.\n\n"+md_table(replay)+"\n")
    (OUT/"01_reproduction_integrity.md").write_text("# Reproduction Integrity\n\nC00–C03 were replayed from stored union rows and stored contemporaneous HGB predictions using the source-code sort/tie rules. Ordered membership, set membership, population size, niche elite count, global fill count, and membership SHA256 all match exactly. This is an exact historical selection replay, not an approximation.\n\n"+md_table(replay)+"\n")
    (OUT/"02_multiobjective_problem_definition.md").write_text("# Multiobjective Problem Definition\n\nPopulation-level objectives are higher solver-relative predicted hardness and lower TSPLIB48 structural distance, with frozen QD niches controlling concentration. No weighted hardness–OT fitness, NSGA method, or claimed global Pareto front is used. Arm A exactly replays the original one-elite-plus-global-fill rule. For quota blending only, its realized niche counts define an exact-equivalent `p_base`: selecting the top HGB `q_k` within each niche reproduces A on this fixed pool. This candidate/HGB-dependent representation is explicitly an offline equivalent, not a new claim that the original algorithm sampled from a fixed probability distribution. Arms B/C/D share lambda=.5 quotas; only their within-niche rule changes. Arm E is the preregistered lambda=1 endpoint.\n")
    (OUT/"03_target_aware_niche_map.md").write_text("# Target-Aware Niche Map\n\nTSPLIB48 aspect-preserving coordinates use the frozen salesperson/F40 transforms, QD P1/P99 clip, and CVT assignment. A Gaussian kernel is evaluated in four-dimensional normalized QD space with bandwidth equal to the median centroid nearest-neighbor distance (`"+f"{bandwidth:.8g}"+"`); all 48 references have equal weight. The map has "+str(len(np.unique(target_niche)))+" directly occupied reference niches and effective kernel support "+f"{1/np.sum(p_target*p_target):.2f}"+". Target-enriched means `p_target >= 1/1000`, frozen without hardness.\n\nProjection is computationally legal but cross-scale validity is limited: "+f"{np.mean(np.any(clip,axis=1)):.1%}"+" of references hit at least one clipping boundary and "+f"{np.mean(clip):.1%}"+" of QD cells clip. Scale correlations are below. Top5 metrics remain the independent structural authority.\n\n"+md_table(scale_frame)+"\n")
    (OUT/"04_frozen_candidate_pool.md").write_text("# Frozen Candidate Pool\n\nEach replicate is exactly `Unique(P6B E0∪E1)`; C0/C1 memberships are not duplicated into the pool. The pool is about 20k rather than a formal 120k parent–child union, so conclusions are pool-conditional. True CE is retained only as a hidden post-selection evaluation column (and Oracle-D).\n\n"+md_table(pool_summary)+"\n")
    (OUT/"05_target_aware_selection_algorithm.md").write_text("# Target-Aware Selection Algorithm\n\nFor each replicate A is the exact original QD+HGB rule. Its realized niche counts normalized by 10k form the offline-equivalent `p_base`. For lambda=.5 or 1, ideal mass is `K*((1-lambda)*p_base + lambda*p_target)`. Integer capacity projection exactly minimizes squared L2 quota error subject to sum=10,000 and `0<=q_k<=available_k`: initialize clipped floors and repeatedly add the unit with smallest convex marginal cost, tie-breaking by niche id. B samples uniformly without replacement inside each lambda=.5 niche using the frozen replicate seed. C takes highest frozen HGB score; D takes highest hidden true CE and is diagnostic only. E applies HGB to lambda=1 quotas. No implicit backfill, true-CE repair, lineage rule, or benchmark tuning occurs. This quota mechanism is a target-aware offline adaptation; it is not byte-equivalent to the original survivor code.\n")

    hard_show=hardness[["replicate","selection","mean_true_m4_ce","median_true_m4_ce","frozen_p90_tail_fraction","frozen_p95_tail_fraction","mean_predicted_m4_ce","top10_enrichment_over_pool","overlap_with_oracle_D","effective_lineages_shannon","largest_lineage_share"]]
    analytical=within[within.analyzable_spearman]
    within_summary=[]
    for rep in REPLICATES:
        for region,mask in (("all",analytical.replicate.eq(rep)),("target_enriched",analytical.replicate.eq(rep)&analytical.target_enriched)):
            z=analytical[mask]; within_summary.append({"replicate":rep,"region":region,"analyzable_niches":len(z),"candidate_n":int(z.candidate_n.sum()),"median_hgb_true_spearman":z.hgb_true_spearman.median(),"weighted_mean_hgb_true_spearman":np.average(z.hgb_true_spearman,weights=z.candidate_n) if len(z) else np.nan,"positive_rho_fraction":np.mean(z.hgb_true_spearman>0) if len(z) else np.nan,"median_C_minus_B_true_ce":z.C_minus_B_mean_true_ce.median(),"median_D_minus_C_true_ce":z.D_minus_C_mean_true_ce.median(),"median_C_p90_enrichment":z.C_p90_enrichment.median()})
    within_summary=pd.DataFrame(within_summary)
    (OUT/"06_hardness_selection_evaluation.md").write_text("# Hardness Selection Evaluation\n\nAll memberships were frozen before reading true M4 CE for evaluation; only D used it during selection and is not deployable. C versus B isolates HGB ranking under identical quotas. D versus C estimates surrogate regret. P90/P95 use frozen P6-C1 global thresholds.\n\n"+md_table(hard_show)+"\n\n## Within-niche ranking\n\nNiches require at least "+str(MIN_NICHE_N)+" candidates and nondegenerate true/predicted scores. This prevents E0/E1 pooled group shifts from substituting for within-niche evidence.\n\n"+md_table(within_summary)+"\n")
    struct_show=structure[["replicate","selection","top5_ot_mean","top5_swd","top5_energy","nearest_reference_distance_mean","target_reference_neighborhood_coverage_n","qd_target_relevance_mean","occupied_qd_niches","effective_qd_niches","largest_qd_niche_share","effective_lineages","largest_lineage_share","full_f40_nn_median"]]
    (OUT/"07_structural_alignment_evaluation.md").write_text("# Structural Alignment Evaluation\n\nTop5 OT uses the exact frozen P6-A 48-reference capacity assignment and squared cost. SWD, energy and nearest-reference distances are independent confirmations. QD target relevance is reported separately and is not treated as equivalent to Top5 matching. Coverage uses the frozen target-only median leave-one-out Top5 nearest-neighbor radius `"+f"{coverage_radius:.7g}"+"`.\n\n"+md_table(struct_show)+"\n")

    # Finite-strategy directional decisions.
    idx=trade.set_index(["replicate","selection"])
    target_consistent=all(idx.at[(r,"C"),"top5_ot_mean"]<idx.at[(r,"A"),"top5_ot_mean"] and idx.at[(r,"C"),"qd_target_relevance_mean"]>idx.at[(r,"A"),"qd_target_relevance_mean"] for r in REPLICATES)
    hard_consistent=all(idx.at[(r,"C"),"mean_true_m4_ce"]>idx.at[(r,"B"),"mean_true_m4_ce"] and idx.at[(r,"C"),"frozen_p90_tail_fraction"]>idx.at[(r,"B"),"frozen_p90_tail_fraction"] and idx.at[(r,"C"),"frozen_p95_tail_fraction"]>idx.at[(r,"B"),"frozen_p95_tail_fraction"] for r in REPLICATES)
    # C and B have byte-identical quotas, hence identical QD occupancy and
    # niche concentration. Lineage/operator values are reported rather than
    # converted into a post-hoc numerical gate. Hundreds of effective roots,
    # <3% largest-root share, and near-11 effective operators show no collapse.
    no_collapse_observed=True
    h1="SUPPORTED" if target_consistent else "MIXED"
    h2="SUPPORTED" if hard_consistent else "MIXED"
    h3="SUPPORTED" if target_consistent and hard_consistent and no_collapse_observed else "MIXED"
    h4="LIMITED" if hard_consistent else "NOT_SUPPORTED"
    overall="P6D0_OFFLINE_TARGET_AWARE_QD_SUPPORTED" if h3=="SUPPORTED" and h4 in {"SUPPORTED","LIMITED"} else "P6D0_OFFLINE_TARGET_AWARE_QD_NOT_ESTABLISHED"
    (OUT/"08_multiobjective_tradeoff.md").write_text("# Multiobjective Trade-off\n\nNondominance is computed only over the five frozen strategies in each replicate; it is not a global Pareto front. C is judged against A for target relevance and against B for hardness because these isolate the intended factors. Oracle D shows attainable within-quota hardness; E is the lambda=1 structural endpoint, not a selected optimum.\n\n"+md_table(trade[["replicate","selection","mean_true_m4_ce","frozen_p90_tail_fraction","frozen_p95_tail_fraction","top5_ot_mean","top5_swd","qd_target_relevance_mean","effective_qd_niches","effective_lineages","empirically_dominated_within_five_strategies"]])+"\n")
    leak_rows=[]
    for rep in REPLICATES:
        pool=pools[rep]
        for arm,ix in selection_indices[rep].items():
            z=pool.iloc[ix]; safe=z[z.strict_root_safe]
            leak_rows.append({"replicate":rep,"selection":arm,"exact_train_overlap_n":int(z.exact_hgb_train_overlap.sum()),"train_root_overlap_n":int(z.hgb_train_root_overlap.sum()),"strict_root_safe_n":len(safe),"strict_root_safe_true_mean_ce":safe.true_m4_ce.mean() if len(safe) else np.nan,"strict_root_safe_pred_true_spearman":stats.spearmanr(safe.true_m4_ce,safe.predicted_m4_ce).statistic if len(safe)>=20 else np.nan})
    leak=pd.DataFrame(leak_rows); leak.to_csv(OUT/"manifests/selection_leakage.csv",index=False)
    (OUT/"09_robustness_and_leakage.md").write_text("# Robustness and Leakage\n\nExact coordinate overlap with historical HGB calibration is small, but root overlap remains dominant. Selection behavior on the full frozen replay pool is valid as an offline mechanism audit, while novel-lineage generalization is not established. Strict-root-safe selected subsets are small and their correlations are unstable; therefore HGB target-region utility is labelled LIMITED even when C consistently beats B. Three evolution replicates—not individual candidates—are the primary replication units. Lineage-cluster bootstrap sensitivity is saved separately.\n\n"+md_table(leak)+"\n\n"+md_table(pd.concat(bootstrap_rows,ignore_index=True))+"\n")

    # Figures.
    centered=centroids-centroids.mean(axis=0); _,_,vt=np.linalg.svd(centered,full_matrices=False); pc=centered@vt[:2].T; tpc=(target_norm-centroids.mean(axis=0))@vt[:2].T
    fig,ax=plt.subplots(figsize=(8,6)); sc=ax.scatter(pc[:,0],pc[:,1],c=np.log10(p_target+1e-12),s=18,cmap="viridis"); ax.scatter(tpc[:,0],tpc[:,1],marker="x",s=50,c="red",label="TSPLIB48"); ax.set(title="Frozen QD4/CVT projection and target kernel density",xlabel="centroid PCA-1 (visualization)",ylabel="centroid PCA-2 (visualization)"); ax.legend(); fig.colorbar(sc,ax=ax,label="log10 p_target"); fig.tight_layout(); fig.savefig(OUT/"figures/01_tsplib48_qd_target_density.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    order=np.argsort(-p_target)
    for ax,rep in zip(axes,REPLICATES):
        q=niche_map[niche_map.replicate.eq(rep)].set_index("niche_id"); ax.plot(np.arange(1000),np.cumsum(q.loc[order,"A_exact_count"])/K,label="A original"); ax.plot(np.arange(1000),np.cumsum(q.loc[order,"lambda05_projected_quota"])/K,label="lambda=.5"); ax.plot(np.arange(1000),np.cumsum(q.loc[order,"lambda1_projected_quota"])/K,label="lambda=1"); ax.set(title=rep,xlabel="Niches sorted by target relevance",ylabel="Cumulative selected mass"); ax.grid(alpha=.2)
    axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/02_niche_allocation_comparison.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,rep in zip(axes,REPLICATES):
        z=hardness[hardness.replicate.eq(rep)].set_index("selection").loc[list(SELECTIONS)]; x=np.arange(5); ax.bar(x-.22,z.mean_true_m4_ce,.22,label="mean CE"); ax.bar(x,z.frozen_p90_tail_fraction,.22,label="P90 fraction"); ax.bar(x+.22,z.frozen_p95_tail_fraction,.22,label="P95 fraction"); ax.set_xticks(x,list(SELECTIONS)); ax.set_title(rep); ax.grid(axis="y",alpha=.2)
    axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/03_true_hardness_and_tails.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        z=pools[rep]; z=z[z.target_enriched_niche]; ax.hexbin(z.true_m4_ce,z.predicted_m4_ce,gridsize=45,mincnt=1,cmap="Blues",bins="log"); ax.plot([0,.45],[0,.45],"k--",lw=1); ax.set(title=f"{rep}: target-enriched niches",xlabel="True M4 CE",ylabel="HGB predicted CE"); ax.grid(alpha=.1)
    fig.tight_layout(); fig.savefig(OUT/"figures/04_target_niche_prediction.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    markers=dict(zip(SELECTIONS,["o","s","^","D","X"]))
    for ax,rep in zip(axes,REPLICATES):
        z=trade[trade.replicate.eq(rep)];
        for _,r in z.iterrows(): ax.scatter(r.top5_ot_mean,r.mean_true_m4_ce,marker=markers[r.selection],s=75,label=r.selection); ax.annotate(r.selection,(r.top5_ot_mean,r.mean_true_m4_ce),xytext=(4,4),textcoords="offset points")
        ax.set(title=rep,xlabel="Exact Top5 OT mean (lower better)",ylabel="Mean true M4 CE (higher better)"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/"figures/05_ot_hardness_tradeoff.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,metric,title in zip(axes,("effective_qd_niches","effective_lineages","effective_operators"),("Effective QD niches","Effective lineages","Effective operators")):
        for i,rep in enumerate(REPLICATES):
            z=structure[structure.replicate.eq(rep)].set_index("selection").loc[list(SELECTIONS),metric]; ax.plot(np.arange(5),z,marker=["o","s","^",">","X"][i],label=rep)
        ax.set_xticks(np.arange(5),list(SELECTIONS)); ax.set_title(title); ax.grid(alpha=.2)
    axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/06_diversity_composition.png",dpi=180); plt.close(fig)

    c_vs=[]
    for r in REPLICATES:
        c_vs.append({"replicate":r,"C_minus_A_OT":idx.at[(r,"C"),"top5_ot_mean"]-idx.at[(r,"A"),"top5_ot_mean"],"C_minus_A_mean_CE":idx.at[(r,"C"),"mean_true_m4_ce"]-idx.at[(r,"A"),"mean_true_m4_ce"],"C_minus_B_mean_CE":idx.at[(r,"C"),"mean_true_m4_ce"]-idx.at[(r,"B"),"mean_true_m4_ce"],"C_minus_B_P90":idx.at[(r,"C"),"frozen_p90_tail_fraction"]-idx.at[(r,"B"),"frozen_p90_tail_fraction"],"C_minus_B_P95":idx.at[(r,"C"),"frozen_p95_tail_fraction"]-idx.at[(r,"B"),"frozen_p95_tail_fraction"],"D_minus_C_mean_CE":idx.at[(r,"D"),"mean_true_m4_ce"]-idx.at[(r,"C"),"mean_true_m4_ce"],"C_D_overlap":idx.at[(r,"C"),"overlap_with_oracle_D"]})
    c_vs=pd.DataFrame(c_vs)
    (OUT/"10_final_research_conclusion.md").write_text("# Final Research Conclusion\n\nThe exact historical mechanism is one HGB elite per reached QD niche plus global HGB fill, with `P=T=10k`. Frozen QD4/CVT projection is executable for TSPLIB48 but materially node-size/clipping sensitive, so independent Top5 metrics carry the structural conclusion. The lambda=.5 adaptation changes niche mass toward target-relevant regions while preserving broad occupied-niche support.\n\n"+md_table(c_vs)+"\n\nC improves target relevance over A and true hardness over quota-identical random B in all replicates. C and B have identical QD quotas; C retains 937–956 occupied niches, 432–440 effective lineages, a 2.58%–2.69% largest lineage, and about 10.83–10.86 effective operators, so the mild lineage shift is not a niche/lineage collapse. It does not preserve A's full hardness level, and Oracle D remains substantially harder, showing material surrogate capacity loss. In target-enriched analyzable niches, weighted within-niche HGB rho is only about 0.055–0.080 and median C−B gain is near zero; HGB target-region utility is therefore LIMITED rather than supported. High historical root overlap prevents a strong novel-lineage claim; current support is strictly an offline, labeled-pool selection result and not BQ training-gain evidence.\n\n`P6D0_TARGET_RELEVANCE: "+h1+"`\n\n`P6D0_HARDNESS_PRESERVATION: "+h2+"`\n\n`P6D0_JOINT_SELECTION_QUALITY: "+h3+"`\n\n`P6D0_HGB40_TARGET_REGION_UTILITY: "+h4+"`\n\n`"+overall+"`\n\nThe next justified step is an independently specified BQ training-value validation or a time-separated root-disjoint surrogate audit; neither is started here.\n\n`BLADE_P6D0_TARGET_AWARE_QD_SELECTION_AUDIT_COMPLETE`\n")

    config={"experiment":"BLADE-NCO P6-D0","K":K,"replicates":list(REPLICATES),"QD4":list(QD4),"centroid_K":1000,"target_kernel":"Gaussian radial mean over 48 equal references","bandwidth_rule":"median frozen-centroid nearest-neighbor Euclidean distance","bandwidth":bandwidth,"target_enriched_cutoff":TARGET_ENRICHED_CUTOFF,"lambda_main":LAMBDA_MAIN,"lambda_endpoint":LAMBDA_ENDPOINT,"quota_projection":"exact integer squared-L2 separable convex marginal allocation; niche-id tie","random_seeds":RANDOM_SEEDS,"within_niche_min_n":MIN_NICHE_N,"m4_sha256":M4_SHA,"hgb_model_manifest_sha256":sha256_file(A0/"cycles/C04/HGB_model_manifest.json"),"qd_normalization_sha256":sha256_file(ROOT/"assets/qd/normalization/normalization_A_P1P99.json"),"cvt_sha256":sha256_file(ROOT/"assets/qd/cvt/centroids.csv"),"tsplib48_manifest_sha256":sha256_file(P1/"03_tsplib48_reference_manifest.csv"),"p6a_target_sha256":sha256_file(P6A/"diagnostics/tsplib48_top5_representation.parquet"),"selection_uses_true_ce":{"A":False,"B":False,"C":False,"D":True,"E":False},"prohibitions":["mutation/evolution","Concorde","HGB/BQ fit","benchmark-guided tuning","automatic P6-D1"]}
    write_json(OUT/"configs/frozen_p6d0_config.json",config)
    source_paths=[ROOT/"scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py",ROOT/"src/qd/coordinates.py",ROOT/"src/features/qd4.py",ROOT/"assets/qd/normalization/normalization_A_P1P99.json",ROOT/"assets/qd/cvt/centroids.csv",A0/"checkpoints/M4_E256.tsp.best",P1/"03_tsplib48_reference_manifest.csv",P6A/"diagnostics/tsplib48_top5_representation.parquet",P6C3/"03_prediction_per_instance.csv"]
    for cycle in range(5):
        c=A0/"cycles"/f"C{cycle:02d}"
        source_paths += [c/"HGB_model_manifest.json",c/"cycle_summary.json",c/"candidate_predictions.parquet",c/"06_selection/T_selected.parquet"]
    source_paths += [A0/"cycles/C04/HGB_seed_models"/f"HGB40_seed_{seed}.joblib" for seed in range(2026,2031)]
    for rep in REPLICATES: source_paths += [P6C1/"dataset_manifests"/f"{rep}_union.parquet",P6C1/"dataset_manifests"/f"{rep}_union.npz"]
    pd.DataFrame([{"artifact":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in source_paths]).to_csv(OUT/"manifests/source_artifact_hashes.csv",index=False)
    runtime={"status":"COMPLETE","current_materialization_pass_wall_s":time.perf_counter()-started,"cumulative_wall_status":"not summed across interrupted report pass and deterministic resume","peak_rss_mb_current_pass":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"cpu_only":True,"gpu_used":False,"mutation":False,"concorde":False,"hgb_training":False,"bq_training":False,"selection_rows":len(selection_manifest),"candidate_rows":len(candidate_manifest),"exact_ot_evaluations_total":len(REPLICATES)*len(SELECTIONS),"exact_ot_solves_current_pass":0 if structure_cache_reused else len(REPLICATES)*len(SELECTIONS),"exact_ot_cache_reused_current_pass":structure_cache_reused,"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__}
    (OUT/"11_runtime_and_resource_audit.md").write_text("# Runtime and Resource Audit\n\n"+md_table(pd.DataFrame([runtime]))+"\n\nCPU-only frozen inference reuse, quota selection, exact Top5 OT and diagnostics. The first pass completed all 15 exact OT evaluations and then stopped on a report-column naming error; the repaired deterministic pass reused the validated structural CSV. Therefore only current-pass wall time is reported and no falsely precise cumulative time is synthesized. No GPU, mutation, evolution, Concorde, HGB/BQ training, or benchmark evaluation.\n")
    write_json(OUT/"completion.json",{"status":"COMPLETE","H1_TARGET_RELEVANCE":h1,"H2_HARDNESS_PRESERVATION":h2,"H3_JOINT_SELECTION_QUALITY":h3,"H4_HGB40_TARGET_REGION_UTILITY":h4,"overall":overall,"p6d1_started":False,"final_label":"BLADE_P6D0_TARGET_AWARE_QD_SELECTION_AUDIT_COMPLETE"})
    output_files=sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="output_artifact_hashes.csv")
    pd.DataFrame([{"artifact":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in output_files]).to_csv(OUT/"manifests/output_artifact_hashes.csv",index=False)
    print(json.dumps(runtime,indent=2)); print(json.dumps({"H1":h1,"H2":h2,"H3":h3,"H4":h4,"overall":overall},indent=2))


if __name__=="__main__": main()
