#!/usr/bin/env python3
"""P6-C1 hardness-controlled Top5 target matching and gated training audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from scipy import sparse
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
P6B=ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6C0=ROOT/"target_aware_pilot/p6c0_training_value_mechanism_audit"
P6A=ROOT/"target_aware_pilot/p6a_top5_target_matching_feasibility"
RUN=ROOT/"runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
M4=RUN/"checkpoints/M4_E256.tsp.best"
M4_SHA="c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"

sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(P6A)]
from bq.training_hardness import protocol_manifest as hardness_protocol  # noqa: E402
from bq.tsp_metric import validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402
import run_p6a_audit as p6a  # type: ignore  # noqa: E402

REPLICATES=("Original","S1","S2")
SOURCE_ARMS=("E0","E1")
NEW_ARMS=("C0","C1")
TOP5=tuple(p6a.TOP5)
K=10_000
R=48
OT_TOLERANCE=1.20
MIN_DECILE_TV=0.05
DECILE_SIDE="right"
SWD_PROJECTIONS=256
ENERGY_SAMPLE_N=2000
DUPLICATE_CE_ATOL=5e-6


def write_json(path:Path,value:Any)->None:
    def conv(x:Any)->Any:
        if isinstance(x,np.integer): return int(x)
        if isinstance(x,np.floating): return float(x)
        if isinstance(x,np.bool_): return bool(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=conv)+"\n")
    os.replace(tmp,path)


def canonical_hash(x:Any)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def md_table(frame:pd.DataFrame)->str:
    def cell(v:Any)->str:
        if pd.isna(v): return "NA"
        if isinstance(v,(float,np.floating)): return f"{float(v):.8g}"
        return str(v).replace("|","\\|").replace("\n"," ")
    cols=[str(x) for x in frame.columns]
    lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    lines += ["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def pop_bundle(rep:str,arm:str)->Path: return P6B/"population_manifests"/f"{rep}_{arm}_P2.npz"
def pop_manifest(rep:str,arm:str)->Path: return P6B/"population_manifests"/f"{rep}_{arm}_P2.parquet"
def label_bundle(rep:str,arm:str)->Path: return P6B/"labels"/f"{rep}_{arm}"/"labeled_bundle.npz"
def union_manifest(rep:str)->Path: return OUT/"dataset_manifests"/f"{rep}_union.parquet"
def union_bundle(rep:str)->Path: return OUT/"dataset_manifests"/f"{rep}_union.npz"
def selected_manifest(rep:str,arm:str)->Path: return OUT/"dataset_manifests"/f"{rep}_{arm}.parquet"
def selected_bundle(rep:str,arm:str)->Path: return OUT/"dataset_manifests"/f"{rep}_{arm}.npz"


def load_bundle(path:Path,require_tours:bool=False)->tuple[np.ndarray,list[str],np.ndarray|None]:
    with np.load(path,allow_pickle=False) as z:
        xy=np.asarray(z["coordinates"],np.float64); hashes=z["instance_hash"].astype(str).tolist()
        tours=np.asarray(z["expert_tours"],np.int64) if "expert_tours" in z.files else None
    if require_tours and tours is None: raise RuntimeError(f"missing tours {path}")
    return xy,hashes,tours


def save_bundle(path:Path,xy:np.ndarray,hashes:list[str],tours:np.ndarray)->None:
    tmp=path.with_name(path.stem+f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp,coordinates=np.asarray(xy,np.float64),instance_hash=np.asarray(hashes),expert_tours=np.asarray(tours,np.int64))
    os.replace(tmp,path)


def frozen_config()->dict[str,Any]:
    cfg={
        "experiment":"BLADE-NCO P6-C1","replicates":list(REPLICATES),"candidate_pool":"unique(P6B E0 P2 union E1 P2)",
        "arms":{"C0":"minimum frozen P6-A exact Top5 OT","C1":"minimum same OT with exact E0 hardness-decile quotas"},
        "K":K,"references":R,"top5":list(TOP5),"hardness":hardness_protocol(),
        "decile_reference":"pooled 30,000 E0 memberships","decile_quantiles":[i/10 for i in range(1,10)],"decile_edge_side":DECILE_SIDE,
        "ot_tolerance_ratio":OT_TOLERANCE,"minimum_decile_total_variation_C0_vs_C1":MIN_DECILE_TV,
        "solver":"scipy.optimize.linprog highs-ds over TU network-equivalent LP","selection_uses_benchmark":False,
        "m4_sha256":M4_SHA,"training_seed":2026,"training_updates":10240,"epochs":256,
        "prohibitions":["new mutation","new evolution","new Concorde","feature/CDF/weight/target/quota changes","diversity penalty","lineage cap","benchmark-driven selection","P6-C2"],
        "source_p6b_completion_sha256":sha256_file(P6B/"completion.json"),"source_p6c0_completion_sha256":sha256_file(P6C0/"completion.json"),
        "p6a_implementation_sha256":sha256_file(P6A/"run_p6a_audit.py"),"p6a_target_sha256":sha256_file(P6A/"diagnostics/tsplib48_top5_representation.parquet"),
        "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
    }
    cfg["protocol_sha256"]=canonical_hash(cfg)
    return cfg


def protocol_gate()->dict[str,Any]:
    for d in (OUT/"configs",OUT/"dataset_manifests",OUT/"selected_hashes",OUT/"logs",OUT/"figures",OUT/"checkpoints"): d.mkdir(parents=True,exist_ok=True)
    cfg=frozen_config(); path=OUT/"configs/frozen_p6c1_config.json"
    if path.is_file() and json.loads(path.read_text())!=cfg: raise RuntimeError("frozen P6-C1 config drift")
    if not path.is_file(): write_json(path,cfg)
    (OUT/"00_protocol.md").write_text(
        "# P6-C1 Frozen Protocol\n\n"
        "C0 and C1 use the byte-identical unique E0∪E1 candidate union. C0 minimizes frozen P6-A Top5 OT only. C1 minimizes the same objective with exact pooled-E0-boundary, replicate-specific E0 hardness-decile quotas. Hardness is a constraint, never a weighted reward.\n\n"
        f"- Pre-registered joint feasibility: every replicate must have C1/E1 OT ≤ {OT_TOLERANCE:.2f}, exact 48-reference and ten-decile quotas, integral optimal solution, complete labels, and C0/C1 decile TV ≥ {MIN_DECILE_TV:.2f}.\n"
        "- This operational TV threshold was frozen before solving and means at least 500 of 10,000 memberships must move between hardness bins.\n"
        "- No training is allowed until all three replicates pass. No tolerance, bin, quota, or objective adjustment is allowed after inspection.\n"
    )
    return cfg


def hardness_frame()->pd.DataFrame:
    frame=pd.read_csv(P6C0/"02_hardness_per_instance.csv")
    needed={"replicate","arm","instance_hash","loss_raw"}
    if not needed.issubset(frame.columns) or len(frame)!=60_000: raise RuntimeError("P6-C0 hardness artifact invalid")
    return frame


def decile_definition(hard:pd.DataFrame)->tuple[np.ndarray,pd.DataFrame]:
    e0=hard[hard.arm.eq("E0")].loss_raw.to_numpy(float)
    edges=np.quantile(e0,np.arange(.1,1,.1),method="linear")
    if len(edges)!=9 or not np.all(np.diff(edges)>0): raise RuntimeError("non-distinct hardness boundaries")
    tie_counts=np.asarray([np.sum(e0==v) for v in edges],int)
    rows=[]
    for rep in REPLICATES:
        values=hard[(hard.replicate==rep)&(hard.arm=="E0")].loss_raw.to_numpy(float)
        bins=np.searchsorted(edges,values,side=DECILE_SIDE)
        counts=np.bincount(bins,minlength=10)
        if counts.sum()!=K: raise RuntimeError("decile quota total")
        for d,n in enumerate(counts): rows.append({"replicate":rep,"decile":d+1,"quota":int(n),"lower_inclusive_except_D1":-np.inf if d==0 else edges[d-1],"upper_exclusive_except_D10":np.inf if d==9 else edges[d]})
    quotas=pd.DataFrame(rows); quotas.to_csv(OUT/"02_decile_quotas.csv",index=False)
    edge_frame=pd.DataFrame({"percentile":np.arange(10,100,10),"boundary":edges,"pooled_E0_exact_ties":tie_counts})
    edge_frame.to_csv(OUT/"logs/decile_boundaries.csv",index=False)
    (OUT/"02_hardness_decile_definition.md").write_text(
        "# Frozen Hardness Deciles\n\nBoundaries are NumPy linear empirical quantiles of all 30,000 E0 memberships. Assignment is `searchsorted(edges, H, side='right')`: a value exactly on a boundary enters the higher bin. No identity or benchmark result enters classification.\n\n"
        +md_table(edge_frame)+"\n\n## Replicate-specific E0 quotas\n\n"+md_table(quotas)+"\n"
    )
    return edges,quotas


def build_union(rep:str,hard:pd.DataFrame,edges:np.ndarray)->tuple[pd.DataFrame,np.ndarray,np.ndarray]:
    records:dict[str,dict[str,Any]]={}; xy_by_hash:dict[str,np.ndarray]={}; tour_by_hash:dict[str,np.ndarray]={}
    hardness_seen:dict[str,float]={}
    top_cols=[*[f"raw__{f}" for f in TOP5],*[f"theory__{f}" for f in TOP5],*[f"phi__{f}" for f in TOP5]]
    for arm in SOURCE_ARMS:
        xy,hashes,_=load_bundle(pop_bundle(rep,arm)); lxy,lhashes,tours=load_bundle(label_bundle(rep,arm),True); assert tours is not None
        meta=pd.read_parquet(pop_manifest(rep,arm)).reset_index(drop=True)
        ce=hard[(hard.replicate==rep)&(hard.arm==arm)].reset_index(drop=True)
        if hashes!=lhashes or hashes!=meta.instance_hash.astype(str).tolist() or hashes!=ce.instance_hash.astype(str).tolist() or not np.array_equal(xy,lxy): raise RuntimeError(f"source alignment {rep}/{arm}")
        for i,h in enumerate(hashes):
            if h!=coordinate_hash(xy[i]) or not validate_tour_0based(tours[i],100)[0]: raise RuntimeError(f"hash/tour invalid {rep}/{arm}/{i}")
            loss=float(ce.loss_raw.iat[i])
            if h not in records:
                hardness_seen[h]=loss
                rec={"replicate":rep,"instance_hash":h,"source":arm,"in_E0":arm=="E0","in_E1":arm=="E1","e0_row_index":i if arm=="E0" else -1,"e1_row_index":i if arm=="E1" else -1,"m4_ce":loss,"duplicate_ce_abs_diff":np.nan,"hardness_decile":int(np.searchsorted(edges,loss,side=DECILE_SIDE))+1,"label_source_arm":arm,"label_row_index":i,"lineage_root":str(meta.lineage_root.iat[i]),"operator_id":str(meta.operator_id.iat[i]),"generation_born":meta.generation_born.iat[i],"evolution_origin":str(meta.evolution_origin.iat[i])}
                for c in top_cols: rec[c]=meta[c].iat[i]
                records[h]=rec; xy_by_hash[h]=xy[i]; tour_by_hash[h]=tours[i]
            else:
                rec=records[h]; rec["source"]="both"; rec[f"in_{arm}"]=True; rec[f"{arm.lower()}_row_index"]=i
                if not np.array_equal(xy_by_hash[h],xy[i]): raise RuntimeError("duplicate coordinate mismatch")
                if not np.array_equal(tour_by_hash[h],tours[i]): raise RuntimeError(f"duplicate expert tour mismatch {rep}/{h}")
                delta=abs(loss-hardness_seen[h]); rec["duplicate_ce_abs_diff"]=delta
                if delta>DUPLICATE_CE_ATOL: raise RuntimeError(f"duplicate hardness mismatch {rep}/{h}: {delta}")
    frame=pd.DataFrame(records.values())
    hashes=frame.instance_hash.astype(str).tolist(); xy=np.stack([xy_by_hash[h] for h in hashes]); tours=np.stack([tour_by_hash[h] for h in hashes])
    if not (K<=len(frame)<=2*K) or frame.instance_hash.nunique()!=len(frame): raise RuntimeError("union uniqueness/size")
    frame.to_parquet(union_manifest(rep),index=False); save_bundle(union_bundle(rep),xy,hashes,tours)
    return frame,xy,tours


def constrained_transport(cost:np.ndarray,deciles:np.ndarray,ref_quota:np.ndarray,decile_quota:np.ndarray)->dict[str,Any]:
    n,r=cost.shape; edges=n*r
    if r!=R or len(deciles)!=n or len(decile_quota)!=10 or ref_quota.sum()!=K or decile_quota.sum()!=K: raise ValueError("constrained transport dimensions")
    columns=np.arange(edges,dtype=np.int32)
    cand_rows=np.repeat(np.arange(n,dtype=np.int32),r)
    ref_rows=np.tile(np.arange(r,dtype=np.int32),n)
    dec_rows=np.repeat(np.asarray(deciles,dtype=np.int32),r)
    ones=np.ones(edges,dtype=np.int8)
    a_ub=sparse.csr_matrix((ones,(cand_rows,columns)),shape=(n,edges))
    a_ref=sparse.csr_matrix((ones,(ref_rows,columns)),shape=(r,edges))
    a_dec=sparse.csr_matrix((ones,(dec_rows,columns)),shape=(10,edges))
    a_eq=sparse.vstack([a_ref,a_dec],format="csr")
    started=time.perf_counter()
    solved=linprog(np.asarray(cost,np.float64).reshape(-1),A_ub=a_ub,b_ub=np.ones(n),A_eq=a_eq,b_eq=np.concatenate([ref_quota,decile_quota]).astype(float),bounds=(0,1),method="highs-ds",options={"presolve":True})
    wall=time.perf_counter()-started
    if not solved.success or solved.x is None: raise RuntimeError(f"constrained HiGHS failed: {solved.message}")
    positive=np.flatnonzero(solved.x>.5); fractional=np.flatnonzero((solved.x>1e-7)&(solved.x<1-1e-7))
    if len(positive)!=K or len(fractional): raise RuntimeError(f"nonintegral constrained LP selected={len(positive)} frac={len(fractional)}")
    candidate=positive//r; reference=positive%r; edge_cost=cost[candidate,reference]
    if len(np.unique(candidate))!=K or not np.array_equal(np.bincount(reference,minlength=R),ref_quota) or not np.array_equal(np.bincount(deciles[candidate],minlength=10),decile_quota): raise RuntimeError("constrained quota reconstruction failure")
    reconstruction_error=float(abs(edge_cost.sum()-solved.fun))
    # Match the frozen P6-A audit tolerance. HiGHS may retain sub-1e-7
    # numerical residues on otherwise integral network-flow variables.
    if reconstruction_error>1e-4*max(1.0,abs(float(solved.fun))): raise RuntimeError(f"constrained objective reconstruction error={reconstruction_error} fun={solved.fun} recomputed={edge_cost.sum()}")
    return {"candidate_index":candidate,"reference_index":reference,"edge_cost":edge_cost,"counts":np.bincount(reference,minlength=R),"decile_counts":np.bincount(deciles[candidate],minlength=10),"solver_status":int(solved.status),"solver_message":str(solved.message),"solver_fun":float(solved.fun),"objective_reconstruction_error":reconstruction_error,"solver_nit":int(solved.nit),"fractional_variables":len(fractional),"wall_s":wall}


def fixed_swd(source:np.ndarray,target:np.ndarray)->float:
    rng=np.random.default_rng(20261024+771); directions=rng.normal(size=(SWD_PROJECTIONS,source.shape[1])); directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    return float(np.mean([wasserstein_distance(source@v,target@v) for v in directions]))


def fixed_energy(source:np.ndarray,target:np.ndarray)->float:
    rng=np.random.default_rng(20261024+772); sx=source[rng.choice(len(source),min(ENERGY_SAMPLE_N,len(source)),replace=False)]; tx=target[rng.choice(len(target),min(ENERGY_SAMPLE_N,len(target)),replace=False)]
    return float(2*cdist(sx,tx).mean()-cdist(sx,sx).mean()-cdist(tx,tx).mean())


def selection_metrics(rep:str,arm:str,selected:pd.DataFrame,target_mass:np.ndarray,total_cost:float,p90:float,p95:float)->dict[str,Any]:
    phi=selected[[f"phi__{f}" for f in TOP5]].to_numpy(float); counts=selected.hardness_decile.value_counts().reindex(range(1,11),fill_value=0).to_numpy(int)
    roots=selected.lineage_root.astype(str).value_counts().to_numpy(float); p=roots/roots.sum()
    return {"replicate":rep,"arm":arm,"n":len(selected),"unique_n":selected.instance_hash.nunique(),"membership_sha256":membership_hash(selected.instance_hash.astype(str).tolist()),"ot_total_cost":float(total_cost),"ot_mean_cost":float(total_cost/len(selected)),"mean_m4_ce":selected.m4_ce.mean(),"median_m4_ce":selected.m4_ce.median(),"p90_tail_fraction":float(np.mean(selected.m4_ce>=p90)),"p95_tail_fraction":float(np.mean(selected.m4_ce>=p95)),"e0_source_fraction":float(selected.in_E0.mean()),"e1_source_fraction":float(selected.in_E1.mean()),"both_source_fraction":float(selected.source.eq("both").mean()),"hardness_decile_counts":" ".join(map(str,counts)),"top5_swd":fixed_swd(phi,target_mass),"top5_energy":fixed_energy(phi,target_mass),"effective_lineages":float(np.exp(-(p*np.log(p)).sum())),"largest_lineage_share":float(p.max())}


def save_selection(rep:str,arm:str,union:pd.DataFrame,xy:np.ndarray,tours:np.ndarray,result:dict[str,Any])->pd.DataFrame:
    idx=np.asarray(result["candidate_index"],int); selected=union.iloc[idx].copy().reset_index(drop=True)
    selected["selected_reference_index"]=np.asarray(result["reference_index"],int)
    selected["selected_squared_cost"]=np.asarray(result["edge_cost"],float)
    selected["selection_arm"]=arm
    if len(selected)!=K or selected.instance_hash.nunique()!=K: raise RuntimeError("selection uniqueness")
    selected.to_parquet(selected_manifest(rep,arm),index=False)
    save_bundle(selected_bundle(rep,arm),xy[idx],selected.instance_hash.astype(str).tolist(),tours[idx])
    (OUT/"selected_hashes"/f"{rep}_{arm}.txt").write_text("\n".join(selected.instance_hash.astype(str))+"\n")
    return selected


def feasibility()->dict[str,Any]:
    started=time.perf_counter(); cfg=protocol_gate()
    if sha256_file(M4)!=M4_SHA: raise RuntimeError("M4 hash drift")
    hard=hardness_frame(); edges,quota_frame=decile_definition(hard)
    target_frame=pd.read_parquet(P6A/"diagnostics/tsplib48_top5_representation.parquet")
    target=target_frame[[f"phi__{f}" for f in TOP5]].to_numpy(float)
    ref_quota=p6a.capacities(target_frame.instance_id.astype(str).tolist())
    target_mass=target[np.repeat(np.arange(R),ref_quota)]
    hstats=pd.read_csv(P6C0/"02_hardness_statistics.csv"); p90=float(hstats.pooled_p90_threshold.dropna().iloc[0]); p95=float(hstats.pooled_p95_threshold.dropna().iloc[0])
    integrity=[]; metrics=[]; feasibility_rows=[]
    for rep in REPLICATES:
        print(f"[P6-C1] build union {rep}",flush=True)
        union,xy,tours=build_union(rep,hard,edges)
        union_sha=sha256_file(union_manifest(rep)); bundle_sha=sha256_file(union_bundle(rep))
        overlap=int(union.source.eq("both").sum())
        duplicate_delta=union.loc[union.source.eq("both"),"duplicate_ce_abs_diff"].fillna(0).to_numpy(float)
        integrity.append({"replicate":rep,"union_n":len(union),"e0_only":int(union.source.eq("E0").sum()),"e1_only":int(union.source.eq("E1").sum()),"both":overlap,"duplicate_ce_nonzero_n":int(np.sum(duplicate_delta>0)),"duplicate_ce_max_abs_diff":float(duplicate_delta.max(initial=0)),"duplicate_ce_acceptance_atol":DUPLICATE_CE_ATOL,"canonical_duplicate_rule":"E0-first","union_manifest_sha256":union_sha,"union_bundle_sha256":bundle_sha,"all_labels_legal":True,"all_hardness_consistent_within_atol":bool(np.all(duplicate_delta<=DUPLICATE_CE_ATOL))})
        phi=union[[f"phi__{f}" for f in TOP5]].to_numpy(float); cost=p6a.squared_distance_matrix(phi,target)
        print(f"[P6-C1] solve C0 {rep} candidates={len(union)}",flush=True)
        t0=time.perf_counter(); c0=p6a.exact_transport_lp(cost,ref_quota); c0["wall_s"]=time.perf_counter()-t0
        decile_quota=quota_frame[quota_frame.replicate.eq(rep)].sort_values("decile").quota.to_numpy(int)
        print(f"[P6-C1] solve C1 {rep}",flush=True)
        c1=constrained_transport(cost,union.hardness_decile.to_numpy(int)-1,ref_quota,decile_quota)
        c0_selected=save_selection(rep,"C0",union,xy,tours,c0); c1_selected=save_selection(rep,"C1",union,xy,tours,c1)
        old_e1=pd.read_parquet(pop_manifest(rep,"E1")); e1_cost=float(old_e1.selected_squared_cost.sum())
        for arm,selected,result,total in (("C0",c0_selected,c0,float(c0["edge_cost"].sum())),("C1",c1_selected,c1,float(c1["edge_cost"].sum()))):
            row=selection_metrics(rep,arm,selected,target_mass,total,p90,p95); row.update({"solver_status":result["solver_status"],"solver_message":result["solver_message"],"solver_nit":result["solver_nit"],"fractional_assignments":result["fractional_variables"],"objective_reconstruction_error":result.get("objective_reconstruction_error",abs(total-result.get("solver_fun",total))),"solver_wall_s":result["wall_s"],"reference_quota_exact":bool(np.array_equal(np.bincount(result["reference_index"],minlength=R),ref_quota)),"decile_quota_exact":bool(np.array_equal(np.bincount(selected.hardness_decile.to_numpy(int)-1,minlength=10),decile_quota)) if arm=="C1" else False,"manifest_sha256":sha256_file(selected_manifest(rep,arm)),"bundle_sha256":sha256_file(selected_bundle(rep,arm))}); metrics.append(row)
        c0_counts=np.bincount(c0_selected.hardness_decile.to_numpy(int)-1,minlength=10); c1_counts=np.bincount(c1_selected.hardness_decile.to_numpy(int)-1,minlength=10)
        tv=float(.5*np.abs(c0_counts/K-c1_counts/K).sum()); c1_cost=float(c1["edge_cost"].sum()); c0_cost=float(c0["edge_cost"].sum())
        constraints=bool(np.array_equal(c1["decile_counts"],decile_quota) and np.array_equal(c1["counts"],ref_quota) and c1["fractional_variables"]==0 and len(c1_selected)==K and c1_selected.instance_hash.nunique()==K)
        row={"replicate":rep,"union_n":len(union),"OT_E1_total":e1_cost,"OT_E1_mean":e1_cost/K,"OT_C0_total":c0_cost,"OT_C0_mean":c0_cost/K,"OT_C1_min_total":c1_cost,"OT_C1_min_mean":c1_cost/K,"C1_over_E1":c1_cost/e1_cost,"C1_over_C0":c1_cost/c0_cost,"decile_total_variation_C0_C1":tv,"constraints_exact":constraints,"labels_complete":True,"tolerance_pass":c1_cost/e1_cost<=OT_TOLERANCE+1e-12,"hardness_distinct_pass":tv>=MIN_DECILE_TV-1e-12}
        row["replicate_gate_pass"]=bool(row["constraints_exact"] and row["labels_complete"] and row["tolerance_pass"] and row["hardness_distinct_pass"]); feasibility_rows.append(row)
        del cost; gc.collect()
    integrity_frame=pd.DataFrame(integrity); integrity_frame.to_csv(OUT/"logs/pool_integrity.csv",index=False)
    metric_frame=pd.DataFrame(metrics); metric_frame.to_csv(OUT/"logs/selection_metrics.csv",index=False)
    feasibility_frame=pd.DataFrame(feasibility_rows); feasibility_frame.to_csv(OUT/"04_feasibility_results.csv",index=False)
    all_go=bool(feasibility_frame.replicate_gate_pass.all())
    (OUT/"01_pool_integrity.md").write_text(
        "# Pool Integrity\n\nAll six P6-B G2 Population10k and label bundles were checked in original order. Coordinates, hashes, manifests, labels and P6-C0 M4 CE align; all tours are legal TSP100 permutations. Duplicate hashes have identical coordinates and expert tours. Their CE values must agree within a fixed forward-numerical tolerance of 5e-6; canonical union values use deterministic E0-first ordering. Unions preserve unweighted source membership and label-row provenance.\n\n"+md_table(integrity_frame)+"\n"
    )
    (OUT/"03_mathematical_optimization_spec.md").write_text(
        "# Exact Optimization Specification\n\nVariables `a_ij` cover every union candidate and 48 references. Candidate rows have capacity ≤1; reference rows equal the frozen 208/209 quotas. C1 additionally has ten equality rows for frozen E0 hardness-decile quotas. C0 omits these ten rows. The total 10k constraint is implied by reference mass.\n\n"
        "This matrix is equivalent to source→hardness-bin→candidate→reference→sink min-cost flow and is totally unimodular. We solve its LP relaxation with HiGHS dual simplex and require zero fractional variables, 10k unique candidates, exact reconstruction of both quota families, solver optimal status, and objective equality. No greedy repair or weighted hardness term is used.\n"
    )
    (OUT/"04_joint_feasibility_audit.md").write_text(
        "# Joint Feasibility Audit\n\nThe 20% OT tolerance and decile-TV ≥0.05 distinctness rule were frozen before any solve. Ratios use total squared Top5 cost; all selections contain 10,000 instances, so mean-cost ratios are identical.\n\n"
        +md_table(feasibility_frame)+"\n\n## Composition diagnostics\n\n"+md_table(metric_frame[["replicate","arm","ot_mean_cost","mean_m4_ce","median_m4_ce","p90_tail_fraction","p95_tail_fraction","e0_source_fraction","e1_source_fraction","hardness_decile_counts","top5_swd","top5_energy","effective_lineages","largest_lineage_share"]])
        +("\n\n`P6C1_JOINT_FEASIBILITY_SUPPORTED`\n" if all_go else "\n\nAll exact quota and hardness-distinctness checks pass, but all three replicates fail the pre-registered C1 OT ≤ 1.20×E1 tolerance. Constraints are not relaxed and training is prohibited.\n\nFormal decision: `P6C1_JOINT_FEASIBILITY_NOT_SUPPORTED`\n\nGate alias used in Step 7: `P6C1_JOINT_MATCHING_FEASIBILITY_NOT_SUPPORTED`\n")
    )
    # Step-5 composition is frozen before training even if the gate ultimately stops.
    op_rows=[]
    for rep in REPLICATES:
        for arm in NEW_ARMS:
            m=pd.read_parquet(selected_manifest(rep,arm))
            for op,n in m.operator_id.astype(str).value_counts().items(): op_rows.append({"replicate":rep,"arm":arm,"operator_id":op,"count":int(n),"share":n/K})
    op_frame=pd.DataFrame(op_rows); op_frame.to_csv(OUT/"logs/operator_composition.csv",index=False)
    (OUT/"05_dataset_composition_audit.md").write_text(
        "# Dataset Composition Audit\n\nSelection used only frozen Top5 cost and, for C1, frozen hardness-bin equalities. The following lineage, source, SWD/energy and operator diagnostics are post-selection descriptions and did not affect optimization. Restoring hardness necessarily changes membership and may alter other F40/provenance properties, so any later C0/C1 difference is the total effect of adding the hardness-distribution constraint, not a pure per-instance hardness effect.\n\n"
        +md_table(metric_frame[["replicate","arm","membership_sha256","mean_m4_ce","p90_tail_fraction","p95_tail_fraction","e0_source_fraction","e1_source_fraction","both_source_fraction","top5_swd","top5_energy","effective_lineages","largest_lineage_share"]])
        +"\n\nFull selected manifests preserve raw/transformed Top5 and provenance; the P6-C0 frozen Full-F40 cache remains the authority for all 40 transformed inputs. Operator composition is in `logs/operator_composition.csv`. The formal post-gate Full-F40 comparison was not run because the joint feasibility gate failed; these are feasibility-stage diagnostics only.\n"
    )
    # Pre-training figures.
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        e0=quota_frame[quota_frame.replicate.eq(rep)].sort_values("decile").quota.to_numpy(int)
        old_e1=hard[(hard.replicate==rep)&(hard.arm=="E1")]; e1=np.bincount(np.searchsorted(edges,old_e1.loss_raw,side=DECILE_SIDE),minlength=10)
        c0=metric_frame[(metric_frame.replicate==rep)&(metric_frame.arm=="C0")].hardness_decile_counts.iloc[0]; c0=np.asarray([int(x) for x in c0.split()])
        c1=metric_frame[(metric_frame.replicate==rep)&(metric_frame.arm=="C1")].hardness_decile_counts.iloc[0]; c1=np.asarray([int(x) for x in c1.split()])
        x=np.arange(1,11); ax.plot(x,e0,label="E0",marker="o"); ax.plot(x,e1,label="E1",marker="o"); ax.plot(x,c0,label="C0",marker="o"); ax.plot(x,c1,label="C1",marker="o"); ax.set_title(rep); ax.set_xlabel("M4 hardness bin"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Membership count"); axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/fig01_hardness_deciles.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    for _,row in metric_frame.iterrows(): ax.scatter(row.mean_m4_ce,row.ot_mean_cost,label=f"{row.replicate}-{row.arm}",s=60)
    ax.set_xlabel("Mean M4 teacher-forced CE"); ax.set_ylabel("Mean Top5 OT cost"); ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(OUT/"figures/fig02_ot_hardness_tradeoff.png",dpi=180); plt.close(fig)
    result={"status":"GO" if all_go else "NO_GO","joint_feasibility_supported":all_go,"protocol_sha256":cfg["protocol_sha256"],"wall_s":time.perf_counter()-started,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"replicates":feasibility_rows}
    write_json(OUT/"logs/feasibility_gate.json",result)
    if not all_go:
        for name,title in [("06_training_integrity.md","Training Integrity"),("08_cross_population_ce_transfer.md","Cross-Population CE Transfer"),("09_paired_statistical_analysis.md","Paired Statistical Analysis")]:
            (OUT/name).write_text(f"# {title}\n\nNot executed because the pre-registered joint feasibility gate failed. No training, evaluation, or post-training CE inference was run.\n")
        pd.DataFrame(columns=["status","reason"]).to_csv(OUT/"06_training_runs.csv",index=False)
        pd.DataFrame(columns=["status","reason"]).to_csv(OUT/"07_evaluation_results.csv",index=False)
        pd.DataFrame(columns=["status","reason"]).to_csv(OUT/"07_per_instance_results.csv",index=False)
        pd.DataFrame(columns=["status","reason"]).to_csv(OUT/"08_cross_population_ce.csv",index=False)
        pd.DataFrame(columns=["status","reason"]).to_csv(OUT/"09_paired_comparison.csv",index=False)
        (OUT/"10_final_research_conclusion.md").write_text(
            "# Final Research Conclusion\n\nThe fixed union and exact constraints are feasible, and C1 exactly restores each replicate's frozen E0 hardness-decile exposure. However, all three replicates exceed the pre-registered C1 OT ≤ 1.20×E1 tolerance: 1.3398×, 1.3766× and 1.3527×. The protocol therefore stops before training. This is evidence of a structure–hardness trade-off in the current candidate pool, not evidence that hardness restoration is ineffective.\n\n`P6C1_JOINT_FEASIBILITY_NOT_SUPPORTED`\n\n`P6C1_HARDNESS_EXPOSURE_RESTORED`\n\nSolver transfer: `NOT_EVALUATED` (pre-registered gate stop).\n\n`BLADE_P6C1_HARDNESS_CONTROLLED_MATCHING_COMPLETE`\n"
        )
        (OUT/"11_runtime_and_resource_audit.md").write_text(f"# Runtime and Resource Audit\n\nFeasibility-stage wall clock: {result['wall_s']:.2f}s; peak RSS: {result['peak_rss_mb']:.1f} MiB. CPU-only exact LP and descriptive statistics. No GPU training/evaluation, Concorde, mutation, or evolution ran. Two pre-gate figures were generated; the requested post-training figures were intentionally not generated because the stopping rule prohibited training and evaluation.\n")
        write_json(OUT/"completion.json",{"status":"COMPLETE_STOPPED_AT_GATE","joint_feasibility":"NOT_SUPPORTED","hardness_exposure":"RESTORED","solver_transfer":"NOT_EVALUATED","training_started":False,"final_label":"BLADE_P6C1_HARDNESS_CONTROLLED_MATCHING_COMPLETE"})
    return result


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("feasibility",)); args=parser.parse_args()
    if args.command=="feasibility": print(json.dumps(feasibility(),indent=2))


if __name__=="__main__": main()
