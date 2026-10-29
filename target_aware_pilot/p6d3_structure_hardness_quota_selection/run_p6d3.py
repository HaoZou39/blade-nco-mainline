#!/usr/bin/env python3
"""P6-D3 frozen within-niche structure/hardness quota experiment.

Selection is performed only over the byte-frozen P6-D0 candidate pools.  The
script reuses existing coordinates, Concorde tours, M4/HGB predictions and the
formal P6-D1 trainer/evaluator.  It never mutates instances, calls Concorde, or
fits a surrogate.
"""
from __future__ import annotations

import argparse, fcntl, gc, hashlib, json, os, platform, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
D0=ROOT/"target_aware_pilot/p6d0_target_aware_qd_selection_audit"
D1=ROOT/"target_aware_pilot/p6d1_target_aware_qd_training_value"
D2=ROOT/"target_aware_pilot/p6d2_selected_population_structure_audit"
C0=ROOT/"target_aware_pilot/p6c0_training_value_mechanism_audit"
C1=ROOT/"target_aware_pilot/p6c1_hardness_controlled_matching"
B6=ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6A=ROOT/"target_aware_pilot/p6a_top5_target_matching_feasibility"
RUN=ROOT/"runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
M4=RUN/"checkpoints/M4_E256.tsp.best"; M4_EVAL=RUN/"cycles/C03/evaluation_work/M4"
M4_SHA="c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
REQUEST=Path("/home/haozou/.codex/attachments/2d32d16f-26c7-43b2-9f73-40a0ae6bdabc/pasted-text.txt")
REPS=("Original","S1","S2"); ARMS=("D25","D50","D75"); RATIOS={"D25":.25,"D50":.50,"D75":.75}; K=10_000
JOB_ORDER=tuple((r,a) for r in REPS for a in ARMS); TRAINING_SEED=2026; BOOT_N=2000; BOOT_SEED=20261110

sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(P6A),str(D1)]
import run_p6a_audit as p6a  # noqa: E402
import run_p6d1 as d1  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.tsp_metric import validate_tour_0based  # noqa: E402
from final_run.science import hardness, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402

TOP5=tuple(p6a.TOP5); DATASETS=tuple(d1.DATASETS); GAP5=tuple(d1.GAP5)
F40=tuple(json.loads((ROOT/"audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json").read_text())["F40"])

def now(): return datetime.now(timezone.utc).isoformat()
def chash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def write_json(path:Path,v:Any):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(v,indent=2,sort_keys=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n"); os.replace(tmp,path)
def md(df:pd.DataFrame): return d1.md_table(df)
def eff(s:pd.Series):
    p=s.astype(str).value_counts(normalize=True).to_numpy(float); return float(np.exp(-(p*np.log(p)).sum())),float(p.max())
def ensure_dirs():
    for p in (OUT,OUT/"configs",OUT/"datasets",OUT/"checkpoints",OUT/"logs",OUT/"hardness",OUT/"figures",OUT/"figures/cases",OUT/"monitoring",OUT/"02_training_configs",OUT/"selected_hashes"): p.mkdir(parents=True,exist_ok=True)
    for name in ("failures.md","repair_actions.md"):
        p=OUT/"monitoring"/name
        if not p.exists(): p.write_text(f"# {name[:-3].replace('_',' ').title()}\n\n")

def load_bundle(path:Path, require_tours=True): return d1.load_bundle(path,require_tours)
def save_bundle(path:Path,xy,hh,tt): return d1.save_bundle(path,xy,hh,tt)
def bundle_path(rep,arm): return OUT/"datasets"/f"{rep}_{arm}.npz"
def manifest_path(rep,arm): return OUT/"datasets"/f"{rep}_{arm}.parquet"
def run_dir(rep,arm): return OUT/"logs"/f"{rep}_{arm}"
def checkpoint(rep,arm): return OUT/"checkpoints"/f"{rep}_{arm}_M5_E256.tsp.best"
def eval_label(rep,arm): return f"P6D3_{rep}_{arm}_M5"
def eval_csv(rep,arm): return run_dir(rep,arm)/"external_benchmark_results.csv"

class Status:
    def update(self,**kw):
        p=OUT/"status.json"; old=json.loads(p.read_text()) if p.exists() else {}; write_json(p,{**old,**kw,"updated_utc":now()})

def training_protocol():
    x=d1.base.base_training_protocol(); x.update({"formal_source":"P6-B/P6-C2 core.train via audited P6-D1 adapter","trainer_sha256":sha256_file(ROOT/"scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py"),"training_order":[f"{r}-{a}" for r,a in JOB_ORDER],"device":"cuda:1"}); return x

def frozen_protocol():
    v={"experiment":"BLADE-NCO P6-D3","request_sha256":sha256_file(REQUEST),"p6d0_completion_sha256":sha256_file(D0/"completion.json"),"p6d1_completion_sha256":sha256_file(D1/"completion.json"),"p6d2_completion_sha256":sha256_file(D2/"completion.json"),"m4_sha256":M4_SHA,"replicates":list(REPS),"arms":RATIOS,"structure_score":"minimum frozen P6-A squared Top5 candidate-reference cost","quota":"deterministic largest remainder; tie niche id","selection_order":"STRUCTURE ascending score then HARDNESS descending frozen HGB; hash tie","training":training_protocol(),"bootstrap_n":BOOT_N,"bootstrap_seed":BOOT_SEED,"monitor_interval":"60m","decision_rules":{"H1":"SUPPORTED iff d_struct and exact OT are monotone non-increasing D25->D50->D75 in all 3 reps; MIXED if either is monotone in >=1 rep","H2":"SUPPORTED iff every D arm retains frozen P90 tail >= B in all reps; MIXED iff at least one fixed D arm does so in all reps","H3":"PRELIMINARILY_SUPPORTED iff one fixed D arm beats C Original70 3/3 and is not worse GAP5 3/3; MIXED iff any fixed arm beats C in 1-2 reps","H4":"SUPPORTED_PRELIMINARY iff one fixed D arm beats B Original70 3/3 and is not worse GAP5 3/3; MIXED iff any arm beats B in 1-2 reps","H5":"SUPPORTED_PRELIMINARY iff one fixed D arm beats M4 Original70 3/3; PARTIAL iff any does in 1-2 reps","H6":"SUPPORTED_EXPLORATORY iff structure ratio has consistent monotone offline tradeoff and same-direction Original70 rank in all reps; MIXED if offline tradeoff holds but gap rank differs","H7":"PROMISING only if H3 and H4 supported and H5 not NOT_SUPPORTED; otherwise NOT_ESTABLISHED unless all gain tests fail"},"prohibitions":["mutation","evolution","Concorde","HGB training","ratio search","lambda change","benchmark-driven selection","P6-D4"],"python":platform.python_version(),"torch":torch.__version__}; v["protocol_sha256"]=chash(v); return v

def protocol_gate():
    ensure_dirs(); v=frozen_protocol(); p=OUT/"configs/frozen_p6d3_config.json"
    if p.exists() and json.loads(p.read_text())!=v: raise RuntimeError("frozen P6-D3 protocol drift")
    if not p.exists(): write_json(p,v)
    (OUT/"00_protocol.md").write_text("# P6-D3 Frozen Protocol\n\nThree preregistered within-niche structure/hardness ratios are selected over the exact P6-D0 pools. Structure uses minimum squared P6-A Top5 reference cost; hardness uses frozen M4-conditioned HGB40. Structure-first, largest-remainder quotas, hash ties, lambda=.5 and all training/evaluation settings are immutable. No mutation, evolution, Concorde, HGB fit, ratio search or benchmark-driven selection is permitted.\n\n- Protocol SHA256: `"+v["protocol_sha256"]+"`\n- M4 SHA256: `"+M4_SHA+"`\n- Training: fresh Adam from M4, seed 2026, 256 epochs, 10,240 updates.\n")
    return v

# Repoint the audited P6-D1 execution primitives to this isolated experiment.
d1.OUT=OUT; d1.ARMS=ARMS; d1.JOB_ORDER=JOB_ORDER; d1.Status=Status; d1.protocol_gate=protocol_gate
d1.bundle_path=bundle_path; d1.manifest_path=manifest_path; d1.run_dir=run_dir; d1.checkpoint=checkpoint; d1.eval_label=eval_label; d1.eval_csv=eval_csv

def largest_remainder(q:np.ndarray,r:float):
    raw=r*q; out=np.floor(raw).astype(int); need=int(round(r*K))-int(out.sum()); order=np.lexsort((np.arange(len(q)),-(raw-out))); out[order[:need]]+=1
    if out.sum()!=int(r*K) or np.any(out>q): raise RuntimeError("role quota allocation")
    return out

def build_selection():
    cfg=protocol_gate()
    for p,label in ((D0,"BLADE_P6D0_TARGET_AWARE_QD_SELECTION_AUDIT_COMPLETE"),(D1,"BLADE_P6D1_TARGET_AWARE_QD_TRAINING_VALUE_COMPLETE"),(D2,"BLADE_P6D2_SELECTED_POPULATION_STRUCTURE_AUDIT_COMPLETE")):
        if json.loads((p/"completion.json").read_text()).get("final_label")!=label: raise RuntimeError(f"incomplete source {p.name}")
    if sha256_file(M4)!=M4_SHA: raise RuntimeError("M4 hash drift")
    cand=pd.read_csv(D0/"04_candidate_manifest.csv"); target=pd.read_parquet(P6A/"diagnostics/tsplib48_top5_representation.parquet")
    phi_cols=[f"phi__{x}" for x in TOP5]; tphi=target[phi_cols].to_numpy(float); refnames=target.instance_id.astype(str).to_numpy()
    selected0=pd.read_csv(D0/"05_selection_manifests.csv"); qmap=pd.read_csv(D0/"03_niche_target_relevance.csv")
    scoreparts=[]; quota_rows=[]; selparts=[]; integrity=[]; role_overlap=[]
    f40=pd.concat([pd.read_parquet(p) for p in sorted((C0/"full_f40_cache").glob("*.parquet"))],ignore_index=True).drop_duplicates("instance_hash")
    if len(f40)!=59678: raise RuntimeError("F40 cache drift")
    for rep in REPS:
        pool=cand[cand.replicate.eq(rep)].copy().reset_index(drop=True); cost=p6a.squared_distance_matrix(pool[phi_cols].to_numpy(float),tphi); nr=cost.argmin(axis=1)
        pool["d_struct"]=cost[np.arange(len(pool)),nr]; pool["nearest_reference"]=refnames[nr]; scoreparts.append(pool[["replicate","instance_hash","qd_niche","d_struct","nearest_reference"]])
        q=qmap[qmap.replicate.eq(rep)].sort_values("niche_id").lambda05_projected_quota.to_numpy(int)
        if q.sum()!=K: raise RuntimeError("D0 quota recovery")
        # Historical B/C exact recovery against D1 training bundles.
        for arm in ("B","C"):
            h=selected0[(selected0.replicate.eq(rep))&selected0.selection.eq(arm)].sort_values("selection_order").instance_hash.astype(str).tolist(); dh=load_bundle(D1/"datasets"/f"{rep}_{arm}.npz")[1]
            if h!=dh: raise RuntimeError(f"D0/D1 membership drift {rep}/{arm}")
        for arm,r in RATIOS.items():
            qs=largest_remainder(q,r); qh=q-qs; chosen=[]; order_n=0
            for niche in range(1000):
                g=pool[pool.qd_niche.eq(niche)]; take=int(q[niche]); ns=int(qs[niche]); nh=int(qh[niche])
                if len(g)<take: raise RuntimeError(f"capacity {rep}/{arm}/{niche}")
                sr=g.sort_values(["d_struct","instance_hash"],ascending=[True,True],kind="stable"); s=sr.head(ns); rem=g[~g.instance_hash.isin(set(s.instance_hash))]; h=rem.sort_values(["predicted_m4_ce","instance_hash"],ascending=[False,True],kind="stable").head(nh)
                ideal_h=set(g.sort_values(["predicted_m4_ce","instance_hash"],ascending=[False,True],kind="stable").head(nh).instance_hash.astype(str)); ideal_s=set(sr.head(ns).instance_hash.astype(str)); inter=len(ideal_s&ideal_h)
                quota_rows.append({"replicate":rep,"arm":arm,"structure_ratio":r,"niche_id":niche,"q_total":take,"q_structure":ns,"q_hardness":nh,"available":len(g),"ideal_rank_intersection_n":inter,"ideal_rank_intersection_over_min_quota":inter/min(ns,nh) if min(ns,nh)>0 else np.nan})
                for role,z in (("STRUCTURE",s),("HARDNESS",h)):
                    z=z.copy(); z["arm"]=arm; z["structure_ratio"]=r; z["selection_role"]=role; z["role_rank"]=np.arange(1,len(z)+1); z["selection_order"]=np.arange(order_n+1,order_n+len(z)+1); order_n+=len(z); chosen.append(z)
            z=pd.concat(chosen,ignore_index=True)
            if len(z)!=K or z.instance_hash.nunique()!=K or z.selection_role.value_counts().to_dict()!={"HARDNESS":int((1-r)*K),"STRUCTURE":int(r*K)}: raise RuntimeError(f"selection invariant {rep}/{arm}")
            got=z.qd_niche.value_counts().reindex(range(1000),fill_value=0).to_numpy(int)
            if not np.array_equal(got,q): raise RuntimeError(f"niche quota drift {rep}/{arm}")
            selparts.append(z); role_overlap.append({"replicate":rep,"arm":arm,"ideal_rank_overlap_n":sum(x["ideal_rank_intersection_n"] for x in quota_rows if x["replicate"]==rep and x["arm"]==arm)})
            (OUT/"selected_hashes"/f"{rep}_{arm}.txt").write_text("\n".join(z.instance_hash.astype(str))+"\n")
        # Restore exact existing labels by coordinate hash.
        ux,uh,ut=load_bundle(C1/"dataset_manifests"/f"{rep}_union.npz"); lookup={h:i for i,h in enumerate(uh)}
        for arm in ARMS:
            z=next(x for x in selparts if x.replicate.iloc[0]==rep and x.arm.iloc[0]==arm); hh=z.instance_hash.astype(str).tolist(); idx=np.asarray([lookup[h] for h in hh]); xy=ux[idx]; tt=ut[idx]
            if [coordinate_hash(x) for x in xy]!=hh or not all(validate_tour_0based(t,100)[0] for t in tt): raise RuntimeError(f"label/hash integrity {rep}/{arm}")
            z.to_parquet(manifest_path(rep,arm),index=False); save_bundle(bundle_path(rep,arm),xy,hh,tt)
            integrity.append({"replicate":rep,"arm":arm,"n":len(hh),"unique_n":len(set(hh)),"structure_n":int((z.selection_role=="STRUCTURE").sum()),"hardness_n":int((z.selection_role=="HARDNESS").sum()),"membership_sha256":membership_hash(hh),"manifest_sha256":sha256_file(manifest_path(rep,arm)),"bundle_sha256":sha256_file(bundle_path(rep,arm)),"source_union_sha256":sha256_file(C1/"dataset_manifests"/f"{rep}_union.npz"),"coordinate_hashes_match":True,"expert_tours_legal":True})
    scores=pd.concat(scoreparts,ignore_index=True); scores.to_csv(OUT/"02_candidate_structural_scores.csv",index=False)
    quotas=pd.DataFrame(quota_rows); quotas.to_csv(OUT/"03_niche_role_quotas.csv",index=False)
    sels=pd.concat(selparts,ignore_index=True); sels.to_csv(OUT/"04_selection_manifests.csv",index=False)
    ints=pd.DataFrame(integrity); ints.to_csv(OUT/"logs/dataset_integrity.csv",index=False)
    (OUT/"01_input_integrity.md").write_text("# Input Integrity\n\nExact P6-D0 pools, lambda=.5 niche quotas, B/C memberships, frozen Top5 transform, M4-conditioned HGB predictions, M4 checkpoint, coordinate hashes, existing tours and provenance pass. No candidate or label was regenerated.\n\n"+md(ints)+"\n")
    (OUT/"02_structural_proximity_definition.md").write_text("# Structural Proximity Definition\n\n`d_struct(X)=min_j FrozenTop5Cost(phi(X),phi(TSPLIB48_j))`, using the exact P6-A transformed five-dimensional representation and squared Euclidean cost. Lower is closer. NA semantics, feature order, scaling and float precision are inherited from the frozen P6-A artifacts; QD centroid distance and raw F40 are not used. This instance score is distinct from population exact OT.\n")
    (OUT/"04_dual_quota_selection_algorithm.md").write_text("# Dual-Quota Selection\n\nFor every frozen niche quota, deterministic largest remainder assigns the global role total. Structure-first selects ascending `d_struct`; hardness then selects descending frozen HGB40 among remaining candidates; ties use coordinate hash. There is no lineage/diversity penalty, threshold, reference quota, oracle CE, backfill across niches, or ratio search.\n\n"+md(pd.DataFrame(role_overlap))+"\n")
    return cfg,sels,f40,tphi,target

def population_audit(sels:pd.DataFrame,f40:pd.DataFrame,tphi:np.ndarray,target:pd.DataFrame):
    selected0=pd.read_csv(D0/"05_selection_manifests.csv"); scores=pd.read_csv(OUT/"02_candidate_structural_scores.csv")
    p90=float(pd.read_csv(C0/"02_hardness_statistics.csv").pooled_p90_threshold.dropna().iloc[0]); p95=float(pd.read_csv(C0/"02_hardness_statistics.csv").pooled_p95_threshold.dropna().iloc[0])
    ref_quota=p6a.capacities(target.instance_id.astype(str).tolist()); target_mass=tphi[np.repeat(np.arange(48),ref_quota)]
    target_pair=np.sqrt(p6a.squared_distance_matrix(tphi,tphi)); np.fill_diagonal(target_pair,np.inf); radius=float(np.median(target_pair.min(axis=1)))
    metrics=[]; op_rows=[]
    allsel=[]
    for rep in REPS:
        hist=selected0[(selected0.replicate.eq(rep))&selected0.selection.isin(["B","C"])].copy(); hist["arm"]=hist.pop("selection"); hist["selection_role"]="ALL"; hist["structure_ratio"]=np.nan
        hist=hist.merge(scores[scores.replicate.eq(rep)].drop(columns="replicate"),on=["instance_hash","qd_niche"],validate="many_to_one")
        new=sels[sels.replicate.eq(rep)].copy(); combined=pd.concat([hist,new],ignore_index=True,sort=False); allsel.append(combined)
        pool=pd.read_csv(D0/"04_candidate_manifest.csv"); pool=pool[pool.replicate.eq(rep)]
        ff=f40[f40.instance_hash.isin(pool.instance_hash)].set_index("instance_hash"); mu=ff[list(F40)].mean(); sd=ff[list(F40)].std(ddof=0).replace(0,1)
        for arm in ("B","C",*ARMS):
            z=combined[combined.arm.eq(arm)].copy(); phi=z[[f"phi__{x}" for x in TOP5]].to_numpy(float); cost=p6a.squared_distance_matrix(phi,tphi); match=p6a.exact_transport_lp(cost,ref_quota)
            near=np.sqrt(cost.min(axis=1)); refmin=np.sqrt(cost.min(axis=0)); refcounts=pd.Series(z.nearest_reference).value_counts(); rp=(refcounts/refcounts.sum()).to_numpy(float)
            fz=((ff.loc[z.instance_hash,list(F40)]-mu)/sd).to_numpy(float); nn=NearestNeighbors(n_neighbors=2,n_jobs=-1).fit(fz).kneighbors(fz,return_distance=True)[0][:,1]
            within=[]
            for _,g in z.groupby("qd_niche"):
                if len(g)>1:
                    x=g[[f"phi__{v}" for v in TOP5]].to_numpy(float); within.extend(NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x,return_distance=True)[0][:,1])
            for role,rz in [("ALL",z)]+([(x,z[z.selection_role.eq(x)]) for x in ("STRUCTURE","HARDNESS")] if arm in ARMS else []):
                lin,largest=eff(rz.lineage_root); roots,_=eff(rz.lineage_root); ops,opmax=eff(rz.operator_id); src=rz.source.astype(str)
                row={"replicate":rep,"arm":arm,"structure_ratio":RATIOS.get(arm,np.nan),"selection_role":role,"n":len(rz),"unique_n":rz.instance_hash.nunique(),"mean_true_m4_ce":rz.true_m4_ce.mean(),"median_true_m4_ce":rz.true_m4_ce.median(),"true_p90":rz.true_m4_ce.quantile(.9),"true_p95":rz.true_m4_ce.quantile(.95),"frozen_p90_tail_fraction":np.mean(rz.true_m4_ce>=p90),"frozen_p95_tail_fraction":np.mean(rz.true_m4_ce>=p95),"mean_predicted_m4_ce":rz.predicted_m4_ce.mean(),"prediction_bias":(rz.predicted_m4_ce-rz.true_m4_ce).mean(),"hardness_decile_hist":" ".join(map(str,np.bincount(rz.true_decile.astype(int)-1,minlength=10))),"mean_d_struct":rz.d_struct.mean(),"median_d_struct":rz.d_struct.median(),"nearest_reference_coverage":rz.nearest_reference.nunique(),"reference_entropy":float(-(rp*np.log(rp)).sum()) if role=="ALL" else np.nan,"largest_reference_share":float(rp.max()) if role=="ALL" else np.nan,"occupied_qd_niches":rz.qd_niche.nunique(),"effective_lineages":lin,"effective_roots":roots,"largest_lineage_fraction":largest,"effective_operators":ops,"largest_operator_fraction":opmax,"E0_fraction":np.mean(src.isin(["E0","both"])),"E1_fraction":np.mean(src.isin(["E1","both"])),"top5_ot_mean":float(match["edge_cost"].mean()) if role=="ALL" else np.nan,"top5_swd":p6a.sliced_wasserstein(phi,target_mass) if role=="ALL" else np.nan,"top5_energy":p6a.energy_distance_multivariate(phi,target_mass) if role=="ALL" else np.nan,"target_reference_neighborhood_coverage_n":int(np.sum(refmin<=radius)) if role=="ALL" else np.nan,"full_f40_nn_median":float(np.median(nn)) if role=="ALL" else np.nan,"within_niche_top5_nn_mean":float(np.mean(within)) if role=="ALL" and within else np.nan}
                metrics.append(row)
                for op,count in rz.operator_id.astype(str).value_counts().items(): op_rows.append({"replicate":rep,"arm":arm,"selection_role":role,"operator_id":op,"count":count,"share":count/len(rz)})
    frame=pd.DataFrame(metrics)
    # Post-selection control contrasts; never used to alter membership.
    base=frame[frame.selection_role.eq("ALL")].set_index(["replicate","arm"])
    frame["hardness_enrichment_over_B"]=frame.apply(lambda r:r.mean_true_m4_ce/base.at[(r.replicate,"B"),"mean_true_m4_ce"] if r.selection_role=="ALL" else np.nan,axis=1)
    frame["hardness_loss_vs_C"]=frame.apply(lambda r:base.at[(r.replicate,"C"),"mean_true_m4_ce"]-r.mean_true_m4_ce if r.selection_role=="ALL" else np.nan,axis=1)
    frame.to_csv(OUT/"05_population_metrics.csv",index=False); pd.DataFrame(op_rows).to_csv(OUT/"logs/operator_composition.csv",index=False)
    (OUT/"05_population_composition.md").write_text("# Population Composition\n\nTrue M4 CE is post-selection only. Population-level exact OT uses frozen 48-reference capacities; instance nearest-reference counts separately expose concentration. Full-F40 and within-niche Top5 NN are structural similarity proxies, not gradient redundancy. All three preregistered arms proceed to training regardless of this table.\n\n"+md(frame)+"\n")
    # Freeze training configs and complete source ledger.
    ints=pd.read_csv(OUT/"logs/dataset_integrity.csv"); cfg=protocol_gate()
    for rep,arm in JOB_ORDER:
        row=ints[(ints.replicate.eq(rep))&ints.arm.eq(arm)].iloc[0]; x={"run_id":f"P6D3_{rep}_{arm}","replicate":rep,"arm":arm,"structure_ratio":RATIOS[arm],"training_seed":TRAINING_SEED,"membership_sha256":row.membership_sha256,"dataset_bundle_sha256":sha256_file(bundle_path(rep,arm)),"m4_sha256":M4_SHA,"protocol_sha256":cfg["protocol_sha256"],"base_training_protocol":training_protocol()}; x["training_config_sha256"]=chash(x); write_json(OUT/"02_training_configs"/f"{rep}_{arm}.json",x)
    hgb_dir=RUN/"cycles/C04"; sources=[REQUEST,D0/"completion.json",D0/"04_candidate_manifest.csv",D0/"03_niche_target_relevance.csv",D0/"05_selection_manifests.csv",D1/"completion.json",D2/"completion.json",M4,P6A/"diagnostics/tsplib48_top5_representation.parquet",ROOT/"assets/qd/cvt/centroids.csv",ROOT/"assets/qd/normalization/normalization_A_P1P99.json",hgb_dir/"HGB_model_manifest.json",hgb_dir/"candidate_predictions.parquet"]
    sources += sorted((hgb_dir/"HGB_seed_models").glob("*.joblib"))
    sources += sorted((C0/"full_f40_cache").glob("*.parquet"))
    sources += [D1/"checkpoints"/f"{r}_{a}_M5_E256.tsp.best" for r in REPS for a in ("B","C")]
    sources += [D1/"logs"/f"{r}_{a}"/"training_integrity.json" for r in REPS for a in ("B","C")]
    sources += [C1/"dataset_manifests"/f"{r}_union.npz" for r in REPS]
    pd.DataFrame([{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sources]).to_csv(OUT/"logs/source_artifact_hashes.csv",index=False)
    # Required offline figures and deterministic role examples.
    allm=frame[frame.selection_role.eq("ALL")]
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3); ax.bar(x-.18,[2500,5000,7500],.36,label="STRUCTURE"); ax.bar(x+.18,[7500,5000,2500],.36,label="HARDNESS"); ax.set_xticks(x,ARMS); ax.set_ylabel("Selected role count"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figures/01_role_quotas.png",dpi=180); plt.close(fig)
    def grouped(metric,ylabel,path):
        fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
        for ax,rep in zip(axes,REPS):
            z=allm[allm.replicate.eq(rep)].set_index("arm").loc[["B","C",*ARMS]]; ax.bar(range(5),z[metric]); ax.set_xticks(range(5),["B","C",*ARMS]); ax.set_title(rep); ax.grid(axis="y",alpha=.2)
        axes[0].set_ylabel(ylabel); fig.tight_layout(); fig.savefig(OUT/"figures"/path,dpi=180); plt.close(fig)
    grouped("top5_ot_mean","Exact Top5 OT mean","02_top5_ot.png"); grouped("mean_true_m4_ce","Mean true M4 CE","03_true_m4_ce.png"); grouped("full_f40_nn_median","Median Full-F40 NN","04_f40_clustering.png")
    combined=pd.concat(allsel,ignore_index=True,sort=False); case_rows=[]
    fig,axes=plt.subplots(3,6,figsize=(18,9))
    for ri,rep in enumerate(REPS):
        ux,uh,ut=load_bundle(C1/"dataset_manifests"/f"{rep}_union.npz"); lookup={h:i for i,h in enumerate(uh)}
        for ai,arm in enumerate(ARMS):
            for oi,role in enumerate(("STRUCTURE","HARDNESS")):
                z=combined[(combined.replicate.eq(rep))&combined.arm.eq(arm)&combined.selection_role.eq(role)].sort_values(["d_struct","instance_hash"]); row=z.iloc[len(z)//2]; idx=lookup[str(row.instance_hash)]; xy=ux[idx]; tour=ut[idx]; ax=axes[ri,ai*2+oi]; cyc=np.r_[tour,tour[0]]; ax.plot(xy[cyc,0],xy[cyc,1],lw=.5); ax.scatter(xy[:,0],xy[:,1],s=3); ax.set_title(f"{rep} {arm} {role[0]}\nCE={row.true_m4_ce:.3f}, d={row.d_struct:.3f}",fontsize=8); ax.axis("off"); case_rows.append({"replicate":rep,"arm":arm,"role":role,"instance_hash":row.instance_hash,"d_struct":row.d_struct,"true_m4_ce":row.true_m4_ce})
    fig.tight_layout(); fig.savefig(OUT/"figures/cases/10_role_cases.png",dpi=180); plt.close(fig); pd.DataFrame(case_rows).to_csv(OUT/"logs/representative_cases.csv",index=False)
    return frame

def audit():
    started=time.perf_counter(); cfg,sels,f40,tphi,target=build_selection(); metrics=population_audit(sels,f40,tphi,target)
    result={"status":"PASS","datasets":9,"candidate_scores":int(pd.read_csv(OUT/"02_candidate_structural_scores.csv",usecols=["instance_hash"]).shape[0]),"selection_rows":len(sels),"population_metric_rows":len(metrics),"protocol_sha256":cfg["protocol_sha256"],"wall_s":time.perf_counter()-started}; write_json(OUT/"logs/audit.json",result); Status().update(phase="audit",status="PASS",completed_runs=0,total_runs=9); return result

def training_manifest():
    rows=[]
    for rep,arm in JOB_ORDER:
        p=run_dir(rep,arm)/"training_integrity.json"; rows.append(json.loads(p.read_text()) if p.exists() else {"replicate":rep,"arm":arm,"run_id":f"P6D3_{rep}_{arm}","status":"PENDING","training_seed":TRAINING_SEED})
    f=pd.DataFrame(rows); f.to_csv(OUT/"06_training_runs.csv",index=False); passed=int(f.status.eq("PASS").sum())
    (OUT/"06_training_integrity.md").write_text(f"# Training Integrity\n\nCompleted: {passed}/9. Every PASS run checks exact frozen input hashes, E000 tensor equality to formal M4, seed 2026, 256 epochs, 10,240 updates, finite weights and fixed E256 selection.\n")
    return f
d1.training_manifest=training_manifest

def train(rep,arm,device): return d1.train(rep,arm,device)
def evaluate(rep,arm,device): return d1.evaluate(rep,arm,device)
def infer_ce(rep,arm,device): return d1.infer_ce(rep,arm,device)

def detail_new(rep,arm,ds): return pd.read_parquet(run_dir(rep,arm)/"evaluation_work"/eval_label(rep,arm)/ds/"instance_metrics.parquet")
def detail_hist(rep,arm,ds): return pd.read_parquet(D1/"logs"/f"{rep}_{arm}"/"evaluation_work"/f"P6D1_{rep}_{arm}_M5"/ds/"instance_metrics.parquet")
def bootstrap(v,key):
    x=np.asarray(v,float); rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6D3|{BOOT_SEED}|{key}".encode()).digest()[:8],"little")); out=np.empty(BOOT_N)
    for i in range(BOOT_N): out[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(np.quantile(out,.025)),float(np.quantile(out,.975))

def finalize():
    started=time.perf_counter(); cfg=protocol_gate(); tr=training_manifest()
    if len(tr)!=9 or not tr.status.eq("PASS").all(): raise RuntimeError("training incomplete")
    for rep,arm in JOB_ORDER:
        if json.loads((run_dir(rep,arm)/"evaluation_integrity.json").read_text()).get("status")!="PASS": raise RuntimeError("evaluation incomplete")
        if json.loads((OUT/"hardness"/f"{rep}_{arm}_on_union.json").read_text()).get("status")!="PASS": raise RuntimeError("CE incomplete")
    hgb_dir=RUN/"cycles/C04"; sources=[REQUEST,D0/"completion.json",D0/"04_candidate_manifest.csv",D0/"03_niche_target_relevance.csv",D0/"05_selection_manifests.csv",D1/"completion.json",D2/"completion.json",M4,P6A/"diagnostics/tsplib48_top5_representation.parquet",ROOT/"assets/qd/cvt/centroids.csv",ROOT/"assets/qd/normalization/normalization_A_P1P99.json",hgb_dir/"HGB_model_manifest.json",hgb_dir/"candidate_predictions.parquet"]
    sources += sorted((hgb_dir/"HGB_seed_models").glob("*.joblib"))+sorted((C0/"full_f40_cache").glob("*.parquet"))+[C1/"dataset_manifests"/f"{r}_union.npz" for r in REPS]+[D1/"checkpoints"/f"{r}_{a}_M5_E256.tsp.best" for r in REPS for a in ("B","C")]+[D1/"logs"/f"{r}_{a}"/"training_integrity.json" for r in REPS for a in ("B","C")]
    pd.DataFrame([{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sources]).to_csv(OUT/"logs/source_artifact_hashes.csv",index=False)
    dev=set(pd.read_csv(ROOT/"target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str)); held=set(json.loads((ROOT/"target_aware_pilot/p1_heldout22_names.json").read_text())["names"])
    models=("M4","B","C",*ARMS); perparts=[]; evalrows=[]; statrows=[]; byrep={}
    for rep in REPS:
        frames={}
        for ds in DATASETS:
            pieces=[]
            for model in ("B","C",*ARMS):
                x=(detail_hist(rep,model,ds) if model in ("B","C") else detail_new(rep,model,ds))[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].copy()
                if not x.valid_tour.all(): raise RuntimeError(f"invalid tour {rep}/{model}/{ds}")
                pieces.append(x.drop(columns="valid_tour").rename(columns={"coordinate_hash":f"hash_{model}","n_nodes":f"n_{model}","optimal_cost_int":f"opt_{model}","gap_percent":f"gap_{model}"}))
            m=pieces[0]
            for x in pieces[1:]: m=m.merge(x,on="instance_id",validate="one_to_one")
            for model in ("C",*ARMS):
                if not (m.hash_B.eq(m[f"hash_{model}"])&m.n_B.eq(m[f"n_{model}"])&m.opt_B.eq(m[f"opt_{model}"])).all(): raise RuntimeError(f"benchmark drift {rep}/{ds}/{model}")
            m4=pd.read_parquet(M4_EVAL/ds/"instance_metrics.parquet")[["instance_id","coordinate_hash","gap_percent"]].rename(columns={"coordinate_hash":"hash_M4","gap_percent":"gap_M4"}); m=m.merge(m4,on="instance_id",validate="one_to_one")
            if not m.hash_B.eq(m.hash_M4).all(): raise RuntimeError(f"M4 drift {ds}")
            f=pd.DataFrame({"replicate":rep,"dataset":ds,"instance_id":m.instance_id.astype(str),"coordinate_hash":m.hash_B,"n_nodes":m.n_B.astype(int),"optimal_cost_int":m.opt_B.astype(np.int64)})
            for model in models: f[f"gap_{model}"]=m[f"gap_{model}"]
            frames[ds]=f; perparts.append(f)
        original=frames["OriginalTSPLIB70"]
        for name,names in (("TSPLIB48",dev),("TSPLIB22",held)):
            f=original[original.instance_id.isin(names)].copy(); f["dataset"]=name
            if len(f)!=len(names): raise RuntimeError(f"split integrity {name}")
            frames[name]=f; perparts.append(f)
        byrep[rep]=frames
        for ds,f in frames.items():
            for model in models:
                cp=M4_SHA if model=="M4" else json.loads(((D1 if model in ("B","C") else OUT)/"logs"/f"{rep}_{model}"/"training_integrity.json").read_text())["checkpoint_sha256"]
                evalrows.append({"replicate":rep,"model":model,"dataset":ds,"n_instances":len(f),"mean_gap":f[f"gap_{model}"].mean(),"checkpoint_sha256":cp})
            for arm in ARMS:
                for comp in ("M4","B","C",*[x for x in ARMS if x!=arm]):
                    v=(f[f"gap_{comp}"]-f[f"gap_{arm}"]).to_numpy(float); lo,hi=bootstrap(v,f"{rep}|{ds}|{arm}|{comp}"); reg=np.clip(-v,0,None); top=np.sort(reg)[-max(1,int(np.ceil(.1*len(v)))):].sum()
                    statrows.append({"replicate":rep,"arm":arm,"comparator":comp,"dataset":ds,"stratum":"ALL","n":len(v),"mean_gap_arm":f[f"gap_{arm}"].mean(),"mean_gap_comparator":f[f"gap_{comp}"].mean(),"mean_delta_comparator_minus_arm":v.mean(),"median_delta":np.median(v),"sample_sd_delta":np.std(v,ddof=1),"paired_ci95_low":lo,"paired_ci95_high":hi,"arm_win_count":int((v>0).sum()),"arm_win_fraction":np.mean(v>0),"worst10pct_share_of_regression":top/reg.sum() if reg.sum()>0 else 0})
        for model in models: evalrows.append({"replicate":rep,"model":model,"dataset":"GAP5","n_instances":sum(len(frames[x]) for x in GAP5),"mean_gap":np.mean([frames[x][f"gap_{model}"].mean() for x in GAP5]),"checkpoint_sha256":M4_SHA if model=="M4" else json.loads(((D1 if model in ("B","C") else OUT)/"logs"/f"{rep}_{model}"/"training_integrity.json").read_text())["checkpoint_sha256"]})
        for arm in ARMS:
            for comp in ("M4","B","C",*[x for x in ARMS if x!=arm]):
                vals=[frames[x][f"gap_{comp}"].to_numpy(float)-frames[x][f"gap_{arm}"].to_numpy(float) for x in GAP5]; rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6D3|GAP5|{rep}|{arm}|{comp}".encode()).digest()[:8],"little")); boots=np.empty(BOOT_N)
                for i in range(BOOT_N): boots[i]=np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in vals])
                means=np.asarray([v.mean() for v in vals]); statrows.append({"replicate":rep,"arm":arm,"comparator":comp,"dataset":"GAP5","stratum":"ALL","n":sum(map(len,vals)),"mean_gap_arm":np.mean([frames[x][f"gap_{arm}"].mean() for x in GAP5]),"mean_gap_comparator":np.mean([frames[x][f"gap_{comp}"].mean() for x in GAP5]),"mean_delta_comparator_minus_arm":means.mean(),"median_delta":np.median(means),"sample_sd_delta":np.std(means,ddof=1),"paired_ci95_low":np.quantile(boots,.025),"paired_ci95_high":np.quantile(boots,.975),"arm_win_count":np.nan,"arm_win_fraction":np.nan,"worst10pct_share_of_regression":np.nan})
    per=pd.concat(perparts,ignore_index=True); per.to_csv(OUT/"07_per_instance_gap.csv",index=False); evaluations=pd.DataFrame(evalrows); evaluations.to_csv(OUT/"07_greedy_evaluation.csv",index=False); paired=pd.DataFrame(statrows); paired.to_csv(OUT/"08_paired_statistics.csv",index=False)
    # Frozen cross-population teacher-forced CE; no training or membership changes.
    selected0=pd.read_csv(D0/"05_selection_manifests.csv",usecols=["replicate","selection","instance_hash"]); selected3=pd.read_csv(OUT/"04_selection_manifests.csv",usecols=["replicate","arm","instance_hash"])
    matrix=[]; ceinst=[]; data_arms=("B","C",*ARMS,"E0","E1")
    for rep in REPS:
        union=pd.read_parquet(C1/"dataset_manifests"/f"{rep}_union.parquet",columns=["instance_hash","m4_ce"]).set_index("instance_hash").m4_ce
        maps={m:pd.read_parquet((D1 if m in ("B","C") else OUT)/"hardness"/f"{rep}_{m}_on_union.parquet").set_index("instance_hash").loss_raw for m in ("B","C",*ARMS)}
        for data_arm in data_arms:
            if data_arm in ("B","C"): hh=selected0[(selected0.replicate.eq(rep))&selected0.selection.eq(data_arm)].instance_hash.astype(str).tolist()
            elif data_arm in ARMS: hh=selected3[(selected3.replicate.eq(rep))&selected3.arm.eq(data_arm)].instance_hash.astype(str).tolist()
            else: hh=load_bundle(B6/"labels"/f"{rep}_{data_arm}"/"labeled_bundle.npz")[1]
            m4=union.loc[hh].to_numpy(float); matrix.append({"replicate":rep,"model":"M4","dataset_arm":data_arm,"n":len(hh),"mean_ce":m4.mean(),"median_ce":np.median(m4),"mean_gain_M4_minus_model":0.0,"improvement_fraction":np.nan})
            for model,s in maps.items():
                val=s.loc[hh].to_numpy(float); gain=m4-val; matrix.append({"replicate":rep,"model":model,"dataset_arm":data_arm,"n":len(hh),"mean_ce":val.mean(),"median_ce":np.median(val),"mean_gain_M4_minus_model":gain.mean(),"improvement_fraction":np.mean(gain>0)})
                ceinst.extend({"replicate":rep,"model":model,"dataset_arm":data_arm,"instance_hash":h,"m4_ce":x,"model_ce":y,"gain_m4_minus_model":x-y} for h,x,y in zip(hh,m4,val))
    matrix=pd.DataFrame(matrix); matrix.to_csv(OUT/"09_cross_population_ce.csv",index=False); pd.DataFrame(ceinst).to_parquet(OUT/"logs/cross_population_ce_per_instance.parquet",index=False)
    (OUT/"09_cross_population_ce.md").write_text("# Cross-Population CE Transfer\n\nPositive gain is M4 CE minus M5 CE. Own-population cells overlap training and demonstrate optimization, not independent generalization. All values use existing expert tours and frozen checkpoints.\n\n"+md(matrix)+"\n")
    pop=pd.read_csv(OUT/"05_population_metrics.csv"); popall=pop[pop.selection_role.eq("ALL")]; trade=popall.merge(evaluations[evaluations.dataset.isin(["OriginalTSPLIB70","TSPLIB48","TSPLIB22","GAP5"])],left_on=["replicate","arm"],right_on=["replicate","model"],how="left")
    rel=popall.merge(evaluations[evaluations.dataset.eq("OriginalTSPLIB70")][["replicate","model","mean_gap"]],left_on=["replicate","arm"],right_on=["replicate","model"],validate="one_to_one").rename(columns={"mean_gap":"original70_gap"}); rel.to_csv(OUT/"logs/structure_hardness_training_relationship.csv",index=False)
    show=rel[["replicate","arm","structure_ratio","mean_true_m4_ce","frozen_p90_tail_fraction","frozen_p95_tail_fraction","mean_d_struct","top5_ot_mean","top5_swd","effective_lineages","original70_gap"]]
    orig=evaluations[evaluations.dataset.eq("OriginalTSPLIB70")].pivot(index="replicate",columns="model",values="mean_gap"); gap5=evaluations[evaluations.dataset.eq("GAP5")].pivot(index="replicate",columns="model",values="mean_gap")
    # Pre-registered decisions.
    monotone=[]
    for rep in REPS:
        z=popall[(popall.replicate.eq(rep))&popall.arm.isin(ARMS)].set_index("arm").loc[list(ARMS)]; monotone.append({"replicate":rep,"d_struct_nonincreasing":bool(np.all(np.diff(z.mean_d_struct)<=1e-12)),"ot_nonincreasing":bool(np.all(np.diff(z.top5_ot_mean)<=1e-12)),"true_ce_nonincreasing":bool(np.all(np.diff(z.mean_true_m4_ce)<=1e-12)),"original70_monotone":bool(np.all(np.diff(orig.loc[rep,list(ARMS)])>=0) or np.all(np.diff(orig.loc[rep,list(ARMS)])<=0))})
    mono=pd.DataFrame(monotone); h1="SUPPORTED" if (mono.d_struct_nonincreasing&mono.ot_nonincreasing).all() else "MIXED" if (mono.d_struct_nonincreasing|mono.ot_nonincreasing).any() else "NOT_SUPPORTED"
    pidx=popall.set_index(["replicate","arm"]); preserve={a:sum(pidx.at[(r,a),"frozen_p90_tail_fraction"]>=pidx.at[(r,"B"),"frozen_p90_tail_fraction"] for r in REPS) for a in ARMS}; h2="SUPPORTED" if all(v==3 for v in preserve.values()) else "MIXED" if any(v==3 for v in preserve.values()) else "NOT_SUPPORTED"
    def wins(comp,table): return {a:int((table[a]<table[comp]).sum()) for a in ARMS}
    wc=wins("C",orig); wbc=wins("B",orig); wm4=wins("M4",orig); gc={a:int((gap5[a]<=gap5.C).sum()) for a in ARMS}; gb={a:int((gap5[a]<=gap5.B).sum()) for a in ARMS}
    h3="PRELIMINARILY_SUPPORTED" if any(wc[a]==3 and gc[a]==3 for a in ARMS) else "MIXED" if any(wc[a]>0 for a in ARMS) else "NOT_SUPPORTED"
    h4="SUPPORTED_PRELIMINARY" if any(wbc[a]==3 and gb[a]==3 for a in ARMS) else "MIXED" if any(wbc[a]>0 for a in ARMS) else "NOT_SUPPORTED"
    h5="SUPPORTED_PRELIMINARY" if any(v==3 for v in wm4.values()) else "PARTIAL" if any(v>0 for v in wm4.values()) else "NOT_SUPPORTED"
    offline_trade=(mono.d_struct_nonincreasing&mono.ot_nonincreasing&mono.true_ce_nonincreasing).all(); gapmono=mono.original70_monotone.all(); h6="SUPPORTED_EXPLORATORY" if offline_trade and gapmono else "MIXED" if offline_trade or (mono.d_struct_nonincreasing&mono.true_ce_nonincreasing).all() else "NOT_SUPPORTED"
    h7="PROMISING" if h3=="PRELIMINARILY_SUPPORTED" and h4=="SUPPORTED_PRELIMINARY" and h5!="NOT_SUPPORTED" else "NOT_SUPPORTED" if h3==h4=="NOT_SUPPORTED" and h5=="NOT_SUPPORTED" else "NOT_ESTABLISHED"
    mean_orig=orig[list(ARMS)].mean(); best=str(mean_orig.idxmin())
    (OUT/"08_ratio_comparison.md").write_text("# Ratio Comparison\n\nB is a random within-niche control and is not a 100% structure endpoint. Positive paired delta in `08_paired_statistics.csv` favors the D arm. TSPLIB48 is target-informed and TSPLIB22 has been repeatedly observed; neither is pristine.\n\n"+md(show)+"\n\nOriginal70:\n\n"+md(orig.reset_index())+"\n\nGAP5:\n\n"+md(gap5.reset_index())+"\n")
    (OUT/"10_structure_hardness_training_relationship.md").write_text("# Structure–Hardness–Training Relationship\n\nOffline true CE was not used for selection, and neither offline metric was used to eliminate an arm. Associations across three evolution replicates do not identify a causal optimal ratio.\n\n"+md(mono)+"\n\n"+md(show)+"\n")
    # Required post-training figures.
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,rep in zip(axes,REPS): z=orig.loc[rep,list(models)]; ax.bar(range(len(z)),z); ax.set_xticks(range(len(z)),z.index,rotation=45); ax.set_title(rep); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Original70 mean gap (%)"); fig.tight_layout(); fig.savefig(OUT/"figures/05_original70_gap.png",dpi=180); plt.close(fig)
    sub=evaluations[evaluations.dataset.isin(["TSPLIB48","TSPLIB22","GAP5"])]
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for ax,(rep,g) in zip(axes,sub.groupby("replicate")):
        pv=g.pivot(index="model",columns="dataset",values="mean_gap").loc[list(models)]; pv.plot.bar(ax=ax); ax.set_title(rep); ax.tick_params(axis="x",rotation=45)
    fig.tight_layout(); fig.savefig(OUT/"figures/06_subgroup_gap5.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,rep in zip(axes,REPS): ax.plot([.25,.5,.75],orig.loc[rep,list(ARMS)],marker="o"); ax.set_title(rep); ax.set_xlabel("Structure quota"); ax.set_ylabel("Original70 gap")
    fig.tight_layout(); fig.savefig(OUT/"figures/07_ratio_vs_gap.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,rep in zip(axes,REPS):
        z=popall[(popall.replicate.eq(rep))&popall.arm.isin(ARMS)].set_index("arm").loc[list(ARMS)]; ax.plot(z.structure_ratio,z.mean_true_m4_ce,marker="o",label="true CE"); ax2=ax.twinx(); ax2.plot(z.structure_ratio,z.top5_ot_mean,marker="s",color="tab:orange",label="OT"); ax.set_title(rep); ax.set_xlabel("Structure quota")
    fig.tight_layout(); fig.savefig(OUT/"figures/08_ratio_ce_ot.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    for ax,rep in zip(axes,REPS):
        p=matrix[matrix.replicate.eq(rep)].pivot(index="model",columns="dataset_arm",values="mean_ce").loc[list(models),list(data_arms)]; im=ax.imshow(p,aspect="auto"); ax.set_xticks(range(len(data_arms)),data_arms,rotation=45); ax.set_yticks(range(len(models)),models); ax.set_title(rep)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label="Mean CE"); fig.savefig(OUT/"figures/09_cross_population_ce.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    final=["# P6-D3 Final Research Conclusion","",f"`H1_STRUCTURAL_CONTROL: {h1}`",f"`H2_HARDNESS_PRESERVATION: {h2}`",f"`H3_TRAINING_VALUE_GAIN: {h3}`",f"`H4_BEATS_RANDOM_BASELINE: {h4}`",f"`H5_ABSOLUTE_IMPROVEMENT: {h5}`",f"`H6_RATIO_TRADEOFF: {h6}`",f"`H7_ONLINE_EVOLUTION_READINESS: {h7}`","",f"Across the three existing evolution replicates, the lowest mean Original70 gap among preregistered D arms is {best} ({mean_orig[best]:.6f}%). This is a development result over only three ratios and does not establish an optimal ratio.","",f"Original70 win counts versus C: {wc}; versus B: {wbc}; versus M4: {wm4}. GAP5 non-worse counts versus C: {gc}; versus B: {gb}.","",md(orig.reset_index()),"",f"Structural monotonicity and hardness trade-off diagnostics: {mono.to_dict(orient='records')}.","","The direct answer is determined by H3/H4: explicit role quotas improve stability only if one fixed ratio wins across all three evolution replicates without systematic GAP5 regression. Own-data CE is not treated as generalization. QD4 cross-size clipping and HGB root-overlap limits remain; PROMISING never means novel-offspring generalization is established.","","No ratio is retuned, and no P6-D4, evolution, Concorde, surrogate training or additional seed is started.","","`BLADE_P6D3_STRUCTURE_HARDNESS_QUOTA_SELECTION_COMPLETE`"]
    (OUT/"11_final_research_conclusion.md").write_text("\n".join(final)+"\n")
    monitor=pd.read_csv(OUT/"monitoring/hourly_status.csv") if (OUT/"monitoring/hourly_status.csv").exists() else pd.DataFrame(); train_s=tr.training_loop_wall_s.sum(); eval_s=sum(json.loads((run_dir(r,a)/"evaluation_integrity.json").read_text())["evaluation_wall_s"] for r,a in JOB_ORDER); ce_s=sum(json.loads((OUT/"hardness"/f"{r}_{a}_on_union.json").read_text())["wall_s"] for r,a in JOB_ORDER)
    (OUT/"12_runtime_resource_audit.md").write_text(f"# Runtime and Resource Audit\n\n- Training: {train_s/3600:.4f} GPU-hours.\n- Evaluation: {eval_s/3600:.4f} GPU-hours.\n- CE inference: {ce_s/3600:.4f} GPU-hours.\n- Monitoring snapshots: {len(monitor)}.\n- New mutation/evolution/Concorde/HGB fitting: 0.\n")
    completion={"status":"COMPLETE","final_label":"BLADE_P6D3_STRUCTURE_HARDNESS_QUOTA_SELECTION_COMPLETE","H1_STRUCTURAL_CONTROL":h1,"H2_HARDNESS_PRESERVATION":h2,"H3_TRAINING_VALUE_GAIN":h3,"H4_BEATS_RANDOM_BASELINE":h4,"H5_ABSOLUTE_IMPROVEMENT":h5,"H6_RATIO_TRADEOFF":h6,"H7_ONLINE_EVOLUTION_READINESS":h7,"best_mean_original70_arm":best,"protocol_sha256":cfg["protocol_sha256"],"final_report_sha256":sha256_file(OUT/"11_final_research_conclusion.md"),"completed_utc":now(),"wall_s":time.perf_counter()-started}; write_json(OUT/"completion.json",completion); Status().update(phase="complete",status="COMPLETE",completed_runs=9,total_runs=9); return completion

def log_failure(rep,arm,phase,attempt):
    with (OUT/"monitoring/failures.md").open("a") as f: f.write(f"## {now()} {rep}-{arm} {phase} attempt {attempt}\n\n```\n{traceback.format_exc()}\n```\n\n")

def refresh_source_ledger():
    hgb_dir=RUN/"cycles/C04"; sources=[REQUEST,D0/"completion.json",D0/"04_candidate_manifest.csv",D0/"03_niche_target_relevance.csv",D0/"05_selection_manifests.csv",D1/"completion.json",D2/"completion.json",M4,P6A/"diagnostics/tsplib48_top5_representation.parquet",ROOT/"assets/qd/cvt/centroids.csv",ROOT/"assets/qd/normalization/normalization_A_P1P99.json",hgb_dir/"HGB_model_manifest.json",hgb_dir/"candidate_predictions.parquet"]
    sources += sorted((hgb_dir/"HGB_seed_models").glob("*.joblib"))+sorted((C0/"full_f40_cache").glob("*.parquet"))+[C1/"dataset_manifests"/f"{r}_union.npz" for r in REPS]+[D1/"checkpoints"/f"{r}_{a}_M5_E256.tsp.best" for r in REPS for a in ("B","C")]+[D1/"logs"/f"{r}_{a}"/"training_integrity.json" for r in REPS for a in ("B","C")]
    frame=pd.DataFrame([{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sources]); frame.to_csv(OUT/"logs/source_artifact_hashes.csv",index=False); return {"status":"PASS","source_artifacts":len(frame),"ledger_sha256":sha256_file(OUT/"logs/source_artifact_hashes.csv")}
def quarantine(rep,arm,attempt):
    q=run_dir(rep,arm)/"quarantine"/f"attempt_{attempt}_{int(time.time())}"; q.mkdir(parents=True,exist_ok=True)
    for p in (run_dir(rep,arm)/"training_cycle",checkpoint(rep,arm),run_dir(rep,arm)/"training_integrity.json"):
        if p.exists(): shutil.move(str(p),str(q/p.name))

def runner(device):
    ensure_dirs(); lock=(OUT/"monitoring/runner.lock").open("w")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return {"status":"RUNNER_ALREADY_ACTIVE"}
    (OUT/"monitoring/runner.pid").write_text(str(os.getpid())+"\n"); Status().update(phase="runner_start",status="RUNNING",runner_pid=os.getpid(),total_runs=9)
    try:
        if not (OUT/"logs/audit.json").exists() or json.loads((OUT/"logs/audit.json").read_text()).get("status")!="PASS": audit()
        for index,(rep,arm) in enumerate(JOB_ORDER,1):
            Status().update(phase="training_queue",status="RUNNING",current_run=f"{rep}-{arm}",queue_index=index,completed_runs=int(training_manifest().status.eq("PASS").sum()),next_run=f"{rep}-{arm}")
            for attempt in range(3):
                try: train(rep,arm,device); break
                except Exception as exc:
                    log_failure(rep,arm,"training",attempt); fatal=any(x in str(exc).lower() for x in ("nan","inf","mismatch","drift","illegal","validity","integrity","quota"))
                    if fatal or attempt>=2: Status().update(status="BLOCKED",phase="training",current_run=f"{rep}-{arm}",error=str(exc)); raise
                    quarantine(rep,arm,attempt+1)
                    with (OUT/"monitoring/repair_actions.md").open("a") as f: f.write(f"## {now()} {rep}-{arm}\n\nTransient training failure; isolated incomplete output and restarted from formal M4 ({attempt+1}/2).\n\n")
        for phase,fn in (("evaluation",evaluate),("ce",infer_ce)):
            for rep,arm in JOB_ORDER:
                for attempt in range(3):
                    try: fn(rep,arm,device); break
                    except Exception as exc:
                        log_failure(rep,arm,phase,attempt)
                        if attempt>=2: Status().update(status="BLOCKED",phase=phase,current_run=f"{rep}-{arm}",error=str(exc)); raise
        return finalize()
    except Exception as exc: Status().update(status="BLOCKED",error=str(exc)); raise
    finally:
        try: (OUT/"monitoring/runner.pid").unlink()
        except FileNotFoundError: pass
        if (OUT/"completion.json").exists(): subprocess.run(["systemctl","--user","disable","--now","blade-p6d3-monitor.timer"],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def alive(pid):
    try: os.kill(pid,0); return True
    except (ProcessLookupError,PermissionError): return False
def monitor(initial=False):
    ensure_dirs(); lock=(OUT/"monitoring/monitor.lock").open("w")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return {"status":"SKIP_OVERLAP"}
    st=json.loads((OUT/"status.json").read_text()) if (OUT/"status.json").exists() else {}; pp=OUT/"monitoring/runner.pid"; pid=int(pp.read_text()) if pp.exists() else -1; running=pid>0 and alive(pid); current=str(st.get("current_run","")); epochs=updates=0; finite=True
    if "-" in current:
        rep,arm=current.rsplit("-",1); curve=run_dir(rep,arm)/"training_cycle/09_bq_training/training_loss_curve.csv"
        if curve.exists():
            try: c=pd.read_csv(curve); epochs=len(c); updates=int(c.optimizer_updates.sum()); finite=bool(np.isfinite(c.mean_training_loss).all())
            except Exception: finite=False
    gpu=""
    try: gpu=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20).stdout.strip().replace("\n","; ")
    except Exception as exc: gpu=str(exc)
    complete=(OUT/"completion.json").exists(); completed=int(training_manifest().status.eq("PASS").sum()) if (OUT/"logs/audit.json").exists() else 0
    snap={"timestamp_utc":now(),"initial_check":bool(initial),"runner_alive":running,"runner_pid":pid if running else "","phase":st.get("phase","UNKNOWN"),"current_run":current,"task_status":st.get("status","UNKNOWN"),"epochs":epochs,"updates":updates,"loss_finite":finite,"gpu_status":gpu,"disk_free_gib":shutil.disk_usage(OUT).free/2**30,"completed_runs":completed,"experiment_complete":complete}
    p=OUT/"monitoring/hourly_status.csv"; pd.DataFrame([snap]).to_csv(p,mode="a",header=not p.exists(),index=False); write_json(OUT/"monitoring/current_status.json",snap)
    with (OUT/"monitoring/hourly_monitor.log").open("a") as f: f.write(json.dumps(snap,sort_keys=True)+"\n")
    if not running and not complete and st.get("status")!="BLOCKED":
        rp=OUT/"monitoring/runner_restart_count.json"; v=json.loads(rp.read_text()) if rp.exists() else {"count":0}
        if v["count"]<2: v["count"]+=1; v["last_restart_utc"]=now(); write_json(rp,v); subprocess.run(["systemctl","--user","start","blade-p6d3-runner.service"],check=False)
    if complete: subprocess.run(["systemctl","--user","disable","--now","blade-p6d3-monitor.timer"],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return snap

def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("audit","train","evaluate","ce","finalize","runner","monitor","ledger")); p.add_argument("--replicate",choices=REPS); p.add_argument("--arm",choices=ARMS); p.add_argument("--device",default="cuda:1"); p.add_argument("--initial",action="store_true"); a=p.parse_args()
    if a.phase=="audit": result=audit()
    elif a.phase=="runner": result=runner(a.device)
    elif a.phase=="monitor": result=monitor(a.initial)
    elif a.phase=="ledger": result=refresh_source_ledger()
    elif a.phase=="finalize": result=finalize()
    else:
        if not a.replicate or not a.arm: p.error("--replicate and --arm required")
        result=train(a.replicate,a.arm,a.device) if a.phase=="train" else evaluate(a.replicate,a.arm,a.device) if a.phase=="evaluate" else infer_ce(a.replicate,a.arm,a.device)
    if result is not None: print(json.dumps(result,indent=2,default=str))
if __name__=="__main__": main()
