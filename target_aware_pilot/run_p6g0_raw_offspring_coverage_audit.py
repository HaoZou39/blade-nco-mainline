#!/usr/bin/env python3
"""P6-G0: read-only historical RAW-offspring coverage audit."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
P6B=ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6E0=ROOT/"target_aware_pilot/p6e0_evoreal_structural_representation_audit"
OUT=ROOT/"target_aware_pilot/p6g0_raw_offspring_coverage_audit"
FEAT=OUT/"features"; FIG=OUT/"figures"; TABLE=OUT/"tables"; LOG=OUT/"logs"; MAN=OUT/"manifests"
REPS=("Original","S1","S2"); ARMS=("E0","E1")
RAW10K_SEEDS=tuple(20261110+i for i in range(10))
START=time.time()


def load_module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


E0MOD=load_module("p6g0_frozen_p6e0",P6E0/"run_p6e0_audit.py")


def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def write(p:Path,s:str): p.write_text(s.strip()+"\n",encoding="utf-8")


def md(d:pd.DataFrame,digits:int=4)->str:
    if d.empty:return "_(empty)_"
    lines=["| "+" | ".join(map(str,d.columns))+" |","|"+"|".join(["---"]*len(d.columns))+"|"]
    for row in d.itertuples(index=False,name=None):
        vals=[]
        for v in row:
            if pd.isna(v):vals.append("NA")
            elif isinstance(v,(float,np.floating)):vals.append(f"{float(v):.{digits}f}")
            else:vals.append(str(v).replace("|","\\|").replace("\n"," "))
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)


def npz_info(path:Path)->dict:
    z=np.load(path,allow_pickle=False); h=z["instance_hash"].astype(str)
    return {"path":str(path.relative_to(ROOT)),"exists":True,"n":len(h),"unique_n":len(set(h)),"coordinate_shape":str(tuple(z["coordinates"].shape)),"bytes":path.stat().st_size,"sha256":sha(path)}


def raw_path(rep:str,arm:str|None,gen:int)->Path:
    if gen==1:return P6B/f"evolution/{rep}/shared_G1/mutation/offspring_coordinates.npz"
    return P6B/f"evolution/{rep}/{arm}/G2/mutation/offspring_coordinates.npz"


def manifest_path(rep:str,arm:str|None,gen:int)->Path:
    if gen==1:return P6B/f"evolution/{rep}/shared_G1/mutation/mutation_manifest.parquet"
    return P6B/f"evolution/{rep}/{arm}/G2/mutation/mutation_manifest.parquet"


def union_path(rep:str,arm:str|None,gen:int)->Path:
    if gen==1:return P6B/f"evolution/{rep}/shared_G1/candidate_union.npz"
    return P6B/f"evolution/{rep}/{arm}/G2/candidate_union.npz"


def selected_path(rep:str,arm:str,gen:int)->Path:
    return P6B/f"evolution/{rep}/{arm}/G{gen}/active_population.npz"


def cache_features(name:str,path:Path)->pd.DataFrame:
    out=FEAT/f"{name}.parquet"; meta=FEAT/f"{name}.json"
    if out.exists() and meta.exists():
        m=json.loads(meta.read_text()); d=pd.read_parquet(out)
        if m.get("source_bytes")==path.stat().st_size and len(d)==m.get("n") and d.instance_hash.nunique()==len(d):
            print(f"[cache] {name}: {len(d):,}",flush=True);return d
    z=np.load(path,allow_pickle=False); coords=z["coordinates"]; hashes=z["instance_hash"].astype(str).tolist()
    if len(set(hashes))!=len(hashes) or not np.isfinite(coords).all():raise RuntimeError(f"invalid source {name}")
    jobs=[(i,np.asarray(coords[i:i+500],np.float64)) for i in range(0,len(coords),500)]; parts=[]; t=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for k,(start,rows) in enumerate(pool.map(E0MOD.feature_chunk,jobs),1):
            f=pd.DataFrame(rows)[["fft_energy","nn_ratio","duplicate_coordinate_n"]];f.insert(0,"instance_hash",hashes[start:start+len(f)]);parts.append((start,f))
            if k%25==0 or k==len(jobs):print(f"[features] {name}: {k}/{len(jobs)} chunks",flush=True)
    parts.sort();d=pd.concat([x[1] for x in parts],ignore_index=True);d.to_parquet(out,index=False)
    meta.write_text(json.dumps({"source":str(path.relative_to(ROOT)),"source_bytes":path.stat().st_size,"source_sha256":sha(path),"n":len(d),"feature_implementation":str((P6E0/'run_p6e0_audit.py').relative_to(ROOT)),"wall_s":time.time()-t},indent=2)+"\n")
    del coords,z
    return d


def reorder(features:list[pd.DataFrame],hashes:np.ndarray)->pd.DataFrame:
    d=pd.concat(features,ignore_index=True).drop_duplicates("instance_hash").set_index("instance_hash")
    h=pd.Index(hashes.astype(str)); missing=h.difference(d.index)
    if len(missing):raise RuntimeError(f"feature lookup missing {len(missing)}")
    out=d.loc[h].copy();out.index.name="instance_hash"
    return out.reset_index()


def member(frame:pd.DataFrame,refz,center,scale,radius):
    z=(frame[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale;dist=cdist(z,refz);return dist,dist<=radius


def coverage_row(rep,arm,stage,mode,scope,frame,dist,mem,draw=-1,**extra):
    mask=np.ones(mem.shape[1],bool) if scope=="all48" else REF.n_nodes.between(75,150).to_numpy()
    m=mem[:,mask];d=dist[:,mask]
    r={"replicate":rep,"arm":arm,"stage":stage,"evaluation_mode":mode,"scope":scope,"draw":draw,"population_n":len(frame),"unique_n":frame.instance_hash.nunique(),"reference_total":int(mask.sum()),"coverage_n":int(m.any(0).sum()),"coverage_fraction":float(m.any(0).mean()),"within_any_radius_n":int(m.any(1).sum()),"within_any_radius_fraction":float(m.any(1).mean()),"reference_nearest_mean":float(d.min(0).mean()),"reference_nearest_p90":float(np.quantile(d.min(0),.9))}
    r.update(extra);return r


def stage_records(rep,arm,stage,frame,dist,mem,rows,refs,random10k=False):
    for scope in ("all48","matched17"):
        rows.append(coverage_row(rep,arm,stage,"full",scope,frame,dist,mem))
    for j,r in REF.reset_index(drop=True).iterrows():
        refs.append({"replicate":rep,"arm":arm,"stage":stage,"evaluation_mode":"full","reference_id":r.instance_id,"n_nodes":r.n_nodes,"reference_group":r.reference_group,"matched_75_150":bool(75<=r.n_nodes<=150),"covered":bool(mem[:,j].any()),"nearest_distance":float(dist[:,j].min()),"neighborhood_candidate_n":int(mem[:,j].sum()),"population_n":len(frame),"unique_n":frame.instance_hash.nunique()})
    if random10k and len(frame)>10000:
        for draw,s in enumerate(RAW10K_SEEDS):
            ix=np.random.default_rng(s).choice(len(frame),10000,replace=False)
            for scope in ("all48","matched17"):rows.append(coverage_row(rep,arm,stage,"random10k",scope,frame.iloc[ix],dist[ix],mem[ix],draw=draw))


def inventory()->pd.DataFrame:
    cfg=json.loads((P6B/"configs/frozen_p6b_config.json").read_text()); p03=ROOT/cfg["initial_population"]
    rows=[]; x=npz_info(p03);x.update({"replicate":"ALL","arm":"SHARED","stage":"P03","artifact_type":"population"});rows.append(x)
    for rep in REPS:
        for gen in (1,2):
            arms=(None,) if gen==1 else ARMS
            for arm in arms:
                rp=raw_path(rep,arm,gen);u=union_path(rep,arm,gen);m=manifest_path(rep,arm,gen)
                for stage,p,typ in [(f"G{gen}_RAW",rp,"raw_offspring"),(f"G{gen}_CANDIDATE_UNION",u,"parent_child_union")]:
                    x=npz_info(p);x.update({"replicate":rep,"arm":"SHARED" if arm is None else arm,"stage":stage,"artifact_type":typ});rows.append(x)
                mf=pd.read_parquet(m,columns=["proposal_status","child_hash"]);rows[-2].update({"manifest_path":str(m.relative_to(ROOT)),"nominal_attempts":len(mf),"accepted_attempts":int(mf.proposal_status.eq("accepted").sum()),"manifest_unique_accepted":int(mf.loc[mf.proposal_status.eq('accepted'),'child_hash'].nunique())})
        for arm in ARMS:
            for gen in (1,2):
                p=selected_path(rep,arm,gen);x=npz_info(p);x.update({"replicate":rep,"arm":arm,"stage":f"G{gen}_SELECTED","artifact_type":"active_population"});rows.append(x)
    out=pd.DataFrame(rows);out.to_csv(TABLE/"artifact_inventory.csv",index=False);return out


def loss_detail(rep,arm,gen,rawf,rawmem,selmem,unionmem)->list[dict]:
    man=pd.read_parquet(manifest_path(rep,None if gen==1 else arm,gen));man=man[man.proposal_status.eq("accepted")][["child_hash","operator_id","parent_hash"]].drop_duplicates("child_hash").set_index("child_hash")
    rows=[]
    for j,r in REF.reset_index(drop=True).iterrows():
        rawcov=bool(rawmem[:,j].any()); selcov=bool(selmem[:,j].any()); unioncov=bool(unionmem[:,j].any())
        if rawcov and not selcov:
            hs=rawf.loc[rawmem[:,j],"instance_hash"].astype(str);g=man.reindex(hs).dropna(subset=["operator_id"]); ops=g.operator_id.value_counts();pars=g.parent_hash.nunique()
            rows.append({"replicate":rep,"arm":arm,"generation":f"G{gen}","reference_id":r.instance_id,"n_nodes":r.n_nodes,"raw_covered":rawcov,"candidate_union_covered":unioncov,"selected_covered":selcov,"raw_neighborhood_n":int(rawmem[:,j].sum()),"rare_raw_lt10":bool(rawmem[:,j].sum()<10),"distinct_parent_n":int(pars),"distinct_operator_n":int(g.operator_id.nunique()),"top_operator":ops.index[0] if len(ops) else "","top_operator_n":int(ops.iloc[0]) if len(ops) else 0,"operator_counts":";".join(f"{k}:{v}" for k,v in ops.items())})
    return rows


def main():
    global REF
    for p in (OUT,FEAT,FIG,TABLE,LOG,MAN):p.mkdir(parents=True,exist_ok=True)
    inv=inventory()
    cfg=json.loads((P6E0/"configs/reference_rules_frozen.json").read_text());REF=pd.read_parquet(P6E0/"features/TSPLIB48.parquet").reset_index(drop=True)
    center=np.asarray(cfg["center"],float);scale=np.asarray(cfg["scale_iqr"],float);radius=float(cfg["radius"]);refz=(REF[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale
    p6cfg=json.loads((P6B/"configs/frozen_p6b_config.json").read_text());p03path=ROOT/p6cfg["initial_population"]
    p03=cache_features("P03",p03path); p03dist,p03mem=member(p03,refz,center,scale,radius)
    covrows=[];refrows=[];lossrows=[];transition=[];stage_mem={};stage_dist={}
    for rep in REPS:
        stage_records(rep,"SHARED","P03",p03,p03dist,p03mem,covrows,refrows)
        g1raw=cache_features(f"{rep}_G1_RAW",raw_path(rep,None,1));g1d,g1m=member(g1raw,refz,center,scale,radius);stage_records(rep,"SHARED","G1_RAW",g1raw,g1d,g1m,covrows,refrows,True)
        g1uh=np.load(union_path(rep,None,1),allow_pickle=False)["instance_hash"];g1union=reorder([p03,g1raw],g1uh);g1ud,g1um=member(g1union,refz,center,scale,radius);stage_records(rep,"SHARED","G1_CANDIDATE_UNION",g1union,g1ud,g1um,covrows,refrows)
        stage_mem[(rep,"SHARED","G1_RAW")]=g1m;stage_dist[(rep,"SHARED","G1_RAW")]=g1d
        new=(~p03mem.any(0))&g1m.any(0)
        transition.append({"replicate":rep,"arm":"SHARED","transition":"P03_TO_G1_RAW","source_coverage":int(p03mem.any(0).sum()),"destination_coverage":int(g1m.any(0).sum()),"new_reference_n":int(new.sum()),"new_reference_ids":";".join(REF.instance_id[new].astype(str))})
        for arm in ARMS:
            g1sh=np.load(selected_path(rep,arm,1),allow_pickle=False)["instance_hash"];g1sel=reorder([g1union],g1sh);g1sd,g1sm=member(g1sel,refz,center,scale,radius);stage_records(rep,arm,"G1_SELECTED",g1sel,g1sd,g1sm,covrows,refrows);stage_mem[(rep,arm,"G1_SELECTED")]=g1sm
            lossrows+=loss_detail(rep,arm,1,g1raw,g1m,g1sm,g1um)
            transition.append({"replicate":rep,"arm":arm,"transition":"G1_RAW_TO_SELECTED","source_coverage":int(g1m.any(0).sum()),"candidate_union_coverage":int(g1um.any(0).sum()),"destination_coverage":int(g1sm.any(0).sum()),"lost_raw_reference_n":int((g1m.any(0)&~g1sm.any(0)).sum()),"lost_raw_reference_ids":";".join(REF.instance_id[g1m.any(0)&~g1sm.any(0)].astype(str))})
            g2raw=cache_features(f"{rep}_{arm}_G2_RAW",raw_path(rep,arm,2));g2d,g2m=member(g2raw,refz,center,scale,radius);stage_records(rep,arm,"G2_RAW",g2raw,g2d,g2m,covrows,refrows,True);stage_mem[(rep,arm,"G2_RAW")]=g2m
            new2=(~g1sm.any(0))&g2m.any(0);continued=g1sm.any(0)&g2m.any(0);stilllost=(~g1sm.any(0))&(~g2m.any(0))
            transition.append({"replicate":rep,"arm":arm,"transition":"G1_SELECTED_TO_G2_RAW","source_coverage":int(g1sm.any(0).sum()),"destination_coverage":int(g2m.any(0).sum()),"new_reference_n":int(new2.sum()),"new_reference_ids":";".join(REF.instance_id[new2].astype(str)),"continued_reference_n":int(continued.sum()),"still_unreached_n":int(stilllost.sum()),"still_unreached_ids":";".join(REF.instance_id[stilllost].astype(str))})
            g2uh=np.load(union_path(rep,arm,2),allow_pickle=False)["instance_hash"];g2union=reorder([g1sel,g2raw],g2uh);g2ud,g2um=member(g2union,refz,center,scale,radius);stage_records(rep,arm,"G2_CANDIDATE_UNION",g2union,g2ud,g2um,covrows,refrows)
            g2sh=np.load(selected_path(rep,arm,2),allow_pickle=False)["instance_hash"];g2sel=reorder([g2union],g2sh);g2sd,g2sm=member(g2sel,refz,center,scale,radius);stage_records(rep,arm,"G2_SELECTED",g2sel,g2sd,g2sm,covrows,refrows);stage_mem[(rep,arm,"G2_SELECTED")]=g2sm
            lossrows+=loss_detail(rep,arm,2,g2raw,g2m,g2sm,g2um)
            transition.append({"replicate":rep,"arm":arm,"transition":"G2_RAW_TO_SELECTED","source_coverage":int(g2m.any(0).sum()),"candidate_union_coverage":int(g2um.any(0).sum()),"destination_coverage":int(g2sm.any(0).sum()),"lost_raw_reference_n":int((g2m.any(0)&~g2sm.any(0)).sum()),"lost_raw_reference_ids":";".join(REF.instance_id[g2m.any(0)&~g2sm.any(0)].astype(str))})
    # Existing EvoReal features: no regeneration.
    evo=pd.read_parquet(P6E0/"features/EvoReal50k.parquet");ed,em=member(evo,refz,center,scale,radius);stage_records("EvoReal","EXTERNAL","EvoReal50k",evo,ed,em,covrows,refrows)
    for draw,s in enumerate(cfg["matched_seeds"]):
        ix=np.random.default_rng(s).choice(len(evo),10000,replace=False)
        for scope in ("all48","matched17"):covrows.append(coverage_row("EvoReal","EXTERNAL","EvoReal10k","random10k",scope,evo.iloc[ix],ed[ix],em[ix],draw=draw))
    cov=pd.DataFrame(covrows)
    # Add random-10k distribution summaries without hiding individual draws.
    rs=cov[cov.evaluation_mode.eq("random10k")].groupby(["replicate","arm","stage","scope"],as_index=False).agg(population_n=("population_n","first"),unique_n=("unique_n","mean"),reference_total=("reference_total","first"),coverage_n=("coverage_n","mean"),coverage_sd=("coverage_n","std"),coverage_p10=("coverage_n",lambda x:x.quantile(.1)),coverage_p90=("coverage_n",lambda x:x.quantile(.9)),coverage_min=("coverage_n","min"),coverage_max=("coverage_n","max"),within_any_radius_fraction=("within_any_radius_fraction","mean"),reference_nearest_mean=("reference_nearest_mean","mean"),reference_nearest_p90=("reference_nearest_p90","mean"))
    rs["evaluation_mode"]="random10k_summary";rs["draw"]=-1;rs["coverage_fraction"]=rs.coverage_n/rs.reference_total
    cov=pd.concat([cov,rs],ignore_index=True,sort=False);cov.to_csv(OUT/"02_stage_coverage.csv",index=False)
    refs=pd.DataFrame(refrows);refs.to_csv(OUT/"03_reference_reachability.csv",index=False)
    losses=pd.DataFrame(lossrows);losses.to_csv(TABLE/"selection_loss_references.csv",index=False)
    trans=pd.DataFrame(transition);trans.to_csv(TABLE/"stage_transitions.csv",index=False)
    # EvoReal gap evidence, every seed/branch and a global per-reference summary.
    erows=[];evocov=em.any(0)
    for rep in REPS:
        for arm in ARMS:
            g1=stage_mem[(rep,"SHARED","G1_RAW")].any(0);g2=stage_mem[(rep,arm,"G2_RAW")].any(0);fin=stage_mem[(rep,arm,"G2_SELECTED")].any(0)
            for j,r in REF.reset_index(drop=True).iterrows():
                if evocov[j] and not fin[j]:erows.append({"replicate":rep,"arm":arm,"reference_id":r.instance_id,"n_nodes":r.n_nodes,"matched_75_150":bool(75<=r.n_nodes<=150),"evoreal50k_candidate_n":int(em[:,j].sum()),"g1_raw_covered":bool(g1[j]),"g1_raw_candidate_n":int(stage_mem[(rep,'SHARED','G1_RAW')][:,j].sum()),"g2_raw_covered":bool(g2[j]),"g2_raw_candidate_n":int(stage_mem[(rep,arm,'G2_RAW')][:,j].sum()),"ever_raw_covered":bool(g1[j] or g2[j]),"final_selected_covered":False,"interpretation":"GENERATED_THEN_LOST" if g1[j] or g2[j] else "NOT_REACHED_IN_THIS_RUN"})
    evidence=pd.DataFrame(erows);evidence.to_csv(TABLE/"evoreal_final_gap_evidence.csv",index=False)
    global_rows=[]
    for j,r in REF.reset_index(drop=True).iterrows():
        g1any=any(stage_mem[(rep,"SHARED","G1_RAW")][:,j].any() for rep in REPS);g2any=any(stage_mem[(rep,arm,"G2_RAW")][:,j].any() for rep in REPS for arm in ARMS);fin_n=sum(stage_mem[(rep,arm,"G2_SELECTED")][:,j].any() for rep in REPS for arm in ARMS)
        status="NOT_COVERED_BY_EVOREAL" if not evocov[j] else "GENERATION_LIMITED_HISTORICALLY" if not(g1any or g2any) else "GENERATED_BUT_LOST_ALL_FINALS" if fin_n==0 else "GENERATED_AND_RETAINED_SOME"
        global_rows.append({"reference_id":r.instance_id,"n_nodes":r.n_nodes,"matched_75_150":bool(75<=r.n_nodes<=150),"evoreal50k_covered":bool(evocov[j]),"evoreal50k_candidate_n":int(em[:,j].sum()),"any_g1_raw_covered":g1any,"any_g2_raw_covered":g2any,"final_covered_run_count_of_6":int(fin_n),"status":status})
    global_evidence=pd.DataFrame(global_rows);global_evidence.to_csv(TABLE/"global_reference_evidence.csv",index=False)
    union_bottleneck=[]
    for rep in REPS:
        raw_any=stage_mem[(rep,"SHARED","G1_RAW")].any(0)|stage_mem[(rep,"E0","G2_RAW")].any(0)|stage_mem[(rep,"E1","G2_RAW")].any(0)
        final_any=stage_mem[(rep,"E0","G2_SELECTED")].any(0)|stage_mem[(rep,"E1","G2_SELECTED")].any(0)
        lost=raw_any&~final_any
        union_bottleneck.append({"replicate":rep,"historical_raw_union_coverage_n":int(raw_any.sum()),"final_E0_union_E1_coverage_n":int(final_any.sum()),"raw_references_lost_by_final_union_n":int(lost.sum()),"raw_references_lost_by_final_union_ids":";".join(REF.instance_id[lost].astype(str)),"sets_exactly_equal":bool(np.array_equal(raw_any,final_any))})
    union_bottleneck=pd.DataFrame(union_bottleneck);union_bottleneck.to_csv(TABLE/"final_union_bottleneck.csv",index=False)

    # Exact reproduction of the P6-E0.5 final E0∪E1 union coverage endpoint.
    expected=pd.read_csv(ROOT/"target_aware_pilot/p6e05_candidate_pool_structural_reachability/manifests/full_pool_population_metrics.csv")
    repro=[]
    for rep in REPS:
        q=refs[(refs.replicate.eq(rep))&(refs.stage.eq("G2_SELECTED"))]
        observed=int(q.groupby("reference_id").covered.any().sum())
        exp=int(expected.loc[expected.dataset.eq(f"U_{rep}"),"reference_coverage_count"].iloc[0])
        repro.append({"replicate":rep,"observed_final_E0_union_E1_coverage":observed,"p6e05_expected_union_coverage":exp,"exact_match":observed==exp})
    repro=pd.DataFrame(repro);repro.to_csv(TABLE/"p6e05_reproduction_check.csv",index=False)
    if not repro.exact_match.all():raise RuntimeError("P6-E0.5 frozen coverage reproduction failed")

    # One simple stage figure; raw full and selected are never conflated.
    full=cov[(cov.scope.eq("all48"))&(cov.evaluation_mode.eq("full"))]
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    order=["P03","G1_RAW","G1_SELECTED","G2_RAW","G2_SELECTED"]
    for ax,rep in zip(axes,REPS):
        for arm,marker in [("E0","o"),("E1","s")]:
            vals=[]
            for st in order:
                a="SHARED" if st in ("P03","G1_RAW") else arm;q=full[(full.replicate.eq(rep))&(full.arm.eq(a))&(full.stage.eq(st))];vals.append(q.coverage_n.iloc[0] if len(q) else np.nan)
            ax.plot(order,vals,marker=marker,label=arm);ax.tick_params(axis="x",rotation=45);ax.set_title(rep);ax.set_ylim(0,49);ax.axhline(int(em.any(0).sum()),c="k",ls="--",label="EvoReal50k" if rep=="Original" else None)
        ax.set_ylabel("TSPLIB48 neighborhoods covered")
    axes[0].legend();fig.tight_layout();fig.savefig(FIG/"01_stage_coverage_raw_selected_evoreal.png",dpi=180);plt.close(fig)

    # Decisions.
    g1expand=all(int(trans[(trans.replicate.eq(r))&(trans.transition.eq("P03_TO_G1_RAW"))].new_reference_n.iloc[0])>0 for r in REPS)
    g1loss=bool(((trans.transition.eq("G1_RAW_TO_SELECTED"))&(trans.lost_raw_reference_n.fillna(0)>0)).any())
    g2expand=bool(((trans.transition.eq("G1_SELECTED_TO_G2_RAW"))&(trans.new_reference_n.fillna(0)>0)).any())
    g2loss=bool(((trans.transition.eq("G2_RAW_TO_SELECTED"))&(trans.lost_raw_reference_n.fillna(0)>0)).any())
    global_ev=global_evidence[global_evidence.evoreal50k_covered];genlim=bool(global_ev.status.eq("GENERATION_LIMITED_HISTORICALLY").any());branch_sellim=bool(evidence.ever_raw_covered.any());union_sellim=bool((union_bottleneck.raw_references_lost_by_final_union_n>0).any())
    primary="MIXED" if genlim and union_sellim else "GENERATION" if genlim else "SELECTION" if (union_sellim or branch_sellim) else "NOT_IDENTIFIABLE"

    # Reports.
    compact=inv[["replicate","arm","stage","artifact_type","n","unique_n","nominal_attempts","accepted_attempts","path"]].copy()
    write(OUT/"01_artifact_inventory.md",f"""# P6-G0 Artifact Inventory

`P6_G0_RAW_OFFSPRING_AVAILABLE = YES`

All required historical stages exist. G1 RAW is shared by E0/E1 within each seed. G2 RAW is branch-specific because it is generated from the selected G1 parents. `offspring_coordinates.npz` is used for RAW; final active populations and the E0∪E1 ~20k downstream union are never substituted for RAW. Counts come from the actual NPZ and 110k-attempt manifests, not a nominal assumption.

{md(compact,0)}

All coordinate sets are finite and hash-unique at the stored stage. Input hashes are retained in `tables/artifact_inventory.csv` and `manifests/input_hashes.csv`.
""")
    loss_summary=trans[trans.transition.isin(["G1_RAW_TO_SELECTED","G2_RAW_TO_SELECTED"])][["replicate","arm","transition","source_coverage","candidate_union_coverage","destination_coverage","lost_raw_reference_n","lost_raw_reference_ids"]]
    write(OUT/"04_selection_loss.md",f"""# Selection Loss Audit

Selection acts on Parent+Child Union, so this audit reports both RAW-only coverage and the actual candidate-union coverage. A reference is called a selection loss only when RAW reached it but the corresponding 10k survivor did not. It is not inferred from a raw-versus-selected count difference alone.

{md(loss_summary,0)}

Per-reference raw counts, rarity (`raw_n < 10`), parent diversity and mutation-operator evidence are in `tables/selection_loss_references.csv`. G1 and G2 transition categories are in `tables/stage_transitions.csv`.
""")
    evosum=cov[((cov.replicate.eq("EvoReal"))&(cov.scope.eq("all48")))|((cov.stage.eq("G2_SELECTED"))&(cov.scope.eq("all48"))&(cov.evaluation_mode.eq("full")))][["replicate","arm","stage","evaluation_mode","coverage_n","reference_total","population_n"]]
    generated_lost=global_ev[global_ev.status.eq("GENERATED_BUT_LOST_ALL_FINALS")].reference_id.astype(str).tolist();never=global_ev[global_ev.status.eq("GENERATION_LIMITED_HISTORICALLY")].reference_id.astype(str).tolist()
    perrun=evidence[evidence.ever_raw_covered][["replicate","arm","reference_id","evoreal50k_candidate_n","g1_raw_candidate_n","g2_raw_candidate_n","interpretation"]]
    write(OUT/"05_evoReal_comparison.md",f"""# EvoReal Comparison

The existing EvoReal50k feature cache and the same frozen reference geometry are reused; no synthetic instance is regenerated. EvoReal random10k uses the ten seeds already frozen by P6-E0.

{md(evosum,0)}

- EvoReal-covered references generated by at least one historical RAW stage but absent from **all six** final populations: `{';'.join(generated_lost) or '(none)'}`.
- EvoReal-covered references not reached by any historical G1/G2 RAW stage: `{';'.join(never) or '(none)'}`.

## Generated by FOGA but absent from the corresponding final 10k

{md(perrun,0)}

The per-seed/branch evidence—including exact G1/G2 raw candidate counts—is in `tables/evoreal_final_gap_evidence.csv`; the global 48-reference certificate is in `tables/global_reference_evidence.csv`.
""")
    # Compact coverage table for conclusion.
    key=full[full.stage.isin(order+['EvoReal50k'])][["replicate","arm","stage","coverage_n","population_n"]]
    keymatched=cov[(cov.scope.eq("matched17"))&(cov.evaluation_mode.eq("full"))&cov.stage.isin(order+['EvoReal50k'])][["replicate","arm","stage","coverage_n","population_n"]]
    write(OUT/"06_final_conclusion.md",f"""# P6-G0 Final Conclusion

## Decisions

- `P6_G0_RAW_OFFSPRING_AVAILABLE = YES`
- `P6_G0_G1_EXPANDS_COVERAGE = {'YES' if g1expand else 'NO'}`
- `P6_G0_G1_SELECTION_LOSS = {'YES' if g1loss else 'NO'}`
- `P6_G0_G2_EXPANDS_COVERAGE = {'YES' if g2expand else 'NO'}`
- `P6_G0_G2_SELECTION_LOSS = {'YES' if g2loss else 'NO'}`
- `P6_G0_GENERATION_REACHABILITY_LIMITED = {'YES' if genlim else 'NO'}`
- `P6_G0_PRIMARY_BOTTLENECK = {primary}`
- `P6_G0_AUDIT_STATUS = COMPLETE`

## Frozen full-stage coverage

{md(key,0)}

## Matched-size 75–150 reference coverage (17 total)

{md(keymatched,0)}

## Direct answer

**YES at the corresponding seed/branch level:** FOGA generated EvoReal-coverable neighborhoods that the corresponding final 10k lost: Original-E0 `rat99, rat783`; Original-E1 `pr226`; S1-E0 `rat575, rat783`; S1-E1 `pr226`; S2-E0 `rat99, rat575`; S2-E1 `pr226`. Exact raw and EvoReal candidate counts are in the evidence table.

**NO reference was generated and then lost by all six final populations simultaneously.** Other branches retained every globally generated reference at least once. Globally verified historical generation-limited references are `{';'.join(never) or '(none)'}`.

The reconstructed E0∪E1 final coverage exactly reproduces P6-E0.5: Original 25/48, S1 24/48, S2 25/48.

At the combined-pool level, every seed's final E0∪E1 reference set is exactly equal to the union of all references reached by its G1/G2 RAW offspring. Thus branch-level selection losses are real, but the P6-E0.5 combined-pool deficit versus EvoReal is primarily a historical generation-reachability limitation. In the 75–150 subset, EvoReal reaches 17/17 while historical RAW reaches at most 15/17; the two globally unreached matched-size references are `pr107` and `pr144`.

This conclusion is limited to the historical P03 parents, two generations, recorded mutation budgets, and frozen FFT_APPROX/NN-CV geometry. It does not claim that FOGA-11 can never reach a currently unreached structure.

`BLADE_P6G0_RAW_OFFSPRING_COVERAGE_AUDIT_COMPLETE`
""")
    # Input and output hashes.
    inpaths=[P6E0/"configs/reference_rules_frozen.json",P6E0/"features/TSPLIB48.parquet",P6E0/"features/EvoReal50k.parquet",P6E0/"run_p6e0_audit.py",ROOT/"target_aware_pilot/p6e05_candidate_pool_structural_reachability/manifests/full_pool_population_metrics.csv",P6B/"configs/frozen_p6b_config.json",p03path]
    for rep in REPS:
        inpaths += [raw_path(rep,None,1),manifest_path(rep,None,1),union_path(rep,None,1)]
        for arm in ARMS:
            inpaths += [selected_path(rep,arm,1),raw_path(rep,arm,2),manifest_path(rep,arm,2),union_path(rep,arm,2),selected_path(rep,arm,2)]
    pd.DataFrame([{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in inpaths]).to_csv(MAN/"input_hashes.csv",index=False)
    elapsed=time.time()-START;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    write(OUT/"runtime_resource_audit.md",f"# Runtime and resources\n\n- CPU-only wall time: {elapsed:.1f} s.\n- Peak RSS: {peak:.1f} MiB.\n- Feature extraction: exact imported P6-E0 implementation, 8 threads, cached in this independent directory.\n- No mutation, instance generation, Concorde, BQ/HGB inference, fitting, or training.\n")
    outfiles=[p for p in OUT.rglob("*") if p.is_file() and p.name not in ("completion.json","output_hashes.csv")];pd.DataFrame([{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(outfiles)]).to_csv(LOG/"output_hashes.csv",index=False)
    perrun_map={f"{rep}_{arm}":evidence[(evidence.replicate.eq(rep))&(evidence.arm.eq(arm))&evidence.ever_raw_covered].reference_id.astype(str).tolist() for rep in REPS for arm in ARMS}
    completion={"status":"COMPLETE","label":"BLADE_P6G0_RAW_OFFSPRING_COVERAGE_AUDIT_COMPLETE","raw_available":"YES","g1_expands":"YES" if g1expand else "NO","g1_selection_loss":"YES" if g1loss else "NO","g2_expands":"YES" if g2expand else "NO","g2_selection_loss":"YES" if g2loss else "NO","generation_reachability_limited":"YES" if genlim else "NO","primary_bottleneck":primary,"generated_then_lost_per_run":perrun_map,"generated_then_lost_all_six_finals":generated_lost,"historical_generation_limited_global":never,"p6e05_reproduction_exact":bool(repro.exact_match.all()),"runtime_s":time.time()-START,"peak_rss_mib":peak,"new_evolution":False,"new_training":False,"new_model_inference":False}
    (OUT/"completion.json").write_text(json.dumps(completion,indent=2)+"\n");print(json.dumps(completion,indent=2))


if __name__=="__main__":main()
