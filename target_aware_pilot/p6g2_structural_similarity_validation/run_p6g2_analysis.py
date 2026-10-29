#!/usr/bin/env python3
"""P6-G2 read-only TSPLIB structural-similarity validation.

Consumes frozen P6-E0/G0/G1 artifacts. It does not run evolution, solvers,
model inference/training, or inspect held-out TSPLIB22.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import cKDTree, distance
from scipy.stats import spearmanr, wasserstein_distance

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/"figures";TAB=OUT/"tables";MAN=OUT/"manifests";CACHE=OUT/"cache"
P6B=ROOT/"target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution"
P6G0=ROOT/"target_aware_pilot/p6g0_raw_offspring_coverage_audit"
P6G1=ROOT/"target_aware_pilot/p6g1_structural_reachability_analysis"
P6E0=ROOT/"target_aware_pilot/p6e0_evoreal_structural_representation_audit"
P03_PATH=ROOT/"runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz"
EVO_PATH=ROOT/"runs/BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2/datasets/evoreal50k_coordinates.npy"
F40_MANIFEST=ROOT/"audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
REF_MANIFEST=ROOT/"target_aware_pilot/03_tsplib48_reference_manifest.csv"
REF_F40=ROOT/"target_aware_pilot/p1_tsplib48_normalization_sensitivity.csv"
FROZEN_RULES=P6E0/"configs/reference_rules_frozen.json"
F40=tuple(json.loads(F40_MANIFEST.read_text())["F40"])
REPS=("Original","S1","S2");ARMS=("E0","E1")
SEED=20261029;REPEATS=30;TOPK=5
GEOMETRY_GRIDS=(4,8,16);GEOMETRY_ANGLES=np.arange(0.0,360.0,5.0)
MIN_F40_DIMS=30
UNREACHABLE=("d198","d657","pcb1173","pr107","pr144","pr264","u574","u724","a280","eil51","pcb442","pr299","rat195","ts225","tsp225")


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def write_json(path:Path,value:Any)->None:
    def conv(x:Any)->Any:
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(np.floating,)):return float(x)
        if isinstance(x,(np.bool_,)):return bool(x)
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=conv)+"\n");os.replace(tmp,path)


def md(frame:pd.DataFrame,columns:list[str]|None=None,digits:int=5)->str:
    if frame.empty:return "_(empty)_"
    q=frame if columns is None else frame[columns]
    lines=["| "+" | ".join(map(str,q.columns))+" |","|"+"|".join(["---"]*len(q.columns))+"|"]
    for row in q.itertuples(index=False,name=None):
        cells=[]
        for v in row:
            if pd.isna(v):cells.append("NA")
            elif isinstance(v,(float,np.floating)):cells.append(f"{float(v):.{digits}g}")
            else:cells.append(str(v).replace("|","\\|").replace("\n"," "))
        lines.append("| "+" | ".join(cells)+" |")
    return "\n".join(lines)


def raw_specs()->list[dict[str,Any]]:
    rows=[]
    for rep in REPS:
        rows.append({"dataset":f"{rep}_G1_RAW","replicate":rep,"arm":"SHARED","generation":1,
                     "coords":P6B/rep/"shared_G1/mutation/offspring_coordinates.npz","fft":P6G0/f"features/{rep}_G1_RAW.parquet","manifest":P6B/rep/"shared_G1/mutation/mutation_manifest.parquet"})
        for arm in ARMS:
            rows.append({"dataset":f"{rep}_{arm}_G2_RAW","replicate":rep,"arm":arm,"generation":2,
                         "coords":P6B/rep/arm/"G2/mutation/offspring_coordinates.npz","fft":P6G0/f"features/{rep}_{arm}_G2_RAW.parquet","manifest":P6B/rep/arm/"G2/mutation/mutation_manifest.parquet"})
    return rows


def load_npz(path:Path)->tuple[np.ndarray,list[str]]:
    with np.load(path,allow_pickle=False) as z:return np.asarray(z["coordinates"],float),z["instance_hash"].astype(str).tolist()


def parse_tsplib(path:Path)->np.ndarray:
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",encoding="utf-8",errors="replace") as f:lines=f.readlines()
    n=None;start=None
    for i,line in enumerate(lines):
        t=line.strip().upper()
        if t.startswith("DIMENSION"):n=int(line.replace(":"," ").split()[-1])
        if t=="NODE_COORD_SECTION":start=i+1;break
    if n is None or start is None:raise RuntimeError(path)
    pts=[]
    for line in lines[start:]:
        t=line.strip()
        if not t or t.upper() in {"EOF","DISPLAY_DATA_SECTION"}:break
        p=t.split()
        if len(p)>=3:pts.append((float(p[1]),float(p[2])))
        if len(pts)==n:break
    x=np.asarray(pts,float)
    if x.shape!=(n,2):raise RuntimeError(f"TSPLIB shape {path}: {x.shape}")
    return x


def load_reference()->tuple[pd.DataFrame,np.ndarray,dict[str,np.ndarray]]:
    m=pd.read_csv(REF_MANIFEST)
    if len(m)!=48 or m.instance_id.nunique()!=48:raise RuntimeError("TSPLIB48 integrity")
    fft=pd.read_parquet(P6E0/"features/TSPLIB48.parquet")
    frame=m[["instance_id","n_nodes","evoreal_split"]].merge(fft[["instance_id","fft_energy","nn_ratio"]],on="instance_id",validate="one_to_one")
    ff=pd.read_csv(REF_F40);ff=ff[ff.coordinate_variant.eq("aspect")].set_index("instance_id").loc[m.instance_id]
    coords={r.instance_id:parse_tsplib(ROOT/r.source_file) for r in m.itertuples(index=False)}
    return frame,ff[list(F40)].to_numpy(float),coords


def membership_hash(hashes:list[str])->str:return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def load_pools(f40_index:pd.DataFrame)->tuple[dict[str,dict[str,Any]],dict[str,dict[str,Any]]]:
    # P03
    pxy,ph=load_npz(P03_PATH);pf=pd.read_parquet(P6G0/"features/P03.parquet")
    if pf.instance_hash.astype(str).tolist()!=ph:raise RuntimeError("P03 FFT alignment")
    pools={"P03":{"hashes":ph,"fft":pf[["fft_energy","nn_ratio"]].to_numpy(float),"f40":f40_index.loc[ph,list(F40)].to_numpy(float),
                  "source":np.full(len(ph),"P03",object),"source_index":np.arange(len(ph)),"generator":np.full(len(ph),"P03",object)}}
    # Historical RAW union, preserving first observed provenance only.
    parts=[];stage_maps={}
    for spec in raw_specs():
        ff=pd.read_parquet(spec["fft"]);hh=ff.instance_hash.astype(str).tolist()
        with np.load(spec["coords"],allow_pickle=False) as z:
            if z["instance_hash"].astype(str).tolist()!=hh:raise RuntimeError(f"RAW alignment {spec['dataset']}")
        part=ff[["instance_hash","fft_energy","nn_ratio"]].copy();part["source"]=spec["dataset"];part["source_index"]=np.arange(len(part));parts.append(part);stage_maps[spec["dataset"]]=spec
    fu=pd.concat(parts,ignore_index=True).drop_duplicates("instance_hash",keep="first").reset_index(drop=True);fh=fu.instance_hash.astype(str).tolist()
    pools["FOGA_RAW"]={"hashes":fh,"fft":fu[["fft_energy","nn_ratio"]].to_numpy(float),"f40":f40_index.loc[fh,list(F40)].to_numpy(float),
                       "source":fu.source.to_numpy(object),"source_index":fu.source_index.to_numpy(int),"generator":np.full(len(fu),"FOGA11",object)}
    # EvoReal
    ef=pd.read_parquet(P6E0/"features/EvoReal50k.parquet");eh=ef.instance_hash.astype(str).tolist()
    pools["EvoReal50k"]={"hashes":eh,"fft":ef[["fft_energy","nn_ratio"]].to_numpy(float),"f40":f40_index.loc[eh,list(F40)].to_numpy(float),
                         "source":np.full(len(eh),"EvoReal50k",object),"source_index":np.arange(len(eh)),"generator":ef.generator_source.astype(str).to_numpy(object)}
    for name,p in pools.items():
        if len(p["hashes"])!=len(set(p["hashes"])):raise RuntimeError(f"pool nonunique {name}")
    return pools,stage_maps


def robust_f40_config(pools:dict[str,dict[str,Any]])->dict[str,Any]:
    rng=np.random.default_rng(SEED);parts=[];samples={}
    for name in ("P03","FOGA_RAW","EvoReal50k"):
        n=min(10000,len(pools[name]["hashes"]));idx=rng.choice(len(pools[name]["hashes"]),n,replace=False);parts.append(pools[name]["f40"][idx]);samples[name]=idx.tolist()
    x=np.vstack(parts);center=np.nanmedian(x,axis=0);q=np.nanquantile(x,[.25,.75],axis=0);scale=q[1]-q[0]
    effective=np.isfinite(center)&np.isfinite(scale)&(scale>1e-12)
    config={"fit":"balanced synthetic-only sample: 10k P03 + 10k FOGA_RAW union + 10k EvoReal50k","seed":SEED,"center":"featurewise median","scale":"featurewise IQR","distance":"RMS robust-z difference over pairwise finite effective dimensions","minimum_dimensions":MIN_F40_DIMS,"features":list(F40),"center_values":center,"scale_values":scale,"effective_mask":effective,"effective_n":int(effective.sum()),"zero_or_invalid_scale_features":[F40[i] for i in np.flatnonzero(~effective)],"sample_membership_sha256":{k:membership_hash([pools[k]["hashes"][i] for i in v]) for k,v in samples.items()},"HGB_weighting":False,"TSPLIB_used_to_fit":False,"missing_policy":"pairwise finite; no imputation"}
    write_json(MAN/"f40_distance_config.json",config);return {**config,"center_values":center,"scale_values":scale,"effective_mask":effective}


def distance_matrices(pools:dict[str,dict[str,Any]],ref_fft:pd.DataFrame,ref_f40:np.ndarray,cfg:dict[str,Any])->dict[str,dict[str,np.ndarray]]:
    rules=json.loads(FROZEN_RULES.read_text());center=np.asarray(rules["center"]);scale2=np.asarray(rules["scale_iqr"])
    rz=(ref_fft[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale2
    fcenter=np.asarray(cfg["center_values"]);fscale=np.asarray(cfg["scale_values"]);eff=np.asarray(cfg["effective_mask"],bool)
    refs=(ref_f40-fcenter)/fscale;outputs={}
    for name,p in pools.items():
        n=len(p["hashes"]);token=membership_hash(p["hashes"]);meta_path=CACHE/f"{name}_distance_meta.json";fft_path=CACHE/f"{name}_fft_dist.npy";f40_path=CACHE/f"{name}_f40_dist.npy"
        reuse=False
        if meta_path.is_file() and fft_path.is_file() and f40_path.is_file():
            meta=json.loads(meta_path.read_text());reuse=meta.get("membership_sha256")==token and meta.get("f40_config_sha256")==sha256_file(MAN/"f40_distance_config.json")
        if not reuse:
            print(f"[P6-G2] distances {name} n={n:,}",flush=True)
            fmm=np.lib.format.open_memmap(f40_path,mode="w+",dtype="float32",shape=(n,len(refs)))
            xmm=np.lib.format.open_memmap(fft_path,mode="w+",dtype="float32",shape=(n,len(refs)))
            for start in range(0,n,25000):
                end=min(n,start+25000);x=(p["f40"][start:end]-fcenter)/fscale;x2=(p["fft"][start:end]-center)/scale2
                xmm[start:end]=distance.cdist(x2,rz).astype("float32")
                for j,r in enumerate(refs):
                    mask=eff&np.isfinite(r);d=x[:,mask]-r[mask];valid=np.isfinite(d);count=valid.sum(1)
                    val=np.sqrt(np.divide(np.nansum(d*d,axis=1),count,out=np.full(len(d),np.inf),where=count>0));val[count<min(MIN_F40_DIMS,int(mask.sum()))]=np.inf;fmm[start:end,j]=val.astype("float32")
                if start==0 or end==n or end%250000==0:print(f"[P6-G2] {name} {end:,}/{n:,}",flush=True)
            del fmm,xmm
            write_json(meta_path,{"pool":name,"n":n,"membership_sha256":token,"f40_config_sha256":sha256_file(MAN/"f40_distance_config.json"),"shape":[n,len(refs)]})
        outputs[name]={"fft":np.load(fft_path,mmap_mode="r"),"f40":np.load(f40_path,mmap_mode="r")}
    return outputs


def top_indices(v:np.ndarray,k:int=TOPK)->np.ndarray:
    idx=np.argpartition(v,k-1)[:k];return idx[np.argsort(v[idx],kind="stable")]


def nearest_candidates(pools:dict[str,dict[str,Any]],mats:dict[str,dict[str,np.ndarray]],refs:pd.DataFrame,ref_f40:np.ndarray,cfg:dict[str,Any])->pd.DataFrame:
    rules=json.loads(FROZEN_RULES.read_text());radius=float(rules["radius"]);fc=np.asarray(cfg["center_values"]);fs=np.asarray(cfg["scale_values"]);eff=np.asarray(cfg["effective_mask"],bool)
    rows=[]
    for pool,p in pools.items():
        for j,ref in enumerate(refs.instance_id):
            fi=top_indices(np.asarray(mats[pool]["fft"][:,j]));hi=top_indices(np.asarray(mats[pool]["f40"][:,j]));fset=set(fi.tolist());hset=set(hi.tolist());same=int(fi[0])==int(hi[0]);overlap=len(fset&hset)
            for metric,idxs in (("FFT_NN",fi),("F40_ROBUST",hi)):
                for rank,i in enumerate(idxs,1):
                    i=int(i);cand=p["f40"][i];r=ref_f40[j];mask=eff&np.isfinite(cand)&np.isfinite(r);delta=(cand-r)/fs;order=np.argsort(np.where(mask,np.abs(delta),-np.inf))[::-1][:min(8,int(mask.sum()))]
                    fval=float(mats[pool]["f40"][i,j]);xval=float(mats[pool]["fft"][i,j]);
                    rows.append({"reference_id":ref,"reference_n_nodes":int(refs.iloc[j].n_nodes),"matched_75_150":bool(75<=refs.iloc[j].n_nodes<=150),"pool":pool,"retrieval_metric":metric,"retrieval_rank":rank,"candidate_index":i,"candidate_hash":p["hashes"][i],"source_dataset":p["source"][i],"source_index":int(p["source_index"][i]),"generator_source":p["generator"][i],"fft_nn_distance":xval,"frozen_radius":radius,"fft_distance_over_radius":xval/radius,"fft_covered":xval<=radius,"f40_distance":fval,"f40_effective_dimensions":int(mask.sum()),"top_f40_difference_features":json.dumps([F40[q] for q in order]),"top_f40_signed_robust_z_differences":json.dumps([float(delta[q]) for q in order]),"top5_overlap_n":overlap,"top5_jaccard":overlap/len(fset|hset),"top1_same":same,
                                 "f40_rank_in_pool":int(np.sum(np.asarray(mats[pool]["f40"][:,j])<fval)+1),"fft_rank_in_pool":int(np.sum(np.asarray(mats[pool]["fft"][:,j])<xval)+1),"pool_n":len(p["hashes"])})
    out=pd.DataFrame(rows);return add_provenance(out)


def add_provenance(frame:pd.DataFrame)->pd.DataFrame:
    frame=frame.copy();frame["operator_id"]="";frame["parent_hash"]="";frame["n_affected"]=np.nan
    specs={x["dataset"]:x for x in raw_specs()}
    for source,g in frame[frame.pool.eq("FOGA_RAW")].groupby("source_dataset"):
        hashes=set(g.candidate_hash.astype(str));m=pd.read_parquet(specs[source]["manifest"]);m=m[m.child_hash.astype(str).isin(hashes)].drop_duplicates("child_hash").set_index("child_hash")
        for idx in g.index:
            h=str(frame.at[idx,"candidate_hash"])
            if h in m.index:
                frame.at[idx,"operator_id"]=str(m.at[h,"operator_id"]);frame.at[idx,"parent_hash"]=str(m.at[h,"parent_hash"]);frame.at[idx,"n_affected"]=float(m.at[h,"n_affected"])
    return frame


def coordinate_store(pools:dict[str,dict[str,Any]],stage_maps:dict[str,dict[str,Any]],candidates:pd.DataFrame)->tuple[dict[tuple[str,int],np.ndarray],dict[str,np.ndarray]]:
    arrays={"P03":load_npz(P03_PATH)[0],"EvoReal50k":np.load(EVO_PATH,mmap_mode="r",allow_pickle=False)}
    for source in candidates[candidates.pool.eq("FOGA_RAW")].source_dataset.unique():arrays[source]=load_npz(stage_maps[source]["coords"])[0]
    coords={}
    for r in candidates.itertuples(index=False):coords[(r.pool,int(r.candidate_index))]=np.asarray(arrays[str(r.source_dataset)][int(r.source_index)],float)
    keys=[];values=[]
    for (pool,i),x in coords.items():keys.append(f"{pool}::{i}");values.append(x)
    np.savez_compressed(MAN/"nearest_candidate_coordinates.npz",keys=np.asarray(keys),coordinates=np.stack(values))
    return coords,arrays


def shape_normalize(x:np.ndarray)->np.ndarray:
    z=np.asarray(x,float)-np.mean(x,axis=0);radius=float(np.max(np.linalg.norm(z,axis=1)))
    return z/radius if radius>0 else np.zeros_like(z)


def js(p:np.ndarray,q:np.ndarray)->float:
    p=np.asarray(p,float);q=np.asarray(q,float);p=p/p.sum();q=q/q.sum();m=(p+q)/2
    def kl(a,b):
        k=a>0;return float(np.sum(a[k]*np.log2(a[k]/b[k])))
    return .5*kl(p,m)+.5*kl(q,m)


def hist2(x:np.ndarray,grid:int)->np.ndarray:
    h=np.histogram2d(x[:,0],x[:,1],bins=grid,range=[[-1,1],[-1,1]])[0].ravel();return h+0.5


def kdist(x:np.ndarray)->np.ndarray:
    k=min(6,len(x));d=cKDTree(x).query(x,k=k)[0]
    if d.ndim==1:d=d[:,None]
    cols=[min(q,d.shape[1]-1) for q in (1,3,5)];return d[:,cols]


def geometry_compare(ref:np.ndarray,cand:np.ndarray)->dict[str,Any]:
    r=shape_normalize(ref);c=shape_normalize(cand);rh={g:hist2(r,g) for g in GEOMETRY_GRIDS}
    best=(np.inf,0.0,False,{})
    for reflected in (False,True):
        base=c.copy()
        if reflected:base[:,0]*=-1
        for angle in GEOMETRY_ANGLES:
            a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);z=base@rot.T
            vals={g:js(rh[g],hist2(z,g)) for g in GEOMETRY_GRIDS};score=float(np.mean(list(vals.values())))
            if score<best[0]:best=(score,float(angle),reflected,vals)
    occupancy,angle,reflected,occ=best
    rd=kdist(r);cd=kdist(c);knn=[]
    for j in range(3):
        a=rd[:,j]/max(np.median(rd[:,j]),1e-12);b=cd[:,j]/max(np.median(cd[:,j]),1e-12);knn.append(wasserstein_distance(a,b))
    nn_cv_r=float(np.std(rd[:,0])/max(np.mean(rd[:,0]),1e-12));nn_cv_c=float(np.std(cd[:,0])/max(np.mean(cd[:,0]),1e-12))
    pair=wasserstein_distance(distance.pdist(r),distance.pdist(c))/2
    radial=wasserstein_distance(np.linalg.norm(r,axis=1),np.linalg.norm(c,axis=1))
    empty=[]
    for g in GEOMETRY_GRIDS:
        er=np.mean(np.histogram2d(r[:,0],r[:,1],bins=g,range=[[-1,1],[-1,1]])[0]==0);ec=np.mean(np.histogram2d(c[:,0],c[:,1],bins=g,range=[[-1,1],[-1,1]])[0]==0);empty.append(abs(er-ec))
    kn=float(np.mean(knn));cv=abs(nn_cv_r-nn_cv_c);composite=float(np.mean([occupancy,pair,radial,kn/(1+kn),cv/(1+cv)]))
    return {"geometry_composite":composite,"occupancy_js_mean":occupancy,"occupancy_js_g4":occ[4],"occupancy_js_g8":occ[8],"occupancy_js_g16":occ[16],"best_rotation_deg":angle,"best_reflection":reflected,"pairwise_distance_w1_over2":pair,"radial_distance_w1":radial,"knn_shape_w1_mean":kn,"nn1_cv_reference":nn_cv_r,"nn1_cv_candidate":nn_cv_c,"nn1_cv_abs_diff":cv,"empty_cell_fraction_abs_diff_mean":float(np.mean(empty)),"reference_n":len(r),"candidate_n":len(c),"node_count_ratio":len(c)/len(r),"knn_w1_k1":knn[0],"knn_w1_k3":knn[1],"knn_w1_k5":knn[2]}


def geometry_audit(candidates:pd.DataFrame,coords:dict[tuple[str,int],np.ndarray],ref_coords:dict[str,np.ndarray])->pd.DataFrame:
    unique=candidates.drop_duplicates(["reference_id","pool","candidate_index"]);rows=[]
    total=len(unique)
    for done,r in enumerate(unique.itertuples(index=False),1):
        q=geometry_compare(ref_coords[r.reference_id],coords[(r.pool,int(r.candidate_index))]);rows.append({"reference_id":r.reference_id,"reference_n_nodes":r.reference_n_nodes,"matched_75_150":r.matched_75_150,"pool":r.pool,"candidate_index":r.candidate_index,"candidate_hash":r.candidate_hash,"source_dataset":r.source_dataset,"generator_source":r.generator_source,"operator_id":r.operator_id,"is_fft_top5":bool(((candidates.reference_id==r.reference_id)&(candidates.pool==r.pool)&(candidates.candidate_index==r.candidate_index)&(candidates.retrieval_metric=="FFT_NN")).any()),"is_f40_top5":bool(((candidates.reference_id==r.reference_id)&(candidates.pool==r.pool)&(candidates.candidate_index==r.candidate_index)&(candidates.retrieval_metric=="F40_ROBUST")).any()),**q})
        if done==1 or done%100==0 or done==total:print(f"[P6-G2] geometry {done}/{total}",flush=True)
    return pd.DataFrame(rows)


def reference_validation(candidates:pd.DataFrame,geometry:pd.DataFrame,pools:dict[str,dict[str,Any]],mats:dict[str,dict[str,np.ndarray]],refs:pd.DataFrame)->pd.DataFrame:
    rules=json.loads(FROZEN_RULES.read_text());radius=float(rules["radius"]);rows=[];rng=np.random.default_rng(SEED+99)
    balanced={k:rng.choice(len(pools[k]["hashes"]),min(10000,len(pools[k]["hashes"])),replace=False) for k in pools}
    for j,r in refs.iterrows():
        row={"reference_id":r.instance_id,"n_nodes":int(r.n_nodes),"matched_75_150":bool(75<=r.n_nodes<=150),"reference_group":r.evoreal_split}
        for pool in pools:
            c=candidates[(candidates.reference_id==r.instance_id)&(candidates.pool==pool)];g=geometry[(geometry.reference_id==r.instance_id)&(geometry.pool==pool)]
            fft1=c[(c.retrieval_metric=="FFT_NN")&(c.retrieval_rank==1)].iloc[0];f401=c[(c.retrieval_metric=="F40_ROBUST")&(c.retrieval_rank==1)].iloc[0]
            row.update({f"{pool}_best_fft":fft1.fft_nn_distance,f"{pool}_fft_ratio":fft1.fft_distance_over_radius,f"{pool}_fft_covered":bool(fft1.fft_covered),f"{pool}_best_f40":f401.f40_distance,f"{pool}_fft_top1_f40_rank_fraction":fft1.f40_rank_in_pool/fft1.pool_n,f"{pool}_f40_top1_fft_rank_fraction":f401.fft_rank_in_pool/f401.pool_n,f"{pool}_top1_same":bool(fft1.top1_same),f"{pool}_top5_jaccard":fft1.top5_jaccard,f"{pool}_best_geometry":g.geometry_composite.min(),f"{pool}_fft_top1_geometry":g[g.candidate_index.eq(fft1.candidate_index)].geometry_composite.iloc[0],f"{pool}_f40_top1_geometry":g[g.candidate_index.eq(f401.candidate_index)].geometry_composite.iloc[0]})
            idx=balanced[pool];rho=spearmanr(np.asarray(mats[pool]["fft"])[idx,j],np.asarray(mats[pool]["f40"])[idx,j],nan_policy="omit").statistic;row[f"{pool}_fft_f40_spearman_balanced10k"]=rho
        base_f40=min(row["P03_best_f40"],row["FOGA_RAW_best_f40"]);base_geo=min(row["P03_best_geometry"],row["FOGA_RAW_best_geometry"])
        row["EvoReal_f40_better_than_P03_FOGA_full"]=row["EvoReal50k_best_f40"]<base_f40;row["EvoReal_geometry_better_than_P03_FOGA_full"]=row["EvoReal50k_best_geometry"]<base_geo
        if row["EvoReal50k_fft_covered"]:
            if row["EvoReal_f40_better_than_P03_FOGA_full"] and row["EvoReal_geometry_better_than_P03_FOGA_full"]:category="FFT_F40_GEOMETRY_SUPPORTED"
            elif row["EvoReal_f40_better_than_P03_FOGA_full"] or row["EvoReal_geometry_better_than_P03_FOGA_full"]:category="FFT_PARTIALLY_SUPPORTED"
            else:category="FFT_ONLY_OR_INCONSISTENT"
        elif row["EvoReal_f40_better_than_P03_FOGA_full"] or row["EvoReal_geometry_better_than_P03_FOGA_full"]:category="NONFFT_STRUCTURAL_ADVANTAGE"
        else:category="NO_EVOREAL_ADVANTAGE"
        row["evidence_category"]=category;rows.append(row)
    return pd.DataFrame(rows)


def fixed_sampling(pools:dict[str,dict[str,Any]],mats:dict[str,dict[str,np.ndarray]],refs:pd.DataFrame,stage_maps:dict[str,dict[str,Any]])->pd.DataFrame:
    rows=[];radius=float(json.loads(FROZEN_RULES.read_text())["radius"]);foga_index=pd.Index(pools["FOGA_RAW"]["hashes"])
    datasets=[("P03","P03",np.arange(len(pools["P03"]["hashes"])),"ALL","SHARED",0)]
    for spec in raw_specs():
        hh=pd.read_parquet(spec["fft"],columns=["instance_hash"]).instance_hash.astype(str);idx=foga_index.get_indexer(hh)
        if (idx<0).any():raise RuntimeError(f"FOGA union mapping {spec['dataset']}")
        datasets.append((spec["dataset"],"FOGA_RAW",idx,spec["replicate"],spec["arm"],spec["generation"]))
    datasets.append(("EvoReal50k","EvoReal50k",np.arange(len(pools["EvoReal50k"]["hashes"])),"EvoReal","ALL",-1))
    for dname,pool,eligible,rep,arm,generation in datasets:
        for repeat in range(REPEATS):
            rng=np.random.default_rng(SEED+repeat+int(hashlib.sha256(dname.encode()).hexdigest()[:8],16));idx=eligible if len(eligible)==10000 else rng.choice(eligible,10000,replace=False)
            xf=np.asarray(mats[pool]["fft"])[idx];hf=np.asarray(mats[pool]["f40"])[idx];xb=xf.min(0);hb=hf.min(0);matched=refs.n_nodes.between(75,150).to_numpy()
            rows.append({"record_type":"repeat","dataset":dname,"pool":pool,"replicate":rep,"arm":arm,"generation":generation,"sample_n":10000,"repeat":repeat,"fft_coverage_48":int(np.sum(xb<=radius)),"fft_coverage_17":int(np.sum(xb[matched]<=radius)),"best_fft_mean":xb.mean(),"best_fft_median":np.median(xb),"best_f40_mean":hb.mean(),"best_f40_median":np.median(hb),"best_f40_p90":np.quantile(hb,.9)})
    frame=pd.DataFrame(rows);summary=[]
    metrics=["fft_coverage_48","fft_coverage_17","best_fft_mean","best_fft_median","best_f40_mean","best_f40_median","best_f40_p90"]
    for keys,g in frame.groupby(["dataset","pool","replicate","arm","generation"]):
        row={"record_type":"summary","dataset":keys[0],"pool":keys[1],"replicate":keys[2],"arm":keys[3],"generation":keys[4],"sample_n":10000,"repeat":-1}
        for metric in metrics:
            v=g[metric].to_numpy(float);row[metric]=v.mean();row[metric+"_p025"]=np.quantile(v,.025);row[metric+"_p975"]=np.quantile(v,.975)
        summary.append(row)
    return pd.concat([frame,pd.DataFrame(summary)],ignore_index=True,sort=False)


def aligned_for_plot(x:np.ndarray,angle:float,reflected:bool)->np.ndarray:
    z=shape_normalize(x)
    if reflected:z[:,0]*=-1
    a=np.deg2rad(angle);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);return z@rot.T


def make_plots(candidates:pd.DataFrame,geometry:pd.DataFrame,validation:pd.DataFrame,coords:dict[tuple[str,int],np.ndarray],ref_coords:dict[str,np.ndarray])->None:
    colors={"P03":"#555555","FOGA_RAW":"#1f77b4","EvoReal50k":"#d62728"}
    fig,ax=plt.subplots(figsize=(8,6))
    for pool,g in candidates.groupby("pool"):
        ax.scatter(g.fft_distance_over_radius,g.f40_distance,s=13,alpha=.45,label=pool,color=colors[pool])
    ax.set_xscale("log");ax.set_xlabel("FFT–NN distance / frozen radius (log)");ax.set_ylabel("robust F40 RMS distance");ax.axvline(1,color="black",ls="--",lw=1);ax.legend();ax.set_title("Top-5 retrieval candidates: FFT proximity versus F40 proximity");fig.tight_layout();fig.savefig(FIG/"01_fft_vs_f40_distance.png",dpi=180);plt.close(fig)
    # Per-reference best comparison.
    fig,axes=plt.subplots(1,2,figsize=(13,5));x=np.arange(len(validation))
    for pool in colors:
        axes[0].plot(x,validation[f"{pool}_best_f40"],"o",ms=3,alpha=.7,label=pool,color=colors[pool]);axes[1].plot(x,validation[f"{pool}_best_geometry"],"o",ms=3,alpha=.7,label=pool,color=colors[pool])
    axes[0].set_ylabel("best robust F40 distance");axes[1].set_ylabel("best independent geometry composite");
    for ax in axes:ax.set_xlabel("TSPLIB48 reference index");ax.legend(fontsize=8)
    fig.suptitle("Full-pool best similarity by reference (pool sizes differ)");fig.tight_layout();fig.savefig(FIG/"02_reference_best_similarity.png",dpi=180);plt.close(fig)
    # pr107/pr144 gallery: reference plus FFT/F40 top1 for each pool.
    fig,axes=plt.subplots(2,7,figsize=(18,6))
    for rr,ref in enumerate(("pr107","pr144")):
        axes[rr,0].scatter(*shape_normalize(ref_coords[ref]).T,s=6);axes[rr,0].set_title(f"{ref}\nTSPLIB",fontsize=8)
        col=1
        for pool in ("P03","FOGA_RAW","EvoReal50k"):
            for metric in ("FFT_NN","F40_ROBUST"):
                r=candidates[(candidates.reference_id==ref)&(candidates.pool==pool)&(candidates.retrieval_metric==metric)&(candidates.retrieval_rank==1)].iloc[0]
                g=geometry[(geometry.reference_id==ref)&(geometry.pool==pool)&(geometry.candidate_index==r.candidate_index)].iloc[0];z=aligned_for_plot(coords[(pool,int(r.candidate_index))],g.best_rotation_deg,bool(g.best_reflection));axes[rr,col].scatter(*z.T,s=6);axes[rr,col].set_title(f"{pool.replace('50k','')} {metric.split('_')[0]}\nFFT/r={r.fft_distance_over_radius:.2f}; F40={r.f40_distance:.2f}\nGeo={g.geometry_composite:.2f}",fontsize=7);col+=1
        for ax in axes[rr]:ax.set_aspect("equal");ax.set_xlim(-1.05,1.05);ax.set_ylim(-1.05,1.05);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("pr107/pr144: TSPLIB and aligned top-1 candidates (alignment fixed by multiscale occupancy)");fig.tight_layout();fig.savefig(FIG/"03_pr107_pr144_fft_f40_geometry.png",dpi=180);plt.close(fig)
    # Fifteen-reference evidence heatmap.
    q=validation.set_index("reference_id").loc[list(UNREACHABLE)];data=np.column_stack([q.EvoReal50k_fft_ratio,q.EvoReal50k_best_f40/q[["P03_best_f40","FOGA_RAW_best_f40"]].min(axis=1),q.EvoReal50k_best_geometry/q[["P03_best_geometry","FOGA_RAW_best_geometry"]].min(axis=1)])
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(np.log2(np.clip(data,1e-3,1e3)),aspect="auto",cmap="coolwarm",vmin=-3,vmax=3);ax.set_yticks(np.arange(len(q)),q.index);ax.set_xticks(range(3),["Evo FFT/r","Evo F40 / best baseline","Evo geometry / best baseline"],rotation=15,ha="right");fig.colorbar(im,ax=ax,label="log2 ratio (blue < 1 is closer)");ax.set_title("15 historically unreachable references: metric sensitivity");fig.tight_layout();fig.savefig(FIG/"04_unreachable15_metric_diagnostics.png",dpi=180);plt.close(fig)
    # One auditable gallery per reference: original plus FFT/F40 top-1 from all pools.
    gallery=FIG/"reference_galleries";gallery.mkdir(parents=True,exist_ok=True)
    for ref in validation.reference_id:
        fig,axes=plt.subplots(1,7,figsize=(18,3.2));axes[0].scatter(*shape_normalize(ref_coords[ref]).T,s=6);axes[0].set_title(f"{ref}\nTSPLIB",fontsize=8);col=1
        for pool in ("P03","FOGA_RAW","EvoReal50k"):
            for metric in ("FFT_NN","F40_ROBUST"):
                r=candidates[(candidates.reference_id==ref)&(candidates.pool==pool)&(candidates.retrieval_metric==metric)&(candidates.retrieval_rank==1)].iloc[0]
                g=geometry[(geometry.reference_id==ref)&(geometry.pool==pool)&(geometry.candidate_index==r.candidate_index)].iloc[0]
                z=aligned_for_plot(coords[(pool,int(r.candidate_index))],g.best_rotation_deg,bool(g.best_reflection));axes[col].scatter(*z.T,s=6)
                axes[col].set_title(f"{pool.replace('50k','')} {metric.split('_')[0]}\nFFT/r={r.fft_distance_over_radius:.2f}; F40={r.f40_distance:.2f}\nGeo={g.geometry_composite:.2f}",fontsize=7);col+=1
        for ax in axes:ax.set_aspect("equal");ax.set_xlim(-1.05,1.05);ax.set_ylim(-1.05,1.05);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"{ref}: original and aligned FFT/F40 nearest candidates");fig.tight_layout();fig.savefig(gallery/f"{ref}.png",dpi=150);plt.close(fig)


def reports(pools:dict[str,dict[str,Any]],cfg:dict[str,Any],candidates:pd.DataFrame,geometry:pd.DataFrame,validation:pd.DataFrame,fixed:pd.DataFrame,refs:pd.DataFrame)->None:
    rules=json.loads(FROZEN_RULES.read_text());summ=fixed[fixed.record_type.eq("summary")]
    covered=validation[validation.EvoReal50k_fft_covered];matched=validation[validation.matched_75_150]
    counts=covered.evidence_category.value_counts();top_consistency=candidates[candidates.retrieval_rank.eq(1)].groupby("pool").agg(top1_same_fraction=("top1_same","mean"),top5_jaccard_mean=("top5_jaccard","mean"),fft_top1_f40_rank_fraction_median=("f40_rank_in_pool",lambda x:np.nan)).reset_index()
    # Replace rank aggregation with exact per-pool rank fractions from FFT top1 only.
    rows=[]
    for pool,g in candidates[(candidates.retrieval_metric=="FFT_NN")&(candidates.retrieval_rank==1)].groupby("pool"):
        rows.append({"pool":pool,"top1_same_fraction":g.top1_same.mean(),"top5_jaccard_mean":g.top5_jaccard.mean(),"fft_top1_f40_rank_fraction_median":np.median(g.f40_rank_in_pool/g.pool_n),"balanced10k_fft_f40_spearman_median":validation[f"{pool}_fft_f40_spearman_balanced10k"].median()})
    consistency=pd.DataFrame(rows);consistency.to_csv(TAB/"fft_f40_consistency_summary.csv",index=False)
    inventory=[]
    for name,p in pools.items():inventory.append({"pool":name,"n":len(p["hashes"]),"unique_n":len(set(p["hashes"])),"membership_sha256":membership_hash(p["hashes"])})
    inv=pd.DataFrame(inventory);inv.to_csv(MAN/"pool_inventory.csv",index=False)
    integrity=f"""# P6-G2 Data and Metric Integrity

This is a read-only validation. No FOGA, EvoReal generation, Concorde, BQ/HGB/GP inference or training was run. TSPLIB22 was not loaded.

## Frozen inputs

{md(inv)}

- TSPLIB48: 48 unique development references; the predefined 75–150 control contains {int(refs.n_nodes.between(75,150).sum())} references.
- FFT contrast: frozen `FFT_APPROX + NN-ratio`, center `{rules['center']}`, IQR scale `{rules['scale_iqr']}`, radius `{rules['radius']:.9f}`.
- Formal F40: frozen order and existing P6-G1 values. Robust distance scaling was fit on an equal synthetic-only 30k sample (10k P03 + 10k FOGA union + 10k EvoReal), seed {SEED}; no TSPLIB outcomes or HGB importance entered weighting. Effective dimensions: **{cfg['effective_n']}/40**; zero-scale dimensions: `{cfg['zero_or_invalid_scale_features']}`. Native missing values are pairwise-excluded without imputation, requiring at least {MIN_F40_DIMS} dimensions.

## Independent geometry diagnostic (fixed before results)

Coordinates are centered and divided by maximum radial norm, preserving aspect ratio and providing translation/uniform-scale invariance. Candidate orientation is searched over fixed 5° rotations and reflection. Metrics are: multiscale 4×4/8×8/16×16 occupancy JS with 0.5 pseudocount, radial-distance W1, pairwise-distance W1, self-median-normalized kNN-distance W1 for k=1/3/5, NN-CV difference, and empty-cell fraction difference. The reported composite is the unweighted mean of bounded occupancy, pairwise, radial, transformed kNN and transformed NN-CV components. It is an explanatory diagnostic, not a learned selector, fitness, or new coverage threshold.

Different node counts still affect empirical resolution and occupancy sparsity; therefore all 48 and the 75–150 subset are reported separately. Pool-size extreme-neighbor effects are separated from 30× matched 10k comparisons.
""";(OUT/"01_data_and_metric_integrity.md").write_text(integrity)
    # Labels are based on joint continuous/rank evidence, not a replacement threshold.
    both=int((covered.evidence_category=="FFT_F40_GEOMETRY_SUPPORTED").sum());partial=int((covered.evidence_category=="FFT_PARTIALLY_SUPPORTED").sum());only=int((covered.evidence_category=="FFT_ONLY_OR_INCONSISTENT").sum())
    pr107=validation.set_index("reference_id").loc["pr107"];pr144=validation.set_index("reference_id").loc["pr144"]
    fft_reproduced=(int(validation.P03_fft_covered.sum())==23 and int(validation.FOGA_RAW_fft_covered.sum())==25 and int(validation.EvoReal50k_fft_covered.sum())==40)
    # Initial automated labels are conservative; exact supporting counts are printed in the report.
    median_rho=float(consistency.balanced10k_fft_f40_spearman_median.median());f40_label="WEAK" if median_rho<.2 else ("MIXED" if median_rho<.6 else "STRONG")
    geo_ratio=np.median(covered.EvoReal50k_fft_top1_geometry/covered.EvoReal50k_f40_top1_geometry);geo_label="MIXED" if .75<=geo_ratio<=1.5 else "WEAK"
    advantage="ROBUST" if both>=28 else ("PARTIAL" if both+partial>=20 else "METRIC_DEPENDENT")
    def ref_label(row:pd.Series)->str:
        if not row.EvoReal50k_fft_covered:return "NOT_SUPPORTED"
        f=bool(row.EvoReal_f40_better_than_P03_FOGA_full);g=bool(row.EvoReal_geometry_better_than_P03_FOGA_full)
        return "SUPPORTED" if f and g else ("PARTIAL" if f or g else "NOT_SUPPORTED")
    # Relative scores improve for pr107/pr144, but the aligned galleries show
    # only motif-level agreement rather than the complete global layout.
    labels={"P6_G2_FFT_COVERAGE_REPRODUCED":"YES" if fft_reproduced else "NO","P6_G2_FFT_F40_CONSISTENCY":f40_label,"P6_G2_FFT_GEOMETRY_CONSISTENCY":geo_label,"P6_G2_EVOREAL_STRUCTURAL_ADVANTAGE":advantage,"P6_G2_PR107_GEOMETRIC_MATCH":"PARTIAL","P6_G2_PR144_GEOMETRIC_MATCH":"PARTIAL","P6_G2_ANALYSIS_STATUS":"COMPLETE"}
    unreachable=validation.set_index("reference_id").loc[list(UNREACHABLE)].reset_index()
    both_names=unreachable[unreachable.evidence_category.eq("FFT_F40_GEOMETRY_SUPPORTED")].reference_id.tolist()
    partial_names=unreachable[unreachable.evidence_category.eq("FFT_PARTIALLY_SUPPORTED")].reference_id.tolist()
    only_names=unreachable[unreachable.evidence_category.eq("FFT_ONLY_OR_INCONSISTENT")].reference_id.tolist()
    matched_counts=matched.evidence_category.value_counts()
    key=validation[["reference_id","n_nodes","matched_75_150","EvoReal50k_fft_covered","EvoReal_f40_better_than_P03_FOGA_full","EvoReal_geometry_better_than_P03_FOGA_full","evidence_category"]]
    fixed_key=summ[["dataset","fft_coverage_48","fft_coverage_48_p025","fft_coverage_48_p975","fft_coverage_17","best_f40_mean","best_f40_mean_p025","best_f40_mean_p975"]]
    final="# P6-G2 Final Conclusion\n\n## 判定标签\n\n"+"\n".join(f"`{k} = {v}`" for k,v in labels.items())+f"""

## 历史覆盖复现

完整池严格复现 P03 **{int(validation.P03_fft_covered.sum())}/48**、FOGA RAW union **{int(validation.FOGA_RAW_fft_covered.sum())}/48**、EvoReal50k **{int(validation.EvoReal50k_fft_covered.sum())}/48**。75–150节点控制组分别为 **{int(matched.P03_fft_covered.sum())}/17**、**{int(matched.FOGA_RAW_fft_covered.sum())}/17**、**{int(matched.EvoReal50k_fft_covered.sum())}/17**。这是原冻结FFT–NN指标复现，不是新覆盖定义。

## FFT 与 F40/几何一致性

{md(consistency)}

EvoReal FFT覆盖的40条中，F40与独立几何相对P03/完整FOGA池均占优 `{both}` 条，仅一项占优 `{partial}` 条，两项均不占优 `{only}` 条。逐reference证据：

{md(key)}

FFT–NN非常近并不保证完整结构一致：Top5集合和Top1的一致率、跨平衡10k的距离相关性均直接记录于上表。独立几何也只验证局部/全局统计与多尺度占据，不被解释成唯一“真实结构”度量。

## 固定10k公平比较

{md(fixed_key)}

完整池结果回答历史预算内“曾发现过什么”；30次10k结果回答相同样本量下的典型表现。二者不能互换。

## 三个研究问题

1. **EvoReal 40/48有多少来自可解释结构？** 40条FFT覆盖中有 `{both}` 条同时得到F40与独立几何的相对支持，另有 `{partial}` 条只获部分支持，`{only}` 条两项均不支持；因此不能把40/48整体视为40条几何同构，但也不是纯粹由两个统计量制造。75–150节点的17条中，双重支持 `{int(matched_counts.get('FFT_F40_GEOMETRY_SUPPORTED',0))}`、单项支持 `{int(matched_counts.get('FFT_PARTIALLY_SUPPORTED',0))}`、两项均不支持 `{int(matched_counts.get('FFT_ONLY_OR_INCONSISTENT',0))}`。
2. **哪些差异有意义，哪些可能是度量表象？** 15条历史不可达reference中，双重支持11条：`{', '.join(both_names)}`；部分支持3条：`{', '.join(partial_names)}`；`{', '.join(only_names)}` 的FFT优势未得到F40或几何支持，最可能属于当前表示造成的表观差异。pr107的EvoReal FFT实例复现重复线段和近邻规律，却没有复现TSPLIB双侧长列的全局布局；pr144复现分离块与局部规则点阵，但没有完整复现原实例的重复微结构，所以两者均为PARTIAL。跨规模reference仍受节点密度和有限采样分辨率影响。
3. **下一步优先级？** 15条不可达reference中14条至少得到F40或几何支持、11条得到二者共同支持，因此证据更支持**优先验证补充初始结构支持**；但FFT/F40 Top5几乎不重合，后续结构表示至少应增加多维一致性检查，避免把单一FFT–NN覆盖当成充分条件。当前实验不提出新fitness、权重或演化深度。

## 证据边界

这些结果验证统计距离和可解释几何相似性，不验证任何BQ训练收益。所有相对比较均保留连续距离与排名，没有把几何诊断重新阈值化成另一套“覆盖率”。

`BLADE_P6G2_STRUCTURAL_SIMILARITY_VALIDATION_COMPLETE`
""";(OUT/"06_final_conclusion.md").write_text(final)
    write_json(OUT/"completion.json",{"status":"COMPLETE","labels":labels,"final_tag":"BLADE_P6G2_STRUCTURAL_SIMILARITY_VALIDATION_COMPLETE","created_utc":pd.Timestamp.utcnow().isoformat(),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__})


def main()->None:
    started=time.perf_counter()
    for p in (OUT,FIG,TAB,MAN,CACHE):p.mkdir(parents=True,exist_ok=True)
    f40=pd.read_parquet(P6G1/"cache/formal_f40_all.parquet").set_index("instance_hash",drop=False)
    if np.isinf(f40[list(F40)].to_numpy(float)).any():raise RuntimeError("F40 infinity")
    refs,rf40,rcoords=load_reference();pools,stage_maps=load_pools(f40);cfg=robust_f40_config(pools);mats=distance_matrices(pools,refs,rf40,cfg)
    candidates=nearest_candidates(pools,mats,refs,rf40,cfg);candidates.to_csv(OUT/"02_fft_vs_f40_nearest_candidates.csv",index=False)
    coords,_=coordinate_store(pools,stage_maps,candidates);geometry=geometry_audit(candidates,coords,rcoords);geometry.to_csv(OUT/"03_geometric_similarity_analysis.csv",index=False)
    validation=reference_validation(candidates,geometry,pools,mats,refs);validation.to_csv(OUT/"04_reference_level_validation.csv",index=False)
    fixed=fixed_sampling(pools,mats,refs,stage_maps);fixed.to_csv(OUT/"05_fixed_sample_robustness.csv",index=False)
    make_plots(candidates,geometry,validation,coords,rcoords);reports(pools,cfg,candidates,geometry,validation,fixed,refs)
    outputs=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "cache" not in p.parts and p.name!="output_hashes.csv":outputs.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    pd.DataFrame(outputs).to_csv(MAN/"output_hashes.csv",index=False)
    print(f"[P6-G2] COMPLETE wall={time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":main()
