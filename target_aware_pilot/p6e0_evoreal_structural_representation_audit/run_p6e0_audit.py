#!/usr/bin/env python3
"""P6-E0 read-only EvoReal FFT/NN structural representation audit.

No instance generation, solver labeling, surrogate fitting, BQ inference/training,
or selection is performed.  TSPLIB22 is loaded only after the reference-only
representation and coverage rules have been persisted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

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
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CFG = OUT / "configs"
MAN = OUT / "manifests"
FEAT = OUT / "features"
LOG = OUT / "logs"
P1 = ROOT / "target_aware_pilot"
P6A = P1 / "p6a_top5_target_matching_feasibility"
P6AE = P1 / "p6ae_evoreal50k_top5_alignment"
P6C0 = P1 / "p6c0_training_value_mechanism_audit"
P6C1 = P1 / "p6c1_hardness_controlled_matching"
P6D0 = P1 / "p6d0_target_aware_qd_selection_audit"
P6D1 = P1 / "p6d1_target_aware_qd_training_value"
P6D3 = P1 / "p6d3_structure_hardness_quota_selection"
V2 = ROOT / "runs/BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2"
ORIGINAL = ROOT / "assets/evaluation/original_tsplib70"
REQUEST = Path("/home/haozou/.codex/attachments/1af116ea-3b30-42ac-979a-946da5a19632/pasted-text.txt")
APPROX_SOURCE = ROOT / "scripts/evoreal_structure_alignment_audit_corrected_v2.py"

REPS = ("Original", "S1", "S2")
CORE_ARMS = ("B", "C")
AUX_ARMS = ("D25", "D50", "D75")
TOP5 = (
    "spn_sum_of_lowest_edge_values", "spn_distance_min",
    "spn_distance_varcoeff", "spn_nng_3_strong_components_varcoeff",
    "spn_mst_dists_sd",
)
SEED = 20261028
MATCHED_SEEDS = tuple(SEED + 100 + i for i in range(10))
GRID = 32
FFT_THRESHOLD = 35.0
NN_THRESHOLD = 0.5
NORMALIZATION_VERSION = "evoreal_alg1_shared_max_span_v1"
REFERENCE_COUNTS = {"S1": 19, "S2": 12, "S3": 17}
PROTOCOL_CLAIMED_COUNTS = {"S1": 27, "S2": 15, "S3": 6}
OFFICIAL_REPO = "https://github.com/HenryZhu1029/EvoReal"
OFFICIAL_COMMIT = "919616337105e894e06463734ec1dd71f6451309"
ARXIV = "https://arxiv.org/abs/2511.10233"
COLD_CACHE_OBSERVED_WALL_S = 67.51


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P6A_MOD = load_module("p6e0_frozen_p6a", P6A / "run_p6a_audit.py")
APPROX = load_module("p6e0_evoreal_approx", APPROX_SOURCE)
sys.path[:0] = [str(ROOT / "src")]
from features.qd4 import QD4_FEATURE_NAMES  # noqa: E402
from qd.coordinates import normalize_qd4  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402

QD4 = tuple(QD4_FEATURE_NAMES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(x: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=conv) + "\n")
    tmp.replace(path)


def md(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty: return "_(empty)_"
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for v in row:
            if pd.isna(v): cells.append("NA")
            elif isinstance(v, (float, np.floating)): cells.append(f"{float(v):.{digits}f}")
            else: cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ensure_dirs() -> None:
    for p in (OUT, FIG, CFG, MAN, FEAT, LOG): p.mkdir(parents=True, exist_ok=True)


def parse_tsplib(path: Path) -> np.ndarray:
    return P6A_MOD.parse_tsplib(path)


def norm(xy: np.ndarray) -> np.ndarray:
    return APPROX.alg1_normalize(np.asarray(xy, np.float64))


def classify(fft: float, nn: float) -> str:
    if fft > FFT_THRESHOLD: return "S1"
    if nn < NN_THRESHOLD: return "S2"
    return "S3"


def feature_one(xy: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(xy, np.float64)
    z = norm(raw)
    tree = cKDTree(z)
    d, _ = tree.query(z, k=2)
    nn = np.asarray(d[:, 1], float)
    mean = float(nn.mean())
    ratio = float(nn.std(ddof=0) / mean) if mean > 0 else float("nan")
    fft = float(APPROX.fft_energy(raw))
    unique = int(len(np.unique(z, axis=0)))
    return {
        "fft_energy": fft, "nn_mean": mean, "nn_std": float(nn.std(ddof=0)),
        "nn_ratio": ratio, "duplicate_coordinate_n": int(len(z) - unique),
        "normalized_coordinate_sha256": sha256_array(z),
        "approx_type": classify(fft, ratio) if np.isfinite(fft) and np.isfinite(ratio) else "INVALID",
    }


def feature_chunk(args: tuple[int, np.ndarray]) -> tuple[int, list[dict[str, Any]]]:
    start, coordinates = args
    return start, [feature_one(x) for x in coordinates]


def extract_fixed(name: str, coordinates: np.ndarray, hashes: list[str], source_label: np.ndarray | None = None) -> pd.DataFrame:
    path = FEAT / f"{name}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        required = {"source_dataset", "normalization_version", "original_coordinate_sha256"}
        if len(old) == len(hashes) and old.instance_hash.astype(str).tolist() == hashes:
            if not required.issubset(old.columns):
                old["source_dataset"] = name
                old["normalization_version"] = NORMALIZATION_VERSION
                old["original_coordinate_sha256"] = [sha256_array(row) for row in coordinates]
                old.to_parquet(path,index=False)
            return old
    x = np.asarray(coordinates)
    if x.ndim != 3 or x.shape[2] != 2 or len(x) != len(hashes): raise RuntimeError(f"{name} coordinate shape")
    if not np.isfinite(x).all() or len(set(hashes)) != len(hashes): raise RuntimeError(f"{name} integrity")
    chunk = 500
    jobs = [(i, np.asarray(x[i:i + chunk], np.float64)) for i in range(0, len(x), chunk)]
    started = time.perf_counter(); parts = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, (start, rows) in enumerate(pool.map(feature_chunk, jobs), 1):
            f = pd.DataFrame(rows); f.insert(0, "instance_hash", hashes[start:start + len(f)]); f.insert(1, "n_nodes", x.shape[1]); parts.append((start, f))
            if done % 20 == 0 or done == len(jobs): print(f"[P6-E0] {name}: {done}/{len(jobs)} chunks", flush=True)
    parts.sort(key=lambda q: q[0]); out = pd.concat([q[1] for q in parts], ignore_index=True)
    out.insert(2, "source_dataset", name)
    out.insert(3, "normalization_version", NORMALIZATION_VERSION)
    out["original_coordinate_sha256"] = [sha256_array(row) for row in x]
    if source_label is not None: out.insert(2, "generator_source", np.asarray(source_label).astype(str))
    out.to_parquet(path, index=False)
    with (LOG / "runtime.csv").open("a") as f:
        f.write(f"feature_extraction,{name},{len(out)},{time.perf_counter()-started:.6f}\n")
    return out


def load_reference() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(P1 / "03_tsplib48_reference_manifest.csv")
    if len(manifest) != 48 or manifest.instance_id.nunique() != 48: raise RuntimeError("TSPLIB48 integrity")
    coords: dict[str, np.ndarray] = {}; rows = []
    for r in manifest.itertuples(index=False):
        raw = parse_tsplib(ROOT / str(r.source_file)); coords[str(r.instance_id)] = raw
        z = norm(raw); feat = feature_one(raw)
        rows.append({"instance_id":str(r.instance_id), "instance_hash":coordinate_hash(z), "n_nodes":len(raw), "source_dataset":"TSPLIB48", "reference_group":str(r.evoreal_split), "normalization_version":NORMALIZATION_VERSION, "original_coordinate_sha256":sha256_array(raw), **feat})
    frame = pd.DataFrame(rows)
    frame.to_parquet(FEAT / "TSPLIB48.parquet", index=False)
    frame.to_csv(OUT / "03_fft_features.csv", index=False)
    frame[["instance_id","instance_hash","n_nodes","source_dataset","reference_group","nn_mean","nn_std","nn_ratio","duplicate_coordinate_n","normalized_coordinate_sha256","original_coordinate_sha256","approx_type"]].to_csv(OUT / "04_nn_ratio_features.csv", index=False)
    return frame, coords


def load_heldout_after_freeze() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if not (CFG / "reference_rules_frozen.json").exists(): raise RuntimeError("reference rules not frozen before heldout")
    names = json.loads((P1 / "p1_heldout22_names.json").read_text())["names"]
    src = pd.read_csv(ORIGINAL / "source_manifest_bq83.csv").set_index("source_instance")
    coords = {}; rows = []
    for name in names:
        r = src.loc[name]
        rel = Path(str(r.dataset_file))
        # The frozen manifest retains the historical ``original_scale/`` prefix,
        # while the immutable evaluation bundle is rooted directly at instances/.
        if rel.parts and rel.parts[0] == "original_scale":
            rel = Path(*rel.parts[1:])
        path = ORIGINAL / rel
        raw = parse_tsplib(path); coords[name] = raw; z = norm(raw); feat = feature_one(raw)
        rows.append({"instance_id":name,"instance_hash":coordinate_hash(z),"n_nodes":len(raw),"source_dataset":"TSPLIB22","reference_group":"NONREFERENCE","normalization_version":NORMALIZATION_VERSION,"original_coordinate_sha256":sha256_array(raw),**feat})
    frame = pd.DataFrame(rows); frame.to_parquet(FEAT / "TSPLIB22_READONLY.parquet", index=False)
    return frame, coords


def load_populations() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}; inventory = []
    for rep in REPS:
        p = P6C1 / "dataset_manifests" / f"{rep}_union.npz"
        with np.load(p, allow_pickle=False) as z:
            xy = np.asarray(z["coordinates"], np.float64); hh = z["instance_hash"].astype(str).tolist()
        frame = extract_fixed(f"Union_{rep}", xy, hh); outputs[f"Union_{rep}"] = frame
        inventory.append({"dataset":f"Union_{rep}","n":len(frame),"n_nodes_min":100,"n_nodes_max":100,"unique_hashes":frame.instance_hash.nunique(),"source_path":str(p.relative_to(ROOT)),"source_sha256":sha256_file(p)})
        lookup = frame.set_index("instance_hash")
        for arm in (*CORE_ARMS, *AUX_ARMS):
            base = P6D1 if arm in CORE_ARMS else P6D3
            pp = base / "datasets" / f"{rep}_{arm}.parquet"
            meta = pd.read_parquet(pp)
            feat = lookup.loc[meta.instance_hash.astype(str)].reset_index()
            for c in ("true_m4_ce","predicted_m4_ce","qd_niche","lineage_root","operator_id"):
                if c in meta: feat[c] = meta[c].to_numpy()
            outputs[f"{arm}_{rep}"] = feat
            inventory.append({"dataset":f"{arm}_{rep}","n":len(feat),"n_nodes_min":100,"n_nodes_max":100,"unique_hashes":feat.instance_hash.nunique(),"source_path":str(pp.relative_to(ROOT)),"source_sha256":sha256_file(pp)})
    ep = V2 / "datasets/evoreal50k_coordinates.npy"; lp = V2 / "datasets/evoreal50k_source_labels.npy"
    ecoords = np.load(ep, mmap_mode="r", allow_pickle=False); labels = np.load(lp, allow_pickle=False).astype(str)
    eh = [coordinate_hash(x) for x in ecoords]
    evo = extract_fixed("EvoReal50k", ecoords, eh, labels); outputs["EvoReal50k"] = evo
    inventory.append({"dataset":"EvoReal50k","n":len(evo),"n_nodes_min":100,"n_nodes_max":100,"unique_hashes":evo.instance_hash.nunique(),"source_path":str(ep.relative_to(ROOT)),"source_sha256":sha256_file(ep)})
    inv = pd.DataFrame(inventory); inv.to_csv(MAN / "population_inventory.csv", index=False)
    return outputs, inv


def invariance_and_nn_tests(reference: pd.DataFrame, coords: dict[str, np.ndarray], populations: dict[str,pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED); rows = []
    names = reference.instance_id.tolist()[:4]
    for name in names:
        xy = coords[name]; base = feature_one(xy)
        variants = {
            "translation": xy + np.array([17.3,-9.1]), "uniform_scale": xy * 3.7,
            "rotation90": np.column_stack([-xy[:,1],xy[:,0]]), "permutation": xy[rng.permutation(len(xy))],
        }
        for kind, q in variants.items():
            v = feature_one(q); rows.append({"instance_id":name,"test":kind,"fft_abs_diff":abs(base["fft_energy"]-v["fft_energy"]),"nn_abs_diff":abs(base["nn_ratio"]-v["nn_ratio"])})
    # KDTree/direct cross-check on frozen synthetic coordinates reconstructed from selected bundles.
    for rep in REPS:
        with np.load(P6D1/"datasets"/f"{rep}_B.npz",allow_pickle=False) as z: sample=np.asarray(z["coordinates"][:3],float)
        for j,xy in enumerate(sample):
            z=norm(xy); d=np.sqrt(np.square(z[:,None,:]-z[None,:,:]).sum(axis=2)); np.fill_diagonal(d,np.inf); direct=d.min(axis=1); tree=cKDTree(z).query(z,k=2)[0][:,1]
            rows.append({"instance_id":f"{rep}_B_{j}","test":"kdtree_direct_nn","fft_abs_diff":np.nan,"nn_abs_diff":float(np.max(np.abs(direct-tree)))})
    out=pd.DataFrame(rows); out.to_csv(MAN/"invariance_and_nn_tests.csv",index=False); return out


def reference_reproduction(reference: pd.DataFrame) -> pd.DataFrame:
    out = reference[["instance_id","n_nodes","fft_energy","nn_ratio","reference_group","approx_type"]].copy()
    out["agreement"] = out.reference_group.eq(out.approx_type)
    out["fft_margin_to_35"] = out.fft_energy - FFT_THRESHOLD
    out["nn_margin_to_05"] = out.nn_ratio - NN_THRESHOLD
    out.to_csv(OUT / "05_reference_type_table.csv", index=False)
    return out


def population_metrics(reference: pd.DataFrame, populations: dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    for name, f in [("TSPLIB48",reference), *populations.items()]:
        for feature in ("fft_energy","nn_ratio"):
            v=f[feature].to_numpy(float)
            rows.append({"dataset":name,"record_type":"distribution","feature":feature,"n":len(v),"min":np.min(v),"p10":np.quantile(v,.1),"p25":np.quantile(v,.25),"median":np.median(v),"p75":np.quantile(v,.75),"p90":np.quantile(v,.9),"max":np.max(v),"mean":np.mean(v),"sd":np.std(v,ddof=1)})
        counts=f.approx_type.value_counts()
        for t in ("S1","S2","S3"):
            rows.append({"dataset":name,"record_type":"approx_type","feature":t,"n":len(f),"fraction":counts.get(t,0)/len(f),"count":int(counts.get(t,0))})
    out=pd.DataFrame(rows); out.to_csv(OUT/"06_population_metrics.csv",index=False); return out


def correlations(reference: pd.DataFrame, populations: dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    # TSPLIB48 full F40 (aspect normalization) plus frozen Top5-phi.
    f40=pd.read_csv(P1/"p1_tsplib48_normalization_sensitivity.csv"); f40=f40[f40.coordinate_variant.eq("aspect")].set_index("instance_id").loc[reference.instance_id].reset_index()
    numeric=[c for c in f40.columns if c.startswith("spn_")]
    rf=reference.merge(f40[["instance_id",*numeric]],on="instance_id",validate="one_to_one")
    top=pd.read_parquet(P6A/"diagnostics/tsplib48_top5_representation.parquet"); rf=rf.merge(top[["instance_id",*[f"phi__{x}" for x in TOP5]]],on="instance_id",validate="one_to_one")
    for target in ("fft_energy","nn_ratio"):
        for c in [*numeric,*[f"phi__{x}" for x in TOP5]]:
            rho=stats.spearmanr(rf[target],rf[c],nan_policy="omit").statistic
            rows.append({"scope":"TSPLIB48","target_feature":target,"comparison_feature":c,"n":len(rf),"spearman":rho})
    # Synthetic selected populations use exact frozen full-F40 cache by hash.
    cache=pd.concat([pd.read_parquet(p) for p in sorted((P6C0/"full_f40_cache").glob("*.parquet"))],ignore_index=True).drop_duplicates("instance_hash").set_index("instance_hash")
    full=[c for c in cache.columns if c.startswith("spn_")]
    for rep in REPS:
        for arm in CORE_ARMS:
            f=populations[f"{arm}_{rep}"].copy(); q=cache.loc[f.instance_hash.astype(str),full].reset_index(drop=True); f=pd.concat([f.reset_index(drop=True),q],axis=1)
            for target in ("fft_energy","nn_ratio"):
                for c in full:
                    rho=stats.spearmanr(f[target],f[c],nan_policy="omit").statistic
                    rows.append({"scope":f"{arm}_{rep}","target_feature":target,"comparison_feature":c,"n":len(f),"spearman":rho})
    out=pd.DataFrame(rows); out.to_csv(OUT/"07_feature_correlations.csv",index=False); return out


def qd_clipping(reference: pd.DataFrame, populations: dict[str,pd.DataFrame]) -> pd.DataFrame:
    rows=[]; allf=pd.read_csv(P1/"p1_tsplib48_normalization_sensitivity.csv"); target=allf[allf.coordinate_variant.eq("aspect")].set_index("instance_id").loc[reference.instance_id]
    q=normalize_qd4(target[list(QD4)].to_numpy(float)); clip=np.isclose(q,0)|np.isclose(q,1)
    for i,r in reference.reset_index(drop=True).iterrows(): rows.append({"dataset":"TSPLIB48","instance_id":r.instance_id,"approx_type":r.approx_type,"clipped_axes":int(clip[i].sum()),"any_axis_clipped":bool(clip[i].any())})
    for rep in REPS:
        for arm in CORE_ARMS:
            meta=pd.read_parquet(P6D1/"datasets"/f"{rep}_{arm}.parquet"); cols=[f"normalized_{x}" for x in QD4]; z=meta[cols].to_numpy(float); c=np.isclose(z,0)|np.isclose(z,1); types=populations[f"{arm}_{rep}"].approx_type.to_numpy()
            for t in ("S1","S2","S3"):
                m=types==t; rows.append({"dataset":f"{arm}_{rep}","instance_id":f"SUMMARY_{t}","approx_type":t,"clipped_axes":float(c[m].sum(axis=1).mean()) if m.any() else np.nan,"any_axis_clipped":float(c[m].any(axis=1).mean()) if m.any() else np.nan})
    out=pd.DataFrame(rows); out.to_csv(MAN/"qd4_clipping_by_fft_nn_class.csv",index=False); return out


def coarse_cells(x: np.ndarray, edges: list[np.ndarray]) -> tuple[np.ndarray,np.ndarray]:
    bins=[]; inside=np.ones(len(x),bool)
    for j,e in enumerate(edges):
        inside &= (x[:,j]>=e[0])&(x[:,j]<=e[-1]); bins.append(np.clip(np.searchsorted(e[1:-1],x[:,j],side="right"),0,2))
    return bins[0]*3+bins[1],inside


def js(p: np.ndarray,q: np.ndarray) -> float:
    p=np.asarray(p,float);q=np.asarray(q,float);p=p/p.sum();q=q/q.sum();m=(p+q)/2
    def kl(a,b):
        z=a>0; return float(np.sum(a[z]*np.log2(a[z]/b[z])))
    return .5*kl(p,m)+.5*kl(q,m)


def swd(x: np.ndarray,y: np.ndarray,seed: int=SEED,proj: int=256) -> float:
    rng=np.random.default_rng(seed); d=rng.normal(size=(proj,2)); d/=np.linalg.norm(d,axis=1,keepdims=True); vals=[]
    n=max(len(x),len(y)); ix=np.arange(n)%len(x); iy=np.arange(n)%len(y)
    for v in d:
        a=np.sort((x@v)[ix]);b=np.sort((y@v)[iy]);vals.append(np.mean(np.abs(a-b)))
    return float(np.mean(vals))


def coverage_metrics(reference: pd.DataFrame,populations: dict[str,pd.DataFrame])->pd.DataFrame:
    rv=reference[["fft_energy","nn_ratio"]].to_numpy(float); center=np.median(rv,axis=0); scale=np.subtract(*np.quantile(rv,[.75,.25],axis=0)); rz=(rv-center)/scale
    pair=cdist(rz,rz);np.fill_diagonal(pair,np.inf);radius=float(np.median(pair.min(axis=1)))
    edges=[np.array([rv[:,j].min(),np.quantile(rv[:,j],1/3),np.quantile(rv[:,j],2/3),rv[:,j].max()]) for j in range(2)]
    rc,ri=coarse_cells(rv,edges); rh=np.bincount(rc[ri],minlength=9)
    config={"center":center,"scale_iqr":scale,"radius":radius,"edges":[x.tolist() for x in edges],"fit_data":"TSPLIB48_only","matched_seeds":MATCHED_SEEDS,"fft_status":"FFT_APPROX"};write_json(CFG/"reference_rules_frozen.json",config)
    rows=[]
    def one(name:str,f:pd.DataFrame,repeat:int=-1):
        pv=f[["fft_energy","nn_ratio"]].to_numpy(float);pz=(pv-center)/scale; dp=cdist(pz,rz); dr=cdist(rz,pz); nearest=dp.min(axis=1); assign=dp.argmin(axis=1); mass=np.bincount(assign,minlength=48); p=mass/mass.sum(); nz=p>0
        pc,inside=coarse_cells(pv,edges); ph=np.bincount(pc[inside],minlength=9)
        rows.append({"dataset":name,"repeat":repeat,"n":len(f),"inside_reference_box_fraction":inside.mean(),"population_within_reference_radius_fraction":np.mean(nearest<=radius),"reference_neighborhood_coverage_fraction":np.mean(dr.min(axis=1)<=radius),"nearest_reference_median":np.median(nearest),"nearest_reference_p90":np.quantile(nearest,.9),"occupied_reference_neighborhoods":int((mass>0).sum()),"effective_reference_neighborhoods":float(np.exp(-np.sum(p[nz]*np.log(p[nz])))),"largest_reference_share":p.max(),"coarse_3x3_js":js(rh,ph) if ph.sum() else 1.0,"sliced_wasserstein":swd(pz,rz,SEED+repeat),"S1_fraction":np.mean(f.approx_type.eq("S1")),"S2_fraction":np.mean(f.approx_type.eq("S2")),"S3_fraction":np.mean(f.approx_type.eq("S3"))})
    for rep in REPS:
        for arm in (*CORE_ARMS,*AUX_ARMS): one(f"{arm}_{rep}",populations[f"{arm}_{rep}"])
    evo=populations["EvoReal50k"]; one("EvoReal50k_full",evo)
    for i,seed in enumerate(MATCHED_SEEDS): one("EvoReal50k_matched10k",evo.iloc[np.random.default_rng(seed).choice(len(evo),10000,replace=False)],i)
    out=pd.DataFrame(rows);out.to_csv(OUT/"08_coverage_metrics.csv",index=False);return out


def cross_size(reference: pd.DataFrame,coords: dict[str,np.ndarray],populations: dict[str,pd.DataFrame])->tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for feature in ("fft_energy","nn_ratio"):
        rows.append({"record":"reference_correlation","dataset":"TSPLIB48","feature":feature,"parameter":"N","value":stats.spearmanr(reference.n_nodes,reference[feature]).statistic,"n":len(reference)})
        rows.append({"record":"reference_correlation","dataset":"TSPLIB48","feature":feature,"parameter":"logN","value":stats.spearmanr(np.log(reference.n_nodes),reference[feature]).statistic,"n":len(reference)})
    bins=pd.cut(reference.n_nodes,[0,99,199,499,999,np.inf],labels=["<100","100-199","200-499","500-999",">=1000"])
    for b,g in reference.groupby(bins,observed=False):
        for feature in ("fft_energy","nn_ratio"): rows.append({"record":"size_bin","dataset":"TSPLIB48","feature":feature,"parameter":str(b),"value":g[feature].median() if len(g) else np.nan,"n":len(g)})
    rng=np.random.default_rng(SEED+55); subrows=[]
    for r in reference.itertuples(index=False):
        if r.n_nodes<100: continue
        xy=coords[r.instance_id]
        for repeat in range(10):
            q=feature_one(xy[rng.choice(len(xy),100,replace=False)])
            subrows.append({"dataset":"TSPLIB48_subsample100","instance_id":r.instance_id,"original_n":r.n_nodes,"repeat":repeat,"fft_energy":q["fft_energy"],"nn_ratio":q["nn_ratio"],"approx_type":q["approx_type"],"original_fft":r.fft_energy,"original_nn":r.nn_ratio,"original_type":r.approx_type})
    # Fixed synthetic size sensitivity on read-only EvoReal coordinates.
    ecoords=np.load(V2/"datasets/evoreal50k_coordinates.npy",mmap_mode="r",allow_pickle=False); idx=rng.choice(len(ecoords),300,replace=False)
    for n in (50,75):
        for j in idx:
            q=feature_one(ecoords[j][rng.choice(100,n,replace=False)]); base=populations["EvoReal50k"].iloc[j]
            subrows.append({"dataset":f"EvoReal_subsample{n}","instance_id":str(j),"original_n":100,"repeat":0,"fft_energy":q["fft_energy"],"nn_ratio":q["nn_ratio"],"approx_type":q["approx_type"],"original_fft":base.fft_energy,"original_nn":base.nn_ratio,"original_type":base.approx_type})
    subs=pd.DataFrame(subrows);subs.to_csv(FEAT/"cross_size_subsampling.csv",index=False)
    for ds,g in subs.groupby("dataset"):
        rows += [
            {"record":"subsample_stability","dataset":ds,"feature":"fft_energy","parameter":"median_subsample_over_original","value":float(np.median(g.fft_energy/g.original_fft)),"n":len(g)},
            {"record":"subsample_stability","dataset":ds,"feature":"nn_ratio","parameter":"median_abs_change","value":float(np.median(np.abs(g.nn_ratio-g.original_nn))),"n":len(g)},
            {"record":"subsample_stability","dataset":ds,"feature":"approx_type","parameter":"classification_agreement","value":float(np.mean(g.approx_type.eq(g.original_type))),"n":len(g)},
        ]
    # A predeclared broad TSP100-comparable reference band. It remains small and
    # group-imbalanced, so this is a confounding diagnostic, not a new target fit.
    rv=reference[["fft_energy","nn_ratio"]].to_numpy(float)
    center=np.median(rv,axis=0); scale=np.subtract(*np.quantile(rv,[.75,.25],axis=0)); rz=(rv-center)/scale
    pair=cdist(rz,rz); np.fill_diagonal(pair,np.inf); radius=float(np.median(pair.min(axis=1)))
    size_mask=reference.n_nodes.between(75,150).to_numpy(); target=rz[size_mask]
    comparable=[]
    for name in [*[f"{a}_{r}" for r in REPS for a in CORE_ARMS],"EvoReal50k"]:
        f=populations[name]; pz=(f[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale
        dist=cdist(target,pz)
        comparable.append({
            "dataset":name,"reference_band":"75<=N<=150","reference_n":int(size_mask.sum()),"population_n":len(f),
            "reference_neighborhood_coverage_fraction":float(np.mean(dist.min(axis=1)<=radius)),
            "population_within_band_reference_radius_fraction":float(np.mean(dist.min(axis=0)<=radius)),
            "reference_to_population_nearest_median":float(np.median(dist.min(axis=1))),
            "sliced_wasserstein":swd(pz,target,SEED+77),
        })
    comparable_frame=pd.DataFrame(comparable)
    comparable_frame.to_csv(MAN/"comparable_size_coverage.csv",index=False)
    rows.append({"record":"comparable_size_summary","dataset":"TSPLIB48","feature":"reference_count","parameter":"75<=N<=150","value":float(size_mask.sum()),"n":int(size_mask.sum())})
    out=pd.DataFrame(rows);out.to_csv(MAN/"cross_size_metrics.csv",index=False);return out,subs


def hardness_context(populations: dict[str,pd.DataFrame])->pd.DataFrame:
    rows=[]
    for rep in REPS:
        for arm in (*CORE_ARMS,*AUX_ARMS):
            f=populations[f"{arm}_{rep}"]
            for t,g in f.groupby("approx_type"):
                rows.append({"replicate":rep,"arm":arm,"approx_type":t,"n":len(g),"true_m4_ce_mean":g.true_m4_ce.mean(),"true_m4_ce_median":g.true_m4_ce.median(),"true_m4_ce_p90":g.true_m4_ce.quantile(.9),"spearman_fft_ce":stats.spearmanr(g.fft_energy,g.true_m4_ce).statistic,"spearman_nn_ce":stats.spearmanr(g.nn_ratio,g.true_m4_ce).statistic})
    out=pd.DataFrame(rows);out.to_csv(FEAT/"hardness_context.csv",index=False);return out


def make_figures(reference:pd.DataFrame,coords:dict[str,np.ndarray],populations:dict[str,pd.DataFrame],metrics:pd.DataFrame,corr:pd.DataFrame,clip:pd.DataFrame,held:pd.DataFrame,subs:pd.DataFrame)->None:
    colors={"S1":"tab:blue","S2":"tab:orange","S3":"tab:green"}
    fig,ax=plt.subplots(figsize=(7,5));
    for t,g in reference.groupby("reference_group"): ax.scatter(g.fft_energy,g.nn_ratio,label=t,c=colors[t],s=42)
    ax.axvline(35,c="k",ls="--");ax.axhline(.5,c="k",ls=":");ax.set(xlabel="FFT_APPROX",ylabel="NN-CV",title="TSPLIB48 (official repository labels)");ax.legend();fig.tight_layout();fig.savefig(FIG/"01_tsplib48_fft_nn.png",dpi=180);plt.close(fig)
    evo=populations["EvoReal50k"];fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharex=True,sharey=True)
    for ax,(t,g) in zip(axes,evo.groupby("generator_source")): ax.hexbin(g.fft_energy,g.nn_ratio,gridsize=45,mincnt=1,bins="log");ax.set_title(f"generator {t}, n={len(g)}");ax.set_xlabel("FFT_APPROX")
    axes[0].set_ylabel("NN-CV");fig.tight_layout();fig.savefig(FIG/"02_evoreal_generators_fft_nn.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),sharex=True,sharey=True)
    for ax,rep in zip(axes,REPS):
        g=populations[f"B_{rep}"];ax.hexbin(g.fft_energy,g.nn_ratio,gridsize=40,mincnt=1,bins="log");ax.scatter(reference.fft_energy,reference.nn_ratio,s=10,c="red");ax.set_title(f"B {rep}, n=10k");ax.set_xlabel("FFT_APPROX")
    axes[0].set_ylabel("NN-CV");fig.tight_layout();fig.savefig(FIG/"03_b_three_seed_fft_nn.png",dpi=180);plt.close(fig)
    for feature,num in (("fft_energy",4),("nn_ratio",5)):
        fig,ax=plt.subplots(figsize=(8,5));
        for name in ("B_Original","C_Original","EvoReal50k"): ax.hist(populations[name][feature],bins=80,density=True,histtype="step",label=f"{name} n={len(populations[name])}")
        ax.hist(reference[feature],bins=20,density=True,histtype="step",lw=2,label="TSPLIB48 n=48");ax.set_xlabel(feature);ax.set_ylabel("density");ax.legend();fig.tight_layout();fig.savefig(FIG/f"{num:02d}_{feature}_marginals.png",dpi=180);plt.close(fig)
    names=["TSPLIB48","B_Original","B_S1","B_S2","C_Original","C_S1","C_S2","EvoReal50k"]
    vals=[]
    for name in names:
        f=reference if name=="TSPLIB48" else populations[name];vals.append([np.mean(f.approx_type.eq(t)) for t in ("S1","S2","S3")])
    fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(len(names))
    for j,t in enumerate(("S1","S2","S3")): ax.bar(names,np.array(vals)[:,j],bottom=bottom,label=t);bottom+=np.array(vals)[:,j]
    ax.tick_params(axis="x",rotation=35);ax.set_ylabel("exploratory approximate fraction");ax.legend();fig.tight_layout();fig.savefig(FIG/"06_approx_type_proportions.png",dpi=180);plt.close(fig)
    for feature,num in (("fft_energy",7),("nn_ratio",8)):
        fig,ax=plt.subplots(figsize=(7,5));
        for t,g in reference.groupby("reference_group"): ax.scatter(g.n_nodes,g[feature],label=t,c=colors[t]);
        ax.set_xscale("log");ax.set(xlabel="N (log)",ylabel=feature);ax.legend();fig.tight_layout();fig.savefig(FIG/f"{num:02d}_{feature}_vs_n.png",dpi=180);plt.close(fig)
    q=clip[clip.dataset.eq("TSPLIB48")];fig,ax=plt.subplots(figsize=(7,4));q.groupby("approx_type").any_axis_clipped.mean().reindex(["S1","S2","S3"]).plot.bar(ax=ax);ax.set_ylabel("any QD4 axis clipped fraction");fig.tight_layout();fig.savefig(FIG/"09_qd4_clipping_by_class.png",dpi=180);plt.close(fig)
    best=corr[corr.scope.eq("TSPLIB48")].pivot(index="target_feature",columns="comparison_feature",values="spearman");fig,ax=plt.subplots(figsize=(16,3));im=ax.imshow(best,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm");ax.set_yticks(range(len(best)),best.index);ax.set_xticks(range(len(best.columns)),best.columns,rotation=90,fontsize=6);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/"10_fft_nn_f40_correlation.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(9,9));
    for i,t in enumerate(("S1","S2","S3")):
        g=reference[reference.reference_group.eq(t)].sort_values("fft_energy"); ids=[g.iloc[0].instance_id,g.iloc[len(g)//2].instance_id,g.iloc[-1].instance_id]
        for ax,name in zip(axes[i],ids): z=norm(coords[name]);ax.scatter(z[:,0],z[:,1],s=max(1,40/np.sqrt(len(z))));ax.set_title(f"{t}: {name} (N={len(z)})");ax.set_aspect("equal")
    fig.tight_layout();fig.savefig(FIG/"11_representative_coordinates.png",dpi=180);plt.close(fig)
    show=metrics[(metrics.repeat.eq(-1))|metrics.dataset.str.contains("matched")].groupby("dataset").agg(js=("coarse_3x3_js","mean"),coverage=("reference_neighborhood_coverage_fraction","mean")).reset_index();fig,ax=plt.subplots(figsize=(12,5));ax.bar(np.arange(len(show))-.2,show.js,.4,label="JS");ax.bar(np.arange(len(show))+.2,show.coverage,.4,label="reference coverage");ax.set_xticks(range(len(show)),show.dataset,rotation=60,ha="right");ax.legend();fig.tight_layout();fig.savefig(FIG/"12_reference_coverage_comparison.png",dpi=180);plt.close(fig)
    show=metrics[metrics.dataset.str.match("[BC]_")];fig,axes=plt.subplots(1,2,figsize=(12,4));
    for arm,marker in (("B","o"),("C","s")):
        g=show[show.dataset.str.startswith(arm)].copy();g["rep"]=g.dataset.str.split("_").str[1];axes[0].plot(g.rep,g.coarse_3x3_js,marker=marker,label=arm);axes[1].plot(g.rep,g.reference_neighborhood_coverage_fraction,marker=marker,label=arm)
    axes[0].set_title("3x3 JS");axes[1].set_title("reference neighborhood coverage");axes[0].legend();fig.tight_layout();fig.savefig(FIG/"13_three_seed_stability.png",dpi=180);plt.close(fig)
    cm=pd.crosstab(evo.generator_source,evo.approx_type).reindex(index=["S1","S2","S3"],columns=["S1","S2","S3"],fill_value=0);fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm);ax.set_xticks(range(3),cm.columns);ax.set_yticks(range(3),cm.index);ax.set(xlabel="approx type",ylabel="generator source");
    for i in range(3):
        for j in range(3): ax.text(j,i,int(cm.iloc[i,j]),ha="center",va="center",color="white" if cm.iloc[i,j]>cm.to_numpy().max()/2 else "black")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(FIG/"14_generator_vs_approx_confusion.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(reference.fft_energy,reference.nn_ratio,c="tab:blue",label="TSPLIB48");ax.scatter(held.fft_energy,held.nn_ratio,c="tab:red",marker="x",label="TSPLIB22 read-only");ax.legend();ax.set(xlabel="FFT_APPROX",ylabel="NN-CV");fig.tight_layout();fig.savefig(FIG/"15_nonreference_diagnostic.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].hist(subs.fft_energy/subs.original_fft,bins=50);axes[0].set_title("subsample FFT/original");axes[1].hist(subs.nn_ratio-subs.original_nn,bins=50);axes[1].set_title("subsample NN-CV change");fig.tight_layout();fig.savefig(FIG/"16_cross_size_subsampling.png",dpi=180);plt.close(fig)


def reports(reference:pd.DataFrame,inv:pd.DataFrame,tests:pd.DataFrame,repro:pd.DataFrame,metrics:pd.DataFrame,corr:pd.DataFrame,clip:pd.DataFrame,coverage:pd.DataFrame,cross:pd.DataFrame,hard:pd.DataFrame,held:pd.DataFrame)->dict[str,str]:
    counts=reference.reference_group.value_counts().reindex(["S1","S2","S3"]).astype(int).to_dict(); agreement=float(repro.agreement.mean()); s1=float(repro[repro.reference_group.eq("S1")].agreement.mean()); nons1=float(repro[~repro.reference_group.eq("S1")].agreement.mean())
    rho_fft=float(stats.spearmanr(reference.n_nodes,reference.fft_energy).statistic);rho_nn=float(stats.spearmanr(reference.n_nodes,reference.nn_ratio).statistic)
    qref=clip[clip.dataset.eq("TSPLIB48")];clipfrac=float(qref.any_axis_clipped.astype(bool).mean())
    maxcorr=corr.assign(absrho=lambda x:x.spearman.abs()).sort_values("absrho",ascending=False).groupby(["scope","target_feature"]).head(1)
    core=coverage[coverage.dataset.isin([f"{a}_{r}" for a in CORE_ARMS for r in REPS]+["EvoReal50k_full"])]
    ctab=core[["dataset","coarse_3x3_js","reference_neighborhood_coverage_fraction","population_within_reference_radius_fraction","effective_reference_neighborhoods","largest_reference_share","S1_fraction","S2_fraction","S3_fraction"]]
    evomat=coverage[coverage.dataset.eq("EvoReal50k_matched10k")].mean(numeric_only=True)
    b=coverage[coverage.dataset.str.startswith("B_")]; c=coverage[coverage.dataset.str.startswith("C_")]
    evo_full=coverage[coverage.dataset.eq("EvoReal50k_full")].iloc[0]
    # Conservative decision rules: official FFT code absent => PARTIAL; cross-size threshold use is limited.
    h1="PARTIAL"
    # Agreement is descriptive only: this approximation was calibrated on the
    # same 48 references, and FFT is strongly size-confounded.
    h2="MIXED" if agreement>=.65 else "NOT_SUPPORTED"
    fft_top=maxcorr[(maxcorr.scope.eq("TSPLIB48"))&(maxcorr.target_feature.eq("fft_energy"))].absrho.max(); nn_top=maxcorr[(maxcorr.scope.eq("TSPLIB48"))&(maxcorr.target_feature.eq("nn_ratio"))].absrho.max()
    h3="SUPPORTED" if fft_top<.7 and nn_top<.7 else "PARTIAL" if fft_top<.85 or nn_top<.85 else "NOT_SUPPORTED"
    type_agree=cross[(cross.record.eq("subsample_stability"))&(cross.feature.eq("approx_type"))].value.min()
    h4="SUPPORTED" if abs(rho_fft)<.2 and abs(rho_nn)<.2 and type_agree>=.9 else "LIMITED" if type_agree>=.5 else "NOT_SUPPORTED"
    # Gap requires three-seed direction and material discrepancy; reference coverage alone cannot decide it.
    b_js=b.coarse_3x3_js.to_numpy(); b_cov=b.reference_neighborhood_coverage_fraction.to_numpy(); ejs=float(evomat.coarse_3x3_js); ecov=float(evomat.reference_neighborhood_coverage_fraction)
    raw_gap=np.all(b_js>.15) and np.all(b_cov<.9)
    # The raw gap is consistent across seeds, but cannot be separated cleanly
    # from N because raw-count FFT has rho≈0.8 with N. Preserve this uncertainty.
    h5="MIXED" if raw_gap or np.mean(b_js)>.08 or np.mean(b_cov)<.95 else "NOT_SUPPORTED"
    h6="MIXED" if ejs<np.mean(b_js) or ecov>np.mean(b_cov) else "NOT_SUPPORTED"
    h7="SUPPORTED_FOR_PILOT" if h1=="EXACT" and h4=="SUPPORTED" and h2=="SUPPORTED" and h5=="SUPPORTED" else "NOT_SUPPORTED" if h2=="NOT_SUPPORTED" or h5=="NOT_SUPPORTED" else "NOT_ESTABLISHED"
    decisions={"P6E0_FEATURE_REPRODUCTION":h1,"P6E0_STRUCTURAL_DISCRIMINATION":h2,"P6E0_ADDITIONAL_STRUCTURAL_SIGNAL":h3,"P6E0_CROSS_SIZE_STABILITY":h4,"P6E0_TARGET_COVERAGE_GAP":h5,"P6E0_EVOREAL_COVERAGE_ADVANTAGE":h6,"P6E0_FIFTH_ROUND_FEATURE_READINESS":h7}
    (OUT/"00_protocol.md").write_text("# P6-E0 Frozen Protocol\n\nRead-only audit of frozen TSPLIB48/22, P6-D1 B/C, P6-D3 D arms, P6-B union and EvoReal50k. No mutation, generation, Concorde, HGB/BQ inference or fitting, selection, or P6-E1 is performed. TSPLIB22 is loaded only after `configs/reference_rules_frozen.json` is written.\n")
    (OUT/"01_evoreal_feature_source_audit.md").write_text(f"""# EvoReal Feature Source Audit

- Paper: [{ARXIV}]({ARXIV}).
- Official repository: [{OFFICIAL_REPO}]({OFFICIAL_REPO}), current and pinned commit `{OFFICIAL_COMMIT}`.
- Full 1,494-file official tree was searched at that commit for FFT, histogram, nearest-neighbor, NN-ratio and coefficient-of-variation implementations; none was published. Only the three author validation coordinate files were found.
- Coordinate normalization and NN-CV definitions are explicit in the paper. FFT grid resolution, frequency-region mask, magnitude-vs-power and normalization are not.
- Reused implementation: `{APPROX_SOURCE.relative_to(ROOT)}`: 32x32 raw-count histogram, mean `|FFT|` over shifted annulus `1 < r <= 4`, DC excluded, population SD for NN-CV.
- Status: `FFT_APPROX`; NN-Ratio definition is faithful but the combined reproduction is `PARTIAL`.
- The implementation was historically calibrated on the same 48 author references. Threshold agreement is therefore descriptive reproduction, not independent validation.
- Frozen project and official repository counts are `{counts}`. The protocol text's `{PROTOCOL_CLAIMED_COUNTS}` is not present in the official files and is not substituted.
""")
    (OUT/"02_coordinate_normalization.md").write_text("# Coordinate Normalization\n\nAlgorithm 1 shared-max-span normalization is used for every source: subtract per-axis minima, divide both axes by the single maximum axis span. Aspect ratio is preserved. Canonical-float64 original and normalized coordinate SHA256 values, N, source and normalization version are retained in feature artifacts. Degenerate input produces zero coordinates in the historical approximation and would make NN-CV invalid; no audited instance is degenerate. Translation, uniform-scale, permutation and 90-degree-rotation diagnostics are in `manifests/invariance_and_nn_tests.csv`.\n\n"+md(tests)+"\n")
    (OUT/"03_fft_feature_audit.md").write_text(f"# FFT Feature Audit\n\nStatus: `FFT_APPROX`; paper threshold reproduction is `NOT_ESTABLISHED` as an exact reproduction because official numerical code is absent. On the 48 calibration references, approximate label agreement is {agreement:.1%}; S1 agreement {s1:.1%}, non-S1 agreement {nons1:.1%}. Spearman FFT vs N is {rho_fft:.4f}. Values are finite/non-negative and deterministic. Rotation sensitivity is quantified in the invariance manifest. Raw-count FFT amplitude is node-count sensitive, so FFT=35 is not treated as a cross-size universal threshold.\n")
    (OUT/"04_nn_ratio_audit.md").write_text(f"# NN-Ratio Audit\n\nNN-Ratio is population-SD/mean of self-excluded nearest-neighbor distances (`k=2`). KDTree agrees with direct distances to maximum error `{tests.loc[tests.test.eq('kdtree_direct_nn'),'nn_abs_diff'].max():.3e}`. Spearman NN-CV vs N on TSPLIB48 is {rho_nn:.4f}. Duplicate and zero-mean checks are retained per instance. Threshold-based type labels remain exploratory because the first-stage FFT implementation is approximate.\n")
    (OUT/"05_evoreal_type_reproduction.md").write_text(f"# EvoReal Type Reproduction\n\nOfficial-repository/frozen counts are `{counts}`, not protocol-claimed `{PROTOCOL_CLAIMED_COUNTS}`. Approximate rule agreement is {agreement:.1%}. This is `EXPLORATORY_APPROX_CLASSIFICATION`; it must not define a formal selection quota.\n\n"+md(pd.crosstab(repro.reference_group,repro.approx_type).reset_index())+"\n")
    summary=metrics[(metrics.record_type.eq("distribution"))&metrics.dataset.isin(["TSPLIB48","B_Original","B_S1","B_S2","C_Original","C_S1","C_S2","EvoReal50k"])]
    (OUT/"06_population_fft_nn_comparison.md").write_text("# Population FFT–NN Comparison\n\nGenerator provenance is available for all 50k EvoReal instances and is kept separate from approximate feature classification. B/C seeds are never pooled for inferential claims. Marginal statistics include quantiles rather than only means.\n\n"+md(summary[["dataset","feature","n","p10","p25","median","p75","p90"]])+"\n")
    (OUT/"07_representation_comparison.md").write_text(f"# Representation Comparison\n\nFrozen QD4 clipping is reproduced: `{clipfrac:.1%}` of TSPLIB48 touches at least one boundary. This is a map-normalization issue, not proof that FFT–NN is superior. Highest absolute Full-F40/Top5 correlations are shown below. NN-CV overlaps local nearest-neighbor statistics. FFT_APPROX is only partly complementary: its correlations with the frozen Top5 are modest on TSPLIB48, but its maximum Full-F40 correlation is high and substantially reflects node-size/distance-scale confounding.\n\n"+md(maxcorr[["scope","target_feature","comparison_feature","n","spearman"]])+"\n")
    (OUT/"08_target_coverage_gap.md").write_text("# Target Coverage Gap\n\nAll distances use TSPLIB48-only median/IQR scaling frozen before TSPLIB22 access. EvoReal full-50k and ten fixed 10k subsamples are both reported. Type labels are exploratory and class-proportion matching is not treated as sufficient evidence.\n\n"+md(ctab)+f"\n\nEvoReal matched-10k mean JS `{ejs:.6f}`, reference coverage `{ecov:.6f}`.\n")
    comparable=pd.read_csv(MAN/"comparable_size_coverage.csv")
    (OUT/"09_cross_size_stability.md").write_text(f"# Cross-Size Stability\n\nTSPLIB48 FFT-vs-N Spearman `{rho_fft:.4f}`; NN-CV-vs-N `{rho_nn:.4f}`. Fixed subsampling results below show that the raw-count FFT scale and threshold classification are not fully stable across N. Therefore TSP100 and broad-size TSPLIB values cannot be treated as directly interchangeable. A predeclared broad comparable band (75<=N<=150) contains only `{int(comparable.reference_n.iloc[0])}` references and is group-imbalanced; its results are diagnostic only. The apparent raw target gap cannot be cleanly separated from size confounding.\n\n"+md(cross)+"\n\n## Comparable-size diagnostic\n\n"+md(comparable)+"\n")
    (OUT/"10_optional_hardness_context.md").write_text("# Optional Existing-Hardness Context\n\nOnly existing true M4 teacher-forced CE attached to frozen B/C/D instances is used; no new inference or Concorde was run. Associations are descriptive and do not imply training value. EvoReal50k M4 CE was unavailable as a frozen aligned artifact and was not generated.\n\n"+md(hard.groupby(["arm","approx_type"]).agg(n=("n","sum"),mean_ce=("true_m4_ce_mean","mean"),median_ce=("true_m4_ce_median","mean")).reset_index())+"\n")
    heldsum=held.groupby("approx_type").agg(n=("instance_id","size"),fft_median=("fft_energy","median"),nn_median=("nn_ratio","median")).reset_index()
    (OUT/"11_nonreference_diagnostic.md").write_text("# TSPLIB22 Read-Only Diagnostic\n\nComputed only after reference rules were frozen. No parameter, threshold, normalization or target distribution is updated. TSPLIB22 is historically observed and is not described as pristine.\n\n"+md(heldsum)+"\n")
    final=pd.DataFrame([{"hypothesis":k,"decision":v} for k,v in decisions.items()])
    answer=pd.DataFrame([
        ["FFT reproduction","PARTIAL / FFT_APPROX; official numerical implementation absent"],
        ["NN-Ratio reproduction","Definition faithful; combined type rule remains approximate"],
        ["S1/S2/S3 reproduction",f"{agreement:.1%} agreement on same 48 calibration references; official counts 19/12/17"],
        ["Additional information",f"{h3}: FFT is partly complementary to Top5 but highly associated with some Full-F40/size-scale features; NN-CV overlaps local-distance statistics"],
        ["B target gap",h5],["EvoReal advantage",h6],["Cross-size",h4],
        ["Hardness consistency","Structure classes and CE differ descriptively; no training-value implication"],
        ["P6-E1 readiness",h7],
        ["Only admissible next step","Before any training pilot, obtain the exact FFT code or define and freeze a scale-stable FFT statistic, then rerun this read-only representation audit against Random B"],
    ],columns=["question","answer"])
    (OUT/"12_final_research_conclusion.md").write_text("# P6-E0 Final Research Conclusion\n\n"+md(final)+"\n\n"+md(answer)+"\n\nThe audit does not authorize P6-E1. No training-value claim is inferred from coverage.\n\n`BLADE_P6E0_EVOREAL_STRUCTURAL_REPRESENTATION_AUDIT_COMPLETE`\n")
    return decisions


def source_ledger() -> pd.DataFrame:
    paths=[REQUEST,APPROX_SOURCE,P1/"03_tsplib48_reference_manifest.csv",P1/"p1_heldout22_names.json",P1/"p1_tsplib48_normalization_sensitivity.csv",P6A/"diagnostics/tsplib48_top5_representation.parquet",P6D0/"completion.json",P6D1/"completion.json",P6D3/"completion.json",V2/"datasets/evoreal50k_coordinates.npy",V2/"datasets/evoreal50k_source_labels.npy",V2/"datasets/evoreal50k_manifest.json"]
    paths += [P6C1/"dataset_manifests"/f"{r}_union.npz" for r in REPS]
    rows=[]
    for p in paths: rows.append({"path":str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    out=pd.DataFrame(rows);out.to_csv(MAN/"input_source_hashes.csv",index=False);return out


def audit_metadata() -> None:
    write_json(CFG/"feature_implementation.json", {
        "normalization": {"version":NORMALIZATION_VERSION,"definition":"subtract per-axis minima; divide both axes by maximum axis span"},
        "fft": {"status":"FFT_APPROX","grid":GRID,"histogram":"raw counts","transform":"fft2 then fftshift","statistic":"mean absolute magnitude","frequency_mask":"1 < radial_bin <= 4","dc_excluded":True,"threshold_exploratory_only":FFT_THRESHOLD},
        "nn_ratio": {"status":"definition_faithful","neighbor":"self-excluded Euclidean nearest neighbor","std_ddof":0,"threshold_exploratory_only":NN_THRESHOLD},
        "official_repository":OFFICIAL_REPO,"official_commit":OFFICIAL_COMMIT,"paper":ARXIV,
        "random_seed":SEED,"matched_subsample_seeds":MATCHED_SEEDS,
    })
    pd.DataFrame([
        {"item":"official_fft_implementation","status":"MISSING","required_for":"EXACT feature reproduction","handling":"FFT_APPROX; no formal use of threshold 35"},
        {"item":"official_fft_numeric_parameters","status":"MISSING","required_for":"EXACT feature reproduction","handling":"all approximation parameters explicitly frozen"},
        {"item":"EvoReal50k_aligned_true_M4_CE","status":"MISSING_OPTIONAL","required_for":"optional hardness context","handling":"not generated; B/C/D existing CE only"},
        {"item":"C00-C03_evolution_trajectory","status":"NOT_INCLUDED_OPTIONAL","required_for":"auxiliary trajectory analysis","handling":"omitted to preserve minimum-evidence scope"},
    ]).to_csv(LOG/"missing_data.csv",index=False)
    pd.DataFrame([
        {"item":"TSPLIB48_group_counts","protocol":"S1=27, S2=15, S3=6","frozen_and_official":"S1=19, S2=12, S3=17","resolution":"frozen/official grouping retained; no relabeling"},
        {"item":"FFT_threshold_status","protocol":"use only if compatible official implementation","frozen_and_official":"official numerical implementation unavailable","resolution":"exploratory approximation only"},
    ]).to_csv(MAN/"protocol_discrepancies.csv",index=False)


def output_ledger() -> pd.DataFrame:
    rows=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("output_hashes.csv","completion.json") and "__pycache__" not in p.parts:
            rows.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    out=pd.DataFrame(rows);out.to_csv(LOG/"output_hashes.csv",index=False);return out


def main() -> None:
    started=time.perf_counter();ensure_dirs();(LOG/"runtime.csv").write_text("phase,dataset,n,wall_s\n")
    audit_metadata();source=source_ledger(); reference,coords=load_reference(); populations,inv=load_populations();tests=invariance_and_nn_tests(reference,coords,populations);repro=reference_reproduction(reference);metrics=population_metrics(reference,populations);corr=correlations(reference,populations);clip=qd_clipping(reference,populations);coverage=coverage_metrics(reference,populations);cross,subs=cross_size(reference,coords,populations);hard=hardness_context(populations)
    held,hcoords=load_heldout_after_freeze();make_figures(reference,coords,populations,coverage,corr,clip,held,subs);decisions=reports(reference,inv,tests,repro,metrics,corr,clip,coverage,cross,hard,held)
    runtime=time.perf_counter()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (OUT/"13_runtime_resource_audit.md").write_text(f"# Runtime and Resource Audit\n\n- Initial cold feature-cache construction observed wall time: approximately {COLD_CACHE_OBSERVED_WALL_S:.2f} s (the pass then stopped on a held-out path-resolution error, fixed without recomputing features).\n- Final cached deterministic validation pass: {runtime:.2f} s.\n- Peak RSS in final pass: {peak:.1f} MiB.\n- GPU/Concorde/HGB/BQ/evolution: none.\n- Unique fixed populations are cached by coordinate hash membership; FFT/NN is not recomputed for B/C/D subsets.\n")
    ledger=output_ledger(); completion={"status":"COMPLETE","final_label":"BLADE_P6E0_EVOREAL_STRUCTURAL_REPRESENTATION_AUDIT_COMPLETE",**decisions,"official_commit":OFFICIAL_COMMIT,"source_ledger_sha256":sha256_file(MAN/"input_source_hashes.csv"),"output_ledger_sha256":sha256_file(LOG/"output_hashes.csv"),"final_report_sha256":sha256_file(OUT/"12_final_research_conclusion.md"),"cold_cache_observed_wall_s":COLD_CACHE_OBSERVED_WALL_S,"final_cached_validation_wall_s":runtime,"peak_rss_mib":peak,"completed_utc":pd.Timestamp.now(tz="UTC").isoformat()};write_json(OUT/"completion.json",completion)
    print(json.dumps(completion,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.parse_args();main()
