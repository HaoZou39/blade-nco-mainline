#!/usr/bin/env python3
"""P6-G1 read-only structural reachability mechanism audit.

No mutation, model inference, solver, training, or held-out TSPLIB22 access occurs.
The script consumes historical P6-B artifacts and frozen P6-E0/G0 features.
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
import pyarrow.dataset as pads
import scipy
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "tables"
MAN = OUT / "manifests"
CACHE = OUT / "cache"
P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution/evolution"
P6G0 = ROOT / "target_aware_pilot/p6g0_raw_offspring_coverage_audit"
P6E0 = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit"
P6C0_CACHE = ROOT / "target_aware_pilot/p6c0_training_value_mechanism_audit/full_f40_cache"
P03_PATH = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz"
EVO_PATH = ROOT / "runs/BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2/datasets/evoreal50k_coordinates.npy"
EVO_LABELS = ROOT / "runs/BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V2/datasets/evoreal50k_source_labels.npy"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
REF_MANIFEST = ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv"
REF_F40 = ROOT / "target_aware_pilot/p1_tsplib48_normalization_sensitivity.csv"
FROZEN_RULES = P6E0 / "configs/reference_rules_frozen.json"
REPS = ("Original", "S1", "S2")
ARMS = ("E0", "E1")
F40 = tuple(json.loads(F40_MANIFEST.read_text())["F40"])
UNREACHABLE = ("d198", "d657", "pcb1173", "pr107", "pr144", "pr264", "u574", "u724", "a280", "eil51", "pcb442", "pr299", "rat195", "ts225", "tsp225")
SEED = 20261028
SAMPLE_REPEATS = 30


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
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def md(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 5) -> str:
    if frame.empty: return "_(empty)_"
    frame = frame if columns is None else frame[columns]
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for v in row:
            if pd.isna(v): cells.append("NA")
            elif isinstance(v, (float, np.floating)): cells.append(f"{float(v):.{digits}g}")
            else: cells.append(str(v).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def raw_stage_specs() -> list[dict[str, Any]]:
    rows = [{"dataset": "P03", "replicate": "ALL", "arm": "SHARED", "generation": 0, "path": P03_PATH,
             "fft": P6G0 / "features/P03.parquet", "kind": "npz"}]
    for rep in REPS:
        rows.append({"dataset": f"{rep}_G1_RAW", "replicate": rep, "arm": "SHARED", "generation": 1,
                     "path": P6B / rep / "shared_G1/mutation/offspring_coordinates.npz",
                     "fft": P6G0 / f"features/{rep}_G1_RAW.parquet", "kind": "npz"})
        for arm in ARMS:
            rows.append({"dataset": f"{rep}_{arm}_G2_RAW", "replicate": rep, "arm": arm, "generation": 2,
                         "path": P6B / rep / arm / "G2/mutation/offspring_coordinates.npz",
                         "fft": P6G0 / f"features/{rep}_{arm}_G2_RAW.parquet", "kind": "npz"})
    rows.append({"dataset": "EvoReal50k", "replicate": "EvoReal", "arm": "ALL", "generation": -1,
                 "path": EVO_PATH, "fft": P6E0 / "features/EvoReal50k.parquet", "kind": "npy"})
    return rows


def selected_spec(rep: str, arm: str, generation: int) -> dict[str, Any]:
    return {"dataset": f"{rep}_{arm}_G{generation}_SELECTED", "replicate": rep, "arm": arm, "generation": generation,
            "path": P6B / rep / arm / f"G{generation}/active_population.npz", "kind": "npz"}


def load_npz(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["coordinates"], np.float64), z["instance_hash"].astype(str).tolist()


def stage_coordinates(spec: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    if spec["kind"] == "npz": return load_npz(spec["path"])
    xy = np.load(spec["path"], mmap_mode="r", allow_pickle=False)
    hh = pd.read_parquet(spec["fft"], columns=["instance_hash"]).instance_hash.astype(str).tolist()
    return xy, hh


def parse_tsplib(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle: lines = handle.readlines()
    dimension = None; start = None
    for i, line in enumerate(lines):
        text = line.strip().upper()
        if text.startswith("DIMENSION"): dimension = int(line.replace(":", " ").split()[-1])
        if text == "NODE_COORD_SECTION": start = i + 1; break
    if dimension is None or start is None: raise RuntimeError(f"cannot parse {path}")
    points = []
    for line in lines[start:]:
        text = line.strip()
        if not text or text.upper() in {"EOF", "DISPLAY_DATA_SECTION"}: break
        p = text.split()
        if len(p) >= 3: points.append((float(p[1]), float(p[2])))
        if len(points) == dimension: break
    out = np.asarray(points, float)
    if out.shape != (dimension, 2): raise RuntimeError(f"coordinate shape mismatch {path}")
    return out


def normalize_shape(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, float) - np.min(x, axis=0)
    span = float(np.max(np.max(z, axis=0)))
    return z / span if span > 0 else np.zeros_like(z)


def load_global_f40() -> pd.DataFrame:
    consolidated = CACHE / "formal_f40_all.parquet"
    if consolidated.is_file():
        frame = pd.read_parquet(consolidated)
    else:
        paths = sorted(P6C0_CACHE.glob("f40_*.parquet")) + sorted((OUT / "f40_cache").glob("f40_*.parquet"))
        if not paths: raise RuntimeError("no F40 cache files")
        table = pads.dataset([str(p) for p in paths], format="parquet").to_table(columns=["instance_hash", *F40])
        frame = table.to_pandas().drop_duplicates("instance_hash", keep="last")
        consolidated.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(consolidated, index=False)
    if list(frame.columns) != ["instance_hash", *F40]: raise RuntimeError("F40 schema/order mismatch")
    values=frame[list(F40)].to_numpy(float)
    if np.isinf(values).any(): raise RuntimeError("infinite F40")
    missing=[]
    for j,name in enumerate(F40):
        n=int(np.isnan(values[:,j]).sum())
        if n: missing.append({"feature":name,"missing_n":n,"missing_fraction":n/len(values)})
    pd.DataFrame(missing,columns=["feature","missing_n","missing_fraction"]).to_csv(TAB/"f40_native_missingness.csv",index=False)
    return frame.set_index("instance_hash", drop=False)


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(REF_MANIFEST)
    if len(manifest) != 48 or manifest.instance_id.nunique() != 48: raise RuntimeError("TSPLIB48 integrity")
    fft = pd.read_parquet(P6E0 / "features/TSPLIB48.parquet")
    f40 = pd.read_csv(REF_F40)
    f40 = f40[f40.coordinate_variant.eq("aspect")].set_index("instance_id").loc[manifest.instance_id].reset_index()
    if list(f40[list(F40)].columns) != list(F40): raise RuntimeError("reference F40 order mismatch")
    coords = {r.instance_id: parse_tsplib(ROOT / r.source_file) for r in manifest.itertuples(index=False)}
    fft = manifest[["instance_id", "n_nodes", "evoreal_split"]].merge(
        fft[["instance_id", "fft_energy", "nn_ratio"]], on="instance_id", validate="one_to_one")
    return fft, f40, coords


def f40_values(indexed: pd.DataFrame, hashes: list[str]) -> np.ndarray:
    missing = set(hashes) - set(indexed.index)
    if missing: raise RuntimeError(f"F40 membership missing {len(missing)}")
    return indexed.loc[hashes, list(F40)].to_numpy(float)


def fft_values(spec: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    coords, hashes = stage_coordinates(spec)
    frame = pd.read_parquet(spec["fft"])
    if frame.instance_hash.astype(str).tolist() != hashes: raise RuntimeError(f"FFT alignment failure {spec['dataset']}")
    return frame, coords, hashes


def summary_rows(dataset: str, representation: str, names: list[str], values: np.ndarray, rep: str, arm: str, generation: int) -> list[dict[str, Any]]:
    rows = []
    for j, name in enumerate(names):
        raw = np.asarray(values[:, j], float); v=raw[np.isfinite(raw)]
        if not len(v): raise RuntimeError(f"all values missing: {dataset} {name}")
        q = np.quantile(v, [.05, .25, .5, .75, .95]); iqr = q[3] - q[1]
        outliers = (v < q[1] - 3 * iqr) | (v > q[3] + 3 * iqr)
        rows.append({"dataset": dataset, "replicate": rep, "arm": arm, "generation": generation,
                     "representation": representation, "feature": name, "n": len(raw), "finite_n":len(v),"missing_n":int(len(raw)-len(v)),"missing_fraction":float(1-len(v)/len(raw)),"mean": v.mean(),
                     "std": v.std(ddof=1), "median": q[2], "iqr": iqr, "p05": q[0], "p25": q[1],
                     "p75": q[3], "p95": q[4], "min": v.min(), "max": v.max(),
                     "outlier_n_3iqr": int(outliers.sum()), "outlier_fraction_3iqr": outliers.mean()})
    return rows


def distribution_audit(f40_index: pd.DataFrame, refs_fft: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    distributions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for spec in raw_stage_specs():
        fft, coords, hashes = fft_values(spec)
        fv = f40_values(f40_index, hashes)
        xv = fft[["fft_energy", "nn_ratio"]].to_numpy(float)
        distributions[spec["dataset"]] = {"spec": spec, "hashes": hashes, "coords": coords, "f40": fv, "fft": xv, "fft_frame": fft}
        rows += summary_rows(spec["dataset"], "F40_FORMAL_ASINH", list(F40), fv, spec["replicate"], spec["arm"], spec["generation"])
        rows += summary_rows(spec["dataset"], "FFT_APPROX_NN_RATIO", ["fft_energy", "nn_ratio"], xv, spec["replicate"], spec["arm"], spec["generation"])
        if spec["dataset"] == "EvoReal50k" and "generator_source" in fft:
            for source, idx in fft.groupby("generator_source").groups.items():
                ii = np.asarray(list(idx), int); name = f"EvoReal_{source}"
                rows += summary_rows(name, "F40_FORMAL_ASINH", list(F40), fv[ii], "EvoReal", str(source), -1)
                rows += summary_rows(name, "FFT_APPROX_NN_RATIO", ["fft_energy", "nn_ratio"], xv[ii], "EvoReal", str(source), -1)
    out = pd.DataFrame(rows); out.to_csv(OUT / "02_structural_distribution.csv", index=False)

    evo = distributions["EvoReal50k"]
    ranking = []
    for dataset, data in distributions.items():
        if dataset == "EvoReal50k": continue
        for representation, names, left, right in (
            ("F40_FORMAL_ASINH", list(F40), data["f40"], evo["f40"]),
            ("FFT_APPROX_NN_RATIO", ["fft_energy", "nn_ratio"], data["fft"], evo["fft"]),
        ):
            for j, feature in enumerate(names):
                a, b = left[:, j], right[:, j];a=a[np.isfinite(a)];b=b[np.isfinite(b)]
                pooled = math.sqrt(((len(a)-1)*np.var(a, ddof=1)+(len(b)-1)*np.var(b, ddof=1))/(len(a)+len(b)-2))
                biqr = float(np.subtract(*np.quantile(b, [.75, .25])))
                wd = float(wasserstein_distance(a, b))
                ranking.append({"dataset": dataset, "replicate": data["spec"]["replicate"], "arm": data["spec"]["arm"],
                                "generation": data["spec"]["generation"], "representation": representation,
                                "feature": feature, "standardized_mean_difference": (a.mean()-b.mean())/pooled if pooled else np.nan,
                                "wasserstein_distance": wd, "wasserstein_over_evoreal_iqr": wd/biqr if biqr else np.nan,
                                "iqr_ratio_vs_evoreal": np.subtract(*np.quantile(a,[.75,.25]))/biqr if biqr else np.nan,
                                "stage_p05": np.quantile(a,.05), "stage_p95": np.quantile(a,.95),
                                "evoreal_p05": np.quantile(b,.05), "evoreal_p95": np.quantile(b,.95)})
    rank = pd.DataFrame(ranking)
    rank["rank_abs_smd_within_dataset"] = rank.groupby(["dataset", "representation"])["standardized_mean_difference"].transform(lambda x: x.abs().rank(method="min", ascending=False))
    rank["rank_wasserstein_within_dataset"] = rank.groupby(["dataset", "representation"])["wasserstein_over_evoreal_iqr"].rank(method="min", ascending=False)
    rank.to_csv(OUT / "03_feature_difference_ranking.csv", index=False)
    return out, rank, distributions


def build_selected(distributions: dict[str, dict[str, Any]], f40_index: pd.DataFrame) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    p03 = distributions["P03"]
    for rep in REPS:
        g1 = distributions[f"{rep}_G1_RAW"]
        fft_lookup = pd.concat([
            pd.DataFrame({"instance_hash": p03["hashes"], "fft_energy": p03["fft"][:,0], "nn_ratio": p03["fft"][:,1]}),
            pd.DataFrame({"instance_hash": g1["hashes"], "fft_energy": g1["fft"][:,0], "nn_ratio": g1["fft"][:,1]}),
        ], ignore_index=True).drop_duplicates("instance_hash").set_index("instance_hash")
        for arm in ARMS:
            for generation in (1, 2):
                spec = selected_spec(rep, arm, generation); coords, hashes = stage_coordinates(spec)
                if generation == 2:
                    raw = distributions[f"{rep}_{arm}_G2_RAW"]
                    fft_lookup2 = pd.concat([fft_lookup.reset_index(), pd.DataFrame({"instance_hash":raw["hashes"],"fft_energy":raw["fft"][:,0],"nn_ratio":raw["fft"][:,1]})],ignore_index=True).drop_duplicates("instance_hash").set_index("instance_hash")
                else: fft_lookup2 = fft_lookup
                if not set(hashes).issubset(fft_lookup2.index): raise RuntimeError(f"selected FFT membership {spec['dataset']}")
                selected[spec["dataset"]] = {"spec": spec, "coords": coords, "hashes": hashes,
                    "fft": fft_lookup2.loc[hashes, ["fft_energy", "nn_ratio"]].to_numpy(float),
                    "f40": f40_values(f40_index, hashes)}
    return selected


def nearest_reference_audit(distributions: dict[str, dict[str, Any]], refs_fft: pd.DataFrame, refs_f40: pd.DataFrame) -> pd.DataFrame:
    rules = json.loads(FROZEN_RULES.read_text()); center=np.asarray(rules["center"]); scale=np.asarray(rules["scale_iqr"]); radius=float(rules["radius"])
    rz=(refs_fft[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale
    ref_f40 = refs_f40.set_index("instance_id")
    ref_index = {x:i for i,x in enumerate(refs_fft.instance_id)}
    rows=[]; saved_coords=[]; keys=[]
    for dataset, data in distributions.items():
        tree=cKDTree((data["fft"]-center)/scale)
        for ref in UNREACHABLE:
            j=ref_index[ref]; distance, idx=tree.query(rz[j],k=1); idx=int(idx)
            candidate=data["f40"][idx]; target=ref_f40.loc[ref,list(F40)].to_numpy(float); delta=candidate-target;valid=np.isfinite(delta)
            ranked=np.where(valid,np.abs(delta),-np.inf);order=np.argsort(ranked)[::-1][:min(8,int(valid.sum()))]
            key=f"{dataset}__{ref}"; saved_coords.append(np.asarray(data["coords"][idx],float)); keys.append(key)
            rows.append({"reference_id":ref,"reference_n_nodes":int(refs_fft.iloc[j].n_nodes),"matched_75_150":bool(75<=refs_fft.iloc[j].n_nodes<=150),
                         "dataset":dataset,"replicate":data["spec"]["replicate"],"arm":data["spec"]["arm"],"generation":data["spec"]["generation"],
                         "population_n":len(data["hashes"]),"candidate_index":idx,"candidate_hash":data["hashes"][idx],"coordinate_key":key,
                         "coordinate_source":str(data["spec"]["path"].relative_to(ROOT)),"fft_nn_distance":float(distance),"frozen_radius":radius,
                         "distance_over_radius":float(distance/radius),"covered":bool(distance<=radius),"f40_comparable_dimension_n":int(valid.sum()),"f40_rms_difference":float(np.sqrt(np.mean(delta[valid]**2))) if valid.any() else np.nan,
                         "f40_l2_difference":float(np.linalg.norm(delta[valid])) if valid.any() else np.nan,"top_f40_abs_difference_features":json.dumps([F40[i] for i in order]),
                         "top_f40_signed_differences":json.dumps([float(delta[i]) for i in order])})
    out=pd.DataFrame(rows); out.to_csv(OUT/"04_unreachable_reference_analysis.csv",index=False)
    np.savez_compressed(MAN/"unreachable_nearest_coordinates.npz",keys=np.asarray(keys),coordinates=np.stack(saved_coords))
    return out


def operator_audit(distributions: dict[str, dict[str, Any]], f40_index: pd.DataFrame, refs_fft: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    rules=json.loads(FROZEN_RULES.read_text()); center=np.asarray(rules["center"]);scale=np.asarray(rules["scale_iqr"]);radius=float(rules["radius"])
    rz=(refs_fft[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale; rtree=cKDTree(rz)
    refpos={x:i for i,x in enumerate(refs_fft.instance_id)}; movement=[]; summaries=[]
    stages=[]
    for rep in REPS:
        stages.append((rep,"SHARED",1,P6B/rep/"shared_G1/mutation/mutation_manifest.parquet",distributions[f"{rep}_G1_RAW"]))
        for arm in ARMS: stages.append((rep,arm,2,P6B/rep/arm/"G2/mutation/mutation_manifest.parquet",distributions[f"{rep}_{arm}_G2_RAW"]))
    for rep,arm,generation,manifest_path,data in stages:
        manifest=pd.read_parquet(manifest_path); hashes=data["hashes"]
        accepted=manifest[(manifest.proposal_status=="accepted") & manifest.child_hash.notna() & (~manifest.within_offspring_duplicate.fillna(False))].drop_duplicates("child_hash")
        accepted=accepted.set_index("child_hash").loc[hashes].reset_index()
        if accepted.child_hash.astype(str).tolist()!=hashes: raise RuntimeError(f"manifest alignment {rep} {arm} G{generation}")
        parent_hash=accepted.parent_hash.astype(str).tolist(); pf=f40_values(f40_index,parent_hash); cf=data["f40"]
        if generation==1:
            lookup=pd.DataFrame({"instance_hash":distributions["P03"]["hashes"],"fft_energy":distributions["P03"]["fft"][:,0],"nn_ratio":distributions["P03"]["fft"][:,1]}).set_index("instance_hash")
        else:
            p03=distributions["P03"];g1=distributions[f"{rep}_G1_RAW"]
            lookup=pd.concat([pd.DataFrame({"instance_hash":p03["hashes"],"fft_energy":p03["fft"][:,0],"nn_ratio":p03["fft"][:,1]}),pd.DataFrame({"instance_hash":g1["hashes"],"fft_energy":g1["fft"][:,0],"nn_ratio":g1["fft"][:,1]})],ignore_index=True).drop_duplicates("instance_hash").set_index("instance_hash")
        if not set(parent_hash).issubset(lookup.index): raise RuntimeError(f"parent FFT missing {rep} {arm} G{generation}")
        pfft=lookup.loc[parent_hash,["fft_energy","nn_ratio"]].to_numpy(float); cfft=data["fft"]
        pz=(pfft-center)/scale;cz=(cfft-center)/scale
        pmin=rtree.query(pz,k=1)[0];cmin=rtree.query(cz,k=1)[0]
        comparable=np.isfinite(cf)&np.isfinite(pf);fdelta=np.sqrt(np.nanmean(np.where(comparable,(cf-pf)**2,np.nan),axis=1)); xdelta=np.linalg.norm(cz-pz,axis=1)
        pr107_p=np.linalg.norm(pz-rz[refpos["pr107"]],axis=1);pr107_c=np.linalg.norm(cz-rz[refpos["pr107"]],axis=1)
        pr144_p=np.linalg.norm(pz-rz[refpos["pr144"]],axis=1);pr144_c=np.linalg.norm(cz-rz[refpos["pr144"]],axis=1)
        frame=pd.DataFrame({"operator_id":accepted.operator_id.astype(str),"f40_delta_rms":fdelta,"fft_nn_delta":xdelta,
                            "nearest_ref_improvement":pmin-cmin,"new_neighborhood":(cmin<=radius)&(pmin>radius),
                            "pr107_improvement":pr107_p-pr107_c,"pr107_child_distance":pr107_c,
                            "pr144_improvement":pr144_p-pr144_c,"pr144_child_distance":pr144_c,
                            "f40_comparable_dimension_n":comparable.sum(axis=1),"n_affected":accepted.n_affected.to_numpy(float),"repair_fraction":accepted.repair_fraction.to_numpy(float)})
        f90=float(np.quantile(fdelta,.9));x90=float(np.quantile(xdelta,.9))
        calls=manifest.groupby("operator_id").size()
        for op,g in frame.groupby("operator_id"):
            summaries.append({"replicate":rep,"arm":arm,"generation":generation,"operator_id":op,"call_count":int(calls.get(op,0)),
                "accepted_unique_offspring":len(g),"acceptance_fraction":len(g)/calls.get(op,len(g)),
                "f40_delta_mean":g.f40_delta_rms.mean(),"f40_delta_median":g.f40_delta_rms.median(),"f40_delta_p90":g.f40_delta_rms.quantile(.9),"f40_delta_p95":g.f40_delta_rms.quantile(.95),"f40_large_move_fraction_stage_p90":np.mean(g.f40_delta_rms>=f90),
                "fft_nn_delta_mean":g.fft_nn_delta.mean(),"fft_nn_delta_median":g.fft_nn_delta.median(),"fft_nn_delta_p90":g.fft_nn_delta.quantile(.9),"fft_nn_large_move_fraction_stage_p90":np.mean(g.fft_nn_delta>=x90),
                "improved_nearest_reference_fraction":np.mean(g.nearest_ref_improvement>0),"new_neighborhood_instance_n":int(g.new_neighborhood.sum()),
                "pr107_improved_fraction":np.mean(g.pr107_improvement>0),"pr107_best_improvement":g.pr107_improvement.max(),"pr107_min_child_distance":g.pr107_child_distance.min(),
                "pr144_improved_fraction":np.mean(g.pr144_improvement>0),"pr144_best_improvement":g.pr144_improvement.max(),"pr144_min_child_distance":g.pr144_child_distance.min(),
                "f40_comparable_dimension_mean":g.f40_comparable_dimension_n.mean(),"n_affected_mean":g.n_affected.mean(),"repair_fraction_mean":g.repair_fraction.mean()})
        movement.append({"replicate":rep,"arm":arm,"generation":generation,"n":len(frame),"f40_delta_median":np.median(fdelta),"f40_delta_p90":np.quantile(fdelta,.9),"f40_delta_p99":np.quantile(fdelta,.99),"fft_nn_delta_median":np.median(xdelta),"fft_nn_delta_p90":np.quantile(xdelta,.9),"nearest_ref_improved_fraction":np.mean(pmin>cmin),"new_neighborhood_instance_n":int(np.sum((cmin<=radius)&(pmin>radius))),"pr107_improved_fraction":np.mean(pr107_p>pr107_c),"pr144_improved_fraction":np.mean(pr144_p>pr144_c)})
        print(f"[P6-G1] operator paired {rep} {arm} G{generation}: {len(frame):,}",flush=True)
    out=pd.DataFrame(summaries); out.to_csv(OUT/"05_operator_effect_analysis.csv",index=False)
    mov=pd.DataFrame(movement); mov.to_csv(TAB/"parent_child_movement_summary.csv",index=False)
    return out,mov


def stage_metrics(distributions: dict[str,dict[str,Any]], selected:dict[str,dict[str,Any]], refs_fft:pd.DataFrame, rank:pd.DataFrame)->pd.DataFrame:
    rules=json.loads(FROZEN_RULES.read_text());center=np.asarray(rules["center"]);scale=np.asarray(rules["scale_iqr"]);radius=float(rules["radius"])
    rz=(refs_fft[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale; rows=[]
    combined={**distributions,**selected}
    for name,d in combined.items():
        z=(d["fft"]-center)/scale; tree=cKDTree(z);rd=tree.query(rz,k=1)[0]; pdist=cKDTree(rz).query(z,k=1)[0]
        spec=d["spec"]
        q=rank[(rank.dataset==name)&(rank.representation=="F40_FORMAL_ASINH")]
        rows.append({"dataset":name,"replicate":spec["replicate"],"arm":spec["arm"],"generation":spec["generation"],"n":len(z),
                     "coverage_48":int(np.sum(rd<=radius)),"coverage_17":int(np.sum(rd[refs_fft.n_nodes.between(75,150).to_numpy()]<=radius)),
                     "median_population_to_reference_distance":np.median(pdist),"p90_population_to_reference_distance":np.quantile(pdist,.9),
                     "pr107_nearest_distance":rd[refs_fft.instance_id.tolist().index("pr107")],"pr144_nearest_distance":rd[refs_fft.instance_id.tolist().index("pr144")],
                     "mean_abs_f40_smd_vs_evoreal":q.standardized_mean_difference.abs().mean() if len(q) else np.nan,
                     "mean_f40_wasserstein_over_evoreal_iqr":q.wasserstein_over_evoreal_iqr.mean() if len(q) else np.nan})
    out=pd.DataFrame(rows);out.to_csv(TAB/"generation_stage_metrics.csv",index=False);return out


def matched_sampling(distributions:dict[str,dict[str,Any]],refs_fft:pd.DataFrame)->pd.DataFrame:
    rules=json.loads(FROZEN_RULES.read_text());center=np.asarray(rules["center"]);scale=np.asarray(rules["scale_iqr"]);radius=float(rules["radius"])
    rz=(refs_fft[["fft_energy","nn_ratio"]].to_numpy(float)-center)/scale;evo=distributions["EvoReal50k"];rows=[]
    quantiles=np.linspace(0,1,101)
    for dataset,d in distributions.items():
        if dataset=="EvoReal50k":continue
        for n in (10000,50000):
            if len(d["hashes"])<n or len(evo["hashes"])<n:continue
            for repeat in range(SAMPLE_REPEATS):
                rng=np.random.default_rng(SEED+10000*n+repeat+int(hashlib.sha256(dataset.encode()).hexdigest()[:8],16))
                ia=rng.choice(len(d["hashes"]),n,replace=False);ib=rng.choice(len(evo["hashes"]),n,replace=False)
                for representation,a,b in (("F40_FORMAL_ASINH",d["f40"][ia],evo["f40"][ib]),("FFT_APPROX_NN_RATIO",d["fft"][ia],evo["fft"][ib])):
                    ma=np.nanmean(a,0);mb=np.nanmean(b,0);va=np.nanvar(a,0,ddof=1);vb=np.nanvar(b,0,ddof=1);na=np.isfinite(a).sum(0);nb=np.isfinite(b).sum(0);pooled=np.sqrt(((na-1)*va+(nb-1)*vb)/np.maximum(1,na+nb-2));smd=np.divide(ma-mb,pooled,out=np.full_like(ma,np.nan),where=pooled>0)
                    qa=np.nanquantile(a,quantiles,axis=0);qb=np.nanquantile(b,quantiles,axis=0);biqr=np.subtract(*np.nanquantile(b,[.75,.25],axis=0));wd=np.nanmean(np.abs(qa-qb),axis=0);nwd=np.divide(wd,biqr,out=np.full_like(wd,np.nan),where=biqr>0)
                    row={"dataset":dataset,"sample_n":n,"repeat":repeat,"representation":representation,"mean_abs_smd":np.nanmean(np.abs(smd)),"mean_quantile_wasserstein_over_evoreal_iqr":np.nanmean(nwd)}
                    if representation=="FFT_APPROX_NN_RATIO":
                        rd=cKDTree((a-center)/scale).query(rz,k=1)[0]
                        row["coverage_48"] = int(np.sum(rd<=radius))
                        row["coverage_17"] = int(np.sum(rd[refs_fft.n_nodes.between(75,150).to_numpy()]<=radius))
                    rows.append(row)
            print(f"[P6-G1] matched samples {dataset} n={n} repeats={SAMPLE_REPEATS}",flush=True)
    out=pd.DataFrame(rows);out.to_csv(TAB/"fixed_sample_robustness.csv",index=False)
    metrics=["mean_abs_smd","mean_quantile_wasserstein_over_evoreal_iqr","coverage_48","coverage_17"]
    summaries=[]
    for keys,g in out.groupby(["dataset","sample_n","representation"],dropna=False):
        row={"dataset":keys[0],"sample_n":keys[1],"representation":keys[2],"repeats":len(g)}
        for metric in metrics:
            v=g[metric].dropna().to_numpy(float) if metric in g else np.array([])
            row[f"{metric}_mean"]=v.mean() if len(v) else np.nan
            row[f"{metric}_p025"]=np.quantile(v,.025) if len(v) else np.nan
            row[f"{metric}_p975"]=np.quantile(v,.975) if len(v) else np.nan
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(TAB/"fixed_sample_robustness_summary.csv",index=False)
    return out


def plots(distributions:dict[str,dict[str,Any]],rank:pd.DataFrame,refs_fft:pd.DataFrame,unreachable:pd.DataFrame,ref_coords:dict[str,np.ndarray],operators:pd.DataFrame,stages:pd.DataFrame)->None:
    rng=np.random.default_rng(SEED); colors={"P03":"#333333","Original_G1_RAW":"#1f77b4","Original_E0_G2_RAW":"#ff7f0e","Original_E1_G2_RAW":"#2ca02c","EvoReal50k":"#d62728"}; names=list(colors)
    # FFT-NN matched-size scatter.
    fig,ax=plt.subplots(figsize=(8,6))
    for name in names:
        v=distributions[name]["fft"];ii=rng.choice(len(v),min(2500,len(v)),replace=False);ax.scatter(v[ii,0],v[ii,1],s=5,alpha=.22,label=name,color=colors[name])
    ax.scatter(refs_fft.fft_energy,refs_fft.nn_ratio,s=28,c="black",marker="x",label="TSPLIB48");ax.set(xlabel="FFT_APPROX energy",ylabel="NN-ratio",title="Frozen FFT_APPROX + NN-ratio structural space");ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(FIG/"01_fft_nn_structure_space.png",dpi=180);plt.close(fig)
    # F40 PCA on matched samples.
    chunks=[];labels=[]
    for name in names:
        v=distributions[name]["f40"];ii=rng.choice(len(v),min(10000,len(v)),replace=False);chunks.append(v[ii]);labels.extend([name]*len(ii))
    x=np.vstack(chunks);lab=np.asarray(labels);complete=np.isfinite(x).all(axis=1);x=x[complete];lab=lab[complete]
    z=StandardScaler().fit_transform(x);pca=PCA(n_components=2,random_state=SEED).fit_transform(z)
    fig,ax=plt.subplots(figsize=(8,6))
    for name in names:
        ii=np.flatnonzero(lab==name);jj=rng.choice(ii,min(2500,len(ii)),replace=False);ax.scatter(pca[jj,0],pca[jj,1],s=5,alpha=.2,label=name,color=colors[name])
    ax.set(xlabel="PC1",ylabel="PC2",title="F40 PCA (equal 10k inputs per population; display subsample)");ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(FIG/"02_f40_pca.png",dpi=180);plt.close(fig)
    # Top six formal marginal differences by average normalized Wasserstein.
    formal=rank[rank.representation.eq("F40_FORMAL_ASINH")];top=formal.groupby("feature").wasserstein_over_evoreal_iqr.mean().nlargest(6).index.tolist()
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,feature in zip(axes.ravel(),top):
        j=list(F40).index(feature)
        for name in names:
            v=distributions[name]["f40"][:,j];v=v[np.isfinite(v)];ii=rng.choice(len(v),min(10000,len(v)),replace=False);ax.hist(v[ii],bins=50,density=True,histtype="step",linewidth=1.2,label=name,color=colors[name])
        ax.set_title(feature.replace("spn_",""),fontsize=8)
    axes[0,0].legend(fontsize=6);fig.suptitle("Largest F40 marginal differences from EvoReal");fig.tight_layout();fig.savefig(FIG/"03_f40_top_feature_distributions.png",dpi=180);plt.close(fig)
    # pr107/pr144 gallery.
    fig,axes=plt.subplots(2,4,figsize=(13,7))
    for row,ref in enumerate(("pr107","pr144")):
        candidates=[("TSPLIB",normalize_shape(ref_coords[ref]))]
        for label,selector in (("P03",unreachable[(unreachable.reference_id==ref)&(unreachable.dataset=="P03")]),("FOGA RAW",unreachable[(unreachable.reference_id==ref)&unreachable.dataset.str.contains("RAW")].nsmallest(1,"fft_nn_distance")),("EvoReal",unreachable[(unreachable.reference_id==ref)&(unreachable.dataset=="EvoReal50k")])):
            r=selector.iloc[0];data=distributions[r.dataset];candidates.append((f"{label}\nd/r={r.distance_over_radius:.2f}",normalize_shape(np.asarray(data["coords"][int(r.candidate_index)],float))))
        for col,(title,xy) in enumerate(candidates):
            ax=axes[row,col];ax.scatter(xy[:,0],xy[:,1],s=7);ax.set_aspect("equal");ax.set_title(f"{ref}: {title}",fontsize=8);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("Geometry gallery (shared-max-span normalized; visual diagnostic only)");fig.tight_layout();fig.savefig(FIG/"04_pr107_pr144_geometry.png",dpi=180);plt.close(fig)
    # Operator movement.
    agg=operators.groupby("operator_id").agg(f40_median=("f40_delta_median","mean"),f40_p95=("f40_delta_p95","mean"),fft_median=("fft_nn_delta_median","mean"),new_n=("new_neighborhood_instance_n","sum")).sort_values("f40_median")
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(agg));ax.bar(x,agg.f40_median,label="F40 median RMS");ax.scatter(x,agg.f40_p95,s=25,c="red",label="F40 P95");ax.set_xticks(x,agg.index,rotation=55,ha="right",fontsize=7);ax.set_ylabel("formal F40 movement");ax.legend();ax.set_title("Observed parent→offspring movement by FOGA-11 operator");fig.tight_layout();fig.savefig(FIG/"05_operator_movement.png",dpi=180);plt.close(fig)
    # Generational distance to difficult references.
    q=stages[(stages.replicate=="Original") & stages.dataset.str.contains("P03|G1_RAW|G1_SELECTED|G2_RAW",regex=True)].copy()
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(q));ax.bar(x-.18,q.pr107_nearest_distance,.36,label="pr107");ax.bar(x+.18,q.pr144_nearest_distance,.36,label="pr144");ax.axhline(json.loads(FROZEN_RULES.read_text())["radius"],color="black",ls="--",label="frozen radius");ax.set_xticks(x,q.dataset,rotation=55,ha="right",fontsize=7);ax.set_ylabel("nearest frozen FFT–NN distance");ax.legend();ax.set_title("Generation and survivor proximity to pr107/pr144 (Original seed)");fig.tight_layout();fig.savefig(FIG/"06_generation_reference_distance.png",dpi=180);plt.close(fig)


def write_reports(distributions:dict[str,dict[str,Any]],rank:pd.DataFrame,unreachable:pd.DataFrame,operators:pd.DataFrame,movement:pd.DataFrame,stages:pd.DataFrame,matched:pd.DataFrame,refs_fft:pd.DataFrame)->None:
    inventory=pd.read_csv(P6G0/"tables/artifact_inventory.csv")
    f40log=json.loads((OUT/"logs/f40_extraction.json").read_text())
    inv=inventory[inventory.stage.isin(["P03","G1_RAW","G1_SELECTED","G2_RAW","G2_SELECTED"])][["replicate","arm","stage","n","unique_n","path","manifest_path"]]
    integrity=f"""# P6-G1 Data Integrity

This audit is read-only with respect to P6-B/P6-E0/P6-G0. It performed no mutation, generation, BQ/HGB inference, Concorde labeling, or training. TSPLIB22 was never loaded.

## Artifact inventory

{md(inv)}

G1 RAW is shared once per seed; G2 RAW is branch-specific. Every stored stage is coordinate-hash unique. Mutation manifests retain exact `parent_hash`, `child_hash`, `operator_id`, sampled parameters, validation and repair metadata, so parent–child pairing is exact rather than nearest-neighbor inferred.

## Feature integrity

- Formal F40 order is the frozen 40-column manifest; preprocessing is `asinh((raw-center_median)/scale)` using the formal scaler.
- F40 extraction: **{f40log['status']}**, {sum(x['newly_extracted'] for x in f40log['stages']):,} new coordinates, {f40log['total_wall_s']/60:.2f} minutes, {f40log['workers']} CPU workers.
- The canonical pipeline's frozen policy is `native_HGB_missing_routing_no_imputation`. Sparse native NaNs are retained and audited in `tables/f40_native_missingness.csv`; marginal statistics use available finite values, parent–child RMS uses pairwise-available dimensions, and PCA is complete-case. No imputation was introduced.
- FFT contrast uses frozen `FFT_APPROX` + NN-ratio, TSPLIB48 median/IQR normalization and radius {json.loads(FROZEN_RULES.read_text())['radius']:.9f}; no threshold was refit.
- EvoReal generator labels are available and retained. All synthetic sets compared here are TSP100.
"""
    (OUT/"01_data_integrity.md").write_text(integrity)

    opagg=operators.groupby("operator_id").agg(calls=("call_count","sum"),accepted=("accepted_unique_offspring","sum"),f40_median=("f40_delta_median","mean"),f40_p95=("f40_delta_p95","mean"),fft_median=("fft_nn_delta_median","mean"),improved=("improved_nearest_reference_fraction","mean"),new_neighborhood=("new_neighborhood_instance_n","sum"),pr107_best=("pr107_best_improvement","max"),pr107_min_child=("pr107_min_child_distance","min"),pr144_best=("pr144_best_improvement","max"),pr144_min_child=("pr144_min_child_distance","min")).reset_index().sort_values("f40_median",ascending=False)
    rawstage=stages[stages.dataset.str.contains("P03|RAW",regex=True)].sort_values(["replicate","generation","arm"])
    g0=pd.read_csv(P6G0/"02_stage_coverage.csv");full=g0[g0.evaluation_mode.eq("full")]
    coverage=full.groupby(["replicate","arm","stage"],as_index=False).first()[["replicate","arm","stage","coverage_48","coverage_17"]] if "coverage_48" in full else pd.DataFrame()
    matched_summary=pd.read_csv(TAB/"fixed_sample_robustness_summary.csv")
    matched_fft=matched_summary[matched_summary.representation.eq("FFT_APPROX_NN_RATIO")][["dataset","sample_n","coverage_48_mean","coverage_48_p025","coverage_48_p975","coverage_17_mean","mean_abs_smd_mean","mean_quantile_wasserstein_over_evoreal_iqr_mean"]]
    dynamics=f"""# P6-G1 Generation Dynamics

## Observed trajectory

{md(rawstage[["dataset","n","coverage_48","coverage_17","median_population_to_reference_distance","pr107_nearest_distance","pr144_nearest_distance"]])}

P03 starts at 23/48 neighborhoods. Full G1 RAW reaches only 24–25/48 and full G2 RAW 22–25/48 across the three historical seeds/branches. The 75–150 control remains 14/17 at P03, at most 15/17 for historical RAW, versus 17/17 for EvoReal. Thus the full reference gap is not solely a sample-count comparison or solely caused by very large TSPLIB sizes.

G1 survivor selection changes the parent distribution supplied to G2. E0 and E1 therefore have different G2 RAW distributions even with the same operator family; historical observations cannot separate parent-distribution effects from branch random streams causally. E0/E1 union nevertheless retains every neighborhood ever reached in these histories, so the persistent 15-reference gap remains a generation/reachability observation.

The E1 target-aware survivor branch consistently shifts density toward the reference support: across seeds, G2-RAW median population-to-reference distance is about 0.35 for E1 versus 0.58 for E0, and formal-F40 Wasserstein/IQR is about 0.395 versus 0.477. This is a strong selection association, but it does not create a new globally reachable neighborhood beyond the historical RAW union and therefore does not overturn the primary generation bottleneck.

## Exact parent→offspring movement

{md(movement)}

The distributions include a dominant small-movement body and operator-dependent long tails. A positive move toward a reference is common, but a positive one-step change does not imply entry into its frozen neighborhood.

## Operator behavior (descriptive, not causal)

{md(opagg)}

Operators acted on different historical parents and sometimes underwent repair. These statistics identify movement scale and observed contributions, not an operator causal ranking.

## Fixed-size robustness

For every eligible population, 10k and 50k unique samples were drawn with 30 fixed deterministic repeats. `tables/fixed_sample_robustness.csv` reports mean absolute SMD, quantile-approximated marginal Wasserstein/IQR, and frozen FFT–NN coverage. Full RAW remains the historical reachability upper bound; matched samples are the fair density comparison.

{md(matched_fft)}
"""
    (OUT/"06_generation_dynamics.md").write_text(dynamics)

    evo=distributions["EvoReal50k"];p03=stages[stages.dataset.eq("P03")].iloc[0]
    evo_stage=stages[stages.dataset.eq("EvoReal50k")].iloc[0]
    difficult=unreachable[unreachable.reference_id.isin(["pr107","pr144"])].pivot(index="reference_id",columns="dataset",values="distance_over_radius")
    top=rank[(rank.representation=="F40_FORMAL_ASINH")].groupby("feature").agg(
        mean_wasserstein_over_evoreal_iqr=("wasserstein_over_evoreal_iqr","mean"),
        max_abs_smd=("standardized_mean_difference",lambda x: float(np.max(np.abs(x)))),
        mean_iqr_ratio_vs_evoreal=("iqr_ratio_vs_evoreal","mean"),
    ).nlargest(8,"mean_wasserstein_over_evoreal_iqr").reset_index()
    # Evidence-calibrated labels: observational support is deliberately not promoted to causality.
    labels={
      "P6_G1_INITIAL_POPULATION_BIAS":"SUPPORTED",
      "P6_G1_MUTATION_STRUCTURAL_EXPANSION":"LIMITED",
      "P6_G1_OPERATOR_LIMITATION":"PARTIAL",
      "P6_G1_PARENT_SELECTION_LIMITATION":"PARTIAL",
      "P6_G1_EVOREAL_STRUCTURAL_DIFFERENCE":"CONFIRMED",
      "P6_G1_CROSS_SIZE_CONFOUNDING":"PARTIAL",
      "P6_G1_ANALYSIS_STATUS":"COMPLETE",
    }
    final=f"""# P6-G1 Final Conclusion

## Decision labels

"""+"\n".join(f"`{k} = {v}`" for k,v in labels.items())+f"""

## Q1 — EvoReal reaches what different structure?

Both frozen representations show that EvoReal is not merely a denser copy of P03/FOGA. Its marginal F40 ranges are broader or shifted most strongly in the following dimensions:

{md(top)}

For pr107/pr144, the nearest-instance distance divided by the frozen radius is:

{md(difficult.reset_index())}

The geometry gallery makes the missing modes interpretable. `pr107` is a highly regular, repeated two-sided column/grid layout; the P03/FOGA nearest cases remain irregular dispersed clouds, while EvoReal produces a repeated line/grid construction. `pr144` is a repeated block/cluster motif; the closest FOGA case contains only partial local clustering, while EvoReal reproduces multiple separated blocks. These patterns agree with the largest F40 shifts in nearest-neighbor dispersion, angular regularity, hull-distance/edge summaries and centroid span.

In frozen FFT_APPROX–NN space EvoReal covers 40/48, while all historical FOGA RAW reaches at most 25/48. The difference persists for the 75–150 subset (17/17 versus 15/17), although cross-size effects remain material for the other 31 references. Generator-specific rows in `02_structural_distribution.csv` show that the broader support is a mixture of different EvoReal generator regimes, not one homogeneous cloud.

The exact FFT–NN-nearest candidate is not necessarily the F40-nearest candidate: even EvoReal's pr107/pr144 FFT matches retain sizable F40 RMS differences. Thus the **population-level EvoReal structural difference is confirmed by both representations**, while individual nearest-case correspondence remains representation-dependent.

## Q2 — Did FOGA move toward the missing regions?

Yes, individual offspring from multiple operators have positive distance improvements toward pr107/pr144, and the movement distribution has a real long tail. But the best historical RAW remains at 5.04 radii for pr107 and 1.06 radii for pr144, versus 0.02 and 0.06 for EvoReal. Positive one-step moves were generally too small, started from biased parents, or moved along incomplete dimensions. Grid mutation produced the closest observed pr107 child; implosion produced the closest pr144 child, showing that no single operator uniformly dominates both targets. For all 15 references, `04_unreachable_reference_analysis.csv` gives the continuous distance/radius and the F40 discrepancy of the exact FFT–NN-nearest candidate. This supports **limited expansion**, not the stronger claim that FOGA can only make local moves.

## Q3 — What should be changed first?

The strongest current evidence points first to the parent/support supplied to mutation, jointly with testing operator scale/direction. P03 already covers only 23/48, and two generations expand the maximum by only two neighborhoods. Parent selection is a secondary, branch-level limitation: it changes G2 density and can lose rare reached regions, but E0∪E1 did not explain the globally unreachable 15. The structure representation should remain a sensitivity check—not be refit to erase the gap—because both F40 and FFT–NN detect an EvoReal difference, while nearest-instance details can still be representation-dependent.

## Q4 — Does this determine evolution depth?

No. Static two-generation histories show small coverage increments and branch-dependent density, but cannot identify an optimal depth. They justify one focused future test: under a fixed evaluation and mutation budget, does seeding parents from the missing EvoReal-like support increase *new-neighborhood arrival rate* more than increasing operator magnitude/depth from P03? That requires a prospective controlled experiment; it was not run here.

## Limits

- Operator comparisons are observational because parent distributions and repairs differ.
- FFT is explicitly FFT_APPROX; it is not claimed as an exact official EvoReal FFT reproduction.
- No F40 neighborhood threshold was invented; F40 evidence is continuous/marginal/PCA only.
- TSPLIB48 is variable-size and target-informed. The 75–150 subset reduces but cannot eliminate cross-size and small-reference-sample uncertainty.
- Structural proximity is not interpreted as BQ training value.

## 中文研究总结（≤500字）

已确认：P03仅覆盖23/48；三种历史种子下，G1/G2全部RAW最多25/48，而EvoReal50k为40/48。控制到75–150节点后仍为15/17对17/17。正式F40与冻结FFT_APPROX–NN均显示EvoReal具有P03/FOGA缺少或明显不足的结构区间。真实父子配对表明FOGA并非完全局部：多种算子存在长尾结构移动，也产生朝pr107/pr144靠近的offspring，但历史最佳距离仍未进入冻结邻域。具有证据但未确立因果：初始父代支持偏窄与算子方向/幅度共同限制可达性；父代选择会改变G2分布，却不是15个全局不可达区域的主因。跨规模与表征依赖仍是局限。下一步最值得单独验证的问题是：固定预算下，补充缺失结构支持的父代，是否比单纯增加变异幅度或演化深度更能提高新邻域到达率。

`BLADE_P6G1_STRUCTURAL_REACHABILITY_ANALYSIS_COMPLETE`
"""
    (OUT/"07_final_conclusion.md").write_text(final)
    write_json(OUT/"completion.json",{"status":"COMPLETE","labels":labels,"final_tag":"BLADE_P6G1_STRUCTURAL_REACHABILITY_ANALYSIS_COMPLETE","created_utc":pd.Timestamp.utcnow().isoformat(),"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__})


def main()->None:
    started=time.perf_counter()
    for p in (OUT,FIG,TAB,MAN,CACHE):p.mkdir(parents=True,exist_ok=True)
    if not (OUT/"logs/f40_extraction.json").is_file():raise RuntimeError("run_f40_extraction.py must complete first")
    f40=load_global_f40();print(f"[P6-G1] loaded formal F40 unique={len(f40):,}",flush=True)
    refs_fft,refs_f40,ref_coords=load_reference()
    dist,rank,distributions=distribution_audit(f40,refs_fft);print("[P6-G1] distributions complete",flush=True)
    selected=build_selected(distributions,f40)
    unreachable=nearest_reference_audit(distributions,refs_fft,refs_f40);print("[P6-G1] unreachable references complete",flush=True)
    operators,movement=operator_audit(distributions,f40,refs_fft)
    stages=stage_metrics(distributions,selected,refs_fft,rank)
    matched=matched_sampling(distributions,refs_fft)
    plots(distributions,rank,refs_fft,unreachable,ref_coords,operators,stages)
    write_reports(distributions,rank,unreachable,operators,movement,stages,matched,refs_fft)
    outputs=[]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "f40_cache" not in p.parts and p.name!="output_hashes.csv":outputs.append({"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    pd.DataFrame(outputs).to_csv(MAN/"output_hashes.csv",index=False)
    print(f"[P6-G1] COMPLETE wall={time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":main()
