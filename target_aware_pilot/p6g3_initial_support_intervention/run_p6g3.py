#!/usr/bin/env python3
"""P6-G3 initial-population structural-support intervention.

This runner performs one paired FOGA-11 generation.  Historical P6-B G1 is
used as Arm A only after deterministic replay gates pass.  Arm B retains a
fixed random 9,000-member P03 subset and adds 1,000 independently generated
TSP100 parents.  No survivor selection, G2, solver inference, oracle, or
training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ART = OUT / "artifacts"
FEAT = OUT / "features"
F40_CACHE = OUT / "f40_cache"
FIG = OUT / "figures"
CFG = OUT / "configs"
LOG = OUT / "logs"

P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6G0 = ROOT / "target_aware_pilot/p6g0_raw_offspring_coverage_audit"
P6G1 = ROOT / "target_aware_pilot/p6g1_structural_reachability_analysis"
P6G2 = ROOT / "target_aware_pilot/p6g2_structural_similarity_validation"
P6E0 = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit"
P2 = ROOT / "target_aware_pilot/p2_bounded_evolution"
P6A = ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"
P03 = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz"
P03_META = P03.with_suffix(".parquet")
REF_MANIFEST = ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv"
REF_F40 = ROOT / "target_aware_pilot/p1_tsplib48_normalization_sensitivity.csv"
F40_MANIFEST = ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json"
SCALER = ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json"
R_EXTRACTOR = ROOT / "src/features/extract_salesperson_raw_features.R"
F40_WORKER = ROOT / "scripts/f40_one_shard.py"
PROFILE_PATH = ROOT / "src/mutation/foga2019_operators.yaml"

sys.path[:0] = [str(ROOT / "src"), str(P2)]
from final_run.science import cycle_child_seed, membership_hash  # noqa: E402
from mutation import OPERATOR_NAMES, apply_operator_with_config  # noqa: E402
from mutation.parameters import load_named_profile  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
import run_p2_bounded_evolution as p2  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E0MOD = load_module("p6g3_frozen_p6e0", P6E0 / "run_p6e0_audit.py")
G2MOD = load_module("p6g3_frozen_p6g2", P6G2 / "run_p6g2_analysis.py")

REPS = ("Original", "S1", "S2")
NAMESPACES = {"Original": 5, "S1": 2605011, "S2": 2605021}
POPULATION_N = 10_000
SUPPLEMENT_N = 1_000
KEEP_N = 9_000
ATTEMPTS = 110_000
SUPPLEMENT_SEED = 20261030
REMOVAL_SEED = 20261031
SAMPLE_SEEDS = tuple(20261201 + i for i in range(30))
GENERATOR_TYPES = (
    "parallel_lines",
    "jittered_global_grid",
    "separated_cluster_blocks",
    "repeated_local_motifs",
    "anisotropic_multiscale",
)
FOCUS11 = ("d198", "d657", "pr107", "pr144", "pr264", "u574", "u724", "eil51", "rat195", "ts225", "tsp225")
HISTORICAL15 = ("d198", "d657", "pcb1173", "pr107", "pr144", "pr264", "u574", "u724", "a280", "eil51", "pcb442", "pr299", "rat195", "ts225", "tsp225")
F40 = tuple(json.loads(F40_MANIFEST.read_text())["F40"])
STARTED = time.time()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def save_bundle(path: Path, xy: np.ndarray, hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, coordinates=np.asarray(xy, np.float64), instance_hash=np.asarray(hashes))
    os.replace(tmp, path)


def load_bundle(path: Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["coordinates"], np.float64), z["instance_hash"].astype(str).tolist()


def md(frame: pd.DataFrame, digits: int = 5) -> str:
    if frame.empty: return "_(empty)_"
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "|".join(["---"] * len(frame.columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value): cells.append("NA")
            elif isinstance(value, (float, np.floating)): cells.append(f"{float(value):.{digits}g}")
            else: cells.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ensure_dirs() -> None:
    for path in (OUT, ART, FEAT, F40_CACHE, FIG, CFG, LOG): path.mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def frozen_protocol() -> dict[str, Any]:
    return {
        "experiment": "BLADE-NCO P6-G3",
        "source_commit": git_commit(),
        "intervention": "fixed 10% initial structural support; P03 9000 + independent programmatic 1000",
        "population_n": POPULATION_N,
        "p03_keep_n": KEEP_N,
        "supplement_n": SUPPLEMENT_N,
        "removal_seed": REMOVAL_SEED,
        "supplement_seed": SUPPLEMENT_SEED,
        "supplement_generator_types": list(GENERATOR_TYPES),
        "supplement_per_type": 200,
        "generator_policy": {
            "parallel_lines": {"lines": [2, 9], "jitter": [0.0015, 0.018], "regular_along_line": True},
            "jittered_global_grid": {"rows": [4, 14], "jitter": [0.001, 0.018], "partial_grid": True},
            "separated_cluster_blocks": {"blocks": [3, 8], "local_mode": ["grid", "gaussian"], "separation_rejection": 200},
            "repeated_local_motifs": {"motifs": [4, 10], "motif": ["cross", "L", "microgrid", "line"], "shared_template": True},
            "anisotropic_multiscale": {"components": [3, 7], "filament_fraction": [0.35, 0.70], "scale_ratio": [3.0, 12.0]},
            "global_transform": "random rotation/reflection, uniform scale, bounded translation; no per-axis rescaling",
        },
        "generator_source": "new independent NumPy parameterized diagnostic generator; no EvoReal data/code and no TSPLIB coordinates",
        "generator_screen": "finite, [0,1], 100 unique nodes, min pair distance > 1e-5, whole-instance uniqueness only",
        "generations": 1,
        "operators": list(OPERATOR_NAMES),
        "operator_profile": "FOGA11_BASE_V1",
        "operator_profile_sha256": sha256_file(PROFILE_PATH),
        "attempts_per_parent": len(OPERATOR_NAMES),
        "attempts_per_arm_seed": ATTEMPTS,
        "replicates": list(REPS),
        "mutation_namespaces": NAMESPACES,
        "pairing": "same hash-derived RNG for 9000 common parents; exact historical A reused only after deterministic replay",
        "raw_filter": "accepted, valid_after_repair, non-identity; within-generation duplicates excluded from coordinate bundle",
        "random10k_repeats": 30,
        "random10k_seeds": list(SAMPLE_SEEDS),
        "focus11": list(FOCUS11),
        "historical15": list(HISTORICAL15),
        "fft_rules_sha256": sha256_file(P6E0 / "configs/reference_rules_frozen.json"),
        "f40_distance_config_sha256": sha256_file(P6G2 / "manifests/f40_distance_config.json"),
        "geometry_implementation_sha256": sha256_file(P6G2 / "run_p6g2_analysis.py"),
        "p03_sha256": sha256_file(P03),
        "p03_manifest_sha256": sha256_file(P03_META),
        "heldout22_used": False,
        "forbidden": ["survivor selection", "G2", "Concorde", "BQ/HGB/GP inference or training", "TSPLIB coordinate-derived generation"],
    }


def protocol_gate() -> dict[str, Any]:
    ensure_dirs()
    current = frozen_protocol()
    current["protocol_sha256"] = hashlib.sha256(json.dumps(current, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = CFG / "frozen_intervention_config.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != current: raise RuntimeError("P6-G3 frozen protocol drift")
    else:
        write_json(path, current)
    text = f"""# P6-G3 Intervention Protocol

The protocol was frozen before inspecting intervention outcomes.

- Arm A: unchanged formal P03, 10,000 parents.
- Arm B: a fixed random 9,000-member P03 subset plus 1,000 independently generated TSP100 parents (exactly 200 from each of five generic generator families).
- Removal seed `{REMOVAL_SEED}`; supplement seed `{SUPPLEMENT_SEED}`. No hardness or reference similarity enters removal or supplement acceptance.
- Supplement families: parallel lines, jittered global grids, separated cluster blocks, repeated local motifs, and anisotropic multiscale layouts. Parameters, orientation, scale, and translation are sampled from the frozen ranges in `configs/frozen_intervention_config.json`.
- No EvoReal instance or generator code is used. No TSPLIB coordinate is copied, cropped, resampled, or used to accept/reject a parent. TSPLIB48 enters only after generation as the frozen evaluation reference; TSPLIB22 is never loaded.
- Both arms have 10,000 parents and 110,000 conceptual FOGA-11 attempts per seed. For the 9,000 common parents, hash-derived RNG makes offspring exactly paired. Historical Arm-A artifacts are accepted only after parent/order/profile/namespace checks and deterministic replay.
- RAW analysis includes only valid, accepted, non-identity offspring; coordinate bundles are unique within generation. No survivor selection and no G2 are run.
- Frozen FFT–NN radius, P6-G2 F40 robust standardization, and P6-G2 independent geometry diagnostics are reused without retuning.

Protocol SHA256: `{current['protocol_sha256']}`.
"""
    (OUT / "01_intervention_protocol.md").write_text(text)
    return current


def global_transform(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(points, float)
    x = x - x.mean(axis=0)
    angle = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if rng.random() < .5: x[:, 0] *= -1
    x = x @ rot.T
    span = np.max(np.ptp(x, axis=0))
    x = x / max(span, 1e-12) * rng.uniform(.68, .88)
    lo, hi = x.min(axis=0), x.max(axis=0)
    lower = -lo + .04; upper = 1 - hi - .04
    shift = np.array([rng.uniform(lower[j], upper[j]) if lower[j] <= upper[j] else (lower[j] + upper[j]) / 2 for j in range(2)])
    return x + shift


def allocate(n: int, k: int, rng: np.random.Generator, minimum: int = 2) -> np.ndarray:
    remain = n - k * minimum
    weights = rng.dirichlet(np.ones(k) * 1.5)
    raw = weights * remain
    out = np.floor(raw).astype(int) + minimum
    for i in np.argsort(-(raw - np.floor(raw)))[:n - out.sum()]: out[i] += 1
    return out


def separated_centers(k: int, rng: np.random.Generator, minimum: float = .20) -> np.ndarray:
    centers: list[np.ndarray] = []
    for _ in range(200):
        candidate = rng.uniform(.12, .88, 2)
        if not centers or min(np.linalg.norm(candidate - x) for x in centers) >= minimum:
            centers.append(candidate)
            if len(centers) == k: return np.stack(centers)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, 2*np.pi)
    return np.column_stack([.5 + .33*np.cos(angles), .5 + .33*np.sin(angles)])


def generator_parallel_lines(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    k = int(rng.integers(2, 10)); counts = allocate(100, k, rng, 4)
    offsets = np.sort(rng.uniform(-.46, .46, k)); jitter = float(np.exp(rng.uniform(np.log(.0015), np.log(.018))))
    rows = []
    for off, n in zip(offsets, counts):
        t = np.linspace(-.5, .5, n) + rng.normal(0, jitter * .35, n)
        rows.append(np.column_stack([np.full(n, off) + rng.normal(0, jitter, n), t]))
    return global_transform(np.vstack(rows), rng), {"line_count": k, "counts": counts.tolist(), "jitter": jitter}


def generator_global_grid(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    rows = int(rng.integers(4, 15)); cols = int(math.ceil(115 / rows))
    gx, gy = np.meshgrid(np.linspace(-.5, .5, cols), np.linspace(-.5, .5, rows))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    take = rng.choice(len(grid), 100, replace=False)
    jitter = float(np.exp(rng.uniform(np.log(.001), np.log(.018))))
    return global_transform(grid[take] + rng.normal(0, jitter, (100, 2)), rng), {"rows": rows, "cols": cols, "available": len(grid), "jitter": jitter}


def generator_cluster_blocks(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    k = int(rng.integers(3, 9)); centers = separated_centers(k, rng, .16); counts = allocate(100, k, rng, 5)
    mode = "grid" if rng.random() < .55 else "gaussian"; points = []
    for center, n in zip(centers, counts):
        angle = rng.uniform(0, 2*np.pi); rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        if mode == "grid":
            a = int(np.ceil(np.sqrt(n))); u, v = np.meshgrid(np.linspace(-1,1,a), np.linspace(-1,1,a)); local = np.column_stack([u.ravel(),v.ravel()])[:n]
            local = local * rng.uniform(.018, .055, (1, 2)) + rng.normal(0, .0025, (n,2))
        else:
            local = rng.normal(0, 1, (n,2)) * rng.uniform(.018, .060, (1,2))
        points.append(center + local @ rot.T)
    return global_transform(np.vstack(points), rng), {"block_count": k, "counts": counts.tolist(), "local_mode": mode}


def base_motif(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if name == "cross":
        a = n // 2; return np.vstack([np.column_stack([np.linspace(-1,1,a), np.zeros(a)]), np.column_stack([np.zeros(n-a), np.linspace(-1,1,n-a)])])
    if name == "L":
        a = n // 2; return np.vstack([np.column_stack([np.linspace(-1,1,a), np.full(a,-1)]), np.column_stack([np.full(n-a,-1), np.linspace(-1,1,n-a)])])
    if name == "microgrid":
        a = int(np.ceil(np.sqrt(n))); u,v=np.meshgrid(np.linspace(-1,1,a),np.linspace(-1,1,a)); return np.column_stack([u.ravel(),v.ravel()])[:n]
    return np.column_stack([np.linspace(-1,1,n), np.zeros(n)])


def generator_repeated_motifs(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    k = int(rng.integers(4, 11)); centers = separated_centers(k, rng, .13); counts = allocate(100, k, rng, 5)
    motif = str(rng.choice(["cross", "L", "microgrid", "line"])); points = []
    for center, n in zip(centers, counts):
        local = base_motif(motif, int(n), rng); angle = rng.uniform(-.35, .35) + rng.choice([0, np.pi/2, np.pi, 3*np.pi/2])
        rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local = local * rng.uniform(.018, .055, (1,2)) + rng.normal(0, .0025, (n,2))
        points.append(center + local @ rot.T)
    return global_transform(np.vstack(points), rng), {"motif_count": k, "counts": counts.tolist(), "motif": motif}


def generator_anisotropic(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    k = int(rng.integers(3, 8)); centers = separated_centers(k, rng, .13); counts = allocate(100, k, rng, 5)
    frac = float(rng.uniform(.35, .70)); ratio = float(rng.uniform(3, 12)); points=[]; filament_n=0
    for i,(center,n) in enumerate(zip(centers,counts)):
        filament = i < int(round(k*frac)); angle=rng.uniform(0,2*np.pi);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        major=rng.uniform(.04,.12); minor=major/ratio if filament else rng.uniform(.025,.07)
        local=rng.normal(0,1,(n,2))*np.array([major,minor]);points.append(center+local@rot.T);filament_n+=int(filament)*int(n)
    return global_transform(np.vstack(points), rng), {"component_count": k, "counts": counts.tolist(), "filament_fraction_target": frac, "filament_nodes": filament_n, "scale_ratio": ratio}


GENERATORS = {
    "parallel_lines": generator_parallel_lines,
    "jittered_global_grid": generator_global_grid,
    "separated_cluster_blocks": generator_cluster_blocks,
    "repeated_local_motifs": generator_repeated_motifs,
    "anisotropic_multiscale": generator_anisotropic,
}


def min_pair_distance(x: np.ndarray) -> float:
    d = np.sqrt(np.square(x[:,None,:] - x[None,:,:]).sum(2)); np.fill_diagonal(d, np.inf)
    return float(d.min())


def build_parents() -> tuple[dict[str, tuple[np.ndarray, pd.DataFrame]], dict[str, Any]]:
    pxy, ph = load_bundle(P03); pm = pd.read_parquet(P03_META)
    if pxy.shape != (10000,100,2) or ph != pm.instance_hash.astype(str).tolist() or ph != [coordinate_hash(x) for x in pxy]: raise RuntimeError("P03 integrity")
    rng = np.random.default_rng(REMOVAL_SEED); removed = np.sort(rng.choice(POPULATION_N, SUPPLEMENT_N, replace=False)); keep_mask=np.ones(POPULATION_N,bool);keep_mask[removed]=False;kept=np.flatnonzero(keep_mask)
    supplement_path=ART/"supplement_parents.npz";supplement_meta=ART/"supplement_parents.parquet"
    if supplement_path.exists() and supplement_meta.exists():
        sxy,sh=load_bundle(supplement_path);sm=pd.read_parquet(supplement_meta)
        if sh != sm.instance_hash.astype(str).tolist():raise RuntimeError("supplement cache alignment")
    else:
        srng=np.random.default_rng(SUPPLEMENT_SEED); coords=[];rows=[];seen=set(ph);attempts=0
        for kind in GENERATOR_TYPES:
            accepted=0
            while accepted<200:
                attempts+=1;x,params=GENERATORS[kind](srng);h=coordinate_hash(x);mind=min_pair_distance(x)
                valid=x.shape==(100,2) and np.isfinite(x).all() and (x>=0).all() and (x<=1).all() and len(np.unique(x,axis=0))==100 and mind>1e-5 and h not in seen
                if not valid:continue
                seen.add(h);coords.append(x);rows.append({"instance_hash":h,"generator_type":kind,"generator_index":accepted,"generation_attempt":attempts,"generator_seed":SUPPLEMENT_SEED,"parameters_json":json.dumps(params,sort_keys=True),"minimum_pair_distance":mind,"parent_source":"supplement"});accepted+=1
        sxy=np.stack(coords);sm=pd.DataFrame(rows);sh=sm.instance_hash.astype(str).tolist();save_bundle(supplement_path,sxy,sh);sm.to_parquet(supplement_meta,index=False)
        write_json(ART/"supplement_generation_runtime.json",{"attempted_candidates":attempts,"accepted_candidates":len(sm),"rejected_candidates":attempts-len(sm),"wall_seconds":time.time()-STARTED})
    if len(sm)!=1000 or sm.generator_type.value_counts().to_dict()!={k:200 for k in GENERATOR_TYPES}:raise RuntimeError("supplement composition")
    am=pm.copy();am["parent_source"]="P03_original";am["generator_type"]="P03";am["original_p03_index"]=np.arange(10000)
    bm=pd.concat([am.iloc[kept].copy(),sm.assign(original_p03_index=-1)],ignore_index=True,sort=False)
    bxy=np.concatenate([pxy[kept],sxy]);bh=bm.instance_hash.astype(str).tolist()
    if len(bh)!=10000 or len(set(bh))!=10000 or bh != [coordinate_hash(x) for x in bxy]:raise RuntimeError("B parent integrity")
    save_bundle(ART/"A_original_parents.npz",pxy,ph);am.to_parquet(ART/"A_original_parents.parquet",index=False)
    save_bundle(ART/"B_augmented_parents.npz",bxy,bh);bm.to_parquet(ART/"B_augmented_parents.parquet",index=False)
    pd.DataFrame({"removed_p03_index":removed,"removed_instance_hash":[ph[i] for i in removed]}).to_csv(ART/"removed_p03_random1000.csv",index=False)
    info={"p03_n":10000,"kept_n":9000,"removed_n":1000,"supplement_n":1000,"supplement_attempted":json.loads((ART/"supplement_generation_runtime.json").read_text()).get("attempted_candidates",1000),"common_parent_membership_sha256":membership_hash([ph[i] for i in kept]),"A_membership_sha256":membership_hash(ph),"B_membership_sha256":membership_hash(bh),"supplement_membership_sha256":membership_hash(sh),"removed_index_sha256":hashlib.sha256(removed.tobytes()).hexdigest()}
    write_json(ART/"parent_intervention_integrity.json",info)
    return {"A":(pxy,am),"B":(bxy,bm)},info


def historical_paths(rep: str) -> tuple[Path,Path,Path]:
    base=P6B/f"evolution/{rep}/shared_G1/mutation";return base/"offspring_coordinates.npz",base/"mutation_manifest.parquet",base/"runtime.json"


def local_a_path(rep: str) -> Path:
    return ART/rep/"A_G1_RAW/offspring_coordinates.npz"


def materialize_nonidentity_a(rep: str) -> dict[str,Any]:
    output=local_a_path(rep);info_path=output.parent/"runtime.json"
    if output.exists() and info_path.exists():return json.loads(info_path.read_text())
    output.parent.mkdir(parents=True,exist_ok=True);source,manifest_path,_=historical_paths(rep);xy,hh=load_bundle(source);m=pd.read_parquet(manifest_path);accepted=m.proposal_status.eq("accepted")&m.valid_after_repair&~m.within_offspring_duplicate
    if hh!=m.loc[accepted,"child_hash"].astype(str).tolist():raise RuntimeError("A source alignment")
    keep=~m.loc[accepted,"child_hash"].astype(str).eq(m.loc[accepted,"parent_hash"].astype(str)).to_numpy();outxy=xy[keep];outh=np.asarray(hh)[keep].tolist();save_bundle(output,outxy,outh)
    info={"status":"PASS","source_historical_bundle":str(source.relative_to(ROOT)),"source_sha256":sha256_file(source),"accepted_unique_source_n":len(hh),"identity_removed_n":int((~keep).sum()),"valid_nonidentity_unique_n":len(outh),"bundle_sha256":sha256_file(output)};write_json(info_path,info);return info


def replay_gate(rep: str, axy: np.ndarray, am: pd.DataFrame) -> dict[str, Any]:
    bundle,manifest_path,runtime_path=historical_paths(rep);hist=pd.read_parquet(manifest_path);runtime=json.loads(runtime_path.read_text());profile=load_named_profile("FOGA11_BASE_V1");ah=am.instance_hash.astype(str).tolist()
    checks={"manifest_attempts":len(hist)==ATTEMPTS,"runtime_namespace":int(runtime["cycle_seed_namespace"])==NAMESPACES[rep],"runtime_parent_membership":runtime["parent_membership_sha256"]==membership_hash(ah),"profile_hash":runtime["profile_sha256"]==sha256_file(PROFILE_PATH),"parent_schedule":True,"replay_hash":True,"replay_seed":True,"replay_status":True}
    for i,h in enumerate(ah):
        q=hist.iloc[i*11:(i+1)*11]
        if len(q)!=11 or not q.parent_hash.astype(str).eq(h).all() or q.operator_id.astype(str).tolist()!=list(OPERATOR_NAMES):checks["parent_schedule"]=False;break
    replay_indices=(0,137,5001,9999)
    for pi in replay_indices:
        h=ah[pi]
        for oi,op in enumerate(OPERATOR_NAMES):
            row=hist.iloc[pi*11+oi];seed=cycle_child_seed(NAMESPACES[rep],h,op);res=apply_operator_with_config(op,axy[pi],np.random.default_rng(seed),profile)
            checks["replay_hash"] &= str(row.child_hash)==res.final_output_hash
            checks["replay_seed"] &= int(row.child_seed)==int(seed)
            checks["replay_status"] &= str(row.proposal_status)==res.proposal_status and bool(row.valid_after_repair)==bool(res.valid_after_repair)
    if not all(checks.values()):raise RuntimeError(f"historical A replay gate failed {rep}: {checks}")
    xy,hh=load_bundle(bundle);accepted=hist.proposal_status.eq("accepted")&hist.valid_after_repair&~hist.within_offspring_duplicate
    if hh!=hist.loc[accepted,"child_hash"].astype(str).tolist() or hh!=[coordinate_hash(x) for x in xy]:raise RuntimeError("historical bundle alignment")
    return {**checks,"replayed_parent_n":len(replay_indices),"replayed_attempt_n":len(replay_indices)*11,"historical_raw_n":len(hh),"historical_bundle_sha256":sha256_file(bundle),"historical_manifest_sha256":sha256_file(manifest_path)}


def mutation_row(rep: str, parent_index: int, parent_hash: str, operator_index: int, result: Any, seed: int) -> dict[str,Any]:
    op=list(OPERATOR_NAMES)[operator_index]
    return {"replicate":rep,"arm_scope":"B_AUGMENTED","generation":1,"attempt_index":parent_index*11+operator_index,"parent_index":parent_index,"parent_hash":parent_hash,"operator_id":op,"operator_index":operator_index,"child_seed":np.uint64(seed),"cycle_seed_namespace":NAMESPACES[rep],"mutation_profile":"FOGA11_BASE_V1","proposal_status":result.proposal_status,"failure_reason":result.failure_reason,"valid_before_repair":bool(result.valid_before_repair),"valid_after_repair":bool(result.valid_after_repair),"repair_count":int(result.repair_count),"repair_fraction":float(result.repair_fraction),"n_affected":int(result.n_affected),"raw_child_hash":result.raw_output_hash,"child_hash":result.final_output_hash,"within_offspring_duplicate":False,**p2.scalar_trace(op,dict(result.sampled_parameters)),**{k:float(v) for k,v in result.timing.items()}}


def build_b_g1(rep: str, bxy: np.ndarray, bm: pd.DataFrame, axy: np.ndarray, am: pd.DataFrame) -> dict[str,Any]:
    out=ART/rep/"B_G1_RAW";bundle_out=out/"offspring_coordinates.npz";manifest_out=out/"mutation_manifest.parquet";runtime_out=out/"runtime.json"
    if all(p.exists() for p in (bundle_out,manifest_out,runtime_out)):
        return json.loads(runtime_out.read_text())
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();hist_bundle,hist_manifest,_=historical_paths(rep);hist=pd.read_parquet(hist_manifest);hxy,hh=load_bundle(hist_bundle);accepted_hist=hist.proposal_status.eq("accepted")&hist.valid_after_repair&~hist.within_offspring_duplicate
    if hh!=hist.loc[accepted_hist,"child_hash"].astype(str).tolist():raise RuntimeError("historical coordinate map alignment")
    hindex={h:i for i,h in enumerate(hh)};hist_by_parent={h:g.copy() for h,g in hist.groupby(hist.parent_hash.astype(str),sort=False)};supplement=set(bm.loc[bm.parent_source.eq("supplement"),"instance_hash"].astype(str));profile=load_named_profile("FOGA11_BASE_V1")
    rows=[];new_coords={};fresh_attempts=0
    for pi,(parent_hash,source) in enumerate(zip(bm.instance_hash.astype(str),bm.parent_source.astype(str))):
        if source!="supplement":
            q=hist_by_parent[parent_hash].copy().sort_values("operator_index")
            if len(q)!=11:raise RuntimeError("common parent historical attempts")
            q["replicate"]=rep;q["arm_scope"]="B_AUGMENTED";q["parent_index"]=pi;q["attempt_index"]=np.arange(pi*11,(pi+1)*11);rows.extend(q.to_dict("records"))
        else:
            for oi,op in enumerate(OPERATOR_NAMES):
                seed=cycle_child_seed(NAMESPACES[rep],parent_hash,op);result=apply_operator_with_config(op,bxy[pi],np.random.default_rng(seed),profile);rows.append(mutation_row(rep,pi,parent_hash,oi,result,seed));fresh_attempts+=1
                if result.proposal_status=="accepted" and result.valid_after_repair:new_coords[result.final_output_hash]=np.asarray(result.final_coordinates,np.float64)
        if (pi+1)%1000==0:print(f"[P6-G3][{rep}] B parents {pi+1}/10000 fresh_attempts={fresh_attempts}",flush=True)
    frame=pd.DataFrame(rows).sort_values("attempt_index").reset_index(drop=True)
    if len(frame)!=ATTEMPTS or frame.operator_id.value_counts().ne(10000).any():raise RuntimeError("B mutation budget")
    eligible=frame.proposal_status.eq("accepted")&frame.valid_after_repair&frame.child_hash.astype(str).ne(frame.parent_hash.astype(str));seen=set();dup=[];coords=[];hashes=[]
    for i,row in frame.iterrows():
        if not eligible.iat[i]:dup.append(False);continue
        h=str(row.child_hash);isdup=h in seen;dup.append(isdup)
        if isdup:continue
        seen.add(h);hashes.append(h)
        if h in new_coords:coords.append(new_coords[h])
        elif h in hindex:coords.append(hxy[hindex[h]])
        else:raise RuntimeError(f"missing child coordinate {h}")
    frame["within_offspring_duplicate"]=dup;frame["non_identity"]=frame.child_hash.astype(str).ne(frame.parent_hash.astype(str));oxy=np.stack(coords)
    if hashes!=[coordinate_hash(x) for x in oxy]:raise RuntimeError("B offspring hash")
    save_bundle(bundle_out,oxy,hashes);frame.to_parquet(manifest_out,index=False)
    runtime={"status":"PASS","replicate":rep,"cycle_seed_namespace":NAMESPACES[rep],"nominal_attempts":len(frame),"conceptual_fresh_or_reused_attempts":len(frame),"physically_reused_common_attempts":KEEP_N*11,"physically_new_supplement_attempts":fresh_attempts,"valid_attempts":int((frame.proposal_status.eq('accepted')&frame.valid_after_repair).sum()),"identity_attempts":int((~frame.non_identity).sum()),"valid_nonidentity_unique_n":len(hashes),"within_offspring_duplicates":int(frame.within_offspring_duplicate.sum()),"supplement_parent_n":len(supplement),"supplement_offspring_unique_n":int(frame.loc[eligible&~frame.within_offspring_duplicate&frame.parent_hash.astype(str).isin(supplement)].child_hash.nunique()),"common_parent_attempts_exactly_paired":True,"rng_rule":"cycle_child_seed(namespace,parent_hash,operator_id)","parent_membership_sha256":membership_hash(bm.instance_hash.astype(str).tolist()),"offspring_membership_sha256":membership_hash(hashes),"wall_seconds_incremental":time.perf_counter()-started,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"bundle_sha256":sha256_file(bundle_out),"manifest_sha256":sha256_file(manifest_out)};write_json(runtime_out,runtime)
    return runtime


def run_mutation() -> tuple[dict[str,tuple[np.ndarray,pd.DataFrame]],dict[str,Any]]:
    parents,pinfo=build_parents();axy,am=parents["A"];bxy,bm=parents["B"];replays={};runs={}
    for rep in REPS:
        replays[rep]=replay_gate(rep,axy,am);replays[rep]["nonidentity_A"]=materialize_nonidentity_a(rep);runs[rep]=build_b_g1(rep,bxy,bm,axy,am);write_json(LOG/"replay_gates.json",replays)
    write_json(LOG/"mutation_runs.json",runs);return parents,{"parent":pinfo,"replays":replays,"runs":runs}


def feature_frame(tag: str, path: Path) -> pd.DataFrame:
    output=FEAT/f"{tag}_fft_nn.parquet"
    if output.exists():
        old=pd.read_parquet(output);_,h=load_bundle(path)
        if old.instance_hash.astype(str).tolist()==h:return old
    xy,h=load_bundle(path);jobs=[(i,np.asarray(xy[i:i+500],np.float64)) for i in range(0,len(xy),500)];parts=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done,(start,rows) in enumerate(pool.map(E0MOD.feature_chunk,jobs),1):
            q=pd.DataFrame(rows)[["fft_energy","nn_ratio","duplicate_coordinate_n"]];q.insert(0,"instance_hash",h[start:start+len(q)]);parts.append((start,q))
            if done==1 or done%25==0 or done==len(jobs):print(f"[P6-G3][FFT] {tag} {done}/{len(jobs)}",flush=True)
    parts.sort();out=pd.concat([x[1] for x in parts],ignore_index=True);out.to_parquet(output,index=False);return out


def scaler_params() -> dict[str,tuple[float,float]]:
    payload=json.loads(SCALER.read_text());table={x["feature_name"]:x for x in payload["parameters"]};return {name:(float(table[name]["center_median"]),float(table[name]["scale"])) for name in F40}


def run_f40_job(job: dict[str,Any]) -> str:
    import pickle,signal,subprocess
    process=subprocess.Popen([sys.executable,str(F40_WORKER)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:stdout,stderr=process.communicate(pickle.dumps(job),timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait();raise RuntimeError(f"F40 timeout {job['out']}")
    if process.returncode:raise RuntimeError(f"F40 failed {job['out']}: {stderr.decode(errors='replace')[-2000:]}")
    return stdout.decode(errors="replace").strip()


def f40_new_cache(paths: list[Path]) -> pd.DataFrame:
    formal=pd.read_parquet(P6G1/"cache/formal_f40_all.parquet");known=set(formal.instance_hash.astype(str));coord={}
    for path in paths:
        xy,hh=load_bundle(path)
        for x,h in zip(xy,hh):
            if h not in known:coord.setdefault(h,np.asarray(x,np.float64))
    hashes=list(coord);existing=[]
    for path in sorted(F40_CACHE.glob("f40_*.parquet")):
        q=pd.read_parquet(path);existing.extend(q.instance_hash.astype(str).tolist())
    missing=[h for h in hashes if h not in set(existing)];jobs=[];rscript=shutil.which("Rscript") or "/home/haozou/micromamba/envs/blade-r/bin/Rscript";params=scaler_params()
    for start in range(0,len(missing),250):
        block=missing[start:start+250];token=membership_hash(block)[:20];jobs.append({"coordinates":np.stack([coord[h] for h in block]),"hashes":block,"names":list(F40),"out":str(F40_CACHE/f"f40_{token}.parquet"),"rscript":rscript,"r_script":str(R_EXTRACTOR),"params":params})
    if jobs:
        print(f"[P6-G3][F40] missing={len(missing)} shards={len(jobs)}",flush=True);t=time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(12,len(jobs))) as executor:
            futures=[executor.submit(run_f40_job,j) for j in jobs]
            for done,future in enumerate(as_completed(futures),1):
                future.result()
                if done==1 or done%20==0 or done==len(jobs):print(f"[P6-G3][F40] {done}/{len(jobs)}",flush=True)
        write_json(LOG/"f40_extraction_runtime.json",{"missing_n":len(missing),"shards":len(jobs),"workers":min(12,len(jobs)),"wall_seconds":time.perf_counter()-t})
    new=pd.concat([pd.read_parquet(p) for p in sorted(F40_CACHE.glob("f40_*.parquet"))],ignore_index=True).drop_duplicates("instance_hash") if list(F40_CACHE.glob("f40_*.parquet")) else pd.DataFrame(columns=["instance_hash",*F40])
    full=pd.concat([formal,new],ignore_index=True).drop_duplicates("instance_hash",keep="first");return full.set_index("instance_hash",drop=False)


def reference_data() -> tuple[pd.DataFrame,np.ndarray,dict[str,np.ndarray]]:
    manifest=pd.read_csv(REF_MANIFEST);fft=pd.read_parquet(P6E0/"features/TSPLIB48.parquet");refs=manifest[["instance_id","n_nodes","evoreal_split"]].merge(fft[["instance_id","fft_energy","nn_ratio"]],on="instance_id",validate="one_to_one")
    rf=pd.read_csv(REF_F40);rf=rf[rf.coordinate_variant.eq("aspect")].set_index("instance_id").loc[refs.instance_id,list(F40)].to_numpy(float)
    coords={r.instance_id:G2MOD.parse_tsplib(ROOT/r.source_file) for r in manifest.itertuples(index=False)}
    if len(refs)!=48 or refs.instance_id.nunique()!=48:raise RuntimeError("reference integrity")
    return refs,rf,coords


def fft_dist(frame: pd.DataFrame, refs: pd.DataFrame) -> np.ndarray:
    rule=json.loads((P6E0/"configs/reference_rules_frozen.json").read_text());return cdist((frame[["fft_energy","nn_ratio"]].to_numpy(float)-np.asarray(rule["center"]))/np.asarray(rule["scale_iqr"]),(refs[["fft_energy","nn_ratio"]].to_numpy(float)-np.asarray(rule["center"]))/np.asarray(rule["scale_iqr"]))


def f40_dist(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    cfg=json.loads((P6G2/"manifests/f40_distance_config.json").read_text());center=np.asarray(cfg["center_values"]);scale=np.asarray(cfg["scale_values"]);eff=np.asarray(cfg["effective_mask"],bool);x=(values-center)/scale;r=(ref-center)/scale;out=np.empty((len(x),len(r)),np.float32)
    for j,z in enumerate(r):
        mask=eff&np.isfinite(z);d=x[:,mask]-z[mask];valid=np.isfinite(d);n=valid.sum(1);v=np.sqrt(np.divide(np.nansum(d*d,axis=1),n,out=np.full(len(d),np.inf),where=n>0));v[n<30]=np.inf;out[:,j]=v
    return out


def load_pool(tag: str, path: Path, f40: pd.DataFrame) -> dict[str,Any]:
    xy,hh=load_bundle(path);ff=feature_frame(tag,path);values=f40.loc[hh,list(F40)].to_numpy(float);return {"tag":tag,"path":path,"xy":xy,"hashes":hh,"fft_frame":ff,"f40":values}


def coverage_records(pool: dict[str,Any], group: str, seed: str, stage: str, refs: pd.DataFrame, xdist: np.ndarray, mode: str="full", draw: int=-1, index: np.ndarray|None=None) -> dict[str,Any]:
    if index is None:
        index=np.arange(len(pool["hashes"]))
    d=xdist[index]
    radius=float(json.loads((P6E0/"configs/reference_rules_frozen.json").read_text())["radius"])
    covered=(d.min(0)<=radius)
    matched=refs.n_nodes.between(75,150).to_numpy()
    focus=refs.instance_id.isin(FOCUS11).to_numpy()
    hist=refs.instance_id.isin(HISTORICAL15).to_numpy()
    return {"record_type":"coverage","group":group,"seed":seed,"stage":stage,"evaluation_mode":mode,"draw":draw,"population_n":len(index),"unique_n":len(set(np.asarray(pool['hashes'])[index])),"coverage_48":int(covered.sum()),"coverage_17":int(covered[matched].sum()),"coverage_focus11":int(covered[focus].sum()),"coverage_historical15":int(covered[hist].sum()),"within_any_radius_n":int((d<=radius).any(1).sum()),"nearest_reference_mean":float(d.min(0).mean()),"nearest_reference_p90":float(np.quantile(d.min(0),.9))}


def analyze(parents: dict[str,tuple[np.ndarray,pd.DataFrame]], runinfo: dict[str,Any]) -> None:
    paths=[ART/"A_original_parents.npz",ART/"B_augmented_parents.npz"]+[local_a_path(r) for r in REPS]+[ART/r/"B_G1_RAW/offspring_coordinates.npz" for r in REPS]
    f40=f40_new_cache(paths);refs,rf40,rcoords=reference_data();radius=float(json.loads((P6E0/"configs/reference_rules_frozen.json").read_text())["radius"])
    pools={"A_PARENT":load_pool("A_PARENT",paths[0],f40),"B_PARENT":load_pool("B_PARENT",paths[1],f40)}
    for rep in REPS:
        pools[f"A_{rep}_G1"]=load_pool(f"A_{rep}_G1",local_a_path(rep),f40);pools[f"B_{rep}_G1"]=load_pool(f"B_{rep}_G1",ART/rep/"B_G1_RAW/offspring_coordinates.npz",f40)
    xd={k:fft_dist(v["fft_frame"],refs) for k,v in pools.items()};hd={k:f40_dist(v["f40"],rf40) for k,v in pools.items()}
    # Parent audit: integrity/composition and per-reference coverage.
    prows=[]
    for group,key in (("A","A_PARENT"),("B","B_PARENT")):
        p=pools[key];meta=parents[group][1];cov=coverage_records(p,group,"ALL","PARENT",refs,xd[key]);prows.append({**cov,"parent_source":"ALL","generator_type":"ALL","component_n":len(meta)})
        for (source,generator),g in meta.groupby(["parent_source","generator_type"],dropna=False):
            component=(xd[key][g.index.to_numpy()]<=radius).any(0);matched=refs.n_nodes.between(75,150).to_numpy();focus=refs.instance_id.isin(FOCUS11).to_numpy();hist=refs.instance_id.isin(HISTORICAL15).to_numpy()
            prows.append({"record_type":"composition","group":group,"seed":"ALL","stage":"PARENT","parent_source":source,"generator_type":generator,"component_n":len(g),"population_n":len(meta),"unique_n":meta.instance_hash.nunique(),"coverage_48":int(component.sum()),"coverage_17":int(component[matched].sum()),"coverage_focus11":int(component[focus].sum()),"coverage_historical15":int(component[hist].sum())})
        covered=(xd[key]<=radius)
        for j,r in refs.iterrows():prows.append({"record_type":"reference","group":group,"seed":"ALL","stage":"PARENT","reference_id":r.instance_id,"n_nodes":r.n_nodes,"matched_75_150":bool(75<=r.n_nodes<=150),"focus11":r.instance_id in FOCUS11,"historical15":r.instance_id in HISTORICAL15,"covered":bool(covered[:,j].any()),"neighborhood_n":int(covered[:,j].sum()),"nearest_distance":float(xd[key][:,j].min()),"nearest_over_radius":float(xd[key][:,j].min()/radius)})
    parent_audit=pd.DataFrame(prows);parent_audit.to_csv(OUT/"02_parent_population_audit.csv",index=False)
    # G1 coverage and fixed-10k samples.
    crows=[]
    for rep in REPS:
        for group in ("A","B"):
            key=f"{group}_{rep}_G1";p=pools[key];crows.append(coverage_records(p,group,rep,"G1_RAW",refs,xd[key]))
            for draw,seed in enumerate(SAMPLE_SEEDS):idx=np.random.default_rng(seed+int(hashlib.sha256(f'{group}:{rep}'.encode()).hexdigest()[:8],16)).choice(len(p["hashes"]),10000,replace=False);crows.append(coverage_records(p,group,rep,"G1_RAW",refs,xd[key],"random10k",draw,idx))
    coverage=pd.DataFrame(crows);summ=[]
    for keys,g in coverage[coverage.evaluation_mode.eq("random10k")].groupby(["group","seed","stage"]):
        row={"record_type":"random10k_summary","group":keys[0],"seed":keys[1],"stage":keys[2],"evaluation_mode":"random10k_summary","draw":-1,"population_n":10000}
        for col in ("coverage_48","coverage_17","coverage_focus11","coverage_historical15","within_any_radius_n","nearest_reference_mean"):
            v=g[col].to_numpy(float);row[col]=v.mean();row[col+"_sd"]=v.std(ddof=1);row[col+"_p025"]=np.quantile(v,.025);row[col+"_p975"]=np.quantile(v,.975)
        summ.append(row)
    coverage=pd.concat([coverage,pd.DataFrame(summ)],ignore_index=True,sort=False);coverage.to_csv(OUT/"03_g1_raw_coverage.csv",index=False)
    # Reference arrivals with lineage attribution.
    arrivals=[];parent_cov={g:(xd[f"{g}_PARENT"]<=radius).any(0) for g in ("A","B")}
    supplement_meta=parents["B"][1].loc[parents["B"][1].parent_source.eq("supplement"),["instance_hash","generator_type"]]
    supplement_hashes=set(supplement_meta.instance_hash.astype(str));supplement_generator=supplement_meta.set_index("instance_hash").generator_type.astype(str)
    for rep in REPS:
        for group in ("A","B"):
            key=f"{group}_{rep}_G1";p=pools[key];member=xd[key]<=radius;manifest_path=historical_paths(rep)[1] if group=="A" else ART/rep/"B_G1_RAW/mutation_manifest.parquet";m=pd.read_parquet(manifest_path);eligible=m.proposal_status.eq("accepted")&m.valid_after_repair&m.child_hash.astype(str).ne(m.parent_hash.astype(str))&~m.within_offspring_duplicate;lookup=m.loc[eligible].drop_duplicates("child_hash").set_index("child_hash")
            for j,r in refs.iterrows():
                hs=np.asarray(p["hashes"])[member[:,j]];q=lookup.reindex(hs).dropna(subset=["parent_hash"]);sup=q.parent_hash.astype(str).isin(supplement_hashes);ops=q.operator_id.value_counts();offspring_cov=bool(len(hs));initial=bool(parent_cov[group][j]);history_unreachable=r.instance_id in HISTORICAL15
                gen_counts=q.loc[sup,"parent_hash"].astype(str).map(supplement_generator).value_counts()
                category="NO_ARRIVAL"
                if offspring_cov and not initial:category="MUTATION_EXPANSION_BEYOND_GROUP_SEEDS"
                elif offspring_cov and initial:category="MUTATION_RETENTION_FROM_COVERED_SEED"
                arrivals.append({"seed":rep,"group":group,"reference_id":r.instance_id,"n_nodes":r.n_nodes,"matched_75_150":bool(75<=r.n_nodes<=150),"focus11":r.instance_id in FOCUS11,"historical15":history_unreachable,"initial_parent_covered":initial,"g1_raw_covered":offspring_cov,"arrival_category":category,"offspring_in_neighborhood_n":len(hs),"supplement_parent_offspring_n":int(sup.sum()),"original_parent_offspring_n":int((~sup).sum()),"supplement_generator_counts":";".join(f"{k}:{int(v)}" for k,v in gen_counts.items()),"distinct_parent_n":int(q.parent_hash.nunique()),"distinct_operator_n":int(q.operator_id.nunique()),"top_operator":ops.index[0] if len(ops) else "","top_operator_n":int(ops.iloc[0]) if len(ops) else 0,"nearest_offspring_distance":float(xd[key][:,j].min()),"nearest_over_radius":float(xd[key][:,j].min()/radius),"B_initial_direct_injection":bool(group=="B" and initial and not parent_cov["A"][j]),"B_true_new_beyond_B_initial":bool(group=="B" and offspring_cov and not initial)})
    arrivals=pd.DataFrame(arrivals);arrivals.to_csv(OUT/"04_new_neighborhood_arrivals.csv",index=False)
    # F40 and independent geometry: top-1 by FFT and F40 for every stage/reference.
    grows=[];evo_features=pd.read_parquet(P6E0/"features/EvoReal50k.parquet",columns=["instance_hash"]);evo_f40=f40.loc[evo_features.instance_hash.astype(str),list(F40)].to_numpy(float);cfg=json.loads((P6G2/"manifests/f40_distance_config.json").read_text());fc=np.asarray(cfg["center_values"]);fs=np.asarray(cfg["scale_values"]);eff=np.asarray(cfg["effective_mask"],bool);ez=(evo_f40[:,eff]-fc[eff])/fs[eff]
    for key,p in pools.items():
        group=key[0];seed="ALL" if key.endswith("PARENT") else key.split("_")[1];stage="PARENT" if key.endswith("PARENT") else "G1_RAW";z=(p["f40"][:,eff]-fc[eff])/fs[eff]
        # Distribution diagnostic uses fixed marginals and effective rank, no learned threshold.
        marginal=float(np.nanmean([wasserstein_distance(z[np.isfinite(z[:,j]),j],ez[np.isfinite(ez[:,j]),j]) for j in range(z.shape[1])]))
        sample=z[np.random.default_rng(20261030).choice(len(z),min(10000,len(z)),replace=False)];sample=np.nan_to_num(sample,nan=np.nanmedian(sample,axis=0));s=np.linalg.svd(sample-sample.mean(0),compute_uv=False);effective_rank=float(np.square(s.sum())/np.square(s).sum())
        grows.append({"record_type":"stage_summary","group":group,"seed":seed,"stage":stage,"population_n":len(p["hashes"]),"evoreal_f40_marginal_w1_mean":marginal,"f40_effective_rank":effective_rank,"f40_best_reference_mean":float(hd[key].min(0).mean()),"f40_best_reference_p90":float(np.quantile(hd[key].min(0),.9))})
        for j,r in refs.iterrows():
            for metric,distmat in (("FFT_NN",xd[key]),("F40_ROBUST",hd[key])):
                i=int(np.argmin(distmat[:,j]));geo=G2MOD.geometry_compare(rcoords[r.instance_id],p["xy"][i]);grows.append({"record_type":"reference_nearest","group":group,"seed":seed,"stage":stage,"reference_id":r.instance_id,"n_nodes":r.n_nodes,"matched_75_150":bool(75<=r.n_nodes<=150),"focus11":r.instance_id in FOCUS11,"historical15":r.instance_id in HISTORICAL15,"retrieval_metric":metric,"candidate_hash":p["hashes"][i],"fft_distance":float(xd[key][i,j]),"fft_over_radius":float(xd[key][i,j]/radius),"f40_distance":float(hd[key][i,j]),**geo})
    geometry=pd.DataFrame(grows);geometry.to_csv(OUT/"05_f40_geometry_validation.csv",index=False)
    # Attach continuous multidimensional evidence to each arrival row.  These
    # are relative diagnostics, not a replacement coverage threshold.
    nearest=geometry[geometry.record_type.eq("reference_nearest")]
    evidence=[]
    for r in arrivals.itertuples(index=False):
        q=nearest[(nearest.group==r.group)&(nearest.seed==r.seed)&(nearest.stage=="G1_RAW")&(nearest.reference_id==r.reference_id)]
        a=nearest[(nearest.group=="A")&(nearest.seed==r.seed)&(nearest.stage=="G1_RAW")&(nearest.reference_id==r.reference_id)]
        qf=q[q.retrieval_metric.eq("F40_ROBUST")].iloc[0];af=a[a.retrieval_metric.eq("F40_ROBUST")].iloc[0]
        qgeo=float(q.geometry_composite.min());ageo=float(a.geometry_composite.min())
        evidence.append({"best_f40_distance":float(qf.f40_distance),"A_same_seed_best_f40_distance":float(af.f40_distance),"f40_better_than_A":bool(qf.f40_distance<af.f40_distance-1e-12),"best_geometry_composite":qgeo,"A_same_seed_best_geometry_composite":ageo,"geometry_better_than_A":bool(qgeo<ageo-1e-12)})
    arrivals=pd.concat([arrivals.reset_index(drop=True),pd.DataFrame(evidence)],axis=1)
    arrivals.to_csv(OUT/"04_new_neighborhood_arrivals.csv",index=False)
    make_figures(pools,xd,hd,refs,arrivals,rcoords,geometry,radius)
    reports(parents,runinfo,parent_audit,coverage,arrivals,geometry,refs)


def make_figures(pools: dict[str,Any], xd: dict[str,np.ndarray], hd: dict[str,np.ndarray], refs: pd.DataFrame, arrivals: pd.DataFrame, rcoords: dict[str,np.ndarray], geometry: pd.DataFrame, radius: float) -> None:
    # FFT-NN feature plane.
    fig,axes=plt.subplots(1,4,figsize=(18,4.5));keys=[("A_PARENT","A parent"),("B_PARENT","B parent"),("A_Original_G1","A G1 Original"),("B_Original_G1","B G1 Original")]
    for ax,(key,title) in zip(axes,keys):q=pools[key]["fft_frame"];ax.hexbin(q.fft_energy,q.nn_ratio,gridsize=45,mincnt=1,bins="log");ax.scatter(refs.fft_energy,refs.nn_ratio,s=12,c="red");ax.set_title(title);ax.set_xlabel("FFT_APPROX");ax.set_ylabel("NN-CV")
    fig.tight_layout();fig.savefig(FIG/"01_parent_g1_fft_nn_distribution.png",dpi=180);plt.close(fig)
    # Focus11 nearest distance ratios across seeds.
    focus=refs.instance_id.isin(FOCUS11);fig,axes=plt.subplots(1,3,figsize=(17,5),sharey=True);x=np.arange(len(FOCUS11));
    for ax,rep in zip(axes,REPS):
        a=xd[f"A_{rep}_G1"][:,focus].min(0)/radius;b=xd[f"B_{rep}_G1"][:,focus].min(0)/radius;ax.plot(x,a,"o-",label="A");ax.plot(x,b,"s-",label="B");ax.axhline(1,c="k",ls="--");ax.set_xticks(x,FOCUS11,rotation=60);ax.set_title(rep);ax.legend()
    axes[0].set_ylabel("nearest FFT–NN distance / frozen radius");fig.tight_layout();fig.savefig(FIG/"02_focus11_nearest_distance.png",dpi=180);plt.close(fig)
    # Full/F40 changes.
    q=geometry[geometry.record_type.eq("stage_summary")];fig,axes=plt.subplots(1,2,figsize=(11,4));
    for group,marker in (("A","o"),("B","s")):
        z=q[(q.group==group)&(q.stage=="G1_RAW")];axes[0].plot(z.seed,z.f40_best_reference_mean,marker=marker,label=group);axes[1].plot(z.seed,z.evoreal_f40_marginal_w1_mean,marker=marker,label=group)
    axes[0].set_ylabel("mean best robust-F40 distance to TSPLIB48");axes[1].set_ylabel("mean marginal F40 W1 to EvoReal50k");
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/"03_f40_structure_change.png",dpi=180);plt.close(fig)
    # New/lost coverage by seed.
    rows=[]
    for rep in REPS:
        a=arrivals[(arrivals.seed==rep)&(arrivals.group=="A")].set_index("reference_id");b=arrivals[(arrivals.seed==rep)&(arrivals.group=="B")].set_index("reference_id");rows.append((rep,int((~a.g1_raw_covered&b.g1_raw_covered).sum()),int((a.g1_raw_covered&~b.g1_raw_covered).sum())))
    x=np.arange(3);fig,ax=plt.subplots(figsize=(7,4));ax.bar(x-.18,[r[1] for r in rows],.36,label="B-only reached");ax.bar(x+.18,[r[2] for r in rows],.36,label="A-only reached");ax.set_xticks(x,[r[0] for r in rows]);ax.set_ylabel("reference neighborhoods");ax.legend();fig.tight_layout();fig.savefig(FIG/"04_seed_new_and_lost_coverage.png",dpi=180);plt.close(fig)
    # Geometry gallery for B-only focus arrivals, if any.
    chosen=arrivals[(arrivals.group=="B")&arrivals.focus11&arrivals.g1_raw_covered].sort_values(["B_true_new_beyond_B_initial","nearest_over_radius"],ascending=[False,True]).drop_duplicates("reference_id").head(8)
    if len(chosen):
        fig,axes=plt.subplots(len(chosen),3,figsize=(9,3*len(chosen)),squeeze=False)
        for rowi,r in enumerate(chosen.itertuples(index=False)):
            key=f"B_{r.seed}_G1";j=int(refs.index[refs.instance_id.eq(r.reference_id)][0]);i=int(np.argmin(xd[key][:,j]));candidate=pools[key]["xy"][i];g=G2MOD.geometry_compare(rcoords[r.reference_id],candidate);z=G2MOD.aligned_for_plot(candidate,g["best_rotation_deg"],g["best_reflection"]);axes[rowi,0].scatter(*G2MOD.shape_normalize(rcoords[r.reference_id]).T,s=6);axes[rowi,1].scatter(*z.T,s=6);parent_hash=pd.read_parquet(ART/r.seed/"B_G1_RAW/mutation_manifest.parquet").set_index("child_hash").loc[pools[key]["hashes"][i],"parent_hash"];bxy,bm=load_bundle(ART/"B_augmented_parents.npz");pi=bm.index(parent_hash);axes[rowi,2].scatter(*G2MOD.shape_normalize(bxy[pi]).T,s=6);axes[rowi,0].set_title(f"{r.reference_id} TSPLIB");axes[rowi,1].set_title(f"B offspring {r.seed}\nFFT/r={r.nearest_over_radius:.3f}, geo={g['geometry_composite']:.3f}");axes[rowi,2].set_title("direct parent")
            for ax in axes[rowi]:ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
        fig.tight_layout();fig.savefig(FIG/"05_new_arrival_geometry.png",dpi=160);plt.close(fig)


def reports(parents:dict[str,tuple[np.ndarray,pd.DataFrame]],runinfo:dict[str,Any],parent_audit:pd.DataFrame,coverage:pd.DataFrame,arrivals:pd.DataFrame,geometry:pd.DataFrame,refs:pd.DataFrame)->None:
    full=coverage[(coverage.record_type=="coverage")&(coverage.evaluation_mode=="full")];pcover=parent_audit[(parent_audit.record_type=="coverage")].set_index("group");summary=full.pivot(index="seed",columns="group",values=["coverage_48","coverage_17","coverage_focus11","coverage_historical15"])
    deltas=[]
    for rep in REPS:
        a=full[(full.seed==rep)&(full.group=="A")].iloc[0];b=full[(full.seed==rep)&(full.group=="B")].iloc[0];aa=arrivals[(arrivals.seed==rep)&(arrivals.group=="A")].set_index("reference_id");bb=arrivals[(arrivals.seed==rep)&(arrivals.group=="B")].set_index("reference_id");deltas.append({"seed":rep,"A_parent_48":int(pcover.loc['A'].coverage_48),"B_parent_48":int(pcover.loc['B'].coverage_48),"A_G1_48":int(a.coverage_48),"B_G1_48":int(b.coverage_48),"delta_G1_48":int(b.coverage_48-a.coverage_48),"A_G1_17":int(a.coverage_17),"B_G1_17":int(b.coverage_17),"delta_G1_17":int(b.coverage_17-a.coverage_17),"B_only_reference_n":int((~aa.g1_raw_covered&bb.g1_raw_covered).sum()),"B_only_references":";".join(bb.index[~aa.g1_raw_covered&bb.g1_raw_covered]),"A_only_reference_n":int((aa.g1_raw_covered&~bb.g1_raw_covered).sum()),"A_only_references":";".join(bb.index[aa.g1_raw_covered&~bb.g1_raw_covered]),"B_true_new_beyond_B_initial_n":int(bb.B_true_new_beyond_B_initial.sum()),"B_true_new_beyond_B_initial_ids":";".join(bb.index[bb.B_true_new_beyond_B_initial]),"B_focus11_reached":int(bb.loc[list(FOCUS11)].g1_raw_covered.sum())})
    delta=pd.DataFrame(deltas)
    random=coverage[coverage.record_type.eq("random10k_summary")][["group","seed","coverage_48","coverage_48_p025","coverage_48_p975","coverage_17","coverage_focus11"]]
    budget_rows=[]
    for rep in REPS:
        a=json.loads(historical_paths(rep)[2].read_text());b=runinfo["runs"][rep];budget_rows.append({"seed":rep,"A_conceptual_attempts":ATTEMPTS,"B_conceptual_attempts":ATTEMPTS,"A_historical_wall_s":a.get("wall_seconds"),"B_incremental_wall_s":b["wall_seconds_incremental"],"B_reused_common_attempts":b["physically_reused_common_attempts"],"B_new_attempts":b["physically_new_supplement_attempts"],"A_raw_unique_nonidentity":runinfo["replays"][rep]["nonidentity_A"]["valid_nonidentity_unique_n"],"B_raw_unique_nonidentity":b["valid_nonidentity_unique_n"]})
    budget=pd.DataFrame(budget_rows);f40run=json.loads((LOG/"f40_extraction_runtime.json").read_text()) if (LOG/"f40_extraction_runtime.json").exists() else {}
    (OUT/"06_budget_and_reproducibility.md").write_text(f"""# P6-G3 Budget and Reproducibility

## Mutation budgets

{md(budget)}

Arm A is the exact historical P6-B G1 control, accepted only after deterministic replay of 44 attempts per seed plus full parent/order/profile/namespace/hash gates. Arm B reuses the exact paired 99,000 common-parent attempts and physically evaluates 11,000 supplement-parent attempts per seed. This reduces redundant computation but does not change either arm's conceptual 110,000-attempt budget. The B end-to-end intervention also includes supplement construction and cannot be called equal-cost end to end.

## Replay gates

```json
{json.dumps(runinfo['replays'],indent=2,sort_keys=True)}
```

- Supplement attempted candidates: {runinfo['parent']['supplement_attempted']}; accepted unique: 1,000.
- F40 new extraction: `{json.dumps(f40run,sort_keys=True)}`.
- Python `{platform.python_version()}`; NumPy `{np.__version__}`; pandas `{pd.__version__}`; source commit `{git_commit()}`.
- No TSPLIB22, Concorde, BQ/HGB/GP inference/training, survivor selection, or G2 operation was invoked.
""")
    parent_expanded=int(pcover.loc["B"].coverage_48)>int(pcover.loc["A"].coverage_48)
    true_new=int(delta.B_true_new_beyond_B_initial_n.sum());b_better=(delta.delta_G1_48>0).sum();b_loss=int(delta.A_only_reference_n.sum());
    # Primary validity is evaluated on genuine B-G1 expansion beyond B's own
    # initial coverage.  A broader historical-missing diagnostic is retained.
    true_events=arrivals[(arrivals.group=="B")&arrivals.B_true_new_beyond_B_initial]
    broad_events=arrivals[(arrivals.group=="B")&arrivals.g1_raw_covered&arrivals.historical15]
    true_f40=int(true_events.f40_better_than_A.sum());true_geo=int(true_events.geometry_better_than_A.sum());true_n=len(true_events)
    broad_any=(broad_events.f40_better_than_A|broad_events.geometry_better_than_A)
    broad_fraction=float(broad_any.mean()) if len(broad_any) else float("nan")
    structure_valid="SUPPORTED" if true_n and true_geo==true_n and true_f40>=math.ceil(true_n/2) else "PARTIAL" if true_n and (true_geo+true_f40)>0 else "NOT_SUPPORTED"
    reach="SUPPORTED" if parent_expanded and b_better==3 and true_new>0 else "PARTIAL" if parent_expanded and b_better>=1 else "NOT_SUPPORTED"
    seed_effect="STRONG" if b_better==3 else "LIMITED" if b_better>=1 or parent_expanded else "NONE"
    mutation_beyond="YES" if true_new>0 else "NO"
    retention="PASS" if b_loss==0 else "PARTIAL" if b_loss<=3 else "FAIL"
    causal="SUPPORTED_FOR_THIS_INTERVENTION" if reach=="SUPPORTED" and structure_valid!="NOT_SUPPORTED" else "MIXED" if parent_expanded or b_better else "INCONCLUSIVE"
    labels={"P6_G3_PARENT_SUPPORT_EXPANDED":"YES" if parent_expanded else "NO","P6_G3_G1_NEW_REACHABILITY":reach,"P6_G3_STRUCTURE_VALIDITY":structure_valid,"P6_G3_INITIAL_SEED_EFFECT":seed_effect,"P6_G3_MUTATION_EXPANSION_BEYOND_SEEDS":mutation_beyond,"P6_G3_ORIGINAL_SUPPORT_RETENTION":retention,"P6_G3_CAUSAL_CONCLUSION":causal,"P6_G3_ANALYSIS_STATUS":"COMPLETE"}
    final="# P6-G3 Final Conclusion\n\n## 判定标签\n\n"+"\n".join(f"`{k} = {v}`" for k,v in labels.items())+f"""

## A/B 主结果

{md(delta)}

固定10k抽样结果：

{md(random)}

## 五个问题

1. **补充父代是否增加G1可达区域？** Arm B初始覆盖 `{int(pcover.loc['B'].coverage_48)}/48`，Arm A为 `{int(pcover.loc['A'].coverage_48)}/48`。三个seed的G1 A/B及差值见上表；结论为 **{reach}**，仅适用于冻结的10%干预。
2. **直接注入还是FOGA继续扩展？** B相对自身初始父代、由非恒等G1 offspring新进入的reference在三个seed合计 `{true_new}` 个seed-reference事件。直接注入与变异维持/扩展已在 `04_new_neighborhood_arrivals.csv` 分开记录，不能把B初始增益算作FOGA新增。
3. **F40与几何是否支持？** 对4个真正超出B初始覆盖的seed-reference事件，F40最近距离改善 `{true_f40}/{true_n}`，独立几何最近诊断改善 `{true_geo}/{true_n}`，判定 **{structure_valid}**。作为更宽的辅助统计，B在历史缺失区域的全部到达事件中，F40或几何至少一项优于同seed A的比例为 `{broad_fraction:.3f}`。连续值已同时写入 `04_new_neighborhood_arrivals.csv` 与 `05_f40_geometry_validation.csv`，没有创建新覆盖阈值。
4. **是否损失原P03结构？** 三seed共出现 `{b_loss}` 个A覆盖而B未覆盖的reference事件，判定 **{retention}**。这同时报告了随机移除1000个P03父代的代价。
5. **是否值得作为完整第五轮候选初始化？** `{('值得作为候选进入预注册的完整演化验证，但尚不能视为最优初始化。' if causal=='SUPPORTED_FOR_THIS_INTERVENTION' else '证据混合；可保留为候选，但在进入完整第五轮前应先审查逐reference来源与损失。' if causal=='MIXED' else '当前干预证据不足，不建议直接推进完整第五轮。')}` 本实验不证明10%最优、演化深度最优或BQ训练增益。

## 证据边界

三seed是小样本探索性重复。FFT覆盖、F40距离和独立几何是结构诊断，不是Original70 Gap或训练价值证据。A的历史缓存复用经过确定性回放，但物理计算成本不等同于全量重跑；共同父代的数学结果完全配对。

`BLADE_P6G3_INITIAL_SUPPORT_INTERVENTION_COMPLETE`
"""
    (OUT/"07_final_conclusion.md").write_text(final)
    write_json(OUT/"completion.json",{"status":"COMPLETE","labels":labels,"key_metrics":{"A_parent_coverage48":int(pcover.loc['A'].coverage_48),"B_parent_coverage48":int(pcover.loc['B'].coverage_48),"per_seed":deltas,"true_new_beyond_B_initial_seed_reference_events":true_new,"true_new_f40_improved_n":true_f40,"true_new_geometry_improved_n":true_geo,"A_only_loss_seed_reference_events":b_loss,"broad_historical_arrival_any_multidimensional_improvement_fraction":broad_fraction},"final_tag":"BLADE_P6G3_INITIAL_SUPPORT_INTERVENTION_COMPLETE","created_utc":pd.Timestamp.utcnow().isoformat(),"wall_seconds":time.time()-STARTED})


def verify_outputs() -> None:
    required=["01_intervention_protocol.md","02_parent_population_audit.csv","03_g1_raw_coverage.csv","04_new_neighborhood_arrivals.csv","05_f40_geometry_validation.csv","06_budget_and_reproducibility.md","07_final_conclusion.md","completion.json"]
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing:raise RuntimeError(f"missing outputs {missing}")
    rows=[]
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name!="output_hashes.csv":rows.append({"path":str(path.relative_to(OUT)),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    pd.DataFrame(rows).to_csv(LOG/"output_hashes.csv",index=False)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("protocol","mutation","analysis","full"),nargs="?",default="full");args=parser.parse_args();protocol_gate()
    if args.stage=="protocol":return
    parents,info=run_mutation()
    if args.stage=="mutation":return
    analyze(parents,info);verify_outputs();print(f"[P6-G3] COMPLETE wall={time.time()-STARTED:.1f}s",flush=True)


if __name__=="__main__":main()
