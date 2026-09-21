"""Scientific primitives used by the frozen autonomous runner."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import yaml

from audit.gp_modeling import safe_scaled_prediction, train_seed, tree_depth, tree_from_dict, tree_nodes
from audit.uniform_bootstrap import child_seed as audited_c00_seed
from bq.checkpoint import load_canonical_checkpoint
from bq.inference import greedy_tour_costs
from bq.training_hardness import teacher_forced_training_hardness
from evaluation.datasets import load_fixed_n100, load_lehd_cross_size, load_original_tsplib70
from evaluation.gap import optimality_gap_percent
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact
from mutation import OPERATOR_NAMES, apply_operator
from oracle.concorde import ORACLE_LEDGER_COLUMNS, canonical_oracle_ledger, resolve_backend, solve_batch
from qd.acquisition import centroid_global_acquire
from qd.archive import CurrentSurrogateScore
from qd.coordinates import assign_nearest, load_centroids, normalize_qd4
from qd.roots import coordinate_hash


ROOT = Path(__file__).resolve().parents[2]
GP_CFG = {
    "population_size": 2000, "generations": 60, "initial_depth_min": 2,
    "initial_depth_max": 4, "maximum_depth": 6, "maximum_nodes": 25,
    "maximum_unique_features": 10, "maximum_constants": 25,
    "tournament_size": 7, "elitism_count": 20, "crossover_probability": .65,
    "subtree_mutation_probability": .15, "point_mutation_probability": .10,
    "hoist_mutation_probability": .05, "reproduction_probability": .05,
    "parsimony_coefficient": 5e-4, "hall_of_fame_size": 100,
    "prediction_invalid_threshold": 1e6, "fitness_prediction_clip": 20.,
    "maximum_validation_clip_fraction": .01, "huber_delta": 1.,
}
HGB_CFG = dict(learning_rate=.05, max_iter=500, max_leaf_nodes=31, max_depth=None,
               min_samples_leaf=20, l2_regularization=1e-4, early_stopping=False,
               random_state=2026)


def _duckdb():
    try:
        import duckdb
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(ROOT / "vendor"))
        import duckdb
    return duckdb


def membership_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def cycle_child_seed(cycle: int, parent_hash: str, operator_id: str) -> int:
    if cycle == 0:
        return audited_c00_seed(parent_hash, operator_id)
    material = f"BLADE_FINAL_20CYCLE_V1|2026|C{cycle:02d}|{parent_hash}|{operator_id}|FOGA11".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def generate_offspring(parents: np.ndarray, parent_hashes: list[str], cycle: int) -> tuple[np.ndarray, pd.DataFrame]:
    if len(parents) != 10_000 or len(parent_hashes) != 10_000 or len(set(parent_hashes)) != 10_000:
        raise ValueError("formal mutation requires exactly 10,000 unique parents")
    coordinates: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parent_index, (parent, parent_hash) in enumerate(zip(parents, parent_hashes, strict=True)):
        for operator_id in OPERATOR_NAMES:
            seed = cycle_child_seed(cycle, parent_hash, operator_id)
            result = apply_operator(operator_id, parent, np.random.default_rng(seed))
            accepted = result.proposal_status == "accepted" and result.valid_after_repair
            dedup_started=time.perf_counter(); duplicate = accepted and result.final_output_hash in seen
            rows.append({
                "parent_index": parent_index, "parent_hash": parent_hash,
                "operator_id": operator_id, "child_seed": np.uint64(seed),
                "child_hash": result.final_output_hash, "raw_child_hash": result.raw_output_hash,
                "valid_before_repair": bool(result.valid_before_repair),
                "valid_after_repair": bool(result.valid_after_repair),
                "repair_count": int(result.repair_count), "proposal_status": result.proposal_status,
                "failure_reason": result.failure_reason, "duplicate": bool(duplicate), "cycle": cycle,
            })
            if accepted and not duplicate:
                seen.add(result.final_output_hash)
                coordinates.append(np.asarray(result.final_coordinates, dtype=np.float64))
            dedup_s=time.perf_counter()-dedup_started
            rows[-1].update({key:float(value) for key,value in result.timing.items()})
            rows[-1]["deduplication_s"]=dedup_s
        if (parent_index + 1) % 1000 == 0:
            print(f"[C{cycle:02d}] offspring parents {parent_index + 1}/10000", flush=True)
    return np.stack(coordinates), pd.DataFrame(rows)


def qd_frame(coordinates: np.ndarray, hashes: list[str]) -> pd.DataFrame:
    raw = extract_qd4_exact(coordinates)
    normalized = normalize_qd4(raw)
    niches = assign_nearest(normalized)
    frame: dict[str, Any] = {"instance_hash": hashes, "niche_id": niches}
    for index, name in enumerate(QD4_FEATURE_NAMES):
        frame[name] = raw[:, index]
        frame[f"normalized_{name}"] = normalized[:, index]
    return pd.DataFrame(frame)


def exact_sparse_archive(meta: pd.DataFrame, hardness: pd.Series, version: str) -> pd.DataFrame:
    data = meta.copy()
    data["exact_hardness"] = data.instance_hash.map(hardness)
    if data.exact_hardness.isna().any() or not np.isfinite(data.exact_hardness).all():
        raise RuntimeError("exact archive has missing/non-finite current hardness")
    data = data.sort_values(["niche_id", "exact_hardness", "instance_hash"], ascending=[True, False, True], kind="stable")
    result = data.drop_duplicates("niche_id", keep="first").sort_values("niche_id").reset_index(drop=True)
    if len(result) > 1000 or result.niche_id.duplicated().any():
        raise RuntimeError("exact sparse archive invariant failure")
    result["hardness_version"] = version
    return result


def robust_transform(hardness: np.ndarray) -> dict[str, Any]:
    y = np.log1p(np.asarray(hardness, dtype=float))
    center = float(np.median(y))
    options = [("IQR", float(np.subtract(*np.percentile(y, [75, 25])))),
               ("1.4826_MAD", 1.4826 * float(np.median(np.abs(y - center)))),
               ("STD", float(np.std(y, ddof=1))), ("ONE", 1.)]
    method, scale = next((name, value) for name, value in options if np.isfinite(value) and value > 1e-12)
    return {"formula": "log1p_LBQ", "center": center, "scale": scale, "scale_method": method}


def _formula_prediction(record: dict[str, Any], x: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    scaled, clip_fraction, valid = safe_scaled_prediction(tree_from_dict(record["tree"]), x, GP_CFG)
    if not valid or clip_fraction > GP_CFG["maximum_validation_clip_fraction"]:
        raise RuntimeError("SPERF formula produced invalid predictions")
    return scaled * transform["scale"] + transform["center"]


def _select_sperf(hall: list[dict[str, Any]], x: np.ndarray, y: np.ndarray, transform: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    for record in hall:
        try:
            prediction = _formula_prediction(record, x, transform)
            rho = float(spearmanr(y, prediction).statistic) if np.ptp(prediction) > 1e-15 else 0.0
            mae = float(np.mean(np.abs(y - prediction)))
            eligible = bool(np.isfinite(prediction).all())
        except Exception:
            rho, mae, eligible = -math.inf, math.inf, False
        candidates.append({**record, "validation_spearman": rho, "validation_mae": mae, "validation_eligible": eligible})
    usable = [item for item in candidates if item["validation_eligible"]]
    if not usable:
        raise RuntimeError("no SPERF-eligible formula")
    best = max(item["validation_spearman"] for item in usable)
    finalists = [item for item in usable if item["validation_spearman"] >= best - 1e-6]
    return min(finalists, key=lambda item: (item["nodes"], item["validation_mae"], item["canonical_formula"])), candidates


def train_gp4(features: np.ndarray, hardness: np.ndarray, output: Path, cycle: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    transform = robust_transform(hardness)
    y = np.log1p(np.asarray(hardness, float))
    scaled = (y - transform["center"]) / transform["scale"]
    formulas, runtimes = [], []
    for position, seed in enumerate(range(2026, 2031), 1):
        print(f"[C{cycle:02d}][GP{cycle}] GP train {position}/5", flush=True)
        started = time.perf_counter()
        log, hall = train_seed(features, scaled, list(QD4_FEATURE_NAMES), GP_CFG, seed)
        selected, candidates = _select_sperf(hall, features, y, transform)
        runtime = time.perf_counter() - started
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        log.to_parquet(seed_dir / "training_log.parquet", index=False)
        (seed_dir / "hof_candidates.json").write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (seed_dir / "formula.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        formulas.append(selected); runtimes.append(runtime)
    model = {"cycle": cycle, "seeds": list(range(2026, 2031)), "transform": transform,
             "formulas": formulas, "aggregate": "median", "seed_runtime_s": runtimes,
             "train_n": len(features), "features": list(QD4_FEATURE_NAMES), "selection": "SPERF"}
    path = output / "gp4_ensemble.joblib"
    joblib.dump(model, path, compress=3)
    return model


def predict_gp4(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    predictions = [_formula_prediction(record, features, model["transform"]) for record in model["formulas"]]
    result = np.median(np.vstack(predictions), axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("GP4 ensemble produced non-finite scores")
    return result


def extract_gp20(coordinates: np.ndarray, hashes: list[str], output: Path) -> pd.DataFrame:
    """Extract the exact frozen salesperson GP20 control features."""
    spec = yaml.safe_load((ROOT / "configs/hgb20_features.yaml").read_text(encoding="utf-8"))
    names = list(spec["features"])
    scaler = json.loads((ROOT / "assets/qd/reference_cloud/formal_raw_scaler.json").read_text(encoding="utf-8"))
    parameters = {item["feature_name"]: (float(item["center_median"]), float(item["scale"])) for item in scaler["parameters"]}
    rscript = os.environ.get("BLADE_RSCRIPT") or shutil.which("Rscript") or "C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe"
    frames=[]
    for start in range(0, len(hashes), 250):
        stop=min(len(hashes),start+250)
        with tempfile.TemporaryDirectory(prefix="blade_final_gp20_") as temporary:
            work=Path(temporary); inp=work/"coordinates.csv"; raw=work/"features.csv"; schema=work/"schema.csv"
            with inp.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.writer(handle,lineterminator="\n"); writer.writerow(["state_id","node_idx","x","y"])
                for instance_hash,points in zip(hashes[start:stop],coordinates[start:stop],strict=True):
                    for node,(x,y) in enumerate(points): writer.writerow([instance_hash,node,repr(float(x)),repr(float(y))])
            process=subprocess.run([rscript,str(ROOT/"src/features/extract_salesperson_raw_features.R"),str(inp),str(raw),str(schema)],capture_output=True,text=True,encoding="utf-8",errors="replace")
            if process.returncode or not raw.is_file(): raise RuntimeError(f"GP20 extraction failed: {process.stderr[-1000:]}")
            data=pd.read_csv(raw)
            if data.state_id.astype(str).tolist()!=hashes[start:stop]: raise RuntimeError("GP20 feature alignment failure")
            frame=pd.DataFrame({"instance_hash":data.state_id.astype(str)})
            for name in names:
                center,scale=parameters[name]; frame[name]=np.arcsinh((data[name].to_numpy(float)-center)/scale)
            frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    result.to_parquet(output,index=False)
    return result


def train_hgb_controls(qd4: np.ndarray, gp20: np.ndarray, hardness: np.ndarray, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    y = np.log1p(np.asarray(hardness, float))
    started = time.perf_counter()
    hgb4 = HistGradientBoostingRegressor(**HGB_CFG).fit(qd4, y)
    hgb4_s = time.perf_counter() - started
    started = time.perf_counter()
    hgb20 = HistGradientBoostingRegressor(**HGB_CFG).fit(gp20, y)
    hgb20_s = time.perf_counter() - started
    started=time.perf_counter(); prediction4=hgb4.predict(qd4); hgb4_predict_s=time.perf_counter()-started
    started=time.perf_counter(); prediction20=hgb20.predict(gp20); hgb20_predict_s=time.perf_counter()-started
    joblib.dump({"model": hgb4, "features": list(QD4_FEATURE_NAMES)}, output / "hgb4.joblib", compress=3)
    names = yaml.safe_load((ROOT / "configs/hgb20_features.yaml").read_text(encoding="utf-8"))["features"]
    joblib.dump({"model": hgb20, "features": names}, output / "hgb20.joblib", compress=3)
    return {"HGB4_train_s": hgb4_s, "HGB4_predict_s":hgb4_predict_s,"HGB4_training_spearman":float(spearmanr(y,prediction4).statistic),
            "HGB20_train_s": hgb20_s,"HGB20_predict_s":hgb20_predict_s,"HGB20_training_spearman":float(spearmanr(y,prediction20).statistic),"acquisition_driver": False}


def hardness(model: torch.nn.Module, coordinates: np.ndarray, tours: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    values = teacher_forced_training_hardness(model, coordinates, tours, device, batch_size=batch_size)["loss_raw"]
    if len(values) != len(coordinates) or not np.isfinite(values).all():
        raise RuntimeError("incomplete/non-finite exact hardness")
    return values


def parse_tours(values: list[str], n: int = 100) -> np.ndarray:
    result = np.asarray([[int(token) for token in str(value).split()] for value in values], dtype=np.int64)
    if result.shape != (len(values), n):
        raise RuntimeError(f"expert tour shape mismatch: {result.shape}")
    return result


def load_anchor() -> tuple[np.ndarray, np.ndarray]:
    with np.load(ROOT / "assets/retention/uniform10k/fixed_uniform_anchor_10k_coordinates.npz", allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    path = (ROOT / "assets/retention/uniform10k/fixed_uniform_anchor_10k_labels.parquet").as_posix().replace("'", "''")
    values = [row[0] for row in _duckdb().sql(f"SELECT expert_tour FROM read_parquet('{path}')").fetchall()]
    tours = parse_tours(values)
    if coordinates.shape != (10_000, 100, 2):
        raise RuntimeError("anchor coordinates mismatch")
    return coordinates, tours


def label_with_cache(coordinates: np.ndarray, hashes: list[str], output: Path, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    backend = resolve_backend(ROOT / "third_party/concorde/concorde")
    known = existing if existing is not None else pd.DataFrame()
    ledger = solve_batch(coordinates, hashes, output / "work", backend, workers=8, existing=known)
    if len(ledger) != len(hashes) or not ledger.verified.all():
        raise RuntimeError("Oracle labeling incomplete")
    return ledger


def attach_oracle_metadata(selected: pd.DataFrame, current_hardness: np.ndarray,
                           ledger: pd.DataFrame) -> pd.DataFrame:
    """Attach one canonical Oracle payload without propagating prior merge suffixes."""
    payload=set(ORACLE_LEDGER_COLUMNS[1:])
    leaked={"oracle_role","FORMAL_SELECTION_OCCURRED"}
    keep=[column for column in selected.columns
          if column not in payload|leaked and not column.endswith("_oracle")]
    clean=selected.loc[:,keep].copy()
    clean["current_hardness"]=np.asarray(current_hardness,dtype=float)
    canonical=canonical_oracle_ledger(ledger)
    labeled=clean.merge(canonical,on="instance_hash",how="left",validate="one_to_one")
    labeled["FORMAL_SELECTION_OCCURRED"]="YES"
    if labeled.columns.duplicated().any(): raise RuntimeError("duplicate columns after canonical Oracle merge")
    return labeled


def select_population(pool: pd.DataFrame, scores: np.ndarray, budget: int, version: str) -> pd.DataFrame:
    hashes = pool.instance_hash.astype(str).tolist()
    score_map = {h: CurrentSurrogateScore(float(v), version) for h, v in zip(hashes, scores, strict=True)}
    niche_map = dict(zip(hashes, pool.niche_id.astype(int), strict=True))
    norm_cols = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
    norm_map = {row.instance_hash: tuple(float(getattr(row, name)) for name in norm_cols) for row in pool.itertuples(index=False)}
    selected = centroid_global_acquire(hashes, score_map, niche_map, norm_map, load_centroids(), budget, current_score_version=version)
    lookup = pool.set_index("instance_hash", drop=False)
    frame = lookup.loc[[item.instance_hash for item in selected]].reset_index(drop=True).copy()
    frame["component"] = [item.component for item in selected]
    frame["centroid_distance"] = [item.centroid_distance for item in selected]
    frame["gp_predicted_log1p_hardness"] = [item.surrogate_score for item in selected]
    return frame


def summary_stats(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    result: dict[str, float | int] = {"n": len(array), "mean": float(np.mean(array)), "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)), "max": float(np.max(array))}
    for q in (75, 90, 95, 99): result[f"P{q}"] = float(np.percentile(array, q))
    return result


def bootstrap_mean_delta(new: np.ndarray, old: np.ndarray, seed: int, replicates: int = 10_000, paired: bool = False) -> dict[str, float]:
    a, b = np.asarray(new, float), np.asarray(old, float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 500):
        stop = min(replicates, start + 500)
        if paired:
            if len(a) != len(b): raise ValueError("paired bootstrap sizes differ")
            indices = rng.integers(0, len(a), size=(stop-start, len(a)))
            estimates[start:stop] = np.mean(a[indices] - b[indices], axis=1)
        else:
            ia = rng.integers(0, len(a), size=(stop-start, len(a)))
            ib = rng.integers(0, len(b), size=(stop-start, len(b)))
            estimates[start:stop] = np.mean(a[ia], axis=1) - np.mean(b[ib], axis=1)
    return {"estimate": float(np.mean(a)-np.mean(b)), "ci95_low": float(np.percentile(estimates, 2.5)), "ci95_high": float(np.percentile(estimates, 97.5)), "replicates": replicates, "seed": seed}


def evaluation_references(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if name in {"Uniform100", "GaussianMixture100", "TSPLIB100"}:
        coordinates = load_fixed_n100(name)
        directory = {"Uniform100":"uniform100", "GaussianMixture100":"gaussian_mixture100", "TSPLIB100":"tsplib100"}[name]
        file = {"TSPLIB100":"concorde/concorde_results.parquet"}.get(name, "concorde/oracle_results.parquet")
        path = (ROOT / "assets/evaluation" / directory / file).as_posix().replace("'", "''")
        rows = _duckdb().sql(f"SELECT instance_id, optimal_cost_int FROM read_parquet('{path}') ORDER BY instance_id").fetchall()
        identifiers = [str(row[0]) for row in rows]
        optimum = np.asarray([row[1] for row in rows], dtype=np.int64)
        return coordinates, optimum, identifiers
    if name == "OriginalTSPLIB70":
        records = list(load_original_tsplib70())
        # Variable-size evaluation is performed one instance at a time.
        return np.asarray([item[1] for item in records], dtype=object), np.asarray([item[2] for item in records]), [item[0] for item in records]
    size = int(name.rsplit("TSP", 1)[1])
    coordinates, expert = load_lehd_cross_size(size)
    optimum = np.empty(len(coordinates), dtype=np.int64)
    from bq.tsp_metric import integer_tour_cost
    for index in range(len(coordinates)):
        tour = expert[index].tolist()
        if len(tour) == size + 1 and tour[0] == tour[-1]: tour = tour[:-1]
        if min(tour) == 1: tour = [value - 1 for value in tour]
        optimum[index] = integer_tour_cost(coordinates[index], tour, 1_000_000)
    return coordinates, optimum, [f"{name}_{index:04d}" for index in range(len(coordinates))]


def evaluate_checkpoint(model: torch.nn.Module, device: torch.device, output: Path, model_name: str, datasets: list[str]) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, name in enumerate(datasets, 1):
        print(f"[{model_name}] external eval {position}/7", flush=True)
        started = time.perf_counter(); coordinates, optimum, identifiers = evaluation_references(name); load_s = time.perf_counter()-started
        inference_started = time.perf_counter(); predicted_parts=[]; valid=0
        if coordinates.dtype == object:
            for points in coordinates:
                original=np.asarray(points,dtype=np.float64); minima=original.min(axis=0); span=original.max(axis=0)-minima
                model_input=(original-minima)/np.where(span>0,span,1.)
                _, tours = greedy_tour_costs(model, model_input[None, ...], device)
                tour=tours[0]; cost=sum(int(np.rint(np.linalg.norm(original[tour[i]]-original[tour[(i+1)%len(tour)]]))) for i in range(len(tour)))
                predicted_parts.append(np.asarray([cost],dtype=np.int64)); valid += 1
        else:
            batch_size = 128 if coordinates.shape[1] <= 100 else (16 if coordinates.shape[1] <= 500 else 4)
            for start in range(0, len(coordinates), batch_size):
                costs, _ = greedy_tour_costs(model, coordinates[start:start+batch_size], device); predicted_parts.append(costs); valid += len(costs)
        inference_s = time.perf_counter()-inference_started
        metric_started=time.perf_counter(); predicted=np.concatenate(predicted_parts); gaps=optimality_gap_percent(predicted,optimum); metric_s=time.perf_counter()-metric_started
        detail=pd.DataFrame({"instance_id":identifiers,"predicted_cost_int":predicted,"optimal_cost_int":optimum,"gap_percent":gaps})
        detail_path=output/f"{name}.parquet"; write_started=time.perf_counter(); detail.to_parquet(detail_path,index=False); write_s=time.perf_counter()-write_started
        stats=summary_stats(gaps)
        rows.append({"model":model_name,"dataset":name,"instance_count":len(gaps),"valid_solutions":valid,
                     "mean_gap_percent":stats["mean"],"median_gap_percent":stats["median"],"P75":stats["P75"],"P90":stats["P90"],"P95":stats["P95"],"max_gap_percent":stats["max"],
                     "dataset_load_s":load_s,"model_inference_s":inference_s,"metric_computation_s":metric_s,"result_write_s":write_s,
                     "total_benchmark_s":load_s+inference_s+metric_s+write_s,"instances_per_s":len(gaps)/max(inference_s,1e-12),"mean_inference_s_per_instance":inference_s/len(gaps)})
    frame=pd.DataFrame(rows); frame.to_csv(output/"external_metrics.csv",index=False); return frame
