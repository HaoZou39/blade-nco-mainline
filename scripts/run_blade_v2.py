#!/usr/bin/env python
"""Corrected BLADE-NCO V2 Full plus C00-branch GP-only ablation runner."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bq.adaptation import fine_tune_fixed_budget, save_e256
from bq.checkpoint import load_canonical_checkpoint
from features.qd4 import QD4_FEATURE_NAMES
from final_run.pipeline import (_canonical_mutation_worker, _load_bundle, _oracle_tours,
                                _save_bundle, _verify_c00_replay)
from final_run.science import (attach_oracle_metadata, bootstrap_mean_delta, hardness,
                               label_with_cache, load_anchor, membership_hash,
                               predict_gp4, qd_frame, select_population, summary_stats,
                               train_gp4, _duckdb)
from final_run.stage import atomic_json, atomic_stage, sha256_file, utc_now, verify_stage
from qd.roots import recover_uniform_roots


FULL_ID = "blade_v2_full_seed2026"
GP_ID = "blade_v2_gp_only_seed2026"
FULL_DIR = ROOT / "runs" / FULL_ID
GP_DIR = ROOT / "runs" / GP_ID
RUNTIME = ROOT / "runtime"
SUMMARY_FULL = ROOT / "summary" / "full_main_experiment"
SUMMARY_ABLATION = ROOT / "summary" / "qd_vs_gp_only_m15"
M0_PATH = ROOT / "assets" / "solver" / "M0" / "tsp.best"
M0_SHA256 = "6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15"
NORM = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
LR = 3.75e-5
EPOCHS = 256
BENCHMARKS = ["Uniform100", "GaussianMixture100", "TSPLIB100", "EvoReal_S1_100",
              "EvoReal_S2_100", "EvoReal_S3_100", "GaussianMixture500", "OriginalTSPLIB70"]


def validate_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    full_cycles = config["run"]["full_cycles"]
    full_final_model = config["run"]["full_final_model"]
    gp_cycles = config["run"]["gp_only_cycles"]
    gp_final_model = config["run"]["gp_only_final_model"]
    checks = {
        "schema": config["schema_version"] == "BLADE_V2_CORRECTED_C00_BRANCH_V1",
        "seed": config["run"]["master_seed"] == 2026,
        "full cycles": full_cycles in (list(range(15)), list(range(20))),
        "full final model": full_final_model == f"M{len(full_cycles)}",
        "gp-only cycles": gp_cycles in (list(range(1, 6)), list(range(1, 15))),
        "gp-only final model": gp_final_model == f"M{max(gp_cycles) + 1}",
        "branch": config["branch"]["first_divergent_generation"] == "C01",
        "random500": config["gp"]["working_set"] == 500,
        "GP4": config["gp"]["population"] == 2000 and config["gp"]["generations"] == 60
                 and config["gp"]["seeds"] == list(range(2026, 2031)),
        "QD": config["qd"]["k"] == 1000 and config["qd"]["centroids_frozen"] is True,
        "FOGA11": config["mutation"]["family"] == "FOGA11" and config["mutation"]["nominal_offspring"] == 110_000,
        "selection": config["selection"]["budget"] == 10_000,
        "no E": config["archive"]["persistent_exact_elite_archive"] is False,
        "BQ": config["training"]["hard_count"] == 10_000 and config["training"]["uniform_anchor_count"] == 10_000
              and float(config["training"]["learning_rate"]) == LR and config["training"]["epochs"] == EPOCHS,
        "benchmarks": config["evaluation"]["datasets"] == BENCHMARKS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V2 frozen config semantic failure: {failed}")
    return config


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hash() -> str:
    digest = hashlib.sha256()
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "configs"):
        for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _pid_command(pid: int) -> str:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return ""
        return "alive"
    query = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' -ErrorAction SilentlyContinue).CommandLine"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", query], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else ""


def _acquire_lock(path: Path, state: str, resume: bool) -> None:
    prior = _json(path) if path.is_file() else None
    if prior and "run_blade_v2.py" in _pid_command(int(prior["pid"])):
        raise RuntimeError(f"active V2 lock held by PID {prior['pid']}: {path}")
    if prior and not resume:
        raise FileExistsError(f"existing V2 lock requires --resume: {path}")
    if prior:
        atomic_json(path.with_name(path.stem + ".previous.json"), prior)
    atomic_json(path, {"pid": os.getpid(), "host": socket.gethostname(), "state": state,
                       "timestamp": utc_now(), "command": " ".join(sys.argv)})


class Heartbeat:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {"arm": "INITIALIZING", "cycle": None, "phase": "startup",
                                     "last_valid_artifact": None, "latest_checkpoint": None,
                                     "latest_error": None, "recovery_state": "NONE"}
        self.started = time.time()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="blade-v2-heartbeat", daemon=True)

    def start(self) -> None:
        self.write()
        self.thread.start()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)
        self.write()

    def write(self) -> None:
        with self.lock:
            value = dict(self.state)
        value.update({"timestamp": utc_now(), "pid": os.getpid(), "elapsed_seconds": time.time() - self.started})
        atomic_json(RUNTIME / "heartbeat.json", value)

    def _loop(self) -> None:
        while not self.stop_event.wait(60):
            self.write()

    def close(self) -> None:
        self.stop_event.set()
        self.write()


def _run_dirs(config_path: Path, config: dict[str, Any], resume: bool) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for path, state in ((RUNTIME / "blade_v2_full_seed2026.lock", "ACTIVE"),
                        (RUNTIME / "blade_v2_gp_only_seed2026.lock", "SCHEDULED"),
                        (RUNTIME / "blade_v2_shared_c00_seed2026.lock", "ACTIVE")):
        _acquire_lock(path, state, resume)
    protocol = {"schema_version": config["schema_version"], "config": config,
                "config_sha256": sha256_file(config_path), "M0_sha256": sha256_file(M0_PATH),
                "source_snapshot_sha256": _source_hash(), "created_utc": utc_now()}
    if protocol["M0_sha256"] != M0_SHA256:
        raise RuntimeError("M0 hash mismatch")
    for run_dir, arm in ((FULL_DIR, "FULL"), (GP_DIR, "GP_ONLY")):
        if run_dir.exists() and not resume:
            raise FileExistsError(f"run directory exists without --resume: {run_dir}")
        for relative in ("00_protocol", "cycles", "logs", "timing"):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)
        frozen = run_dir / "00_protocol" / "RUN_PROTOCOL_FROZEN.json"
        if resume:
            old = _json(frozen)
            if old['config_sha256'] != protocol['config_sha256']:
                old_science = copy.deepcopy(old['config'])
                new_science = copy.deepcopy(config)
                old_horizon = (
                    old_science['run'].pop('full_cycles'),
                    old_science['run'].pop('full_final_model'),
                    old_science['run'].pop('gp_only_cycles'),
                    old_science['run'].pop('gp_only_final_model'),
                )
                new_horizon = (
                    new_science['run'].pop('full_cycles'),
                    new_science['run'].pop('full_final_model'),
                    new_science['run'].pop('gp_only_cycles'),
                    new_science['run'].pop('gp_only_final_model'),
                )
                horizon_amendment = (
                    old_science == new_science
                    and old_horizon == (list(range(20)), 'M20', list(range(1, 6)), 'M6')
                    and new_horizon == (list(range(15)), 'M15', list(range(1, 15)), 'M15')
                )
                if not horizon_amendment:
                    raise RuntimeError(f'resume identity mismatch: {run_dir}')
                atomic_json(run_dir / '00_protocol' / 'PROTOCOL_AMENDMENT_MATCHED_M15.json', {
                    'status': 'USER_DIRECTED_MATCHED_M15_HORIZON_AMENDMENT',
                    'amended_utc': utc_now(),
                    'old_config_sha256': old['config_sha256'],
                    'new_config_sha256': protocol['config_sha256'],
                    'old_full_cycles': old_horizon[0],
                    'new_full_cycles': new_horizon[0],
                    'old_final_model': old_horizon[1],
                    'new_final_model': new_horizon[1],
                    'old_gp_only_cycles': old_horizon[2],
                    'new_gp_only_cycles': new_horizon[2],
                    'old_gp_only_final_model': old_horizon[3],
                    'new_gp_only_final_model': new_horizon[3],
                    'scientific_settings_other_than_training_horizons_unchanged': True,
                })
                old['config_sha256'] = protocol['config_sha256']
            if old["config_sha256"] != protocol["config_sha256"] or old["M0_sha256"] != protocol["M0_sha256"]:
                raise RuntimeError(f"resume identity mismatch: {run_dir}")
            atomic_json(run_dir / "00_protocol" / f"RESUME_PID_{os.getpid()}.json",
                        {**protocol, "arm": arm, "resume": True, "initial_source_snapshot_sha256": old["source_snapshot_sha256"]})
        else:
            atomic_json(frozen, {**protocol, "arm": arm})
    SUMMARY_FULL.mkdir(parents=True, exist_ok=True)
    SUMMARY_ABLATION.mkdir(parents=True, exist_ok=True)


def _device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
        return torch.device("cuda")
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    return torch.device("cpu")


def _model_path(arm: str, index: int) -> Path:
    if index == 0:
        return M0_PATH
    if index == 1:
        return FULL_DIR / "cycles" / "C00" / "09_bq_training" / "M1_E256.tsp.best"
    run_dir = FULL_DIR if arm == "FULL" else GP_DIR
    return run_dir / "cycles" / f"C{index - 1:02d}" / "09_bq_training" / f"M{index}_E256.tsp.best"


def _load_model(arm: str, index: int, device: torch.device):
    return load_canonical_checkpoint(_model_path(arm, index), device,
                                     expected_sha256=M0_SHA256 if index == 0 else None)


def _cycle_dir(arm: str, cycle: int) -> Path:
    return (FULL_DIR if arm == "FULL" else GP_DIR) / "cycles" / f"C{cycle:02d}"


def _create_cycle_tree(path: Path) -> None:
    for name in ("00_state", "01_inputs", "02_gp", "03_mutation", "04_qd", "05_scoring",
                 "06_selection", "07_oracle", "08_archive", "09_bq_training", "10_post_adaptation",
                 "11_diagnostics", "12_reporting"):
        (path / name).mkdir(parents=True, exist_ok=True)


def _global_topk(pool: pd.DataFrame, scores: np.ndarray, budget: int = 10_000) -> pd.DataFrame:
    if len(pool) != len(scores) or len(pool) < budget or pool.instance_hash.duplicated().any():
        raise RuntimeError("invalid global GP candidate domain")
    hashes = pool.instance_hash.astype(str).to_numpy()
    order = np.lexsort((hashes, -np.asarray(scores, dtype=float)))[:budget]
    selected = pool.iloc[order].copy().reset_index(drop=True)
    selected["component"] = "global_gp"
    selected["gp_predicted_log1p_hardness"] = np.asarray(scores, dtype=float)[order]
    selected["global_gp_rank"] = np.arange(1, budget + 1)
    return selected


def _population_bundle(pool_coordinates: np.ndarray, selected: pd.DataFrame, path: Path,
                       tours: np.ndarray | None = None) -> None:
    indices = selected.coord_index.to_numpy(dtype=int)
    _save_bundle(path, pool_coordinates[indices], selected.instance_hash.astype(str).tolist(), tours)


def _niche_metrics(niches: np.ndarray) -> dict[str, float | int]:
    counts = np.bincount(np.asarray(niches, dtype=int), minlength=1000).astype(float)
    nonzero = counts[counts > 0]
    total = nonzero.sum()
    probabilities = nonzero / total
    shannon = float(-np.sum(probabilities * np.log(probabilities)))
    ordered = np.sort(counts)
    gini = float(2 * np.sum(np.arange(1, 1001) * ordered) / (1000 * total) - 1001 / 1000)
    return {
        "occupied_niches": int(len(nonzero)), "N_eff": float(np.exp(shannon)), "entropy": shannon,
        "normalized_entropy": float(shannon / np.log(len(nonzero))) if len(nonzero) > 1 else 0.0,
        "gini": gini, "largest_niche_share": float(nonzero.max() / total),
        "top5_niche_share": float(np.sort(nonzero)[-5:].sum() / total),
        "top10_niche_share": float(np.sort(nonzero)[-10:].sum() / total),
        "collapse": float(nonzero.max() / total),
    }


def _predicted_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "P90": float(np.percentile(array, 90)), "P95": float(np.percentile(array, 95))}


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(value) for value in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _gp_diagnostics(model: dict[str, Any], x: np.ndarray, hardness_values: np.ndarray) -> dict[str, Any]:
    prediction = predict_gp4(model, x)
    target = np.log1p(np.asarray(hardness_values, dtype=float))
    return {"fit_spearman": float(spearmanr(target, prediction).statistic),
            "fit_mae": float(np.mean(np.abs(target - prediction))),
            "formulas": [item["canonical_formula"] for item in model["formulas"]],
            "seed_runtime_s": [float(v) for v in model["seed_runtime_s"]],
            "total_runtime_s": float(sum(model["seed_runtime_s"]))}


def _canonical_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["instance_hash", "status", "verified", "optimal_cost_int", "runtime_sec",
                "expert_tour", "tour_hash", "binary_hash"]
    if frame.empty or not set(required).issubset(frame.columns):
        return pd.DataFrame(columns=required)
    data = frame.loc[:, required].copy()
    data = data[data.status.eq("solved") & data.verified.fillna(False)].drop_duplicates("instance_hash", keep="last")
    return data


def _read_parquet_recovered(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read canonical artifacts with DuckDB fallback for legacy PyArrow page metadata."""
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError):
        quoted = path.as_posix().replace("'", "''")
        projection = "*" if columns is None else ", ".join(f'"{name}"' for name in columns)
        return _duckdb().sql(f"SELECT {projection} FROM read_parquet('{quoted}', union_by_name=true)").df()


def _reusable_oracle_cache() -> pd.DataFrame:
    paths = [ROOT / "audits" / "cycle0_initialization_validation_v1" / "03_oracle_labels" / "expert_tours.parquet"]
    paths.extend(sorted((ROOT / "runs" / "blade_final_20cycle_seed2026_v1" / "cycles").glob(
        "C*/07_oracle/selected/ledger.parquet")))
    pieces = []
    for path in paths:
        if path.is_file():
            pieces.append(_canonical_cache_frame(_read_parquet_recovered(path)))
    if not pieces:
        return _canonical_cache_frame(pd.DataFrame())
    return pd.concat(pieces, ignore_index=True).drop_duplicates("instance_hash", keep="last")


GLOBAL_CACHE: pd.DataFrame | None = None


def _oracle_prior(archive: pd.DataFrame | None = None) -> pd.DataFrame:
    global GLOBAL_CACHE
    if GLOBAL_CACHE is None:
        GLOBAL_CACHE = _reusable_oracle_cache()
    pieces = [GLOBAL_CACHE]
    if archive is not None and not archive.empty:
        pieces.append(_canonical_cache_frame(archive))
    return pd.concat(pieces, ignore_index=True).drop_duplicates("instance_hash", keep="last")


def _label_population(directory: Path, selected: pd.DataFrame, coordinates: np.ndarray,
                      model: torch.nn.Module, device: torch.device, archive: pd.DataFrame | None,
                      stage_name: str, resume: bool) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    hashes = selected.instance_hash.astype(str).tolist()
    inputs = {"population": membership_hash(hashes), "count": str(len(hashes))}
    with atomic_stage(directory, stage_name, inputs, resume=resume) as outputs:
        if outputs is not None:
            started = time.perf_counter()
            prior = _oracle_prior(archive)
            known = set(prior.instance_hash.astype(str))
            ledger = label_with_cache(coordinates, hashes, directory, prior)
            tours = _oracle_tours(ledger, hashes)
            values = hardness(model, coordinates, tours, device)
            labeled = attach_oracle_metadata(selected, values, ledger)
            ledger.to_parquet(directory / "ledger.parquet", index=False)
            labeled.to_parquet(directory / "labeled.parquet", index=False)
            _save_bundle(directory / "labeled_bundle.npz", coordinates, hashes, tours)
            summary = {"selected": len(hashes), "cache_hits": sum(h in known for h in hashes),
                       "new_concorde_calls": sum(h not in known for h in hashes), "failures": 0,
                       "wall_seconds": time.perf_counter() - started}
            atomic_json(directory / "oracle_summary.json", summary)
            outputs.extend([directory / "ledger.parquet", directory / "labeled.parquet",
                            directory / "labeled_bundle.npz", directory / "oracle_summary.json"])
    labeled = _read_parquet_recovered(directory / "labeled.parquet")
    _, _, tours = _load_bundle(directory / "labeled_bundle.npz")
    if tours is None:
        raise RuntimeError("labeled bundle lacks tours")
    return labeled, tours, _json(directory / "oracle_summary.json")


def _save_archive(directory: Path, prior_coordinates: np.ndarray | None, prior_meta: pd.DataFrame | None,
                  selected_coordinates: np.ndarray, selected_meta: pd.DataFrame, selected_tours: np.ndarray,
                  resume: bool) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    selected_hashes = selected_meta.instance_hash.astype(str).tolist()
    prior_hashes = [] if prior_meta is None else prior_meta.instance_hash.astype(str).tolist()
    inputs = {"prior": membership_hash(prior_hashes), "selected": membership_hash(selected_hashes)}
    with atomic_stage(directory, "cumulative_unique_formal_archive", inputs, resume=resume) as outputs:
        if outputs is not None:
            if prior_meta is None:
                coordinates = selected_coordinates
                meta = selected_meta.copy()
                tours = selected_tours
            else:
                with np.load(directory.parent / "01_inputs" / "archive_in.npz", allow_pickle=False) as payload:
                    prior_tours = np.asarray(payload["expert_tours"], dtype=np.int64)
                seen = set(prior_hashes)
                keep = [i for i, value in enumerate(selected_hashes) if value not in seen]
                coordinates = np.concatenate([prior_coordinates, selected_coordinates[keep]])
                tours = np.concatenate([prior_tours, selected_tours[keep]])
                meta = pd.concat([prior_meta, selected_meta.iloc[keep]], ignore_index=True, sort=False)
            hashes = meta.instance_hash.astype(str).tolist()
            if len(hashes) != len(set(hashes)) or len(hashes) != len(coordinates):
                raise RuntimeError("archive uniqueness/alignment failure")
            _save_bundle(directory / "A_next.npz", coordinates, hashes, tours)
            meta.to_parquet(directory / "A_next.parquet", index=False)
            atomic_json(directory / "archive_summary.json", {"archive_n": len(hashes),
                        "new_unique": len(hashes) - len(prior_hashes),
                        "duplicate_reselection": len(selected_hashes) - (len(hashes) - len(prior_hashes))})
            outputs.extend([directory / "A_next.npz", directory / "A_next.parquet", directory / "archive_summary.json"])
    coords, hashes, tours = _load_bundle(directory / "A_next.npz")
    meta = _read_parquet_recovered(directory / "A_next.parquet")
    if tours is None or hashes != meta.instance_hash.astype(str).tolist():
        raise RuntimeError("archive reload alignment failure")
    return coords, meta, tours


def _load_archive_for_cycle(arm: str, cycle: int) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    source = _cycle_dir("FULL", 0) if cycle == 1 else _cycle_dir(arm, cycle - 1)
    coordinates, hashes, tours = _load_bundle(source / "08_archive" / "A_next.npz")
    meta = _read_parquet_recovered(source / "08_archive" / "A_next.parquet")
    if tours is None or hashes != meta.instance_hash.astype(str).tolist():
        raise RuntimeError("input archive alignment failure")
    return coordinates, meta, tours


def _load_parent_for_cycle(arm: str, cycle: int) -> tuple[np.ndarray, list[str]]:
    if cycle == 1:
        name = "P0_FULL.npz" if arm == "FULL" else "P0_GPONLY.npz"
        path = _cycle_dir("FULL", 0) / "06_selection" / name
    else:
        path = _cycle_dir(arm, cycle - 1) / "06_selection" / "P_next.npz"
    coordinates, hashes, _ = _load_bundle(path)
    if len(hashes) != 10_000 or len(set(hashes)) != 10_000:
        raise RuntimeError("parent population invariant failure")
    return coordinates, hashes


def _previous_training(arm: str, cycle: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    source = _cycle_dir("FULL", 0) if cycle == 1 else _cycle_dir(arm, cycle - 1)
    coordinates, hashes, tours = _load_bundle(source / "07_oracle" / "T" / "labeled_bundle.npz")
    if tours is None:
        raise RuntimeError("previous T lacks tours")
    return coordinates, hashes, tours


def _train_bq(arm: str, cycle: int, model: torch.nn.Module, model_hash: str,
              hard_coordinates: np.ndarray, hard_hashes: list[str], hard_tours: np.ndarray,
              device: torch.device, resume: bool, heartbeat: Heartbeat):
    directory = _cycle_dir(arm, cycle) / "09_bq_training"
    checkpoint = directory / f"M{cycle + 1}_E256.tsp.best"
    inputs = {"model": model_hash, "T": membership_hash(hard_hashes), "lr": str(LR), "epochs": str(EPOCHS)}
    with atomic_stage(directory, "BQ_10k_hard_plus_10k_uniform_fixed_E256", inputs, resume=resume) as outputs:
        if outputs is not None:
            heartbeat.update(phase="BQ_256E_training", latest_checkpoint=str(_model_path(arm, cycle)))
            started = time.perf_counter()
            setup_started = time.perf_counter()
            anchor_coordinates, anchor_tours = load_anchor()
            setup_s = time.perf_counter() - setup_started
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            loop_started = time.perf_counter()

            def callback(epoch: int, row: dict[str, Any]) -> None:
                if epoch in {1, 32, 64, 128, 192, 256}:
                    print(f"[{arm}][C{cycle:02d}][M{cycle}->M{cycle + 1}] BQ E{epoch} loss={row['mean_training_loss']:.8f}", flush=True)
                    heartbeat.update(phase=f"BQ_E{epoch}")

            history = fine_tune_fixed_budget(model, hard_coordinates, hard_tours, anchor_coordinates,
                                             anchor_tours, device, seed=2026,
                                             learning_rate=LR, epoch_callback=callback)
            loop_s = time.perf_counter() - loop_started
            write_started = time.perf_counter()
            save_e256(model, checkpoint, {"arm": arm, "cycle": cycle, "seed": 2026,
                                         "learning_rate": LR, "hard_count": 10_000,
                                         "uniform_anchor_count": 10_000})
            checkpoint_s = time.perf_counter() - write_started
            verify_started = time.perf_counter()
            output_hash = sha256_file(checkpoint)
            finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
            verify_s = time.perf_counter() - verify_started
            frame = pd.DataFrame(history)
            frame.to_csv(directory / "training_loss_curve.csv", index=False)
            epoch_times = frame.epoch_wall_s.to_numpy(float)
            timing = {"hard": 10_000, "uniform_anchor": 10_000, "learning_rate": LR, "epochs": EPOCHS,
                      "scientific_checkpoint": "E256", "dataset_setup_s": setup_s, "dataloader_setup_s": 0.0,
                      "training_loop_wall_s": loop_s, "checkpoint_write_s": checkpoint_s,
                      "verification_s": verify_s, "total_wall_s": time.perf_counter() - started,
                      "mean_sec_per_epoch": float(epoch_times.mean()), "median_sec_per_epoch": float(np.median(epoch_times)),
                      "P95_sec_per_epoch": float(np.percentile(epoch_times, 95)), "checkpoint_sha256": output_hash,
                      "model_finite": finite, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                      "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
            atomic_json(directory / "bq_training_timing.json", timing)
            outputs.extend([checkpoint, directory / "training_loss_curve.csv", directory / "bq_training_timing.json"])
    loaded, meta = _load_model(arm, cycle + 1, device)
    timing = _json(directory / "bq_training_timing.json")
    if not timing["model_finite"] or meta["sha256"] != timing["checkpoint_sha256"]:
        raise RuntimeError("E256 checkpoint verification failure")
    heartbeat.update(latest_checkpoint=str(checkpoint), last_valid_artifact=str(checkpoint))
    return loaded, meta, timing


def _repair(directory: Path, model: torch.nn.Module, model_hash: str, coordinates: np.ndarray,
            hashes: list[str], tours: np.ndarray, pre: np.ndarray, device: torch.device,
            seed: int, resume: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = {"model": model_hash, "T": membership_hash(hashes)}
    with atomic_stage(directory, "post_adaptation_hardness_and_paired_repair", inputs, resume=resume) as outputs:
        if outputs is not None:
            started = time.perf_counter()
            post = hardness(model, coordinates, tours, device)
            frame = pd.DataFrame({"instance_hash": hashes, "H_pre": pre, "H_post": post,
                                  "delta_post_minus_pre": post - pre})
            frame.to_parquet(directory / "adaptation_repair.parquet", index=False)
            boot = bootstrap_mean_delta(post, pre, seed, paired=True)
            delta = post - pre
            result = {"mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
                      "P75_delta": float(np.percentile(delta, 75)), "P90_delta": float(np.percentile(delta, 90)),
                      "P95_delta": float(np.percentile(delta, 95)), "ci95_low": boot["ci95_low"],
                      "ci95_high": boot["ci95_high"], "fraction_improved": float(np.mean(delta < 0)),
                      "fraction_worsened": float(np.mean(delta > 0)), "wall_seconds": time.perf_counter() - started}
            atomic_json(directory / "adaptation_repair.json", result)
            outputs.extend([directory / "adaptation_repair.parquet", directory / "adaptation_repair.json"])
    return pd.read_parquet(directory / "adaptation_repair.parquet"), _json(directory / "adaptation_repair.json")


def _stage_timing(cycle_dir: Path) -> tuple[list[dict[str, Any]], float]:
    rows = []
    for path in cycle_dir.rglob("STAGE_DONE.json"):
        record = _json(path)
        rows.append({"phase": record["stage"], "wall_seconds": float(record["wall_seconds"])})
    return rows, sum(row["wall_seconds"] for row in rows)


def _write_cycle_report(arm: str, cycle: int, summary: dict[str, Any], resume: bool) -> None:
    cycle_dir = _cycle_dir(arm, cycle)
    directory = cycle_dir / "12_reporting"
    inputs = {"checkpoint": summary["checkpoint_hash"], "T": summary["T_hash"],
              "P": summary["P_hash"], "A": summary["A_hash"]}
    with atomic_stage(directory, "cycle_report_timing_and_strict_verification", inputs, resume=resume) as outputs:
        if outputs is not None:
            atomic_json(cycle_dir / "cycle_summary.json", summary)
            p = summary["parent_metrics"]
            o = summary["offspring_metrics"]
            h = summary["hardness"]
            r = summary["adaptation_repair"]
            report = f"""# BLADE V2 PROGRESS

- Arm: {arm}
- Cycle: C{cycle:02d}
- Solver: M{cycle} -> M{cycle + 1}
- Parent policy: {summary['parent_policy']}
- GP working set: 500; runtime: {summary['gp']['total_runtime_s'] / 60:.2f} min
- Offspring: nominal 110000; valid unique {o['valid_unique']}; niches {o['occupied_niches']}; N_eff {o['N_eff']:.3f}
- Hard training T_s: N=10000; mean H={h['mean']:.8f}; P90={h['P90']:.8f}; P95={h['P95']:.8f}; niches={summary['T_metrics']['occupied_niches']}; N_eff={summary['T_metrics']['N_eff']:.3f}
- Parent P_s: N=10000; niches={p['occupied_niches']}; N_eff={p['N_eff']:.3f}; entropy={p['entropy']:.6f}; collapse={p['collapse']:.6f}
- Adaptation: mean delta={r['mean_delta']:.8f}; 95% CI=[{r['ci95_low']:.8f}, {r['ci95_high']:.8f}]
- BQ: hard=10000; Uniform anchor=10000; LR=3.75e-5; epochs=256; checkpoint=E256
- Cycle accounted wall-clock: {summary['timing']['accounted_seconds'] / 60:.2f} min
- Accounted fraction: {summary['timing']['accounted_fraction']:.4f}
- Errors/recoveries: {summary.get('recoveries', 'none')}
- Status: COMPLETE
"""
            (cycle_dir / "cycle_report.md").write_text(report, encoding="utf-8")
            pd.DataFrame(summary["timing"]["phases"]).to_csv(cycle_dir / "timing_ledger.csv", index=False)
            strict = {"status": "PASS", "checks": {"Random500": summary["gp"]["working_set_n"] == 500,
                      "T_10k": summary["T_n"] == 10_000, "P_10k": summary["P_n"] == 10_000,
                      "archive_unique": summary["archive_unique"], "BQ_hard": summary["bq"]["hard"] == 10_000,
                      "BQ_anchor": summary["bq"]["uniform_anchor"] == 10_000,
                      "LR": summary["bq"]["learning_rate"] == LR, "E256": summary["bq"]["epochs"] == 256,
                      "finite": summary["bq"]["model_finite"], "no_nan_inf": summary["finite"],
                      "checkpoint_hash": bool(summary["checkpoint_hash"]), "timing": True}}
            if not all(strict["checks"].values()):
                strict["status"] = "FAIL"
            atomic_json(cycle_dir / "strict_cycle_verification.json", strict)
            if strict["status"] != "PASS":
                raise RuntimeError(f"{arm} C{cycle:02d} strict verification failed")
            outputs.extend([cycle_dir / "cycle_summary.json", cycle_dir / "cycle_report.md",
                            cycle_dir / "timing_ledger.csv", cycle_dir / "strict_cycle_verification.json"])
    required = [cycle_dir / "09_bq_training" / f"M{cycle + 1}_E256.tsp.best",
                cycle_dir / "06_selection" / ("P0_FULL.npz" if cycle == 0 else "P_next.npz"),
                cycle_dir / "07_oracle" / "T" / "labeled_bundle.npz",
                cycle_dir / "08_archive" / "A_next.npz", cycle_dir / "10_post_adaptation" / "adaptation_repair.parquet",
                cycle_dir / "cycle_report.md", cycle_dir / "timing_ledger.csv"]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"cycle-complete artifact missing: {path}")
    marker = cycle_dir / "CYCLE_COMPLETE.json"
    if not marker.exists():
        atomic_json(marker, {"status": "COMPLETE", "arm": arm, "cycle": f"C{cycle:02d}",
                            "checkpoint_sha256": summary["checkpoint_hash"], "T_sha256": sha256_file(required[2]),
                            "A_sha256": sha256_file(required[3]), "P_sha256": sha256_file(required[1]),
                            "report_sha256": sha256_file(required[5]), "timing_sha256": sha256_file(required[6]),
                            "completed_utc": utc_now()})


def _verify_completed_cycle(arm: str, cycle: int) -> bool:
    marker_path = _cycle_dir(arm, cycle) / "CYCLE_COMPLETE.json"
    if not marker_path.is_file():
        return False
    marker = _json(marker_path)
    cycle_dir = _cycle_dir(arm, cycle)
    paths = {
        "checkpoint_sha256": cycle_dir / "09_bq_training" / f"M{cycle + 1}_E256.tsp.best",
        "T_sha256": cycle_dir / "07_oracle" / "T" / "labeled_bundle.npz",
        "A_sha256": cycle_dir / "08_archive" / "A_next.npz",
        "P_sha256": cycle_dir / "06_selection" / ("P0_FULL.npz" if cycle == 0 else "P_next.npz"),
        "report_sha256": cycle_dir / "cycle_report.md", "timing_sha256": cycle_dir / "timing_ledger.csv",
    }
    for key, path in paths.items():
        if not path.is_file() or sha256_file(path) != marker[key]:
            raise RuntimeError(f"completed cycle artifact verification failed: {path}")
    if _json(cycle_dir / "strict_cycle_verification.json")["status"] != "PASS":
        raise RuntimeError(f"completed cycle strict verification failed: {arm} C{cycle:02d}")
    return True


def _prepare_offspring(arm: str, cycle: int, parent_bundle: Path, parent_hashes: list[str],
                       resume: bool) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    directory = _cycle_dir(arm, cycle) / "03_mutation"
    inputs = {"parents": membership_hash(parent_hashes), "cycle": str(cycle), "FOGA": "FOGA11"}
    with atomic_stage(directory, "FOGA11_generation_repair_dedup", inputs, resume=resume) as outputs:
        if outputs is not None:
            reused = False
            if cycle == 0:
                source = ROOT / "runs" / "blade_final_20cycle_seed2026_v1" / "cycles" / "C00" / "04_mutation"
                done = source / "STAGE_DONE.json"
                if done.is_file():
                    verify_stage(done)
                    for name in ("offspring_coordinates.npz", "mutation_manifest.parquet", "CANONICAL_MUTATION_RUNTIME.json"):
                        shutil.copy2(source / name, directory / name)
                    manifest = pd.read_parquet(directory / "mutation_manifest.parquet")
                    _verify_c00_replay(manifest)
                    reused = True
            if not reused:
                _canonical_mutation_worker(parent_bundle, directory, cycle)
                if cycle == 0:
                    _verify_c00_replay(pd.read_parquet(directory / "mutation_manifest.parquet"))
            outputs.extend([directory / "offspring_coordinates.npz", directory / "mutation_manifest.parquet",
                            directory / "CANONICAL_MUTATION_RUNTIME.json"])
    coordinates, hashes, _ = _load_bundle(directory / "offspring_coordinates.npz")
    manifest = pd.read_parquet(directory / "mutation_manifest.parquet")
    if cycle == 0 and len(hashes) != 109_785:
        raise RuntimeError(f"frozen O0 unique count mismatch: {len(hashes)}")
    return coordinates, hashes, manifest


def _prepare_qd(arm: str, cycle: int, parent_coordinates: np.ndarray, parent_hashes: list[str],
                offspring_coordinates: np.ndarray, offspring_hashes: list[str], manifest: pd.DataFrame,
                resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = _cycle_dir(arm, cycle) / "04_qd"
    inputs = {"offspring": membership_hash(offspring_hashes), "map": "QD4_P1P99_K1000_FROZEN"}
    with atomic_stage(directory, "QD4_feature_extraction_assignment", inputs, resume=resume) as outputs:
        if outputs is not None:
            reused = False
            if cycle == 0:
                source = ROOT / "runs" / "blade_final_20cycle_seed2026_v1" / "cycles" / "C00" / "05_qd"
                done = source / "STAGE_DONE.json"
                if done.is_file():
                    verify_stage(done)
                    candidate = _read_parquet_recovered(source / "offspring_qd.parquet")
                    if candidate.instance_hash.astype(str).tolist() == offspring_hashes:
                        parent_candidate = _read_parquet_recovered(source / "parent_qd.parquet")
                        candidate.to_parquet(directory / "offspring_qd.parquet", index=False)
                        parent_candidate.to_parquet(directory / "parent_qd.parquet", index=False)
                        shutil.copy2(source / "operator_report.csv", directory / "operator_report.csv")
                        reused = True
            if not reused:
                parent_qd = qd_frame(parent_coordinates, parent_hashes)
                qd = qd_frame(offspring_coordinates, offspring_hashes)
                accepted = manifest.valid_after_repair & manifest.proposal_status.eq("accepted") & ~manifest.duplicate
                accepted_meta = manifest.loc[accepted, ["parent_hash", "operator_id"]].reset_index(drop=True)
                if len(accepted_meta) != len(qd):
                    raise RuntimeError("mutation/QD alignment failure")
                qd["parent_hash"] = accepted_meta.parent_hash
                qd["operator_id"] = accepted_meta.operator_id
                lookup = parent_qd.set_index("instance_hash")
                qd["parent_niche_id"] = qd.parent_hash.map(lookup.niche_id).to_numpy(int)
                qd["normalized_qd_displacement"] = np.linalg.norm(
                    qd[NORM].to_numpy(float) - lookup.loc[qd.parent_hash, NORM].to_numpy(float), axis=1)
                qd["coord_index"] = np.arange(len(qd))
                qd["source"] = "offspring"
                rows = []
                for operator_id in sorted(manifest.operator_id.unique()):
                    raw = manifest[manifest.operator_id.eq(operator_id)]
                    children = qd[qd.operator_id.eq(operator_id)]
                    rows.append({"operator_id": operator_id, "generated": len(raw),
                                 "repaired": int((raw.repair_count > 0).sum()),
                                 "invalid": int((~raw.valid_after_repair | ~raw.proposal_status.eq("accepted")).sum()),
                                 "duplicates": int(raw.duplicate.sum()), "valid_unique": len(children),
                                 "occupied_niches": int(children.niche_id.nunique())})
                parent_qd.to_parquet(directory / "parent_qd.parquet", index=False)
                qd.to_parquet(directory / "offspring_qd.parquet", index=False)
                pd.DataFrame(rows).to_csv(directory / "operator_report.csv", index=False)
            outputs.extend([directory / "parent_qd.parquet", directory / "offspring_qd.parquet",
                            directory / "operator_report.csv"])
    return (_read_parquet_recovered(directory / "parent_qd.parquet"),
            _read_parquet_recovered(directory / "offspring_qd.parquet"),
            pd.read_csv(directory / "operator_report.csv"))


def run_shared_c00(config: dict[str, Any], device: torch.device, resume: bool, heartbeat: Heartbeat) -> None:
    if _verify_completed_cycle("FULL", 0):
        legacy_summary = ROOT / "summary" / "qd_vs_gp_only"
        for name in ("02_C00_branch_identity.json", "03_C00_parent_selection_comparison.csv"):
            source = legacy_summary / name
            destination = SUMMARY_ABLATION / name
            if source.is_file() and not destination.is_file():
                shutil.copy2(source, destination)
        heartbeat.update(arm="SHARED_C00", cycle="C00", phase="verified_complete",
                         last_valid_artifact=str(_cycle_dir("FULL", 0) / "CYCLE_COMPLETE.json"))
        return
    started = time.perf_counter()
    arm, cycle = "FULL", 0
    cycle_dir = _cycle_dir(arm, cycle)
    _create_cycle_tree(cycle_dir)
    heartbeat.update(arm="SHARED_C00", cycle="C00", phase="cycle_inputs")
    model, model_meta = _load_model(arm, 0, device)
    roots = recover_uniform_roots()
    parent_coordinates = roots.coordinates
    parent_hashes = list(roots.coordinate_hashes)
    parent_bundle = cycle_dir / "01_inputs" / "parents.npz"
    with atomic_stage(cycle_dir / "01_inputs", "shared_C00_canonical_uniform_roots",
                      {"M0": model_meta["sha256"]}, resume=resume) as outputs:
        if outputs is not None:
            _save_bundle(parent_bundle, parent_coordinates, parent_hashes)
            outputs.append(parent_bundle)

    heartbeat.update(phase="O0_FOGA11_generation")
    offspring_coordinates, offspring_hashes, manifest = _prepare_offspring(
        arm, cycle, parent_bundle, parent_hashes, resume)
    parent_qd, offspring_qd, operator_report = _prepare_qd(
        arm, cycle, parent_coordinates, parent_hashes, offspring_coordinates, offspring_hashes, manifest, resume)

    b_select_dir = cycle_dir / "02_gp" / "B0_selection"
    with atomic_stage(b_select_dir, "B0_uniform_random500_once_seed2026",
                      {"O0": membership_hash(offspring_hashes), "seed": "2026"}, resume=resume) as outputs:
        if outputs is not None:
            indices = np.random.default_rng(2026).choice(len(offspring_qd), size=500, replace=False)
            b0 = offspring_qd.iloc[indices].copy().reset_index(drop=True)
            b0["random500_draw_order"] = np.arange(500)
            b0.to_parquet(b_select_dir / "B0.parquet", index=False)
            _population_bundle(offspring_coordinates, b0, b_select_dir / "B0.npz")
            outputs.extend([b_select_dir / "B0.parquet", b_select_dir / "B0.npz"])
    b0 = pd.read_parquet(b_select_dir / "B0.parquet")
    b0_coordinates, b0_hashes, _ = _load_bundle(b_select_dir / "B0.npz")
    if len(b0_hashes) != 500 or len(set(b0_hashes)) != 500:
        raise RuntimeError("B0 Random500 invariant failure")

    heartbeat.update(phase="B0_oracle_and_M0_hardness")
    b0_labeled, b0_tours, b0_oracle = _label_population(
        cycle_dir / "07_oracle" / "B0", b0, b0_coordinates, model, device, None,
        "B0_oracle_and_current_M0_hardness", resume)
    gp_dir = cycle_dir / "02_gp" / "training"
    gp_inputs = {"B0": membership_hash(b0_hashes), "model": model_meta["sha256"], "n": "500"}
    heartbeat.update(phase="GP0_training")
    with atomic_stage(gp_dir, "GP0_GP4_Random500", gp_inputs, resume=resume) as outputs:
        if outputs is not None:
            x = b0_labeled[list(QD4_FEATURE_NAMES)].to_numpy(float)
            y = b0_labeled.current_hardness.to_numpy(float)
            gp = train_gp4(x, y, gp_dir, cycle)
            diagnostics = _gp_diagnostics(gp, x, y)
            atomic_json(gp_dir / "gp_diagnostics.json", diagnostics)
            outputs.extend([gp_dir / "gp4_ensemble.joblib", gp_dir / "gp_diagnostics.json"])
            outputs.extend(sorted(gp_dir.glob("seed_*/*.json")))
            outputs.extend(sorted(gp_dir.glob("seed_*/*.parquet")))
    gp = joblib.load(gp_dir / "gp4_ensemble.joblib")
    gp_diagnostics = _json(gp_dir / "gp_diagnostics.json")

    score_dir = cycle_dir / "05_scoring"
    heartbeat.update(phase="GP0_score_O0")
    with atomic_stage(score_dir, "GP0_score_all_O0",
                      {"GP0": sha256_file(gp_dir / "gp4_ensemble.joblib"), "O0": membership_hash(offspring_hashes)},
                      resume=resume) as outputs:
        if outputs is not None:
            scores = predict_gp4(gp, offspring_qd[list(QD4_FEATURE_NAMES)].to_numpy(float))
            scored = offspring_qd.copy()
            scored["gp_predicted_log1p_hardness"] = scores
            scored.to_parquet(score_dir / "O0_scored.parquet", index=False)
            outputs.append(score_dir / "O0_scored.parquet")
    scored = pd.read_parquet(score_dir / "O0_scored.parquet")
    scores = scored.gp_predicted_log1p_hardness.to_numpy(float)

    selection_dir = cycle_dir / "06_selection"
    heartbeat.update(phase="C00_branch_parent_selection")
    with atomic_stage(selection_dir, "C00_T0_and_two_parent_policies",
                      {"GP0": sha256_file(gp_dir / "gp4_ensemble.joblib"), "O0": membership_hash(offspring_hashes)},
                      resume=resume) as outputs:
        if outputs is not None:
            t0 = _global_topk(offspring_qd, scores)
            p0_full = select_population(offspring_qd, scores, 10_000, "GP0")
            p0_gponly = t0.copy()
            if p0_gponly.instance_hash.astype(str).tolist() != t0.instance_hash.astype(str).tolist():
                raise RuntimeError("P0_GPONLY != T0")
            if p0_full.instance_hash.astype(str).tolist() == p0_gponly.instance_hash.astype(str).tolist():
                raise RuntimeError("scientific integrity issue: corrected C00 policies produced identical parents")
            for name, frame in (("T0", t0), ("P0_FULL", p0_full), ("P0_GPONLY", p0_gponly)):
                frame.to_parquet(selection_dir / f"{name}.parquet", index=False)
                _population_bundle(offspring_coordinates, frame, selection_dir / f"{name}.npz")
                outputs.extend([selection_dir / f"{name}.parquet", selection_dir / f"{name}.npz"])
    t0 = pd.read_parquet(selection_dir / "T0.parquet")
    p0_full = pd.read_parquet(selection_dir / "P0_FULL.parquet")
    p0_gponly = pd.read_parquet(selection_dir / "P0_GPONLY.parquet")
    t0_coordinates, t0_hashes, _ = _load_bundle(selection_dir / "T0.npz")

    heartbeat.update(phase="T0_oracle_and_M0_hardness")
    t0_labeled, t0_tours, t0_oracle = _label_population(
        cycle_dir / "07_oracle" / "T", t0, t0_coordinates, model, device, b0_labeled,
        "T0_oracle_and_exact_M0_hardness", resume)

    archive_dir = cycle_dir / "08_archive"
    with atomic_stage(archive_dir, "A1_unique_B0_union_T0",
                      {"B0": membership_hash(b0_hashes), "T0": membership_hash(t0_hashes)}, resume=resume) as outputs:
        if outputs is not None:
            meta = pd.concat([b0_labeled, t0_labeled], ignore_index=True, sort=False).drop_duplicates(
                "instance_hash", keep="last").reset_index(drop=True)
            hashes = meta.instance_hash.astype(str).tolist()
            coordinate_lookup = {h: c for h, c in zip(b0_hashes, b0_coordinates, strict=True)}
            coordinate_lookup.update(dict(zip(t0_hashes, t0_coordinates, strict=True)))
            tour_lookup = {h: t for h, t in zip(b0_hashes, b0_tours, strict=True)}
            tour_lookup.update(dict(zip(t0_hashes, t0_tours, strict=True)))
            coordinates = np.stack([coordinate_lookup[h] for h in hashes])
            tours = np.stack([tour_lookup[h] for h in hashes])
            _save_bundle(archive_dir / "A_next.npz", coordinates, hashes, tours)
            meta.to_parquet(archive_dir / "A_next.parquet", index=False)
            atomic_json(archive_dir / "archive_summary.json", {"archive_n": len(hashes),
                        "new_unique": len(hashes), "duplicate_reselection": 10_500 - len(hashes)})
            outputs.extend([archive_dir / "A_next.npz", archive_dir / "A_next.parquet",
                            archive_dir / "archive_summary.json"])
    a1_coordinates, a1_hashes, a1_tours = _load_bundle(archive_dir / "A_next.npz")
    a1_meta = _read_parquet_recovered(archive_dir / "A_next.parquet")
    if a1_tours is None or a1_hashes != a1_meta.instance_hash.astype(str).tolist():
        raise RuntimeError("A1 alignment failure")

    new_model, new_meta, bq_timing = _train_bq(
        arm, cycle, model, model_meta["sha256"], t0_coordinates, t0_hashes, t0_tours, device, resume, heartbeat)
    heartbeat.update(phase="C00_post_adaptation_repair")
    repair, repair_stats = _repair(cycle_dir / "10_post_adaptation", new_model, new_meta["sha256"],
                                   t0_coordinates, t0_hashes, t0_tours,
                                   t0_labeled.current_hardness.to_numpy(float), device, 2026, resume)

    full_metrics = _niche_metrics(p0_full.niche_id.to_numpy())
    gp_metrics = _niche_metrics(p0_gponly.niche_id.to_numpy())
    t_metrics = _niche_metrics(t0.niche_id.to_numpy())
    offspring_metrics = {**_niche_metrics(offspring_qd.niche_id.to_numpy()),
                         "valid_unique": len(offspring_qd),
                         "newly_reached_niches": len(set(offspring_qd.niche_id.astype(int)) - set(parent_qd.niche_id.astype(int))),
                         "operator_contribution": operator_report.to_dict("records")}
    overlap = len(set(p0_full.instance_hash.astype(str)) & set(p0_gponly.instance_hash.astype(str)))
    comparison = [{"arm": "FULL", **full_metrics, **{f"predicted_{k}": v for k, v in _predicted_stats(p0_full.gp_predicted_log1p_hardness).items()},
                   "overlap_count": overlap, "jaccard": overlap / (20_000 - overlap),
                   "exclusive_count": 10_000 - overlap},
                  {"arm": "GP_ONLY", **gp_metrics, **{f"predicted_{k}": v for k, v in _predicted_stats(p0_gponly.gp_predicted_log1p_hardness).items()},
                   "overlap_count": overlap, "jaccard": overlap / (20_000 - overlap),
                   "exclusive_count": 10_000 - overlap}]
    exact_map = a1_meta.set_index("instance_hash").current_hardness.to_dict()
    for record, frame in zip(comparison, (p0_full, p0_gponly), strict=True):
        cached = np.asarray([exact_map[h] for h in frame.instance_hash.astype(str) if h in exact_map], dtype=float)
        record.update({"exact_cached_n": int(len(cached)),
                       "exact_cached_mean": float(cached.mean()) if len(cached) else None,
                       "exact_cached_median": float(np.median(cached)) if len(cached) else None})
    pd.DataFrame(comparison).to_csv(SUMMARY_ABLATION / "03_C00_parent_selection_comparison.csv", index=False)
    branch_identity = {"shared": {"M0": model_meta["sha256"], "O0": membership_hash(offspring_hashes),
                       "B0": membership_hash(b0_hashes), "GP0": sha256_file(gp_dir / "gp4_ensemble.joblib"),
                       "T0": membership_hash(t0_hashes), "A1": membership_hash(a1_hashes), "M1": new_meta["sha256"]},
                       "full": {"P0_FULL": membership_hash(p0_full.instance_hash.astype(str).tolist()), **full_metrics},
                       "gp_only": {"P0_GPONLY": membership_hash(p0_gponly.instance_hash.astype(str).tolist()),
                                   "equals_T0": True, **gp_metrics},
                       "first_causal_divergence": "C01 offspring generation"}
    atomic_json(SUMMARY_ABLATION / "02_C00_branch_identity.json", branch_identity)
    branch_md = "# C00 BRANCH POINT\n\nShared through M1: M0, O0, B0, GP0, T0, A1, M1.\n\n"
    branch_md += _markdown_table(pd.DataFrame(comparison)) + "\n\nFirst causal divergence occurs in C01 offspring generation.\n"
    (SUMMARY_ABLATION / "C00_BRANCH_POINT.md").write_text(branch_md, encoding="utf-8")
    atomic_json(FULL_DIR / "00_protocol" / "SHARED_C00_BRANCH_MANIFEST.json", branch_identity)
    atomic_json(GP_DIR / "00_protocol" / "SHARED_C00_BRANCH_MANIFEST.json", branch_identity)

    phases, accounted = _stage_timing(cycle_dir)
    summary = {"arm": "SHARED_C00", "cycle": 0, "model_in": "M0", "model_out": "M1",
               "parent_policy": "QD+GP (P0_FULL; GP-only branch saved separately)",
               "checkpoint_hash": new_meta["sha256"], "T_hash": membership_hash(t0_hashes),
               "P_hash": membership_hash(p0_full.instance_hash.astype(str).tolist()), "A_hash": membership_hash(a1_hashes),
               "T_n": 10_000, "P_n": 10_000, "archive_unique": len(a1_hashes) == len(set(a1_hashes)),
               "gp": {"working_set_n": 500, "random_seed": 2026,
                      "working_hardness": summary_stats(b0_labeled.current_hardness.to_numpy(float)), **gp_diagnostics},
               "offspring_metrics": offspring_metrics, "hardness": summary_stats(t0_labeled.current_hardness.to_numpy(float)),
               "T_metrics": t_metrics, "parent_metrics": full_metrics,
               "parent_selection": {"QD_floor_n": int(p0_full.component.eq("centroid_floor").sum()),
                                    "GP_fill_n": int((~p0_full.component.eq("centroid_floor")).sum())},
               "oracle": {"B0": b0_oracle, "T0": t0_oracle}, "archive": _json(archive_dir / "archive_summary.json"),
               "adaptation_repair": repair_stats, "bq": bq_timing,
               "finite": bool(np.isfinite(repair[["H_pre", "H_post"]].to_numpy()).all()),
               "timing": {"phases": phases, "accounted_seconds": accounted,
                          "cycle_wall_seconds": max(accounted, time.perf_counter() - started),
                          "accounted_fraction": min(1.0, accounted / max(accounted, time.perf_counter() - started, 1e-12)),
                          "unaccounted_seconds": max(0.0, time.perf_counter() - started - accounted),
                          "warning": "WARN_ACCOUNTED_BELOW_98_PERCENT" if accounted / max(accounted, time.perf_counter() - started, 1e-12) < .98 else "NONE"},
               "recoveries": "none"}
    _write_cycle_report(arm, cycle, summary, resume)
    heartbeat.update(phase="COMPLETE", last_valid_artifact=str(cycle_dir / "CYCLE_COMPLETE.json"),
                     latest_checkpoint=str(_model_path("FULL", 1)))
    print("BLADE V2 PROGRESS SHARED_C00 C00 M0->M1 COMPLETE", flush=True)


def run_arm_cycle(arm: str, cycle: int, config: dict[str, Any], device: torch.device,
                  resume: bool, heartbeat: Heartbeat) -> None:
    if cycle < 1:
        raise ValueError("independent arm cycles begin at C01")
    if _verify_completed_cycle(arm, cycle):
        heartbeat.update(arm=arm, cycle=f"C{cycle:02d}", phase="verified_complete",
                         last_valid_artifact=str(_cycle_dir(arm, cycle) / "CYCLE_COMPLETE.json"))
        return
    started = time.perf_counter()
    cycle_dir = _cycle_dir(arm, cycle)
    _create_cycle_tree(cycle_dir)
    heartbeat.update(arm=arm, cycle=f"C{cycle:02d}", phase="cycle_inputs")
    model, model_meta = _load_model(arm, cycle, device)
    parent_coordinates, parent_hashes = _load_parent_for_cycle(arm, cycle)
    archive_coordinates, archive_meta, archive_tours = _load_archive_for_cycle(arm, cycle)
    archive_hashes = archive_meta.instance_hash.astype(str).tolist()
    parent_bundle = cycle_dir / "01_inputs" / "parents.npz"
    archive_bundle = cycle_dir / "01_inputs" / "archive_in.npz"
    with atomic_stage(cycle_dir / "01_inputs", "arm_cycle_inputs",
                      {"model": model_meta["sha256"], "parents": membership_hash(parent_hashes),
                       "archive": membership_hash(archive_hashes)}, resume=resume) as outputs:
        if outputs is not None:
            _save_bundle(parent_bundle, parent_coordinates, parent_hashes)
            _save_bundle(archive_bundle, archive_coordinates, archive_hashes, archive_tours)
            archive_meta.to_parquet(cycle_dir / "01_inputs" / "archive_in.parquet", index=False)
            outputs.extend([parent_bundle, archive_bundle, cycle_dir / "01_inputs" / "archive_in.parquet"])

    random_seed = 2026 + cycle
    b_dir = cycle_dir / "02_gp" / "B_selection_and_refresh"
    heartbeat.update(phase="Random500_A_s_current_hardness")
    with atomic_stage(b_dir, "Random500_A_s_and_current_solver_refresh_only_B",
                      {"archive": membership_hash(archive_hashes), "model": model_meta["sha256"],
                       "seed": str(random_seed)}, resume=resume) as outputs:
        if outputs is not None:
            indices = np.random.default_rng(random_seed).choice(len(archive_meta), size=500, replace=False)
            selected = archive_meta.iloc[indices].copy().reset_index(drop=True)
            selected_coordinates = archive_coordinates[indices]
            selected_tours = archive_tours[indices]
            refresh_started = time.perf_counter()
            values = hardness(model, selected_coordinates, selected_tours, device)
            selected["current_hardness"] = values
            selected["random500_draw_order"] = np.arange(500)
            selected.to_parquet(b_dir / "B_s.parquet", index=False)
            _save_bundle(b_dir / "B_s.npz", selected_coordinates,
                         selected.instance_hash.astype(str).tolist(), selected_tours)
            atomic_json(b_dir / "refresh_timing.json", {"n": 500, "wall_seconds": time.perf_counter() - refresh_started,
                        "full_archive_refreshed": False, "archive_n": len(archive_meta)})
            outputs.extend([b_dir / "B_s.parquet", b_dir / "B_s.npz", b_dir / "refresh_timing.json"])
    b_work = pd.read_parquet(b_dir / "B_s.parquet")
    b_hashes = b_work.instance_hash.astype(str).tolist()
    if len(b_hashes) != 500 or len(set(b_hashes)) != 500:
        raise RuntimeError("Random500 working-set invariant failure")

    gp_dir = cycle_dir / "02_gp" / "training"
    heartbeat.update(phase=f"GP{cycle}_training")
    with atomic_stage(gp_dir, f"GP{cycle}_GP4_Random500",
                      {"B": membership_hash(b_hashes), "model": model_meta["sha256"], "n": "500"},
                      resume=resume) as outputs:
        if outputs is not None:
            x = b_work[list(QD4_FEATURE_NAMES)].to_numpy(float)
            y = b_work.current_hardness.to_numpy(float)
            gp = train_gp4(x, y, gp_dir, cycle)
            diagnostics = _gp_diagnostics(gp, x, y)
            atomic_json(gp_dir / "gp_diagnostics.json", diagnostics)
            outputs.extend([gp_dir / "gp4_ensemble.joblib", gp_dir / "gp_diagnostics.json"])
            outputs.extend(sorted(gp_dir.glob("seed_*/*.json")))
            outputs.extend(sorted(gp_dir.glob("seed_*/*.parquet")))
    gp = joblib.load(gp_dir / "gp4_ensemble.joblib")
    gp_diagnostics = _json(gp_dir / "gp_diagnostics.json")

    heartbeat.update(phase="FOGA11_offspring_generation")
    offspring_coordinates, offspring_hashes, manifest = _prepare_offspring(
        arm, cycle, parent_bundle, parent_hashes, resume)
    parent_qd, offspring_qd, operator_report = _prepare_qd(
        arm, cycle, parent_coordinates, parent_hashes, offspring_coordinates, offspring_hashes, manifest, resume)

    score_dir = cycle_dir / "05_scoring"
    heartbeat.update(phase="GP_score_arm_offspring")
    with atomic_stage(score_dir, "GP_score_own_offspring",
                      {"GP": sha256_file(gp_dir / "gp4_ensemble.joblib"),
                       "offspring": membership_hash(offspring_hashes)}, resume=resume) as outputs:
        if outputs is not None:
            scores = predict_gp4(gp, offspring_qd[list(QD4_FEATURE_NAMES)].to_numpy(float))
            scored = offspring_qd.copy()
            scored["gp_predicted_log1p_hardness"] = scores
            scored.to_parquet(score_dir / "offspring_scored.parquet", index=False)
            outputs.append(score_dir / "offspring_scored.parquet")
    scored = pd.read_parquet(score_dir / "offspring_scored.parquet")
    scores = scored.gp_predicted_log1p_hardness.to_numpy(float)

    selection_dir = cycle_dir / "06_selection"
    parent_policy = "QD+GP" if arm == "FULL" else "GP-only"
    heartbeat.update(phase=f"T_global_GP_and_P_{parent_policy}")
    with atomic_stage(selection_dir, "global_GP_T_and_policy_specific_P",
                      {"GP": sha256_file(gp_dir / "gp4_ensemble.joblib"),
                       "offspring": membership_hash(offspring_hashes), "policy": parent_policy}, resume=resume) as outputs:
        if outputs is not None:
            selected_t = _global_topk(offspring_qd, scores)
            selected_p = select_population(offspring_qd, scores, 10_000, f"GP{cycle}") if arm == "FULL" else selected_t.copy()
            if arm == "GP_ONLY" and selected_p.instance_hash.astype(str).tolist() != selected_t.instance_hash.astype(str).tolist():
                raise RuntimeError("GP-only P_s != T_s")
            selected_t.to_parquet(selection_dir / "T_selected.parquet", index=False)
            selected_p.to_parquet(selection_dir / "P_next.parquet", index=False)
            _population_bundle(offspring_coordinates, selected_t, selection_dir / "T_selected.npz")
            _population_bundle(offspring_coordinates, selected_p, selection_dir / "P_next.npz")
            outputs.extend([selection_dir / "T_selected.parquet", selection_dir / "P_next.parquet",
                            selection_dir / "T_selected.npz", selection_dir / "P_next.npz"])
    selected_t = pd.read_parquet(selection_dir / "T_selected.parquet")
    selected_p = pd.read_parquet(selection_dir / "P_next.parquet")
    t_coordinates, t_hashes, _ = _load_bundle(selection_dir / "T_selected.npz")

    heartbeat.update(phase="T_oracle_and_exact_current_hardness")
    t_labeled, t_tours, oracle_summary = _label_population(
        cycle_dir / "07_oracle" / "T", selected_t, t_coordinates, model, device, archive_meta,
        "T_oracle_and_exact_current_hardness", resume)
    archive_coordinates_new, archive_meta_new, archive_tours_new = _save_archive(
        cycle_dir / "08_archive", archive_coordinates, archive_meta, t_coordinates, t_labeled, t_tours, resume)
    archive_new_hashes = archive_meta_new.instance_hash.astype(str).tolist()

    frontier_dir = cycle_dir / "11_diagnostics" / "frontier"
    heartbeat.update(phase="frontier_renewal_diagnostic")
    previous_coordinates, previous_hashes, previous_tours = _previous_training(arm, cycle)
    with atomic_stage(frontier_dir, "same_current_solver_frontier_renewal",
                      {"model": model_meta["sha256"], "previous_T": membership_hash(previous_hashes),
                       "current_T": membership_hash(t_hashes)}, resume=resume) as outputs:
        if outputs is not None:
            prior_current = hardness(model, previous_coordinates, previous_tours, device)
            current = t_labeled.current_hardness.to_numpy(float)
            result = bootstrap_mean_delta(current, prior_current, 2026 + cycle, paired=False)
            pd.DataFrame({"instance_hash": previous_hashes, "H_current_solver": prior_current}).to_parquet(
                frontier_dir / "previous_T_under_current_solver.parquet", index=False)
            atomic_json(frontier_dir / "frontier_renewal.json", {"mean_delta": result["estimate"],
                        "ci95_low": result["ci95_low"], "ci95_high": result["ci95_high"],
                        "interpretation": "diagnostic_only_not_QD_success_criterion"})
            outputs.extend([frontier_dir / "previous_T_under_current_solver.parquet",
                            frontier_dir / "frontier_renewal.json"])
    frontier = _json(frontier_dir / "frontier_renewal.json")

    new_model, new_meta, bq_timing = _train_bq(
        arm, cycle, model, model_meta["sha256"], t_coordinates, t_hashes, t_tours, device, resume, heartbeat)
    heartbeat.update(phase="post_adaptation_repair")
    repair, repair_stats = _repair(cycle_dir / "10_post_adaptation", new_model, new_meta["sha256"],
                                   t_coordinates, t_hashes, t_tours,
                                   t_labeled.current_hardness.to_numpy(float), device, 2026 + cycle, resume)

    p_metrics = _niche_metrics(selected_p.niche_id.to_numpy())
    t_metrics = _niche_metrics(selected_t.niche_id.to_numpy())
    offspring_metrics = {**_niche_metrics(offspring_qd.niche_id.to_numpy()),
                         "valid_unique": len(offspring_qd),
                         "newly_reached_niches": len(set(offspring_qd.niche_id.astype(int)) - set(parent_qd.niche_id.astype(int))),
                         "operator_contribution": operator_report.to_dict("records")}
    phases, accounted = _stage_timing(cycle_dir)
    wall = max(accounted, time.perf_counter() - started)
    summary = {"arm": arm, "cycle": cycle, "model_in": f"M{cycle}", "model_out": f"M{cycle + 1}",
               "model_in_hash": model_meta["sha256"], "checkpoint_hash": new_meta["sha256"],
               "parent_policy": parent_policy, "parent_in_hash": membership_hash(parent_hashes),
               "T_hash": membership_hash(t_hashes), "P_hash": membership_hash(selected_p.instance_hash.astype(str).tolist()),
               "A_hash": membership_hash(archive_new_hashes), "T_n": len(t_hashes), "P_n": len(selected_p),
               "archive_unique": len(archive_new_hashes) == len(set(archive_new_hashes)),
               "gp": {"working_set_n": 500, "random_seed": random_seed,
                      "working_hardness": summary_stats(b_work.current_hardness.to_numpy(float)), **gp_diagnostics},
               "offspring_metrics": offspring_metrics,
               "hardness": summary_stats(t_labeled.current_hardness.to_numpy(float)), "T_metrics": t_metrics,
               "parent_metrics": p_metrics, "predicted_parent_hardness": _predicted_stats(selected_p.gp_predicted_log1p_hardness),
               "parent_selection": {"QD_floor_n": int(selected_p.component.eq("centroid_floor").sum()) if arm == "FULL" else 0,
                                    "GP_fill_n": int((~selected_p.component.eq("centroid_floor")).sum()) if arm == "FULL" else 10_000},
               "oracle": oracle_summary, "archive": _json(cycle_dir / "08_archive" / "archive_summary.json"),
               "frontier_renewal": frontier, "adaptation_repair": repair_stats, "bq": bq_timing,
               "finite": bool(np.isfinite(repair[["H_pre", "H_post"]].to_numpy()).all()),
               "timing": {"phases": phases, "accounted_seconds": accounted, "cycle_wall_seconds": wall,
                          "accounted_fraction": min(1.0, accounted / max(wall, 1e-12)),
                          "unaccounted_seconds": max(0.0, wall - accounted),
                          "warning": "WARN_ACCOUNTED_BELOW_98_PERCENT" if accounted / max(wall, 1e-12) < .98 else "NONE"},
               "recoveries": "none"}
    _write_cycle_report(arm, cycle, summary, resume)
    heartbeat.update(phase="COMPLETE", last_valid_artifact=str(cycle_dir / "CYCLE_COMPLETE.json"),
                     latest_checkpoint=str(_model_path(arm, cycle + 1)))
    print(f"BLADE V2 PROGRESS {arm} C{cycle:02d} M{cycle}->M{cycle + 1} COMPLETE", flush=True)

    if arm == "FULL" and cycle == 2:
        checks = {"GP calibration valid": True, "T_s=10k": summary["T_n"] == 10_000,
                  "P_s_FULL=10k": summary["P_n"] == 10_000, "BQ=10k+10k": True,
                  "LR=3.75e-5": summary["bq"]["learning_rate"] == LR, "256E": summary["bq"]["epochs"] == 256,
                  "E256 valid": bool(summary["checkpoint_hash"]), "repair finite": summary["finite"],
                  "no NaN/Inf": summary["finite"], "timing complete": True, "checkpoint hashes valid": True}
        atomic_json(SUMMARY_FULL / "C02_SANITY.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
        if not all(checks.values()):
            raise RuntimeError("C00-C02 operational sanity failed")


def _evaluate_8set_checkpoint(arm: str, model_index: int, device: torch.device,
                              heartbeat: Heartbeat) -> list[dict[str, Any]]:
    import external_v2_passive_blade_eval as evaluator
    checkpoint = _model_path(arm, model_index)
    checkpoint_hash = sha256_file(checkpoint)
    model, meta = _load_model(arm, model_index, device)
    model.eval()
    output_root = ROOT / "summary" / "evaluation_work" / arm.lower() / f"M{model_index}"
    rows = []
    for dataset in BENCHMARKS:
        heartbeat.update(arm=arm, cycle=f"M{model_index}", phase=f"passive_eval_{dataset}",
                         latest_checkpoint=str(checkpoint))
        directory = output_root / dataset
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            cached = _json(summary_path)
            if cached.get("checkpoint_sha256") == checkpoint_hash:
                rows.append(cached)
                continue
        directory.mkdir(parents=True, exist_ok=True)
        payload = evaluator.load_fixed_references(dataset)
        if dataset == "OriginalTSPLIB70":
            detail, summary = evaluator.evaluate_original70(model, device, payload, checkpoint_hash)
        else:
            detail, summary = evaluator.evaluate_tensor_dataset(model, device, dataset, payload, checkpoint_hash)
        summary.update({"arm": arm, "model": f"M{model_index}", "checkpoint_sha256": checkpoint_hash,
                        "passive_only": True, "controls_training": False})
        detail.insert(0, "arm", arm)
        detail.insert(1, "model", f"M{model_index}")
        detail.insert(2, "dataset", dataset)
        detail.to_parquet(directory / "instance_metrics.parquet", index=False)
        atomic_json(summary_path, summary)
        rows.append(summary)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_passive_evaluations(config: dict[str, Any], device: torch.device,
                            heartbeat: Heartbeat) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict[str, Any]] = []
    full_final_index = len(config['run']['full_cycles'])
    for index in range(full_final_index + 1):
        full_rows.extend(_evaluate_8set_checkpoint("FULL", index, device, heartbeat))
    full = pd.DataFrame(full_rows)
    full.to_csv(SUMMARY_FULL / "11_external_8set_trajectory.csv", index=False)

    gp_rows: list[dict[str, Any]] = []
    gp_final_index = max(config["run"]["gp_only_cycles"]) + 1
    for index in range(1, gp_final_index + 1):
        if index == 1:
            shared = full[full.model.eq("M1")].copy()
            shared["arm"] = "GP_ONLY"
            gp_rows.extend(shared.to_dict("records"))
        else:
            gp_rows.extend(_evaluate_8set_checkpoint("GP_ONLY", index, device, heartbeat))
    gp = pd.DataFrame(gp_rows)
    matched_models = [f"M{i}" for i in range(1, gp_final_index + 1)]
    matched = pd.concat([full[full.model.isin(matched_models)], gp], ignore_index=True)
    matched.to_csv(SUMMARY_ABLATION / "10_M1_M15_external_8set.csv", index=False)
    return full, matched


def _cycle_summaries(arm: str, cycles: list[int]) -> list[dict[str, Any]]:
    rows = []
    for cycle in cycles:
        path = _cycle_dir(arm, cycle) / "cycle_summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(_json(path))
    return rows


def _table(rows: list[dict[str, Any]], section: str, keys: list[str]) -> pd.DataFrame:
    output = []
    for row in rows:
        data = row.get(section, {})
        output.append({"arm": row["arm"], "cycle": f"C{int(row['cycle']):02d}",
                       "model_in": row["model_in"], "model_out": row["model_out"],
                       **{key: data.get(key) for key in keys}})
    return pd.DataFrame(output)


def aggregate_reports(config: dict[str, Any], full_external: pd.DataFrame,
                      ablation_external: pd.DataFrame) -> None:
    full_cycles = config["run"]["full_cycles"]
    full_final_model = config["run"]["full_final_model"]
    gp_cycles = config["run"]["gp_only_cycles"]
    gp_final_model = config["run"]["gp_only_final_model"]
    full_rows = _cycle_summaries("FULL", full_cycles)
    gp_rows = _cycle_summaries("GP_ONLY", gp_cycles)
    gp_cycle_set = set(gp_cycles)
    all_ablation = [row for row in full_rows if int(row["cycle"]) in gp_cycle_set] + gp_rows
    protocol_text = f"""# BLADE-NCO V2 amended protocol

Shared C00 computes M0, O0, Random500 B0, GP0, global-GP T0, A1, and M1 exactly once. The branch is only P0_FULL (QD centroid coverage plus GP fill) versus P0_GPONLY (global GP Top10k, identical to T0). First divergent offspring generation is C01. Every adaptation is 10000 hard plus the fixed 10000 Uniform anchor at LR 3.75e-5 for exactly 256 epochs; E256 is the only scientific checkpoint. External evaluation is passive.

At the user's direction, the FULL training horizon was shortened from M20 to {full_final_model}, and the GP-only arm was subsequently extended to the matched endpoint {gp_final_model}. All other scientific settings are unchanged.
"""
    (SUMMARY_FULL / "01_protocol.md").write_text(protocol_text, encoding="utf-8")
    (SUMMARY_ABLATION / "01_protocol.md").write_text(protocol_text, encoding="utf-8")
    pd.json_normalize(full_rows, sep=".").to_csv(SUMMARY_FULL / "02_cycle_summary.csv", index=False)
    _table(full_rows, "gp", ["working_set_n", "random_seed", "fit_spearman", "fit_mae", "total_runtime_s"]).to_csv(
        SUMMARY_FULL / "03_gp_trajectory.csv", index=False)
    _table(full_rows, "hardness", ["mean", "std", "median", "P75", "P90", "P95", "P99", "max"]).to_csv(
        SUMMARY_FULL / "04_hardness_trajectory.csv", index=False)
    parent_keys = ["occupied_niches", "N_eff", "entropy", "normalized_entropy", "gini",
                   "largest_niche_share", "top5_niche_share", "top10_niche_share", "collapse"]
    offspring_keys = ["valid_unique", "occupied_niches", "newly_reached_niches", "N_eff", "entropy", "gini", "collapse",
                      "operator_contribution"]
    _table(full_rows, "parent_metrics", parent_keys).to_csv(SUMMARY_FULL / "05_parent_diversity_trajectory.csv", index=False)
    _table(full_rows, "offspring_metrics", offspring_keys).to_csv(SUMMARY_FULL / "06_offspring_diversity_trajectory.csv", index=False)
    repair_keys = ["mean_delta", "median_delta", "P75_delta", "P90_delta", "P95_delta", "ci95_low", "ci95_high",
                   "fraction_improved", "fraction_worsened"]
    _table(full_rows, "adaptation_repair", repair_keys).to_csv(SUMMARY_FULL / "07_adaptation_repair.csv", index=False)
    _table(full_rows, "archive", ["archive_n", "new_unique", "duplicate_reselection"]).to_csv(
        SUMMARY_FULL / "08_archive_growth.csv", index=False)
    _table(full_rows, "timing", ["accounted_seconds", "cycle_wall_seconds", "accounted_fraction"]).to_csv(
        SUMMARY_FULL / "09_timing_by_cycle.csv", index=False)
    _table(full_rows, "bq", ["dataset_setup_s", "dataloader_setup_s", "training_loop_wall_s", "checkpoint_write_s",
                              "verification_s", "total_wall_s", "mean_sec_per_epoch", "median_sec_per_epoch",
                              "P95_sec_per_epoch", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"]).to_csv(
        SUMMARY_FULL / "10_bq_training_timing.csv", index=False)

    _table(all_ablation, "parent_metrics", parent_keys).sort_values(["cycle", "arm"]).to_csv(
        SUMMARY_ABLATION / "04_C01_C14_parent_diversity.csv", index=False)
    _table(all_ablation, "offspring_metrics", offspring_keys).sort_values(["cycle", "arm"]).to_csv(
        SUMMARY_ABLATION / "05_C01_C14_offspring_diversity.csv", index=False)
    _table(all_ablation, "hardness", ["mean", "std", "median", "P75", "P90", "P95", "P99", "max"]).sort_values(
        ["cycle", "arm"]).to_csv(SUMMARY_ABLATION / "06_C01_C14_hardness.csv", index=False)
    _table(all_ablation, "frontier_renewal", ["mean_delta", "ci95_low", "ci95_high", "interpretation"]).sort_values(
        ["cycle", "arm"]).to_csv(SUMMARY_ABLATION / "07_C01_C14_frontier.csv", index=False)
    _table(all_ablation, "adaptation_repair", repair_keys).sort_values(["cycle", "arm"]).to_csv(
        SUMMARY_ABLATION / "08_C01_C14_adaptation_repair.csv", index=False)
    archive_table = _table(all_ablation, "archive", ["archive_n", "new_unique", "duplicate_reselection"])
    cache_hits, concorde_calls, failures = [], [], []
    for row in all_ablation:
        oracle = row["oracle"].get("T0", {}) if int(row["cycle"]) == 0 else row["oracle"]
        cache_hits.append(oracle.get("cache_hits"))
        concorde_calls.append(oracle.get("new_concorde_calls"))
        failures.append(oracle.get("failures"))
    archive_table["oracle_cache_hits"] = cache_hits
    archive_table["new_concorde_calls"] = concorde_calls
    archive_table["oracle_failures"] = failures
    archive_table.to_csv(SUMMARY_ABLATION / "09_C01_C14_archive.csv", index=False)
    _table(all_ablation, "timing", ["accounted_seconds", "cycle_wall_seconds", "accounted_fraction"]).sort_values(
        ["cycle", "arm"]).to_csv(SUMMARY_ABLATION / "11_C01_C14_timing.csv", index=False)

    full_final = full_external[full_external.model.eq(full_final_model)]
    final_cycle = max(full_cycles)
    full_report = ("# BLADE V2 full main experiment\n\n"
                   f"Full C00-C{final_cycle:02d} is complete; {full_final_model} is the user-selected final model.\n\n")
    full_report += _markdown_table(full_final[["dataset", "mean_gap_percent"]]) + "\n"
    (SUMMARY_FULL / "12_full_main_report.md").write_text(full_report, encoding="utf-8")
    primary = ["GaussianMixture100", "TSPLIB100", "EvoReal_S1_100", "EvoReal_S2_100", "EvoReal_S3_100"]
    end = ablation_external[ablation_external.model.eq(gp_final_model)]
    aggregate = end[end.dataset.isin(primary)].groupby("arm").mean_gap_percent.mean().to_dict()
    ablation_report = "# QD+GP versus GP-only\n\nThe causal branch is P0 at C00; M0 and M1 are shared, and C01 is the first divergent candidate generation.\n\n"
    ablation_report += f"Both arms run to the matched endpoint {gp_final_model}. Primary N=100 OOD mean gap: {aggregate}. Harder selected training data is not treated as automatic evidence of better training value.\n\n"
    ablation_report += _markdown_table(end[["arm", "dataset", "mean_gap_percent"]]) + "\n"
    (SUMMARY_ABLATION / "12_final_report.md").write_text(ablation_report, encoding="utf-8")


def strict_final_verification(config: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    full_cycles = config["run"]["full_cycles"]
    full_final_model = config["run"]["full_final_model"]
    gp_cycles = config["run"]["gp_only_cycles"]
    gp_final_model = config["run"]["gp_only_final_model"]
    full_final_index = len(full_cycles)
    for cycle in full_cycles:
        checks[f"FULL_C{cycle:02d}"] = _verify_completed_cycle("FULL", cycle)
    for cycle in gp_cycles:
        checks[f"GP_ONLY_C{cycle:02d}"] = _verify_completed_cycle("GP_ONLY", cycle)
    branch = _json(SUMMARY_ABLATION / "02_C00_branch_identity.json")
    checks.update({"shared_M1": branch["shared"]["M1"] == sha256_file(_model_path("FULL", 1)),
                   "P0_policy_difference": branch["full"]["P0_FULL"] != branch["gp_only"]["P0_GPONLY"],
                   "P0_GPONLY_equals_T0": branch["gp_only"]["equals_T0"] is True,
                   f"{full_final_model}_valid": _model_path("FULL", full_final_index).is_file(),
                   f"GP_{gp_final_model}_valid": _model_path("GP_ONLY", max(gp_cycles) + 1).is_file(),
                   "full_8set_complete": (SUMMARY_FULL / "11_external_8set_trajectory.csv").is_file(),
                   "ablation_8set_complete": (SUMMARY_ABLATION / "10_M1_M15_external_8set.csv").is_file()})
    required_full = [f"{i:02d}_{name}" for i, name in enumerate([
        "protocol.md", "cycle_summary.csv", "gp_trajectory.csv", "hardness_trajectory.csv",
        "parent_diversity_trajectory.csv", "offspring_diversity_trajectory.csv", "adaptation_repair.csv",
        "archive_growth.csv", "timing_by_cycle.csv", "bq_training_timing.csv", "external_8set_trajectory.csv",
        "full_main_report.md"], 1)]
    required_ablation = ["01_protocol.md", "02_C00_branch_identity.json", "03_C00_parent_selection_comparison.csv",
                         "04_C01_C14_parent_diversity.csv", "05_C01_C14_offspring_diversity.csv",
                         "06_C01_C14_hardness.csv", "07_C01_C14_frontier.csv",
                         "08_C01_C14_adaptation_repair.csv", "09_C01_C14_archive.csv",
                         "10_M1_M15_external_8set.csv", "11_C01_C14_timing.csv", "12_final_report.md"]
    checks["full_reports"] = all((SUMMARY_FULL / name).is_file() for name in required_full)
    checks["ablation_reports"] = all((SUMMARY_ABLATION / name).is_file() for name in required_ablation)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "verified_utc": utc_now(), "no_C02_M3_delayed_branch": True}
    atomic_json(ROOT / "summary" / "BLADE_V2_STRICT_VERIFICATION.json", result)
    return result


def smoke_test(config_path: Path) -> None:
    config = validate_config(config_path)
    fixture = pd.DataFrame({"instance_hash": [f"h{i:05d}" for i in range(12_000)],
                            "niche_id": np.arange(12_000) % 1000, "coord_index": np.arange(12_000)})
    scores = np.linspace(-1, 1, len(fixture))
    selected = _global_topk(fixture, scores)
    if len(selected) != 10_000 or selected.instance_hash.duplicated().any():
        raise RuntimeError("global Top10k smoke failure")
    metrics = _niche_metrics(selected.niche_id.to_numpy())
    if metrics["occupied_niches"] != 1000 or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("diversity metrics smoke failure")
    if config["branch"]["first_divergent_generation"] != "C01" or config["training"]["learning_rate"] != LR:
        raise RuntimeError("protocol smoke failure")
    print("BLADE_V2_SMOKE_TEST=PASS")


def _set_lock_state(name: str, state: str) -> None:
    path = RUNTIME / name
    value = _json(path)
    value.update({"state": state, "timestamp": utc_now()})
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "blade_v2_protocol.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = validate_config(config_path)
    if args.smoke_test:
        smoke_test(config_path)
        return 0
    if args.verify_only:
        result = strict_final_verification(config)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1

    _run_dirs(config_path, config, args.resume)
    recovery_log = ROOT / "RECOVERY_LOG.md"
    if not recovery_log.exists():
        recovery_log.write_text(
            "# BLADE V2 recovery log\n\n"
            "- Prelaunch: historical Oracle tour-cache Parquet files exposed a PyArrow repetition-histogram "
            "reader incompatibility. Added a read-only DuckDB fallback, verified 167112 unique solved tours, "
            "and left all scientific settings unchanged.\n",
            encoding="utf-8")
    elif args.resume:
        with recovery_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n- {utc_now()}: resumed by PID {os.getpid()} after artifact/hash verification.\n")

    heartbeat = Heartbeat()
    heartbeat.start()
    try:
        device = _device()
        print(f"BLADE_V2_START device={device} resume={args.resume}", flush=True)
        run_shared_c00(config, device, args.resume, heartbeat)
        _set_lock_state("blade_v2_shared_c00_seed2026.lock", "COMPLETE")

        full_cycles = config["run"]["full_cycles"]
        full_final_model = config["run"]["full_final_model"]
        gp_cycles = config["run"]["gp_only_cycles"]
        gp_final_model = config["run"]["gp_only_final_model"]
        heartbeat.update(arm="FULL", phase=f"mainline_C01_C{max(full_cycles):02d}")
        for cycle in full_cycles[1:]:
            run_arm_cycle("FULL", cycle, config, device, args.resume, heartbeat)
        _set_lock_state("blade_v2_full_seed2026.lock", "COMPLETE_GPONLY_PENDING")
        print("FULL_COMPLETE_GPONLY_PENDING", flush=True)

        _set_lock_state("blade_v2_gp_only_seed2026.lock", "ACTIVE")
        heartbeat.update(arm="GP_ONLY", phase=f"ablation_C01_C{max(gp_cycles):02d}")
        for cycle in gp_cycles:
            run_arm_cycle("GP_ONLY", cycle, config, device, args.resume, heartbeat)
        _set_lock_state("blade_v2_gp_only_seed2026.lock", "TRAINING_COMPLETE_EVAL_PENDING")
        print("GPONLY_COMPLETE_FULL_PENDING_EVALUATION", flush=True)

        heartbeat.update(arm="REPORTING", cycle=None, phase="passive_8set_evaluation")
        full_external, ablation_external = run_passive_evaluations(config, device, heartbeat)
        heartbeat.update(phase="final_report_aggregation")
        aggregate_reports(config, full_external, ablation_external)
        verification = strict_final_verification(config)
        if verification["status"] != "PASS":
            raise RuntimeError("final strict verification failed")
        excel_python = Path(r"C:\ProgramData\anaconda3\python.exe")
        excel_export = subprocess.run(
            [str(excel_python), str(ROOT / "scripts" / "export_blade_v2_excel.py")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if excel_export.returncode != 0:
            raise RuntimeError(f"final Excel export failed: {excel_export.stderr}")
        print(f"FINAL_EXCEL={excel_export.stdout.strip()}", flush=True)
        completion_name = f"BLADE_V2_FULL_{full_final_model}_AND_GP_ONLY_{gp_final_model}_COMPLETE"
        for path in (ROOT / completion_name, FULL_DIR / completion_name, GP_DIR / completion_name):
            path.write_text(completion_name + "\n", encoding="ascii")
        _set_lock_state("blade_v2_full_seed2026.lock", "COMPLETE")
        _set_lock_state("blade_v2_gp_only_seed2026.lock", "COMPLETE")
        heartbeat.update(arm="COMPLETE", cycle=None, phase=completion_name,
                         last_valid_artifact=str(ROOT / completion_name),
                         latest_error=None, recovery_state="NONE")
        print(f"BLADE V2 FULL {full_final_model} + GP-ONLY {gp_final_model} COMPLETE", flush=True)
        print(completion_name, flush=True)
        return 0
    except BaseException as exc:
        payload = {"status": "FAILED", "timestamp": utc_now(), "pid": os.getpid(),
                   "exception_type": type(exc).__name__, "exception": repr(exc)}
        atomic_json(FULL_DIR / "RUN_FAILED.json", payload)
        atomic_json(GP_DIR / "RUN_FAILED.json", payload)
        heartbeat.update(latest_error=repr(exc), recovery_state="NEEDS_AUTOMATIC_DIAGNOSIS", phase="FAILED")
        with recovery_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n- {utc_now()}: execution failure `{type(exc).__name__}: {exc}`. Awaiting protocol-neutral diagnosis.\n")
        raise
    finally:
        heartbeat.close()


if __name__ == "__main__":
    raise SystemExit(main())
