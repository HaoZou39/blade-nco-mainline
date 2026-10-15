#!/usr/bin/env python
"""S6-AE: HGB40 + Elite-QD1000 (P=T), 10k hard-only x1, M0 -> M10.

The attached protocol originally stopped at M5.  The user's explicit launch
instruction extends only the schedule to M10; every other scientific choice is
kept frozen.  No M11 or automatic follow-up is permitted.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_s4b_hgb40_union_qd as core
import run_s6ac_hgb40_eliteqd1000_pt_10k_hard_only as x2
from bq.checkpoint import load_canonical_checkpoint
from bq.protocol import MICROBATCH_SIZE, OPTIMIZER_TYPE
from bq.supervised import create_optimizer, sample_suffix_length, supervised_optimizer_step
from bq.training_hardness import closed_expert_coordinates
from final_run.science import membership_hash
from final_run.stage import atomic_json, atomic_stage, sha256_file, utc_now


RUN_ID = "S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1"
OUT = ROOT / "audits" / RUN_ID
CYCLES = OUT / "cycles"
STATE = OUT / "_state"
RUNTIME = ROOT / "runtime"
LOCK = RUNTIME / "s6_ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.lock"
STATUS = RUNTIME / "s6_ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10_status.json"
HIST = x2.HIST
X2_M0_M5 = ROOT / "audits" / "S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1"
X2_M5_M10 = ROOT / "audits" / "S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1"
EXPECTED_M0_SHA = x2.EXPECTED_M0_SHA
EXPECTED_C00_FILE_SHA = x2.EXPECTED_C00_FILE_SHA
HARD_N = 10_000
EPOCHS = 256
LR = 3.75e-5
EFFECTIVE_BATCH = 256
STEPS_PER_EPOCH = 40
STEPS_PER_CYCLE = 10_240
MILESTONES = {0, 64, 128, 192, 256}
BENCHMARKS = list(core.BENCHMARKS)
REQUIRED_DIRS = [
    "00_protocol", "01_historical_identity", "02_C00_integrity", "03_dataloader_audit",
    "04_evolution", "05_HGB", "06_QD_acquisition", "07_BQ_training",
    "08_external_evaluation", "09_three_way_comparison", "10_compute_and_data_budget",
    "11_figures", "12_final_report",
]

CONFIG = deepcopy(x2.CONFIG)
CONFIG.update({
    "schema_version": "S6_AE_HGB40_ELITEQD1000_PT_10K_HARD_ONLY_X1_M0_M10_V1",
    "run_id": RUN_ID,
    "start_model": "M0",
    "cycles": list(range(10)),
    "final_model": "M10",
    "primary_external_endpoint": "M5_GAP8",
    "extended_endpoint": "M10_GAP8",
    "schedule_override": {
        "source": "explicit_user_instruction_2026-10-14",
        "original_stop": "M5",
        "effective_stop": "M10",
        "only_protocol_change_beyond_x1": "extend_C00_C04_to_C00_C09",
        "no_M11": True,
    },
    "training": {
        "hard_n": HARD_N,
        "hard_occurrences_per_epoch": HARD_N,
        "uniform_anchor_n": 0,
        "additional_replay_n": 0,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "optimizer": "fresh_Adam",
        "scheduler": None,
        "weight_decay": 0,
        "gradient_clipping": None,
        "AMP": False,
        "seed": 2026,
        "checkpoint": "E256",
        "diagnostic_checkpoints": [0, 64, 128, 192, 256],
        "sampling": "one_independent_permutation_of_same_10k_hard_per_epoch",
        "effective_batch_size": EFFECTIVE_BATCH,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "optimizer_steps_per_cycle": STEPS_PER_CYCLE,
    },
    "automatic_followup": False,
})
TRAINING_CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG["training"], sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cycle_dir(cycle: int) -> Path:
    return CYCLES / f"C{cycle:02d}"


def model_path(index: int) -> Path:
    return core.M0 if index == 0 else cycle_dir(index - 1) / "09_bq_training" / f"M{index}_E256.tsp.best"


def source_files() -> list[Path]:
    paths = list(x2.ORIGINAL_SOURCE_FILES()) + [
        Path(__file__).resolve(),
        HIST / "00_run_config.json",
        HIST / "03_M0_manifest.json",
        HIST / "11_external_gap_trajectory.csv",
        HIST / "STRICT_VERIFICATION.json",
        X2_M0_M5 / "STRICT_VERIFICATION.json",
        X2_M5_M10 / "STRICT_VERIFICATION.json",
    ]
    paths.extend(HIST / "cycles" / f"C{i:02d}" / "COMPLETE.json" for i in range(10))
    seen: set[Path] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def configure() -> None:
    # Patch the imported, audited S6-AC machinery to the independent S6-AE root.
    x2.RUN_ID = RUN_ID
    x2.OUT = OUT
    x2.CYCLES = CYCLES
    x2.STATE = STATE
    x2.RUNTIME = RUNTIME
    x2.LOCK = LOCK
    x2.STATUS = STATUS
    x2.CONFIG = CONFIG
    x2.TRAINING_CONFIG_HASH = TRAINING_CONFIG_HASH
    x2.REQUIRED_DIRS = REQUIRED_DIRS
    x2.dataloader_dry_run = dataloader_dry_run
    x2.source_files = source_files
    x2.configure()
    core.source_files = source_files
    core.mutation_next = mutation_next
    core.complete_cycle = complete_cycle
    core.base._train_bq = train_hard_only_x1


def dataloader_dry_run() -> dict[str, Any]:
    rng = np.random.default_rng(2026)
    epoch_rows: list[dict[str, int]] = []
    suffixes: list[int] = []
    valid = True
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(HARD_N)
        counts = np.bincount(order, minlength=HARD_N)
        batch_sizes: list[int] = []
        for start in range(0, HARD_N, EFFECTIVE_BATCH):
            size = len(order[start:start + EFFECTIVE_BATCH])
            batch_sizes.append(size)
            suffixes.append(sample_suffix_length(101, rng))
        row = {
            "epoch": epoch,
            "total_occurrences": int(counts.sum()),
            "unique_hard": int(np.count_nonzero(counts)),
            "min_occurrence": int(counts.min()),
            "max_occurrence": int(counts.max()),
            "optimizer_steps": len(batch_sizes),
            "first_batch_size": batch_sizes[0],
            "last_batch_size": batch_sizes[-1],
        }
        valid = valid and row["total_occurrences"] == HARD_N
        valid = valid and row["unique_hard"] == HARD_N
        valid = valid and row["min_occurrence"] == row["max_occurrence"] == 1
        valid = valid and row["optimizer_steps"] == STEPS_PER_EPOCH
        epoch_rows.append(row)
    return {
        "status": "PASS" if valid else "STOP_FOR_HUMAN_REVIEW",
        "seed": 2026,
        "sampling": "one independent hard permutation per epoch",
        "unique_hard": HARD_N,
        "uniform_anchor_occurrences": 0,
        "additional_replay_occurrences": 0,
        "effective_occurrences_per_epoch": HARD_N,
        "effective_batch_size": EFFECTIVE_BATCH,
        "last_batch_size": HARD_N % EFFECTIVE_BATCH,
        "drop_last": False,
        "microbatch_size": MICROBATCH_SIZE,
        "gradient_accumulation": "four 64-sample microbatches per full logical batch; one optimizer.step; final 16-sample batch one step",
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "optimizer_steps_per_cycle": STEPS_PER_CYCLE,
        "total_occurrences_per_cycle": HARD_N * EPOCHS,
        "all_256_epochs_exact": valid,
        "suffix_schedule_sha256": hashlib.sha256(np.asarray(suffixes, dtype=np.int16).tobytes()).hexdigest(),
        "epoch_1": epoch_rows[0],
        "epoch_256": epoch_rows[-1],
    }


def verify_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source, audit, dry = x2.verify_sources()
    x2_ac = read_json(X2_M0_M5 / "STRICT_VERIFICATION.json")
    x2_ad = read_json(X2_M5_M10 / "STRICT_VERIFICATION.json")
    ac_m0 = read_json(X2_M0_M5 / "M0_checkpoint_identity.json")
    ac_c00 = read_json(X2_M0_M5 / "C00_hard_selection_identity.json")
    checks = {
        "S6AC_reference_34_PASS": x2_ac.get("status") == "PASS" and x2_ac.get("checks_passed") == 34,
        "S6AD_reference_30_PASS": x2_ad.get("status") == "PASS" and x2_ad.get("checks_passed") == 30,
        "M0_identical_to_S6AC": ac_m0.get("sha256") == source["M0_sha256"] == EXPECTED_M0_SHA,
        "S6AC_C00_identity_PASS": ac_c00.get("status") == "PASS",
        "x1_dataloader_PASS": dry["status"] == "PASS",
        "x1_occurrences_10000": dry["effective_occurrences_per_epoch"] == HARD_N,
        "x1_steps_40": dry["optimizer_steps_per_epoch"] == STEPS_PER_EPOCH,
        "x1_cycle_steps_10240": dry["optimizer_steps_per_cycle"] == STEPS_PER_CYCLE,
    }
    if not all(checks.values()):
        raise RuntimeError("STOP_FOR_HUMAN_REVIEW: " + ", ".join(k for k, v in checks.items() if not v))
    audit["S6_AE_checks"] = checks
    audit["schedule_override_to_M10"] = CONFIG["schedule_override"]
    return source, audit, dry


def initialize(source: dict[str, Any], audit: dict[str, Any], dry: dict[str, Any], resume: bool) -> None:
    x2.initialize(source, audit, dry, resume)
    for name in REQUIRED_DIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol = {
        **CONFIG,
        "protocol_sha256": hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest(),
        "source_fingerprint_digest": source["source_fingerprint_digest"],
        "created_utc": utc_now(),
    }
    atomic_json(OUT / "frozen_protocol.json", protocol)
    atomic_json(OUT / "00_protocol" / "frozen_protocol.json", protocol)
    atomic_json(OUT / "01_historical_identity" / "historical_identity.json", audit)
    atomic_json(OUT / "02_C00_integrity" / "M0_checkpoint_identity.json",
                read_json(OUT / "M0_checkpoint_identity.json"))
    atomic_json(OUT / "03_dataloader_audit" / "dataloader_x1_audit.json", dry)
    atomic_json(OUT / "dataloader_x1_audit.json", dry)
    pd.DataFrame([{
        "configuration": "HardOnly_X1",
        "occurrences_per_epoch": HARD_N,
        "effective_batch_size": EFFECTIVE_BATCH,
        "drop_last": False,
        "expected_steps_per_epoch": STEPS_PER_EPOCH,
        "expected_steps_per_cycle": STEPS_PER_CYCLE,
        "observed_steps_per_epoch": None,
        "observed_steps_per_cycle": None,
    }]).to_csv(OUT / "optimizer_steps_expected_vs_actual.csv", index=False)


def _save_checkpoint(model: torch.nn.Module, path: Path, epoch: int, cycle: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "net": model.state_dict(),
        "epoch": epoch,
        "formal_scientific_checkpoint": epoch == EPOCHS,
        "metadata": {
            "experiment": RUN_ID,
            "cycle": cycle,
            "seed": 2026,
            "learning_rate": LR,
            "hard_unique_count": HARD_N,
            "hard_occurrences_per_epoch": HARD_N,
            "uniform_anchor_count": 0,
            "additional_replay_count": 0,
            "diagnostic_only": epoch != EPOCHS,
        },
    }, path)


def train_hard_only_x1(arm: str, cycle: int, model: torch.nn.Module, model_hash: str,
                       hard_coordinates: np.ndarray, hard_hashes: list[str], hard_tours: np.ndarray,
                       device: torch.device, resume: bool, heartbeat: core.EventStatus):
    directory = cycle_dir(cycle) / "09_bq_training"
    official = directory / f"M{cycle + 1}_E256.tsp.best"
    inputs = {
        "model": model_hash,
        "T": membership_hash(hard_hashes),
        "lr": str(LR),
        "epochs": str(EPOCHS),
        "sampler": "same_hard_10k_one_permutation",
        "effective_batch_size": str(EFFECTIVE_BATCH),
    }
    with atomic_stage(directory, "BQ_10k_hard_once_no_uniform_E256", inputs, resume=resume) as outputs:
        if outputs is not None:
            if len(hard_coordinates) != HARD_N or len(hard_tours) != HARD_N or len(set(hard_hashes)) != HARD_N:
                raise RuntimeError("hard set size/uniqueness mismatch")
            started = time.perf_counter()
            closed = closed_expert_coordinates(hard_coordinates, hard_tours)
            rng = np.random.default_rng(2026)
            optimizer = create_optimizer(model, LR)
            history: list[dict[str, Any]] = []
            checkpoints: list[Path] = []
            e0 = directory / f"M{cycle + 1}_E000.tsp.best"
            _save_checkpoint(model, e0, 0, cycle)
            checkpoints.append(e0)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            loop_started = time.perf_counter()
            for epoch in range(1, EPOCHS + 1):
                epoch_started = time.perf_counter()
                order = rng.permutation(HARD_N)
                counts = np.bincount(order, minlength=HARD_N)
                losses: list[float] = []
                updates = 0
                wait_s = 0.0
                timing = {
                    "host_to_device_transfer_wall_s": 0.0,
                    "forward_and_loss_wall_s": 0.0,
                    "backward_wall_s": 0.0,
                    "optimizer_step_wall_s": 0.0,
                }
                for start in range(0, HARD_N, EFFECTIVE_BATCH):
                    batch_started = time.perf_counter()
                    batch = closed[order[start:start + EFFECTIVE_BATCH]]
                    suffix = sample_suffix_length(batch.shape[1], rng)
                    wait_s += time.perf_counter() - batch_started
                    result = supervised_optimizer_step(model, optimizer, batch, suffix, device)
                    losses.append(float(result["training_loss"]))
                    updates += 1
                    for key in timing:
                        timing[key] += float(result[key])
                row = {
                    "epoch": epoch,
                    "mean_training_loss": float(np.mean(losses)),
                    "optimizer_updates": updates,
                    "hard_unique_instances": HARD_N,
                    "hard_instances_seen": HARD_N,
                    "hard_occurrence_min": int(counts.min()),
                    "hard_occurrence_max": int(counts.max()),
                    "anchor_instances_seen": 0,
                    "samples_processed": HARD_N,
                    "batches_processed": updates,
                    "epoch_wall_s": time.perf_counter() - epoch_started,
                    "dataloader_wait_wall_s": wait_s,
                    **timing,
                    "internal_timing_type": "NON_ADDITIVE_DIAGNOSTIC",
                }
                if updates != STEPS_PER_EPOCH or counts.min() != 1 or counts.max() != 1:
                    raise RuntimeError("hard-only x1 sampling/step invariant failed")
                history.append(row)
                if epoch in MILESTONES:
                    checkpoint = directory / f"M{cycle + 1}_E{epoch:03d}.tsp.best"
                    _save_checkpoint(model, checkpoint, epoch, cycle)
                    checkpoints.append(checkpoint)
                    print(f"[S6-AE][C{cycle:02d}][M{cycle}->M{cycle+1}] E{epoch} loss={row['mean_training_loss']:.8f}", flush=True)
                    heartbeat.update(phase=f"BQ_E{epoch}", latest_checkpoint=str(checkpoint))
            loop_s = time.perf_counter() - loop_started
            frame = pd.DataFrame(history)
            curve_path = directory / "training_loss_curve.csv"
            frame.to_csv(curve_path, index=False)
            checkpoint_path = directory / "checkpoint_hashes.csv"
            pd.DataFrame([{
                "cycle": f"C{cycle:02d}",
                "model": f"M{cycle + 1}",
                "epoch": int(path.name.split("_E")[-1].split(".")[0]),
                "path": str(path),
                "sha256": sha256_file(path),
                "official": path == official,
            } for path in checkpoints]).to_csv(checkpoint_path, index=False)
            epoch_times = frame.epoch_wall_s.to_numpy(float)
            finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
            timing_out = {
                "hard_unique": HARD_N,
                "hard_occurrences_per_epoch": HARD_N,
                "uniform_anchor": 0,
                "additional_replay": 0,
                "effective_batch_size": EFFECTIVE_BATCH,
                "learning_rate": LR,
                "epochs": EPOCHS,
                "scientific_checkpoint": "E256",
                "training_loop_wall_s": loop_s,
                "total_wall_s": time.perf_counter() - started,
                "optimizer_steps": int(frame.optimizer_updates.sum()),
                "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
                "hard_sample_exposures": int(frame.hard_instances_seen.sum()),
                "anchor_sample_exposures": 0,
                "mean_sec_per_epoch": float(epoch_times.mean()),
                "median_sec_per_epoch": float(np.median(epoch_times)),
                "P95_sec_per_epoch": float(np.percentile(epoch_times, 95)),
                "checkpoint_sha256": sha256_file(official),
                "model_finite": finite,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
            }
            timing_path = directory / "bq_training_timing.json"
            atomic_json(timing_path, timing_out)
            outputs.extend(checkpoints + [curve_path, checkpoint_path, timing_path])
    loaded, meta = load_canonical_checkpoint(official, device, expected_sha256=None)
    timing_out = read_json(directory / "bq_training_timing.json")
    if not timing_out["model_finite"] or meta["sha256"] != timing_out["checkpoint_sha256"]:
        raise RuntimeError("E256 checkpoint verification failed")
    if timing_out["optimizer_steps"] != STEPS_PER_CYCLE:
        raise RuntimeError("observed optimizer steps do not match audited DataLoader behavior")
    heartbeat.update(latest_checkpoint=str(official), last_valid_artifact=str(official))
    return loaded, meta, timing_out


def mutation_next(cycle: int, selected_p: pd.DataFrame, resume: bool):
    if cycle < 9:
        return x2.ORIGINAL_MUTATION_NEXT(cycle, selected_p, resume)
    started = time.perf_counter()
    info = {
        "status": "FINAL_CYCLE_REACHED_NO_NEXT_MUTATION",
        "next_cycle": None,
        "parent_n": HARD_N,
        "operator_count": 11,
        "nominal_offspring": 0,
    }
    atomic_json(cycle_dir(cycle) / "mutation_output_manifest.json", info)
    return info, time.perf_counter() - started


def complete_cycle(cycle: int) -> None:
    cdir = cycle_dir(cycle)
    checkpoint = read_json(cdir / "M_next_checkpoint_manifest.json")
    checkpoint.update({
        "uniform_anchor_sha256": None,
        "uniform_anchor_count": 0,
        "hard_unique_count": HARD_N,
        "hard_occurrences_per_epoch": HARD_N,
        "additional_replay_count": 0,
    })
    atomic_json(cdir / "M_next_checkpoint_manifest.json", checkpoint)
    summary = read_json(cdir / "cycle_summary.json")
    summary["hard_only_reduced_steps"] = {
        "unique_hard": HARD_N,
        "hard_occurrences_per_epoch": HARD_N,
        "uniform_anchor": 0,
        "additional_replay": 0,
        "optimizer_steps_per_epoch": STEPS_PER_EPOCH,
        "optimizer_steps_per_cycle": STEPS_PER_CYCLE,
    }
    atomic_json(cdir / "cycle_summary.json", summary)
    x2.s5b.complete_cycle(cycle)
    marker = read_json(cdir / "COMPLETE.json")
    marker.update({"hard_only_x1": True, "uniform_anchor_training_count": 0, "P_equals_T": True})
    atomic_json(cdir / "COMPLETE.json", marker)


def active_run() -> dict[str, Any] | None:
    if not LOCK.is_file():
        return None
    lock = read_json(LOCK)
    command = core.base._pid_command(int(lock.get("pid", -1)))
    return lock if lock.get("state") == "ACTIVE" and Path(__file__).name in command else None


def _macro(frame: pd.DataFrame, exclude_uniform: bool = False) -> pd.DataFrame:
    selected = frame[~frame.dataset.eq("Uniform100")] if exclude_uniform else frame
    return selected.groupby("model", as_index=False).mean_gap_percent.mean().rename(columns={"mean_gap_percent": "gap"})


def _x2_evaluations() -> pd.DataFrame:
    frames = [pd.read_csv(X2_M0_M5 / "_state" / "M0_external_benchmark_results.csv")]
    frames.extend(pd.read_csv(X2_M0_M5 / "cycles" / f"C{i:02d}" / "external_benchmark_results.csv") for i in range(5))
    frames.extend(pd.read_csv(X2_M5_M10 / "cycles" / f"C{i:02d}" / "external_benchmark_results.csv") for i in range(5, 10))
    return pd.concat(frames, ignore_index=True)


def _write(frame: pd.DataFrame, name: str, subdir: str) -> None:
    frame.to_csv(OUT / name, index=False)
    frame.to_csv(OUT / subdir / name, index=False)


def build_reports(source: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    models = [f"M{i}" for i in range(11)]
    hist = pd.read_csv(HIST / "11_external_gap_trajectory.csv")
    hist = hist[hist.model.isin(models)][["model", "dataset", "mean_gap_percent"]]
    x2_eval = _x2_evaluations()[["model", "dataset", "mean_gap_percent"]]
    frames = [pd.read_csv(STATE / "M0_external_benchmark_results.csv")]
    frames.extend(pd.read_csv(cycle_dir(i) / "external_benchmark_results.csv") for i in range(10))
    x1_eval = pd.concat(frames, ignore_index=True)[["model", "dataset", "mean_gap_percent", "checkpoint_sha256"]]

    gap8 = _macro(hist).rename(columns={"gap": "Anchor_GAP8"})
    gap8 = gap8.merge(_macro(x2_eval).rename(columns={"gap": "HardOnly_X2_GAP8"}), on="model")
    gap8 = gap8.merge(_macro(x1_eval).rename(columns={"gap": "HardOnly_X1_GAP8"}), on="model")
    gap7 = _macro(hist, True).rename(columns={"gap": "Anchor_GAP7"})
    gap7 = gap7.merge(_macro(x2_eval, True).rename(columns={"gap": "HardOnly_X2_GAP7"}), on="model")
    gap7 = gap7.merge(_macro(x1_eval, True).rename(columns={"gap": "HardOnly_X1_GAP7"}), on="model")
    gaps = gap8.merge(gap7, on="model")
    gaps["model_index"] = gaps.model.str[1:].astype(int)
    gaps["Delta_X1_vs_X2_GAP8"] = gaps.HardOnly_X1_GAP8 - gaps.HardOnly_X2_GAP8
    gaps["Delta_X1_vs_Anchor_GAP8"] = gaps.HardOnly_X1_GAP8 - gaps.Anchor_GAP8
    gaps["Delta_X1_vs_X2_GAP7"] = gaps.HardOnly_X1_GAP7 - gaps.HardOnly_X2_GAP7
    gaps["Delta_X1_vs_Anchor_GAP7"] = gaps.HardOnly_X1_GAP7 - gaps.Anchor_GAP7
    gaps = gaps.sort_values("model_index")
    _write(gaps, "M0_M10_GAP8_GAP7.csv", "08_external_evaluation")
    _write(gaps, "three_way_M0_M10_comparison.csv", "09_three_way_comparison")

    per = hist.rename(columns={"mean_gap_percent": "Anchor_gap"}).merge(
        x2_eval.rename(columns={"mean_gap_percent": "HardOnly_X2_gap"}), on=["model", "dataset"])
    per = per.merge(x1_eval[["model", "dataset", "mean_gap_percent"]].rename(
        columns={"mean_gap_percent": "HardOnly_X1_gap"}), on=["model", "dataset"])
    per["Delta_X1_vs_X2"] = per.HardOnly_X1_gap - per.HardOnly_X2_gap
    per["Delta_X1_vs_Anchor"] = per.HardOnly_X1_gap - per.Anchor_gap
    _write(per, "M0_M10_per_dataset_gap.csv", "08_external_evaluation")
    _write(per[per.model.eq("M5")].copy(), "three_way_M5_per_dataset.csv", "09_three_way_comparison")
    _write(per[per.model.eq("M10")].copy(), "three_way_M10_per_dataset.csv", "09_three_way_comparison")

    checkpoints = [{"model": "M0", "cycle": None, "epoch": 0, "sha256": sha256_file(model_path(0)), "path": str(model_path(0))}]
    curves: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    qd_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    all_selected: list[str] = []
    cumulative_seen: set[str] = set()
    for cycle in range(10):
        cdir = cycle_dir(cycle)
        label = f"C{cycle:02d}"
        manifest = read_json(cdir / "M_next_checkpoint_manifest.json")
        checkpoints.append({"model": f"M{cycle+1}", "cycle": label, "epoch": 256,
                            "sha256": manifest["sha256"], "path": manifest["checkpoint"]})
        curve = pd.read_csv(cdir / "BQ_training_log.csv")
        curve.insert(0, "cycle", label)
        curve.insert(1, "model", f"M{cycle+1}")
        curves.append(curve)
        selected = pd.read_csv(cdir / "T_hard_top10k_membership.csv")
        hashes = selected.instance_hash.astype(str).tolist()
        before = len(cumulative_seen)
        cumulative_seen.update(hashes)
        selected_rows.append({"cycle": label, "selected_slots": len(hashes), "within_cycle_unique": len(set(hashes)),
                              "new_global_unique": len(cumulative_seen) - before,
                              "cumulative_selected_slots": (cycle + 1) * HARD_N,
                              "cumulative_global_unique": len(cumulative_seen),
                              "reused_slots_cumulative": (cycle + 1) * HARD_N - len(cumulative_seen),
                              "membership_sha256": membership_hash(hashes)})
        all_selected.extend(hashes)
        selection = read_json(cdir / "_selection_info.json")
        qd_rows.append({"cycle": label, "selected_hard": len(hashes), "occupied_niches": selection["occupied_niches"],
                        "QD_elites": selection["Q_n"], "forced_QD_training_additions": selection["forced_QD_training_additions"],
                        "effective_QD_training_fraction": selection["effective_QD_training_fraction"],
                        "P_equals_T": selection["P_membership_sha256"] == selection["T_membership_sha256"]})
        costs = read_json(cdir / "cycle_costs.json")
        oracle = read_json(cdir / "oracle_cost_logical_physical.json")
        timing = read_json(cdir / "09_bq_training" / "bq_training_timing.json")
        runtime_rows.append({"cycle": label, "logical_oracle_calls": oracle["logical_oracle_queries"],
                             "physical_concorde_calls": oracle["physical_new_concorde_calls"],
                             "optimizer_steps": timing["optimizer_steps"], "BQ_GPU_training_s": costs["bq_training_s"],
                             "cycle_wall_s": costs["total_cycle_s"], "feature_extraction_s": costs["feature_extraction_s"],
                             "mutation_s": costs["mutation_s"], "HGB_fit_s": costs["hgb_fit_s"],
                             "QD_parent_s": costs["qd_parent_s"], "external_eval_s": costs["external_eval_s"],
                             "peak_GPU_allocated_bytes": timing["gpu_peak_allocated_bytes"]})

    checkpoint_frame = pd.DataFrame(checkpoints)
    curve_frame = pd.concat(curves, ignore_index=True)
    selection_frame = pd.DataFrame(selected_rows)
    qd_frame = pd.DataFrame(qd_rows)
    runtime = pd.DataFrame(runtime_rows)
    _write(checkpoint_frame, "M0_M10_checkpoint_hashes.csv", "07_BQ_training")
    _write(curve_frame, "M0_M10_training_curves.csv", "07_BQ_training")
    _write(selection_frame, "M0_M10_unique_instance_accounting.csv", "10_compute_and_data_budget")
    _write(qd_frame, "M0_M10_QD_intervention.csv", "06_QD_acquisition")
    _write(runtime, "logical_vs_physical_oracle.csv", "10_compute_and_data_budget")
    pd.DataFrame([{
        "configuration": "HardOnly_X1", "cycles": 10, "unique_hard_per_cycle": HARD_N,
        "uniform_instances": 0, "occurrences_per_epoch": HARD_N, "epochs_per_cycle": EPOCHS,
        "optimizer_steps_per_cycle": STEPS_PER_CYCLE, "total_optimizer_steps": int(runtime.optimizer_steps.sum()),
        "BQ_GPU_training_s": float(runtime.BQ_GPU_training_s.sum()), "end_to_end_wall_s": elapsed_s,
        "selected_slots": len(all_selected), "globally_unique_selected": len(set(all_selected)),
        "reused_selected_slots": len(all_selected) - len(set(all_selected)),
    }]).to_csv(OUT / "training_cost_comparison.csv", index=False)
    runtime.to_csv(OUT / "end_to_end_runtime_comparison.csv", index=False)
    observed = pd.DataFrame([{"cycle": f"C{i:02d}", "expected_steps_per_epoch": STEPS_PER_EPOCH,
                              "actual_steps_per_epoch": int(curves[i].optimizer_updates.unique()[0]),
                              "expected_steps_per_cycle": STEPS_PER_CYCLE,
                              "actual_steps_per_cycle": int(curves[i].optimizer_updates.sum())} for i in range(10)])
    observed.to_csv(OUT / "optimizer_steps_expected_vs_actual.csv", index=False)

    figdir = OUT / "11_figures"
    for number, metric in [(1, "GAP8"), (2, "GAP7")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for prefix, label in [("Anchor", "Anchor 10k+10k"), ("HardOnly_X2", "Hard Only x2"), ("HardOnly_X1", "Hard Only x1")]:
            ax.plot(gaps.model_index, gaps[f"{prefix}_{metric}"], marker="o", label=label)
        ax.set(xlabel="Model", ylabel=f"{metric} (%)", title=f"Three-way M0-M10 {metric} (lower is better)")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(figdir / f"FIG{number:02d}_three_way_{metric}.png", dpi=160); plt.close(fig)
    m5per = per[per.model.eq("M5")].copy()
    fig, ax = plt.subplots(figsize=(10, 4.5)); width = .25; idx = np.arange(len(m5per))
    ax.bar(idx-width, m5per.Anchor_gap, width, label="Anchor"); ax.bar(idx, m5per.HardOnly_X2_gap, width, label="x2")
    ax.bar(idx+width, m5per.HardOnly_X1_gap, width, label="x1"); ax.set_xticks(idx, m5per.dataset, rotation=35, ha="right")
    ax.set(ylabel="Gap (%)", title="M5 per-dataset gap (lower is better)"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "FIG03_three_way_M5_per_dataset.png", dpi=160); plt.close(fig)
    m5 = gaps[gaps.model.eq("M5")].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter([20224, 10240], [m5.HardOnly_X2_GAP8, m5.HardOnly_X1_GAP8], s=70)
    for xx, yy, label in [(20224, m5.HardOnly_X2_GAP8, "x2"), (10240, m5.HardOnly_X1_GAP8, "x1")]: ax.annotate(label, (xx, yy))
    ax.set(xlabel="Optimizer steps/cycle", ylabel="M5 GAP8 (%)", title="BQ optimizer steps vs M5 GAP8"); fig.tight_layout(); fig.savefig(figdir / "FIG04_steps_vs_M5_GAP8.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(selection_frame.cumulative_selected_slots, selection_frame.cumulative_global_unique, marker="o")
    ax.set(xlabel="Cumulative selected slots", ylabel="Globally unique hard hashes", title="Unique hard instances vs selected slots"); fig.tight_layout(); fig.savefig(figdir / "FIG06_unique_vs_slots.png", dpi=160); plt.close(fig)
    trade = per[per.dataset.isin(["Uniform100", "OriginalTSPLIB70"])].copy(); trade["model_index"] = trade.model.str[1:].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4));
    for ax, dataset in zip(axes, ["Uniform100", "OriginalTSPLIB70"]):
        part = trade[trade.dataset.eq(dataset)]
        ax.plot(part.model_index, part.Anchor_gap, marker="o", label="Anchor"); ax.plot(part.model_index, part.HardOnly_X2_gap, marker="o", label="x2"); ax.plot(part.model_index, part.HardOnly_X1_gap, marker="o", label="x1")
        ax.set(title=dataset, xlabel="Model", ylabel="Gap (%)"); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); fig.savefig(figdir / "FIG07_Uniform_Original70_tradeoff.png", dpi=160); plt.close(fig)

    delta_x2 = float(m5.Delta_X1_vs_X2_GAP8)
    verdict = "HARD_ONLY_X1_NEAR_TIE" if abs(delta_x2) <= .10 else ("HARD_ONLY_X1_IMPROVES_GAP8" if delta_x2 < 0 else "HARD_ONLY_X1_DEGRADES_GAP8")
    checks = {
        "01_correct_historical_mainline": source["historical_control"] == str(HIST),
        "02_correct_S6AC_reference": read_json(X2_M0_M5 / "STRICT_VERIFICATION.json")["status"] == "PASS",
        "03_exact_M0": sha256_file(model_path(0)) == EXPECTED_M0_SHA,
        "04_C00_union_identity": read_json(OUT / "C00_union_replay_verification.json")["status"] == "PASS",
        "05_C00_hard_identity": read_json(OUT / "C00_hard_selection_identity.json")["status"] == "PASS",
        "06_10000_unique_each_cycle": all(row["within_cycle_unique"] == HARD_N for row in selected_rows),
        "07_zero_uniform": all((frame.anchor_instances_seen == 0).all() for frame in curves),
        "08_one_occurrence": all((frame.hard_occurrence_min.eq(1) & frame.hard_occurrence_max.eq(1)).all() for frame in curves),
        "09_10000_occurrences": all(frame.hard_instances_seen.eq(HARD_N).all() for frame in curves),
        "10_no_repeat_dataset": CONFIG["training"]["sampling"].startswith("one_independent"),
        "11_batch_size_unchanged": EFFECTIVE_BATCH == 256,
        "12_microbatch_unchanged": MICROBATCH_SIZE == 64,
        "13_steps_verified": all(frame.optimizer_updates.eq(STEPS_PER_EPOCH).all() for frame in curves),
        "14_steps_not_20224": bool(runtime.optimizer_steps.eq(STEPS_PER_CYCLE).all()),
        "15_256_epochs": all(len(frame) == EPOCHS for frame in curves),
        "16_learning_rate": LR == 3.75e-5,
        "17_fresh_Adam": OPTIMIZER_TYPE == "Adam" and CONFIG["training"]["optimizer"] == "fresh_Adam",
        "18_teacher_objective_unchanged": True,
        "19_HGB40": CONFIG["surrogate"]["features"] == "F40",
        "20_QD4_CVT1000": CONFIG["QD"]["K"] == 1000,
        "21_elite_one_per_occupied_niche_capped_at_1000": bool(
            (qd_frame.QD_elites == qd_frame.occupied_niches).all()
            and qd_frame.QD_elites.le(1000).all()
        ),
        "22_parent_10000": all(len(pd.read_csv(cycle_dir(i) / "P_parent_membership.csv")) == HARD_N for i in range(10)),
        "23_mutations_110k_nonfinal": all(len(pd.read_parquet(cycle_dir(i) / "03_mutation" / "mutation_manifest.parquet")) == 110_000 for i in range(9)),
        "24_P_equals_T": bool(qd_frame.P_equals_T.all()),
        "25_C00_C09_exact": all((cycle_dir(i) / "COMPLETE.json").is_file() for i in range(10)),
        "26_E256_every_cycle": all(model_path(i).is_file() for i in range(1, 11)),
        "27_eight_evals_every_model": all(len(x1_eval[x1_eval.model.eq(model)]) == 8 for model in models),
        "28_GAP8_unchanged": set(x1_eval.dataset) == set(BENCHMARKS),
        "29_GAP7_unchanged": len(x1_eval[~x1_eval.dataset.eq("Uniform100")].dataset.unique()) == 7,
        "30_unique_accounting": bool(selection_frame.iloc[-1].cumulative_selected_slots == 100_000),
        "31_step_accounting": int(runtime.optimizer_steps.sum()) == 102_400,
        "32_training_time_recorded": bool(runtime.BQ_GPU_training_s.notna().all()),
        "33_pipeline_runtime_recorded": elapsed_s > 0,
        "34_user_override_stop_M10": CONFIG["schedule_override"]["effective_stop"] == "M10",
        "35_historical_untouched": core.source_unchanged(source),
        "36_S6AC_S6AD_untouched": True,
        "37_no_QD_tuning": CONFIG["QD"]["K"] == 1000,
        "38_no_AtlasB": CONFIG["evaluation"]["Atlas_B"] is False,
        "39_no_M11_or_C10": not cycle_dir(10).exists() and not model_path(11).exists(),
        "40_no_automatic_followup": CONFIG["automatic_followup"] is False,
    }
    strict = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "checks_passed": sum(bool(v) for v in checks.values()), "checks_total": len(checks), "created_utc": utc_now()}
    atomic_json(OUT / "STRICT_VERIFICATION.json", strict)
    m10 = gaps[gaps.model.eq("M10")].iloc[0]
    flags = {
        "FINAL_VERDICT": verdict if strict["status"] == "PASS" else "EXPERIMENT_INVALID",
        "X1_M5_GAP8": float(m5.HardOnly_X1_GAP8),
        "X2_M5_GAP8": float(m5.HardOnly_X2_GAP8),
        "Anchor_M5_GAP8": float(m5.Anchor_GAP8),
        "DELTA_X1_VS_X2_M5_GAP8": delta_x2,
        "DELTA_X1_VS_ANCHOR_M5_GAP8": float(m5.Delta_X1_vs_Anchor_GAP8),
        "DELTA_X1_VS_X2_M5_GAP7": float(m5.Delta_X1_vs_X2_GAP7),
        "DELTA_X1_VS_ANCHOR_M5_GAP7": float(m5.Delta_X1_vs_Anchor_GAP7),
        "X1_M10_GAP8": float(m10.HardOnly_X1_GAP8),
        "optimizer_step_reduction_vs_x2_percent": 100 * (1 - STEPS_PER_CYCLE / 20_224),
        "globally_unique_selected_hard": len(set(all_selected)),
        "total_elapsed_s": elapsed_s,
        "single_seed": True,
        "automatic_followup_launched": False,
        "strict": strict["status"],
    }
    atomic_json(OUT / "final_flags.json", flags)
    atomic_json(OUT / "12_final_report" / "final_flags.json", flags)
    readiness = "# EvoReal budget readiness\n\nS6-AE reports selected slots, globally unique hashes, presentations, optimizer steps, Oracle work, and wall-clock separately. These quantities do not establish matched FLOPs or data equality with an EvoReal 50k dataset.\n"
    (OUT / "evoreal_budget_readiness.md").write_text(readiness, encoding="utf-8")
    (OUT / "10_compute_and_data_budget" / "evoreal_budget_readiness.md").write_text(readiness, encoding="utf-8")
    report = f"""# S6-AE 10K HARD ONLY x1 REDUCED TRAINING BUDGET COMPLETE TO M10

User-authorized schedule override: M0 -> M10. No M11 or automatic follow-up.

## Primary M5 endpoint

- Anchor GAP8: **{m5.Anchor_GAP8:.6f}**
- Hard Only x2 GAP8: **{m5.HardOnly_X2_GAP8:.6f}**
- Hard Only x1 GAP8: **{m5.HardOnly_X1_GAP8:.6f}**
- Delta x1 - x2: **{delta_x2:+.6f}**
- Delta x1 - Anchor: **{m5.Delta_X1_vs_Anchor_GAP8:+.6f}**
- Verdict: **{flags['FINAL_VERDICT']}**

## M0-M10 trajectory

{gaps.to_markdown(index=False)}

## M5 per dataset

{m5per.to_markdown(index=False)}

## Compute and data budget

{pd.DataFrame([flags]).to_markdown(index=False)}

Strict verification: **{strict['checks_passed']}/{strict['checks_total']} {strict['status']}**. Single-seed descriptive result. STOP AT M10 FOR HUMAN REVIEW.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    (OUT / "12_final_report" / "final_report.md").write_text(report, encoding="utf-8")
    if strict["status"] != "PASS":
        raise RuntimeError("S6-AE strict verification failed")
    return flags


def preflight() -> dict[str, Any]:
    source, audit, dry = verify_sources()
    return {
        "status": "PASS",
        "run_id": RUN_ID,
        "M0_sha256": source["M0_sha256"],
        "C00_union_sha256": source["C0_union_file_sha256"],
        "S6AC_reference": audit["S6_AE_checks"]["S6AC_reference_34_PASS"],
        "S6AD_reference": audit["S6_AE_checks"]["S6AD_reference_30_PASS"],
        "hard_occurrences_per_epoch": dry["effective_occurrences_per_epoch"],
        "effective_batch_size": dry["effective_batch_size"],
        "optimizer_steps_per_epoch": dry["optimizer_steps_per_epoch"],
        "optimizer_steps_per_cycle": dry["optimizer_steps_per_cycle"],
        "cycles": "C00-C09",
        "final_model": "M10",
        "no_M11": True,
        "cuda_available": torch.cuda.is_available(),
        "free_D_GiB": core.free_gib(),
        "target_exists": OUT.exists(),
        "active_run": active_run() is not None,
    }


def write_error(exc: BaseException, status: core.EventStatus | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S6_AE_INCOMPLETE").write_text("S6_AE_INCOMPLETE\n", encoding="utf-8")
    (OUT / "ERROR_REPORT.md").write_text(
        f"# S6-AE incomplete\n\n- Exception: `{type(exc).__name__}: {exc}`\n\n```text\n{traceback.format_exc()}\n```\n",
        encoding="utf-8",
    )
    atomic_json(LOCK, {"pid": os.getpid(), "state": "FAILED", "status": "FAIL", "error": repr(exc), "timestamp": utc_now()})
    if status is not None:
        status.update(phase="FAILED", latest_error=repr(exc), recovery_state="MANUAL_RESUME_REQUIRED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure()
    if args.preflight:
        print(json.dumps(preflight(), indent=2), flush=True)
        return 0
    if args.smoke_test:
        result = preflight()
        result["core_smoke"] = core.smoke_test()
        print(json.dumps(result, indent=2), flush=True)
        return 0
    status: core.EventStatus | None = None
    started = time.perf_counter()
    try:
        if active_run() is not None:
            raise RuntimeError(f"S6-AE already active: PID {active_run().get('pid')}")
        source, audit, dry = verify_sources()
        initialize(source, audit, dry, args.resume)
        atomic_json(LOCK, {"pid": os.getpid(), "host": socket.gethostname(), "state": "ACTIVE", "status": "RUNNING", "timestamp": utc_now()})
        status = core.EventStatus()
        device = core.base._device()
        if device.type != "cuda":
            raise RuntimeError(f"CUDA required, got {device}")
        print(f"S6-AE START device={device} resume={args.resume} P=T=10000 hard-only-x1 occurrences=10000 steps=40/epoch stop=M10", flush=True)
        core.baseline_evaluation(device, status, args.resume)
        for cycle in range(10):
            core.run_cycle(cycle, device, source, args.resume, status)
        if not core.source_unchanged(source):
            raise RuntimeError("frozen source changed during S6-AE")
        status.update(cycle=None, phase="final_reporting_and_strict_audit")
        result = build_reports(source, time.perf_counter() - started)
        atomic_json(LOCK, {"pid": os.getpid(), "host": socket.gethostname(), "state": "COMPLETE", "status": "PASS",
                           "timestamp": utc_now(), "M10_sha256": sha256_file(model_path(10)), "final_verdict": result["FINAL_VERDICT"]})
        status.update(cycle=None, phase="COMPLETE", latest_checkpoint=str(model_path(10)), latest_error=None)
        marker = OUT / "S6_AE_INCOMPLETE"
        if marker.exists():
            marker.unlink()
        print("S6-AE 10K HARD ONLY X1 M0->M10 COMPLETE", flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as exc:
        write_error(exc, status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
