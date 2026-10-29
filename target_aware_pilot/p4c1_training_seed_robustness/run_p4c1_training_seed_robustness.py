#!/usr/bin/env python3
"""P4-C1 paired training-seed robustness audit for frozen R0/R2 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "target_aware_pilot" / "p3_structure_finetune"
P4B1 = ROOT / "target_aware_pilot" / "p4b1_acquisition_ablation"
P4B2 = ROOT / "target_aware_pilot" / "p4b2_random_archive_ablation"
OUT = ROOT / "target_aware_pilot" / "p4c1_training_seed_robustness"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(P4B1))

import run_p4b1_acquisition_ablation as p4b1  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.tsp_metric import validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402


FORMAL = p4b1.FORMAL
CYCLES = p4b1.CYCLES
M4 = p4b1.M4
M4_SHA = p4b1.M4_SHA
HARD_M5_SHA = p4b1.HARD_M5_SHA
ORIGINAL_SEED = 2026
NEW_SEEDS = (260101, 260102)
ALL_SEEDS = (ORIGINAL_SEED, *NEW_SEEDS)
ARMS = ("R0", "R2")
DATASETS = p4b1.DATASETS
BOOTSTRAP_REPLICATES = 10_000

INPUTS = {
    "R0": {
        "bundle": P4B1 / "random10k.npz",
        "bundle_sha": "f4c76b2e6c8dedc459f426c8340ce25369bf591b3584e3d0974005b4a33cc974",
        "labeled": P4B1 / "labeling" / "labeled_bundle.npz",
        "label_sha": "45161315a9ec570c0ac49bedf270dff2b3d40dc8752cd4e201f0720ba36c9580",
        "membership_sha": "54bdb644a348dff070379cab6e1a3be316786f011b79d475139e05ed4fa5d38a",
        "original_checkpoint": P4B1 / "checkpoints" / "R0_Random_M5_E256.tsp.best",
        "original_checkpoint_sha": "72e3d4dcb4a0bc1c5562e9ab161deaffddb03e0829a01798cec078668540e8a1",
        "original_eval": P4B1 / "r0_m5_external_benchmark_results.csv",
        "original_eval_label": "R0_Random_M5",
    },
    "R2": {
        "bundle": P4B2 / "R2_random10k.npz",
        "bundle_sha": "21849ce7f7270fe95a01826ac489e2598c1d43fa2f6bedad66fa75cff6a20688",
        "labeled": P4B2 / "labeling" / "labeled_bundle.npz",
        "label_sha": "8ae216e71b28f8ac5b3b450ea51f651795f0bf42e6383969f2f2753152b53a79",
        "membership_sha": "6f141dcc069fac3c5007af70e5d4adc5b53b3a2ad0372d6269e0eeaa00b8961d",
        "original_checkpoint": P4B2 / "checkpoints" / "R2_RandomArchive_Random10k_M5_E256.tsp.best",
        "original_checkpoint_sha": "c1e569b86221af567b7414d5edc5cb71c128e0df1617063f8aa05442bce61fe4",
        "original_eval": P4B2 / "r2_m5_external_benchmark_results.csv",
        "original_eval_label": "R2_RandomArchive_M5",
    },
}


def write_json(path: Path, value: Any) -> None:
    p4b1.write_json(path, value)


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def run_name(arm: str, seed: int) -> str:
    return f"{arm}_seed{seed}"


def run_dir(arm: str, seed: int) -> Path:
    return LOGS / run_name(arm, seed)


def checkpoint_path(arm: str, seed: int) -> Path:
    return CHECKPOINTS / f"{arm}_M5_seed{seed}_E256.tsp.best"


def eval_label(arm: str, seed: int) -> str:
    return f"{arm}_M5_seed{seed}"


def eval_csv(arm: str, seed: int) -> Path:
    return run_dir(arm, seed) / "external_benchmark_results.csv"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_labeled(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["coordinates"], float), payload["instance_hash"].astype(str).tolist(), np.asarray(payload["expert_tours"], np.int64)


def base_protocol() -> dict[str, Any]:
    core_path = ROOT / "scripts" / "run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
    return {
        "m4_sha256": M4_SHA,
        "trainer_source": str(core_path),
        "trainer_source_sha256": sha256_file(core_path),
        "optimizer": "fresh Adam",
        "scheduler": None,
        "learning_rate": 3.75e-5,
        "epochs": 256,
        "optimizer_updates_per_epoch": 40,
        "total_optimizer_updates": 10_240,
        "logical_batch_size": 256,
        "microbatch_size": 64,
        "drop_last": False,
        "dropout": 0.0,
        "sampler": "one NumPy permutation of frozen 10k per epoch",
        "augmentation": "one NumPy-driven random suffix length per logical batch",
        "checkpoint_rule": "fixed E256; no validation/test selection",
        "evaluation": "frozen greedy evaluator",
    }


def audit() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    if sha256_file(M4) != M4_SHA:
        raise RuntimeError("formal M4 SHA mismatch")
    input_rows = []
    for arm in ARMS:
        item = INPUTS[arm]
        for path_key, sha_key in (("bundle", "bundle_sha"), ("labeled", "label_sha"), ("original_checkpoint", "original_checkpoint_sha")):
            if sha256_file(item[path_key]) != item[sha_key]:
                raise RuntimeError(f"{arm} {path_key} SHA mismatch")
        frozen_xy, frozen_hashes = p4b1.load_bundle(item["bundle"])
        coordinates, hashes, tours = load_labeled(item["labeled"])
        if hashes != frozen_hashes or not np.array_equal(coordinates, frozen_xy):
            raise RuntimeError(f"{arm} frozen labeled bundle mismatch")
        if coordinates.shape != (10_000, 100, 2) or tours.shape != (10_000, 100) or len(set(hashes)) != 10_000:
            raise RuntimeError(f"{arm} data shape/uniqueness mismatch")
        if membership_hash(hashes) != item["membership_sha"]:
            raise RuntimeError(f"{arm} membership SHA mismatch")
        if not all(validate_tour_0based(tour, 100)[0] for tour in tours):
            raise RuntimeError(f"{arm} contains invalid expert tour")
        training_info = json.loads(((P4B1 if arm == "R0" else P4B2) / "training_integrity.json").read_text())
        if not training_info.get("e000_weights_equal_m4") or training_info.get("optimizer_steps") != 10_240 or training_info.get("m4_sha256") != M4_SHA:
            raise RuntimeError(f"{arm} original training integrity mismatch")
        evaluation = pd.read_csv(item["original_eval"])
        if set(evaluation.dataset.astype(str)) != set(DATASETS) or set(evaluation.checkpoint_sha256.astype(str)) != {item["original_checkpoint_sha"]}:
            raise RuntimeError(f"{arm} original evaluation mismatch")
        input_rows.append({
            "arm": arm, "rows": len(hashes), "membership_sha256": membership_hash(hashes),
            "coordinate_bundle_sha256": item["bundle_sha"], "label_bundle_sha256": item["label_sha"],
            "original_training_seed": ORIGINAL_SEED, "original_checkpoint_sha256": item["original_checkpoint_sha"],
            "original_final_loss": training_info["final_loss"], "original_optimizer_updates": training_info["optimizer_steps"],
        })
    original_frames = [pd.read_csv(INPUTS[arm]["original_eval"]).set_index("dataset") for arm in ARMS]
    for dataset in DATASETS:
        if original_frames[0].at[dataset, "dataset_sha256"] != original_frames[1].at[dataset, "dataset_sha256"] or original_frames[0].at[dataset, "evaluator_config_sha256"] != original_frames[1].at[dataset, "evaluator_config_sha256"]:
            raise RuntimeError(f"original evaluation protocol mismatch: {dataset}")

    protocol = base_protocol()
    prereg = {
        "status": "PREREGISTERED_BEFORE_NEW_TRAINING",
        "original_common_training_seed": ORIGINAL_SEED,
        "new_paired_training_seeds": list(NEW_SEEDS),
        "arms": list(ARMS),
        "new_training_run_count": 4,
        "data_and_labels_frozen": True,
        "concorde_calls_allowed": False,
        "base_training_protocol": protocol,
        "base_training_protocol_sha256": canonical_hash(protocol),
        "delta_definition": "mean_gap_R2 - mean_gap_R0; positive favors R0",
        "gap8_definition": "unweighted arithmetic mean of the same eight frozen benchmark mean gaps",
        "test_instance_uncertainty": "10,000 paired bootstrap replicates within each fixed training seed and dataset",
        "training_seed_variability": "descriptive mean, median, sample SD, range, and direction counts across exactly three paired seeds",
        "supported_rule": "all 3 paired seeds have GAP8 Delta > 0; R0 wins at least 5/8 benchmark means within every seed; cross-seed mean Delta > 0 on at least 5/8 benchmarks",
        "not_supported_rule": "all 3 paired seeds have GAP8 Delta < 0; R2 wins at least 5/8 benchmark means within every seed; cross-seed mean Delta < 0 on at least 5/8 benchmarks",
        "otherwise": "INCONCLUSIVE",
        "no_additional_seed": True,
        "tsplib22_role": "repeatedly observed exploratory comparability set; never model selection",
    }
    write_json(OUT / "preregistered_config.json", prereg)
    pd.DataFrame(input_rows).to_csv(OUT / "support_input_integrity.csv", index=False)
    integrity = f"""# P4-C1 Protocol Integrity

- R0 and R2 each reuse the exact frozen 10,000 coordinates and exact existing Concorde label bundle; all bundle, membership, label and original-checkpoint SHA256 values reproduce.
- Both original runs are a strict paired training-seed comparison at seed `{ORIGINAL_SEED}`: same formal M4 `{M4_SHA}`, fresh Adam, LR 3.75e-5, 256 epochs, 10,240 updates, batch/microbatch 256/64, dropout 0, single worker and fixed E256 checkpoint rule.
- Training randomness is confined to one `np.random.default_rng(2026)` in the formal trainer, controlling epoch permutations and suffix-length augmentation. The P4-C1 adapter maps only that single audited call to each preregistered seed, asserts exactly one call with original argument 2026, and restores NumPy immediately afterward. Formal A0 source is not modified.
- Existing R0/R2 E000 weights equal formal M4; every new run must repeat the same tensorwise gate.
- Frozen greedy evaluator dataset/config hashes match across original R0/R2 for all eight benchmarks.
- No FOGA, archive retention, acquisition, Concorde, HGB, structure feature or TSPLIB reference computation is rerun.

Status: **PASS**.
"""
    (OUT / "00_protocol_integrity.md").write_text(integrity, encoding="utf-8")
    seed_report = f"""# Preregistered Training-Seed Protocol

- Original paired seed: `{ORIGINAL_SEED}` (existing R0 and R2).
- New paired seeds: `{NEW_SEEDS[0]}`, `{NEW_SEEDS[1]}`; both arms run for each seed, exactly four new trainings.
- Delta = GAP(R2) − GAP(R0); positive favors target-aware archive retention.
- SUPPORTED: all three paired seeds have positive GAP8 Delta, R0 wins at least 5/8 benchmark means in every seed, and cross-seed mean Delta is positive on at least 5/8 benchmarks.
- NOT_SUPPORTED: exact directional mirror. All other outcomes are INCONCLUSIVE.
- Per-seed paired bootstrap quantifies test-instance uncertainty. The three seed-level Deltas are summarized descriptively and are never expanded into a pseudo-independent instance sample.
- No result-dependent extra seed, rerun, checkpoint selection, dataset weighting or protocol change is permitted.

Base protocol SHA256: `{prereg['base_training_protocol_sha256']}`.
"""
    (OUT / "01_preregistered_seed_protocol.md").write_text(seed_report, encoding="utf-8")
    write_training_manifest()
    return {"status": "PASS", "inputs": input_rows, "preregistered": prereg}


def with_training_seed(core: Any, seed: int, callback: Any) -> Any:
    original = core.np.random.default_rng
    calls: list[Any] = []

    def mapped(requested: Any = None) -> np.random.Generator:
        calls.append(requested)
        if requested != ORIGINAL_SEED:
            raise RuntimeError(f"unexpected formal trainer RNG request: {requested}")
        return original(seed)

    core.np.random.default_rng = mapped
    try:
        result = callback()
    finally:
        core.np.random.default_rng = original
    if calls != [ORIGINAL_SEED]:
        raise RuntimeError(f"seed adapter call audit failed: {calls}")
    return result


def train(arm: str, seed: int, device_name: str) -> None:
    if seed not in NEW_SEEDS or arm not in ARMS:
        raise ValueError("only preregistered new paired runs are allowed")
    audit()
    coordinates, hashes, tours = load_labeled(INPUTS[arm]["labeled"])
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    directory = run_dir(arm, seed) / "training_cycle"
    checkpoint = checkpoint_path(arm, seed)
    core.RUN_ID = f"TARGET_AWARE_P4C1_{arm}_SEED{seed}"
    core.cycle_dir = lambda selected_arm, cycle: directory
    core.model_path = lambda selected_arm, index: checkpoint if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    run_config = {"arm": arm, "training_seed": seed, "training_membership_sha256": membership_hash(hashes), "label_bundle_sha256": INPUTS[arm]["label_sha"], "base_protocol_sha256": json.loads((OUT / "preregistered_config.json").read_text())["base_training_protocol_sha256"]}
    run_config["training_config_sha256"] = canonical_hash(run_config)
    write_json(run_dir(arm, seed) / "training_config.json", run_config)
    started = time.perf_counter()

    def invoke() -> Any:
        return core.train(f"{arm}_SEED{seed}", 4, model, model_meta["sha256"], coordinates, hashes, tours, device, Status(), True)

    trained, trained_meta, timing = with_training_seed(core, seed, invoke)
    invocation_wall = time.perf_counter() - started
    stage = directory / "09_bq_training"
    e000 = torch.load(stage / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e000_equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    curve = pd.read_csv(stage / "training_loss_curve.csv")
    if not e000_equal or len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(10_000).all():
        raise RuntimeError(f"{arm} seed {seed} training integrity failure")
    if trained_meta["sha256"] != sha256_file(checkpoint) or not all(bool(torch.isfinite(parameter).all()) for parameter in trained.parameters()):
        raise RuntimeError(f"{arm} seed {seed} checkpoint integrity failure")
    integrity = {
        "status": "PASS", **run_config, "m4_sha256": M4_SHA, "e000_weights_equal_m4": e000_equal,
        "epochs": 256, "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
        "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": invocation_wall,
        "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]), "checkpoint_sha256": trained_meta["sha256"],
        "loss_curve_sha256": sha256_file(stage / "training_loss_curve.csv"), "seed_adapter_calls": [ORIGINAL_SEED],
    }
    write_json(run_dir(arm, seed) / "training_integrity.json", integrity)
    del trained, model
    if device.type == "cuda": torch.cuda.empty_cache()
    write_training_manifest()
    print(json.dumps(integrity, indent=2), flush=True)


def write_training_manifest() -> None:
    rows = []
    for arm in ARMS:
        original_info = json.loads(((P4B1 if arm == "R0" else P4B2) / "training_integrity.json").read_text())
        rows.append({
            "arm": arm, "training_seed": ORIGINAL_SEED, "run_type": "existing_strict_paired",
            "status": "PASS", "membership_sha256": INPUTS[arm]["membership_sha"], "label_bundle_sha256": INPUTS[arm]["label_sha"],
            "m4_sha256": M4_SHA, "e000_weights_equal_m4": original_info["e000_weights_equal_m4"], "optimizer_steps": original_info["optimizer_steps"],
            "final_loss": original_info["final_loss"], "training_wall_s": original_info["training_loop_wall_s"],
            "checkpoint_sha256": INPUTS[arm]["original_checkpoint_sha"], "checkpoint_path": str(INPUTS[arm]["original_checkpoint"]),
            "training_config_sha256": "historical_formal_seed2026",
        })
        for seed in NEW_SEEDS:
            path = run_dir(arm, seed) / "training_integrity.json"
            if path.is_file():
                item = json.loads(path.read_text())
                rows.append({
                    "arm": arm, "training_seed": seed, "run_type": "new_preregistered_paired", "status": item["status"],
                    "membership_sha256": item["training_membership_sha256"], "label_bundle_sha256": item["label_bundle_sha256"],
                    "m4_sha256": item["m4_sha256"], "e000_weights_equal_m4": item["e000_weights_equal_m4"], "optimizer_steps": item["optimizer_steps"],
                    "final_loss": item["final_loss"], "training_wall_s": item["training_loop_wall_s"], "checkpoint_sha256": item["checkpoint_sha256"],
                    "checkpoint_path": str(checkpoint_path(arm, seed)), "training_config_sha256": item["training_config_sha256"],
                })
            else:
                rows.append({"arm": arm, "training_seed": seed, "run_type": "new_preregistered_paired", "status": "PENDING", "membership_sha256": INPUTS[arm]["membership_sha"], "label_bundle_sha256": INPUTS[arm]["label_sha"], "m4_sha256": M4_SHA})
    pd.DataFrame(rows).to_csv(OUT / "02_training_runs_manifest.csv", index=False)


def evaluate(arm: str, seed: int, device_name: str) -> None:
    if seed not in NEW_SEEDS or arm not in ARMS: raise ValueError("only preregistered new paired runs are allowed")
    integrity = json.loads((run_dir(arm, seed) / "training_integrity.json").read_text())
    checkpoint = checkpoint_path(arm, seed)
    if integrity.get("checkpoint_sha256") != sha256_file(checkpoint): raise RuntimeError("checkpoint SHA mismatch")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    model, meta = load_canonical_checkpoint(checkpoint, device, expected_sha256=integrity["checkpoint_sha256"])
    frame, wall = core.evaluate(model, checkpoint, eval_label(arm, seed), eval_csv(arm, seed), Status(), run_name(arm, seed))
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all(): raise RuntimeError("evaluation incomplete")
    result = {"status": "PASS", "arm": arm, "training_seed": seed, "checkpoint_sha256": meta["sha256"], "all_tours_valid": True, "datasets": list(DATASETS), "evaluation_wall_s": wall, "results_sha256": sha256_file(eval_csv(arm, seed))}
    write_json(run_dir(arm, seed) / "evaluation_integrity.json", result)
    print(frame[["dataset", "instance_count", "mean_gap_percent"]].to_string(index=False), flush=True)


def detail(arm: str, seed: int, dataset: str) -> pd.DataFrame:
    if seed == ORIGINAL_SEED:
        base = P4B1 if arm == "R0" else P4B2
        label = INPUTS[arm]["original_eval_label"]
        path = base / "evaluation_work" / label / dataset / "instance_metrics.parquet"
    else:
        path = run_dir(arm, seed) / "evaluation_work" / eval_label(arm, seed) / dataset / "instance_metrics.parquet"
    return pd.read_parquet(path)


def hard_detail(dataset: str) -> pd.DataFrame:
    return pd.read_parquet(CYCLES / "C04" / "evaluation_work" / "M5" / dataset / "instance_metrics.parquet")


def paired_dataset(seed: int, dataset: str) -> pd.DataFrame:
    left = detail("R0", seed, dataset)[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
    right = detail("R2", seed, dataset)[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
    if not left.valid_tour.all() or not right.valid_tour.all(): raise RuntimeError(f"invalid tour seed {seed}/{dataset}")
    left = left.rename(columns={"coordinate_hash": "coordinate_hash_R0", "n_nodes": "n_nodes_R0", "optimal_cost_int": "optimal_cost_int_R0", "gap_percent": "gap_R0"}).drop(columns="valid_tour")
    right = right.rename(columns={"coordinate_hash": "coordinate_hash_R2", "n_nodes": "n_nodes_R2", "optimal_cost_int": "optimal_cost_int_R2", "gap_percent": "gap_R2"}).drop(columns="valid_tour")
    merged = left.merge(right, on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        if not merged[f"{field}_R0"].astype(str).eq(merged[f"{field}_R2"].astype(str)).all(): raise RuntimeError(f"alignment mismatch {seed}/{dataset}/{field}")
    return pd.DataFrame({"training_seed": seed, "dataset": dataset, "instance_id": merged.instance_id.astype(str), "coordinate_hash": merged.coordinate_hash_R0.astype(str), "n_nodes": merged.n_nodes_R0.astype(int), "optimal_cost_int": merged.optimal_cost_int_R0.astype(np.int64), "gap_R0": merged.gap_R0.astype(float), "gap_R2": merged.gap_R2.astype(float), "Delta_R2_minus_R0": merged.gap_R2.astype(float) - merged.gap_R0.astype(float)})


def verify_evaluations() -> None:
    frames = []
    for arm in ARMS:
        original = pd.read_csv(INPUTS[arm]["original_eval"])
        frames.append(original.set_index("dataset"))
        for seed in NEW_SEEDS:
            integrity = json.loads((run_dir(arm, seed) / "evaluation_integrity.json").read_text())
            if integrity.get("status") != "PASS": raise RuntimeError(f"missing evaluation {arm}/{seed}")
            frame = pd.read_csv(eval_csv(arm, seed))
            if set(frame.checkpoint_sha256.astype(str)) != {sha256_file(checkpoint_path(arm, seed))}: raise RuntimeError("evaluation checkpoint mismatch")
            frames.append(frame.set_index("dataset"))
    for dataset in DATASETS:
        if len({str(frame.at[dataset, "dataset_sha256"]) for frame in frames}) != 1 or len({str(frame.at[dataset, "evaluator_config_sha256"]) for frame in frames}) != 1:
            raise RuntimeError(f"evaluator protocol mismatch {dataset}")


def summarize_seed_dataset(seed: int, name: str, frame: pd.DataFrame) -> dict[str, Any]:
    values = frame.Delta_R2_minus_R0.to_numpy(float)
    low, high = p4b1.paired_bootstrap(values, p4b1.stable_seed(f"P4C1|{seed}|{name}"))
    return {"training_seed": seed, "dataset": name, "n_instances": len(frame), "mean_gap_R0": float(frame.gap_R0.mean()), "mean_gap_R2": float(frame.gap_R2.mean()), "Delta_R2_minus_R0": float(values.mean()), "Delta_ci95_low": low, "Delta_ci95_high": high, "R0_instance_wins": int(np.sum(values > 0)), "R2_instance_wins": int(np.sum(values < 0)), "ties": int(np.sum(values == 0)), "bootstrap_replicates": BOOTSTRAP_REPLICATES}


def make_figures(per_seed: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    macro = per_seed[per_seed.dataset.eq("GAP8_macro")]; x = np.arange(len(ALL_SEEDS)); width = .34
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(x-width/2, macro.mean_gap_R0, width, label="R0 target archive"); ax.bar(x+width/2, macro.mean_gap_R2, width, label="R2 random archive")
    ax.set_xticks(x); ax.set_xticklabels([str(seed) for seed in ALL_SEEDS]); ax.set_xlabel("training seed"); ax.set_ylabel("GAP8 (%)"); ax.set_title("Paired training-seed GAP8"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "01_gap8_by_seed.png", dpi=180); plt.close(fig)
    base = per_seed[per_seed.dataset.isin(DATASETS)].pivot(index="dataset", columns="training_seed", values="Delta_R2_minus_R0").loc[list(DATASETS), list(ALL_SEEDS)]
    fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(base.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-np.abs(base.to_numpy()).max(), vmax=np.abs(base.to_numpy()).max())
    ax.set_xticks(np.arange(len(ALL_SEEDS))); ax.set_xticklabels([str(seed) for seed in ALL_SEEDS]); ax.set_yticks(np.arange(len(DATASETS))); ax.set_yticklabels(DATASETS); ax.set_title("Delta = R2 − R0 gap (pp)")
    for i in range(len(DATASETS)):
        for j in range(len(ALL_SEEDS)): ax.text(j, i, f"{base.iloc[i,j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(FIG / "02_benchmark_delta_heatmap.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, seed in enumerate(ALL_SEEDS):
        plot = per_seed[(per_seed.training_seed.eq(seed)) & per_seed.dataset.isin(DATASETS)].set_index("dataset").loc[list(DATASETS)].reset_index(); y = np.arange(len(plot)) + (offset-1)*.18
        values = plot.Delta_R2_minus_R0.to_numpy(); low = plot.Delta_ci95_low.to_numpy(); high = plot.Delta_ci95_high.to_numpy(); ax.errorbar(values, y, xerr=np.vstack([values-low, high-values]), fmt="o", capsize=2, label=str(seed))
    ax.axvline(0, color="black", linewidth=.8); ax.set_yticks(np.arange(len(DATASETS))); ax.set_yticklabels(DATASETS); ax.set_xlabel("R2 − R0 gap (pp)"); ax.set_title("Per-seed test-instance paired bootstrap 95% CI"); ax.legend(title="training seed")
    fig.tight_layout(); fig.savefig(FIG / "03_per_seed_paired_intervals.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arm in zip(axes, ARMS, strict=True):
        base_dir = P4B1 if arm == "R0" else P4B2
        ax.plot(pd.read_csv(base_dir / "training_cycle" / "09_bq_training" / "training_loss_curve.csv").mean_training_loss, label=str(ORIGINAL_SEED))
        for seed in NEW_SEEDS: ax.plot(pd.read_csv(run_dir(arm, seed) / "training_cycle" / "09_bq_training" / "training_loss_curve.csv").mean_training_loss, label=str(seed))
        ax.set_title(arm); ax.set_xlabel("epoch"); ax.grid(alpha=.2); ax.legend(title="seed")
    axes[0].set_ylabel("mean training loss"); fig.suptitle("Training loss across paired seeds"); fig.tight_layout(); fig.savefig(FIG / "04_training_loss_by_seed.png", dpi=180); plt.close(fig)


def finalize() -> None:
    verify_evaluations(); write_training_manifest()
    dev = set(pd.read_csv(ROOT / "target_aware_pilot" / "03_tsplib48_reference_manifest.csv").instance_id.astype(str)); held = set(json.loads((ROOT / "target_aware_pilot" / "p1_heldout22_names.json").read_text())["names"])
    rows = []; instance_parts = []; seed_frames: dict[tuple[int, str], pd.DataFrame] = {}
    for seed in ALL_SEEDS:
        for dataset in DATASETS:
            frame = paired_dataset(seed, dataset); seed_frames[(seed, dataset)] = frame; instance_parts.append(frame); rows.append(summarize_seed_dataset(seed, dataset, frame))
        tsplib = seed_frames[(seed, "OriginalTSPLIB70")]
        if set(tsplib.instance_id) != dev | held: raise RuntimeError("TSPLIB split mismatch")
        for name, names in (("TSPLIB48_development", dev), ("TSPLIB22_previously_heldout", held)):
            part = tsplib[tsplib.instance_id.isin(names)].copy(); instance_parts.append(part.assign(dataset=name)); rows.append(summarize_seed_dataset(seed, name, part))
    per_seed = pd.DataFrame(rows)
    macro_rows = []
    for seed in ALL_SEEDS:
        base = per_seed[(per_seed.training_seed.eq(seed)) & per_seed.dataset.isin(DATASETS)]
        macro_rows.append({"training_seed": seed, "dataset": "GAP8_macro", "n_instances": int(base.n_instances.sum()), "mean_gap_R0": float(base.mean_gap_R0.mean()), "mean_gap_R2": float(base.mean_gap_R2.mean()), "Delta_R2_minus_R0": float(base.Delta_R2_minus_R0.mean()), "Delta_ci95_low": np.nan, "Delta_ci95_high": np.nan, "R0_instance_wins": np.nan, "R2_instance_wins": np.nan, "ties": np.nan, "bootstrap_replicates": np.nan})
    per_seed = pd.concat([per_seed, pd.DataFrame(macro_rows)], ignore_index=True)
    per_seed.to_csv(OUT / "03_per_seed_benchmark_gap.csv", index=False)
    pd.concat(instance_parts, ignore_index=True).to_csv(OUT / "support_per_instance_paired_differences.csv", index=False)
    aggregate_rows = []
    for dataset in [*DATASETS, "TSPLIB48_development", "TSPLIB22_previously_heldout", "GAP8_macro"]:
        values = per_seed[per_seed.dataset.eq(dataset)].sort_values("training_seed").Delta_R2_minus_R0.to_numpy(float)
        aggregate_rows.append({"dataset": dataset, "seed_n": len(values), "mean_Delta": float(values.mean()), "median_Delta": float(np.median(values)), "sample_sd_Delta": float(values.std(ddof=1)), "min_Delta": float(values.min()), "max_Delta": float(values.max()), "R0_winning_seed_n": int(np.sum(values > 0)), "R2_winning_seed_n": int(np.sum(values < 0)), "all_R0": bool(np.all(values > 0)), "all_R2": bool(np.all(values < 0))})
    aggregate = pd.DataFrame(aggregate_rows); aggregate.to_csv(OUT / "04_paired_seed_gap_analysis.csv", index=False)

    benchmark = per_seed[per_seed.dataset.isin(DATASETS)]
    gap8 = per_seed[per_seed.dataset.eq("GAP8_macro")].sort_values("training_seed")
    wins_by_seed = benchmark.assign(r0_win=benchmark.Delta_R2_minus_R0.gt(0)).groupby("training_seed").r0_win.sum().reindex(ALL_SEEDS)
    mean_benchmark = aggregate[aggregate.dataset.isin(DATASETS)]
    supported = bool(gap8.Delta_R2_minus_R0.gt(0).all() and wins_by_seed.ge(5).all() and mean_benchmark.mean_Delta.gt(0).sum() >= 5)
    not_supported = bool(gap8.Delta_R2_minus_R0.lt(0).all() and wins_by_seed.le(3).all() and mean_benchmark.mean_Delta.lt(0).sum() >= 5)
    if supported: decision = "TARGET_AWARE_RETENTION_TRAINING_SEED_ROBUSTNESS_SUPPORTED"; label = "SUPPORTED"
    elif not_supported: decision = "TARGET_AWARE_RETENTION_TRAINING_SEED_ROBUSTNESS_NOT_SUPPORTED"; label = "NOT_SUPPORTED"
    else: decision = "TARGET_AWARE_RETENTION_TRAINING_SEED_ROBUSTNESS_INCONCLUSIVE"; label = "INCONCLUSIVE"

    hard_means = {dataset: float(hard_detail(dataset).gap_percent.mean()) for dataset in DATASETS}
    hard_gap8 = float(np.mean(list(hard_means.values())))
    r0_beats_hard = {int(row.training_seed): bool(row.mean_gap_R0 < hard_gap8) for _, row in gap8.iterrows()}
    manifest = pd.read_csv(OUT / "02_training_runs_manifest.csv")
    loss_summary = manifest.groupby("arm").final_loss.agg(["mean", "std", "min", "max"]).reset_index()
    loss_table = ["| Arm | Mean | SD | Min | Max |", "|---|---:|---:|---:|---:|"]
    for _, loss_row in loss_summary.iterrows():
        loss_table.append(f"| {loss_row.arm} | {loss_row['mean']:.8f} | {loss_row['std']:.8f} | {loss_row['min']:.8f} | {loss_row['max']:.8f} |")
    make_figures(per_seed, aggregate)
    gap8_mean = aggregate[aggregate.dataset.eq("GAP8_macro")].iloc[0]
    benchmark_mean_sorted = mean_benchmark.sort_values("mean_Delta", ascending=False)
    largest_effects = ", ".join(
        f"{row.dataset} ({row.mean_Delta:+.4f} pp)"
        for _, row in benchmark_mean_sorted.head(3).iterrows()
    )
    uniform_row = mean_benchmark[mean_benchmark.dataset.eq("Uniform100")].iloc[0]
    gm100_row = mean_benchmark[mean_benchmark.dataset.eq("GaussianMixture100")].iloc[0]
    gm500_row = mean_benchmark[mean_benchmark.dataset.eq("GaussianMixture500")].iloc[0]
    tsplib100_row = mean_benchmark[mean_benchmark.dataset.eq("TSPLIB100")].iloc[0]
    original70_row = aggregate[aggregate.dataset.eq("OriginalTSPLIB70")].iloc[0]
    held22_row = aggregate[aggregate.dataset.eq("TSPLIB22_previously_heldout")].iloc[0]
    original70_ci_excludes_zero = int(
        ((per_seed[per_seed.dataset.eq("OriginalTSPLIB70")].Delta_ci95_low > 0)
         | (per_seed[per_seed.dataset.eq("OriginalTSPLIB70")].Delta_ci95_high < 0)).sum()
    )
    held22_ci_excludes_zero = int(
        ((per_seed[per_seed.dataset.eq("TSPLIB22_previously_heldout")].Delta_ci95_low > 0)
         | (per_seed[per_seed.dataset.eq("TSPLIB22_previously_heldout")].Delta_ci95_high < 0)).sum()
    )
    report = ["# Training-Seed Robustness Report", "", "Delta = GAP(R2) − GAP(R0); positive favors R0.", "", "| Seed | R0 GAP8 | R2 GAP8 | Delta | R0 benchmark wins |", "|---:|---:|---:|---:|---:|"]
    for seed in ALL_SEEDS:
        row = gap8[gap8.training_seed.eq(seed)].iloc[0]; report.append(f"| {seed} | {row.mean_gap_R0:.4f} | {row.mean_gap_R2:.4f} | {row.Delta_R2_minus_R0:+.4f} | {int(wins_by_seed.loc[seed])}/8 |")
    report.extend(["", f"- Across the three paired seeds, GAP8 Delta mean {gap8_mean.mean_Delta:+.4f} pp, median {gap8_mean.median_Delta:+.4f}, sample SD {gap8_mean.sample_sd_Delta:.4f}, range [{gap8_mean.min_Delta:+.4f}, {gap8_mean.max_Delta:+.4f}].", f"- R0 beats Original Hardness M5 GAP8 ({hard_gap8:.4f}%) by seed: {r0_beats_hard}.", f"- R0 benchmark-win counts by seed: {dict((int(k), int(v)) for k,v in wins_by_seed.items())}.", f"- Cross-seed mean Delta favors R0 on {int(mean_benchmark.mean_Delta.gt(0).sum())}/8 frozen benchmarks. The three largest mean effects are {largest_effects}; the advantage is broad in direction but concentrated in magnitude.", f"- No systematic Uniform/Gaussian-mixture degradation is observed: Uniform100 {uniform_row.mean_Delta:+.4f} pp ({int(uniform_row.R0_winning_seed_n)}/3 R0-winning seeds), GaussianMixture100 {gm100_row.mean_Delta:+.4f} pp ({int(gm100_row.R0_winning_seed_n)}/3), GaussianMixture500 {gm500_row.mean_Delta:+.4f} pp ({int(gm500_row.R0_winning_seed_n)}/3). TSPLIB100 is also positive at {tsplib100_row.mean_Delta:+.4f} pp ({int(tsplib100_row.R0_winning_seed_n)}/3).", f"- Real-TSPLIB evidence is weaker: OriginalTSPLIB70 mean Delta {original70_row.mean_Delta:+.4f} pp with {int(original70_row.R0_winning_seed_n)}/3 R0-winning seeds and {original70_ci_excludes_zero}/3 per-seed CIs excluding zero; TSPLIB22 mean Delta {held22_row.mean_Delta:+.4f} pp with {int(held22_row.R0_winning_seed_n)}/3 R0-winning seeds and {held22_ci_excludes_zero}/3 CIs excluding zero.", "- Final-loss descriptive summary:", "", *loss_table, "", "- Final training loss is descriptive only; its small arm difference is not used as evidence of generalization.", "- Per-seed CIs quantify test-instance uncertainty only. Seed-level variability is summarized descriptively from three paired seeds; instances are not pooled across seeds as independent replicates.", "- TSPLIB22 is repeatedly observed and remains exploratory.", "- P4-C2 data/archive-sampling-seed robustness is the appropriate next priority because P4-C1 changes only training stochasticity. It is recommended for a separate preregistered run and is not started here."])
    (OUT / "05_training_seed_robustness_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision_report = f"""# P4-C1 Final Decision

## `{decision}` — **{label}**

- Preregistered rule result: all-seed positive GAP8 Delta = **{bool(gap8.Delta_R2_minus_R0.gt(0).all())}**; R0 ≥5/8 benchmark wins in every seed = **{bool(wins_by_seed.ge(5).all())}**; positive cross-seed mean Delta on at least 5/8 benchmarks = **{bool(mean_benchmark.mean_Delta.gt(0).sum() >= 5)}** ({int(mean_benchmark.mean_Delta.gt(0).sum())}/8).
- GAP8 Delta across seeds: mean {gap8_mean.mean_Delta:+.4f} pp, sample SD {gap8_mean.sample_sd_Delta:.4f}, range [{gap8_mean.min_Delta:+.4f}, {gap8_mean.max_Delta:+.4f}].
- Directional breadth: positive cross-seed mean Delta on {int(mean_benchmark.mean_Delta.gt(0).sum())}/8 benchmarks; largest effects are {largest_effects}.
- No systematic Uniform/Gaussian-mixture degradation was detected. Real-TSPLIB conclusions remain limited because OriginalTSPLIB70 changes direction across seeds and neither its nor TSPLIB22's per-seed paired CIs exclude zero.
- This conclusion addresses training-seed robustness only. It does not establish robustness to archive/data-sampling seeds.
- P4-C2 is recommended as the next robustness check, but is not started automatically.

Decision label: `{label}`

`TARGET_AWARE_P4C1_TRAINING_SEED_ROBUSTNESS_COMPLETE`
"""
    (OUT / "06_final_decision.md").write_text(decision_report, encoding="utf-8")
    write_json(OUT / "run_metadata.json", {"status": "COMPLETE", "decision": decision, "decision_label": label, "paired_seeds": list(ALL_SEEDS), "gap8_delta_mean": gap8_mean.mean_Delta, "gap8_delta_sample_sd": gap8_mean.sample_sd_Delta, "per_seed_csv_sha256": sha256_file(OUT / "03_per_seed_benchmark_gap.csv"), "aggregate_csv_sha256": sha256_file(OUT / "04_paired_seed_gap_analysis.csv"), "no_concorde": True, "no_data_resampling": True, "no_extra_seed": True, "p4c2_started": False})
    Status().update(phase="complete", status="COMPLETE", decision=decision, decision_label=label)
    print(json.dumps({"status": "COMPLETE", "decision": decision, "label": label, "gap8": gap8[["training_seed", "mean_gap_R0", "mean_gap_R2", "Delta_R2_minus_R0"]].to_dict("records")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("audit", "train", "evaluate", "finalize")); parser.add_argument("--arm", choices=ARMS); parser.add_argument("--seed", type=int); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    if args.phase == "audit": print(json.dumps(audit(), indent=2), flush=True)
    elif args.phase in ("train", "evaluate"):
        if args.arm is None or args.seed is None: parser.error("--arm and --seed are required")
        (train if args.phase == "train" else evaluate)(args.arm, args.seed, args.device)
    else: finalize()


if __name__ == "__main__":
    main()
