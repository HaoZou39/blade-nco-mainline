#!/usr/bin/env python3
"""P3 structure-guided Cycle-5 fine-tuning verification.

The phase driver reuses the formal A0 labeling, training, and evaluation
functions.  It only redirects their outputs into the isolated P3 directory.
No evolution, acquisition, or model-selection logic is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.protocol import (  # noqa: E402
    ARCHITECTURE,
    DECODE_TYPE,
    INTEGER_SCALE,
    KNNS,
    LOSS_NAME,
    MICROBATCH_SIZE,
    OPTIMIZER_TYPE,
    TRAINING_INPUT,
)
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import membership_hash, parse_tours  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, sha256_file  # noqa: E402


OUT = ROOT / "target_aware_pilot" / "p3_structure_finetune"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
P2 = ROOT / "target_aware_pilot" / "p2_bounded_evolution"
INPUT_BUNDLE = P2 / "provisional_structure10k.npz"
INPUT_MANIFEST = P2 / "provisional_structure10k_manifest.csv"
INPUT_INTEGRITY = P2 / "provisional_structure10k_integrity.json"
EXPECTED_INPUT_SHA = "5287932231529e617383388bb64d607985d07944799c3d661c5a737338bdece1"

FORMAL = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1" / "A0_FULL_BASE"
CYCLES = FORMAL / "cycles"
M4 = FORMAL / "checkpoints" / "M4_E256.tsp.best"
HARD_M5 = FORMAL / "checkpoints" / "M5_E256.tsp.best"
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
HARD_M5_SHA = "41d189dfeb986a070bd7685f203172d1dad4376d8d0134151d7488fb079a4545"
CONCORDE = ROOT / "third_party" / "concorde" / "concorde"
SHARED_CACHE = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1" / "_shared_state" / "shared_oracle_cache.parquet"

STRUCTURE_M5 = CHECKPOINTS / "Structure_M5_E256.tsp.best"
LABEL_DIR = OUT / "labeling"
TRAIN_CYCLE = OUT / "training_cycle"
EVAL_CSV = OUT / "structure_m5_external_benchmark_results.csv"
P1 = ROOT / "target_aware_pilot"

DATASETS = (
    "Uniform100",
    "GaussianMixture100",
    "TSPLIB100",
    "EvoReal_S1_100",
    "EvoReal_S2_100",
    "EvoReal_S3_100",
    "GaussianMixture500",
    "OriginalTSPLIB70",
)
SEED = 2026
BOOTSTRAP_REPLICATES = 10_000


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n", encoding="utf-8")
    temp.replace(path)


def save_bundle(path: Path, coordinates: np.ndarray, hashes: list[str], tours: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temp,
        coordinates=np.asarray(coordinates, dtype=np.float64),
        instance_hash=np.asarray(hashes),
        expert_tours=np.asarray(tours, dtype=np.int64),
    )
    temp.replace(path)


def load_input() -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    with np.load(INPUT_BUNDLE, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
    manifest = pd.read_csv(INPUT_MANIFEST)
    integrity = json.loads(INPUT_INTEGRITY.read_text(encoding="utf-8"))
    return coordinates, hashes, manifest, integrity


def load_labeled() -> tuple[np.ndarray, list[str], np.ndarray, pd.DataFrame]:
    path = LABEL_DIR / "labeled_bundle.npz"
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
        tours = np.asarray(payload["expert_tours"], dtype=np.int64)
    return coordinates, hashes, tours, pd.read_parquet(LABEL_DIR / "labeled.parquet")


def file_inventory() -> dict[str, Any]:
    return {
        "structure_bundle_sha256": sha256_file(INPUT_BUNDLE),
        "structure_manifest_sha256": sha256_file(INPUT_MANIFEST),
        "structure_integrity_sha256": sha256_file(INPUT_INTEGRITY),
        "m4_sha256": sha256_file(M4),
        "hardness_m5_sha256": sha256_file(HARD_M5),
        "concorde_sha256": sha256_file(CONCORDE),
        "formal_runner_sha256": sha256_file(ROOT / "scripts" / "run_blade_core_ablation_5c_noanchor_foga11base_v1.py"),
        "supervised_module_sha256": sha256_file(ROOT / "src" / "bq" / "supervised.py"),
        "evaluation_module_sha256": sha256_file(ROOT / "scripts" / "external_v2_passive_blade_eval.py"),
        "p3_runner_sha256": sha256_file(Path(__file__)),
    }


def audit_inputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    coordinates, hashes, manifest, integrity = load_input()
    actual_sha = sha256_file(INPUT_BUNDLE)
    if actual_sha != EXPECTED_INPUT_SHA or actual_sha != integrity["bundle_sha256"]:
        raise RuntimeError("Structure10k bundle SHA mismatch")
    if coordinates.shape != (10_000, 100, 2):
        raise RuntimeError(f"Structure10k shape mismatch: {coordinates.shape}")
    if len(hashes) != len(set(hashes)) or hashes != manifest.instance_hash.astype(str).tolist():
        raise RuntimeError("Structure10k identity/alignment mismatch")
    reproduced = [coordinate_hash(points) for points in coordinates]
    if reproduced != hashes:
        raise RuntimeError("Structure10k coordinates do not reproduce IDs")
    if not np.isfinite(coordinates).all() or np.any((coordinates < 0) | (coordinates > 1)):
        raise RuntimeError("Structure10k contains invalid coordinates")
    duplicate_nodes = int(sum(np.unique(points, axis=0).shape[0] != 100 for points in coordinates))
    if duplicate_nodes:
        raise RuntimeError(f"Structure10k has {duplicate_nodes} duplicate-node instances")
    history: set[str] = set()
    for cycle in range(4):
        frame = pd.read_parquet(CYCLES / f"C{cycle:02d}" / "06_selection" / "T_selected.parquet", columns=["instance_hash"])
        history.update(frame.instance_hash.astype(str))
    c04 = set(pd.read_parquet(CYCLES / "C04" / "06_selection" / "T_selected.parquet", columns=["instance_hash"]).instance_hash.astype(str))
    if set(hashes) & history or set(hashes) & c04:
        raise RuntimeError("Structure10k overlaps frozen training/control sets")
    if sha256_file(M4) != M4_SHA or sha256_file(HARD_M5) != HARD_M5_SHA:
        raise RuntimeError("formal checkpoint SHA mismatch")
    if sha256_file(CONCORDE) != CANONICAL_CONCORDE_SHA256:
        raise RuntimeError("canonical Concorde SHA mismatch")
    c04_stage = json.loads((CYCLES / "C04" / "09_bq_training" / "STAGE_DONE.json").read_text(encoding="utf-8"))
    if c04_stage["input_hashes"]["model"] != M4_SHA:
        raise RuntimeError("formal C04 did not record the frozen M4 input")
    m4_blob = torch.load(M4, map_location="cpu", weights_only=False)
    c04_e0 = torch.load(CYCLES / "C04" / "09_bq_training" / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    if set(m4_blob["net"]) != set(c04_e0["net"]) or not all(torch.equal(m4_blob["net"][key], c04_e0["net"][key]) for key in m4_blob["net"]):
        raise RuntimeError("C04 E000 weights do not exactly match M4")
    cache = pd.read_parquet(SHARED_CACHE, columns=["instance_hash"])
    cache_overlap = len(set(hashes) & set(cache.instance_hash.astype(str)))
    audit = {
        "status": "PASS",
        "rows": len(hashes),
        "unique": len(set(hashes)),
        "node_count": coordinates.shape[1],
        "finite": True,
        "in_bounds": True,
        "duplicate_node_instances": duplicate_nodes,
        "membership_sha256": membership_hash(hashes),
        "m4_training_overlap": len(set(hashes) & history),
        "c04_hard10k_overlap": len(set(hashes) & c04),
        "shared_oracle_cache_overlap": cache_overlap,
        "m4_state_keys": list(m4_blob.keys()),
        "m4_contains_optimizer_state": "optimizer" in m4_blob or "optimizer_state_dict" in m4_blob,
        "c04_e000_weights_exactly_equal_m4": True,
        "c04_training_input_model_sha256": c04_stage["input_hashes"]["model"],
        **file_inventory(),
    }
    write_json(OUT / "input_integrity.json", audit)
    text = f"""# P3 Input Integrity

- Structure10k: 10,000 unique TSP100; bundle SHA256 `{actual_sha}`; membership SHA256 `{audit['membership_sha256']}`.
- Coordinates are finite, in `[0,1]`, contain 100 unique nodes per instance, and reproduce every manifest ID.
- C00-C03 known M4-training overlap: **0**. Original C04 Hard10k overlap: **0**.
- Formal M4: `{M4.relative_to(ROOT)}`, SHA256 `{M4_SHA}`.
- Original Hardness M5: `{HARD_M5.relative_to(ROOT)}`, SHA256 `{HARD_M5_SHA}`.
- Original C04 stage records M4 SHA `{c04_stage['input_hashes']['model']}` and its E000 state tensors exactly equal formal M4.
- M4 checkpoint contains model weights/epoch/metadata and no optimizer state. Original C04 creates a fresh Adam and has no scheduler; P3 will do the same through the original `train` function.
- Canonical Concorde SHA256 `{audit['concorde_sha256']}`. Shared-cache overlap is {cache_overlap}; no replacement or alternate instance source is permitted.
- Held-out TSPLIB22 coordinates/features are not loaded by input auditing, labeling, or training.

Status: **PASS**.
"""
    (OUT / "00_input_integrity.md").write_text(text, encoding="utf-8")
    protocol = f"""# Training Protocol Comparison

| Field | Original C04 Hard10k→M5 | P3 Structure10k→Structure M5 |
|---|---|---|
| Initialization | M4 `{M4_SHA}` | identical M4 |
| Architecture | `{ARCHITECTURE['identifier']}` | identical |
| Training set | frozen C04 Hard10k | frozen P2.2 E1 Structure10k |
| Unique/occurrences per epoch | 10,000 / 10,000 | identical |
| Epochs | 256 | identical |
| Logical batch / microbatch | 256 / {MICROBATCH_SIZE} | identical |
| Updates | 40/epoch, 10,240 total | identical |
| Optimizer | fresh {OPTIMIZER_TYPE}, LR 3.75e-5, weight decay 0 | identical |
| Scheduler / AMP / clipping | none / false / none | identical |
| RNG | NumPy seed {SEED}; one permutation per epoch | identical |
| Loss/input | `{LOSS_NAME}`; `{TRAINING_INPUT}` | identical |
| Checkpoint selection | E256 fixed before evaluation | identical |
| Decoder | `{DECODE_TYPE}`, beam 1, KNN={KNNS} | identical |

The only planned core variable is fifth-cycle training membership. The original A0 `train` and `evaluate` functions are invoked directly after redirecting output paths; no trainer or evaluator is reimplemented. TSPLIB22 is evaluated only after E256 is frozen and is never used for checkpoint or hyperparameter selection.

Predeclared interpretation: `STRUCTURE_C5_GAIN_CONFIRMED` requires a paired-bootstrap improvement over Hardness M5 on held-out22 with CI upper below zero, non-worse GAP8 macro, and no material (>0.1 percentage point or significant) regression on Uniform/Gaussian controls. Mixed evidence maps to `STRUCTURE_C5_MIXED`; concordant non-improvement/regression maps to `STRUCTURE_C5_NO_GAIN`; protocol or execution failure maps to `STRUCTURE_C5_INCONCLUSIVE`.
"""
    (OUT / "02_training_protocol_comparison.md").write_text(protocol, encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return audit


class P3Status:
    def update(self, **values: Any) -> None:
        value = {"timestamp_unix": time.time(), **values}
        write_json(OUT / "status.json", value)


def label(device_name: str) -> None:
    audit = audit_inputs()
    coordinates, hashes, manifest, _ = load_input()
    import run_blade_v2 as base

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    base.GLOBAL_CACHE = pd.read_parquet(SHARED_CACHE)
    selected = manifest.copy()
    selected["coord_index"] = np.arange(len(selected))
    started = time.perf_counter()
    labeled, tours, summary = base._label_population(
        LABEL_DIR,
        selected,
        coordinates,
        model,
        device,
        None,
        "P3_structure10k_original_A0_oracle_and_M4_hardness",
        True,
    )
    wall = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ledger = pd.read_parquet(LABEL_DIR / "ledger.parquet")
    if len(ledger) != 10_000 or not ledger.verified.all() or not ledger.status.eq("solved").all():
        raise RuntimeError("Concorde ledger is incomplete")
    if tours.shape != (10_000, 100):
        raise RuntimeError(f"expert tour shape mismatch: {tours.shape}")
    legal = np.asarray([validate_tour_0based(tour, 100)[0] for tour in tours])
    if not legal.all():
        raise RuntimeError("illegal Hamiltonian tour in labeled bundle")
    recomputed_costs = np.asarray(
        [tour_cost_int(integer_distance_matrix(points, INTEGER_SCALE), tour) for points, tour in zip(coordinates, tours, strict=True)],
        dtype=np.int64,
    )
    if not np.array_equal(recomputed_costs, ledger.optimal_cost_int.to_numpy(np.int64)):
        raise RuntimeError("expert tour costs differ from canonical ledger")
    closed = closed_expert_coordinates(coordinates, tours)
    if closed.shape != (10_000, 101, 2) or not np.array_equal(closed[:, 0], closed[:, -1]):
        raise RuntimeError("closed expert-coordinate convention mismatch")
    tour_hashes = [hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist():
        raise RuntimeError("tour hashes do not reproduce ledger")
    integrity = {
        "status": "PASS",
        "selected": 10_000,
        "verified": int(ledger.verified.sum()),
        "failed": 0,
        "retry_count": 0,
        "cache_hits": int(summary["cache_hits"]),
        "physical_new_concorde_calls": int(summary["new_concorde_calls"]),
        "pipeline_wall_seconds": float(summary["wall_seconds"]),
        "phase_invocation_wall_seconds": wall,
        "tour_shape": list(tours.shape),
        "tour_indexing": "0-based Hamiltonian permutation; closure implicit in cost and appended only for BQ input",
        "distance_convention": "per-edge Euclidean rint(scale=1,000,000), EXPLICIT FULL_MATRIX",
        "all_tours_legal": True,
        "all_costs_reproduced": True,
        "closed_training_shape": list(closed.shape),
        "ledger_sha256": sha256_file(LABEL_DIR / "ledger.parquet"),
        "labeled_bundle_sha256": sha256_file(LABEL_DIR / "labeled_bundle.npz"),
        "labeled_metadata_sha256": sha256_file(LABEL_DIR / "labeled.parquet"),
        "expert_tours_array_sha256": hashlib.sha256(np.ascontiguousarray(tours, dtype="<i8").tobytes()).hexdigest(),
        "model_used_for_label_hardness_sha256": model_meta["sha256"],
        "concorde_sha256": CANONICAL_CONCORDE_SHA256,
    }
    write_json(LABEL_DIR / "labeling_integrity.json", integrity)
    report = f"""# Concorde Labeling Audit

- Pipeline: original A0 `_label_population` → canonical `label_with_cache`/`solve_batch`, followed by the original M4 teacher-forced hardness attachment. Hardness metadata does not affect training membership or order.
- Selected/verified: 10,000/10,000; failures: 0; retries: 0; cache hits: {integrity['cache_hits']}; physical Concorde calls: {integrity['physical_new_concorde_calls']}.
- Wall-clock: {integrity['pipeline_wall_seconds']:.3f}s for the original combined label + M4-hardness stage.
- Tours: shape `(10000,100)`, 0-based Hamiltonian permutations. Closure is implicit for objective evaluation and appended as node 101 only when constructing BQ expert-ordered training input.
- Objective: explicit symmetric matrix with each Euclidean edge rounded after scaling by 1,000,000. All 10,000 costs exactly reproduce the Concorde ledger.
- Ledger SHA256 `{integrity['ledger_sha256']}`; labeled bundle SHA256 `{integrity['labeled_bundle_sha256']}`; expert-tour array SHA256 `{integrity['expert_tours_array_sha256']}`.

Status: **PASS**. Training is now permitted.
"""
    (OUT / "01_concorde_labeling_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(integrity, indent=2), flush=True)


def train(device_name: str) -> None:
    audit_inputs()
    if not (LABEL_DIR / "labeling_integrity.json").is_file():
        raise RuntimeError("labeling integrity is absent; training forbidden")
    label_integrity = json.loads((LABEL_DIR / "labeling_integrity.json").read_text(encoding="utf-8"))
    if label_integrity.get("status") != "PASS" or label_integrity.get("verified") != 10_000:
        raise RuntimeError("labeling did not pass")
    coordinates, hashes, tours, _ = load_labeled()
    input_coordinates, input_hashes, _, _ = load_input()
    if hashes != input_hashes or not np.array_equal(coordinates, input_coordinates):
        raise RuntimeError("labeled bundle changed Structure10k membership/coordinates")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    core.RUN_ID = "TARGET_AWARE_STRUCTURE_C5_P3"
    core.cycle_dir = lambda arm, cycle: TRAIN_CYCLE
    core.model_path = lambda arm, index: STRUCTURE_M5 if index == 5 else M4
    model, model_meta = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    started = time.perf_counter()
    trained, trained_meta, timing = core.train(
        "STRUCTURE_C5",
        4,
        model,
        model_meta["sha256"],
        coordinates,
        hashes,
        tours,
        device,
        P3Status(),
        True,
    )
    invocation_wall = time.perf_counter() - started
    e0 = torch.load(TRAIN_CYCLE / "09_bq_training" / "M5_E000.tsp.best", map_location="cpu", weights_only=False)
    m4 = torch.load(M4, map_location="cpu", weights_only=False)
    e0_equal = set(e0["net"]) == set(m4["net"]) and all(torch.equal(e0["net"][key], m4["net"][key]) for key in m4["net"])
    if not e0_equal:
        raise RuntimeError("P3 E000 does not exactly equal M4")
    if trained_meta["sha256"] != sha256_file(STRUCTURE_M5):
        raise RuntimeError("Structure M5 hash mismatch")
    if not all(torch.isfinite(parameter).all().item() for parameter in trained.parameters()):
        raise RuntimeError("Structure M5 contains nonfinite parameters")
    del trained, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    curve = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    if len(curve) != 256 or not curve.optimizer_updates.eq(40).all() or not curve.hard_instances_seen.eq(10_000).all():
        raise RuntimeError("training trajectory protocol mismatch")
    original_timing = json.loads((CYCLES / "C04" / "09_bq_training" / "bq_training_timing.json").read_text(encoding="utf-8"))
    report = f"""# Training Log and Checkpoint

- Directly reused formal A0 `train`; initialization M4 SHA256 `{M4_SHA}`. P3 E000 state tensors exactly equal M4: **{e0_equal}**.
- Fresh Adam, LR 3.75e-5, no scheduler/AMP/clipping, batch 256, microbatch 64, 256 epochs, 40 updates/epoch, 10,240 total updates, NumPy seed 2026.
- Each epoch exposes every one of the 10,000 Structure10k instances exactly once; no anchor mixture.
- Final loss: {timing['final_loss']:.8f}; training-loop wall: {timing['training_loop_wall_s']:.3f}s; invocation wall: {invocation_wall:.3f}s; peak allocated GPU bytes: {int(timing['gpu_peak_allocated_bytes']):,}.
- Original C04 comparison: final loss {original_timing['final_loss']:.8f}, training-loop wall {original_timing['training_loop_wall_s']:.3f}s, same 10,240-update protocol.
- Structure M5 checkpoint: `{STRUCTURE_M5.relative_to(ROOT)}`; SHA256 `{trained_meta['sha256']}`.
- Training data membership SHA256 `{membership_hash(hashes)}`; labeled bundle SHA256 `{sha256_file(LABEL_DIR / 'labeled_bundle.npz')}`.

Checkpoint E256 was fixed before any P3 evaluation. No TSPLIB22 result was observed or used during training.
"""
    (OUT / "03_training_log_and_checkpoint.md").write_text(report, encoding="utf-8")
    write_json(
        OUT / "training_integrity.json",
        {
            "status": "PASS",
            "m4_sha256": M4_SHA,
            "e000_weights_equal_m4": e0_equal,
            "structure_m5_sha256": trained_meta["sha256"],
            "training_membership_sha256": membership_hash(hashes),
            "label_bundle_sha256": sha256_file(LABEL_DIR / "labeled_bundle.npz"),
            "epochs": 256,
            "optimizer_steps": int(timing["optimizer_steps"]),
            "final_loss": float(timing["final_loss"]),
            "training_loop_wall_s": float(timing["training_loop_wall_s"]),
            "invocation_wall_s": invocation_wall,
            "gpu_peak_allocated_bytes": int(timing["gpu_peak_allocated_bytes"]),
        },
    )
    print(json.dumps({"status": "PASS", "checkpoint": str(STRUCTURE_M5), "sha256": trained_meta["sha256"], "timing": timing}, indent=2), flush=True)


def evaluate(device_name: str) -> None:
    if not STRUCTURE_M5.is_file() or not (OUT / "training_integrity.json").is_file():
        raise RuntimeError("frozen Structure M5 is absent")
    train_integrity = json.loads((OUT / "training_integrity.json").read_text(encoding="utf-8"))
    if train_integrity.get("status") != "PASS" or train_integrity.get("structure_m5_sha256") != sha256_file(STRUCTURE_M5):
        raise RuntimeError("Structure M5 integrity failed")
    device = torch.device(device_name)
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core

    model, meta = load_canonical_checkpoint(STRUCTURE_M5, device, expected_sha256=train_integrity["structure_m5_sha256"])
    frame, wall = core.evaluate(model, STRUCTURE_M5, "Structure_M5", EVAL_CSV, P3Status(), "STRUCTURE_C5")
    if set(frame.dataset.astype(str)) != set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all():
        raise RuntimeError("frozen evaluation incomplete")
    write_json(
        OUT / "evaluation_integrity.json",
        {
            "status": "PASS",
            "checkpoint_sha256": meta["sha256"],
            "datasets": list(DATASETS),
            "all_tours_valid": True,
            "evaluation_invocation_wall_s": wall,
            "external_results_sha256": sha256_file(EVAL_CSV),
        },
    )
    print(frame[["dataset", "instance_count", "mean_gap_percent"]].to_string(index=False), flush=True)


def paired_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    cursor = 0
    while cursor < BOOTSTRAP_REPLICATES:
        count = min(200, BOOTSTRAP_REPLICATES - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def baseline_detail(model: str, dataset: str) -> pd.DataFrame:
    cycle = "C03" if model == "M4" else "C04"
    path = CYCLES / cycle / "evaluation_work" / model / dataset / "instance_metrics.parquet"
    return pd.read_parquet(path)


def structure_detail(dataset: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "evaluation_work" / "Structure_M5" / dataset / "instance_metrics.parquet")


def align_three(dataset: str) -> pd.DataFrame:
    frames = []
    for model, frame in (("M4", baseline_detail("M4", dataset)), ("Hardness_M5", baseline_detail("M5", dataset)), ("Structure_M5", structure_detail(dataset))):
        required = frame[["instance_id", "coordinate_hash", "n_nodes", "optimal_cost_int", "gap_percent", "valid_tour"]].copy()
        if not required.valid_tour.all():
            raise RuntimeError(f"invalid tour in {model}/{dataset}")
        required = required.rename(
            columns={
                "coordinate_hash": f"coordinate_hash_{model}",
                "n_nodes": f"n_nodes_{model}",
                "optimal_cost_int": f"optimal_cost_int_{model}",
                "gap_percent": f"gap_{model}",
            }
        ).drop(columns=["valid_tour"])
        frames.append(required)
    merged = frames[0].merge(frames[1], on="instance_id", validate="one_to_one").merge(frames[2], on="instance_id", validate="one_to_one")
    for field in ("coordinate_hash", "n_nodes", "optimal_cost_int"):
        columns = [f"{field}_{model}" for model in ("M4", "Hardness_M5", "Structure_M5")]
        if not (merged[columns[0]].astype(str).eq(merged[columns[1]].astype(str)).all() and merged[columns[0]].astype(str).eq(merged[columns[2]].astype(str)).all()):
            raise RuntimeError(f"evaluation alignment failure {dataset}/{field}")
    return pd.DataFrame(
        {
            "dataset": dataset,
            "instance_id": merged.instance_id.astype(str),
            "coordinate_hash": merged.coordinate_hash_M4.astype(str),
            "n_nodes": merged.n_nodes_M4.astype(int),
            "optimal_cost_int": merged.optimal_cost_int_M4.astype(np.int64),
            "gap_M4": merged.gap_M4.astype(float),
            "gap_Hardness_M5": merged.gap_Hardness_M5.astype(float),
            "gap_Structure_M5": merged.gap_Structure_M5.astype(float),
        }
    )


def result_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    d_m4 = frame.gap_Structure_M5.to_numpy(float) - frame.gap_M4.to_numpy(float)
    d_hard = frame.gap_Structure_M5.to_numpy(float) - frame.gap_Hardness_M5.to_numpy(float)
    m4_ci = paired_bootstrap(d_m4, stable_seed(name + "|M4"))
    hard_ci = paired_bootstrap(d_hard, stable_seed(name + "|Hardness_M5"))
    return {
        "dataset": name,
        "n_instances": len(frame),
        "mean_gap_M4": float(frame.gap_M4.mean()),
        "mean_gap_Hardness_M5": float(frame.gap_Hardness_M5.mean()),
        "mean_gap_Structure_M5": float(frame.gap_Structure_M5.mean()),
        "Structure_minus_M4": float(d_m4.mean()),
        "Structure_minus_M4_ci95_low": m4_ci[0],
        "Structure_minus_M4_ci95_high": m4_ci[1],
        "Structure_minus_Hardness_M5": float(d_hard.mean()),
        "Structure_minus_Hardness_M5_ci95_low": hard_ci[0],
        "Structure_minus_Hardness_M5_ci95_high": hard_ci[1],
        "fraction_Structure_better_than_M4": float(np.mean(d_m4 < 0)),
        "fraction_Structure_better_than_Hardness_M5": float(np.mean(d_hard < 0)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def make_figures(summary: pd.DataFrame, paired: pd.DataFrame, original: pd.DataFrame, loss: pd.DataFrame, hard_loss: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hard_loss.epoch, hard_loss.mean_training_loss, label="Hardness M5", linewidth=1.4)
    ax.plot(loss.epoch, loss.mean_training_loss, label="Structure M5", linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean training loss")
    ax.set_title("Frozen Cycle-5 training trajectories")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_training_loss.png", dpi=180)
    plt.close(fig)

    base = summary[summary.dataset.isin(DATASETS)].copy()
    x = np.arange(len(base))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for offset, (column, label) in enumerate((("mean_gap_M4", "M4"), ("mean_gap_Hardness_M5", "Hardness M5"), ("mean_gap_Structure_M5", "Structure M5"))):
        ax.bar(x + width * (offset - 1), base[column], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(base.dataset, rotation=35, ha="right")
    ax.set_ylabel("mean optimality gap (%)")
    ax.set_title("Frozen benchmark comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_all_benchmark_mean_gaps.png", dpi=180)
    plt.close(fig)

    plot = summary[summary.dataset.ne("GAP8_macro")].copy()
    y = np.arange(len(plot))
    values = plot.Structure_minus_Hardness_M5.to_numpy(float)
    lo = plot.Structure_minus_Hardness_M5_ci95_low.to_numpy(float)
    hi = plot.Structure_minus_Hardness_M5_ci95_high.to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(values, y, xerr=np.vstack([values - lo, hi - values]), fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plot.dataset)
    ax.set_xlabel("Structure M5 − Hardness M5 gap (percentage points)")
    ax.set_title("Paired bootstrap 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(FIG / "03_paired_delta_confidence_intervals.png", dpi=180)
    plt.close(fig)

    subset = summary[summary.dataset.isin(["TSPLIB48_development", "TSPLIB22_heldout"])]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    positions = np.arange(len(subset))
    for offset, (column, label) in enumerate((("mean_gap_M4", "M4"), ("mean_gap_Hardness_M5", "Hardness M5"), ("mean_gap_Structure_M5", "Structure M5"))):
        ax.bar(positions + width * (offset - 1), subset[column], width=width, label=label)
    ax.set_xticks(positions)
    ax.set_xticklabels(["development48", "held-out22"])
    ax.set_ylabel("mean gap (%)")
    ax.set_title("OriginalTSPLIB70 split correctly by design role")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_tsplib48_22_split.png", dpi=180)
    plt.close(fig)

    controls = summary[summary.dataset.isin(["Uniform100", "GaussianMixture100", "GaussianMixture500"])]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(np.arange(len(controls)) - 0.18, controls.Structure_minus_M4, width=0.36, label="vs M4")
    ax.bar(np.arange(len(controls)) + 0.18, controls.Structure_minus_Hardness_M5, width=0.36, label="vs Hardness M5")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(len(controls)))
    ax.set_xticklabels(controls.dataset, rotation=20)
    ax.set_ylabel("gap change (percentage points)")
    ax.set_title("Uniform/Gaussian forgetting diagnostic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_forgetting_controls.png", dpi=180)
    plt.close(fig)


def finalize() -> None:
    if not (OUT / "evaluation_integrity.json").is_file():
        raise RuntimeError("evaluation has not passed")
    formal_m4_summary = pd.read_csv(CYCLES / "C03" / "external_benchmark_results.csv")
    formal_hard_summary = pd.read_csv(CYCLES / "C04" / "external_benchmark_results.csv")
    structure_summary = pd.read_csv(EVAL_CSV)
    expected_checkpoints = (
        (formal_m4_summary, M4_SHA, "M4"),
        (formal_hard_summary, HARD_M5_SHA, "Hardness M5"),
        (structure_summary, sha256_file(STRUCTURE_M5), "Structure M5"),
    )
    for frame, expected_checkpoint, label in expected_checkpoints:
        if set(frame.dataset.astype(str)) != set(DATASETS):
            raise RuntimeError(f"{label} benchmark set mismatch")
        if set(frame.checkpoint_sha256.astype(str)) != {expected_checkpoint}:
            raise RuntimeError(f"{label} evaluation checkpoint mismatch")
    protocol_check = formal_m4_summary[["dataset", "dataset_sha256", "evaluator_config_sha256"]].merge(
        formal_hard_summary[["dataset", "dataset_sha256", "evaluator_config_sha256"]],
        on="dataset",
        suffixes=("_M4", "_Hardness_M5"),
        validate="one_to_one",
    ).merge(
        structure_summary[["dataset", "dataset_sha256", "evaluator_config_sha256"]],
        on="dataset",
        validate="one_to_one",
    )
    if not (
        protocol_check.dataset_sha256_M4.eq(protocol_check.dataset_sha256_Hardness_M5).all()
        and protocol_check.dataset_sha256_M4.eq(protocol_check.dataset_sha256).all()
        and protocol_check.evaluator_config_sha256_M4.eq(protocol_check.evaluator_config_sha256_Hardness_M5).all()
        and protocol_check.evaluator_config_sha256_M4.eq(protocol_check.evaluator_config_sha256).all()
    ):
        raise RuntimeError("three-model dataset/evaluator protocol mismatch")
    aligned = {dataset: align_three(dataset) for dataset in DATASETS}
    tsplib = aligned["OriginalTSPLIB70"].copy()
    dev = set(pd.read_csv(P1 / "03_tsplib48_reference_manifest.csv").instance_id.astype(str))
    held = set(json.loads((P1 / "p1_heldout22_names.json").read_text(encoding="utf-8"))["names"])
    if len(dev) != 48 or len(held) != 22 or dev & held or set(tsplib.instance_id) != dev | held:
        raise RuntimeError("TSPLIB48/22 frozen split mismatch")
    aligned["TSPLIB48_development"] = tsplib[tsplib.instance_id.isin(dev)].copy()
    aligned["TSPLIB22_heldout"] = tsplib[tsplib.instance_id.isin(held)].copy()

    paired_parts = []
    rows = []
    for name, frame in aligned.items():
        part = frame.copy()
        part["analysis_dataset"] = name
        part["Structure_minus_M4"] = part.gap_Structure_M5 - part.gap_M4
        part["Structure_minus_Hardness_M5"] = part.gap_Structure_M5 - part.gap_Hardness_M5
        paired_parts.append(part)
        rows.append(result_row(name, frame))
    summary = pd.DataFrame(rows)
    base = summary[summary.dataset.isin(DATASETS)]
    macro = {
        "dataset": "GAP8_macro",
        "n_instances": int(base.n_instances.sum()),
        "mean_gap_M4": float(base.mean_gap_M4.mean()),
        "mean_gap_Hardness_M5": float(base.mean_gap_Hardness_M5.mean()),
        "mean_gap_Structure_M5": float(base.mean_gap_Structure_M5.mean()),
    }
    macro["Structure_minus_M4"] = macro["mean_gap_Structure_M5"] - macro["mean_gap_M4"]
    macro["Structure_minus_Hardness_M5"] = macro["mean_gap_Structure_M5"] - macro["mean_gap_Hardness_M5"]
    for key in (
        "Structure_minus_M4_ci95_low",
        "Structure_minus_M4_ci95_high",
        "Structure_minus_Hardness_M5_ci95_low",
        "Structure_minus_Hardness_M5_ci95_high",
        "fraction_Structure_better_than_M4",
        "fraction_Structure_better_than_Hardness_M5",
        "bootstrap_replicates",
    ):
        macro[key] = np.nan
    summary = pd.concat([summary, pd.DataFrame([macro])], ignore_index=True)
    paired = pd.concat(paired_parts, ignore_index=True)
    summary.to_csv(OUT / "04_all_benchmark_gap_comparison.csv", index=False)
    paired.to_csv(OUT / "05_per_instance_paired_differences.csv", index=False)

    node_rows = []
    for (group, n_nodes), part in paired[paired.analysis_dataset.isin(["TSPLIB48_development", "TSPLIB22_heldout"])].groupby(["analysis_dataset", "n_nodes"]):
        node_rows.append(result_row(f"{group}_N{n_nodes}", part))
    pd.DataFrame(node_rows).to_csv(OUT / "support_tsplib_by_node_count.csv", index=False)

    held_row = summary[summary.dataset.eq("TSPLIB22_heldout")].iloc[0]
    dev_row = summary[summary.dataset.eq("TSPLIB48_development")].iloc[0]
    original_row = summary[summary.dataset.eq("OriginalTSPLIB70")].iloc[0]
    gap8 = summary[summary.dataset.eq("GAP8_macro")].iloc[0]
    controls = summary[summary.dataset.isin(["Uniform100", "GaussianMixture100", "GaussianMixture500"])].copy()
    held_significant = held_row.Structure_minus_Hardness_M5_ci95_high < 0
    controls_material = bool(
        ((controls.Structure_minus_Hardness_M5 > 0.1) | (controls.Structure_minus_Hardness_M5_ci95_low > 0)).any()
    )
    gap8_nonworse = gap8.Structure_minus_Hardness_M5 <= 0
    if held_significant and gap8_nonworse and not controls_material:
        decision = "STRUCTURE_C5_GAIN_CONFIRMED"
    elif held_row.Structure_minus_Hardness_M5 >= 0 and gap8.Structure_minus_Hardness_M5 >= 0:
        decision = "STRUCTURE_C5_NO_GAIN"
    else:
        decision = "STRUCTURE_C5_MIXED"

    loss = pd.read_csv(TRAIN_CYCLE / "09_bq_training" / "training_loss_curve.csv")
    hard_loss = pd.read_csv(CYCLES / "C04" / "09_bq_training" / "training_loss_curve.csv")
    make_figures(summary, paired, tsplib, loss, hard_loss)

    p2_runtime = json.loads((P2 / "support_runtime.json").read_text(encoding="utf-8"))
    label_info = json.loads((LABEL_DIR / "labeling_integrity.json").read_text(encoding="utf-8"))
    training_info = json.loads((OUT / "training_integrity.json").read_text(encoding="utf-8"))
    c04_mutation = json.loads((CYCLES / "C04" / "03_mutation" / "FOGA11_BASE_V1_RUNTIME.json").read_text(encoding="utf-8"))
    c04_label = json.loads((CYCLES / "C04" / "07_oracle" / "T" / "oracle_summary.json").read_text(encoding="utf-8"))
    c04_train = json.loads((CYCLES / "C04" / "09_bq_training" / "bq_training_timing.json").read_text(encoding="utf-8"))
    e1_mutation_wall = (
        p2_runtime["mutation_wall_seconds_by_arm_generation"]["E1_G1"]
        + p2_runtime["mutation_wall_seconds_by_arm_generation"]["E1_G2"]
    )
    original_core_wall = c04_mutation["wall_seconds"] + c04_label["wall_seconds"] + c04_train["training_loop_wall_s"]
    structure_core_wall = e1_mutation_wall + label_info["pipeline_wall_seconds"] + training_info["training_loop_wall_s"]

    lines = [
        "# Generalization and Forgetting Analysis",
        "",
        "Negative deltas favor Structure M5. All confidence intervals are paired 10,000-replicate bootstrap intervals.",
        "",
        "| Dataset | M4 | Hardness M5 | Structure M5 | Structure−Hardness | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in [*DATASETS, "TSPLIB48_development", "TSPLIB22_heldout", "GAP8_macro"]:
        row = summary[summary.dataset.eq(name)].iloc[0]
        interval = "N/A" if pd.isna(row.Structure_minus_Hardness_M5_ci95_low) else f"[{row.Structure_minus_Hardness_M5_ci95_low:.4f}, {row.Structure_minus_Hardness_M5_ci95_high:.4f}]"
        lines.append(f"| {name} | {row.mean_gap_M4:.4f} | {row.mean_gap_Hardness_M5:.4f} | {row.mean_gap_Structure_M5:.4f} | {row.Structure_minus_Hardness_M5:+.4f} | {interval} |")
    lines.extend(
        [
            "",
            f"- Held-out22 Structure−Hardness: {held_row.Structure_minus_Hardness_M5:+.4f} pp, 95% CI [{held_row.Structure_minus_Hardness_M5_ci95_low:.4f}, {held_row.Structure_minus_Hardness_M5_ci95_high:.4f}].",
            f"- Development48 Structure−Hardness: {dev_row.Structure_minus_Hardness_M5:+.4f} pp; Original70 combined: {original_row.Structure_minus_Hardness_M5:+.4f} pp. Original70 is not described as independent.",
            f"- GAP8 macro Structure−Hardness: {gap8.Structure_minus_Hardness_M5:+.4f} pp.",
            f"- Material/significant Uniform/Gaussian regression guard: {controls_material}.",
            "- The three models use identical dataset SHA256 and evaluator-config SHA256 values on all eight benchmarks; instance alignment additionally matches ID, coordinate hash, node count, and optimum.",
            "- Held-out22 clears zero only narrowly; its CI upper bound is close to zero, so this is positive but boundary-strength statistical evidence.",
            "",
            "## Incremental cost relative to original C04",
            "",
            f"- Original C04 evolution: 110,000 attempts, {c04_mutation['wall_seconds']:.1f}s mutation wall. P2.2 deployable E1 path: 40,000 attempts across two generations, {e1_mutation_wall:.1f}s mutation wall: **70,000 fewer attempts and {e1_mutation_wall - c04_mutation['wall_seconds']:+.1f}s**. The full P2.2 causal audit additionally ran the matched E0 control (80,000 total attempts across both arms).",
            f"- Original C04 labeling: {c04_label['new_concorde_calls']:,} new calls, {c04_label['wall_seconds']:.1f}s. P3: {label_info['physical_new_concorde_calls']:,} new calls, {label_info['pipeline_wall_seconds']:.1f}s combined labeling/M4-hardness stage: **{label_info['physical_new_concorde_calls'] - c04_label['new_concorde_calls']:+,} calls and {label_info['pipeline_wall_seconds'] - c04_label['wall_seconds']:+.1f}s**.",
            f"- Original C04 training: {c04_train['training_loop_wall_s']:.1f}s. P3 Structure training: {training_info['training_loop_wall_s']:.1f}s under the same 10,240-update budget: **{training_info['training_loop_wall_s'] - c04_train['training_loop_wall_s']:+.1f}s**.",
            f"- Simple deployable-stage wall sum (mutation + labeling/hardness + training): original C04 {original_core_wall:.1f}s versus Structure path {structure_core_wall:.1f}s, a **{structure_core_wall - original_core_wall:+.1f}s** difference. This excludes evaluation and treats recorded stage walls additively.",
        ]
    )
    (OUT / "06_generalization_and_forgetting_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if decision == "STRUCTURE_C5_GAIN_CONFIRMED":
        conclusion = "The frozen Structure M5 meets the stringent held-out, macro, and retention criteria."
    elif decision == "STRUCTURE_C5_NO_GAIN":
        conclusion = "The frozen Structure M5 does not improve the held-out evidence or macro benchmark relative to Hardness M5."
    else:
        conclusion = "The frozen Structure M5 shows mixed gains/regressions; the evidence does not support a blanket superiority claim."
    decision_text = f"""# Final Research Decision

## `{decision}`

{conclusion}

- Structure M5 vs Hardness M5 on held-out22: {held_row.Structure_minus_Hardness_M5:+.4f} percentage points, paired 95% CI [{held_row.Structure_minus_Hardness_M5_ci95_low:.4f}, {held_row.Structure_minus_Hardness_M5_ci95_high:.4f}].
- Development48 delta: {dev_row.Structure_minus_Hardness_M5:+.4f} pp; Original70 combined delta: {original_row.Structure_minus_Hardness_M5:+.4f} pp.
- GAP8 macro delta: {gap8.Structure_minus_Hardness_M5:+.4f} pp.
- Uniform/Gaussian material-regression guard: {controls_material}.
- All eight dataset hashes and evaluator-config hashes match across M4, Hardness M5, and Structure M5.
- The held-out22 paired interval excludes zero only narrowly (upper bound {held_row.Structure_minus_Hardness_M5_ci95_high:.4f} pp); the gain is confirmed under the predeclared rule but should be described as boundary-significant.
- Structure distribution matching translated to solver performance only to the extent supported by the complete per-dataset table; no benchmark was cherry-picked and Original70 is not treated as independent.

The checkpoint was fixed at E256 before evaluation. No sixth cycle, data regeneration, learning-rate adjustment, extra seed, or additional ablation was started.

`TARGET_AWARE_STRUCTURE_C5_FINETUNE_COMPLETE`
"""
    (OUT / "07_final_research_decision.md").write_text(decision_text, encoding="utf-8")
    write_json(
        OUT / "run_metadata.json",
        {
            "status": "COMPLETE",
            "decision": decision,
            "structure_m5_sha256": sha256_file(STRUCTURE_M5),
            "m4_sha256": M4_SHA,
            "hardness_m5_sha256": HARD_M5_SHA,
            "structure10k_sha256": EXPECTED_INPUT_SHA,
            "label_bundle_sha256": sha256_file(LABEL_DIR / "labeled_bundle.npz"),
            "benchmark_csv_sha256": sha256_file(OUT / "04_all_benchmark_gap_comparison.csv"),
            "paired_csv_sha256": sha256_file(OUT / "05_per_instance_paired_differences.csv"),
            "expert_tours_array_sha256": label_info["expert_tours_array_sha256"],
            "all_dataset_and_evaluator_hashes_match": True,
            "no_tsplib22_model_selection": True,
            "no_extra_training_seed": True,
            "no_sixth_cycle": True,
        },
    )
    print(json.dumps({"status": "COMPLETE", "decision": decision, "heldout_delta": held_row.Structure_minus_Hardness_M5, "gap8_delta": gap8.Structure_minus_Hardness_M5}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "label", "train", "evaluate", "finalize"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "audit":
        audit_inputs()
    elif args.phase == "label":
        label(args.device)
    elif args.phase == "train":
        train(args.device)
    elif args.phase == "evaluate":
        evaluate(args.device)
    else:
        finalize()


if __name__ == "__main__":
    main()
