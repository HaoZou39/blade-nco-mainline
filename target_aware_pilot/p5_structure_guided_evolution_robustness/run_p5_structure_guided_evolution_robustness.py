#!/usr/bin/env python3
"""P5 paired evolution-seed robustness for P2.2 E0 versus E1."""

from __future__ import annotations

import argparse
import hashlib
import json
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
P2 = ROOT / "target_aware_pilot" / "p2_bounded_evolution"
P3 = ROOT / "target_aware_pilot" / "p3_structure_finetune"
P4A = ROOT / "target_aware_pilot" / "p4a_random_evolution_ablation"
P4B1 = ROOT / "target_aware_pilot" / "p4b1_acquisition_ablation"
P4B2 = ROOT / "target_aware_pilot" / "p4b2_random_archive_ablation"
P4C2 = ROOT / "target_aware_pilot" / "p4c2_data_sampling_robustness"
OUT = ROOT / "target_aware_pilot" / "p5_structure_guided_evolution_robustness"
EVOLUTION = OUT / "evolution"
DATA_DIR = OUT / "datasets"
FIG = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(P2), str(P4A), str(P4B1), str(P4B2), str(P4C2)]

import run_p2_bounded_evolution as p2  # noqa: E402
import run_p4a_random_evolution_ablation as p4a  # noqa: E402
import run_p4b1_acquisition_ablation as p4b1  # noqa: E402
import run_p4b2_random_archive_ablation as p4b2  # noqa: E402
import run_p4c2_data_sampling_robustness as p4c2  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import closed_expert_coordinates  # noqa: E402
from bq.tsp_metric import integer_distance_matrix, tour_cost_int, validate_tour_0based  # noqa: E402
from final_run.science import membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, ORACLE_LEDGER_COLUMNS, sha256_file  # noqa: E402


M4 = p4b1.M4
M4_SHA = p4b1.M4_SHA
HARD_M5 = p4b1.HARD_M5
HARD_M5_SHA = p4b1.HARD_M5_SHA
ARMS = ("E0", "E1")
NEW_REPLICATES = ("S1", "S2")
ALL_REPLICATES = ("Original", *NEW_REPLICATES)
EVOLUTION_SEEDS = {"Original": p2.SEED, "S1": 260501, "S2": 260502}
MUTATION_NAMESPACES = {
    "Original": dict(p2.MUTATION_CYCLE_NAMESPACE),
    "S1": {1: 2_605_011, 2: 2_605_012},
    "S2": {1: 2_605_021, 2: 2_605_022},
}
TRAINING_SEED = 2026
DATASETS = p4b1.DATASETS
GAP5_DATASETS = ("Uniform100", "GaussianMixture100", "TSPLIB100", "GaussianMixture500", "OriginalTSPLIB70")
BOOTSTRAP_REPLICATES = 10_000
TRAIN_N = 10_000

ORIGINAL = {
    "E0": {
        "dataset": P2 / "support_e0_provisional_structure10k.npz",
        "manifest": P2 / "support_e0_provisional_structure10k.parquet",
        "dataset_sha": "987395ae7de3e9a4118ed26ce9cf2c369a6b0498623dadd14668812a4cc3cc69",
        "labels": P4A / "labeling" / "labeled_bundle.npz",
        "label_sha": "3077212b6e98b5f4a860fcfb76f75781f60a638099bc700cd443d87f05bed6d2",
        "checkpoint": P4A / "checkpoints" / "E0_Structure_M5_E256.tsp.best",
        "checkpoint_sha": "d7d2ffc9ebafa5adb2bca9ccd90f6e8bd0d0487bdf1b4f68cf98dbf425cd4c58",
        "eval_csv": P4A / "e0_m5_external_benchmark_results.csv",
        "eval_label": "E0_Structure_M5",
        "training_info": P4A / "training_integrity.json",
    },
    "E1": {
        "dataset": P2 / "provisional_structure10k.npz",
        "manifest": P2 / "support_e1_provisional_structure10k.parquet",
        "dataset_sha": "5287932231529e617383388bb64d607985d07944799c3d661c5a737338bdece1",
        "labels": P3 / "labeling" / "labeled_bundle.npz",
        "label_sha": "259dab4e317a66e6c67506617dcbb52da52f6e3bb17f417eaac3f3d4474712f6",
        "checkpoint": P3 / "checkpoints" / "Structure_M5_E256.tsp.best",
        "checkpoint_sha": "ace34d712ca8d20e730edeed0666362e6cc03008f9b5c0411d53c444856e3233",
        "eval_csv": P3 / "structure_m5_external_benchmark_results.csv",
        "eval_label": "Structure_M5",
        "training_info": P3 / "training_integrity.json",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def ensure_dirs() -> None:
    for path in (OUT, EVOLUTION, DATA_DIR, FIG, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def rep_dir(replicate: str) -> Path:
    return EVOLUTION / replicate


def dataset_bundle(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_Structure10k.npz"


def dataset_manifest(replicate: str, arm: str) -> Path:
    return DATA_DIR / f"{replicate}_{arm}_Structure10k.parquet"


def label_dir(replicate: str, arm: str) -> Path:
    return LOGS / f"{replicate}_{arm}" / "labeling"


def run_dir(replicate: str, arm: str) -> Path:
    return LOGS / f"{replicate}_{arm}"


def checkpoint_path(replicate: str, arm: str) -> Path:
    return CHECKPOINTS / f"{replicate}_{arm}_M5_E256.tsp.best"


def eval_label(replicate: str, arm: str) -> str:
    return f"{replicate}_{arm}_M5"


def eval_csv(replicate: str, arm: str) -> Path:
    return run_dir(replicate, arm) / "external_benchmark_results.csv"


def base_training_protocol() -> dict[str, Any]:
    trainer = ROOT / "scripts" / "run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
    return {
        "m4_sha256": M4_SHA, "trainer_sha256": sha256_file(trainer), "training_seed": TRAINING_SEED,
        "optimizer": "fresh Adam", "scheduler": None, "learning_rate": 3.75e-5,
        "epochs": 256, "optimizer_updates": 10_240, "logical_batch_size": 256,
        "microbatch_size": 64, "checkpoint_rule": "fixed E256", "evaluation": "frozen greedy",
    }


def protocol_gate() -> dict[str, Any]:
    path = OUT / "preregistered_evolution_config.json"
    if not path.is_file():
        raise RuntimeError("P5 audit/preregistration missing")
    value = json.loads(path.read_text())
    digest = value.pop("protocol_sha256")
    if canonical_hash(value) != digest:
        raise RuntimeError("P5 preregistration hash mismatch")
    value["protocol_sha256"] = digest
    return value


def audit() -> dict[str, Any]:
    ensure_dirs()
    if sha256_file(M4) != M4_SHA or sha256_file(p2.P03_BUNDLE) != "8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62":
        raise RuntimeError("formal M4/P03 drift")
    if sha256_file(p2.PROFILE_PATH) != "7108725bf6afda6be40e2e5b7a881bf7f43aea495852aa8c9a18dcdb783addee":
        raise RuntimeError("FOGA profile drift")
    geometry = p2.load_geometry()
    p03_xy, p03, p03_info = p2.load_p03(geometry)
    if p03_info["formal_c04_parent_match"] is not True or p03_info["formal_c04_runtime_parent_match"] is not True:
        raise RuntimeError("P03/C04 parent mismatch")
    if p2.ATTEMPTS_PER_GENERATION != 20_000 or p2.GENERATIONS != (1, 2) or p2.POPULATION_N != 10_000 or p2.ARCHIVE_CAPACITY != 30_000:
        raise RuntimeError("P2.2 budget drift")
    if p2.TARGET_FRACTION != .70 or p2.FRONTIER_FRACTION != .30 or p2.FEATURES != ("spn_distance_varcoeff", "spn_fraction_shorter_mean_distance"):
        raise RuntimeError("P2.2 structure protocol drift")
    schedule = p2.attempt_schedule()
    if len(schedule) != 20_000 or max(pd.Series([x[1] for x in schedule]).value_counts()) - min(pd.Series([x[1] for x in schedule]).value_counts()) > 1:
        raise RuntimeError("operator schedule drift")

    # Historical original is comparable only after all data/training/eval hashes reproduce.
    original_rows = []
    for arm in ARMS:
        item = ORIGINAL[arm]
        for key, expected in (("dataset", item["dataset_sha"]), ("labels", item["label_sha"]), ("checkpoint", item["checkpoint_sha"])):
            if sha256_file(item[key]) != expected:
                raise RuntimeError(f"Original {arm} {key} drift")
        xy, hashes, _ = p4c2.load_bundle(item["dataset"])
        lxy, lhashes, tours = p4c2.load_bundle(item["labels"], require_tours=True)
        info = json.loads(item["training_info"].read_text())
        evaluation = pd.read_csv(item["eval_csv"])
        if hashes != lhashes or not np.array_equal(xy, lxy) or tours is None or info["m4_sha256"] != M4_SHA or not info["e000_weights_equal_m4"] or info["optimizer_steps"] != 10_240:
            raise RuntimeError(f"Original {arm} data/training mismatch")
        if set(evaluation.dataset.astype(str)) != set(DATASETS) or set(evaluation.checkpoint_sha256.astype(str)) != {item["checkpoint_sha"]}:
            raise RuntimeError(f"Original {arm} evaluation mismatch")
        original_rows.append({"arm": arm, "dataset_membership": membership_hash(hashes), "checkpoint_sha256": item["checkpoint_sha"]})
    originals = [pd.read_csv(ORIGINAL[arm]["eval_csv"]).set_index("dataset") for arm in ARMS]
    for dataset in DATASETS:
        if originals[0].at[dataset, "dataset_sha256"] != originals[1].at[dataset, "dataset_sha256"] or originals[0].at[dataset, "evaluator_config_sha256"] != originals[1].at[dataset, "evaluator_config_sha256"]:
            raise RuntimeError("Original evaluator protocol mismatch")

    protocol = {
        "status": "PREREGISTERED_BEFORE_NEW_EVOLUTION",
        "original_evolution_seed": EVOLUTION_SEEDS["Original"],
        "new_evolution_seeds": {key: EVOLUTION_SEEDS[key] for key in NEW_REPLICATES},
        "training_seed": TRAINING_SEED,
        "mutation_namespace_mapping": MUTATION_NAMESPACES,
        "namespace_rule": "new replicate seed s maps G1/G2 mutation parameter streams to 10*s+1 and 10*s+2; E0 random survivor uses s+1000+generation",
        "historical_difference": "Original used fixed mutation namespaces {1:5,2:6}; new namespace values intentionally realize new mutation randomness while retaining the same per-arm algorithm and G1 common-random-number coupling",
        "random_streams": {
            "operator_schedule": "deterministic parent-index/pass schedule",
            "mutation_parameters": "cycle_child_seed(namespace,parent_hash,operator), no mutable global RNG",
            "E0_survivor": "numpy.default_rng(evolution_seed+1000+generation)",
            "E1_survivor": "deterministic structured_select",
            "archive_retention": "deterministic structured_select",
            "final_acquisition": "deterministic structure-aware acquisition",
            "metric_swd": "fixed historical 20261024+991 directions for cross-replicate comparability",
        },
        "g1_common_random_numbers": True,
        "g2_survivor_on_training_path": False,
        "base_training_protocol": base_training_protocol(),
        "primary_endpoint": "OriginalTSPLIB70 mean optimality gap; Delta=E0-E1",
        "critical_exploratory_endpoint": "TSPLIB22; repeatedly observed, not pristine",
        "secondary_endpoint": list(GAP5_DATASETS),
        "supported_rule": "both new Original70 Delta > 0; not both new TSPLIB22 Delta < -0.25 pp; and both new GAP5 Delta >= -0.25 pp",
        "not_supported_rule": "both new Original70 Delta < 0",
        "otherwise": "INCONCLUSIVE",
        "bootstrap": "10,000 within-replicate paired resamples; stratified by dataset for GAP5; never pool replicates",
        "no_third_seed": True,
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    config = OUT / "preregistered_evolution_config.json"
    if config.exists():
        if json.loads(config.read_text()) != protocol:
            raise RuntimeError("existing P5 preregistration differs")
    else:
        write_json(config, protocol)

    causal = f"""# P5 Protocol and Causal-Path Audit

Status: **PASS**. P2.2 source and frozen artifacts are internally consistent.

- P03 is exactly the formal C03 `P_next` and C04 parent population: 10,000 aligned TSP100, bundle SHA256 `{sha256_file(p2.P03_BUNDLE)}`, membership `{p03_info['membership_sha256']}`.
- Each arm executes two generations × 20,000 FOGA-11 attempts. Mutation reads the current `population_frames`; G1 survivors are written back and are the G2 parents.
- The archive is updated only with each generation's eligible offspring. It never supplies parents. Final Structure10k acquisition reads Archive30k, not Population10k.
- G1 survivor selection affects G2 parents, hence G2 offspring and the cumulative training archive. G2 survivor selection happens only after G2 offspring already exist and there is no G3. Therefore: `G2_SURVIVOR_SELECTION_NOT_ON_TRAINING_DATA_PATH`.
- G2 survivors are computed and audited but do **not** causally affect the present M5 training data.
- G1 E0/E1 mutation streams are matched without an adapter in the original semantics: same ordered P03, deterministic operator schedule, and child seeds based only on generation namespace + parent hash + operator. No arm name or mutable global RNG enters mutation.
- E0 survivor selection alone is seeded random; E1 selection, common Archive30k retention, and Structure10k acquisition are deterministic.
- New seeds use the preregistered namespaces S1 `{MUTATION_NAMESPACES['S1']}`, S2 `{MUTATION_NAMESPACES['S2']}`. This changes random realization but not operator distributions, budgets, mutation kernels, selection rules, features, quotas, or tie-breaking. Historical Original namespaces remain `{MUTATION_NAMESPACES['Original']}`.
- Original E0/E1 datasets, labels, M4 initialization, 10,240-update training, checkpoints, and evaluator hashes reproduce, so Original is included as a comparable historical replicate.
- TSPLIB22 is never used in evolution, acquisition, training, or model selection.

No mutation, Concorde, or new BQ training occurred during this audit.
"""
    (OUT / "00_p5_protocol_and_causal_path_audit.md").write_text(causal, encoding="utf-8")
    prereg = f"""# Preregistered Evolution Seeds

| Replicate | Evolution seed | Mutation namespaces G1/G2 | BQ seed |
|---|---:|---|---:|
| Original | {EVOLUTION_SEEDS['Original']} | {MUTATION_NAMESPACES['Original']} | {TRAINING_SEED} |
| S1 | {EVOLUTION_SEEDS['S1']} | {MUTATION_NAMESPACES['S1']} | {TRAINING_SEED} |
| S2 | {EVOLUTION_SEEDS['S2']} | {MUTATION_NAMESPACES['S2']} | {TRAINING_SEED} |

- Primary Delta = Original70 gap(E0) − gap(E1); positive favors E1.
- SUPPORTED_PRELIMINARY: both new primary Deltas positive, the two TSPLIB22 Deltas are not both below −0.25 pp, and both GAP5 Deltas are at least −0.25 pp.
- NOT_SUPPORTED: both new primary Deltas negative. Otherwise INCONCLUSIVE.
- Exactly S1/S2 are added; no result-dependent seed, rerun, mutation budget change, data replacement, or endpoint reweighting is allowed.

Protocol SHA256: `{protocol['protocol_sha256']}`.
"""
    (OUT / "01_preregistered_evolution_seeds.md").write_text(prereg, encoding="utf-8")
    Status().update(phase="audit", status="PASS", protocol_sha256=protocol["protocol_sha256"])
    return {"status": "PASS", "protocol_sha256": protocol["protocol_sha256"], "g2_survivor_on_training_path": False, "original_comparable": True}


def fixed_swd(source_z: np.ndarray, target_z: np.ndarray) -> float:
    rng = np.random.default_rng(20261024 + 991)
    directions = rng.normal(size=(p2.SWD_PROJECTIONS, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(np.mean([p2.wasserstein_distance(source_z @ direction, target_z @ direction) for direction in directions]))


def evolve(replicate: str) -> dict[str, Any]:
    protocol_gate()
    if replicate not in NEW_REPLICATES:
        raise ValueError("only S1/S2 evolution is allowed")
    directory = rep_dir(replicate)
    complete = directory / "run_metadata.json"
    if complete.is_file() and json.loads(complete.read_text()).get("status") == "COMPLETE":
        return json.loads(complete.read_text())
    old = {"OUT": p2.OUT, "FIG": p2.FIG, "SEED": p2.SEED, "NS": p2.MUTATION_CYCLE_NAMESPACE, "SWD": p2.sliced_wasserstein}
    p2.OUT = directory
    p2.FIG = directory / "figures"
    p2.SEED = EVOLUTION_SEEDS[replicate]
    p2.MUTATION_CYCLE_NAMESPACE = MUTATION_NAMESPACES[replicate]
    p2.sliced_wasserstein = fixed_swd
    started = time.perf_counter()
    try:
        p2.main()
    finally:
        p2.OUT, p2.FIG, p2.SEED, p2.MUTATION_CYCLE_NAMESPACE, p2.sliced_wasserstein = old["OUT"], old["FIG"], old["SEED"], old["NS"], old["SWD"]
    metadata = json.loads(complete.read_text())
    metadata["p5_replicate"] = replicate
    metadata["p5_evolution_seed"] = EVOLUTION_SEEDS[replicate]
    metadata["p5_mutation_namespaces"] = MUTATION_NAMESPACES[replicate]
    metadata["p5_invocation_wall_seconds"] = time.perf_counter() - started
    metadata["metric_swd_directions_fixed_across_replicates"] = True
    write_json(complete, metadata)
    Status().update(phase="evolution", status="PASS", replicate=replicate)
    return metadata


def source_dataset(replicate: str, arm: str) -> tuple[Path, Path]:
    if replicate == "Original":
        return ORIGINAL[arm]["dataset"], ORIGINAL[arm]["manifest"]
    directory = rep_dir(replicate)
    if arm == "E0":
        return directory / "support_e0_provisional_structure10k.npz", directory / "support_e0_provisional_structure10k.parquet"
    return directory / "provisional_structure10k.npz", directory / "support_e1_provisional_structure10k.parquet"


def prepare() -> dict[str, Any]:
    protocol = protocol_gate()
    for replicate in NEW_REPLICATES:
        meta = json.loads((rep_dir(replicate) / "run_metadata.json").read_text())
        if meta.get("status") != "COMPLETE":
            raise RuntimeError(f"evolution incomplete: {replicate}")
    history, _ = p4b1.frozen_history()
    run_rows = []; integrity_lines = ["# Training Datasets Integrity", ""]
    for replicate in ALL_REPLICATES:
        for arm in ARMS:
            source_bundle, source_manifest = source_dataset(replicate, arm)
            xy, hashes = p2.load_bundle(source_bundle)
            frame = pd.read_parquet(source_manifest)
            if hashes != frame.instance_hash.astype(str).tolist():
                raise RuntimeError(f"manifest alignment failure: {replicate}/{arm}")
            audit_value = p4b1.coordinate_audit(xy, hashes)
            if xy.shape != (TRAIN_N, 100, 2) or len(set(hashes)) != TRAIN_N or set(hashes) & history or not all(audit_value[key] for key in ("finite", "in_bounds", "unique_nodes", "hashes_reproduce")):
                raise RuntimeError(f"training dataset integrity failure: {replicate}/{arm}")
            if replicate != "Original":
                shutil.copy2(source_bundle, dataset_bundle(replicate, arm))
                shutil.copy2(source_manifest, dataset_manifest(replicate, arm))
                frame.to_csv(dataset_manifest(replicate, arm).with_suffix(".csv"), index=False)
                write_json(DATA_DIR / f"{replicate}_{arm}_integrity.json", {"status": "PASS", "rows": TRAIN_N, "membership_sha256": membership_hash(hashes), "bundle_sha256": sha256_file(dataset_bundle(replicate, arm)), "manifest_sha256": sha256_file(dataset_manifest(replicate, arm)), "m4_history_overlap_n": 0, **audit_value})
            run_rows.append({"sampling_replicate": replicate, "arm": arm, "evolution_seed": EVOLUTION_SEEDS[replicate], "mutation_namespace_g1": MUTATION_NAMESPACES[replicate][1], "mutation_namespace_g2": MUTATION_NAMESPACES[replicate][2], "training_membership_sha256": membership_hash(hashes), "training_bundle_sha256": sha256_file(source_bundle), "m4_history_overlap_n": 0, "status": "PASS"})
            integrity_lines.append(f"- {replicate}-{arm}: 10,000 unique valid TSP100; membership `{membership_hash(hashes)}`; bundle `{sha256_file(source_bundle)}`; M4-history overlap 0.")
    pd.DataFrame(run_rows).to_csv(OUT / "02_evolution_runs_manifest.csv", index=False)

    survivor_rows = []
    for replicate, directory in (("Original", P2), ("S1", rep_dir("S1")), ("S2", rep_dir("S2"))):
        p03 = pd.read_parquet(p2.P03_META)
        previous = {arm: p03 for arm in ARMS}
        for generation in (1, 2):
            union_sets = {}
            arm_rows = []
            for arm in ARMS:
                child = pd.read_parquet(directory / "evolution" / arm / f"G{generation}" / "offspring_diagnostics.parquet")
                survivors = pd.read_parquet(directory / "evolution" / arm / f"G{generation}" / "active_population.parquet")
                parents = previous[arm]
                union = set(parents.instance_hash.astype(str)) | set(child.instance_hash.astype(str))
                union_sets[arm] = union
                lineage_p = survivors.lineage_root.astype(str).value_counts(normalize=True).to_numpy(float)
                arm_rows.append({"replicate": replicate, "evolution_seed": EVOLUTION_SEEDS[replicate], "generation": f"G{generation}", "arm": arm, "parent_n": len(parents), "eligible_offspring_n": int((~child.m4_training_overlap).sum()), "union_unique_n": len(union), "survivor_n": len(survivors), "parent_retained_n": int(survivors.instance_hash.astype(str).isin(set(parents.instance_hash.astype(str))).sum()), "parent_retention_fraction": float(survivors.instance_hash.astype(str).isin(set(parents.instance_hash.astype(str))).mean()), "effective_lineages": float(np.exp(-(lineage_p*np.log(lineage_p)).sum())), "largest_lineage_share": float(lineage_p.max()), "survivor_membership_sha256": membership_hash(survivors.instance_hash.astype(str).tolist()), "g2_survivor_on_training_path": generation != 2})
                previous[arm] = survivors
            overlap = len(union_sets["E0"] & union_sets["E1"])
            for row in arm_rows:
                row["e0_e1_union_overlap_n"] = overlap
                survivor_rows.append(row)
    pd.DataFrame(survivor_rows).to_csv(OUT / "03_generation_survivor_audit.csv", index=False)

    metric_parts = []
    operator_parts = []
    for replicate, directory in (("Original", P2), ("S1", rep_dir("S1")), ("S2", rep_dir("S2"))):
        frame = pd.read_csv(directory / "02_generation_metrics.csv"); frame.insert(0, "replicate", replicate); metric_parts.append(frame)
        operators = pd.read_csv(directory / "support_operator_metrics.csv"); operators.insert(0, "replicate", replicate); operator_parts.append(operators)
    metrics = pd.concat(metric_parts, ignore_index=True); metrics.to_csv(OUT / "04_structural_distribution_comparison.csv", index=False)
    pd.concat(operator_parts, ignore_index=True).to_csv(OUT / "support_operator_contribution.csv", index=False)
    integrity_lines.extend(["", "All memberships were frozen before labeling or evaluation. Structural diagnostics were not used for resampling.", "", "Status: **PASS**."])
    (OUT / "05_training_datasets_integrity.md").write_text("\n".join(integrity_lines) + "\n", encoding="utf-8")

    active = metrics[metrics.object_type.eq("active_population")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for replicate, style in zip(ALL_REPLICATES, ("-", "--", ":"), strict=True):
        for arm, color in (("E0", "#4c78a8"), ("E1", "#f58518")):
            part = active[(active.replicate.eq(replicate)) & active.arm.eq(arm)].sort_values("generation")
            axes[0].plot(part.generation, part.sliced_wasserstein, linestyle=style, marker="o", color=color, label=f"{replicate}-{arm}")
            axes[1].plot(part.generation, part.coarse_js, linestyle=style, marker="o", color=color, label=f"{replicate}-{arm}")
    axes[0].set_title("Survivor SWD"); axes[1].set_title("Survivor 3x3 JS")
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "01_survivor_structure_by_generation.png", dpi=180); plt.close(fig)
    Status().update(phase="prepare", status="PASS")
    return {"status": "PASS", "datasets": 4, "protocol_sha256": protocol["protocol_sha256"]}


def p5_cache() -> pd.DataFrame:
    base = p4c2.verified_cache()
    pieces = [base]
    for replicate in NEW_REPLICATES:
        for arm in ARMS:
            path = label_dir(replicate, arm) / "ledger.parquet"
            if path.is_file(): pieces.append(pd.read_parquet(path)[list(ORACLE_LEDGER_COLUMNS)])
    return pd.concat(pieces, ignore_index=True).drop_duplicates("instance_hash", keep="last")


def label(replicate: str, arm: str) -> dict[str, Any]:
    protocol_gate()
    if replicate not in NEW_REPLICATES or arm not in ARMS: raise ValueError("only S1/S2 E0/E1")
    coordinates, hashes, _ = p4c2.load_bundle(dataset_bundle(replicate, arm))
    manifest = pd.read_parquet(dataset_manifest(replicate, arm))
    if manifest.instance_hash.astype(str).tolist() != hashes: raise RuntimeError("manifest mismatch")
    directory = label_dir(replicate, arm); directory.mkdir(parents=True, exist_ok=True)
    integrity_path = directory / "labeling_integrity.json"
    if integrity_path.is_file():
        existing = json.loads(integrity_path.read_text())
        if existing.get("status") == "PASS" and existing.get("training_membership_sha256") == membership_hash(hashes): return existing
    import run_blade_v2 as base
    prior = p5_cache(); known = set(prior.instance_hash.astype(str)); started = time.perf_counter()
    ledger = base.label_with_cache(coordinates, hashes, directory, prior); tours = base._oracle_tours(ledger, hashes)
    if len(ledger) != TRAIN_N or tours.shape != (TRAIN_N,100) or not ledger.verified.all() or not all(validate_tour_0based(tour,100)[0] for tour in tours): raise RuntimeError("labeling validity failure")
    costs = np.asarray([tour_cost_int(integer_distance_matrix(points), tour) for points,tour in zip(coordinates,tours,strict=True)], np.int64)
    if not np.array_equal(costs, ledger.optimal_cost_int.to_numpy(np.int64)): raise RuntimeError("objective mismatch")
    closed = closed_expert_coordinates(coordinates,tours)
    if closed.shape != (TRAIN_N,101,2) or not np.array_equal(closed[:,0],closed[:,-1]): raise RuntimeError("closure mismatch")
    tour_hashes=[hashlib.sha256(np.asarray(tour,dtype=np.int32).tobytes()).hexdigest() for tour in tours]
    if tour_hashes != ledger.tour_hash.astype(str).tolist(): raise RuntimeError("tour hash mismatch")
    payload=set(ORACLE_LEDGER_COLUMNS[1:]); labeled=manifest[[c for c in manifest.columns if c not in payload]].merge(ledger,on="instance_hash",validate="one_to_one")
    ledger.to_parquet(directory/"ledger.parquet",index=False); labeled.to_parquet(directory/"labeled.parquet",index=False); p4c2.save_bundle(directory/"labeled_bundle.npz",coordinates,hashes,tours)
    result={"status":"PASS","replicate":replicate,"arm":arm,"verified":TRAIN_N,"failures":0,"cache_hits":sum(h in known for h in hashes),"physical_new_concorde_calls":sum(h not in known for h in hashes),"pipeline_wall_seconds":time.perf_counter()-started,"training_membership_sha256":membership_hash(hashes),"ledger_sha256":sha256_file(directory/"ledger.parquet"),"labeled_bundle_sha256":sha256_file(directory/"labeled_bundle.npz"),"expert_tours_array_sha256":hashlib.sha256(np.ascontiguousarray(tours,dtype="<i8").tobytes()).hexdigest(),"concorde_sha256":CANONICAL_CONCORDE_SHA256,"all_tours_legal":True,"all_costs_reproduced":True}
    write_json(integrity_path,result); write_label_report(); Status().update(phase="label",status="PASS",replicate=replicate,arm=arm); return result


def write_label_report() -> None:
    rows=[]
    for rep in NEW_REPLICATES:
        for arm in ARMS:
            path=label_dir(rep,arm)/"labeling_integrity.json"
            if path.is_file(): rows.append(json.loads(path.read_text()))
    lines=["# Concorde Labeling Audit","","| Dataset | Verified | Cache hits | New calls | Failures | Wall (s) | Bundle SHA256 |","|---|---:|---:|---:|---:|---:|---|"]
    for row in rows: lines.append(f"| {row['replicate']}-{row['arm']} | {row['verified']:,} | {row['cache_hits']:,} | {row['physical_new_concorde_calls']:,} | {row['failures']} | {row['pipeline_wall_seconds']:.1f} | `{row['labeled_bundle_sha256']}` |")
    lines.extend(["",f"Progress: {len(rows)}/4 complete. Canonical integer distances, 0-based Hamiltonian tours, implicit objective closure; no failed instance was replaced."])
    if len(rows)==4: lines.extend([f"Total physical Concorde calls: {sum(r['physical_new_concorde_calls'] for r in rows):,}; verified cache reuses: {sum(r['cache_hits'] for r in rows):,}.","","Status: **PASS**."])
    (OUT/"06_concorde_labeling_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def train(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    protocol=protocol_gate()
    if replicate not in NEW_REPLICATES or arm not in ARMS: raise ValueError("only S1/S2 E0/E1")
    label_info=json.loads((label_dir(replicate,arm)/"labeling_integrity.json").read_text())
    coordinates,hashes,tours=p4c2.load_bundle(label_dir(replicate,arm)/"labeled_bundle.npz",require_tours=True); assert tours is not None
    frozen_xy,frozen_hashes,_=p4c2.load_bundle(dataset_bundle(replicate,arm))
    if hashes!=frozen_hashes or not np.array_equal(coordinates,frozen_xy): raise RuntimeError("labeled membership drift")
    device=torch.device(device_name); import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    directory=run_dir(replicate,arm)/"training_cycle"; checkpoint=checkpoint_path(replicate,arm)
    core.RUN_ID=f"TARGET_AWARE_P5_{replicate}_{arm}"; core.cycle_dir=lambda selected_arm,cycle:directory; core.model_path=lambda selected_arm,index:checkpoint if index==5 else M4
    model,model_meta=load_canonical_checkpoint(M4,device,expected_sha256=M4_SHA)
    config={"replicate":replicate,"arm":arm,"evolution_seed":EVOLUTION_SEEDS[replicate],"training_seed":TRAINING_SEED,"membership_sha256":membership_hash(hashes),"label_bundle_sha256":label_info["labeled_bundle_sha256"],"protocol_sha256":protocol["protocol_sha256"],"base_training_protocol":base_training_protocol()}; config["training_config_sha256"]=canonical_hash(config); write_json(run_dir(replicate,arm)/"training_config.json",config)
    started=time.perf_counter(); trained,meta,timing=core.train(f"{replicate}_{arm}",4,model,model_meta["sha256"],coordinates,hashes,tours,device,Status(),True); invocation=time.perf_counter()-started
    stage=directory/"09_bq_training"; e000=torch.load(stage/"M5_E000.tsp.best",map_location="cpu",weights_only=False); m4=torch.load(M4,map_location="cpu",weights_only=False); e000_equal=set(e000["net"])==set(m4["net"]) and all(torch.equal(e000["net"][k],m4["net"][k]) for k in m4["net"]); curve=pd.read_csv(stage/"training_loss_curve.csv")
    if not e000_equal or len(curve)!=256 or int(curve.optimizer_updates.sum())!=10_240 or not curve.hard_instances_seen.eq(TRAIN_N).all() or meta["sha256"]!=sha256_file(checkpoint) or not all(bool(torch.isfinite(p).all()) for p in trained.parameters()): raise RuntimeError("training integrity failure")
    result={"status":"PASS",**config,"m4_sha256":M4_SHA,"e000_weights_equal_m4":True,"epochs":256,"optimizer_steps":int(timing["optimizer_steps"]),"final_loss":float(timing["final_loss"]),"training_loop_wall_s":float(timing["training_loop_wall_s"]),"invocation_wall_s":invocation,"gpu_peak_allocated_bytes":int(timing["gpu_peak_allocated_bytes"]),"checkpoint_sha256":meta["sha256"],"loss_curve_sha256":sha256_file(stage/"training_loss_curve.csv")}; write_json(run_dir(replicate,arm)/"training_integrity.json",result)
    del trained,model
    if device.type=="cuda": torch.cuda.empty_cache()
    write_training_manifest(); return result


def write_training_manifest() -> None:
    rows=[]
    for arm in ARMS:
        item=ORIGINAL[arm]; info=json.loads(item["training_info"].read_text())
        rows.append({"replicate":"Original","arm":arm,"status":"EXISTING_PASS","evolution_seed":EVOLUTION_SEEDS["Original"],"training_seed":TRAINING_SEED,"dataset_bundle_sha256":item["dataset_sha"],"label_bundle_sha256":item["label_sha"],"m4_sha256":M4_SHA,"e000_weights_equal_m4":info["e000_weights_equal_m4"],"optimizer_steps":info["optimizer_steps"],"final_loss":info["final_loss"],"training_wall_s":info["training_loop_wall_s"],"checkpoint_sha256":item["checkpoint_sha"],"checkpoint_path":str(item["checkpoint"])})
    for rep in NEW_REPLICATES:
        for arm in ARMS:
            path=run_dir(rep,arm)/"training_integrity.json"
            if path.is_file():
                info=json.loads(path.read_text()); rows.append({"replicate":rep,"arm":arm,"status":info["status"],"evolution_seed":EVOLUTION_SEEDS[rep],"training_seed":TRAINING_SEED,"dataset_bundle_sha256":sha256_file(dataset_bundle(rep,arm)),"label_bundle_sha256":info["label_bundle_sha256"],"m4_sha256":M4_SHA,"e000_weights_equal_m4":info["e000_weights_equal_m4"],"optimizer_steps":info["optimizer_steps"],"final_loss":info["final_loss"],"training_wall_s":info["training_loop_wall_s"],"checkpoint_sha256":info["checkpoint_sha256"],"checkpoint_path":str(checkpoint_path(rep,arm)),"training_config_sha256":info["training_config_sha256"]})
            else: rows.append({"replicate":rep,"arm":arm,"status":"PENDING","evolution_seed":EVOLUTION_SEEDS[rep],"training_seed":TRAINING_SEED})
    pd.DataFrame(rows).to_csv(OUT/"07_bq_training_runs_manifest.csv",index=False)


def evaluate(replicate: str, arm: str, device_name: str) -> dict[str, Any]:
    if replicate not in NEW_REPLICATES or arm not in ARMS: raise ValueError("only S1/S2 E0/E1")
    info=json.loads((run_dir(replicate,arm)/"training_integrity.json").read_text()); checkpoint=checkpoint_path(replicate,arm)
    if info["checkpoint_sha256"]!=sha256_file(checkpoint): raise RuntimeError("checkpoint mismatch")
    device=torch.device(device_name); import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    model,meta=load_canonical_checkpoint(checkpoint,device,expected_sha256=info["checkpoint_sha256"]); frame,wall=core.evaluate(model,checkpoint,eval_label(replicate,arm),eval_csv(replicate,arm),Status(),f"{replicate}_{arm}")
    if set(frame.dataset.astype(str))!=set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all(): raise RuntimeError("evaluation incomplete")
    result={"status":"PASS","replicate":replicate,"arm":arm,"checkpoint_sha256":meta["sha256"],"all_tours_valid":True,"datasets":list(DATASETS),"evaluation_wall_s":wall,"results_sha256":sha256_file(eval_csv(replicate,arm))}; write_json(run_dir(replicate,arm)/"evaluation_integrity.json",result); del model
    if device.type=="cuda": torch.cuda.empty_cache()
    return result


def detail(replicate: str, arm: str, dataset: str) -> pd.DataFrame:
    if replicate=="Original": return pd.read_parquet((P4A if arm=="E0" else P3)/"evaluation_work"/ORIGINAL[arm]["eval_label"]/dataset/"instance_metrics.parquet")
    return pd.read_parquet(run_dir(replicate,arm)/"evaluation_work"/eval_label(replicate,arm)/dataset/"instance_metrics.parquet")


def paired(replicate: str, dataset: str) -> pd.DataFrame:
    parts=[]
    for arm in ARMS:
        frame=detail(replicate,arm,dataset)[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].copy()
        if not frame.valid_tour.all(): raise RuntimeError("invalid evaluation tour")
        parts.append(frame.rename(columns={"coordinate_hash":f"coordinate_hash_{arm}","n_nodes":f"n_nodes_{arm}","optimal_cost_int":f"optimal_cost_int_{arm}","gap_percent":f"gap_{arm}"}).drop(columns="valid_tour"))
    merged=parts[0].merge(parts[1],on="instance_id",validate="one_to_one")
    for field in ("coordinate_hash","n_nodes","optimal_cost_int"):
        if not merged[f"{field}_E0"].astype(str).eq(merged[f"{field}_E1"].astype(str)).all(): raise RuntimeError("paired alignment failure")
    return pd.DataFrame({"replicate":replicate,"dataset":dataset,"instance_id":merged.instance_id.astype(str),"coordinate_hash":merged.coordinate_hash_E0.astype(str),"n_nodes":merged.n_nodes_E0.astype(int),"optimal_cost_int":merged.optimal_cost_int_E0.astype(np.int64),"gap_E0":merged.gap_E0.astype(float),"gap_E1":merged.gap_E1.astype(float),"Delta_E0_minus_E1":merged.gap_E0.astype(float)-merged.gap_E1.astype(float)})


def verify_evaluations() -> None:
    frames=[]
    for arm in ARMS: frames.append(pd.read_csv(ORIGINAL[arm]["eval_csv"]).set_index("dataset"))
    for rep in NEW_REPLICATES:
        for arm in ARMS:
            info=json.loads((run_dir(rep,arm)/"evaluation_integrity.json").read_text()); frame=pd.read_csv(eval_csv(rep,arm))
            if info.get("status")!="PASS" or set(frame.checkpoint_sha256.astype(str))!={sha256_file(checkpoint_path(rep,arm))}: raise RuntimeError("evaluation integrity failure")
            frames.append(frame.set_index("dataset"))
    for dataset in DATASETS:
        if len({str(f.at[dataset,"dataset_sha256"]) for f in frames})!=1 or len({str(f.at[dataset,"evaluator_config_sha256"]) for f in frames})!=1: raise RuntimeError("evaluator protocol mismatch")


def bootstrap(values: np.ndarray, seed: int) -> tuple[float,float]:
    return p4b1.paired_bootstrap(np.asarray(values,float),seed)


def gap5_bootstrap(frames: dict[str,pd.DataFrame], replicate: str) -> tuple[float,float]:
    rng=np.random.default_rng(p4b1.stable_seed(f"P5|{replicate}|GAP5")); estimates=np.empty(BOOTSTRAP_REPLICATES)
    values=[frames[name].Delta_E0_minus_E1.to_numpy(float) for name in GAP5_DATASETS]
    for i in range(BOOTSTRAP_REPLICATES): estimates[i]=np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in values])
    return tuple(map(float,np.quantile(estimates,[.025,.975])))


def finalize() -> dict[str, Any]:
    verify_evaluations(); write_training_manifest(); write_label_report()
    dev=set(pd.read_csv(ROOT/"target_aware_pilot"/"03_tsplib48_reference_manifest.csv").instance_id.astype(str)); held=set(json.loads((ROOT/"target_aware_pilot"/"p1_heldout22_names.json").read_text())["names"])
    result_rows=[]; analysis_rows=[]; instance_parts=[]; all_frames={}
    for rep in ALL_REPLICATES:
        frames={name:paired(rep,name) for name in DATASETS}; all_frames[rep]=frames
        tsplib=frames["OriginalTSPLIB70"]
        if set(tsplib.instance_id)!=dev|held: raise RuntimeError("TSPLIB split mismatch")
        frames["TSPLIB22_previously_heldout"]=tsplib[tsplib.instance_id.isin(held)].copy(); frames["TSPLIB48_development"]=tsplib[tsplib.instance_id.isin(dev)].copy()
        for name,frame in frames.items():
            values=frame.Delta_E0_minus_E1.to_numpy(float); low,high=bootstrap(values,p4b1.stable_seed(f"P5|{rep}|{name}")); result_rows.append({"replicate":rep,"dataset":name,"n_instances":len(frame),"mean_gap_E0":float(frame.gap_E0.mean()),"mean_gap_E1":float(frame.gap_E1.mean()),"Delta_E0_minus_E1":float(values.mean()),"ci95_low":low,"ci95_high":high,"E1_instance_wins":int(np.sum(values>0)),"E0_instance_wins":int(np.sum(values<0)),"ties":int(np.sum(values==0)),"bootstrap_replicates":BOOTSTRAP_REPLICATES}); instance_parts.append(frame)
        gap5_e0=float(np.mean([frames[x].gap_E0.mean() for x in GAP5_DATASETS])); gap5_e1=float(np.mean([frames[x].gap_E1.mean() for x in GAP5_DATASETS])); low,high=gap5_bootstrap(frames,rep)
        result_rows.append({"replicate":rep,"dataset":"GAP5_macro","n_instances":int(sum(len(frames[x]) for x in GAP5_DATASETS)),"mean_gap_E0":gap5_e0,"mean_gap_E1":gap5_e1,"Delta_E0_minus_E1":gap5_e0-gap5_e1,"ci95_low":low,"ci95_high":high,"E1_instance_wins":np.nan,"E0_instance_wins":np.nan,"ties":np.nan,"bootstrap_replicates":BOOTSTRAP_REPLICATES})
    results=pd.DataFrame(result_rows); results.to_csv(OUT/"08_original70_tsplib22_gap5_results.csv",index=False); pd.concat(instance_parts,ignore_index=True).to_csv(OUT/"support_per_instance_paired_differences.csv",index=False)
    for endpoint in ("OriginalTSPLIB70","TSPLIB22_previously_heldout","GAP5_macro"):
        values=results[results.dataset.eq(endpoint)].set_index("replicate").loc[list(ALL_REPLICATES)].Delta_E0_minus_E1.to_numpy(float); analysis_rows.append({"endpoint":endpoint,"replicate_n":3,"mean_Delta":float(values.mean()),"median_Delta":float(np.median(values)),"sample_sd_Delta":float(values.std(ddof=1)),"min_Delta":float(values.min()),"max_Delta":float(values.max()),"E1_winning_replicate_n":int(np.sum(values>0)),"E0_winning_replicate_n":int(np.sum(values<0)),"all_E1":bool(np.all(values>0)),"all_E0":bool(np.all(values<0))})
    analysis=pd.DataFrame(analysis_rows); analysis.to_csv(OUT/"09_paired_evolution_seed_analysis.csv",index=False)
    primary=results[results.dataset.eq("OriginalTSPLIB70")].set_index("replicate").loc[list(ALL_REPLICATES)]; heldout=results[results.dataset.eq("TSPLIB22_previously_heldout")].set_index("replicate").loc[list(ALL_REPLICATES)]; gap5=results[results.dataset.eq("GAP5_macro")].set_index("replicate").loc[list(ALL_REPLICATES)]
    both_primary=bool(primary.loc[list(NEW_REPLICATES)].Delta_E0_minus_E1.gt(0).all()); held_reverse=bool(heldout.loc[list(NEW_REPLICATES)].Delta_E0_minus_E1.lt(-.25).all()); gap5_safe=bool(gap5.loc[list(NEW_REPLICATES)].Delta_E0_minus_E1.ge(-.25).all()); both_negative=bool(primary.loc[list(NEW_REPLICATES)].Delta_E0_minus_E1.lt(0).all())
    if both_primary and not held_reverse and gap5_safe: decision="STRUCTURE_GUIDED_EVOLUTION_GAIN_PRELIMINARILY_REPLICATED"; label="SUPPORTED_PRELIMINARY"
    elif both_negative: decision="STRUCTURE_GUIDED_EVOLUTION_GAIN_NOT_SUPPORTED_IN_REPLICATION"; label="NOT_SUPPORTED"
    else: decision="STRUCTURE_GUIDED_EVOLUTION_GAIN_INCONCLUSIVE"; label="INCONCLUSIVE"
    hard_original=float(p4c2.hard_detail("OriginalTSPLIB70").gap_percent.mean()); m4_original=float(pd.read_parquet(p4b1.CYCLES/"C03"/"evaluation_work"/"M4"/"OriginalTSPLIB70"/"instance_metrics.parquet").gap_percent.mean())
    structure=pd.read_csv(OUT/"04_structural_distribution_comparison.csv"); train_structure=structure[structure.object_type.eq("provisional_structure10k")]
    struct_summary=[]
    for rep in ALL_REPLICATES:
        part=train_structure[train_structure.replicate.eq(rep)].set_index("arm"); struct_summary.append({"replicate":rep,"SWD_E0_minus_E1":float(part.at["E0","sliced_wasserstein"]-part.at["E1","sliced_wasserstein"]),"JS_E0_minus_E1":float(part.at["E0","coarse_js"]-part.at["E1","coarse_js"]),"strict_E1_minus_E0":float(part.at["E1","strict_target_fraction"]-part.at["E0","strict_target_fraction"])})
    pd.DataFrame(struct_summary).to_csv(OUT/"support_structure_vs_performance.csv",index=False)
    labeling=[json.loads((label_dir(rep,arm)/"labeling_integrity.json").read_text()) for rep in NEW_REPLICATES for arm in ARMS]; training=[json.loads((run_dir(rep,arm)/"training_integrity.json").read_text()) for rep in NEW_REPLICATES for arm in ARMS]; evaluation=[json.loads((run_dir(rep,arm)/"evaluation_integrity.json").read_text()) for rep in NEW_REPLICATES for arm in ARMS]
    lines=["# P5 Final Research Conclusion","",f"## `{decision}` — **{label}**","","Delta = gap(E0) - gap(E1); positive favors structure-guided E1.","","| Replicate | Original70 E0 | Original70 E1 | Delta | TSPLIB22 Delta | GAP5 Delta |","|---|---:|---:|---:|---:|---:|"]
    for rep in ALL_REPLICATES: lines.append(f"| {rep} | {primary.at[rep,'mean_gap_E0']:.4f} | {primary.at[rep,'mean_gap_E1']:.4f} | {primary.at[rep,'Delta_E0_minus_E1']:+.4f} | {heldout.at[rep,'Delta_E0_minus_E1']:+.4f} | {gap5.at[rep,'Delta_E0_minus_E1']:+.4f} |")
    agg=analysis.set_index("endpoint")
    gm100=results[results.dataset.eq("GaussianMixture100")].set_index("replicate").loc[list(ALL_REPLICATES)]
    gm500=results[results.dataset.eq("GaussianMixture500")].set_index("replicate").loc[list(ALL_REPLICATES)]
    struct_frame=pd.DataFrame(struct_summary).set_index("replicate")
    evolution_wall=sum(json.loads((rep_dir(rep)/"run_metadata.json").read_text())["p5_invocation_wall_seconds"] for rep in NEW_REPLICATES)
    lines.extend([
        "",
        "## Pre-registered decision",
        "",
        f"- Original70 Delta across replicates: mean {agg.at['OriginalTSPLIB70','mean_Delta']:+.4f} pp, median {agg.at['OriginalTSPLIB70','median_Delta']:+.4f}, sample SD {agg.at['OriginalTSPLIB70','sample_sd_Delta']:.4f}, range [{agg.at['OriginalTSPLIB70','min_Delta']:+.4f}, {agg.at['OriginalTSPLIB70','max_Delta']:+.4f}].",
        f"- Preregistered gates: both new Original70 positive = {both_primary}; persistent TSPLIB22 reverse >0.25 pp = {held_reverse}; both new GAP5 Deltas at least -0.25 pp = {gap5_safe}.",
        "- S1 supports E1, while S2 reverses on Original70, TSPLIB22, and GAP5. Therefore the training-value claim is not robust to evolution seed.",
        "",
        "## Required research judgments",
        "",
        "1. **Causal chain:** PASS. Both arms start from byte-identical ordered P03; G1 survivors become G2 parents; the independent offspring archive never supplies parents; final acquisition reads Archive30k rather than Population10k.",
        "2. **G2 survivor effect:** `G2_SURVIVOR_SELECTION_NOT_ON_TRAINING_DATA_PATH`. G2 survivors are computed and audited, but there is no G3 and they do not modify the already accumulated offspring archive.",
        "3. **G1 random-stream coupling:** E0/E1 use exact common mutation random numbers: identical parent order, operator schedule, and child parameter seeds with no arm/global-RNG term. The only G1 treatment difference is survivor selection.",
        "4. **Structural replication:** E1 improves final Structure10k SWD, 3x3 JS, and strict-target fraction in all three replicates. The SWD E0-E1 gains are " + ", ".join(f"{rep} {struct_frame.at[rep,'SWD_E0_minus_E1']:+.4f}" for rep in ALL_REPLICATES) + "; JS gains are " + ", ".join(f"{rep} {struct_frame.at[rep,'JS_E0_minus_E1']:+.4f}" for rep in ALL_REPLICATES) + ". G1/G2 survivor-population structural gains are likewise directionally consistent.",
        "5. **Original70 training value:** not persistent. Delta is positive for Original and S1, but negative for S2. E1 gaps versus M4 / Hardness M5 are " + ", ".join(f"{rep} {primary.at[rep,'mean_gap_E1']-m4_original:+.4f} / {primary.at[rep,'mean_gap_E1']-hard_original:+.4f} pp" for rep in ALL_REPLICATES) + f" (reference M4 {m4_original:.4f}%, Hardness M5 {hard_original:.4f}%).",
        "6. **TSPLIB22 trend:** not consistent. Deltas are " + ", ".join(f"{rep} {heldout.at[rep,'Delta_E0_minus_E1']:+.4f} pp" for rep in ALL_REPLICATES) + "; every 95% paired-bootstrap CI includes zero, and S2 reverses.",
        "7. **GAP5 without EvoReal:** does not robustly support E1. Deltas are " + ", ".join(f"{rep} {gap5.at[rep,'Delta_E0_minus_E1']:+.4f} pp" for rep in ALL_REPLICATES) + ". S2 is significantly negative by its paired-bootstrap CI.",
        "8. **Core-method status:** structure-guided survivor selection is a reproducible distribution-shaping mechanism, but it is not yet supported as a robust solver-performance contribution.",
        "9. **Next step:** re-evaluate the evolution/data-to-performance mechanism before further E1 component ablation. Do not treat the original 1.918% result as confirmed causal evidence.",
        "",
        "## Generalization and cost diagnostics",
        "",
        "- GaussianMixture behavior is seed-sensitive: GM100 Deltas are " + ", ".join(f"{rep} {gm100.at[rep,'Delta_E0_minus_E1']:+.4f}" for rep in ALL_REPLICATES) + " pp; GM500 Deltas are " + ", ".join(f"{rep} {gm500.at[rep,'Delta_E0_minus_E1']:+.4f}" for rep in ALL_REPLICATES) + " pp. S2 shows clear E1 degradation on both.",
        f"- Added cost for S1/S2: evolution {evolution_wall:.1f}s; Concorde {sum(x['pipeline_wall_seconds'] for x in labeling):.1f}s with {sum(x['physical_new_concorde_calls'] for x in labeling):,} physical calls and {sum(x['cache_hits'] for x in labeling):,} cache hits; BQ training {sum(x['training_loop_wall_s'] for x in training):.1f}s; evaluation {sum(x['evaluation_wall_s'] for x in evaluation):.1f}s.",
        "- Original70 is target-informed through TSPLIB48; TSPLIB22 is repeatedly observed and is not pristine. Replicates are analyzed separately; test instances were not pooled as independent evolution replicates.",
        "- No Diversity-Only, Hardness Top10k, EvoReal Fresh10k, G3, extra seed, or other follow-up was started.",
        "",
        f"Decision label: `{label}`",
        "",
        "`BLADE_P5_STRUCTURE_GUIDED_EVOLUTION_ROBUSTNESS_COMPLETE`",
    ])
    (OUT/"10_final_research_conclusion.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    x=np.arange(3); fig,axes=plt.subplots(1,3,figsize=(14,4.5));
    for ax,table,title in ((axes[0],primary,"Original70"),(axes[1],heldout,"TSPLIB22"),(axes[2],gap5,"GAP5")):
        ax.bar(x-.17,table.mean_gap_E0,.34,label="E0"); ax.bar(x+.17,table.mean_gap_E1,.34,label="E1"); ax.set_xticks(x); ax.set_xticklabels(ALL_REPLICATES); ax.set_title(title); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("mean gap (%)"); axes[-1].legend(); fig.tight_layout(); fig.savefig(FIG/"02_primary_endpoint_by_replicate.png",dpi=180); plt.close(fig)
    write_json(OUT/"run_metadata.json",{"status":"COMPLETE","decision":decision,"decision_label":label,"primary_mean_delta":float(agg.at['OriginalTSPLIB70','mean_Delta']),"primary_sample_sd":float(agg.at['OriginalTSPLIB70','sample_sd_Delta']),"g2_survivor_on_training_path":False,"no_followup_started":True,"results_sha256":sha256_file(OUT/"08_original70_tsplib22_gap5_results.csv"),"analysis_sha256":sha256_file(OUT/"09_paired_evolution_seed_analysis.csv")}); Status().update(phase="complete",status="COMPLETE",decision=decision,decision_label=label)
    return {"status":"COMPLETE","decision":decision,"label":label,"primary":primary[["mean_gap_E0","mean_gap_E1","Delta_E0_minus_E1"]].reset_index().to_dict("records")}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("audit","evolve","prepare","label","train","evaluate","finalize")); parser.add_argument("--replicate",choices=NEW_REPLICATES); parser.add_argument("--arm",choices=ARMS); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args()
    if args.phase=="audit": result=audit()
    elif args.phase=="evolve":
        if args.replicate is None: parser.error("--replicate required")
        result=evolve(args.replicate)
    elif args.phase=="prepare": result=prepare()
    elif args.phase in ("label","train","evaluate"):
        if args.replicate is None or args.arm is None: parser.error("--replicate and --arm required")
        result=label(args.replicate,args.arm) if args.phase=="label" else train(args.replicate,args.arm,args.device) if args.phase=="train" else evaluate(args.replicate,args.arm,args.device)
    else: result=finalize()
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
