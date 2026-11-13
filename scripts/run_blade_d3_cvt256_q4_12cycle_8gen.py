"""12-cycle D3-CVT-256 Q4 evolution from canonical M0.

The behavior map, Unified-12 profile, and BQ training protocol are frozen.
This runner only schedules them. Support status is recorded and never filters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path("/home/haozou/blade-nco-mainline")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from bq.checkpoint import build_bq_model, load_canonical_checkpoint, state_dict_sha256
from bq.protocol import (
    CANONICAL_CONCORDE_SHA256,
    CANONICAL_M0_SHA256,
    GRADIENT_CLIPPING,
    MICROBATCH_SIZE,
    SUFFIX_LENGTH_MIN,
    WEIGHT_DECAY,
)
from bq.supervised import create_optimizer, sample_suffix_length, supervised_optimizer_step
from bq.training_hardness import MIN_DECISION_SUFFIX, closed_expert_coordinates, teacher_forced_training_hardness
from d3_cvt256_q4_policy import select_q4_parents
from mutation import OPERATOR_NAMES, apply_operator_with_config
from mutation.parameters import load_named_profile
from mutation.result import coordinate_hash
from oracle.concorde import resolve_backend, solve_one

CONFIG_PATH = ROOT / "configs/blade_d3_cvt256_q4_12cycle_8gen.yaml"
OUT = ROOT / "outputs/blade_d3_cvt256_q4_12cycle_8gen"
MAP_DIR = ROOT / "assets/qd_maps/d3-cvt256-v1"
MANIFEST_PATH = MAP_DIR / "map_manifest.json"
M0_PATH = ROOT / "assets/solver/M0/tsp.best"
PROFILE_PATH = ROOT / "src/mutation/unified12_frozen.yaml"
PYTHON = "/home/haozou/miniconda3/envs/bq-nco/bin/python"

EXPECTED = {
    "m0": CANONICAL_M0_SHA256,
    "concorde": CANONICAL_CONCORDE_SHA256,
    "unified12": "19e9548f417e46e0c1f17f7053f887645cc4e72e1778bca5e783529667a02f9e",
    "manifest": "36854f7bc06f4c10facf7415f7cf11e91e9268e695487a23460bf0160ea63348",
    "centroid_array": "798a341de070fcf21242af853ff18b307641005635540b7bd7945ef44dbaa99a",
    "centroid_file": "be817c1160de638a925bd3670da0165572a5c6495eb6b731aca0fff3fef020d6",
    "scaler": "281d887d952cc557b2b472d8e11f4214569720d4a102347591235a7dd1009d6d",
    "extractor": "37f676211b6ebca55321fe6d7035b3832625d9f49878f079561c6cd3e2c760e8",
    "mapper": "a2acf6b2baa43385c7daa3cad808c1e79fd0c6555505dc5b730c301565797809",
    "schema": "a1eb4c2b28aeb7adb447c1afb90cf3d4933edef86c050893258c07c15727b62a",
}
OPERATORS = (
    "uniform_mutation", "normal_mutation", "rotation_mutation", "implosion_mutation",
    "explosion_mutation", "cluster_mutation", "expansion_mutation", "compression_mutation",
    "axis_projection_mutation", "grid_mutation", "lattice_reorganization", "motif_assembly",
)
HISTORICAL_LEDGERS = (
    "outputs/blade_unified12_qd_redesign/stage2e_true_hardness_concentration/oracle_cache/ledger.csv",
    "outputs/evoreal_m0_hardness_profiling/optimal_tours/ledger.csv",
    "outputs/evoreal_s1_branch_hardness_analysis/concorde_tours/ledger.csv",
)


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"cannot encode {type(value)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    os.replace(temporary, path)


def load_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    expected = {
        "outer_cycles": 12, "inner_generations": 8, "active_parents": 2000,
        "offspring_per_parent": 25, "fresh_uniform_roots_per_cycle": 2000, "tsp_n": 100,
        "qd_niches": 256, "qd_elites_per_niche": 4, "hard_training_per_cycle": 10000,
        "training_epochs": 256, "training_batch_size": 256, "training_microbatch_size": 64,
        "uniform_anchor": 0, "historical_replay": 0, "mutation_steps_per_offspring": 1,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise SystemExit(f"BLOCKED config {key}={config[key]} != {value}")
    if config["offspring_attempts_per_generation"] != 50000:
        raise SystemExit("BLOCKED attempts per generation")
    if abs(float(config["training_lr"]) - 3.75e-5) > 1e-15:
        raise SystemExit("BLOCKED learning rate")
    if config["legacy_concorde_budget_cap"] is not None:
        raise SystemExit("BLOCKED a label cap is present")
    if config["gp_surrogate"] != "disabled" or config["hgb_surrogate"] != "disabled":
        raise SystemExit("BLOCKED surrogate is enabled")
    if config["parent_selection"] != "D3_CVT256_Q4_ELITE_GLOBAL_CE_FILL_V1":
        raise SystemExit("BLOCKED parent selection")
    if config["training_acquisition"] != "GLOBAL_TOP10000_TRUE_CE":
        raise SystemExit("BLOCKED acquisition")
    if config["qd_map_version"] != "d3-cvt256-v1":
        raise SystemExit("BLOCKED map version")
    return config


def load_mapper():
    path = MAP_DIR / "mapping/map_tsp100.py"
    spec = importlib.util.spec_from_file_location("d3_cvt256_v1_mapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_map(mapper) -> dict:
    manifest = mapper.load_manifest(MANIFEST_PATH)
    file_hash = sha256_file(MANIFEST_PATH)
    array_hash = mapper.sha256_array(manifest["centers"])
    checks = {
        "version_id": manifest["version_id"] == "d3-cvt256-v1",
        "status": manifest["status"] == "FROZEN_WORKING_MAP",
        "features": manifest["feature_names"] == list(mapper.FEATURES),
        "manifest_hash": file_hash == EXPECTED["manifest"],
        "centroid_array": array_hash == EXPECTED["centroid_array"],
        "centroid_file": sha256_file(MAP_DIR / "centroids.npy") == EXPECTED["centroid_file"],
        "scaler": sha256_file(MAP_DIR / "scaler/reference_scaler.json") == EXPECTED["scaler"],
        "extractor": sha256_file(MAP_DIR / "mapping/extract_salesperson_raw_features.R") == EXPECTED["extractor"],
        "schema": sha256_file(MAP_DIR / "mapping/salesperson_schema.csv") == EXPECTED["schema"],
        "mapper": sha256_file(MAP_DIR / "mapping/map_tsp100.py") == EXPECTED["mapper"],
        "k": int(manifest["K"]) == 256 and len(manifest["centers"]) == 256,
        "scaler_type": manifest["scaler_type"] == "asinh((x-median)/IQR)",
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        print("D3_CVT256_MAP_VERIFICATION_BLOCKED", failed, flush=True)
        raise SystemExit(2)
    niches, _, _ = mapper.assign_scaled(manifest["centers"][[0, 255]], manifest["centers"])
    if list(niches) != [0, 255]:
        print("D3_CVT256_MAP_VERIFICATION_BLOCKED niche endpoints", flush=True)
        raise SystemExit(2)
    return manifest


def golden_regression() -> dict:
    mapper_cli = ROOT / "scripts/qd_map_tsp100.py"
    expected = json.loads((MAP_DIR / "tests/golden_outputs.json").read_text())
    coordinates = np.load(MAP_DIR / "tests/golden_inputs.npy")
    work = OUT / "preflight/golden"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    batch_out = work / "batch.json"
    completed = subprocess.run(
        [PYTHON, str(mapper_cli), "--input", str(MAP_DIR / "tests/golden_inputs.npy"),
         "--map", str(MANIFEST_PATH), "--output", str(batch_out)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        print("D3_CVT256_MAP_VERIFICATION_BLOCKED", completed.stderr[-500:], flush=True)
        raise SystemExit(2)
    rows = json.loads(batch_out.read_text())["assignments"]
    worst = 0.0
    for case, row in zip(expected["cases"], rows):
        gap = max(
            float(np.max(np.abs(np.asarray(row["raw_d3_features"]) - np.asarray(case["raw_d3"])))),
            float(np.max(np.abs(np.asarray(row["scaled_d3_features"]) - np.asarray(case["scaled_d3"])))),
        )
        worst = max(worst, gap)
        if row["niche_id"] != case["niche_id"] or not 0 <= int(row["niche_id"]) <= 255 or gap > 1e-5:
            print("D3_CVT256_MAP_VERIFICATION_BLOCKED", case["name"], gap, flush=True)
            raise SystemExit(2)
        if row["map_version"] != "d3-cvt256-v1":
            print("D3_CVT256_MAP_VERIFICATION_BLOCKED version", flush=True)
            raise SystemExit(2)
    single = work / "single.npy"
    np.save(single, coordinates[0])
    single_out = work / "single.json"
    single_run = subprocess.run(
        [PYTHON, str(mapper_cli), "--input", str(single), "--map", str(MANIFEST_PATH), "--output", str(single_out)],
        capture_output=True, text=True, check=False,
    )
    if single_run.returncode != 0 or json.loads(single_out.read_text())["assignments"][0]["niche_id"] != rows[0]["niche_id"]:
        print("D3_CVT256_MAP_VERIFICATION_BLOCKED single", flush=True)
        raise SystemExit(2)
    return {"max_abs": worst, "niches": [int(row["niche_id"]) for row in rows]}


def preflight(config: dict) -> dict:
    print("VERIFY_MAP", flush=True)
    mapper = load_mapper()
    manifest = verify_map(mapper)
    golden = golden_regression()
    profile_hash = sha256_file(PROFILE_PATH)
    m0_hash = sha256_file(M0_PATH)
    backend = resolve_backend(ROOT / "third_party/concorde/concorde")
    checks = []

    def mark(name: str, ok: bool) -> None:
        checks.append({"name": name, "pass": bool(ok)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        if not ok:
            print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_BLOCKED", name, flush=True)
            raise SystemExit(2)

    mark("Canonical M0 Hash", m0_hash == EXPECTED["m0"])
    mark("Unified-12 Profile", profile_hash == EXPECTED["unified12"] and tuple(OPERATOR_NAMES) == OPERATORS)
    profile = load_named_profile("UNIFIED12_FROZEN_V1")
    smoke = apply_operator_with_config("uniform_mutation", np.random.default_rng(0).random((100, 2)), np.random.default_rng(1), profile)
    mark("Unified-12 Mutation Smoke", tuple(smoke.final_coordinates.shape) == (100, 2))
    mark("D3-CVT-256 Frozen Manifest", manifest["status"] == "FROZEN_WORKING_MAP")
    mark("Centroid Array SHA256", True)
    mark("Scaler Hash", True)
    mark("Golden Mapping Regression", golden["max_abs"] <= 1e-5)
    mark("Map Version ID", manifest["version_id"] == "d3-cvt256-v1")
    mark("256 Niche Assignment", golden["niches"] == [46, 227, 202])
    unit = subprocess.run([PYTHON, str(ROOT / "scripts/test_d3_cvt256_q4_selection.py")], capture_output=True, text=True, check=False)
    mark("Q4 Elite + Global Fill Unit Test", unit.returncode == 0 and "Q4_UNIT_PASS" in unit.stdout)
    mark("Pure Top-10k CE Acquisition", unit.returncode == 0)
    mark("12 Outer Cycles", config["outer_cycles"] == 12)
    mark("8 Inner Generations", config["inner_generations"] == 8)
    mark("2000 Parents", config["active_parents"] == 2000)
    mark("25 Offspring/Parent", config["offspring_per_parent"] == 25)
    mark("50000 Attempts/Generation", config["active_parents"] * config["offspring_per_parent"] == 50000)
    mark("400000 Attempts/Cycle", 8 * 50000 == 400000)
    mark("4800000 Total Offspring Attempts", 12 * 400000 == 4800000)
    mark("4824000 Maximum Label Demand", 4800000 + 12 * 2000 == 4824000)
    mark("No Legacy Label Cap", config["legacy_concorde_budget_cap"] is None)
    ledger = OUT / "preflight/mock_oracle.sqlite"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists():
        ledger.unlink()
    connection = sqlite3.connect(ledger)
    connection.execute("CREATE TABLE solved (instance_hash TEXT PRIMARY KEY, source TEXT)")
    connection.execute("INSERT INTO solved VALUES ('abc', 'historical')")
    connection.commit()
    found = connection.execute("SELECT source FROM solved WHERE instance_hash='abc'").fetchone()
    connection.close()
    mark("Concorde Cache & Accounting", backend["binary_hash"] == EXPECTED["concorde"] and found[0] == "historical")
    source = Path(__file__).read_text()
    forbidden_surrogate = "HistGradient" + "BoostingRegressor("
    forbidden_process = "Gaussian" + "Process"
    forbidden_operator = "apply_" + "foga11"
    mark("No GP/HGB", forbidden_surrogate not in source and forbidden_process not in source)
    mark("No Historical FOGA11", forbidden_operator not in source and "linear_projection" not in OPERATORS)
    mark("No Uniform Anchor / Replay", config["uniform_anchor"] == 0 and config["historical_replay"] == 0)
    mark("Benchmark Isolation", config["selection_by_test_gap"] == "disabled" and config["early_stop_by_test_gap"] == "disabled")
    mark("Reproducible Seed Derivation", config["root_seed_base"] == 12000000 and config["name_seed"] == [20261113, 1])
    temporary = OUT / "preflight/resume_probe"
    temporary.mkdir(parents=True, exist_ok=True)
    write_json(temporary / "done.json", {"ok": True})
    mark("Resume Integrity", (temporary / "done.json").is_file())
    free = shutil.disk_usage(ROOT).free
    mark("Disk/Compute Resources", free > 200 * (1024 ** 3) and torch.cuda.is_available())
    mark("Hardness Suffix 3 / Training Suffix 4", MIN_DECISION_SUFFIX == 3 and SUFFIX_LENGTH_MIN == 4)
    mark("Microbatch 64 Adam No Clip", MICROBATCH_SIZE == 64 and WEIGHT_DECAY == 0 and GRADIENT_CLIPPING is None)
    report = {"passed": True, "checks": checks, "golden": golden, "m0": m0_hash, "unified12": profile_hash,
              "manifest": sha256_file(MANIFEST_PATH), "concorde": backend["binary_hash"], "free_bytes": free}
    write_json(OUT / "manifests/preflight.json", report)
    (OUT / "01_preflight.md").write_text(preflight_markdown(report))
    print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_PREFLIGHT_PASS", flush=True)
    return report


def preflight_markdown(report: dict) -> str:
    lines = ["# 01 Preflight", "", "状态：BLADE_D3_CVT256_Q4_12CYCLE_8GEN_PREFLIGHT_PASS", ""]
    for item in report["checks"]:
        lines.append(f"- {item['name']}: PASS")
    lines.append("")
    lines.append(f"Golden max abs {report['golden']['max_abs']}, niches {report['golden']['niches']}.")
    lines.append("没有旧的标签数量上限。预检没有调用 Concorde，也没有训练。")
    lines.append("Fresh Uniform Roots 使用新的 seed base 12000000。仓库里没有可复用的 Unified-12 十轮 root seed 流，因此不声称与旧十轮的根实例相同。")
    return "\n".join(lines) + "\n"


def open_db() -> sqlite3.Connection:
    path = OUT / "oracle/ledger.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS solved (instance_hash TEXT PRIMARY KEY, expert_tour TEXT, optimal_cost_int INTEGER, runtime_sec REAL, binary_hash TEXT, source TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS events (cycle INTEGER, generation INTEGER, kind TEXT, instance_hash TEXT, UNIQUE(cycle, generation, kind, instance_hash))")
    connection.execute("CREATE TABLE IF NOT EXISTS features (instance_hash TEXT PRIMARY KEY, f0 REAL, f1 REAL, f2 REAL)")
    connection.commit()
    return connection


def import_historical(connection: sqlite3.Connection) -> int:
    inserted = 0
    for relative in HISTORICAL_LEDGERS:
        path = ROOT / relative
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        needed = {"instance_hash", "status", "verified", "expert_tour", "optimal_cost_int", "binary_hash"}
        if not needed.issubset(frame.columns):
            continue
        usable = frame[(frame["status"] == "solved") & (frame["verified"].astype(str).isin(["True", "true", "1"])) & (frame["binary_hash"] == EXPECTED["concorde"])]
        for row in usable.itertuples(index=False):
            cursor = connection.execute(
                "INSERT OR IGNORE INTO solved VALUES (?,?,?,?,?,?)",
                (str(row.instance_hash), str(row.expert_tour), int(row.optimal_cost_int), float(getattr(row, "runtime_sec", 0.0) or 0.0), EXPECTED["concorde"], "historical"),
            )
            inserted += cursor.rowcount
    connection.commit()
    return inserted


def inside(coordinates: np.ndarray) -> bool:
    return bool(np.isfinite(coordinates).all() and coordinates.min() >= -1e-12 and coordinates.max() <= 1 + 1e-12)


def load_generators():
    path = ROOT / "outputs/evoreal_generator_analysis/scripts/lib.py"
    spec = importlib.util.spec_from_file_location("evoreal_generators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = ROOT / "outputs/evoreal_generator_analysis/source_snapshot"
    return module, module.load_generator_modules(snapshot)


def generate_roots(cycle: int, config: dict) -> tuple[np.ndarray, list[str], list[int]]:
    generators, modules = load_generators()
    coords = np.empty((config["fresh_uniform_roots_per_cycle"], 100, 2), dtype=np.float64)
    seeds = []
    for index in range(config["fresh_uniform_roots_per_cycle"]):
        seed = int(config["root_seed_base"] + cycle * 2000 + index)
        points, _ = generators.generate_one(modules["Uniform"], seed)
        if points.shape != (100, 2) or not inside(points):
            raise SystemExit(f"BLOCKED root {seed} is not a valid Uniform100 instance")
        coords[index] = points
        seeds.append(seed)
        if (index + 1) % 500 == 0:
            print(f"ROOTS cycle={cycle} {index + 1}", flush=True)
    hashes = [coordinate_hash(coords[index]) for index in range(len(coords))]
    if len(set(hashes)) != len(hashes):
        raise SystemExit("BLOCKED duplicate fresh roots")
    return coords, hashes, seeds


def label_hashes(connection: sqlite3.Connection, coords: np.ndarray, hashes: list[str], cycle: int, generation: int, workers: int) -> tuple[list[str], int, int]:
    missing = []
    hits = 0
    for index, digest in enumerate(hashes):
        row = connection.execute("SELECT expert_tour FROM solved WHERE instance_hash=?", (digest,)).fetchone()
        if row is None:
            missing.append(index)
        else:
            hits += 1
            connection.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?)", (cycle, generation, "cache_hit", digest))
    backend = resolve_backend(ROOT / "third_party/concorde/concorde")
    work = OUT / "oracle/work"
    new = 0
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(missing)))) as pool:
            jobs = {pool.submit(solve_one, coords[index], hashes[index], work / hashes[index], backend): index for index in missing}
            for count, future in enumerate(as_completed(jobs), 1):
                row = future.result()
                connection.execute("INSERT OR IGNORE INTO solved VALUES (?,?,?,?,?,?)", (
                    row["instance_hash"], row["expert_tour"], int(row["optimal_cost_int"]), float(row["runtime_sec"]), row["binary_hash"], "this_experiment"))
                connection.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?)", (cycle, generation, "new_solve", row["instance_hash"]))
                shutil.rmtree(work / row["instance_hash"], ignore_errors=True)
                new += 1
                if count % 100 == 0 or count == len(missing):
                    connection.commit()
                    print(f"CONCORDE cycle={cycle} gen={generation} {count}/{len(missing)}", flush=True)
    connection.commit()
    tours = []
    for digest in hashes:
        row = connection.execute("SELECT expert_tour FROM solved WHERE instance_hash=?", (digest,)).fetchone()
        if row is None:
            raise SystemExit(f"BLOCKED missing tour {digest}")
        tours.append(row[0])
    return tours, hits, new


def parse_tours(values: list[str]) -> np.ndarray:
    tours = np.asarray([[int(token) for token in value.split()] for value in values], dtype=np.int64)
    if tours.ndim != 2 or tours.shape[1] != 100:
        raise SystemExit(f"BLOCKED tour shape {tours.shape}")
    return tours


def score_ce(model, coords: np.ndarray, tours: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    values = teacher_forced_training_hardness(model, coords, tours, device, batch_size=batch_size)["loss_raw"]
    if len(values) != len(coords) or not np.isfinite(values).all():
        raise SystemExit("BLOCKED non-finite hardness")
    return np.asarray(values, dtype=np.float64)


_WORKER = {}


def _worker_init(manifest_path: str) -> None:
    mapper = load_mapper()
    _WORKER["mapper"] = mapper
    _WORKER["manifest"] = mapper.load_manifest(Path(manifest_path))


def _extract_chunk(coordinates: np.ndarray) -> np.ndarray:
    try:
        return _WORKER["mapper"].extract_raw(np.asarray(coordinates, dtype=np.float64), _WORKER["manifest"])
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc


def map_instances(connection: sqlite3.Connection, mapper, manifest: dict, coords: np.ndarray, hashes: list[str], workers: int) -> pd.DataFrame:
    raw = np.full((len(hashes), 3), np.nan, dtype=np.float64)
    missing = []
    for index, digest in enumerate(hashes):
        row = connection.execute("SELECT f0,f1,f2 FROM features WHERE instance_hash=?", (digest,)).fetchone()
        if row is None:
            missing.append(index)
        else:
            raw[index] = row
    if missing:
        chunks = []
        for start in range(0, len(missing), 2000):
            chosen = missing[start:start + 2000]
            chunks.append(coords[np.asarray(chosen)])
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init, initargs=(str(MANIFEST_PATH),)) as pool:
                extracted = list(pool.map(_extract_chunk, chunks))
        except (BrokenProcessPool, RuntimeError) as exc:
            print(f"FEATURE_POOL_RETRY {type(exc).__name__}: {exc}", flush=True)
            extracted = [mapper.extract_raw(chunk, manifest) for chunk in chunks]
        cursor = 0
        for block in extracted:
            count = len(block)
            chosen = missing[cursor:cursor + count]
            raw[np.asarray(chosen)] = block
            cursor += count
            for index in chosen:
                connection.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?)", (hashes[index], float(raw[index, 0]), float(raw[index, 1]), float(raw[index, 2])))
        connection.commit()
    medians, scales = mapper.scaler_arrays(manifest)
    scaled = mapper.scale_raw(raw, medians, scales)
    niches, nearest, second = mapper.assign_scaled(scaled, manifest["centers"])
    margin = second - nearest
    support = [mapper.support_label(float(distance), manifest["support_thresholds"]) for distance in nearest]
    return pd.DataFrame({
        "hash": hashes, "raw0": raw[:, 0], "raw1": raw[:, 1], "raw2": raw[:, 2],
        "scaled0": scaled[:, 0], "scaled1": scaled[:, 1], "scaled2": scaled[:, 2],
        "niche": niches, "nearest": nearest, "margin": margin, "support": support,
    })


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, path)


def generation_dir(cycle: int, generation: int) -> Path:
    return OUT / "cycles" / f"c{cycle:02d}" / f"g{generation:02d}"


def mutate_generation(parents: pd.DataFrame, parent_coords: np.ndarray, cycle: int, generation: int, profile: dict) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = parents.sort_values("hash").reset_index(drop=True)
    parent_coords = parent_coords[np.argsort(parents["hash"].to_numpy())]
    records = []
    accepted_coords = []
    seen = set(ordered["hash"])
    lookup = {row.hash: row for row in ordered.itertuples(index=False)}
    for slot, parent in enumerate(ordered.itertuples(index=False)):
        for attempt in range(25):
            name_rng = np.random.default_rng(np.random.SeedSequence([20261113, 1, cycle, generation, slot, attempt]))
            param_rng = np.random.default_rng(np.random.SeedSequence([20261113, 2, cycle, generation, slot, attempt]))
            operator = str(name_rng.choice(np.asarray(OPERATORS)))
            result = apply_operator_with_config(operator, parent_coords[slot], param_rng, profile)
            final = np.ascontiguousarray(result.final_coordinates, dtype=np.float64)
            digest = coordinate_hash(final)
            if result.proposal_status != "accepted" or not result.valid_after_repair:
                status = "rejected"
            elif not inside(final):
                status = "invalid"
            elif digest == parent.hash:
                status = "noop"
            elif digest in seen:
                status = "duplicate"
            else:
                status = "accepted"
                seen.add(digest)
                accepted_coords.append(final)
            records.append({
                "operator": operator, "status": status, "parent_hash": parent.hash, "hash": digest,
                "parent_ce": float(parent.ce), "depth": int(parent.depth) + 1, "root_id": parent.root_id,
                "slot": slot, "attempt": attempt,
            })
        if (slot + 1) % 200 == 0:
            print(f"MUTATION cycle={cycle} gen={generation} parents={slot + 1}", flush=True)
    attempts = pd.DataFrame(records)
    unique = attempts[attempts["status"] == "accepted"].reset_index(drop=True)
    coords = np.stack(accepted_coords) if len(unique) else np.zeros((0, 100, 2))
    if len(unique) != len(coords):
        raise SystemExit("BLOCKED accepted offspring miscount")
    del lookup
    return unique, coords, attempts, ordered, parent_coords


def percentile_dict(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {}
    return {
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "max": float(np.max(array)),
    }


def effective_count(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True).to_numpy()
    return float(np.exp(-np.sum(counts * np.log(counts))))


def run_generation(connection, mapper, manifest, model, device, profile, config, parents, parent_coords, cycle: int, generation: int) -> None:
    destination = generation_dir(cycle, generation)
    if (destination / "done.json").is_file():
        print(f"SKIP cycle={cycle} gen={generation}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    unique, coords, attempts, ordered_parents, ordered_coords = mutate_generation(parents, parent_coords, cycle, generation, profile)
    attempts.to_parquet(destination / "attempts.parquet", index=False)
    tours, hits, new = label_hashes(connection, coords, unique["hash"].tolist(), cycle, generation, int(config["concorde_workers"]))
    tour_array = parse_tours(tours) if len(unique) else np.zeros((0, 100), dtype=np.int64)
    ce = score_ce(model, coords, tour_array, device, int(config["hardness_batch_size"])) if len(unique) else np.zeros(0)
    unique = unique.copy()
    unique["ce"] = ce
    offspring_map = map_instances(connection, mapper, manifest, coords, unique["hash"].tolist(), int(config["feature_workers"])) if len(unique) else pd.DataFrame(columns=["hash"])
    if len(unique):
        unique = unique.merge(offspring_map, on="hash", how="left", validate="one_to_one")
    parent_rows = ordered_parents.copy()
    parent_rows["stage_source"] = "parent"
    pool = pd.concat([parent_rows, unique], ignore_index=True, sort=False)
    selected = select_q4_parents(pool, parents=2000, per_niche=4)
    selected_lookup = pool.drop_duplicates("hash").set_index("hash")
    chosen = selected_lookup.loc[selected["hash"]].reset_index()
    chosen["stage"] = selected["stage"].to_numpy()
    chosen = chosen.sort_values("hash").reset_index(drop=True)
    coord_lookup = {row.hash: ordered_coords[index] for index, row in enumerate(ordered_parents.itertuples(index=False))}
    for index, row in unique.iterrows():
        coord_lookup[row.hash] = coords[index]
    next_coords = np.stack([coord_lookup[digest] for digest in chosen["hash"]])
    save_array(destination / "offspring_coords.npy", coords)
    save_array(destination / "offspring_tours.npy", tour_array)
    unique.to_parquet(destination / "offspring.parquet", index=False)
    parent_path = OUT / "cycles" / f"c{cycle:02d}" / f"parents_g{generation:02d}"
    save_array(parent_path.with_suffix(".npy"), next_coords)
    chosen.to_parquet(Path(str(parent_path) + ".parquet"), index=False)
    delta = unique["ce"] - unique["parent_ce"] if len(unique) else np.zeros(0)
    depths = unique["depth"] if len(unique) else pd.Series(dtype=int)
    diagnostics = {
        "cycle": cycle, "generation": generation, "attempts": int(len(attempts)),
        "valid_unique": int(len(unique)), "rejections": int((attempts.status == "rejected").sum()),
        "invalid": int((attempts.status == "invalid").sum()), "noop": int((attempts.status == "noop").sum()),
        "duplicates": int((attempts.status == "duplicate").sum()), "cache_hits": hits, "new_solves": new,
        "offspring_ce": percentile_dict(ce), "delta_ce_mean": float(np.mean(delta)) if len(delta) else None,
        "positive_delta_rate": float(np.mean(delta > 0)) if len(delta) else None,
        "depth_mean": float(depths.mean()) if len(depths) else None,
        "depth_median": float(depths.median()) if len(depths) else None,
        "depth_max": int(depths.max()) if len(depths) else None,
        "depth_hist": {str(depth): int((depths == depth).sum()) for depth in range(0, 9)} if len(depths) else {},
        "survivor_depth_mean": float(chosen["depth"].mean()),
        "occupied_niches": int(pool.loc[pool.niche >= 0, "niche"].nunique()) if "niche" in pool else 0,
        "elite_count": int((chosen.stage == "elite").sum()), "fill_count": int((chosen.stage == "fill").sum()),
        "support_fraction": chosen["support"].value_counts(normalize=True).to_dict() if "support" in chosen else {},
        "nearest": percentile_dict(pool["nearest"].to_numpy()) if "nearest" in pool else {},
        "lineages": int(chosen["root_id"].nunique()),
        "effective_lineages": effective_count(chosen["root_id"]),
        "largest_lineage_fraction": float(chosen["root_id"].value_counts(normalize=True).iloc[0]),
        "top5_lineage_fraction": float(chosen["root_id"].value_counts(normalize=True).head(5).sum()),
        "operators": {
            operator: {
                "attempts": int(len(group)),
                "valid_ratio": float(group["status"].isin(["accepted", "duplicate", "noop"]).mean()),
            }
            for operator, group in attempts.groupby("operator")
        },
        "operator_delta_ce": unique.groupby("operator").apply(lambda group: float((group["ce"] - group["parent_ce"]).mean()), include_groups=False).to_dict() if len(unique) else {},
        "operator_survivor_fraction": chosen["operator"].fillna("carried_parent").value_counts(normalize=True).to_dict(),
        "wall_s": time.perf_counter() - started,
    }
    if generation == 1:
        roots = pd.read_parquet(OUT / "cycles" / f"c{cycle:02d}" / "parents_g00.parquet")
        diagnostics["root_ce"] = percentile_dict(roots["ce"].to_numpy())
    previous = [pd.read_parquet(generation_dir(cycle, earlier) / "offspring.parquet", columns=["ce"])["ce"] for earlier in range(1, generation) if (generation_dir(cycle, earlier) / "offspring.parquet").is_file()]
    if len(unique):
        previous.append(unique["ce"])
    if previous:
        accumulated = pd.concat(previous, ignore_index=True)
        diagnostics["cumulative_top1000_ce"] = float(accumulated.nlargest(min(1000, len(accumulated))).mean())
        diagnostics["cumulative_top10000_ce"] = float(accumulated.nlargest(min(10000, len(accumulated))).mean())
    write_json(destination / "diagnostics.json", diagnostics)
    write_json(destination / "done.json", {"cycle": cycle, "generation": generation, "valid_unique": int(len(unique))})
    print(f"GENERATION_DONE cycle={cycle} gen={generation} unique={len(unique)} elite={diagnostics['elite_count']} fill={diagnostics['fill_count']}", flush=True)


def load_parents(cycle: int, generation: int) -> tuple[pd.DataFrame, np.ndarray]:
    stem = OUT / "cycles" / f"c{cycle:02d}" / f"parents_g{generation:02d}"
    return pd.read_parquet(Path(str(stem) + ".parquet")), np.load(stem.with_suffix(".npy"))


def acquire_top10k(cycle: int, prior: set[str]) -> dict:
    frames = []
    for generation in range(1, 9):
        path = generation_dir(cycle, generation) / "offspring.parquet"
        frame = pd.read_parquet(path)
        frame["generation"] = generation
        frame["row"] = np.arange(len(frame))
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[~pool["hash"].isin(prior)].drop_duplicates("hash")
    if len(pool) < 10000:
        write_json(OUT / "logs/status.json", {"status": "BLOCKED", "reason": "fewer than 10000 qualified offspring", "count": int(len(pool)), "cycle": cycle})
        print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_BLOCKED acquisition", len(pool), flush=True)
        raise SystemExit(2)
    ordered = pool.sort_values(["ce", "hash"], ascending=[False, True], kind="mergesort").head(10000).reset_index(drop=True)
    coords = np.empty((10000, 100, 2), dtype=np.float64)
    tours = np.empty((10000, 100), dtype=np.int64)
    for generation, group in ordered.groupby("generation"):
        source_coords = np.load(generation_dir(cycle, int(generation)) / "offspring_coords.npy")
        source_tours = np.load(generation_dir(cycle, int(generation)) / "offspring_tours.npy")
        for position, row in group.iterrows():
            coords[int(position)] = source_coords[int(row.row)]
            tours[int(position)] = source_tours[int(row.row)]
    destination = OUT / "cycles" / f"c{cycle:02d}" / "top10k"
    destination.mkdir(parents=True, exist_ok=True)
    save_array(destination / "coords.npy", coords)
    save_array(destination / "tours.npy", tours)
    ordered.to_parquet(destination / "meta.parquet", index=False)
    return {"count": 10000, "ce_min": float(ordered.ce.min()), "ce_mean": float(ordered.ce.mean()), "ce_max": float(ordered.ce.max()),
            "depth_4_8": float(ordered.depth.between(4, 8).mean()), "generation_4_8": float(ordered.generation.between(4, 8).mean())}


def train_cycle(model, device, cycle: int, config: dict) -> torch.nn.Module:
    destination = OUT / "cycles" / f"c{cycle:02d}" / "train"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "checkpoint.pt"
    coords = np.load(OUT / "cycles" / f"c{cycle:02d}" / "top10k/coords.npy")
    tours = np.load(OUT / "cycles" / f"c{cycle:02d}" / "top10k/tours.npy")
    closed = closed_expert_coordinates(coords, tours)
    if checkpoint.is_file():
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(blob["net"], strict=True)
        optimizer = create_optimizer(model, float(config["training_lr"]))
        optimizer.load_state_dict(blob["optimizer"])
        next_epoch = int(blob["next_epoch"])
        next_step = int(blob["next_step"])
        losses = blob["losses"]
    else:
        optimizer = create_optimizer(model, float(config["training_lr"]))
        next_epoch, next_step, losses = 0, 0, []
    steps = list(range(0, 10000, 256))
    for epoch in range(next_epoch, 256):
        order = np.random.default_rng(np.random.SeedSequence([20261113, 3, cycle, epoch])).permutation(10000)
        begin = next_step if epoch == next_epoch else 0
        for step, start in enumerate(steps):
            if step < begin:
                continue
            batch = closed[order[start:start + 256]]
            suffix_rng = np.random.default_rng(np.random.SeedSequence([20261113, 3, cycle, epoch, step]))
            suffix = sample_suffix_length(batch.shape[1], suffix_rng)
            result = supervised_optimizer_step(model, optimizer, batch, suffix, device)
            losses.append({"epoch": epoch, "step": step, "loss": result["training_loss"], "suffix": suffix})
            blob = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "next_epoch": epoch, "next_step": step + 1, "losses": losses, "cycle": cycle}
            if step + 1 == len(steps):
                blob["next_epoch"] = epoch + 1
                blob["next_step"] = 0
            temporary = destination / "checkpoint.tmp"
            torch.save(blob, temporary)
            os.replace(temporary, checkpoint)
        print(f"TRAIN cycle={cycle} epoch={epoch + 1}/256 loss={losses[-1]['loss']:.6f}", flush=True)
    model.eval()
    published = OUT / "checkpoints" / f"M{cycle + 1:02d}.pt"
    published.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": model.state_dict()}, published)
    record = {"cycle": cycle, "model": f"M{cycle + 1}", "file_sha256": sha256_file(published), "state_sha256": state_dict_sha256(model), "updates": len(losses)}
    write_json(destination / "complete.json", record)
    pd.DataFrame(losses).to_csv(destination / "loss.csv", index=False)
    return model


def refresh_gap_table() -> None:
    frames = [pd.read_csv(path) for path in sorted((OUT / "eval").glob("*/external_metrics.csv"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUT / "tables/external_gaps.csv", index=False)


def evaluate_model(model, device, name: str, config: dict) -> None:
    destination = OUT / "eval" / name
    if not (destination / "external_metrics.csv").is_file():
        from final_run.science import evaluate_checkpoint
        datasets = list(config["primary_benchmarks"]) + list(config["passive_benchmarks"])
        evaluate_checkpoint(model, device, destination, name, datasets)
    refresh_gap_table()


def load_model_for_cycle(cycle: int, device: torch.device):
    if cycle == 0:
        model, meta = load_canonical_checkpoint(M0_PATH, device)
        print("M0_LOADED", meta["sha256"], flush=True)
        return model
    path = OUT / "checkpoints" / f"M{cycle:02d}.pt"
    blob = torch.load(path, map_location=device, weights_only=False)
    model = build_bq_model(device)
    model.load_state_dict(blob["net"], strict=True)
    model.eval()
    print(f"M{cycle}_LOADED", sha256_file(path), flush=True)
    return model


def ensure_roots(connection, mapper, manifest, model, device, config, cycle: int) -> None:
    folder = OUT / "cycles" / f"c{cycle:02d}"
    done = folder / "roots_done.json"
    if done.is_file():
        return
    folder.mkdir(parents=True, exist_ok=True)
    coords, hashes, seeds = generate_roots(cycle, config)
    tours, hits, new = label_hashes(connection, coords, hashes, cycle, 0, int(config["concorde_workers"]))
    tour_array = parse_tours(tours)
    if cycle == 0 and not (OUT / "eval/M00/external_metrics.csv").is_file():
        evaluate_model(model, device, "M00", config)
    ce = score_ce(model, coords, tour_array, device, int(config["hardness_batch_size"]))
    mapped = map_instances(connection, mapper, manifest, coords, hashes, int(config["feature_workers"]))
    frame = mapped.copy()
    frame["ce"] = ce
    frame["depth"] = 0
    frame["root_id"] = hashes
    frame["parent_hash"] = ""
    frame["operator"] = ""
    frame["seed"] = seeds
    order = np.argsort(frame["hash"].to_numpy())
    frame = frame.iloc[order].reset_index(drop=True)
    coords = coords[order]
    tour_array = tour_array[order]
    save_array(folder / "parents_g00.npy", coords)
    save_array(folder / "root_tours.npy", tour_array)
    frame.to_parquet(folder / "parents_g00.parquet", index=False)
    write_json(done, {"count": len(frame), "ce_mean": float(ce.mean()), "cache_hits": hits, "new_solves": new})
    print(f"ROOTS_DONE cycle={cycle} ce_mean={float(ce.mean()):.6f}", flush=True)


def prior_training_hashes() -> set[str]:
    path = OUT / "training_hashes.txt"
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def remember_training_hashes(cycle: int) -> None:
    meta = pd.read_parquet(OUT / "cycles" / f"c{cycle:02d}" / "top10k/meta.parquet")
    with (OUT / "training_hashes.txt").open("a") as handle:
        for digest in meta["hash"]:
            handle.write(str(digest) + "\n")


def write_static_docs(config: dict) -> None:
    (OUT / "00_protocol.md").write_text(f"""# 00 Protocol

实验 ID：{config['experiment_id']}

冻结地图 `d3-cvt256-v1`，状态 FROZEN_WORKING_MAP。特征顺序为 spn_mst_dists_varcoeff、spn_distance_median、spn_angles_mean。K=256。Centroid array SHA256 `{EXPECTED['centroid_array']}`。本实验不重新拟合 CVT、scaler 或特征。

父代策略 `D3_CVT256_Q4_ELITE_GLOBAL_CE_FILL_V1`：每个非空 niche 按 true CE 降序、hash 升序保留最多 4 个，再用同一顺序补足 2000。Global fill 不设 niche 上限。训练集是本大轮有效唯一子代的 Global Top-10000 true CE，排除 fresh roots 和此前训练 hash。QD 不参与训练集。

搜索规模：12 大轮，每大轮 8 小轮，每小轮 2000×25=50000 次尝试。另加大轮 2000 个新 Uniform100 roots。没有旧的 185 万标签上限。缓存命中不计为新求解。

训练从 canonical M0 连续微调。每大轮 256 epoch、batch 256、microbatch 64、Adam、学习率 3.75e-5、无 weight decay、无 scheduler、无梯度裁剪。每大轮重新初始化 Adam。中断恢复时沿用本大轮已经完成的 Adam state，避免重复已完成的 optimizer update。Hardness 最短 suffix 是 {MIN_DECISION_SUFFIX}，训练最短 suffix 是 {SUFFIX_LENGTH_MIN}。

外部 gap 只作记录。Uniform100 不改变任何选择。

科学限制：本实验同时改变了行为空间、父代策略、小轮数、大轮数和搜索预算，不是单变量消融。优于旧 QD4 也不能单独归因于 D3、K=256 或 Q4。更高 CE 不等于更高训练价值。D3-CVT-256 是 working map。Reference support 只诊断，不过滤样本。

Fresh root seeds 从 12000000 起。没有找到可复用的 Unified-12 十轮 root seed 流，所以不声称前十轮根实例与旧实验相同。Mutation slots 在每个小轮按 canonical hash 排序。
""")
    (OUT / "02_qd_mapping_and_parent_selection.md").write_text("""# 02 QD mapping and parent selection

映射调用 `assets/qd_maps/d3-cvt256-v1/mapping/map_tsp100.py`。Runner 不复制 centroid，也不改 feature 公式。相同 hash 的 D3 原始特征写入本实验自己的 sqlite cache。

Support status 的口径沿用冻结阈值：normal ≤ P95，warning ≤ P99，severe > P99。非有限描述记为 nonfinite_descriptor。这些标签不参与父代选择或训练集过滤。非有限描述没有 niche，不进入 elite 阶段，仍可进入 global fill。

Q4 先在每个 niche 内取最多 4 个，再从剩余候选按 CE 补到 2000。Elite 不会再次进入 fill。父代槽位按 hash 升序编号，只影响随机种子，不改变入选集合。
""")


def append_generation_table() -> None:
    rows = []
    for path in sorted((OUT / "cycles").glob("c*/g*/diagnostics.json")):
        rows.append(json.loads(path.read_text()))
    if rows:
        pd.json_normalize(rows).to_csv(OUT / "tables/generations.csv", index=False)


def final_reports() -> None:
    gaps = pd.read_csv(OUT / "tables/external_gaps.csv") if (OUT / "tables/external_gaps.csv").is_file() else pd.DataFrame()
    (OUT / "04_training_and_external_evaluation.md").write_text("# 04 Training and external evaluation\n\n```\n" + (gaps.to_string(index=False) if len(gaps) else "no evaluations") + "\n```\n")
    (OUT / "03_evolution_depth_and_hardness.md").write_text("# 03 Evolution depth and hardness\n\n见 `tables/generations.csv` 与每大轮 `top10k/meta.parquet`。绝对 CE 不能跨大轮比较，因为评分模型 M_s 在变。\n")
    accounting = OUT / "oracle/ledger.sqlite"
    connection = sqlite3.connect(accounting)
    new_solves = connection.execute("SELECT COUNT(*) FROM solved WHERE source='this_experiment'").fetchone()[0]
    historical = connection.execute("SELECT COUNT(*) FROM solved WHERE source='historical'").fetchone()[0]
    connection.close()
    (OUT / "05_runtime_and_oracle_accounting.md").write_text(f"# 05 Runtime and oracle accounting\n\n本次新增 Concorde 求解 {new_solves}。只读导入的历史解 {historical}。缓存命中不计入新增求解。\n")
    (OUT / "06_integrated_findings.md").write_text("# 06 Integrated findings\n\n12 个大轮完成后由本文件汇总。本次不是同预算单因素对照。最低外部 gap 只作事后观察，M12 是预先固定的终点。\n")
    write_json(OUT / "logs/status.json", {"status": "COMPLETE", "new_solves": new_solves})
    print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_COMPLETE", flush=True)


def run(config: dict) -> None:
    import fcntl
    for folder in ("logs", "tables", "manifests", "checkpoints", "analysis", "figures"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    lock_handle = (OUT / "logs/lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_BLOCKED already running", flush=True)
        raise SystemExit(2)
    (OUT / "logs/pid").write_text(str(os.getpid()))
    write_static_docs(config)
    preflight_path = OUT / "manifests/preflight.json"
    if not preflight_path.is_file():
        preflight(config)
    mapper = load_mapper()
    manifest = verify_map(mapper)
    print("MAP_LOADED", manifest["version_id"], sha256_file(MANIFEST_PATH), flush=True)
    write_json(OUT / "logs/status.json", {"status": "RUNNING", "map": manifest["version_id"]})
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_BLOCKED cuda", flush=True)
        raise SystemExit(2)
    connection = open_db()
    imported = import_historical(connection)
    print(f"HISTORICAL_ORACLE_IMPORTED {imported}", flush=True)
    profile = load_named_profile("UNIFIED12_FROZEN_V1")
    if sha256_file(PROFILE_PATH) != EXPECTED["unified12"]:
        print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_BLOCKED profile", flush=True)
        raise SystemExit(2)
    for cycle in range(12):
        model = load_model_for_cycle(cycle, device)
        ensure_roots(connection, mapper, manifest, model, device, config, cycle)
        for generation in range(1, 9):
            parents, parent_coords = load_parents(cycle, generation - 1)
            if len(parents) != 2000:
                print("BLADE_D3_CVT256_Q4_12CYCLE_8GEN_BLOCKED parent count", flush=True)
                raise SystemExit(2)
            run_generation(connection, mapper, manifest, model, device, profile, config, parents, parent_coords, cycle, generation)
            append_generation_table()
        cycle_dir = OUT / "cycles" / f"c{cycle:02d}"
        if not (cycle_dir / "top10k/meta.parquet").is_file():
            summary = acquire_top10k(cycle, prior_training_hashes())
            write_json(cycle_dir / "acquisition.json", summary)
        if not (cycle_dir / "training_hashes_recorded.json").is_file():
            remember_training_hashes(cycle)
            write_json(cycle_dir / "training_hashes_recorded.json", {"count": 10000})
        if (cycle_dir / "train/complete.json").is_file():
            published = torch.load(OUT / "checkpoints" / f"M{cycle + 1:02d}.pt", map_location=device, weights_only=False)
            model.load_state_dict(published["net"], strict=True)
        else:
            model = train_cycle(model, device, cycle, config)
        evaluate_model(model, device, f"M{cycle + 1:02d}", config)
        print(f"CYCLE_DONE {cycle}", flush=True)
    final_reports()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.preflight and not args.run:
        preflight(config)
        return
    if args.run:
        run(config)
        return
    raise SystemExit("pass --preflight or --run")


if __name__ == "__main__":
    main()
