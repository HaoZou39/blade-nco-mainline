#!/usr/bin/env python
"""DT1-A1 fixed-M5 paired pilot.

The script reads frozen DT1-A0/S6-AE artifacts and writes only under the
DT1-A1 audit directory.  It never invokes mutation, training, HGB fitting, or
an evolutionary cycle.  Stages are resumable and must run in order.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WORK = OUT / "work"
A0 = ROOT / "audits" / "dt1a_operator_hardness_preflight_v1"
RUN = ROOT / "audits" / "S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1"
CYCLES = RUN / "cycles"
C04 = CYCLES / "C04"
C05 = CYCLES / "C05"
CHECKPOINT = C04 / "09_bq_training" / "M5_E256.tsp.best"
M5_SHA256 = "ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c"
CONCORDE_SHA256 = "38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0"
SAMPLE_SEED = 20261027
TAU = 0.12629708954782198


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    projection = "*" if columns is None else ",".join(f'"{name}"' for name in columns)
    return duckdb.execute(f"select {projection} from read_parquet(?)", [str(path)]).df()


def read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    projection = "*" if columns is None else ",".join(f'"{name}"' for name in columns)
    return duckdb.execute(f"select {projection} from read_csv_auto(?)", [str(path)]).df()


def coordinate_hash(coordinates: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(coordinates, dtype="<f8").tobytes(order="C")).hexdigest()


def verify_a0() -> dict[str, Any]:
    manifest = json.loads((A0 / "AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    checked: list[dict[str, Any]] = []
    for row in manifest["files"]:
        path = A0 / row["path"]
        digest = sha256_file(path)
        if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
            raise RuntimeError(f"DT1-A0 manifest mismatch: {path}")
        checked.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest})
    integrity = json.loads((A0 / "integrity_summary.json").read_text(encoding="utf-8"))
    for row in integrity["source_files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen source mismatch: {path}")
    if sha256_file(CHECKPOINT) != M5_SHA256:
        raise RuntimeError("M5 checkpoint SHA-256 mismatch")
    if sha256_file(ROOT / "third_party" / "concorde" / "concorde") != CONCORDE_SHA256:
        raise RuntimeError("Concorde SHA-256 mismatch")

    parents = pd.read_csv(A0 / "03_parent_sampling_manifest.csv")
    attempts = pd.read_csv(A0 / "04_operator_attempt_manifest.csv")
    availability = pd.read_csv(A0 / "05_current_checkpoint_label_availability.csv")
    pilot_parents = parents[parents.pilot30].copy()
    pilot_attempts = attempts[attempts.pilot30].copy()
    assertions = {
        "frozen_parent_sample_300": len(parents) == 300 and parents.instance_hash.nunique() == 300,
        "pilot_parent_sample_30": len(pilot_parents) == 30 and pilot_parents.instance_hash.nunique() == 30,
        "frozen_attempts_3300": len(attempts) == 3300,
        "frozen_valid_unique_3281": int(attempts.evaluable_child.sum()) == 3281,
        "frozen_failures_19": int((~attempts.evaluable_child).sum()) == 19,
        "pilot_attempts_330": len(pilot_attempts) == 330,
        "pilot_valid_326": int(pilot_attempts.evaluable_child.sum()) == 326,
        "pilot_failures_4": int((~pilot_attempts.evaluable_child).sum()) == 4,
        "eleven_per_parent": bool(pilot_attempts.groupby("parent_hash").operator_id.nunique().eq(11).all()),
        "one_row_per_pair": bool(pilot_attempts.groupby(["parent_hash", "operator_id"]).size().eq(1).all()),
        "checkpoint_sha256": integrity["checkpoint"]["sha256"] == M5_SHA256,
        "full_parent_missing_214": availability.query("entity_role == 'parent' and availability_class != 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique() == 214,
        "full_child_missing_3044": availability.query("entity_role == 'child_attempt' and evaluable_candidate and availability_class != 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique() == 3044,
        "selected_child_labels_237": availability.query("entity_role == 'child_attempt' and availability_class == 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique() == 237,
    }
    if not all(assertions.values()):
        raise RuntimeError(f"DT1-A0 preflight assertion failed: {assertions}")
    return {"assertions": assertions, "checked_a0_files": checked}


def collect_tour_cache() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for cycle in range(10):
        path = CYCLES / f"C{cycle:02d}" / "07_oracle" / "T" / "ledger.parquet"
        frame = read_parquet(path)
        frame = frame[(frame.status == "solved") & frame.verified.fillna(False)].copy()
        frame["tour_source"] = str(path.relative_to(ROOT)).replace("\\", "/")
        pieces.append(frame)
    tours = pd.concat(pieces, ignore_index=True)
    conflict = tours.groupby("instance_hash").tour_hash.nunique(dropna=True)
    if int(conflict.max()) > 1:
        raise RuntimeError("conflicting cached tour hashes")
    return tours.sort_values(["instance_hash", "tour_source"]).drop_duplicates("instance_hash")


def stage_preflight() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    audit = verify_a0()
    parents = pd.read_csv(A0 / "03_parent_sampling_manifest.csv")
    attempts = pd.read_csv(A0 / "04_operator_attempt_manifest.csv")
    availability = pd.read_csv(A0 / "05_current_checkpoint_label_availability.csv")
    parents = parents[parents.pilot30].sort_values("sample_order").copy()
    attempts = attempts[attempts.pilot30].sort_values(["sample_order", "operator_id"]).copy()
    avail = availability[availability.pilot30].copy()

    # Exact 30-row sample manifest with all eleven immutable attempt IDs.
    attempt_ids = attempts.groupby("parent_hash").apply(
        lambda x: ";".join(f"C05:{int(r.parent_index)}:{r.operator_id}" for r in x.sort_values("operator_id").itertuples()),
        include_groups=False,
    )
    parent_out = parents.copy()
    parent_out["original_operator_attempt_ids"] = parent_out.instance_hash.map(attempt_ids)
    parent_out["operator_attempt_count"] = parent_out.instance_hash.map(attempts.groupby("parent_hash").size())
    parent_out["pilot_protocol"] = "DT1A1_FIXED_M5_PAIRED_PILOT_V1"
    parent_out.to_csv(OUT / "02_30parent_sample_manifest.csv", index=False)

    # Recover exact parent/child coordinates without regeneration.
    with np.load(C04 / "06_selection" / "P_next.npz", allow_pickle=False) as payload:
        parent_coords = np.asarray(payload["coordinates"], dtype=np.float64)
        parent_hashes = payload["instance_hash"].astype(str)
    with np.load(C05 / "03_mutation" / "offspring_coordinates.npz", allow_pickle=False) as payload:
        child_coords = np.asarray(payload["coordinates"], dtype=np.float64)
        child_hashes = payload["instance_hash"].astype(str)

    records: list[dict[str, Any]] = []
    coordinates: list[np.ndarray] = []
    coord_hashes: list[str] = []
    for row in parents.itertuples(index=False):
        index = int(row.parent_coord_index)
        instance_hash = str(row.instance_hash)
        points = parent_coords[index]
        if parent_hashes[index] != instance_hash or coordinate_hash(points) != instance_hash:
            raise RuntimeError(f"parent coordinate identity mismatch: {instance_hash}")
        av = avail[(avail.entity_role == "parent") & (avail.entity_hash == instance_hash)].iloc[0]
        records.append({
            "sample_order": int(row.sample_order), "entity_role": "parent", "parent_hash": instance_hash,
            "operator_id": "", "entity_hash": instance_hash, "evaluable_candidate": True,
            "proposal_status": "parent", "failure_reason": "", "coordinate_source": row.parent_coordinate_source,
            "coordinate_index": index, "preflight_availability_class": av.availability_class,
            "preexisting_M5_LBQ": av.M5_true_LBQ, "preexisting_M5_sources": av.M5_label_sources,
            "preexisting_tour_source": av.tour_source,
        })
        coordinates.append(points); coord_hashes.append(instance_hash)
    for row in attempts.itertuples(index=False):
        valid = bool(row.evaluable_child)
        points = None
        index = int(row.child_coord_index) if valid else -1
        instance_hash = str(row.child_hash)
        if valid:
            points = child_coords[index]
            if child_hashes[index] != instance_hash or coordinate_hash(points) != instance_hash:
                raise RuntimeError(f"child coordinate identity mismatch: {instance_hash}")
        av = avail[(avail.entity_role == "child_attempt") & (avail.entity_hash == instance_hash) & (avail.operator_id == row.operator_id)].iloc[0]
        records.append({
            "sample_order": int(row.sample_order), "entity_role": "child_attempt", "parent_hash": str(row.parent_hash),
            "operator_id": str(row.operator_id), "entity_hash": instance_hash,
            "evaluable_candidate": valid, "proposal_status": str(row.proposal_status),
            "failure_reason": "" if pd.isna(row.failure_reason) else str(row.failure_reason),
            "coordinate_source": str(row.child_coordinate_source) if valid else "",
            "coordinate_index": index, "preflight_availability_class": av.availability_class,
            "preexisting_M5_LBQ": av.M5_true_LBQ, "preexisting_M5_sources": av.M5_label_sources,
            "preexisting_tour_source": av.tour_source,
        })
        if valid:
            coordinates.append(points); coord_hashes.append(instance_hash)
    base = pd.DataFrame(records)
    if len(base) != 360 or int(base.evaluable_candidate.sum()) != 356:
        raise RuntimeError("pilot entity accounting mismatch")
    if len(set(coord_hashes)) != 356:
        raise RuntimeError("pilot evaluable hashes are not unique")
    np.savez_compressed(
        WORK / "pilot_evaluable_coordinates.npz",
        coordinates=np.stack(coordinates).astype(np.float64),
        instance_hash=np.asarray(coord_hashes),
    )
    base.to_csv(WORK / "evaluation_base.csv", index=False)

    parent_missing = base.query("entity_role == 'parent' and preexisting_M5_LBQ.isna()").entity_hash.nunique()
    child_missing = base.query("entity_role == 'child_attempt' and evaluable_candidate and preexisting_M5_LBQ.isna()").entity_hash.nunique()
    new_tours = base.query("evaluable_candidate and preexisting_tour_source.isna()").entity_hash.nunique()
    gate = {
        **audit,
        "status": "PASS",
        "pilot_parents": 30,
        "pilot_scheduled_attempts": 330,
        "pilot_valid_unique_children": 326,
        "pilot_failures": 4,
        "missing_M5_parent_labels": int(parent_missing),
        "missing_M5_child_labels": int(child_missing),
        "missing_M5_labels_total": int(parent_missing + child_missing),
        "new_concorde_required": int(new_tours),
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": M5_SHA256,
        "sample_seed": SAMPLE_SEED,
        "sample_manifest_sha256": sha256_file(A0 / "03_parent_sampling_manifest.csv"),
        "attempt_manifest_sha256": sha256_file(A0 / "04_operator_attempt_manifest.csv"),
        "coordinate_bundle_sha256": sha256_file(WORK / "pilot_evaluable_coordinates.npz"),
    }
    expected = (parent_missing, child_missing, new_tours)
    if expected != (24, 312, 312):
        raise RuntimeError(f"pilot gap mismatch: {expected}")
    write_json(WORK / "preflight_gate.json", gate)
    print("DT1A1_PREFLIGHT_GATE=PASS", flush=True)


def stage_oracle() -> None:
    gate = json.loads((WORK / "preflight_gate.json").read_text(encoding="utf-8"))
    if gate["status"] != "PASS" or sha256_file(CHECKPOINT) != M5_SHA256:
        raise RuntimeError("preflight gate/checkpoint changed")
    base = pd.read_csv(WORK / "evaluation_base.csv")
    with np.load(WORK / "pilot_evaluable_coordinates.npz", allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
    lookup = {value: coordinates[i] for i, value in enumerate(hashes)}
    cache = collect_tour_cache()
    cache_hashes = set(cache.instance_hash.astype(str))
    needed = base[base.evaluable_candidate].entity_hash.astype(str).tolist()
    missing = [value for value in needed if value not in cache_hashes]
    if len(missing) != 312:
        raise RuntimeError(f"expected 312 new Concorde calls, got {len(missing)}")

    sys.path.insert(0, str(ROOT / "src"))
    from oracle.concorde import resolve_backend, solve_one

    backend = resolve_backend(ROOT / "third_party" / "concorde" / "concorde")
    if backend["binary_hash"] != CONCORDE_SHA256:
        raise RuntimeError("resolved Concorde binary mismatch")
    work_dir = WORK / "concorde_work"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_wall = time.perf_counter(); started_cpu = time.process_time()

    def solve_with_retry(instance_hash: str) -> dict[str, Any]:
        first_error = ""
        for attempt in (0, 1):
            try:
                row = solve_one(lookup[instance_hash], instance_hash, work_dir / instance_hash, backend)
                row["retry_count"] = attempt
                row["first_error"] = first_error
                return row
            except Exception as exc:
                if attempt == 0:
                    first_error = repr(exc)
                else:
                    return {"instance_hash": instance_hash, "status": "failed", "verified": False,
                            "retry_count": 1, "first_error": first_error, "final_error": repr(exc)}
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(solve_with_retry, value): value for value in missing}
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row["status"] == "solved" and bool(row["verified"]): rows.append(row)
            else: failures.append(row)
            if count % 50 == 0 or count == len(missing):
                print(f"DT1A1_CONCORDE_PROGRESS={count}/{len(missing)}", flush=True)
    wall = time.perf_counter() - started_wall; cpu = time.process_time() - started_cpu
    if failures:
        pd.DataFrame(failures).to_csv(WORK / "concorde_failures.csv", index=False)
        raise RuntimeError(f"Concorde failures after retry: {len(failures)}")
    new_ledger = pd.DataFrame(rows).sort_values("instance_hash").reset_index(drop=True)
    if len(new_ledger) != 312 or not bool(new_ledger.verified.all()):
        raise RuntimeError("new Concorde ledger incomplete")
    new_ledger["tour_source"] = "audits/dt1a_operator_hardness_pilot_v1/work/new_concorde_ledger.parquet"
    new_ledger.to_parquet(WORK / "new_concorde_ledger.parquet", index=False)

    cache = cache[cache.instance_hash.astype(str).isin(needed)].copy()
    all_tours = pd.concat([cache, new_ledger], ignore_index=True)
    all_tours = all_tours.sort_values(["instance_hash", "tour_source"]).drop_duplicates("instance_hash")
    all_tours = all_tours.set_index("instance_hash").loc[needed].reset_index()
    if len(all_tours) != 356 or not bool(all_tours.verified.all()):
        raise RuntimeError("pilot tour inventory incomplete")
    all_tours.to_parquet(WORK / "pilot_tours.parquet", index=False)
    timing = {
        "status": "PASS", "requested": 356, "cache_hits": 44, "new_concorde_calls": 312,
        "failures": 0, "retries": int(new_ledger.retry_count.sum()), "workers": 8,
        "wall_seconds": wall, "python_process_cpu_seconds": cpu,
        "solver_runtime_seconds_sum": float(new_ledger.runtime_sec.sum()),
        "solver_runtime_seconds_mean": float(new_ledger.runtime_sec.mean()),
        "solver_runtime_seconds_median": float(new_ledger.runtime_sec.median()),
        "backend": backend, "platform": platform.platform(), "python": platform.python_version(),
    }
    write_json(WORK / "concorde_timing.json", timing)
    print(f"DT1A1_CONCORDE_COMPLETE wall_s={wall:.6f} retries={timing['retries']}", flush=True)


def stage_lbq() -> None:
    if sha256_file(CHECKPOINT) != M5_SHA256:
        raise RuntimeError("M5 checkpoint changed")
    base = pd.read_csv(WORK / "evaluation_base.csv")
    with np.load(WORK / "pilot_evaluable_coordinates.npz", allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        hashes = payload["instance_hash"].astype(str).tolist()
    coord_lookup = {value: coordinates[i] for i, value in enumerate(hashes)}
    tours = read_parquet(WORK / "pilot_tours.parquet").set_index("instance_hash")
    missing = base[base.evaluable_candidate & base.preexisting_M5_LBQ.isna()].entity_hash.astype(str).tolist()
    if len(missing) != 336 or len(set(missing)) != 336:
        raise RuntimeError(f"expected 336 missing unique M5 labels, got {len(missing)}/{len(set(missing))}")
    eval_coords = np.stack([coord_lookup[value] for value in missing])
    eval_tours = np.asarray([[int(token) for token in str(tours.loc[value, "expert_tour"]).split()] for value in missing], dtype=np.int64)
    if eval_tours.shape != (336, 100):
        raise RuntimeError(f"unexpected expert tour shape {eval_tours.shape}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the preregistered pilot timing path")
    sys.path.insert(0, str(ROOT / "src"))
    from bq.checkpoint import load_canonical_checkpoint
    from bq.training_hardness import protocol_manifest, teacher_forced_training_hardness

    device = torch.device("cuda:0")
    load_wall_start = time.perf_counter(); load_cpu_start = time.process_time()
    model, model_meta = load_canonical_checkpoint(CHECKPOINT, device, expected_sha256=M5_SHA256)
    torch.cuda.synchronize(device)
    load_wall = time.perf_counter() - load_wall_start; load_cpu = time.process_time() - load_cpu_start
    event_start = torch.cuda.Event(enable_timing=True); event_end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter(); cpu_start = time.process_time()
    event_start.record()
    result = teacher_forced_training_hardness(model, eval_coords, eval_tours, device, batch_size=128)
    event_end.record(); torch.cuda.synchronize(device)
    wall = time.perf_counter() - wall_start; cpu = time.process_time() - cpu_start
    gpu_ms = float(event_start.elapsed_time(event_end))
    values = np.asarray(result["loss_raw"], dtype=float)
    logs = np.asarray(result["loss_log"], dtype=float)
    if len(values) != 336 or not np.isfinite(values).all() or not np.allclose(logs, np.log1p(values), atol=1e-12):
        raise RuntimeError("M5 hardness result invalid")
    labels = pd.DataFrame({
        "instance_hash": missing, "M5_true_LBQ": values, "M5_log1p_LBQ": logs,
        "checkpoint_sha256": M5_SHA256, "label_source": "new_DT1A1_teacher_forced_M5",
        "batch_size": 128, "batch_wall_seconds": wall, "batch_cpu_seconds": cpu,
        "batch_cuda_event_seconds": gpu_ms / 1000.0,
    })
    labels.to_parquet(WORK / "new_M5_lbq_labels.parquet", index=False)
    timing = {
        "status": "PASS", "new_LBQ_calls": 336, "reused_M5_labels": 20,
        "device": str(device), "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "checkpoint_load_wall_seconds": load_wall, "checkpoint_load_cpu_seconds": load_cpu,
        "evaluation_wall_seconds": wall, "evaluation_cpu_seconds": cpu,
        "cuda_event_seconds": gpu_ms / 1000.0, "batch_size": 128,
        "checkpoint": model_meta, "teacher_forced_protocol": protocol_manifest(),
        "failures": 0, "retries": 0, "parameter_updates": 0,
    }
    write_json(WORK / "lbq_timing.json", timing)

    label_map = dict(zip(labels.instance_hash.astype(str), labels.M5_true_LBQ.astype(float)))
    final = base.copy()
    final["M5_true_LBQ"] = final.preexisting_M5_LBQ
    needs = final.evaluable_candidate & final.M5_true_LBQ.isna()
    final.loc[needs, "M5_true_LBQ"] = final.loc[needs, "entity_hash"].map(label_map)
    final["M5_log1p_LBQ"] = np.where(final.evaluable_candidate, np.log1p(final.M5_true_LBQ.astype(float)), np.nan)
    final["label_reused"] = final.evaluable_candidate & final.preexisting_M5_LBQ.notna()
    final["label_new"] = needs
    final["label_source"] = np.where(final.label_reused, final.preexisting_M5_sources, np.where(final.label_new, "audits/dt1a_operator_hardness_pilot_v1/work/new_M5_lbq_labels.parquet", "NOT_APPLICABLE"))
    tour_frame = read_parquet(WORK / "pilot_tours.parquet", ["instance_hash", "tour_source", "tour_hash", "binary_hash", "runtime_sec"])
    final = final.merge(tour_frame.rename(columns={"instance_hash": "entity_hash", "runtime_sec": "concorde_runtime_sec"}), on="entity_hash", how="left", validate="many_to_one")
    final["checkpoint_identity"] = "M5_E256_entering_C05"
    final["checkpoint_sha256"] = M5_SHA256
    if final.loc[final.evaluable_candidate, "M5_true_LBQ"].isna().any():
        raise RuntimeError("final evaluation manifest has missing valid labels")
    if final.loc[~final.evaluable_candidate, "M5_true_LBQ"].notna().any():
        raise RuntimeError("invalid attempts unexpectedly labeled")
    final.to_csv(OUT / "03_true_lbq_evaluation_manifest.csv", index=False)
    print(f"DT1A1_LBQ_COMPLETE wall_s={wall:.6f} gpu_s={gpu_ms/1000:.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "oracle", "lbq", "all"], default="all")
    args = parser.parse_args()
    if args.stage in {"preflight", "all"}: stage_preflight()
    if args.stage in {"oracle", "all"}: stage_oracle()
    if args.stage in {"lbq", "all"}: stage_lbq()


if __name__ == "__main__":
    main()
