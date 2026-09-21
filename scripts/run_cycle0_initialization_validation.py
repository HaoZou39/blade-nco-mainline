"""Cycle-0 initialization validation audit (staged, resumable).

The default ``phase-a`` stage replays the frozen one-sweep manifest without
regenerating or replacing it, audits canonical repair, and freezes all Phase-B
geometry-only memberships before any oracle is invoked.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit.uniform_bootstrap import child_seed
from features.qd4 import QD4_FEATURE_NAMES, extract_qd4_exact
from mutation import OPERATOR_NAMES, apply_operator
from qd.coordinates import load_centroids, normalize_qd4
from qd.roots import recover_uniform_roots
from oracle.concorde import resolve_backend, solve_batch

AUDIT = ROOT / "audits" / "cycle0_initialization_validation_v1"
BOOT = ROOT / "audits" / "uniform_rooted_qd_bootstrap_v1"
OLD = Path("D:/NCO") / "BLADE-NCO"
NORMALIZED = [f"normalized_{name}" for name in QD4_FEATURE_NAMES]
MASTER_SEED = 2026


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def membership_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def old_status() -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(OLD), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(OLD), "status", "--porcelain=v1", "--untracked-files=no"], text=True)
    return {
        "git_head": head,
        "tracked_status_entries": len([x for x in status.splitlines() if x]),
        "tracked_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def prepare_protocol() -> None:
    for name in (
        "00_protocol", "01_repair_semantics", "02_gp0_calibration_design",
        "03_oracle_labels", "04_gp4_d1", "05_gp4_d2", "06_gp4_d5",
        "07_hgb_controls", "08_prediction_evaluation", "09_within_niche_ranking",
        "10_acquisition_evaluation", "11_decision", "12_figures",
    ):
        (AUDIT / name).mkdir(parents=True, exist_ok=True)
    protocol = {
        "audit": "BLADE_CYCLE0_INITIALIZATION_VALIDATION_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ROOT), "master_seed": MASTER_SEED,
        "phase_b_gate": ["A1", "A2"], "calibration_domain": "OFFSPRING_ONLY",
        "calibration_depths": [1, 2, 5], "dev_quantiles": [i / 100 for i in range(5, 100, 10)],
        "qd4": list(QD4_FEATURE_NAMES), "normalization": "deep-reference P1/P99 + clip [0,1]",
        "cvt_k": 1000, "cvt_refit": False,
        "gp4": {"population": 2000, "generations": 60, "fitness": "Huber", "selector": "SPERF", "seeds": [2026,2027,2028,2029,2030], "ensemble": "median"},
        "bootstrap": {"unit": "niche", "replicates": 10000, "seed": 2026},
        "forbidden": {"final_D0": True, "BQ_finetune": True, "external_evaluation": True, "final_runner": True, "runs_population": True},
        "old_project_baseline": old_status(),
    }
    write_json(AUDIT / "00_protocol/frozen_protocol.json", protocol)
    sources = {}
    for rel in (
        "audits/uniform_rooted_qd_bootstrap_v1/02_offspring_manifest.parquet",
        "audits/uniform_rooted_qd_bootstrap_v1/04_root_qd_assignments.parquet",
        "audits/uniform_rooted_qd_bootstrap_v1/05_offspring_qd_assignments.parquet",
        "assets/qd/roots/root_hashes.txt", "assets/qd/normalization/normalization_A_P1P99.json",
        "assets/qd/cvt/centroids.csv", "src/mutation/foga2019_operators.yaml",
        "assets/solver/M0/tsp.best",
    ):
        sources[rel] = record(ROOT / rel)
    write_json(AUDIT / "00_protocol/source_manifest.json", {"frozen_before_oracle_labeling": True, "sources": sources})


def phase_a_and_design() -> None:
    prepare_protocol()
    manifest = pd.read_parquet(BOOT / "02_offspring_manifest.parquet")
    child_qd = pd.read_parquet(BOOT / "05_offspring_qd_assignments.parquet")
    root_qd = pd.read_parquet(BOOT / "04_root_qd_assignments.parquet")
    if len(manifest) != 110000 or len(child_qd) != 109785:
        raise RuntimeError("frozen offspring audit row-count integrity failure")
    roots = recover_uniform_roots()
    replay_failures: list[dict[str, Any]] = []
    repaired_rows: list[dict[str, Any]] = []
    raw_qd_coords: list[np.ndarray] = []
    final_qd_coords: list[np.ndarray] = []
    selected_coords: dict[str, bytes] = {}
    # Replay in exact frozen manifest order. It validates both raw and repaired hashes.
    for i, row in enumerate(manifest.itertuples(index=False)):
        expected_seed = child_seed(row.root_hash, row.operator_id)
        if int(row.child_seed) != expected_seed:
            replay_failures.append({"row": i, "reason": "seed", "expected": expected_seed, "actual": int(row.child_seed)})
            continue
        result = apply_operator(row.operator_id, roots.coordinates[int(row.root_index)], np.random.default_rng(expected_seed))
        if result.raw_output_hash != row.raw_child_hash or result.final_output_hash != row.child_hash:
            replay_failures.append({"row": i, "reason": "hash", "raw": result.raw_output_hash, "final": result.final_output_hash})
        if bool(result.valid_before_repair) != bool(row.valid_before_repair) or result.repair_count != int(row.repair_count):
            replay_failures.append({"row": i, "reason": "repair_metadata"})
        if result.repair_count:
            displacement = np.linalg.norm(result.final_coordinates - result.raw_coordinates, axis=1)
            changed = displacement > 0
            reasons = result.lineage_metadata.get("repair_reasons", {})
            repaired_rows.append({
                "manifest_row": i, "root_index": int(row.root_index), "instance_hash": row.child_hash,
                "raw_instance_hash": row.raw_child_hash, "operator_id": row.operator_id,
                "proposal_status": row.proposal_status, "repair_count": int(result.repair_count),
                "coordinate_components_changed": int(np.count_nonzero(result.final_coordinates != result.raw_coordinates)),
                "nodes_changed": int(changed.sum()), "fraction_nodes_changed": float(changed.mean()),
                "changed_node_mean_displacement": float(displacement[changed].mean()),
                "changed_node_max_displacement": float(displacement[changed].max()),
                "instance_mean_displacement": float(displacement.mean()),
                "instance_max_displacement": float(displacement.max()),
                "out_of_bounds_or_nonfinite_count": len(reasons.get("out_of_bounds_or_nonfinite", [])),
                "duplicate_or_too_close_count": len(reasons.get("duplicate_or_too_close", [])),
            })
            raw_qd_coords.append(result.raw_coordinates)
            final_qd_coords.append(result.final_coordinates)
        if (i + 1) % 10000 == 0:
            print(f"PHASE_A_REPLAY={i+1}/110000", flush=True)
    repair = pd.DataFrame(repaired_rows)
    if len(repair) != 38093:
        raise RuntimeError(f"repair replay count is {len(repair)}, expected 38093")
    print("PHASE_A_QD4_BEFORE_AFTER=START", flush=True)
    raw_qd = extract_qd4_exact(np.stack(raw_qd_coords))
    final_qd = extract_qd4_exact(np.stack(final_qd_coords))
    raw_norm, final_norm = normalize_qd4(raw_qd), normalize_qd4(final_qd)
    for j, name in enumerate(QD4_FEATURE_NAMES):
        repair[f"{name}_before"] = raw_qd[:, j]
        repair[f"{name}_after"] = final_qd[:, j]
        repair[f"normalized_{name}_before"] = raw_norm[:, j]
        repair[f"normalized_{name}_after"] = final_norm[:, j]
    repair["normalized_qd4_repair_displacement"] = np.linalg.norm(final_norm - raw_norm, axis=1)
    repair["qd4_before_scientific_status"] = "finite_geometric_diagnostic_outside_admissible_domain"
    repair.to_parquet(AUDIT / "01_repair_semantics/repair_instance_metrics.parquet", index=False)

    valid_meta = manifest[(manifest.proposal_status == "accepted") & manifest.valid_after_repair & ~manifest.duplicate_status].copy()
    valid = child_qd.merge(valid_meta[["child_hash", "valid_before_repair", "repair_count"]], on="child_hash", how="left", validate="one_to_one")
    valid["group"] = np.where(valid.repair_count.gt(0), "GROUP_REPAIRED", "GROUP_NATIVE")
    root_niches = set(root_qd.niche_id.astype(int)); all_new = set(valid.niche_id.astype(int)) - root_niches
    group_summary: dict[str, Any] = {}
    for group, frame in valid.groupby("group"):
        new_mask = ~frame.niche_id.isin(root_niches)
        values = frame.normalized_qd_displacement.to_numpy(float)
        group_summary[group] = {
            "count": len(frame), "fraction": len(frame)/len(valid), "occupied_niches": int(frame.niche_id.nunique()),
            "new_niches_beyond_roots": len(set(frame.niche_id.astype(int))-root_niches),
            "same_parent_niche_rate": float(frame.niche_id.eq(frame.parent_niche_id).mean()),
            "migration_rate": float(frame.niche_id.ne(frame.parent_niche_id).mean()),
            "root_empty_entry_rate": float(new_mask.mean()),
            "mean_qd_displacement": float(np.mean(values)), "median_qd_displacement": float(np.median(values)),
            "P90_qd_displacement": float(np.quantile(values,.9)), "P95_qd_displacement": float(np.quantile(values,.95)),
        }
    native_niches = set(valid.loc[valid.group.eq("GROUP_NATIVE"), "niche_id"].astype(int))-root_niches
    repaired_niches = set(valid.loc[valid.group.eq("GROUP_REPAIRED"), "niche_id"].astype(int))-root_niches
    contribution = {"new_total": len(all_new), "native_discovered": len(native_niches), "repaired_discovered": len(repaired_niches),
                    "reachable_by_both": len(native_niches & repaired_niches), "native_only": len(native_niches-repaired_niches),
                    "repaired_only": len(repaired_niches-native_niches)}
    operator_rows = []
    for op in OPERATOR_NAMES:
        raw = manifest[manifest.operator_id.eq(op)]
        frame = valid[valid.operator_id.eq(op)]
        native = frame[frame.group.eq("GROUP_NATIVE")]; fixed = frame[frame.group.eq("GROUP_REPAIRED")]
        operator_rows.append({
            "operator": op, "raw_n": len(raw), "repair_n": int(raw.repair_count.gt(0).sum()),
            "repair_rate": float(raw.repair_count.gt(0).mean()), "final_valid_n": len(frame),
            "occupied_niches": int(frame.niche_id.nunique()), "new_niches_beyond_roots": len(set(frame.niche_id.astype(int))-root_niches),
            "native_new_niches": len(set(native.niche_id.astype(int))-root_niches),
            "repaired_new_niches": len(set(fixed.niche_id.astype(int))-root_niches),
            "median_qd_displacement": float(frame.normalized_qd_displacement.median()),
            "P90_qd_displacement": float(frame.normalized_qd_displacement.quantile(.9)),
        })
    ops = pd.DataFrame(operator_rows)
    ops.to_csv(AUDIT / "01_repair_semantics/repair_operator_summary.csv", index=False)
    qd_summary = {"group_summary": group_summary, "new_niche_contribution": contribution,
                  "repair_magnitude": repair.select_dtypes(include=[np.number]).describe(percentiles=[.5,.9,.95]).to_dict()}
    write_json(AUDIT / "01_repair_semantics/repair_qd_summary.json", qd_summary)

    classification = "A4" if replay_failures else ("A2" if contribution["repaired_only"] > 0 else "A1")
    label = {"A1":"CANONICAL_DOMAIN_HANDLING_CONFIRMED", "A2":"CANONICAL_REPAIR_BUT_STRUCTURAL_EFFECT_REQUIRES_REVIEW", "A4":"REPAIR_AUDIT_INTEGRITY_FAILURE"}[classification]
    decision = {"classification": classification, "label": label, "replay_failures": replay_failures,
                "replayed_children": len(manifest), "raw_hashes_match": not replay_failures, "final_hashes_match": not replay_failures,
                "historical_consistency": True, "mainline_semantics_changed": False, "new_niche_contribution": contribution}
    write_json(AUDIT / "01_repair_semantics/phase_A_decision.json", decision)
    write_json(AUDIT / "11_decision/phase_A_final.json", decision)
    (AUDIT / "01_repair_semantics/repair_provenance.md").write_text(REPAIR_PROVENANCE, encoding="utf-8")
    make_repair_figures(ops, valid, contribution)
    if classification not in ("A1", "A2"):
        print(f"PHASE_A_REPAIR_CLASSIFICATION={classification}", flush=True)
        return
    freeze_design(child_qd)


def _calibration_hashes(depth: int) -> list[str]:
    payload=json.loads((AUDIT/f"02_gp0_calibration_design/cal_d{depth}_manifest.json").read_text())
    return [str(row["instance_hash"]) for row in payload["members"]]


def reconstruct_selected() -> tuple[list[str], np.ndarray, pd.DataFrame]:
    dev=pd.read_parquet(AUDIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet")
    d5=_calibration_hashes(5); hashes=list(dict.fromkeys(d5+dev.instance_hash.astype(str).tolist()))
    wanted=set(hashes); manifest=pd.read_parquet(BOOT/"02_offspring_manifest.parquet")
    picked=manifest[manifest.child_hash.isin(wanted)].copy()
    if len(picked)!=len(wanted): raise RuntimeError(f"cannot map {len(wanted)-len(picked)} selected hashes to frozen manifest")
    roots=recover_uniform_roots(); coordinate_map={}
    for count,row in enumerate(picked.itertuples(index=False),1):
        result=apply_operator(row.operator_id,roots.coordinates[int(row.root_index)],np.random.default_rng(int(row.child_seed)))
        if result.final_output_hash!=row.child_hash: raise RuntimeError(f"selected replay mismatch {row.child_hash}")
        coordinate_map[row.child_hash]=result.final_coordinates
        if count%500==0: print(f"SELECTED_REPLAY={count}/{len(picked)}",flush=True)
    coords=np.stack([coordinate_map[h] for h in hashes])
    roles=[]; d5set=set(d5); devset=set(dev.instance_hash.astype(str))
    for h in hashes: roles.append("CAL_D5_AND_DEV" if h in d5set and h in devset else ("CAL_D5" if h in d5set else "CYCLE0_GP_DEV"))
    identity=pd.DataFrame({"instance_hash":hashes,"oracle_role":roles})
    return hashes,coords,identity


def oracle_labels() -> None:
    decision=json.loads((AUDIT/"01_repair_semantics/phase_A_decision.json").read_text())
    if decision["classification"] not in ("A1","A2"): raise RuntimeError("Phase B forbidden by Phase A gate")
    seal=AUDIT/"02_gp0_calibration_design/PRE_ORACLE_SEAL.json"
    if not seal.is_file() or not json.loads(seal.read_text())["sealed"]: raise RuntimeError("pre-oracle seal missing")
    hashes,coords,identity=reconstruct_selected()
    coord_path=AUDIT/"03_oracle_labels/selected_coordinates.npz"
    np.savez_compressed(coord_path,instance_hash=np.asarray(hashes),coordinates=coords)
    backend=resolve_backend(ROOT/"third_party/concorde/concorde")
    ledger_path=AUDIT/"03_oracle_labels/concorde_ledger.parquet"
    existing=pd.read_parquet(ledger_path) if ledger_path.is_file() else None
    ledger=solve_batch(coords,hashes,AUDIT/"03_oracle_labels/_concorde_work",backend,workers=8,existing=existing)
    ledger.to_parquet(ledger_path,index=False)
    identity.merge(ledger,on="instance_hash",validate="one_to_one").to_parquet(AUDIT/"03_oracle_labels/expert_tours.parquet",index=False)
    write_json(AUDIT/"03_oracle_labels/label_manifest.json",{
        "status":"EXPERT_TOURS_COMPLETE_HARDNESS_PENDING","rows":len(ledger),"membership_sha256":membership_hash(hashes),
        "cal_d5_n":len(_calibration_hashes(5)),"dev_n":len(pd.read_parquet(AUDIT/'02_gp0_calibration_design/cycle0_gp_dev.parquet')),
        "deduplicated_union_n":len(hashes),"concorde":backend,"M0_hardness_pending":True,
        "artifacts":{"coordinates":record(coord_path),"concorde_ledger":record(ledger_path),
                     "expert_tours":record(AUDIT/'03_oracle_labels/expert_tours.parquet')}})
    sources=json.loads((AUDIT/"00_protocol/source_manifest.json").read_text()); sources["localized_concorde"]=record(ROOT/"third_party/concorde/concorde")
    write_json(AUDIT/"00_protocol/source_manifest.json",sources)
    print("CONCORDE_LABELING_COMPLETE",flush=True)


def freeze_design(child_qd: pd.DataFrame) -> None:
    centroids = load_centroids()
    rows: list[dict[str, Any]] = []
    hashes = child_qd.instance_hash.astype(str).to_numpy()
    qd = child_qd[NORMALIZED].to_numpy(float)
    niches = child_qd.niche_id.to_numpy(int)
    for niche in sorted(np.unique(niches)):
        idx = np.flatnonzero(niches == niche)
        distance = np.linalg.norm(qd[idx] - centroids[niche], axis=1)
        first_ties = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0, atol=1e-15))
        first = int(first_ties[np.argmin(hashes[idx[first_ties]])])
        selected_local = [first]
        minimum = np.linalg.norm(qd[idx] - qd[idx[first]], axis=1)
        minimum[first] = -np.inf
        while len(selected_local) < len(idx):
            best = float(minimum.max())
            tied = np.flatnonzero(np.isclose(minimum, best, rtol=0, atol=1e-15))
            chosen = int(tied[np.argmin(hashes[idx[tied]])])
            selected_local.append(chosen)
            minimum = np.minimum(minimum, np.linalg.norm(qd[idx] - qd[idx[chosen]], axis=1))
            minimum[selected_local] = -np.inf
        for rank, local in enumerate(selected_local, 1):
            item = {"niche_id": int(niche), "structural_rank": rank, "instance_hash": hashes[idx[local]],
                    "centroid_distance": float(distance[local]), "normalized_qd4": qd[idx[local]].tolist()}
            for j, col in enumerate(NORMALIZED): item[col] = float(qd[idx[local], j])
            rows.append(item)
    order = pd.DataFrame(rows)
    order.to_parquet(AUDIT / "02_gp0_calibration_design/structural_probe_order.parquet", index=False)
    cal_sets = {}
    for depth in (1,2,5):
        frame = order[order.structural_rank.le(depth)].copy()
        cal_sets[depth] = set(frame.instance_hash)
        payload = {"name": f"CAL_D{depth}", "frozen_before_oracle_labeling": True, "difficulty_blind": True,
                   "nominal_probes_per_niche": depth, "total_train_n": len(frame), "occupied_niches_represented": int(frame.niche_id.nunique()),
                   "shortage_niches": int((order.groupby("niche_id").size() < depth).sum()),
                   "ordered_membership_sha256": membership_hash(frame.instance_hash.astype(str).tolist()),
                   "source_order_sha256": sha256_file(AUDIT / "02_gp0_calibration_design/structural_probe_order.parquet"),
                   "members": frame[["niche_id","structural_rank","instance_hash"]].to_dict("records")}
        write_json(AUDIT / f"02_gp0_calibration_design/cal_d{depth}_manifest.json", payload)
    if not cal_sets[1] <= cal_sets[2] <= cal_sets[5]: raise RuntimeError("calibration nesting failure")
    dev_rows=[]; quantiles=np.arange(.05,1,.10)
    for niche, group in order.groupby("niche_id", sort=True):
        remain=group[~group.instance_hash.isin(cal_sets[5])].sort_values(["centroid_distance","instance_hash"], kind="mergesort")
        if len(remain) <= 10:
            chosen=remain.copy(); chosen["target_quantile"] = np.linspace(.05,.95,len(chosen)) if len(chosen) else []
        else:
            available=remain.reset_index(drop=True); used=set(); picks=[]
            empirical=np.arange(len(available))/(len(available)-1)
            for q in quantiles:
                delta=np.abs(empirical-q); cand=np.flatnonzero(np.isclose(delta,delta.min(),rtol=0,atol=1e-15))
                # If the closest point was already used, take the closest unused point; hash breaks equal distance.
                choices=sorted((float(abs(empirical[i]-q)), str(available.iloc[i].instance_hash), int(i)) for i in range(len(available)) if i not in used)
                pick=choices[0][2]; used.add(pick); row=available.iloc[pick].copy(); row["target_quantile"]=q; picks.append(row)
            chosen=pd.DataFrame(picks)
        dev_rows.extend(chosen.to_dict("records"))
    dev=pd.DataFrame(dev_rows).sort_values(["niche_id","target_quantile","instance_hash"],kind="mergesort").reset_index(drop=True)
    if set(dev.instance_hash)&cal_sets[5] or dev.instance_hash.duplicated().any(): raise RuntimeError("dev leakage/duplication")
    dev_path=AUDIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet"; dev.to_parquet(dev_path,index=False)
    write_json(AUDIT/"02_gp0_calibration_design/cycle0_gp_dev_manifest.json",{
        "name":"CYCLE0_GP_DEV","frozen_before_oracle_labeling":True,"difficulty_blind":True,"development_only":True,
        "total_n":len(dev),"occupied_niches_represented":int(dev.niche_id.nunique()),"target_per_niche":10,
        "shortage_niches":int((dev.groupby('niche_id').size()<10).sum()),"excludes_CAL_D5":True,
        "ordered_membership_sha256":membership_hash(dev.instance_hash.astype(str).tolist()),"artifact":record(dev_path)})
    # Re-hash every frozen membership together as a pre-oracle seal.
    design_files=list((AUDIT/"02_gp0_calibration_design").glob("*"))
    write_json(AUDIT/"02_gp0_calibration_design/PRE_ORACLE_SEAL.json",{
        "sealed":True,"oracle_labels_observed":False,"files":{p.name:record(p) for p in sorted(design_files)}})
    print(f"PHASE_A_REPAIR_CLASSIFICATION={json.loads((AUDIT/'01_repair_semantics/phase_A_decision.json').read_text())['classification']}")
    print(f"D1_TRAIN_N={len(cal_sets[1])}\nD2_TRAIN_N={len(cal_sets[2])}\nD5_TRAIN_N={len(cal_sets[5])}\nGP_DEV_N={len(dev)}",flush=True)


def make_repair_figures(ops: pd.DataFrame, valid: pd.DataFrame, contribution: dict[str, int]) -> None:
    fig,ax=plt.subplots(figsize=(11,5)); ax.bar(ops.operator,100*ops.repair_rate); ax.set_ylabel("repair-required (%)"); ax.tick_params(axis='x',rotation=50); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/01_repair_rate_by_operator.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); labels=['native only','both','repaired only']; vals=[contribution['native_only'],contribution['reachable_by_both'],contribution['repaired_only']]; ax.bar(labels,vals); ax.set_ylabel("new niches beyond roots"); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/02_native_vs_repaired_new_niche_contribution.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); groups=[valid.loc[valid.group.eq(x),'normalized_qd_displacement'] for x in ['GROUP_NATIVE','GROUP_REPAIRED']]; ax.boxplot(groups,labels=['native','repaired'],showfliers=False); ax.set_ylabel('normalized parent-child QD4 displacement'); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/03_native_vs_repaired_qd_displacement.png",dpi=160); plt.close(fig)


REPAIR_PROVENANCE = """# Repair provenance

The raw output is produced by each operator's mutator and returned as `RawMutation`; the shared `execute_operator` function in `src/mutation/base.py` then calls `check_validity` and `repair_coordinates` from `src/mutation/repair_adapter.py`.

The adapter marks every non-finite/out-of-[0,1] node and the later-index node of each pair at Euclidean distance below `1e-5`. It deterministically replaces only those indexed coordinate pairs with i.i.d. Uniform([0,1]^2) draws from the materialized `repair_seed`, repeating for at most 20 passes. The union of touched node indices is recorded. A proposal fails when the touched-node fraction exceeds 0.20 or final validity fails.

This preserves instance cardinality, node ordering, unaffected coordinates, scale convention, and all legal geometry. It performs no clipping, scaling, translation, or normalization. Duplicate/too-close points are handled by resampling the later-index endpoint. There is no operator-specific repair branch: all 11 operators use the same execution adapter.

Historical inspection found the same domain-repair concept in the pre-consolidation development pipeline (`src/blade_nco/operator_bank/validity.py`). The frozen consolidated implementation was introduced by development commit `003afd4a6f13d8faa86e9f3118034eac5b36b233` and describes itself as a BLADE-specific deterministic replay/repair/validity compatibility contract around the Bossek et al. FOGA/tspgen geometry, not as a line-by-line upstream R repair implementation. Mainline's operator modules and YAML are byte-identical to that frozen development implementation, except `parameters.py` only resolves the colocated mainline configuration path.

Raw repaired outputs are finite, so QD4 is numerically calculable as a geometric diagnostic. Because every such raw output violates the admissible unit-square/spacing contract, pre-repair QD4 is explicitly tagged as outside the scientifically admissible candidate domain and is not used for calibration, training, or acquisition.
"""


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("stage",choices=["phase-a","oracle"],default="phase-a",nargs="?")
    args=parser.parse_args()
    if args.stage=="phase-a": phase_a_and_design()
    else: oracle_labels()
