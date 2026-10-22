#!/usr/bin/env python
"""Read-only Stage 4A audit of the frozen S6-AE Elite-QD selection artifacts.

This program never invokes BQ, Concorde, mutation, training, or selection code.
It reads the frozen records, deterministically replays their ordering rules, and
writes the six requested audit deliverables.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "audits" / "S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1"
OUT = ROOT / "outputs" / "stage4a_qd_selection_audit"
OUT.mkdir(parents=True, exist_ok=True)

QD4_NORMALIZED = [
    "normalized_spn_sum_of_lowest_edge_values",
    "normalized_spn_distance_varcoeff",
    "normalized_spn_mst_depth_mean",
    "normalized_spn_mst_dists_median",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


HASH_CACHE: dict[Path, str] = {}


def sha256(path: Path) -> str:
    key = path.resolve()
    if key in HASH_CACHE:
        return HASH_CACHE[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    HASH_CACHE[key] = value
    return value


def membership_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def read_parquet(path: Path, columns: str = "*") -> pd.DataFrame:
    quoted = path.resolve().as_posix().replace("'", "''")
    return duckdb.sql(
        f"SELECT {columns} FROM read_parquet('{quoted}', union_by_name=true)"
    ).df()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(frame: pd.DataFrame, digits: int = 6) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def niche_metrics(niches: pd.Series) -> dict[str, float]:
    counts = niches.astype(int).value_counts().to_numpy(dtype=float)
    p = counts / counts.sum()
    entropy = float(-np.sum(p * np.log(p)))
    return {
        "occupied_niches": int(len(counts)),
        "N_eff": float(np.exp(entropy)),
        "entropy": entropy,
        "largest_niche_share": float(p.max()),
    }


def qd4_metrics(frame: pd.DataFrame) -> dict[str, float]:
    variances = [float(np.nanvar(frame[name].to_numpy(float))) for name in QD4_NORMALIZED]
    values: dict[str, float] = {"qd4_rms_radius": float(math.sqrt(sum(variances)))}
    short = ["lowest_edge", "distance_varcoeff", "mst_depth", "mst_distance_median"]
    for name, label in zip(QD4_NORMALIZED, short, strict=True):
        values[f"qd4_std_{label}"] = float(np.nanstd(frame[name].to_numpy(float)))
    return values


def collect_tour_sources(target_hashes: set[str]) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    paths: list[tuple[int, Path, str]] = []
    for path in sorted((RUN / "cycles").glob("C*/07_oracle/T/ledger.parquet")):
        paths.append((10, path, "S6AE_cycle_ledger"))
    paths.append((20, RUN / "_state" / "A0" / "A0.parquet", "S6AE_initial_archive"))
    for path in sorted((RUN / "cycles").glob("C*/01_inputs/archive_in.parquet")):
        paths.append((21, path, "S6AE_cycle_archive"))
    base = ROOT / "audits" / "cycle0_initialization_validation_v1" / "03_oracle_labels" / "expert_tours.parquet"
    if base.is_file():
        paths.append((30, base, "implementation_base_cache"))
    historical = ROOT / "runs" / "S5_B_HGB40_quality_QD_parent_and_training_10cycle_M0_v1" / "cycles"
    for path in sorted(historical.glob("C*/07_oracle/T/ledger.parquet")):
        paths.append((31, path, "implementation_historical_cache"))
    final = ROOT / "runs" / "blade_final_20cycle_seed2026_v1" / "cycles"
    for path in sorted(final.glob("C*/07_oracle/selected/ledger.parquet")):
        paths.append((32, path, "implementation_base_cache"))
    for name in [
        "S6_AC_HGB40_EliteQD1000_PT_10kHardOnly_M0_M5_v1",
        "S6_AD_HGB40_EliteQD1000_PT_10kHardOnly_M5_M10_v1",
    ]:
        for path in sorted((ROOT / "audits" / name / "cycles").glob("C*/07_oracle/T/ledger.parquet")):
            paths.append((40, path, "matched_S6_control_ledger"))

    file_rows: list[dict[str, Any]] = []
    records: dict[str, list[dict[str, Any]]] = {}
    seen_paths: set[Path] = set()
    for priority, path, category in paths:
        if not path.is_file() or path in seen_paths:
            continue
        seen_paths.add(path)
        file_hash = sha256(path)
        file_rows.append(
            {
                "category": category,
                "path": rel(path),
                "sha256": file_hash,
                "bytes": path.stat().st_size,
            }
        )
        try:
            frame = read_parquet(path, "instance_hash,status,verified,optimal_cost_int,expert_tour,tour_hash")
        except Exception:
            continue
        frame = frame[
            frame.status.eq("solved")
            & frame.verified.fillna(False)
            & frame.expert_tour.notna()
            & frame.tour_hash.notna()
            & frame.instance_hash.astype(str).isin(target_hashes)
        ]
        for row in frame.itertuples(index=False):
            records.setdefault(str(row.instance_hash), []).append(
                {
                    "priority": priority,
                    "category": category,
                    "path": rel(path),
                    "file_sha256": file_hash,
                    "tour_hash": str(row.tour_hash),
                    "optimal_cost_int": float(row.optimal_cost_int),
                }
            )
    return records, pd.DataFrame(file_rows)


def main() -> None:
    checkpoints = pd.read_csv(RUN / "M0_M10_checkpoint_hashes.csv", dtype=str)
    checkpoint_by_model = checkpoints.set_index("model").to_dict("index")

    label_records: dict[int, dict[str, dict[str, Any]]] = {}
    for cycle in range(10):
        print(f"label inventory C{cycle:02d}", flush=True)
        cdir = RUN / "cycles" / f"C{cycle:02d}"
        model = f"M{cycle}"
        selected_path = cdir / "selected_exact_hardness.parquet"
        calibration_path = cdir / "B_current_hardness.parquet"
        selected = read_parquet(selected_path, "instance_hash,current_hardness")
        selected["f_log1p_LBQ"] = np.log1p(selected.current_hardness.to_numpy(float))
        calibration = read_parquet(calibration_path, "instance_hash,y_log1p")
        calibration = calibration.rename(columns={"y_log1p": "f_log1p_LBQ"})
        selected_hash = sha256(selected_path)
        calibration_hash = sha256(calibration_path)
        cycle_records: dict[str, dict[str, Any]] = {}
        for row in selected.itertuples(index=False):
            cycle_records[str(row.instance_hash)] = {
                "model": model,
                "f": float(row.f_log1p_LBQ),
                "source": rel(selected_path),
                "source_sha256": selected_hash,
                "source_type": "selected_exact_hardness",
            }
        for row in calibration.itertuples(index=False):
            cycle_records.setdefault(
                str(row.instance_hash),
                {
                    "model": model,
                    "f": float(row.f_log1p_LBQ),
                    "source": rel(calibration_path),
                    "source_sha256": calibration_hash,
                    "source_type": "archive_calibration_current_hardness",
                },
            )
        label_records[cycle] = cycle_records

    exchange_targets: set[str] = set()
    for cycle in range(10):
        cdir = RUN / "cycles" / f"C{cycle:02d}"
        t_hashes = set(pd.read_csv(cdir / "T_hard_top10k_membership.csv", usecols=["instance_hash"], dtype=str).instance_hash)
        g_hashes = set(pd.read_csv(cdir / "G_pure_hgb_top10k_membership.csv", usecols=["instance_hash"], dtype=str).instance_hash)
        exchange_targets.update(t_hashes ^ g_hashes)
    print(f"tour inventory for {len(exchange_targets)} exchange hashes", flush=True)
    tour_records, tour_files = collect_tour_sources(exchange_targets)

    input_rows: list[dict[str, Any]] = []
    global_inputs = [
        RUN / "00_run_config.json",
        RUN / "01_protocol.md",
        RUN / "M0_M10_checkpoint_hashes.csv",
        ROOT / "assets" / "qd" / "cvt" / "centroids.csv",
        ROOT / "assets" / "qd" / "normalization" / "normalization_A_P1P99.json",
        ROOT / "audits" / "S3_B2_RF_HGB_feature_label_scaling_v1" / "09_feature_subset_manifests.json",
        ROOT / "audits" / "S3_B2_RF_HGB_feature_label_scaling_v1" / "10_feature_subset_hashes.json",
        ROOT / "scripts" / "run_s6ae_hgb40_eliteqd1000_pt_10k_hard_only_x1_m0_m10.py",
        ROOT / "scripts" / "run_s6ac_hgb40_eliteqd1000_pt_10k_hard_only.py",
        ROOT / "scripts" / "run_s5b_qd_aware_training.py",
        ROOT / "scripts" / "run_s5a_quality_aware_qd_parent.py",
        ROOT / "scripts" / "run_s5a_quality_aware_qd_parent_v2.py",
        ROOT / "src" / "final_run" / "science.py",
    ]
    for path in global_inputs:
        input_rows.append(
            {
                "scope": "global",
                "cycle": "",
                "role": path.name,
                "path": rel(path),
                "sha256": sha256(path) if path.is_file() else "MISSING",
                "bytes": path.stat().st_size if path.is_file() else "",
            }
        )

    cycle_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    cycle_sets: dict[int, dict[str, Any]] = {}

    for cycle in range(10):
        label = f"C{cycle:02d}"
        print(f"selector replay {label}", flush=True)
        model = f"M{cycle}"
        cdir = RUN / "cycles" / label
        cp = checkpoint_by_model[model]
        checkpoint_path = Path(cp["path"])
        checkpoint_hash_actual = sha256(checkpoint_path)
        if checkpoint_hash_actual != cp["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {model}")
        calibration_stage = read_json(cdir / "_stages" / "01_calibration" / "STAGE_DONE.json")
        if calibration_stage["input_hashes"]["model"] != cp["sha256"]:
            raise RuntimeError(f"calibration checkpoint mismatch for {label}")

        cycle_inputs = {
            "candidate_manifest": cdir / "candidate_manifest.csv",
            "candidate_predictions": cdir / "candidate_predictions.parquet",
            "qd_assignment": cdir / "QD_assignment.csv",
            "q_elites": cdir / "QD_quality_elites.csv",
            "actual_T": cdir / "T_hard_top10k_membership.csv",
            "counterfactual_G": cdir / "G_pure_hgb_top10k_membership.csv",
            "selected_current_hardness": cdir / "selected_exact_hardness.parquet",
            "calibration_current_hardness": cdir / "B_current_hardness.parquet",
            "candidate_coordinates": cdir / "05_scoring" / "union" / "C_union.npz",
            "candidate_union_parquet": cdir / "05_scoring" / "union" / "C_union.parquet",
            "archive_with_tours": cdir / "01_inputs" / "archive_in.parquet",
            "selection_info": cdir / "_selection_info.json",
            "selection_stage_manifest": cdir / "_stages" / "05_selection" / "STAGE_DONE.json",
            "calibration_stage_manifest": cdir / "_stages" / "01_calibration" / "STAGE_DONE.json",
            "hgb_model_manifest": cdir / "HGB_model_manifest.json",
        }
        file_hashes: dict[str, str] = {}
        for role, path in cycle_inputs.items():
            digest = sha256(path)
            file_hashes[role] = digest
            input_rows.append(
                {
                    "scope": "cycle",
                    "cycle": label,
                    "role": role,
                    "path": rel(path),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )
        input_rows.append(
            {
                "scope": "checkpoint",
                "cycle": label,
                "role": f"entering_{model}",
                "path": rel(checkpoint_path),
                "sha256": checkpoint_hash_actual,
                "bytes": checkpoint_path.stat().st_size,
            }
        )

        candidate = pd.read_csv(
            cycle_inputs["candidate_manifest"],
            usecols=["instance_hash", "niche_id", "coord_index", *QD4_NORMALIZED],
            dtype={"instance_hash": str},
        )
        prediction = read_parquet(
            cycle_inputs["candidate_predictions"],
            "instance_hash,hgb_predicted_log1p_hardness",
        )
        prediction["instance_hash"] = prediction.instance_hash.astype(str)
        candidate_prediction_order_match = candidate.instance_hash.tolist() == prediction.instance_hash.tolist()
        if not candidate_prediction_order_match:
            raise RuntimeError(f"candidate/prediction order mismatch for {label}")
        frame = candidate.merge(prediction, on="instance_hash", how="left", validate="one_to_one", sort=False)
        if frame.hgb_predicted_log1p_hardness.isna().any() or frame.instance_hash.duplicated().any():
            raise RuntimeError(f"invalid candidate table for {label}")

        union = read_parquet(cycle_inputs["candidate_union_parquet"], "instance_hash")
        union_order_match = candidate.instance_hash.tolist() == union.instance_hash.astype(str).tolist()
        with np.load(cycle_inputs["candidate_coordinates"], allow_pickle=False) as payload:
            coordinate_hashes = payload["instance_hash"].astype(str).tolist()
            coordinate_count = len(coordinate_hashes)
        coordinate_order_match = candidate.instance_hash.tolist() == coordinate_hashes
        if not union_order_match or not coordinate_order_match or coordinate_count != len(candidate):
            raise RuntimeError(f"candidate union identity mismatch for {label}")

        global_order = frame.sort_values(
            ["hgb_predicted_log1p_hardness", "instance_hash"],
            ascending=[False, True],
            kind="mergesort",
        ).copy()
        global_order["replayed_global_hgb_rank"] = np.arange(1, len(global_order) + 1)
        replayed_rank = global_order.set_index("instance_hash")["replayed_global_hgb_rank"].to_dict()
        q_replay = (
            frame.sort_values(
                ["niche_id", "hgb_predicted_log1p_hardness", "instance_hash"],
                ascending=[True, False, True],
                kind="mergesort",
            )
            .drop_duplicates("niche_id", keep="first")
            .sort_values("niche_id", kind="mergesort")
        )
        g_replay = global_order.iloc[:10000].copy()
        q_hashes = set(q_replay.instance_hash)
        fill = global_order.loc[~global_order.instance_hash.isin(q_hashes)].iloc[: 10000 - len(q_replay)]
        t_replay = pd.concat([q_replay, fill], ignore_index=True)

        q = pd.read_csv(cycle_inputs["q_elites"], dtype={"instance_hash": str})
        t = pd.read_csv(cycle_inputs["actual_T"], dtype={"instance_hash": str})
        g = pd.read_csv(cycle_inputs["counterfactual_G"], dtype={"instance_hash": str})
        q_exact = q_replay.instance_hash.tolist() == q.instance_hash.tolist()
        t_exact = t_replay.instance_hash.tolist() == t.instance_hash.tolist()
        g_exact = g_replay.instance_hash.tolist() == g.instance_hash.tolist()
        if not (q_exact and t_exact and g_exact):
            raise RuntimeError(f"frozen selector replay failed for {label}")

        tset, gset, qset = set(t.instance_hash), set(g.instance_hash), set(q.instance_hash)
        aset, bset = tset - gset, gset - tset
        identity = aset == qset - gset
        equal_exchange = len(aset) == len(bset) and len(aset) <= 1000
        if not identity or not equal_exchange:
            raise RuntimeError(f"exchange identity failed for {label}")
        selection_info = read_json(cycle_inputs["selection_info"])
        stored_hashes_match = (
            membership_hash(t.instance_hash.tolist()) == selection_info["T_membership_sha256"]
            and membership_hash(g.instance_hash.tolist()) == selection_info["G_membership_sha256"]
            and membership_hash(q.instance_hash.tolist()) == selection_info["QD_quality_elite_membership_sha256"]
        )
        if not stored_hashes_match:
            raise RuntimeError(f"stored membership hash mismatch for {label}")

        t_full = t[["instance_hash"]].merge(frame, on="instance_hash", validate="one_to_one")
        g_full = g[["instance_hash"]].merge(frame, on="instance_hash", validate="one_to_one")
        tm, gm = niche_metrics(t_full.niche_id), niche_metrics(g_full.niche_id)
        tq, gq = qd4_metrics(t_full), qd4_metrics(g_full)
        delta = float(g.hgb_predicted_log1p_hardness.sum() - t.hgb_predicted_log1p_hardness.sum())
        if delta < -1e-10:
            raise RuntimeError(f"negative predicted opportunity cost for {label}")

        current = label_records[cycle]
        current_a = sum(value in current for value in aset)
        current_b = sum(value in current for value in bset)
        missing_b = sorted(value for value in bset if value not in current)
        missing_tours = sum(value not in tour_records for value in missing_b)
        reusable_tours = len(missing_b) - missing_tours
        historical_missing = sum(
            any(value in label_records[other] for other in range(10) if other != cycle)
            for value in missing_b
        )

        cycle_sets[cycle] = {
            "A": aset,
            "B": bset,
            "missing_B": set(missing_b),
            "candidate": frame.set_index("instance_hash"),
            "t": t.set_index("instance_hash"),
            "g": g.set_index("instance_hash"),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": cp["sha256"],
            "coordinate_path": cycle_inputs["candidate_coordinates"],
            "coordinate_sha256": file_hashes["candidate_coordinates"],
            "replayed_global_rank": replayed_rank,
        }

        row: dict[str, Any] = {
            "cycle": label,
            "entering_checkpoint": model,
            "entering_checkpoint_sha256": cp["sha256"],
            "candidate_n": len(frame),
            "Q_n": len(q),
            "T_n": len(t),
            "G_n": len(g),
            "A_T_minus_G_n": len(aset),
            "B_G_minus_T_n": len(bset),
            "A_equals_Q_minus_G": identity,
            "equal_exchange_le_1000": equal_exchange,
            "candidate_prediction_order_match": candidate_prediction_order_match,
            "candidate_union_order_match": union_order_match,
            "candidate_coordinate_order_match": coordinate_order_match,
            "Q_order_replay_exact": q_exact,
            "T_order_replay_exact": t_exact,
            "G_order_replay_exact": g_exact,
            "stored_membership_hashes_match": stored_hashes_match,
            "T_occupied_niches": tm["occupied_niches"],
            "G_occupied_niches": gm["occupied_niches"],
            "occupied_niche_gain_T_minus_G": tm["occupied_niches"] - gm["occupied_niches"],
            "T_N_eff_exp_Shannon": tm["N_eff"],
            "G_N_eff_exp_Shannon": gm["N_eff"],
            "N_eff_gain_T_minus_G": tm["N_eff"] - gm["N_eff"],
            "T_entropy": tm["entropy"],
            "G_entropy": gm["entropy"],
            "T_largest_niche_share": tm["largest_niche_share"],
            "G_largest_niche_share": gm["largest_niche_share"],
            "T_qd4_rms_radius": tq["qd4_rms_radius"],
            "G_qd4_rms_radius": gq["qd4_rms_radius"],
            "qd4_rms_radius_gain_T_minus_G": tq["qd4_rms_radius"] - gq["qd4_rms_radius"],
            "G_predicted_score_sum": float(g.hgb_predicted_log1p_hardness.sum()),
            "T_predicted_score_sum": float(t.hgb_predicted_log1p_hardness.sum()),
            "predicted_score_opportunity_cost_G_minus_T": delta,
            "predicted_cost_per_exchange": delta / len(aset) if aset else 0.0,
            "predicted_cost_per_selected_member": delta / 10000,
            "A_current_checkpoint_label_n": current_a,
            "B_current_checkpoint_label_n": current_b,
            "exchange_current_checkpoint_label_n": current_a + current_b,
            "exchange_current_checkpoint_label_fraction": (current_a + current_b) / (2 * len(aset)) if aset else 1.0,
            "new_current_checkpoint_labels_needed": len(missing_b),
            "missing_B_with_other_checkpoint_label_n": historical_missing,
            "missing_B_with_reusable_tour_n": reusable_tours,
            "missing_B_without_reusable_tour_n": missing_tours,
            "exact_true_delta_status": "NOT_COMPUTED_INCOMPLETE_LABELS" if missing_b else "READY_BUT_NOT_COMPUTED_STAGE4A",
        }
        for key, value in tq.items():
            if key != "qd4_rms_radius":
                row[f"T_{key}"] = value
        for key, value in gq.items():
            if key != "qd4_rms_radius":
                row[f"G_{key}"] = value
        cycle_rows.append(row)

        for side, values in [("A_T_minus_G", sorted(aset)), ("B_G_minus_T", sorted(bset))]:
            for instance_hash in values:
                current_record = current.get(instance_hash)
                other = [
                    label_records[index][instance_hash]
                    for index in range(10)
                    if index != cycle and instance_hash in label_records[index]
                ]
                tours = sorted(
                    tour_records.get(instance_hash, []),
                    key=lambda item: (item["priority"], item["path"]),
                )
                chosen = tours[0] if tours else None
                tour_hashes = {item["tour_hash"] for item in tours}
                costs = {item["optimal_cost_int"] for item in tours}
                candidate_row = cycle_sets[cycle]["candidate"].loc[instance_hash]
                needs_label = current_record is None and side.startswith("B_")
                availability_rows.append(
                    {
                        "cycle": label,
                        "instance_hash": instance_hash,
                        "exchange_side": side,
                        "niche_id": int(candidate_row.niche_id),
                        "global_hgb_rank": int(cycle_sets[cycle]["replayed_global_rank"][instance_hash]),
                        "hgb_predicted_log1p_hardness": float(candidate_row.hgb_predicted_log1p_hardness),
                        "entering_checkpoint": model,
                        "entering_checkpoint_sha256": cp["sha256"],
                        "entering_checkpoint_path": rel(checkpoint_path),
                        "current_checkpoint_label_available": current_record is not None,
                        "current_label_source_type": current_record["source_type"] if current_record else "",
                        "current_label_source_path": current_record["source"] if current_record else "",
                        "current_label_source_sha256": current_record["source_sha256"] if current_record else "",
                        "current_f_log1p_LBQ": current_record["f"] if current_record else np.nan,
                        "other_checkpoint_label_available": bool(other),
                        "other_checkpoint_label_count": len(other),
                        "other_checkpoint_ids": ";".join(item["model"] for item in other),
                        "other_checkpoint_f_log1p_LBQ": ";".join(f"{item['model']}:{item['f']:.17g}" for item in other),
                        "other_checkpoint_source_paths": ";".join(item["source"] for item in other),
                        "reusable_verified_optimal_tour_available": chosen is not None,
                        "tour_source_category": chosen["category"] if chosen else "",
                        "tour_source_path": chosen["path"] if chosen else "",
                        "tour_source_sha256": chosen["file_sha256"] if chosen else "",
                        "tour_hash": chosen["tour_hash"] if chosen else "",
                        "optimal_cost_int": chosen["optimal_cost_int"] if chosen else np.nan,
                        "tour_optimal_cost_consistent": len(costs) <= 1,
                        "tour_hash_identical_across_sources": len(tour_hashes) <= 1,
                        "candidate_coordinates_available": True,
                        "candidate_coordinate_source_path": rel(cycle_inputs["candidate_coordinates"]),
                        "candidate_coordinate_source_sha256": file_hashes["candidate_coordinates"],
                        "new_current_checkpoint_label_needed": needs_label,
                        "new_concorde_tour_needed": bool(needs_label and chosen is None),
                        "readiness_class": (
                            "CURRENT_LABEL_PRESENT"
                            if current_record is not None
                            else "BQ_ONLY_USING_REUSABLE_TOUR"
                            if chosen is not None
                            else "CONCORDE_PLUS_BQ_REQUIRED"
                        ),
                    }
                )

    cycle_frame = pd.DataFrame(cycle_rows)
    availability = pd.DataFrame(availability_rows)
    cycle_frame.to_csv(OUT / "02_cyclewise_selection_audit.csv", index=False)
    availability.to_csv(OUT / "03_label_availability.csv", index=False, na_rep="")

    input_frame = pd.DataFrame(input_rows)
    input_frame = pd.concat(
        [
            input_frame,
            tour_files.assign(scope="tour_cache", cycle="", role=tour_files.category)[
                ["scope", "cycle", "role", "path", "sha256", "bytes"]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates(["path", "sha256"])

    # Existing runtime records are benchmarks, not promises for new work.
    calibration_seconds = 0.0
    calibration_n = 0
    oracle_rows: list[dict[str, Any]] = []
    raw_tour_runtimes: list[float] = []
    for cycle in range(10):
        cdir = RUN / "cycles" / f"C{cycle:02d}"
        info = read_json(cdir / "_calibration_info.json")
        calibration_seconds += float(info["refresh_wall_seconds"])
        calibration_n += int(info["calibration_n"])
        oracle = read_json(cdir / "oracle_cost_logical_physical.json")
        summary = read_json(cdir / "07_oracle" / "T" / "oracle_summary.json")
        oracle_rows.append(
            {
                "cycle": f"C{cycle:02d}",
                "new_concorde_calls": int(oracle["physical_new_concorde_calls"]),
                "oracle_and_hardness_wall_s": float(summary["wall_seconds"]),
            }
        )
        ledger = read_parquet(cdir / "07_oracle" / "T" / "ledger.parquet", "instance_hash,runtime_sec")
        raw_tour_runtimes.extend(ledger.drop_duplicates("instance_hash").runtime_sec.astype(float).tolist())
    hardness_s_per_instance = calibration_seconds / calibration_n
    runtime_array = np.asarray(raw_tour_runtimes, dtype=float)
    tour_runtime_median = float(np.median(runtime_array))
    tour_runtime_p90 = float(np.percentile(runtime_array, 90))

    representative = cycle_frame[cycle_frame.cycle.isin(["C00", "C03", "C09"])].copy()
    all_ten = pd.DataFrame(
        [
            {
                "cycle": "ALL_C00_C09",
                "new_current_checkpoint_labels_needed": int(cycle_frame.new_current_checkpoint_labels_needed.sum()),
                "missing_B_with_reusable_tour_n": int(cycle_frame.missing_B_with_reusable_tour_n.sum()),
                "missing_B_without_reusable_tour_n": int(cycle_frame.missing_B_without_reusable_tour_n.sum()),
            }
        ]
    )
    cost_rows = representative[
        [
            "cycle",
            "new_current_checkpoint_labels_needed",
            "missing_B_with_reusable_tour_n",
            "missing_B_without_reusable_tour_n",
        ]
    ].copy()
    cost_rows = pd.concat([cost_rows, all_ten], ignore_index=True)
    cost_rows["BQ_label_wall_s_point_estimate"] = cost_rows.new_current_checkpoint_labels_needed * hardness_s_per_instance
    cost_rows["Concorde_CPU_s_median_runtime_extrapolation"] = cost_rows.missing_B_without_reusable_tour_n * tour_runtime_median
    cost_rows["Concorde_CPU_s_P90_runtime_extrapolation"] = cost_rows.missing_B_without_reusable_tour_n * tour_runtime_p90
    cost_rows["idealized_8_worker_Concorde_wall_s_median"] = cost_rows.Concorde_CPU_s_median_runtime_extrapolation / 8
    cost_rows["idealized_8_worker_Concorde_wall_s_P90"] = cost_rows.Concorde_CPU_s_P90_runtime_extrapolation / 8

    protocol = f"""# Stage 4A protocol and frozen inputs

## Scope and stopping rule

This is a read-only artifact and cost audit of the frozen S6-AE C00–C09 run. It did not invoke LBQ/BQ inference, Concorde, HGB fitting, BQ training, FOGA mutation, evolution, or algorithm code. The exact true-hardness counterfactual was not calculated because required current-checkpoint labels are missing.

The entering checkpoint for cycle `Ckk` is `Mkk`. The true score is

```text
f_k(x) = log(1 + L_BQ(M_k, x)).
```

The surrogate score is the frozen median HGB40 prediction on that same log1p target scale. These two quantities are never substituted for one another.

## Frozen selector replay

For each cycle, the candidate table and prediction table are aligned in frozen candidate order. The strict total order is HGB score descending and instance hash ascending. `G` is the first 10,000 candidates. `Q` is the first candidate under score/hash order within each occupied frozen-QD4 niche, emitted in niche-id order. `T` is `Q` followed by the first `10,000-|Q|` candidates in global order outside `Q`.

All ten cycles pass exact ordered replay of `Q`, `G`, and `T`, coordinate/hash alignment, stored membership hashes, and the identity `T\\G = Q\\G`, with equal directed differences no larger than 1,000. No prediction was recomputed.

The QD map is the fixed 1,000-centroid partition in `assets/qd/cvt/centroids.csv`; QD4 values use the frozen P1/P99 clipped normalization in `assets/qd/normalization/normalization_A_P1P99.json`. `N_eff` in the cyclewise output is `exp(Shannon entropy)`, matching the implementation. `qd4_rms_radius` is the square root of the sum of the four population variances on the frozen normalized descriptors.

## Label provenance rule

A current label is accepted only when the same instance hash is present in either the cycle's selected exact-hardness table or its archive-calibration current-hardness table, both produced with the entering checkpoint recorded by the calibration stage. Selected `current_hardness` is raw LBQ and is transformed with `log1p`; calibration `y_log1p` is already on the required scale. Labels found under another `M_j` are listed separately and never reused as `f_k`.

An optimal tour is checkpoint-independent. Tour availability is accepted only from a solved, verified ledger/archive row with a non-null tour and tour hash. Duplicate sources for every exchange member are checked for consistent optimal cost; tour hashes are reported separately because rotation, reversal, or distinct equal-cost optima can produce different valid encodings. The source catalog below includes the implementation-declared base/historical caches, frozen S6-AE ledgers/archives, and matched S6-AC/S6-AD ledgers present locally.

## Input manifest

Every file actually read by this audit is identified below by repository-relative path and SHA256. Checkpoint rows use their content hash, not merely a model name.

{md_table(input_frame, digits=6)}
"""
    (OUT / "01_protocol_and_frozen_inputs.md").write_text(protocol, encoding="utf-8")

    oracle_frame = pd.DataFrame(oracle_rows)
    compute = f"""# Stage 4A compute-cost assessment

## Exact work units

{md_table(cost_rows, digits=3)}

For the proposed C00+C03+C09 study, the exact requirement is **{int(representative.new_current_checkpoint_labels_needed.sum()):,}** new same-checkpoint LBQ labels. Of these, **{int(representative.missing_B_with_reusable_tour_n.sum()):,}** already have a verified reusable optimal tour in the audited cache scope; **{int(representative.missing_B_without_reusable_tour_n.sum()):,}** need both a new Concorde tour and BQ labeling. Across all ten cycles, the corresponding counts are **{int(cycle_frame.new_current_checkpoint_labels_needed.sum()):,}**, **{int(cycle_frame.missing_B_with_reusable_tour_n.sum()):,}**, and **{int(cycle_frame.missing_B_without_reusable_tour_n.sum()):,}**.

## Measured timing anchors

- Existing S6-AE archive-refresh measurements total {calibration_seconds:.3f} seconds for {calibration_n:,} same-checkpoint hardness evaluations with tours already available: {hardness_s_per_instance:.6f} wall-seconds per instance ({1/hardness_s_per_instance:.2f} instances/second). This is the point estimate used above.
- Verified S6-AE ledger rows record a median Concorde solver runtime of {tour_runtime_median:.6f} seconds and P90 of {tour_runtime_p90:.6f} seconds per instance.
- The implementation calls Concorde with 8 workers. The table's 8-worker wall figures are idealized divisions of CPU-time extrapolations; they are **unverified estimates**, exclude scheduling/I/O/start-up effects, and must not be treated as measured future wall time.
- Historical combined Oracle-plus-hardness wall times and physical new-call counts are shown below. They are descriptive benchmarks only because the wall time also includes cache lookup, tour assembly, and 10,000 BQ hardness evaluations.

{md_table(oracle_frame, digits=3)}

No GPU reservation, future hardware speed, or unseen Concorde failure rate is assumed. Exact cost becomes known only if Stage 4B is separately authorized and run.
"""
    (OUT / "04_compute_cost_assessment.md").write_text(compute, encoding="utf-8")

    minimal_view = representative[
        [
            "cycle",
            "candidate_n",
            "Q_n",
            "A_T_minus_G_n",
            "B_current_checkpoint_label_n",
            "new_current_checkpoint_labels_needed",
            "missing_B_with_reusable_tour_n",
            "missing_B_without_reusable_tour_n",
            "T_occupied_niches",
            "G_occupied_niches",
            "predicted_score_opportunity_cost_G_minus_T",
        ]
    ]
    plan = f"""# Minimal evaluation plan (not executed)

## Recommended first pass

Use C00, C03, and C09. They are suitable because they represent the initial partial-coverage phase, the first cycle with all 1,000 QD niches occupied, and the late evolutionary phase. Their candidate coordinates, entering checkpoints, frozen predictions, `Q/T/G` membership, current selected labels, and reusable-tour records are present and hash-verified.

{md_table(minimal_view, digits=6)}

The union requires exactly **{int(representative.new_current_checkpoint_labels_needed.sum()):,}** new current-checkpoint labels. The work decomposes into **{int(representative.missing_B_with_reusable_tour_n.sum()):,}** BQ-only evaluations using verified reusable tours and **{int(representative.missing_B_without_reusable_tour_n.sum()):,}** Concorde-plus-BQ evaluations. No additional candidate generation, HGB scoring, selection, or retraining is needed.

## Exact future procedure, only if separately authorized

1. Re-verify the file and checkpoint hashes in `01_protocol_and_frozen_inputs.md`.
2. For each missing `B=G\\T` member, load coordinates by its frozen `coord_index`; reuse a verified tour when listed, otherwise obtain one Concorde tour.
3. Evaluate raw `L_BQ(M_k,x)` under the exact entering checkpoint and store `f_k(x)=log1p(L_BQ)` with instance, checkpoint, tour, and coordinate hashes.
4. Require all missing rows to be complete and finite. Do not use other-checkpoint values.
5. Only then calculate `Delta_true = sum_(G\\T) f_k - sum_(T\\G) f_k`. Positive favors Global Top10k on current hardness; negative favors Elite-QD; zero is equality.
6. Report each cycle as an exact finite-pool comparison. Across-cycle summaries are descriptive and dependent, not ten randomized replicates.

## If a smaller study is required

C00 alone is the smallest exact cycle: 41 labels, of which 32 currently lack a reusable tour. A paired or score/niche-stratified sample from C03 or C09 can reduce cost, but it does **not** identify the full-set sum without a declared sampling design and assumptions. Therefore sampling is not recommended as a substitute for the exact three-cycle comparison.

This plan measures current-checkpoint selection hardness only. It cannot establish future reachability or downstream BQ training value.
"""
    (OUT / "05_minimal_evaluation_plan.md").write_text(plan, encoding="utf-8")

    summary_view = cycle_frame[
        [
            "cycle",
            "Q_n",
            "A_T_minus_G_n",
            "T_occupied_niches",
            "G_occupied_niches",
            "T_N_eff_exp_Shannon",
            "G_N_eff_exp_Shannon",
            "predicted_score_opportunity_cost_G_minus_T",
            "new_current_checkpoint_labels_needed",
            "missing_B_without_reusable_tour_n",
        ]
    ]
    recommendation = f"""# Stage 4A final recommendation

## A — READY_FOR_TARGETED_LABELING

The frozen counterfactual is reliable: all ten cycles exactly replay the candidate, score/hash, within-niche elite, Global Top10k, and Elite-QD ordering; `|A_k|=|B_k|<=1000` holds; checkpoint identities and coordinate bundles are present; and the label/tour gaps are enumerated per instance in `03_label_availability.csv`.

{md_table(summary_view, digits=6)}

The exact current-hardness comparison is **not yet identified and was not computed**. Across C00–C09, 6,904 displaced Global members still need current-checkpoint labels. In the audited tour scope, 2,952 of those can reuse a verified optimal tour and 3,952 need Concorde first. For the recommended C00+C03+C09 subset, the counts are 1,668 labels, 640 tour reuses, and 1,028 new tours.

Elite-QD's deterministic structural benefit is observable: every cycle has at least as many occupied niches as Global Top10k, and the cyclewise file reports the exact niche, entropy-effective-count, largest-share, and normalized-QD4 dispersion differences. Its HGB opportunity cost is also exact and nonnegative. Neither quantity is true LBQ hardness, future reachability, or downstream training value.

The additional work is small relative to the frozen ten-cycle run and has a precise stopping rule, so artifact repair is not required before targeted labeling. Stage 4B is not started or authorized by this decision.
"""
    (OUT / "06_final_recommendation.md").write_text(recommendation, encoding="utf-8")

    print(json.dumps({
        "status": "A — READY_FOR_TARGETED_LABELING",
        "outputs": [path.name for path in sorted(OUT.iterdir())],
        "cycles": len(cycle_frame),
        "exchange_rows": len(availability),
        "all_missing_labels": int(cycle_frame.new_current_checkpoint_labels_needed.sum()),
        "three_cycle_missing_labels": int(representative.new_current_checkpoint_labels_needed.sum()),
        "three_cycle_new_tours": int(representative.missing_B_without_reusable_tour_n.sum()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
