#!/usr/bin/env python
"""Read-only DT1-A0 audit builder.

Reads frozen S6-AE artifacts and writes only inside this audit directory.  It
does not invoke mutation, BQ, Concorde, training, or checkpoint loading.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "audits" / "S6_AE_HGB40_EliteQD1000_PT_10kHardOnly_X1_M0_M10_v1"
CYCLES = RUN / "cycles"
C05 = CYCLES / "C05"
C04 = CYCLES / "C04"
SEED = 20261027
SAMPLE_N = 300
PILOT_N = 30
QD_COLUMNS = [
    "normalized_spn_sum_of_lowest_edge_values",
    "normalized_spn_distance_varcoeff",
    "normalized_spn_mst_depth_mean",
    "normalized_spn_mst_dists_median",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}|{SEED}|{value}".encode()).hexdigest()


def parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    names = "*" if columns is None else ",".join(f'"{x}"' for x in columns)
    return duckdb.execute(f"select {names} from read_parquet(?)", [str(path)]).df()


def csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    names = "*" if columns is None else ",".join(f'"{x}"' for x in columns)
    return duckdb.execute(f"select {names} from read_csv_auto(?)", [str(path)]).df()


def allocate_stratified_sample(frame: pd.DataFrame) -> pd.DataFrame:
    strata = ["qd_region", "origin_operator", "entering_component"]
    sizes = frame.groupby(strata, dropna=False).size().rename("stratum_size").reset_index()
    if len(sizes) > SAMPLE_N:
        raise RuntimeError(f"{len(sizes)} nonempty strata exceed sample budget {SAMPLE_N}")
    sizes["sample_n"] = 1
    remaining = SAMPLE_N - len(sizes)
    capacity = sizes.stratum_size - 1
    if remaining:
        ideal = remaining * capacity / capacity.sum()
        extra = np.floor(ideal).astype(int)
        left = remaining - int(extra.sum())
        order = sorted(
            range(len(sizes)),
            key=lambda i: (-(float(ideal.iloc[i]) - int(extra.iloc[i])), stable_key("quota", "|".join(map(str, sizes.loc[i, strata])))),
        )
        for index in order[:left]:
            extra.iloc[index] += 1
        sizes.sample_n += extra
    if int(sizes.sample_n.sum()) != SAMPLE_N or bool((sizes.sample_n > sizes.stratum_size).any()):
        raise RuntimeError("invalid deterministic stratum allocation")

    ranked = frame.merge(sizes, on=strata, validate="many_to_one")
    ranked["sampling_key"] = ranked.instance_hash.map(lambda x: stable_key("parent-sample", str(x)))
    ranked = ranked.sort_values(strata + ["sampling_key", "instance_hash"])
    ranked["within_stratum_rank"] = ranked.groupby(strata, sort=False).cumcount() + 1
    selected = ranked[ranked.within_stratum_rank <= ranked.sample_n].copy()
    selected["inclusion_probability"] = selected.sample_n / selected.stratum_size
    selected["analysis_weight"] = selected.stratum_size / selected.sample_n
    selected["sample_order_key"] = selected.instance_hash.map(lambda x: stable_key("sample-order", str(x)))
    selected = selected.sort_values(["sample_order_key", "instance_hash"]).reset_index(drop=True)
    selected.insert(0, "sample_order", np.arange(1, len(selected) + 1))

    # Pilot allocation is outcome-blind: 3 from eight lineage groups and 2
    # from the remaining three, with group order and rows fixed by SHA-256.
    origins = sorted(selected.origin_operator.unique(), key=lambda x: stable_key("pilot-origin", str(x)))
    quotas = {origin: 3 if i < PILOT_N - 2 * len(origins) else 2 for i, origin in enumerate(origins)}
    selected["pilot_key"] = selected.instance_hash.map(lambda x: stable_key("pilot-parent", str(x)))
    pilot_hashes: set[str] = set()
    for origin, group in selected.groupby("origin_operator"):
        take = quotas[str(origin)]
        pilot_hashes.update(group.sort_values(["pilot_key", "instance_hash"]).head(take).instance_hash.astype(str))
    if len(pilot_hashes) != PILOT_N:
        raise RuntimeError(f"pilot allocation yielded {len(pilot_hashes)} instead of {PILOT_N}")
    selected["pilot30"] = selected.instance_hash.astype(str).isin(pilot_hashes)
    return selected


def label_inventory() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for cycle in range(10):
        directory = CYCLES / f"C{cycle:02d}"
        b_path = directory / "B_current_hardness.parquet"
        t_path = directory / "selected_exact_hardness.parquet"
        if b_path.is_file():
            frame = parquet(b_path, ["instance_hash", "current_LBQ"]).rename(columns={"current_LBQ": "true_LBQ"})
            frame["checkpoint"] = f"M{cycle}"
            frame["label_source"] = str(b_path.relative_to(ROOT)).replace("\\", "/")
            pieces.append(frame)
        if t_path.is_file():
            frame = parquet(t_path, ["instance_hash", "current_hardness"]).rename(columns={"current_hardness": "true_LBQ"})
            frame["checkpoint"] = f"M{cycle}"
            frame["label_source"] = str(t_path.relative_to(ROOT)).replace("\\", "/")
            pieces.append(frame)
    labels = pd.concat(pieces, ignore_index=True)
    spread = labels.groupby(["instance_hash", "checkpoint"]).true_LBQ.agg(lambda x: float(x.max() - x.min()))
    # The repository's canonical cross-check tolerance for repeated BQ
    # forward evaluations is 1e-4. Preserve sources separately and reject
    # only discrepancies beyond that frozen numerical tolerance.
    if len(spread) and float(spread.max()) > 1e-4:
        raise RuntimeError(f"conflicting exact labels: max spread {float(spread.max())}")
    return labels


def tour_inventory() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for cycle in range(10):
        path = CYCLES / f"C{cycle:02d}" / "07_oracle" / "T" / "ledger.parquet"
        if path.is_file():
            frame = parquet(path, ["instance_hash", "status", "verified", "runtime_sec", "tour_hash"])
            frame = frame[(frame.status == "solved") & frame.verified.fillna(False)].copy()
            frame["tour_source"] = str(path.relative_to(ROOT)).replace("\\", "/")
            pieces.append(frame)
    tours = pd.concat(pieces, ignore_index=True)
    conflicts = tours.groupby("instance_hash").tour_hash.nunique(dropna=True)
    if len(conflicts) and int(conflicts.max()) > 1:
        raise RuntimeError("conflicting tour hashes for an identical instance hash")
    return tours.sort_values(["instance_hash", "tour_source"]).drop_duplicates("instance_hash")


def main() -> None:
    # Parent sampling frame: metadata only. No LBQ fields are read here.
    parents = parquet(C05 / "04_qd" / "parent_qd.parquet")
    archive = parquet(C05 / "01_inputs" / "archive_in.parquet", ["instance_hash", "operator_id", "first_cycle_observed"])
    components = csv(C04 / "P_parent_membership.csv", ["instance_hash", "component"])
    parents = parents.merge(archive, on="instance_hash", how="left", validate="one_to_one")
    parents = parents.merge(components, on="instance_hash", how="left", validate="one_to_one")
    parents = parents.rename(columns={"operator_id": "origin_operator", "component": "entering_component"})
    if parents.origin_operator.isna().any() or parents.entering_component.isna().any():
        raise RuntimeError("parent lineage/component provenance incomplete")
    parents["qd_region"] = parents[QD_COLUMNS].ge(0.5).astype(int).astype(str).agg("".join, axis=1)

    mutation_path = C05 / "03_mutation" / "mutation_manifest.parquet"
    mutation = parquet(mutation_path)
    if len(parents) != 10_000 or parents.instance_hash.nunique() != 10_000:
        raise RuntimeError("C05 parent population is not 10,000 unique hashes")
    counts = mutation.groupby("parent_hash").operator_id.nunique()
    if len(mutation) != 110_000 or len(counts) != 10_000 or not bool(counts.eq(11).all()):
        raise RuntimeError("mutation manifest is not a complete 10,000 x 11 schedule")
    if mutation.groupby(["parent_hash", "operator_id"]).size().max() != 1:
        raise RuntimeError("parent/operator schedule is not one-to-one")

    # Verify exact parent and accepted-child coordinate order against frozen NPZ bundles.
    with np.load(C04 / "06_selection" / "P_next.npz", allow_pickle=False) as payload:
        input_parent_hashes = payload["instance_hash"].astype(str)
    indexed = mutation[["parent_index", "parent_hash"]].drop_duplicates().sort_values("parent_index")
    if not np.array_equal(indexed.parent_hash.astype(str).to_numpy(), input_parent_hashes):
        raise RuntimeError("mutation parent_index does not match frozen C04 P_next coordinate order")
    usable_all = mutation.proposal_status.eq("accepted") & mutation.valid_after_repair & ~mutation.duplicate
    mutation["child_coord_index"] = np.nan
    mutation.loc[usable_all, "child_coord_index"] = np.arange(int(usable_all.sum()))
    with np.load(C05 / "03_mutation" / "offspring_coordinates.npz", allow_pickle=False) as payload:
        child_hashes = payload["instance_hash"].astype(str)
    if not np.array_equal(mutation.loc[usable_all, "child_hash"].astype(str).to_numpy(), child_hashes):
        raise RuntimeError("accepted mutation rows do not match frozen child coordinate order")

    sample = allocate_stratified_sample(parents)
    sample["sampling_seed"] = SEED
    sample["parent_coordinate_source"] = str((C04 / "06_selection" / "P_next.npz").relative_to(ROOT)).replace("\\", "/")
    sample["parent_coord_index"] = sample.instance_hash.map(dict(zip(input_parent_hashes, range(len(input_parent_hashes)))))
    parent_columns = [
        "sample_order", "pilot30", "instance_hash", "parent_coord_index", "parent_coordinate_source",
        "niche_id", *QD_COLUMNS, "qd_region", "candidate_age", "origin_operator",
        "first_cycle_observed", "entering_component", "stratum_size", "sample_n",
        "within_stratum_rank", "inclusion_probability", "analysis_weight", "sampling_seed",
        "sampling_key", "pilot_key",
    ]
    sample[parent_columns].to_csv(OUT / "03_parent_sampling_manifest.csv", index=False)

    attempts = mutation[mutation.parent_hash.isin(set(sample.instance_hash))].copy()
    attempts = attempts.merge(
        sample[["sample_order", "pilot30", "instance_hash"]].rename(columns={"instance_hash": "parent_hash"}),
        on="parent_hash", validate="many_to_one",
    )
    attempts["evaluable_child"] = attempts.proposal_status.eq("accepted") & attempts.valid_after_repair & ~attempts.duplicate
    attempts["child_coordinate_source"] = np.where(
        attempts.evaluable_child,
        str((C05 / "03_mutation" / "offspring_coordinates.npz").relative_to(ROOT)).replace("\\", "/"),
        "",
    )
    child_meta = csv(C05 / "candidate_manifest.csv", ["instance_hash", "niche_id", "parent_hash", "operator_id"])
    child_meta = child_meta[child_meta.parent_hash.notna()].rename(columns={"instance_hash": "child_hash", "niche_id": "child_niche_id"})
    predictions = parquet(C05 / "candidate_predictions.parquet", ["instance_hash", "hgb_predicted_log1p_hardness"]).rename(columns={"instance_hash": "child_hash"})
    selected = csv(C05 / "T_hard_top10k_membership.csv", ["instance_hash", "component"]).rename(columns={"instance_hash": "child_hash", "component": "T05_component"})
    attempts = attempts.merge(child_meta, on=["child_hash", "parent_hash", "operator_id"], how="left", validate="one_to_one")
    attempts = attempts.merge(predictions, on="child_hash", how="left", validate="many_to_one")
    attempts = attempts.merge(selected, on="child_hash", how="left", validate="many_to_one")
    attempts["entered_T05"] = attempts.T05_component.notna()
    attempts = attempts.sort_values(["sample_order", "operator_id"]).reset_index(drop=True)
    attempt_columns = [
        "sample_order", "pilot30", "parent_index", "parent_hash", "operator_id", "child_seed",
        "child_hash", "raw_child_hash", "valid_before_repair", "valid_after_repair", "repair_count",
        "proposal_status", "failure_reason", "duplicate", "evaluable_child", "child_coord_index",
        "child_coordinate_source", "child_niche_id", "hgb_predicted_log1p_hardness", "entered_T05",
        "T05_component", "cycle", "input_validation_s", "parameter_sampling_s", "raw_mutation_s",
        "raw_validation_s", "repair_s", "finalize_and_hash_s", "total_operator_s", "deduplication_s",
    ]
    attempts[attempt_columns].to_csv(OUT / "04_operator_attempt_manifest.csv", index=False)

    labels = label_inventory()
    tours = tour_inventory()
    m5 = labels[labels.checkpoint == "M5"].groupby("instance_hash").agg(
        M5_true_LBQ=("true_LBQ", "first"),
        M5_label_sources=("label_source", lambda x: ";".join(sorted(set(x)))),
    ).reset_index()
    other = labels[labels.checkpoint != "M5"].groupby("instance_hash").agg(
        other_checkpoint_labels=("checkpoint", lambda x: ";".join(sorted(set(x), key=lambda y: int(y[1:])))),
        other_label_sources=("label_source", lambda x: ";".join(sorted(set(x)))),
    ).reset_index()
    tour_short = tours[["instance_hash", "runtime_sec", "tour_hash", "tour_source"]].rename(columns={"runtime_sec": "cached_tour_runtime_sec"})

    parent_entities = sample[["sample_order", "pilot30", "instance_hash"]].rename(columns={"instance_hash": "entity_hash"})
    parent_entities["entity_role"] = "parent"
    parent_entities["operator_id"] = ""
    parent_entities["evaluable_candidate"] = True
    child_entities = attempts[["sample_order", "pilot30", "operator_id", "child_hash", "evaluable_child"]].rename(
        columns={"child_hash": "entity_hash", "evaluable_child": "evaluable_candidate"}
    )
    child_entities["entity_role"] = "child_attempt"
    availability = pd.concat([parent_entities, child_entities], ignore_index=True)
    availability = availability.merge(m5.rename(columns={"instance_hash": "entity_hash"}), on="entity_hash", how="left")
    availability = availability.merge(other.rename(columns={"instance_hash": "entity_hash"}), on="entity_hash", how="left")
    availability = availability.merge(tour_short.rename(columns={"instance_hash": "entity_hash"}), on="entity_hash", how="left")
    availability["availability_class"] = np.select(
        [
            ~availability.evaluable_candidate,
            availability.M5_true_LBQ.notna(),
            availability.other_checkpoint_labels.notna(),
            availability.tour_hash.notna(),
        ],
        ["E_INVALID_REJECTED_OR_DUPLICATE", "A_M5_TRUE_LBQ_AVAILABLE", "B_OTHER_CHECKPOINT_LABEL_ONLY", "C_TOUR_AVAILABLE_NO_LABEL"],
        default="D_NO_LABEL_NO_TOUR",
    )
    availability = availability.sort_values(["sample_order", "entity_role", "operator_id"]).reset_index(drop=True)
    availability.to_csv(OUT / "05_current_checkpoint_label_availability.csv", index=False)

    checkpoint = C04 / "09_bq_training" / "M5_E256.tsp.best"
    profile = ROOT / "src" / "mutation" / "foga11_high_v1.yaml"
    worker = ROOT / "scripts" / "foga11_high_v1_mutation_worker.py"
    source_files = [
        C04 / "06_selection" / "P_next.npz", C05 / "04_qd" / "parent_qd.parquet",
        mutation_path, C05 / "03_mutation" / "offspring_coordinates.npz",
        C05 / "candidate_predictions.parquet", C05 / "T_hard_top10k_membership.csv",
        C05 / "B_current_hardness.parquet", C05 / "selected_exact_hardness.parquet",
        C05 / "03_mutation" / "STAGE_DONE.json", C04 / "M_next_checkpoint_manifest.json",
        RUN / "07_mutation_manifest.json", checkpoint, profile, worker,
        ROOT / "src" / "mutation" / "__init__.py", ROOT / "src" / "bq" / "training_hardness.py",
    ]
    class_counts = availability.groupby(["entity_role", "availability_class"]).size().to_dict()
    summary = {
        "status": "PASS",
        "sampling_seed": SEED,
        "sample_parent_count": len(sample),
        "pilot_parent_count": int(sample.pilot30.sum()),
        "nonempty_sampling_strata": int(sample[["qd_region", "origin_operator", "entering_component"]].drop_duplicates().shape[0]),
        "sample_qd_regions": int(sample.qd_region.nunique()),
        "sample_origin_operators": int(sample.origin_operator.nunique()),
        "sample_entering_components": int(sample.entering_component.nunique()),
        "scheduled_attempts": len(attempts),
        "accepted_unique_attempts": int(attempts.evaluable_child.sum()),
        "rejected_or_excluded_attempts": int((~attempts.evaluable_child).sum()),
        "duplicate_attempts": int(attempts.duplicate.sum()),
        "unique_valid_child_hashes": int(attempts.loc[attempts.evaluable_child, "child_hash"].nunique()),
        "availability_counts": {f"{role}|{klass}": int(value) for (role, klass), value in class_counts.items()},
        "unique_parent_M5_labels_available": int(availability.query("entity_role == 'parent' and availability_class == 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique()),
        "unique_child_M5_labels_available": int(availability.query("entity_role == 'child_attempt' and availability_class == 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique()),
        "unique_parent_labels_missing": int(availability.query("entity_role == 'parent' and availability_class != 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique()),
        "unique_child_labels_missing": int(availability.query("entity_role == 'child_attempt' and evaluable_candidate and availability_class != 'A_M5_TRUE_LBQ_AVAILABLE'").entity_hash.nunique()),
        "unique_missing_tours_for_evaluable_unlabeled": int(availability.query("evaluable_candidate and availability_class == 'D_NO_LABEL_NO_TOUR'").entity_hash.nunique()),
        "checkpoint": {
            "identity": "M5_E256 entering C05",
            "path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(checkpoint),
            "manifest_path": str((C04 / "M_next_checkpoint_manifest.json").relative_to(ROOT)).replace("\\", "/"),
        },
        "source_files": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_files
        ],
    }
    (OUT / "integrity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    deliverable_names = [
        "01_research_question_and_protocol.md", "02_frozen_artifact_integrity.md",
        "03_parent_sampling_manifest.csv", "04_operator_attempt_manifest.csv",
        "05_current_checkpoint_label_availability.csv", "06_oracle_cache_and_compute_budget.md",
        "07_preregistered_statistical_analysis.md", "08_lyapunov_and_hitting_time_feasibility.md",
        "09_final_decision.md", "integrity_summary.json", "build_preflight.py",
    ]
    generated = [OUT / name for name in deliverable_names if (OUT / name).is_file()]
    manifest = {
        "builder": "build_preflight.py",
        "prohibited_computation_invoked": False,
        "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in generated],
    }
    (OUT / "AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
