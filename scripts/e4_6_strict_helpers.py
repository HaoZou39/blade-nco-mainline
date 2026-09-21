"""Independent replay checks for the E4.6 corrected-union continuation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from final_run.science import select_population


BUDGET = 10_000


def _global_topk(pool: pd.DataFrame, scores: np.ndarray) -> list[str]:
    hashes = pool.instance_hash.astype(str).to_numpy()
    order = np.lexsort((hashes, -np.asarray(scores, dtype=float)))[:BUDGET]
    return pool.iloc[order].instance_hash.astype(str).tolist()


def _parent_first_union(previous: list[str], offspring: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in [*previous, *offspring]:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def verify_continuation(
    run_dir: Path,
    read_parquet: Callable[..., pd.DataFrame],
    start_cycle: int = 5,
    end_cycle: int = 14,
) -> dict[str, Any]:
    checks = {
        "later_union_definition": True,
        "T_selected_from_union": True,
        "T_pure_global_GP": True,
        "P_selected_from_union": True,
        "P_QD_floor_GP_fill": True,
        "provenance_preserved": True,
        "age_propagation_exact": True,
        "archive_update_exact": True,
    }
    details: list[dict[str, Any]] = []
    previous_p = read_parquet(run_dir / f"C{start_cycle - 1:02d}" / "06_selection" / "P_next.parquet")
    previous_age = dict(zip(
        previous_p.instance_hash.astype(str),
        previous_p.candidate_age.astype(int),
        strict=True,
    ))
    prior_archive = set(read_parquet(
        run_dir / f"C{start_cycle - 1:02d}" / "08_archive" / "A_next.parquet",
        ["instance_hash"],
    ).instance_hash.astype(str))

    for cycle in range(start_cycle, end_cycle + 1):
        directory = run_dir / f"C{cycle:02d}"
        previous = read_parquet(
            run_dir / f"C{cycle - 1:02d}" / "06_selection" / "P_next.parquet"
        ).instance_hash.astype(str).tolist()
        offspring = read_parquet(
            directory / "04_qd" / "offspring_qd.parquet", ["instance_hash"]
        ).instance_hash.astype(str).tolist()
        union = read_parquet(directory / "05_scoring" / "C_union_scored.parquet")
        selected_t = read_parquet(directory / "06_selection" / "T_selected.parquet")
        selected_p = read_parquet(directory / "06_selection" / "P_next.parquet")
        ledger = read_parquet(directory / "06_selection" / "parent_age_ledger.parquet")
        archive = set(read_parquet(
            directory / "08_archive" / "A_next.parquet", ["instance_hash"]
        ).instance_hash.astype(str))

        union_hashes = union.instance_hash.astype(str).tolist()
        expected_union = _parent_first_union(previous, offspring)
        union_unique = len(union_hashes) == len(set(union_hashes))
        union_definition = union_hashes == expected_union
        pset, oset = set(previous), set(offspring)
        provenance = (
            union.is_previous_parent.astype(bool).tolist()
            == [value in pset for value in union_hashes]
            and union.is_new_offspring.astype(bool).tolist()
            == [value in oset for value in union_hashes]
            and union.coord_index.astype(int).tolist() == list(range(len(union)))
        )
        expected_age = np.asarray([
            previous_age[value] + 1 if value in previous_age else 0
            for value in union_hashes
        ], dtype=int)
        age_pool_ok = np.array_equal(union.candidate_age.to_numpy(int), expected_age)

        scores = union.gp_predicted_log1p_hardness.to_numpy(float)
        t_expected = _global_topk(union, scores)
        t_actual = selected_t.instance_hash.astype(str).tolist()
        p_expected = select_population(union, scores, BUDGET, f"GP{cycle}").instance_hash.astype(str).tolist()
        p_actual = selected_p.instance_hash.astype(str).tolist()
        t_ok = union_unique and t_actual == t_expected and len(t_actual) == BUDGET
        p_ok = union_unique and p_actual == p_expected and len(p_actual) == BUDGET
        components_ok = selected_p.component.astype(str).isin(["centroid_floor", "gp_fill"]).all()
        ledger_ok = (
            ledger.instance_hash.astype(str).tolist() == p_actual
            and ledger.candidate_age.astype(int).tolist()
            == selected_p.candidate_age.astype(int).tolist()
        )
        archive_expected = prior_archive | set(t_actual)
        archive_ok = archive == archive_expected

        checks["later_union_definition"] &= union_definition and union_unique
        checks["T_selected_from_union"] &= t_ok
        checks["T_pure_global_GP"] &= t_ok and selected_t.component.astype(str).eq("global_gp").all()
        checks["P_selected_from_union"] &= p_ok
        checks["P_QD_floor_GP_fill"] &= p_ok and bool(components_ok)
        checks["provenance_preserved"] &= provenance
        checks["age_propagation_exact"] &= age_pool_ok and ledger_ok
        checks["archive_update_exact"] &= archive_ok

        details.append({
            "cycle": f"C{cycle:02d}",
            "union_unique": bool(union_unique),
            "pool_definition": bool(union_definition),
            "T_order_exact": bool(t_ok),
            "P_order_exact": bool(p_ok),
            "provenance": bool(provenance),
            "age_pool": bool(age_pool_ok),
            "age_ledger": bool(ledger_ok),
            "archive_update": bool(archive_ok),
        })
        previous_age = dict(zip(
            selected_p.instance_hash.astype(str),
            selected_p.candidate_age.astype(int),
            strict=True,
        ))
        prior_archive = archive

    return {
        "checks": {key: bool(value) for key, value in checks.items()},
        "details": details,
    }
