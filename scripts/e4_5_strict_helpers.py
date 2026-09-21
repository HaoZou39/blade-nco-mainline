"""Independent strict selection/age verifier for the E4.5 union run."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd

from final_run.science import select_population


BUDGET = 10_000


def _global_topk(pool: pd.DataFrame, scores: np.ndarray) -> list[str]:
    hashes = pool.instance_hash.astype(str).to_numpy()
    order = np.lexsort((hashes, -np.asarray(scores, dtype=float)))[:BUDGET]
    return pool.iloc[order].instance_hash.astype(str).tolist()


def verify_union_run(run_dir: Path, read_parquet: Callable[..., pd.DataFrame]) -> dict[str, Any]:
    checks = {
        "C0_union_definition": True,
        "later_union_definition": True,
        "T_selected_from_union": True,
        "T_pure_global_GP": True,
        "P_selected_from_union": True,
        "P_QD_floor_GP_fill": True,
        "provenance_preserved": True,
        "age_propagation_exact": True,
    }
    details: list[dict[str, Any]] = []
    previous_age: dict[str, int] = {}
    for cycle in range(5):
        directory = run_dir / f"C{cycle:02d}"
        union = read_parquet(directory / "05_scoring" / "C_union_scored.parquet")
        selected_t = read_parquet(directory / "06_selection" / "T_selected.parquet")
        selected_p = read_parquet(directory / "06_selection" / "P_next.parquet")
        ledger = read_parquet(directory / "06_selection" / "parent_age_ledger.parquet")
        hashes = union.instance_hash.astype(str).tolist()
        unique = len(hashes) == len(set(hashes))
        provenance = (union.is_previous_parent.notna().all() and union.is_new_offspring.notna().all()
                      and union.coord_index.astype(int).tolist() == list(range(len(union))))
        if cycle == 0:
            pool_ok = int(union.is_previous_parent.sum()) == BUDGET and bool((union.candidate_age == 0).all())
            checks["C0_union_definition"] &= pool_ok
        else:
            expected_age = np.asarray([
                previous_age[value] + 1 if value in previous_age else 0
                for value in union.instance_hash.astype(str)
            ], dtype=int)
            pool_ok = np.array_equal(union.candidate_age.to_numpy(int), expected_age)
            checks["later_union_definition"] &= pool_ok
            checks["age_propagation_exact"] &= pool_ok
        scores = union.gp_predicted_log1p_hardness.to_numpy(float)
        t_expected = _global_topk(union, scores)
        t_actual = selected_t.instance_hash.astype(str).tolist()
        p_replay = select_population(union, scores, BUDGET, f"GP{cycle}")
        p_expected = p_replay.instance_hash.astype(str).tolist()
        p_actual = selected_p.instance_hash.astype(str).tolist()
        t_ok = unique and t_actual == t_expected and len(t_actual) == BUDGET
        p_ok = unique and p_actual == p_expected and len(p_actual) == BUDGET
        components_ok = selected_p.component.astype(str).isin(["centroid_floor", "gp_fill"]).all()
        ledger_ok = (ledger.instance_hash.astype(str).tolist() == p_actual
                     and ledger.candidate_age.astype(int).tolist() == selected_p.candidate_age.astype(int).tolist())
        checks["T_selected_from_union"] &= t_ok
        checks["T_pure_global_GP"] &= t_ok and selected_t.component.astype(str).eq("global_gp").all()
        checks["P_selected_from_union"] &= p_ok
        checks["P_QD_floor_GP_fill"] &= p_ok and components_ok
        checks["provenance_preserved"] &= bool(provenance)
        checks["age_propagation_exact"] &= bool(ledger_ok)
        previous_age = dict(zip(selected_p.instance_hash.astype(str), selected_p.candidate_age.astype(int), strict=True))
        details.append({"cycle": f"C{cycle:02d}", "union_unique": unique, "pool_definition": pool_ok,
                        "T_order_exact": t_ok, "P_order_exact": p_ok,
                        "provenance": bool(provenance), "age_ledger": bool(ledger_ok)})
    return {"checks": {key: bool(value) for key, value in checks.items()}, "details": details}
