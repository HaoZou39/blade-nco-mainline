"""Q4 elite plus global-CE fill for the frozen D3-CVT-256 map.

Niche ids are an input. This module does not fit centers or read hardness gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_q4_parents(frame: pd.DataFrame, parents: int = 2000, per_niche: int = 4) -> pd.DataFrame:
    """Return exactly `parents` rows.

    Elites are the top `per_niche` finite-CE rows of each niche, ordered by
    true CE descending and canonical hash ascending. Global fill uses the same
    order on the rows that were not already chosen. Fill may add more than
    `per_niche` rows to a niche.
    """

    required = {"hash", "ce", "niche"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    work = frame.copy()
    work["hash"] = work["hash"].astype(str)
    work["ce"] = work["ce"].astype(float)
    work["niche"] = work["niche"].astype(int)
    work = work[np.isfinite(work["ce"].to_numpy())]
    work = work.sort_values(["ce", "hash"], ascending=[False, True], kind="mergesort")
    work = work.drop_duplicates("hash", keep="first")
    if work["hash"].duplicated().any():
        raise RuntimeError("duplicate hash survived Q4 deduplication")
    eligible = work[work["niche"] >= 0]
    elites = []
    for _, group in eligible.groupby("niche", sort=True):
        elites.append(group.head(per_niche))
    elite = pd.concat(elites, ignore_index=True) if elites else work.iloc[0:0]
    chosen = set(elite["hash"])
    rest = work[~work["hash"].isin(chosen)]
    need = parents - len(elite)
    if need < 0:
        raise RuntimeError(f"elite count {len(elite)} exceeds parent budget {parents}")
    if len(rest) < need:
        raise RuntimeError(f"unique candidates {len(work)} cannot fill {parents} parents")
    fill = rest.head(need)
    if set(elite["hash"]) & set(fill["hash"]):
        raise RuntimeError("global fill repeated a niche elite")
    selected = pd.concat([elite, fill], ignore_index=True)
    selected["stage"] = ["elite"] * len(elite) + ["fill"] * len(fill)
    if len(selected) != parents or selected["hash"].nunique() != parents:
        raise RuntimeError("Q4 parent count is not the required unique budget")
    return selected
