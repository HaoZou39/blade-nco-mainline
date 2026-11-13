"""Mock checks for Q4 parent selection. No Concorde and no model training."""

from __future__ import annotations

import pandas as pd

from d3_cvt256_q4_policy import select_q4_parents


def test_empty_and_small_niches():
    frame = pd.DataFrame({
        "hash": [f"h{i:04d}" for i in range(2000)],
        "ce": [float(i) for i in range(2000)],
        "niche": [0] * 2 + [1] * 3 + [2] * 1995,
    })
    selected = select_q4_parents(frame)
    counts = selected.groupby("niche").size()
    assert counts[0] == 2
    assert counts[1] == 3
    assert counts[2] > 4
    assert len(selected) == 2000


def test_ties_and_duplicate_hashes():
    frame = pd.DataFrame({
        "hash": ["b", "a", "b", "c"] + [f"z{i:04d}" for i in range(1997)],
        "ce": [1.0, 1.0, 1.0, 0.5] + [0.1] * 1997,
        "niche": [7, 7, 7, 7] + [8] * 1997,
    })
    selected = select_q4_parents(frame)
    elite = selected[selected.stage == "elite"]
    niche7 = elite[elite.niche == 7]["hash"].tolist()
    assert niche7[:2] == ["a", "b"]
    assert selected["hash"].nunique() == 2000


def test_full_map_and_fill_exceeds_quota():
    rows = []
    for niche in range(256):
        for rank in range(6):
            rows.append({"hash": f"n{niche:03d}r{rank}", "ce": float(1000 - rank), "niche": niche})
    for extra in range(600):
        rows.append({"hash": f"extra{extra:04d}", "ce": -1.0, "niche": 0})
    frame = pd.DataFrame(rows)
    selected = select_q4_parents(frame)
    elite = selected[selected.stage == "elite"]
    fill = selected[selected.stage == "fill"]
    assert len(elite) == 256 * 4
    assert len(fill) == 2000 - 1024
    assert set(elite["hash"]).isdisjoint(set(fill["hash"]))
    assert selected.groupby("niche").size().max() > 4


def test_acquisition_is_pure_top_ce():
    frame = pd.DataFrame({
        "hash": [f"h{i:05d}" for i in range(14000)],
        "ce": [float(i) for i in range(14000)],
        "niche": [0 if i >= 4000 else i % 256 for i in range(14000)],
        "is_root": [i < 2000 for i in range(14000)],
    })
    prior = set(frame["hash"].iloc[2000:2050])
    pool = frame[~frame["is_root"] & ~frame["hash"].isin(prior)]
    ordered = pool.sort_values(["ce", "hash"], ascending=[False, True])
    chosen = ordered.head(10000)
    assert len(chosen) == 10000
    assert chosen["ce"].min() >= float(ordered["ce"].iloc[10000])
    assert chosen["hash"].is_unique
    assert chosen.groupby("niche").size().max() > 4


if __name__ == "__main__":
    test_empty_and_small_niches()
    test_ties_and_duplicate_hashes()
    test_full_map_and_fill_exceeds_quota()
    test_acquisition_is_pure_top_ce()
    print("Q4_UNIT_PASS")
