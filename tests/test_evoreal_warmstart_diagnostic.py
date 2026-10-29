from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_evoreal_warmstart_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("warmstart_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
warmstart = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warmstart)


def test_largest_remainder_counts_frozen_mixture() -> None:
    assert warmstart.largest_remainder_counts(10_000) == {"S1": 3958, "S2": 2500, "S3": 3542}
    assert warmstart.largest_remainder_counts(50_000) == {"S1": 19792, "S2": 12500, "S3": 17708}
    assert sum(warmstart.largest_remainder_counts(7).values()) == 7


def test_qd_hgb_select_elite_then_fill_is_unique_and_deterministic() -> None:
    n = 40
    meta = pd.DataFrame({
        "instance_hash": [f"{i:064x}" for i in range(n)],
        "niche_id": np.repeat(np.arange(10), 4),
        "coord_index": np.arange(n),
    })
    scores = np.linspace(-1.0, 1.0, n)
    first = warmstart.qd_hgb_select(meta, scores, budget=20)
    second = warmstart.qd_hgb_select(meta, scores, budget=20)
    assert first.instance_hash.tolist() == second.instance_hash.tolist()
    assert len(first) == 20
    assert first.instance_hash.nunique() == 20
    assert int((first.selection_component == "qd_hgb_elite").sum()) == 10
    assert first.iloc[:10].niche_id.nunique() == 10


def test_protocol_budget_invariants() -> None:
    assert warmstart.CALIBRATION_N == 10_000
    assert warmstart.TRAIN_N == 10_000
    assert warmstart.LOGICAL_BATCH == 256
    assert warmstart.EPOCHS == 256
    assert warmstart.UPDATES_PER_EPOCH == 40
    assert warmstart.UPDATES_PER_ARM == 10_240
    assert warmstart.LEARNING_RATE == 3.75e-5
    assert warmstart.FORMAL_CONCORDE_TIMEOUT == 3_000.0


def test_effective_niche_count() -> None:
    assert warmstart.effective_niche_count([1, 1, 1]) == 1.0
    assert np.isclose(warmstart.effective_niche_count([1, 2, 3, 4]), 4.0)
