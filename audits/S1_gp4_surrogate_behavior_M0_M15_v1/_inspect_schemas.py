import json
import sys
from pathlib import Path

import joblib

ROOT = Path(r"D:\NCO\BLADE-NCO-mainline")
sys.path.insert(0, str(ROOT / "src"))
from final_run.science import _duckdb

FULL = ROOT / "runs" / "blade_v2_full_seed2026"


def rp(path: Path, n=3):
    quoted = path.as_posix().replace("'", "''")
    df = _duckdb().sql(f"SELECT * FROM read_parquet('{quoted}', union_by_name=true) LIMIT {n}").df()
    cnt = _duckdb().sql(f"SELECT COUNT(*) AS n FROM read_parquet('{quoted}', union_by_name=true)").df().n.iloc[0]
    return df, int(cnt)


items = [
    ("T0", FULL / "cycles" / "C00" / "06_selection" / "T0.parquet"),
    ("T1", FULL / "cycles" / "C01" / "06_selection" / "T_selected.parquet"),
    ("Tlab0", FULL / "cycles" / "C00" / "07_oracle" / "T" / "labeled.parquet"),
    ("Tlab1", FULL / "cycles" / "C01" / "07_oracle" / "T" / "labeled.parquet"),
    ("O0", FULL / "cycles" / "C00" / "05_scoring" / "O0_scored.parquet"),
    ("O1", FULL / "cycles" / "C01" / "05_scoring" / "offspring_scored.parquet"),
    ("tours", ROOT / "audits" / "cycle0_initialization_validation_v1" / "03_oracle_labels" / "expert_tours.parquet"),
    ("B0lab", FULL / "cycles" / "C00" / "07_oracle" / "B0" / "labeled.parquet"),
]
for label, path in items:
    df, n = rp(path)
    print(f"\n=== {label} n={n} ===")
    print(list(df.columns))
    extra = {}
    for col in ("component", "source", "FORMAL_SELECTION_OCCURRED"):
        if col in df.columns:
            extra[col] = df[col].unique().tolist()
    if extra:
        print("uniques sample", extra)

gp = joblib.load(FULL / "cycles" / "C00" / "02_gp" / "training" / "gp4_ensemble.joblib")
print("\nGP keys", list(gp.keys()))
print("seeds", gp.get("seeds"))
f = gp["formulas"][0]
print("formula keys", sorted(f.keys()))
print("canonical", f.get("canonical_formula"))
print("nodes", f.get("nodes"), "depth", f.get("depth"), "feature_indices", f.get("feature_indices"))
print("transform", gp["transform"])
formula = json.loads((FULL / "cycles" / "C00" / "02_gp" / "training" / "seed_2026" / "formula.json").read_text(encoding="utf-8"))
print("formula.json keys", sorted(formula.keys()))
print("n_constants", formula.get("n_constants"), formula.get("constants"))
print("gp_diagnostics", json.loads((FULL / "cycles" / "C00" / "02_gp" / "training" / "gp_diagnostics.json").read_text(encoding="utf-8")))
