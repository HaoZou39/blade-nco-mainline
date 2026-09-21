from pathlib import Path
import ast
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent

print("=== XLSX sheets ===")
xl = pd.ExcelFile(SRC / "Normalized_TSPlib95_Labels.xlsx")
print(xl.sheet_names)
for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet)
    print(f"\nSHEET {sheet!r} shape={df.shape}")
    print("columns:", list(df.columns))
    print(df.head(8).to_string())
    print("dtypes:", df.dtypes.to_dict())


def parse_type_file(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = ast.literal_eval(line)
        name = str(rec[0])
        xy = np.asarray(rec[2:], dtype=np.float64).reshape(-1, 2)
        rows.append({"name": name, "opt": rec[1], "N": int(xy.shape[0]), "xy": xy})
    return rows

print("\n=== TYPE FILES ===")
for split in ["S1", "S2", "S3"]:
    rows = parse_type_file(SRC / f"tsplib_type{split}.txt")
    print(split, "n=", len(rows), "names=", [r["name"] for r in rows], "N=", [r["N"] for r in rows])
