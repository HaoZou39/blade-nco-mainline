from __future__ import annotations
import hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]; sys.path.insert(0,str(ROOT/"src"))
BOOT=ROOT/"audits/uniform_rooted_qd_bootstrap_v1"; C0=ROOT/"audits/cycle0_initialization_validation_v1"; E1=ROOT/"runs/audits/gp4_calibration_sample_size_v1"; RUN=ROOT/"runs/blade_final_20cycle_seed2026_v1"; M0=ROOT/"assets/solver/M0/tsp.best"
O0_QD=BOOT/"05_offspring_qd_assignments.parquet"; O0_MANIFEST=BOOT/"02_offspring_manifest.parquet"; ORDER=C0/"02_gp0_calibration_design/structural_probe_order.parquet"; OLD_JSON=C0/"02_gp0_calibration_design/cal_d1_manifest.json"; HIST_HARD=C0/"03_oracle_labels/exact_hardness.parquet"; HIST_TOURS=C0/"03_oracle_labels/expert_tours.parquet"; P0_CSV=E1/"P0_MEMBERSHIP_AND_FEATURES.csv"; CENTROIDS=ROOT/"assets/qd/cvt/centroids.csv"; NORMALIZATION=ROOT/"assets/qd/normalization/normalization_A_P1P99.json"; E1_SCRIPT=E1/"run_audit.py"
spec=importlib.util.spec_from_file_location("e1_gp4",E1_SCRIPT)
if spec is None or spec.loader is None: raise RuntimeError("cannot load E1 GP4")
e1=importlib.util.module_from_spec(spec); spec.loader.exec_module(e1)
GP_CFG=e1.GP_CFG; FEATURES=list(e1.QD4_FEATURE_NAMES); NORM=[f"normalized_{x}" for x in FEATURES]
SEEDS=tuple(range(2026,2031)); SAMPLE_SEED=2026; O0_N=109785; OLD_N=435; RANDOM_N=500; P0_N=1267
INPUTS=[O0_QD,O0_MANIFEST,ORDER,OLD_JSON,HIST_HARD,HIST_TOURS,P0_CSV,CENTROIDS,NORMALIZATION,M0,E1_SCRIPT,ROOT/"src/audit/gp_modeling.py",ROOT/"src/final_run/science.py"]

def now(): return datetime.now(timezone.utc).isoformat()
def jdefault(x):
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,default=jdefault)+"\n",encoding="utf-8"); tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def member_sha(v): return hashlib.sha256(("\n".join(map(str,v))+"\n").encode()).hexdigest()
def cfg_sha(): return hashlib.sha256(json.dumps(GP_CFG,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def input_hashes():
    missing=[str(p) for p in INPUTS if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    return {str(p):sha(p) for p in INPUTS}
def md(f,cols):
    z=f[cols].copy()
    for c in z:
        if pd.api.types.is_float_dtype(z[c]): z[c]=z[c].map(lambda v:"" if pd.isna(v) else f"{v:.6f}")
    return "\n".join(["| "+" | ".join(cols)+" |","|"+"|".join("---" for _ in cols)+"|",*["| "+" | ".join(map(str,r))+" |" for r in z.itertuples(index=False,name=None)]])
def write_protocol():
    (OUT/"01_protocol.md").write_text(f"""# Cycle-0 GP bootstrap audit protocol

Question: does one deterministic uniform Random500 from frozen O0 retain at least 90% of historical Centroid435 ENR10?

- Frozen: M0, N=100, O0={O0_N:,}, QD4/P1-P99/K=1000, canonical GP4 2000x60, seeds 2026-2030, teacher-forced H0, five-formula median.
- Arms only: OLD_CENTROID435 and RANDOM500.
- Random500: coordinate-hash sorted O0, first 500 indices of `default_rng(2026).permutation`; one draw, no reroll.
- Evaluation: exact E1/E2 P0, removing only overlap with either arm.
- Gate: `ENR10_RANDOM500 / ENR10_OLD_CENTROID435 >= 0.90`, valid formulas, integrity pass.

No offspring generation, external benchmark, BQ update, GP retuning, or formal-run write.
""",encoding="utf-8")
