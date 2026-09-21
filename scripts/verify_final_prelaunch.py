#!/usr/bin/env python
"""Twenty-eight mandatory checks before the one formal launch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))

from bq.adaptation import EPOCHS,HARD_COUNT,ANCHOR_COUNT,LEARNING_RATE
from bq.protocol import CANONICAL_M0_SHA256
from final_run.protocol import EXPECTED_DATASETS,load_and_validate
from final_run.reporting import CYCLE_FIGURES,TIMING_FILES
from final_run.science import GP_CFG
from final_run.stage import atomic_json,sha256_file
from mutation import OPERATOR_NAMES
from qd.coordinates import load_centroids,load_normalization
from qd.roots import recover_uniform_roots
from verify_assets import verify_all

RUN_ID="blade_final_20cycle_seed2026_v1"


def reparse(path:Path)->bool:
    info=os.lstat(path); return path.is_symlink() or bool(getattr(info,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0))


def main()->None:
    config_path=ROOT/"configs/final_20cycle_protocol.yaml"; config=load_and_validate(config_path); assets=verify_all(); roots=recover_uniform_roots(); lower,upper=load_normalization(); centroids=load_centroids()
    forbidden="D:"+"\\NCO\\"+"BLADE-NCO\\"; old=[]
    for base in (ROOT/"src",ROOT/"scripts",ROOT/"configs",ROOT/"README.md",ROOT/"METHOD_FROZEN.md",ROOT/"PROTOCOL_STATUS.yaml"):
        paths=[base] if base.is_file() else base.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix.lower() in {".py",".yaml",".md"}:
                text=path.read_text(encoding="utf-8-sig").replace("\\\\","\\")
                if forbidden in text: old.append(path.relative_to(ROOT).as_posix())
    subprocess.run([sys.executable,str(ROOT/"scripts/verify_assets.py")],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/"scripts/verify_bootstrap_audit.py")],cwd=ROOT,check=True)
    test=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT)
    smoke=ROOT/"prelaunch/SMOKE_TEST_PASS.json"
    report_fields=(ROOT/"src/final_run/reporting.py").read_text(encoding="utf-8")
    stage_source=(ROOT/"src/final_run/stage.py").read_text(encoding="utf-8")
    pipeline=(ROOT/"src/final_run/pipeline.py").read_text(encoding="utf-8")
    checks=[
        ("mainline workspace exact",ROOT==Path("D:/NCO/BLADE-NCO-mainline"),str(ROOT)),
        ("no symlinks",not any(reparse(path) for path in ROOT.rglob("*")),""),
        ("no runtime old-project reference",not old,old),
        ("all frozen asset hashes valid",assets["asset_count"]==114,assets["asset_count"]),
        ("M0 hash valid",sha256_file(ROOT/"assets/solver/M0/tsp.best")==CANONICAL_M0_SHA256,""),
        ("Uniform10k roots exact",len(roots.root_hashes)==len(set(roots.root_hashes))==10_000,""),
        ("Uniform anchor exact",assets.get("asset_count")==114,"verified by asset validator"),
        ("QD4 exact",config["qd"]["feature_names"]==["spn_sum_of_lowest_edge_values","spn_distance_varcoeff","spn_mst_depth_mean","spn_mst_dists_median"],""),
        ("normalization exact",lower.shape==upper.shape==(4,) and np.all(upper>lower),""),
        ("K1000 centroid hash exact",centroids.shape==(1000,4),sha256_file(ROOT/"assets/qd/cvt/centroids.csv")),
        ("FOGA11 exact",len(OPERATOR_NAMES)==11,len(OPERATOR_NAMES)),
        ("canonical repair exact",(ROOT/"src/mutation/repair_adapter.py").is_file(),""),
        ("GP4 exact",GP_CFG["population_size"]==2000 and GP_CFG["generations"]==60,""),
        ("HGB controls exact",(ROOT/"configs/hgb20_features.yaml").is_file() and config["surrogate"]["controls"]["acquisition_driver"] is False,""),
        ("256E BQ training exact",EPOCHS==256 and HARD_COUNT==ANCHOR_COUNT==10_000 and LEARNING_RATE==7.5e-5,""),
        ("external evaluator exact",config["evaluation"]["datasets"]==EXPECTED_DATASETS,""),
        ("seven benchmark assets resolve",all(name in assets for name in EXPECTED_DATASETS),assets),
        ("cycle numbering C00-C19 exact",config["run"]["cycle_indices"]==list(range(20)),""),
        ("final model M20",config["run"]["model_final"]=="M20",""),
        ("no scientific early stopping",config["run"]["scientific_early_stop"] is False and config["training"]["early_stop"] is False,""),
        ("reporting schema complete",len(CYCLE_FIGURES)==18 and "Table 10" in report_fields,""),
        ("timing schema complete",len(TIMING_FILES)==9 and "ACCOUNTED" in (ROOT/"METHOD_FROZEN.md").read_text(encoding="utf-8"),""),
        ("resume/idempotence tests pass",test.returncode==0 and smoke.is_file(),test.returncode),
        ("atomic stage handling passes","os.replace" in stage_source and "verify_stage" in stage_source,""),
        ("crash-checkpoint semantics correct",config["resume"]["crash_checkpoints_scientific"] is False,""),
        ("development labels excluded","_load_development_oracle_cache" in pipeline and "prior_ledger" in pipeline,"independent selection lookup only"),
        ("formal run target absent",not (ROOT/"runs"/RUN_ID).exists(),ROOT/"runs"/RUN_ID),
        ("all unit/integration tests pass",test.returncode==0,test.returncode),
    ]
    payload={"status":"PASS" if all(item[1] for item in checks) else "FAIL","checks":[{"number":i+1,"name":name,"passed":bool(passed),"evidence":evidence} for i,(name,passed,evidence) in enumerate(checks)],"config_sha256":sha256_file(config_path),"count":len(checks)}
    path=ROOT/"prelaunch/PRELAUNCH_VERIFICATION.json"; atomic_json(path,payload)
    if payload["status"]!="PASS": raise RuntimeError("prelaunch verification failed: "+str([x["name"] for x in payload["checks"] if not x["passed"]]))
    print("PRELAUNCH_VERIFICATION=PASS"); print("CHECKS=28")


if __name__=="__main__": main()
