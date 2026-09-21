#!/usr/bin/env python
"""Canonical entrypoint for the frozen BLADE-NCO 20-cycle run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from final_run.pipeline import run
from final_run.protocol import load_and_validate
from final_run.reporting import CYCLE_FIGURES, write_cycle_outputs
from final_run.stage import atomic_stage, sha256_file
from mutation import OPERATOR_NAMES, apply_operator
from qd.acquisition import centroid_global_acquire
from qd.archive import CurrentSurrogateScore
from qd.coordinates import load_centroids
from qd.roots import recover_uniform_roots


RUN_ID="blade_final_20cycle_seed2026_v1"


def smoke_test(config_path:Path)->None:
    load_and_validate(config_path)
    with tempfile.TemporaryDirectory(prefix="blade_final_smoke_") as temporary:
        work=Path(temporary)
        roots=recover_uniform_roots(); parent=roots.coordinates[0]
        children=[]
        for index,name in enumerate(OPERATOR_NAMES):
            result=apply_operator(name,parent,np.random.default_rng(9000+index))
            if result.proposal_status=="accepted" and result.valid_after_repair: children.append(result.final_coordinates)
        if not children: raise RuntimeError("mutation smoke fixture produced no valid child")
        centers=load_centroids(); hashes=[f"smoke_{i:03d}" for i in range(12)]; niches={h:i%3 for i,h in enumerate(hashes)}
        points={h:tuple(centers[niches[h]]+((i+1)*1e-6)) for i,h in enumerate(hashes)}
        scores={h:CurrentSurrogateScore(float(i),"SMOKE") for i,h in enumerate(hashes)}
        chosen=centroid_global_acquire(hashes,scores,niches,points,centers,7,current_score_version="SMOKE")
        if len(chosen)!=7 or sum(item.component=="centroid_floor" for item in chosen)!=3: raise RuntimeError("acquisition smoke failure")
        stage=work/"stage"; artifact=stage/"artifact.txt"
        with atomic_stage(stage,"smoke_atomic_stage",{},resume=False) as outputs:
            artifact.write_text("fixture\n",encoding="ascii"); outputs.append(artifact)
        with atomic_stage(stage,"smoke_atomic_stage",{},resume=True) as outputs:
            if outputs is not None: raise RuntimeError("resume did not skip completed stage")
        cycle=work/"cycle"; cycle.mkdir()
        output=write_cycle_outputs(cycle,{"cycle":0,"model_in":"M0","model_out":"M1"},{"fixture":np.asarray([1.,2.,3.])})
        if len(output)!=20 or not all((cycle/"13_figures"/name).is_file() for name in CYCLE_FIGURES): raise RuntimeError("report smoke failure")
        marker=ROOT/"prelaunch"/"SMOKE_TEST_PASS.json"; marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps({"status":"PASS","formal_run_directory_touched":False,"config_sha256":sha256_file(config_path)},indent=2)+"\n",encoding="utf-8")
    print("SMOKE_TEST=PASS")


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--run-id",default=RUN_ID)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke-test",action="store_true")
    args=parser.parse_args()
    config_path=args.config.resolve(); config=load_and_validate(config_path)
    if args.run_id!=RUN_ID: raise ValueError(f"formal run id must be {RUN_ID}")
    if args.smoke_test:
        if args.resume: raise ValueError("smoke test cannot use --resume")
        smoke_test(config_path); return
    command=" ".join(sys.argv)
    print("BLADE_FINAL_20CYCLE_FORMAL_RUN_STARTING",flush=True)
    run(config,config_path,args.run_id,args.resume,command)
    print("BLADE_FINAL_20CYCLE_EXECUTION_COMPLETE",flush=True)


if __name__=="__main__": main()
