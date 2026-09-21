"""Compute exact M0 teacher-forced hardness in an isolated Torch process."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from bq.checkpoint import load_canonical_checkpoint
from bq.protocol import default_canonical_checkpoint
from bq.training_hardness import teacher_forced_training_hardness, protocol_manifest
from bq.utils import sha256_file

AUDIT=ROOT/"audits/cycle0_initialization_validation_v1"; LABEL=AUDIT/"03_oracle_labels"

def rec(path): return {"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":sha256_file(path)}

bundle=np.load(LABEL/"selected_coordinates.npz"); hashes=bundle["instance_hash"].astype(str).tolist(); coords=bundle["coordinates"]
tours=pd.read_parquet(LABEL/"expert_tours.parquet")
if tours.instance_hash.astype(str).tolist()!=hashes: raise RuntimeError("tour-coordinate order mismatch")
tour_array=np.asarray([[int(x) for x in value.split()] for value in tours.expert_tour],dtype=np.int64)
torch.set_num_threads(max(1,min(16,__import__('os').cpu_count() or 1))); device=torch.device("cpu")
model,meta=load_canonical_checkpoint(default_canonical_checkpoint(),device)
print(f"M0_TEACHER_FORCED_START={len(coords)}",flush=True)
hardness=teacher_forced_training_hardness(model,coords,tour_array,device,batch_size=128)
result=tours.copy()
for key,value in hardness.items(): result[key]=value
result["H0"]=result["loss_raw"]; result["H0_log1p"]=result["loss_log"]
out=LABEL/"exact_hardness.parquet"; result.to_parquet(out,index=False)
manifest=json.loads((LABEL/"label_manifest.json").read_text())
manifest.update({"status":"EXACT_LABELING_COMPLETE","M0":meta,"M0_hardness_pending":False,
                 "teacher_forced_protocol":protocol_manifest(),"parameter_updates":0})
manifest["artifacts"]["exact_hardness"]=rec(out)
(LABEL/"label_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("ORACLE_LABELING_COMPLETE",flush=True)
