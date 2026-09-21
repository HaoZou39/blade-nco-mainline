"""Run pinned Concorde on the pre-sealed Cycle-0 audit union."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from oracle.concorde import resolve_backend, solve_batch
import hashlib
AUDIT=ROOT/"audits/cycle0_initialization_validation_v1"; label=AUDIT/"03_oracle_labels"
bundle=np.load(label/"selected_coordinates.npz"); hashes=bundle["instance_hash"].astype(str).tolist(); coords=bundle["coordinates"]
existing=pd.read_parquet(label/"concorde_ledger.parquet") if (label/"concorde_ledger.parquet").is_file() else None
backend=resolve_backend(ROOT/"third_party/concorde/concorde")
ledger=solve_batch(coords,hashes,label/"_concorde_work",backend,workers=8,existing=existing)
ledger.to_parquet(label/"concorde_ledger.parquet",index=False)
roles=[]
d5=json.loads((AUDIT/"02_gp0_calibration_design/cal_d5_manifest.json").read_text()); d5set={x['instance_hash'] for x in d5['members']}
devset=set(pd.read_parquet(AUDIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet").instance_hash.astype(str))
for h in hashes: roles.append("CAL_D5_AND_DEV" if h in d5set&devset else ("CAL_D5" if h in d5set else "CYCLE0_GP_DEV"))
tours=pd.DataFrame({"instance_hash":hashes,"oracle_role":roles}).merge(ledger,on="instance_hash",validate="one_to_one")
tours.to_parquet(label/"expert_tours.parquet",index=False)
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def rec(p): return {"path":str(p.resolve()),"bytes":p.stat().st_size,"sha256":file_sha(p)}
payload={"status":"EXPERT_TOURS_COMPLETE_HARDNESS_PENDING","rows":len(ledger),"deduplicated_union_n":len(hashes),
 "membership_sha256":__import__('hashlib').sha256(("\n".join(hashes)+"\n").encode()).hexdigest(),"concorde":backend,
 "M0_hardness_pending":True,"artifacts":{"coordinates":rec(label/'selected_coordinates.npz'),"concorde_ledger":rec(label/'concorde_ledger.parquet'),"expert_tours":rec(label/'expert_tours.parquet')}}
(label/"label_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("CONCORDE_LABELING_COMPLETE",flush=True)
