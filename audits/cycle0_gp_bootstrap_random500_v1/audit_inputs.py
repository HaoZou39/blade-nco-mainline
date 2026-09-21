"""Input layer using the canonical formal C00 O0 coordinate bundle."""
import json
import numpy as np
from audit_common import FORMAL_O0_COORDS,FORMAL_O0_STAGE,O0_N
from audit_inputs_base import *

def replay_random_coordinates(rnd):
    stage=json.loads(FORMAL_O0_STAGE.read_text(encoding="utf-8"))
    expected=stage["outputs"][str(FORMAL_O0_COORDS)]["sha256"]
    if sha(FORMAL_O0_COORDS)!=expected or expected!="d4a2c80ff8be2116c6449da2601e3ab8ad226c5457f42b9a4e10982e3a2c36cb":
        raise RuntimeError("formal O0 coordinate bundle stage hash mismatch")
    bundle=np.load(FORMAL_O0_COORDS,allow_pickle=False)
    hashes=bundle["instance_hash"].astype(str)
    coordinates=bundle["coordinates"]
    if coordinates.shape!=(O0_N,100,2) or len(set(hashes))!=O0_N:
        raise RuntimeError("formal O0 coordinate bundle shape/uniqueness failure")
    index={value:i for i,value in enumerate(hashes)}
    wanted=rnd.instance_hash.astype(str).tolist()
    missing=[value for value in wanted if value not in index]
    if missing: raise RuntimeError(f"Random500 missing from formal O0 bundle: {len(missing)}")
    result=np.stack([coordinates[index[value]] for value in wanted]).astype(np.float64,copy=False)
    if result.shape!=(500,100,2): raise RuntimeError(str(result.shape))
    print("RANDOM500_CANONICAL_COORDINATES=500/500",flush=True)
    return result
