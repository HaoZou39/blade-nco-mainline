"""Common configuration with the formal C00 coordinate bundle sealed as input."""
from audit_common_base import *

FORMAL_O0_COORDS=RUN/"cycles/C00/04_mutation/offspring_coordinates.npz"
FORMAL_O0_STAGE=RUN/"cycles/C00/04_mutation/STAGE_DONE.json"
INPUTS_EXTENDED=[*INPUTS,FORMAL_O0_COORDS,FORMAL_O0_STAGE]

def input_hashes():
    missing=[str(p) for p in INPUTS_EXTENDED if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    return {str(p):sha(p) for p in INPUTS_EXTENDED}
