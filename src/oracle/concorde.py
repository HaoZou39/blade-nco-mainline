"""Pinned Concorde CLI wrapper for explicit integer-distance TSPs."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_CONCORDE_SHA256="38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0"
INTEGER_SCALE=1_000_000
ORACLE_LEDGER_COLUMNS=("instance_hash","status","verified","optimal_cost_int","runtime_sec",
                       "expert_tour","tour_hash","binary_hash")


def canonical_oracle_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only solver-result fields, excluding metadata accumulated by archives."""
    missing=[column for column in ORACLE_LEDGER_COLUMNS if column not in frame.columns]
    if missing: raise ValueError(f"Oracle ledger missing canonical columns: {missing}")
    result=frame.loc[:,ORACLE_LEDGER_COLUMNS].copy()
    if result.instance_hash.astype(str).duplicated().any(): raise ValueError("Oracle ledger has duplicate instance hashes")
    return result

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def integer_distance_matrix(coords: np.ndarray, scale: int=INTEGER_SCALE) -> np.ndarray:
    delta=np.asarray(coords,float)[:,None,:]-np.asarray(coords,float)[None,:,:]
    result=np.rint(np.sqrt(np.sum(delta*delta,axis=-1))*scale).astype(np.int64); np.fill_diagonal(result,0); return result

def validate_tour_0based(tour: list[int], n: int) -> tuple[bool,str]:
    return (True,"") if len(tour)==n and sorted(tour)==list(range(n)) else (False,"tour is not a Hamiltonian permutation")

def normalize_tour_0based(tour: list[int], n: int) -> list[int]:
    out=list(map(int,tour))
    if len(out)==n+1 and out[0]==out[-1]: out=out[:-1]
    if len(out)==n and min(out)>=1 and max(out)==n: out=[x-1 for x in out]
    ok,msg=validate_tour_0based(out,n)
    if not ok: raise ValueError(msg)
    return out

def tour_cost_int(dist: np.ndarray,tour: list[int]) -> int:
    return int(sum(int(dist[tour[i],tour[(i+1)%len(tour)]]) for i in range(len(tour))))

def write_explicit_tsp(path: Path, dist: np.ndarray, name: str) -> None:
    lines=[f"NAME: {name}","TYPE: TSP",f"DIMENSION: {len(dist)}","EDGE_WEIGHT_TYPE: EXPLICIT","EDGE_WEIGHT_FORMAT: FULL_MATRIX","EDGE_WEIGHT_SECTION"]
    lines.extend(" ".join(map(str,row)) for row in dist); lines.append("EOF"); path.write_text("\n".join(lines)+"\n",encoding="ascii")


def tour_hash(tour: list[int]) -> str:
    return hashlib.sha256(np.asarray(tour, dtype=np.int32).tobytes()).hexdigest()


def windows_to_wsl_path(path: Path) -> str:
    value = str(path.resolve())
    return f"/mnt/{value[0].lower()}{value[2:].replace(chr(92), '/')}" if len(value) > 1 and value[1] == ":" else value.replace("\\", "/")


def resolve_backend(executable: Path | None = None, wsl_distro: str = "Ubuntu") -> dict[str, Any]:
    path = Path(executable).resolve() if executable is not None else (Path(__file__).resolve().parents[2]/"third_party/concorde/concorde").resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != CANONICAL_CONCORDE_SHA256: raise RuntimeError(f"Concorde SHA mismatch: {digest}")
    is_elf = path.read_bytes()[:4] == b"\x7fELF"
    return {"backend":"concorde_cli", "binary":str(path), "binary_hash":digest,
            "invoke_mode":"wsl" if is_elf and shutil.which("wsl") else "native", "wsl_distro":wsl_distro}


def _parse_tour(path: Path, n: int) -> list[int]:
    values=[int(x) for x in re.findall(r"-?\d+",path.read_text(encoding="utf-8",errors="replace"))]
    if values and values[0] == n and len(values) in (n+1,n+2): values=values[1:]
    if len(values)==n+1 and values[0]==values[-1]: values=values[:-1]
    return normalize_tour_0based(values,n)


def _reported(text: str) -> int | None:
    for pattern in (r"Optimal Solution:\s*([0-9]+)",r"Optimal\s+([0-9]+)",r"Value\s*=\s*([0-9]+)"):
        match=re.search(pattern,text,re.I)
        if match: return int(match.group(1))
    return None


def solve_one(coords: np.ndarray, instance_hash: str, work: Path, backend: dict[str,Any], timeout: float=300.0) -> dict[str,Any]:
    work.mkdir(parents=True,exist_ok=True); dist=integer_distance_matrix(coords,INTEGER_SCALE); n=len(coords)
    tsp=work/f"{instance_hash}.tsp"; sol=work/f"{instance_hash}.sol"; write_explicit_tsp(tsp,dist,instance_hash)
    if backend["invoke_mode"]=="wsl":
        inner=f"cd '{windows_to_wsl_path(work)}' && '{windows_to_wsl_path(Path(backend['binary']))}' -x -o '{sol.name}' '{tsp.name}'"
        cmd=["wsl","-d",backend["wsl_distro"],"--","bash","-lc",inner]; cwd=None
    else: cmd=[backend["binary"],"-x","-o",sol.name,tsp.name]; cwd=str(work)
    start=time.perf_counter(); proc=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
    elapsed=time.perf_counter()-start; text=(proc.stdout or "")+"\n"+(proc.stderr or "")
    if not sol.is_file(): raise RuntimeError(f"Concorde {instance_hash} exit={proc.returncode}: {text[-500:]}")
    tour=_parse_tour(sol,n); legal,msg=validate_tour_0based(tour,n); cost=tour_cost_int(dist,tour); stated=_reported(text)
    verified=legal and (stated is None or stated==cost)
    if not verified: raise RuntimeError(f"Concorde verification failed {instance_hash}: {msg}, stated={stated}, cost={cost}")
    return {"instance_hash":instance_hash,"status":"solved","verified":True,"optimal_cost_int":cost,"runtime_sec":elapsed,
            "expert_tour":" ".join(map(str,tour)),"tour_hash":tour_hash(tour),"binary_hash":backend["binary_hash"]}


def solve_batch(coords: np.ndarray, hashes: list[str], work: Path, backend: dict[str,Any], workers: int=8, existing: pd.DataFrame|None=None) -> pd.DataFrame:
    done={}
    if existing is not None and len(existing):
        canonical=canonical_oracle_ledger(existing)
        done={str(r.instance_hash):r._asdict() for r in canonical.itertuples(index=False) if bool(r.verified) and r.status=="solved"}
    pending=[(i,h) for i,h in enumerate(hashes) if h not in done]
    with ThreadPoolExecutor(max_workers=max(1,min(workers,len(pending)))) as pool:
        jobs={pool.submit(solve_one,coords[i],h,work/h,backend):h for i,h in pending}
        for count,future in enumerate(as_completed(jobs),1):
            row=future.result(); done[row["instance_hash"]]=row
            if count%100==0: print(f"CONCORDE_PROGRESS={len(done)}/{len(hashes)}",flush=True)
    return canonical_oracle_ledger(pd.DataFrame([done[h] for h in hashes]))
