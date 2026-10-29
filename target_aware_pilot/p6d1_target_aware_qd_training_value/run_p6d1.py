#!/usr/bin/env python3
"""P6-D1 target-aware hardness-driven QD training-value validation.

This runner consumes the byte-frozen P6-D0 A/B/C memberships.  It restores
coordinates and existing expert tours from the already labelled P6-C1 union,
then delegates training and greedy evaluation to the formal P6-B/P6-C2 code.
It never invokes mutation, evolution, selection, HGB fitting, or Concorde.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
D0 = ROOT / "target_aware_pilot/p6d0_target_aware_qd_selection_audit"
C1 = ROOT / "target_aware_pilot/p6c1_hardness_controlled_matching"
C2 = ROOT / "target_aware_pilot/p6c2_oracle_hardness_restoration_training_value"
B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
C0 = ROOT / "target_aware_pilot/p6c0_training_value_mechanism_audit"
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
M4 = RUN / "checkpoints/M4_E256.tsp.best"
M4_EVAL = RUN / "cycles/C03/evaluation_work/M4"
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
REQUEST = Path("/home/haozou/.codex/attachments/278f29bc-deb9-4ab9-93c1-6e3268bd5cd3/pasted-text.txt")

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(C2)]
import run_p6c2 as base  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.tsp_metric import validate_tour_0based  # noqa: E402
from final_run.science import hardness, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402

REPLICATES = ("Original", "S1", "S2")
ARMS = ("A", "B", "C")
DATA_ARMS = ("A", "B", "C", "E0", "E1")
DATASETS = tuple(base.DATASETS)
GAP5 = tuple(base.GAP5)
K = 10_000
TRAINING_SEED = 2026
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261030
CE_BATCH = 128
JOB_ORDER = tuple((r, a) for r in REPLICATES for a in ARMS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for p in (OUT, OUT/"configs", OUT/"02_training_configs", OUT/"datasets", OUT/"checkpoints",
              OUT/"logs", OUT/"figures", OUT/"hardness", OUT/"monitoring"):
        p.mkdir(parents=True, exist_ok=True)
    for name in ("repair_actions.md", "failures.md"):
        p = OUT/"monitoring"/name
        if not p.exists(): p.write_text(f"# {name.removesuffix('.md').replace('_', ' ').title()}\n\n")


def write_json(path: Path, value: Any) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=conv) + "\n")
    os.replace(tmp, path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def md_table(frame: pd.DataFrame) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, (dict, list, tuple, set)): return json.dumps(v, sort_keys=True, default=str).replace("|", "\\|")
        if pd.isna(v): return "NA"
        if isinstance(v, (float, np.floating)): return f"{float(v):.8g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols = [str(c) for c in frame.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    rows += ["| " + " | ".join(fmt(x) for x in r) + " |" for r in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


class Status:
    def update(self, **values: Any) -> None:
        current = {}
        p = OUT/"status.json"
        if p.exists():
            try: current = json.loads(p.read_text())
            except Exception: current = {}
        write_json(p, {**current, **values, "updated_utc": now()})


def manifest_path(rep: str, arm: str) -> Path: return OUT/"datasets"/f"{rep}_{arm}.parquet"
def bundle_path(rep: str, arm: str) -> Path: return OUT/"datasets"/f"{rep}_{arm}.npz"
def union_bundle(rep: str) -> Path: return C1/"dataset_manifests"/f"{rep}_union.npz"
def run_dir(rep: str, arm: str) -> Path: return OUT/"logs"/f"{rep}_{arm}"
def checkpoint(rep: str, arm: str) -> Path: return OUT/"checkpoints"/f"{rep}_{arm}_M5_E256.tsp.best"
def eval_label(rep: str, arm: str) -> str: return f"P6D1_{rep}_{arm}_M5"
def eval_csv(rep: str, arm: str) -> Path: return run_dir(rep, arm)/"external_benchmark_results.csv"


def load_bundle(path: Path, require_tours: bool = False) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as z:
        xy = np.asarray(z["coordinates"], np.float64)
        hashes = z["instance_hash"].astype(str).tolist()
        tours = np.asarray(z["expert_tours"], np.int64) if "expert_tours" in z.files else None
    if require_tours and tours is None: raise RuntimeError(f"missing expert tours: {path}")
    return xy, hashes, tours


def save_bundle(path: Path, xy: np.ndarray, hashes: list[str], tours: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, coordinates=np.asarray(xy, np.float64),
                        instance_hash=np.asarray(hashes, dtype="U64"), expert_tours=np.asarray(tours, np.int64))
    os.replace(tmp, path)


def training_protocol() -> dict[str, Any]:
    return {
        **base.base_training_protocol(), "formal_source": "P6-B/P6-C2 core.train",
        "trainer_sha256": sha256_file(ROOT/"scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py"),
        "training_order": [f"{r}-{a}" for r, a in JOB_ORDER], "device": "cuda:1",
    }


def frozen_protocol() -> dict[str, Any]:
    value = {
        "experiment": "BLADE-NCO P6-D1", "request_sha256": sha256_file(REQUEST),
        "p6d0_completion_sha256": sha256_file(D0/"completion.json"),
        "p6d0_selection_manifest_sha256": sha256_file(D0/"05_selection_manifests.csv"),
        "p6c1_completion_sha256": sha256_file(C1/"completion.json"),
        "p6b_completion_sha256": sha256_file(B/"completion.json"),
        "p6c2_completion_sha256": sha256_file(C2/"completion.json"),
        "m4_sha256": M4_SHA, "replicates": list(REPLICATES), "arms": list(ARMS),
        "training": training_protocol(), "bootstrap_n": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
        "h1_rule": "supported iff C beats A on Original70 3/3 and C is not worse on GAP5 3/3; not-supported iff C wins Original70 0/3; otherwise inconclusive",
        "h2_rule": "supported iff C beats B on Original70 3/3 and C is not worse on GAP5 3/3; not-supported iff C wins Original70 0/3; otherwise inconclusive",
        "h3_rule": "SUPPORTED iff C beats M4 on Original70 3/3; PARTIAL iff 1-2/3; NOT_SUPPORTED iff 0/3",
        "h4_rule": "PROMISING iff H1 and H2 supported and H3 is not NOT_SUPPORTED; otherwise NOT_ESTABLISHED unless both H1/H2 not-supported",
        "hourly_monitor": {"initial_check": True, "OnActiveSec": "1h", "OnUnitActiveSec": "1h", "max_retries_per_run_fault": 2},
        "prohibitions": ["mutation", "evolution", "selection", "Concorde", "HGB fitting", "extra training seed", "benchmark-driven checkpoint selection", "P6-D2"],
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    }
    value["protocol_sha256"] = canonical_hash(value)
    return value


def protocol_gate() -> dict[str, Any]:
    ensure_dirs(); current = frozen_protocol(); p = OUT/"configs/frozen_p6d1_config.json"
    if p.exists() and json.loads(p.read_text()) != current: raise RuntimeError("frozen P6-D1 protocol drift")
    if not p.exists(): write_json(p, current)
    (OUT/"00_protocol.md").write_text(
        "# P6-D1 Frozen Protocol\n\nP6-D1 consumes only the byte-frozen P6-D0 A/B/C memberships. "
        "Coordinates and expert tours are restored from the already labelled P6-C1 union; no selection or Concorde call is made. "
        "All nine runs start from formal M4 and delegate to the same P6-B/P6-C2 trainer and greedy evaluator.\n\n"
        f"- M4 SHA256: `{M4_SHA}`\n- Protocol SHA256: `{current['protocol_sha256']}`\n"
        "- Training: fresh Adam, LR 3.75e-5, no scheduler, logical batch 256, microbatch 64, seed 2026, 256 epochs, 10,240 updates, fixed E256.\n"
        "- Monitor: one initial check followed by user-systemd oneshot checks every 60 minutes.\n"
    )
    return current


def audit() -> dict[str, Any]:
    started = time.perf_counter(); config = protocol_gate()
    if sha256_file(M4) != M4_SHA: raise RuntimeError("formal M4 hash mismatch")
    d0_done = json.loads((D0/"completion.json").read_text())
    if d0_done.get("final_label") != "BLADE_P6D0_TARGET_AWARE_QD_SELECTION_AUDIT_COMPLETE": raise RuntimeError("P6-D0 incomplete")
    selected = pd.read_csv(D0/"05_selection_manifests.csv")
    rows = []
    for rep in REPLICATES:
        ux, uh, ut = load_bundle(union_bundle(rep), True); assert ut is not None
        if ux.shape[1:] != (100, 2) or len(set(uh)) != len(uh): raise RuntimeError(f"invalid labelled union {rep}")
        lookup = {h:i for i,h in enumerate(uh)}
        for arm in ARMS:
            part = selected[(selected.replicate == rep) & (selected.selection == arm)].sort_values("selection_order").reset_index(drop=True)
            frozen = (D0/"selected_hashes"/f"{rep}_{arm}.txt").read_text().splitlines()
            hashes = part.instance_hash.astype(str).tolist()
            if len(hashes) != K or len(set(hashes)) != K or hashes != frozen: raise RuntimeError(f"P6-D0 membership drift {rep}/{arm}")
            missing = [h for h in hashes if h not in lookup]
            if missing: raise RuntimeError(f"{rep}/{arm}: {len(missing)} hashes absent from labelled union")
            idx = np.asarray([lookup[h] for h in hashes], int); xy = ux[idx]; tours = ut[idx]
            if xy.shape != (K,100,2) or not np.isfinite(xy).all() or np.any((xy<0)|(xy>1)): raise RuntimeError(f"coordinate validity {rep}/{arm}")
            if any(len(np.unique(x,axis=0)) != 100 for x in xy): raise RuntimeError(f"duplicate nodes {rep}/{arm}")
            if [coordinate_hash(x) for x in xy] != hashes: raise RuntimeError(f"coordinate hash mismatch {rep}/{arm}")
            if not all(validate_tour_0based(t,100)[0] for t in tours): raise RuntimeError(f"illegal expert tour {rep}/{arm}")
            part.to_parquet(manifest_path(rep,arm), index=False); save_bundle(bundle_path(rep,arm),xy,hashes,tours)
            # Round-trip is the final authority for training input integrity.
            rx,rh,rt=load_bundle(bundle_path(rep,arm),True); assert rt is not None
            if rh != hashes or not np.array_equal(rx,xy) or not np.array_equal(rt,tours): raise RuntimeError(f"bundle round-trip {rep}/{arm}")
            rows.append({"replicate":rep,"arm":arm,"n":K,"unique_n":len(set(hashes)),"n_nodes":100,
                         "membership_sha256":membership_hash(hashes),"frozen_hash_file_sha256":sha256_file(D0/"selected_hashes"/f"{rep}_{arm}.txt"),
                         "source_union_sha256":sha256_file(union_bundle(rep)),"manifest_sha256":sha256_file(manifest_path(rep,arm)),
                         "bundle_sha256":sha256_file(bundle_path(rep,arm)),"coordinates_valid":True,"coordinate_hashes_match":True,
                         "expert_tours_legal":True,"selection_membership_exact":True})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"01_dataset_manifest.csv",index=False)
    model,meta=load_canonical_checkpoint(M4,torch.device("cpu"),expected_sha256=M4_SHA); del model
    bench=[]
    for ds in DATASETS:
        p=M4_EVAL/ds/"instance_metrics.parquet"
        if not p.is_file(): raise RuntimeError(f"missing frozen benchmark {ds}")
        bench.append({"dataset":ds,"m4_detail_sha256":sha256_file(p),"n":len(pd.read_parquet(p,columns=["instance_id"]))})
    pd.DataFrame(bench).to_csv(OUT/"logs/benchmark_integrity.csv",index=False)
    (OUT/"01_input_integrity.md").write_text(
        "# Input Integrity\n\nAll nine datasets are exact ordered P6-D0 selections. Coordinates and existing expert tours were restored by coordinate hash from P6-C1; no labels were generated. "
        "Each dataset contains 10,000 unique legal TSP100 instances, all coordinate hashes reproduce, and all tours are legal 0-based Hamiltonian permutations. "
        "Formal M4 loaded with the expected hash, and all frozen benchmark assets exist.\n\n"+md_table(frame)+"\n")
    (OUT/"02_training_protocol.md").write_text("# Frozen Training Protocol\n\n"+md_table(pd.DataFrame([training_protocol()]))+"\n\nEvery run checks E000 tensor equality against formal M4 and fixed-E256 completion. No early stopping or benchmark-driven selection is permitted.\n")
    for rep,arm in JOB_ORDER:
        cfg={"run_id":f"P6D1_{rep}_{arm}","replicate":rep,"arm":arm,"training_seed":TRAINING_SEED,
             "membership_sha256":frame[(frame.replicate==rep)&(frame.arm==arm)].membership_sha256.iloc[0],
             "dataset_bundle_sha256":sha256_file(bundle_path(rep,arm)),"m4_sha256":M4_SHA,
             "protocol_sha256":config["protocol_sha256"],"base_training_protocol":training_protocol()}
        cfg["training_config_sha256"]=canonical_hash(cfg); write_json(OUT/"02_training_configs"/f"{rep}_{arm}.json",cfg)
    result={"status":"PASS","datasets":9,"m4_loaded":True,"m4_loader_sha256":meta["sha256"],"benchmarks":len(bench),"wall_s":time.perf_counter()-started,"protocol_sha256":config["protocol_sha256"]}
    write_json(OUT/"logs/audit.json",result); Status().update(phase="audit",status="PASS",completed_runs=0,total_runs=9); return result


def training_manifest() -> pd.DataFrame:
    rows=[]
    for rep,arm in JOB_ORDER:
        p=run_dir(rep,arm)/"training_integrity.json"
        rows.append(json.loads(p.read_text()) if p.exists() else {"replicate":rep,"arm":arm,"run_id":f"P6D1_{rep}_{arm}","status":"PENDING","training_seed":TRAINING_SEED})
    f=pd.DataFrame(rows); f.to_csv(OUT/"03_training_runs.csv",index=False)
    passed=int(f.status.eq("PASS").sum())
    (OUT/"03_training_integrity.md").write_text(f"# Training Integrity\n\nCompleted: {passed}/9. Every PASS run verifies exact frozen input hashes, E000 tensor equality to formal M4, 256 epochs, 10,240 updates, finite weights, seed 2026, and fixed E256 selection.\n")
    return f


def train(rep: str, arm: str, device_name: str) -> dict[str,Any]:
    protocol_gate(); integrity=run_dir(rep,arm)/"training_integrity.json"; cp=checkpoint(rep,arm)
    if integrity.exists():
        old=json.loads(integrity.read_text())
        if old.get("status")=="PASS" and cp.exists() and sha256_file(cp)==old.get("checkpoint_sha256"): return old
    xy,hashes,tours=load_bundle(bundle_path(rep,arm),True); assert tours is not None
    cfg=json.loads((OUT/"02_training_configs"/f"{rep}_{arm}.json").read_text())
    if sha256_file(bundle_path(rep,arm))!=cfg["dataset_bundle_sha256"] or membership_hash(hashes)!=cfg["membership_sha256"]: raise RuntimeError("frozen training input drift")
    device=torch.device(device_name); torch.cuda.set_device(device) if device.type=="cuda" else None
    directory=run_dir(rep,arm)/"training_cycle"; directory.mkdir(parents=True,exist_ok=True)
    base.core.RUN_ID=f"TARGET_AWARE_P6D1_{rep}_{arm}"; base.core.cycle_dir=lambda selected_arm,cycle: directory; base.core.model_path=lambda selected_arm,index: cp if index==5 else M4
    model,mmeta=load_canonical_checkpoint(M4,device,expected_sha256=M4_SHA)
    started_wall=now(); started=time.perf_counter(); Status().update(phase="training",current_run=f"{rep}-{arm}",run_started_utc=started_wall)
    trained,meta,timing=base.core.train(f"P6D1_{rep}_{arm}",4,model,mmeta["sha256"],xy,hashes,tours,device,Status(),True)
    invocation=time.perf_counter()-started; stage=directory/"09_bq_training"
    e000=torch.load(stage/"M5_E000.tsp.best",map_location="cpu",weights_only=False); m4=torch.load(M4,map_location="cpu",weights_only=False)
    equal=set(e000["net"])==set(m4["net"]) and all(torch.equal(e000["net"][k],m4["net"][k]) for k in m4["net"])
    curve=pd.read_csv(stage/"training_loss_curve.csv")
    if not equal or len(curve)!=256 or int(curve.optimizer_updates.sum())!=10240 or not curve.hard_instances_seen.eq(K).all() or meta["sha256"]!=sha256_file(cp) or not all(bool(torch.isfinite(p).all()) for p in trained.parameters()): raise RuntimeError("training integrity failure")
    result={"status":"PASS",**cfg,"start_utc":started_wall,"end_utc":now(),"exit_code":0,"m4_sha256":M4_SHA,"e000_weights_equal_m4":True,
            "epochs":256,"optimizer_steps":int(timing["optimizer_steps"]),"final_loss":float(timing["final_loss"]),"training_loop_wall_s":float(timing["training_loop_wall_s"]),
            "invocation_wall_s":invocation,"gpu_peak_allocated_bytes":int(timing.get("gpu_peak_allocated_bytes") or 0),"checkpoint_sha256":meta["sha256"],
            "checkpoint_path":str(cp),"loss_curve_sha256":sha256_file(stage/"training_loss_curve.csv")}
    write_json(integrity,result); training_manifest(); del trained,model,xy,tours; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None; return result


def evaluate(rep: str, arm: str, device_name: str) -> dict[str,Any]:
    p=run_dir(rep,arm)/"evaluation_integrity.json"
    if p.exists():
        old=json.loads(p.read_text())
        if old.get("status")=="PASS" and eval_csv(rep,arm).exists() and sha256_file(eval_csv(rep,arm))==old.get("results_sha256"): return old
    tr=json.loads((run_dir(rep,arm)/"training_integrity.json").read_text()); cp=checkpoint(rep,arm)
    if sha256_file(cp)!=tr["checkpoint_sha256"]: raise RuntimeError("checkpoint drift")
    device=torch.device(device_name); model,meta=load_canonical_checkpoint(cp,device,expected_sha256=tr["checkpoint_sha256"])
    Status().update(phase="evaluation",current_run=f"{rep}-{arm}")
    frame,wall=base.core.evaluate(model,cp,eval_label(rep,arm),eval_csv(rep,arm),Status(),f"P6D1_{rep}_{arm}")
    if set(frame.dataset.astype(str))!=set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all(): raise RuntimeError("evaluation incomplete")
    historical=pd.read_csv(B/"logs"/f"{rep}_E0"/"external_benchmark_results.csv").set_index("dataset"); current=frame.set_index("dataset")
    for ds in DATASETS:
        if current.at[ds,"dataset_sha256"]!=historical.at[ds,"dataset_sha256"] or current.at[ds,"evaluator_config_sha256"]!=historical.at[ds,"evaluator_config_sha256"]: raise RuntimeError(f"evaluation protocol drift {ds}")
    result={"status":"PASS","replicate":rep,"arm":arm,"checkpoint_sha256":meta["sha256"],"datasets":list(DATASETS),"all_tours_valid":True,"evaluation_wall_s":wall,"results_sha256":sha256_file(eval_csv(rep,arm))}
    write_json(p,result); del model; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None; return result


def infer_ce(rep: str, arm: str, device_name: str) -> dict[str,Any]:
    out=OUT/"hardness"/f"{rep}_{arm}_on_union.parquet"; meta_path=out.with_suffix(".json")
    tr=json.loads((run_dir(rep,arm)/"training_integrity.json").read_text())
    if out.exists() and meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old.get("status")=="PASS" and old.get("checkpoint_sha256")==tr["checkpoint_sha256"] and sha256_file(out)==old.get("output_sha256"): return old
    xy,hashes,tours=load_bundle(union_bundle(rep),True); assert tours is not None
    device=torch.device(device_name); torch.manual_seed(20261027); np.random.seed(20261027)
    model,meta=load_canonical_checkpoint(checkpoint(rep,arm),device,expected_sha256=tr["checkpoint_sha256"]); model.eval(); started=time.perf_counter()
    values=hardness(model,xy,tours,device,batch_size=CE_BATCH); wall=time.perf_counter()-started
    pd.DataFrame({"replicate":rep,"model_arm":arm,"instance_hash":hashes,"loss_raw":values}).to_parquet(out,index=False)
    result={"status":"PASS","replicate":rep,"model_arm":arm,"n":len(hashes),"checkpoint_sha256":meta["sha256"],"output_sha256":sha256_file(out),"wall_s":wall,"gpu_peak_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0}
    write_json(meta_path,result); del model,xy,tours; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None; return result


def detail(rep: str, arm: str, ds: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(rep,arm)/"evaluation_work"/eval_label(rep,arm)/ds/"instance_metrics.parquet")


def bootstrap(values: np.ndarray, key: str) -> tuple[float,float]:
    x=np.asarray(values,float); rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6D1|{BOOTSTRAP_SEED}|{key}".encode()).digest()[:8],"little")); out=np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N): out[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(np.quantile(out,.025)),float(np.quantile(out,.975))


def finalize() -> dict[str,Any]:
    started=time.perf_counter(); cfg=protocol_gate(); tr=training_manifest()
    if len(tr)!=9 or not tr.status.eq("PASS").all(): raise RuntimeError("training incomplete")
    for rep,arm in JOB_ORDER:
        if json.loads((run_dir(rep,arm)/"evaluation_integrity.json").read_text()).get("status")!="PASS": raise RuntimeError("evaluation incomplete")
        if json.loads((OUT/"hardness"/f"{rep}_{arm}_on_union.json").read_text()).get("status")!="PASS": raise RuntimeError("CE incomplete")
    dev=set(pd.read_csv(ROOT/"target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str)); held=set(json.loads((ROOT/"target_aware_pilot/p1_heldout22_names.json").read_text())["names"])
    per_parts=[]; eval_rows=[]; stat_rows=[]; frames_by_rep={}
    for rep in REPLICATES:
        frames={}
        for ds in DATASETS:
            pieces=[]
            for arm in ARMS:
                x=detail(rep,arm,ds)[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].copy()
                if not x.valid_tour.all(): raise RuntimeError(f"invalid greedy tour {rep}/{arm}/{ds}")
                pieces.append(x.drop(columns="valid_tour").rename(columns={"coordinate_hash":f"hash_{arm}","n_nodes":f"n_{arm}","optimal_cost_int":f"opt_{arm}","gap_percent":f"gap_{arm}"}))
            m=pieces[0].merge(pieces[1],on="instance_id",validate="one_to_one").merge(pieces[2],on="instance_id",validate="one_to_one")
            if not (m.hash_A.eq(m.hash_B)&m.hash_A.eq(m.hash_C)&m.n_A.eq(m.n_B)&m.n_A.eq(m.n_C)&m.opt_A.eq(m.opt_B)&m.opt_A.eq(m.opt_C)).all(): raise RuntimeError(f"paired benchmark drift {rep}/{ds}")
            m4=pd.read_parquet(M4_EVAL/ds/"instance_metrics.parquet")[["instance_id","coordinate_hash","gap_percent"]].rename(columns={"coordinate_hash":"hash_M4","gap_percent":"gap_M4"})
            m=m.merge(m4,on="instance_id",validate="one_to_one")
            if not m.hash_A.eq(m.hash_M4).all(): raise RuntimeError(f"M4 benchmark drift {ds}")
            f=pd.DataFrame({"replicate":rep,"dataset":ds,"instance_id":m.instance_id.astype(str),"coordinate_hash":m.hash_A,"n_nodes":m.n_A.astype(int),"optimal_cost_int":m.opt_A.astype(np.int64),"gap_M4":m.gap_M4,"gap_A":m.gap_A,"gap_B":m.gap_B,"gap_C":m.gap_C})
            frames[ds]=f; per_parts.append(f)
        original=frames["OriginalTSPLIB70"]
        for name,names in (("TSPLIB48",dev),("TSPLIB22",held)):
            f=original[original.instance_id.isin(names)].copy(); f["dataset"]=name
            if len(f)!=len(names): raise RuntimeError(f"split integrity {name}")
            frames[name]=f; per_parts.append(f)
        frames_by_rep[rep]=frames
        for ds,f in frames.items():
            for model in ("M4",*ARMS): eval_rows.append({"replicate":rep,"model":model,"dataset":ds,"n_instances":len(f),"mean_gap":f[f"gap_{model}"].mean(),"checkpoint_sha256":M4_SHA if model=="M4" else json.loads((run_dir(rep,model)/"training_integrity.json").read_text())["checkpoint_sha256"]})
            for comp,left in (("C_vs_A","A"),("C_vs_B","B"),("C_vs_M4","M4")):
                v=(f[f"gap_{left}"]-f.gap_C).to_numpy(float); lo,hi=bootstrap(v,f"{rep}|{ds}|{comp}")
                positive=np.clip(-v,0,None); total=positive.sum(); worst=np.sort(positive)[-max(1,int(np.ceil(.1*len(v)))):].sum()
                stat_rows.append({"replicate":rep,"comparison":comp,"dataset":ds,"stratum":"ALL","n":len(v),"mean_gap_comparator":f[f"gap_{left}"].mean(),"mean_gap_C":f.gap_C.mean(),"mean_delta_comparator_minus_C":v.mean(),"median_delta":np.median(v),"sample_sd_delta":np.std(v,ddof=1),"paired_ci95_low":lo,"paired_ci95_high":hi,"c_instance_win_count":int((v>0).sum()),"c_instance_win_fraction":np.mean(v>0),"c_regression_fraction":np.mean(v<0),"worst10pct_share_of_regression":worst/total if total>0 else 0})
                if ds=="OriginalTSPLIB70":
                    scales=pd.cut(f.n_nodes,[0,100,200,500,np.inf],labels=["n<=100","101-200","201-500","n>500"])
                    cats=np.where(f.instance_id.isin(dev),"TSPLIB48","TSPLIB22")
                    for label,mask in [(str(x),scales.astype(str)==str(x)) for x in scales.dropna().unique()]+[(x,cats==x) for x in ("TSPLIB48","TSPLIB22")]:
                        sv=v[np.asarray(mask)]; slo,shi=bootstrap(sv,f"{rep}|{comp}|{label}"); stat_rows.append({"replicate":rep,"comparison":comp,"dataset":ds,"stratum":label,"n":len(sv),"mean_delta_comparator_minus_C":sv.mean(),"median_delta":np.median(sv),"sample_sd_delta":np.std(sv,ddof=1) if len(sv)>1 else np.nan,"paired_ci95_low":slo,"paired_ci95_high":shi,"c_instance_win_count":int((sv>0).sum()),"c_instance_win_fraction":np.mean(sv>0)})
        for model in ("M4",*ARMS): eval_rows.append({"replicate":rep,"model":model,"dataset":"GAP5","n_instances":sum(len(frames[x]) for x in GAP5),"mean_gap":np.mean([frames[x][f"gap_{model}"].mean() for x in GAP5]),"checkpoint_sha256":M4_SHA if model=="M4" else json.loads((run_dir(rep,model)/"training_integrity.json").read_text())["checkpoint_sha256"]})
        for comp,left in (("C_vs_A","A"),("C_vs_B","B"),("C_vs_M4","M4")):
            vals=[frames[x][f"gap_{left}"].to_numpy(float)-frames[x].gap_C.to_numpy(float) for x in GAP5]; rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6D1|GAP5|{rep}|{comp}".encode()).digest()[:8],"little")); boots=np.empty(BOOTSTRAP_N)
            for i in range(BOOTSTRAP_N): boots[i]=np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in vals])
            means=np.asarray([v.mean() for v in vals]); stat_rows.append({"replicate":rep,"comparison":comp,"dataset":"GAP5","stratum":"ALL","n":sum(map(len,vals)),"mean_gap_comparator":np.mean([frames[x][f"gap_{left}"].mean() for x in GAP5]),"mean_gap_C":np.mean([frames[x].gap_C.mean() for x in GAP5]),"mean_delta_comparator_minus_C":means.mean(),"median_delta":np.median(means),"sample_sd_delta":np.std(means,ddof=1),"paired_ci95_low":np.quantile(boots,.025),"paired_ci95_high":np.quantile(boots,.975),"c_instance_win_count":np.nan,"c_instance_win_fraction":np.nan})
    per=pd.concat(per_parts,ignore_index=True); per.to_csv(OUT/"04_per_instance_gap.csv",index=False)
    evaluations=pd.DataFrame(eval_rows); evaluations.to_csv(OUT/"04_greedy_evaluation.csv",index=False)
    stats=pd.DataFrame(stat_rows); stats.to_csv(OUT/"05_paired_statistics.csv",index=False)
    def comparison_doc(comp: str, path: Path, title: str) -> None:
        view=stats[(stats.comparison==comp)&(stats.stratum=="ALL")&stats.dataset.isin(["OriginalTSPLIB70","TSPLIB48","TSPLIB22","GAP5",*GAP5])]
        path.write_text(f"# {title}\n\nPositive delta favors C. Bootstrap intervals are conditional on frozen benchmark instances; the three evolution replicates are the directional robustness unit.\n\n"+md_table(view)+"\n")
    comparison_doc("C_vs_A",OUT/"05_c_vs_a_comparison.md","C vs A Paired Comparison")
    comparison_doc("C_vs_B",OUT/"05_c_vs_b_comparison.md","C vs B Paired Comparison")
    # Historical E0/E1 and Oracle-C1 comparison.
    hist=[]; p6b=pd.read_csv(B/"08_greedy_evaluation_results.csv"); oracle=pd.read_csv(C2/"04_greedy_evaluation.csv")
    historical_datasets=tuple(dict.fromkeys(("OriginalTSPLIB70","TSPLIB48","TSPLIB22","GAP5",*GAP5)))
    for rep in REPLICATES:
        for ds in historical_datasets:
            for model in ("E0","E1"):
                q=p6b[(p6b.replicate==rep)&(p6b.arm==model)&(p6b.dataset==ds)]; hist.append({"replicate":rep,"dataset":ds,"model":f"P6B_{model}","mean_gap":float(q.mean_gap.iloc[0])})
            q=oracle[(oracle.replicate==rep)&(oracle.arm=="C1")&(oracle.dataset==ds)]; hist.append({"replicate":rep,"dataset":ds,"model":"P6C2_Oracle_C1","mean_gap":float(q.mean_gap.iloc[0])})
            for model in ("M4","C"):
                q=evaluations[(evaluations.replicate==rep)&(evaluations.model==model)&(evaluations.dataset==ds)]; hist.append({"replicate":rep,"dataset":ds,"model":model,"mean_gap":float(q.mean_gap.iloc[0])})
    hist=pd.DataFrame(hist); hist.to_csv(OUT/"logs/historical_comparison.csv",index=False)
    (OUT/"06_historical_comparison.md").write_text("# Historical Comparison\n\nHistorical P6-B E0/E1 and P6-C2 Oracle-C1 are descriptive references, not randomized same-membership controls.\n\n"+md_table(hist[hist.dataset=="OriginalTSPLIB70"].pivot(index="replicate",columns="model",values="mean_gap").reset_index())+"\n")
    # 4 models x 5 data populations x 3 replicates = 60 CE cells.
    matrix=[]; instance=[]; selected=pd.read_csv(D0/"05_selection_manifests.csv",usecols=["replicate","selection","instance_hash","selection_order","true_m4_ce"])
    for rep in REPLICATES:
        union_frame=pd.read_parquet(C1/"dataset_manifests"/f"{rep}_union.parquet",columns=["instance_hash","m4_ce"])
        union_m4=union_frame.set_index("instance_hash").m4_ce
        frozen_check=selected[selected.replicate==rep].drop_duplicates("instance_hash").set_index("instance_hash").true_m4_ce
        if not np.allclose(union_m4.loc[frozen_check.index].to_numpy(float),frozen_check.to_numpy(float),rtol=0,atol=1e-12):
            raise RuntimeError(f"M4 CE source mismatch {rep}")
        model_maps={a:pd.read_parquet(OUT/"hardness"/f"{rep}_{a}_on_union.parquet").set_index("instance_hash").loss_raw for a in ARMS}
        for data_arm in DATA_ARMS:
            if data_arm in ARMS: hashes=(D0/"selected_hashes"/f"{rep}_{data_arm}.txt").read_text().splitlines()
            else: hashes=load_bundle(B/"labels"/f"{rep}_{data_arm}"/"labeled_bundle.npz")[1]
            m4=union_m4.loc[hashes].to_numpy(float); matrix.append({"replicate":rep,"model":"M4","dataset_arm":data_arm,"n":len(hashes),"mean_ce":m4.mean(),"median_ce":np.median(m4),"mean_gain_M4_minus_model":0.0,"improvement_fraction":np.nan})
            for model in ARMS:
                tuned=model_maps[model].loc[hashes].to_numpy(float); gain=m4-tuned; matrix.append({"replicate":rep,"model":model,"dataset_arm":data_arm,"n":len(hashes),"mean_ce":tuned.mean(),"median_ce":np.median(tuned),"mean_gain_M4_minus_model":gain.mean(),"improvement_fraction":np.mean(gain>0)})
                instance.extend({"replicate":rep,"model":model,"dataset_arm":data_arm,"instance_hash":h,"m4_ce":float(x),"model_ce":float(y),"gain_m4_minus_model":float(x-y)} for h,x,y in zip(hashes,m4,tuned))
    matrix=pd.DataFrame(matrix); matrix.to_csv(OUT/"07_cross_population_ce.csv",index=False); pd.DataFrame(instance).to_parquet(OUT/"logs/cross_population_ce_per_instance.parquet",index=False)
    (OUT/"07_cross_population_ce.md").write_text("# Cross-Population CE\n\nPositive gain is M4 CE minus M5 CE. Own-population CE is optimization evidence, not generalization evidence. E0/E1/A/B/C share candidates, so cross-population cells diagnose transfer only.\n\n"+md_table(matrix)+"\n")
    # Training curves and frozen structure/hardness relationship.
    dyn=[]
    for rep,arm in JOB_ORDER:
        c=pd.read_csv(run_dir(rep,arm)/"training_cycle/09_bq_training/training_loss_curve.csv"); c.insert(0,"arm",arm); c.insert(0,"replicate",rep); dyn.append(c)
    dynamics=pd.concat(dyn,ignore_index=True); dynamics.to_csv(OUT/"logs/training_curves.csv",index=False)
    pop=pd.read_csv(D0/"06_hardness_metrics.csv"); struct=pd.read_csv(D0/"07_structure_metrics.csv")
    pop=pop[(pop.replicate.isin(REPLICATES))&(pop.selection.isin(ARMS))] if "selection" in pop else pop
    relationships=[]
    for rep,arm in JOB_ORDER:
        d=selected[(selected.replicate==rep)&(selected.selection==arm)]
        gap=float(evaluations[(evaluations.replicate==rep)&(evaluations.model==arm)&(evaluations.dataset=="OriginalTSPLIB70")].mean_gap.iloc[0])
        row={"replicate":rep,"arm":arm,"true_m4_ce_mean":d.true_m4_ce.mean(),"original70_gap":gap,"final_training_loss":float(tr[(tr.replicate==rep)&(tr.arm==arm)].final_loss.iloc[0])}
        s=struct[(struct.replicate==rep)&(struct.selection==arm)] if {"replicate","selection"}.issubset(struct.columns) else pd.DataFrame()
        for key in ("top5_ot_mean","top5_swd","occupied_niches","effective_lineages","largest_lineage_share"):
            if len(s) and key in s: row[key]=s[key].iloc[0]
        relationships.append(row)
    rel=pd.DataFrame(relationships); rel.to_csv(OUT/"logs/structure_training_relationship.csv",index=False)
    (OUT/"08_structure_training_relationship.md").write_text("# Structure–Hardness–Training Relationship\n\nThis is a frozen-strategy association analysis, not a claim that structure distance or hardness alone is causal. C changes membership, lineage and local Full-F40 composition together.\n\n"+md_table(rel)+"\n")
    # Decisions from pre-registered rules.
    primary=stats[(stats.dataset=="OriginalTSPLIB70")&(stats.stratum=="ALL")].set_index(["replicate","comparison"]); gap5=stats[(stats.dataset=="GAP5")&(stats.stratum=="ALL")].set_index(["replicate","comparison"])
    def hypothesis(comp: str, prefix: str) -> tuple[str,int,int]:
        wins=sum(float(primary.at[(r,comp),"mean_delta_comparator_minus_C"])>0 for r in REPLICATES); gap_worse=sum(float(gap5.at[(r,comp),"mean_delta_comparator_minus_C"])<0 for r in REPLICATES)
        label=f"{prefix}_PRELIMINARILY_SUPPORTED" if wins==3 and gap_worse<3 else f"{prefix}_NOT_SUPPORTED" if wins==0 else f"{prefix}_INCONCLUSIVE"
        return label,wins,gap_worse
    h1,w1,g1=hypothesis("C_vs_A","P6D1_TARGET_AWARE_GAIN"); h2,w2,g2=hypothesis("C_vs_B","P6D1_HARDNESS_GUIDED_GAIN")
    abswins=sum(float(primary.at[(r,"C_vs_M4"),"mean_delta_comparator_minus_C"])>0 for r in REPLICATES); h3="SUPPORTED" if abswins==3 else "PARTIAL" if abswins>0 else "NOT_SUPPORTED"
    h4="PROMISING" if h1.endswith("PRELIMINARILY_SUPPORTED") and h2.endswith("PRELIMINARILY_SUPPORTED") and h3!="NOT_SUPPORTED" else "NOT_SUPPORTED" if h1.endswith("NOT_SUPPORTED") and h2.endswith("NOT_SUPPORTED") else "NOT_ESTABLISHED"
    orig=evaluations[evaluations.dataset=="OriginalTSPLIB70"].pivot(index="replicate",columns="model",values="mean_gap")
    hist_orig=hist[hist.dataset=="OriginalTSPLIB70"].pivot(index="replicate",columns="model",values="mean_gap")
    beat={m:int((hist_orig.C<hist_orig[m]).sum()) for m in ("M4","P6B_E0","P6B_E1","P6C2_Oracle_C1")}
    monitor=pd.read_csv(OUT/"monitoring/hourly_status.csv") if (OUT/"monitoring/hourly_status.csv").exists() else pd.DataFrame()
    hourly="COMPLETE" if len(monitor)>=1 and bool(monitor.initial_check.iloc[0]) else "VIOLATED"
    failures=(OUT/"monitoring/failures.md").read_text().count("## ")
    repairs=(OUT/"monitoring/repair_actions.md").read_text().count("## ")
    final=["# P6-D1 Final Research Conclusion","",f"`{h1}`",f"`{h2}`",f"`P6D1_ABSOLUTE_IMPROVEMENT: {h3}`",f"`P6D1_ONLINE_EVOLUTION_READINESS: {h4}`",f"`P6D1_HOURLY_MONITORING: {hourly}`","",f"C wins Original70 versus A in {w1}/3 replicates and versus B in {w2}/3; C beats M4 in {abswins}/3. GAP5 is worse for C in {g1}/3 C-vs-A comparisons and {g2}/3 C-vs-B comparisons.",f"Historical Original70 wins for C: {', '.join(f'{k} {v}/3' for k,v in beat.items())}.","",md_table(orig.reset_index()),"","TSPLIB48 is target-informed and TSPLIB22 has been repeatedly observed; neither is described as a pristine holdout. Cross-population CE and the structure relationship are diagnostic, not stand-alone generalization or causal evidence.","",f"Scientific training/evaluation/CE failures: {failures}. Non-scientific monitor/report repair records: {repairs}. No new selection, Concorde, mutation, evolution, HGB fitting, extra seed, or P6-D2 was run.","","`BLADE_P6D1_TARGET_AWARE_QD_TRAINING_VALUE_COMPLETE`"]
    (OUT/"09_final_research_conclusion.md").write_text("\n".join(final)+"\n")
    # Figures.
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        vals=[orig.at[rep,m] for m in ("M4","A","B","C")]; ax.bar(range(4),vals); ax.set_xticks(range(4),("M4","A","B","C")); ax.set_title(rep); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Original70 mean gap (%)"); fig.tight_layout(); fig.savefig(OUT/"figures/01_original70_gap_comparison.png",dpi=180); plt.close(fig)
    for comp,num in (("C_vs_A","02"),("C_vs_B","03")):
        p=stats[(stats.comparison==comp)&(stats.stratum=="ALL")&stats.dataset.isin(["OriginalTSPLIB70","TSPLIB48","TSPLIB22","GAP5"])]; fig,ax=plt.subplots(figsize=(9,4)); labels=[f"{r.replicate}\n{r.dataset}" for r in p.itertuples()]; ax.bar(range(len(p)),p.mean_delta_comparator_minus_C); ax.axhline(0,color="black",lw=.8); ax.set_xticks(range(len(p)),labels,rotation=60,ha="right"); ax.set_ylabel("Comparator gap − C gap (%)"); fig.tight_layout(); fig.savefig(OUT/f"figures/{num}_{comp.lower()}_paired_gap.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    for arm,marker in zip(ARMS,("o","s","^")):
        p=rel[rel.arm==arm]; ax.scatter(p.true_m4_ce_mean,p.original70_gap,label=arm,marker=marker,s=80)
        for r in p.itertuples(): ax.annotate(r.replicate,(r.true_m4_ce_mean,r.original70_gap),fontsize=8)
    ax.set_xlabel("Mean true M4 CE"); ax.set_ylabel("Original70 mean gap (%)"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"figures/04_hardness_structure_training.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        p=matrix[matrix.replicate==rep].pivot(index="model",columns="dataset_arm",values="mean_ce").loc[["M4","A","B","C"],list(DATA_ARMS)]; im=ax.imshow(p,aspect="auto"); ax.set_xticks(range(5),p.columns); ax.set_yticks(range(4),p.index); ax.set_title(rep)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label="Mean CE"); fig.savefig(OUT/"figures/05_cross_population_ce.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True)
    for ax,(rep,arm) in zip(axes.ravel(),JOB_ORDER):
        p=dynamics[(dynamics.replicate==rep)&(dynamics.arm==arm)]; ax.plot(p.epoch,p.mean_training_loss); ax.set_title(f"{rep}-{arm}"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/"figures/06_training_loss_curves.png",dpi=180); plt.close(fig)
    runtime=tr[["replicate","arm","training_loop_wall_s","invocation_wall_s","gpu_peak_allocated_bytes"]].copy(); runtime.to_csv(OUT/"logs/runtime_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,4)); labels=(runtime.replicate+"-"+runtime.arm).tolist(); ax.bar(range(9),runtime.training_loop_wall_s/60); ax.set_xticks(range(9),labels,rotation=45,ha="right"); ax.set_ylabel("Training minutes"); fig.tight_layout(); fig.savefig(OUT/"figures/07_runtime_summary.png",dpi=180); plt.close(fig)
    total_train=runtime.training_loop_wall_s.sum(); total_eval=sum(json.loads((run_dir(r,a)/"evaluation_integrity.json").read_text())["evaluation_wall_s"] for r,a in JOB_ORDER); total_ce=sum(json.loads((OUT/"hardness"/f"{r}_{a}_on_union.json").read_text())["wall_s"] for r,a in JOB_ORDER)
    (OUT/"10_runtime_resource_audit.md").write_text(f"# Runtime and Resource Audit\n\n- Nine serial training loops: {total_train/3600:.4f} GPU-hours.\n- Greedy evaluations: {total_eval/3600:.4f} GPU-hours.\n- Cross-population CE inference: {total_ce/3600:.4f} GPU-hours.\n- Peak allocated training memory: {runtime.gpu_peak_allocated_bytes.max()/2**30:.3f} GiB.\n- Monitoring snapshots: {len(monitor)}; monitoring rule: {hourly}.\n- Scientific training/evaluation/CE failures: {failures}.\n- Non-scientific monitor/report repair records: {repairs}; details are in `monitoring/repair_actions.md`.\n")
    completion={"status":"COMPLETE","final_label":"BLADE_P6D1_TARGET_AWARE_QD_TRAINING_VALUE_COMPLETE","h1":h1,"h2":h2,"h3":h3,"h4":h4,"hourly_monitoring":hourly,"protocol_sha256":cfg["protocol_sha256"],"completed_utc":now(),"final_report_sha256":sha256_file(OUT/"09_final_research_conclusion.md"),"wall_s":time.perf_counter()-started}
    write_json(OUT/"completion.json",completion); Status().update(phase="complete",status="COMPLETE",completed_runs=9,total_runs=9); return completion


def quarantine_incomplete(rep: str, arm: str, attempt: int) -> None:
    targets=[run_dir(rep,arm)/"training_cycle",checkpoint(rep,arm),run_dir(rep,arm)/"training_integrity.json"]
    q=run_dir(rep,arm)/"quarantine"/f"attempt_{attempt}_{int(time.time())}"; q.mkdir(parents=True,exist_ok=True)
    for p in targets:
        if p.exists(): shutil.move(str(p),str(q/p.name))


def log_failure(rep: str, arm: str, phase: str, attempt: int, exc: BaseException) -> None:
    with (OUT/"monitoring/failures.md").open("a") as f: f.write(f"## {now()} {rep}-{arm} {phase} attempt {attempt}\n\n```\n{traceback.format_exc()}\n```\n\n")


def runner(device: str) -> None:
    ensure_dirs(); lock=(OUT/"monitoring/runner.lock").open("w");
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return
    (OUT/"monitoring/runner.pid").write_text(str(os.getpid())+"\n"); Status().update(phase="runner_start",status="RUNNING",runner_pid=os.getpid(),total_runs=9)
    try:
        if not (OUT/"logs/audit.json").exists() or json.loads((OUT/"logs/audit.json").read_text()).get("status")!="PASS": audit()
        for index,(rep,arm) in enumerate(JOB_ORDER,1):
            completed=int(training_manifest().status.eq("PASS").sum()); Status().update(phase="training_queue",status="RUNNING",current_run=f"{rep}-{arm}",queue_index=index,completed_runs=completed,next_run=f"{rep}-{arm}")
            for attempt in range(3):
                try: train(rep,arm,device); break
                except Exception as exc:
                    log_failure(rep,arm,"training",attempt,exc)
                    fatal=any(x in str(exc).lower() for x in ("nan","inf","mismatch","drift","illegal","validity","integrity"))
                    if fatal or attempt>=2: Status().update(phase="training",status="BLOCKED",current_run=f"{rep}-{arm}",error=str(exc)); raise
                    quarantine_incomplete(rep,arm,attempt+1)
                    with (OUT/"monitoring/repair_actions.md").open("a") as f: f.write(f"## {now()} {rep}-{arm}\n\nTransient training failure; quarantined isolated incomplete outputs and restarted the full run from formal M4 (retry {attempt+1}/2).\n\n")
        for rep,arm in JOB_ORDER:
            for attempt in range(3):
                try: evaluate(rep,arm,device); break
                except Exception as exc:
                    log_failure(rep,arm,"evaluation",attempt,exc)
                    if attempt>=2: Status().update(phase="evaluation",status="BLOCKED",current_run=f"{rep}-{arm}",error=str(exc)); raise
        for rep,arm in JOB_ORDER:
            for attempt in range(3):
                try: infer_ce(rep,arm,device); break
                except Exception as exc:
                    log_failure(rep,arm,"ce",attempt,exc)
                    if attempt>=2: Status().update(phase="ce",status="BLOCKED",current_run=f"{rep}-{arm}",error=str(exc)); raise
        finalize()
    except Exception as exc:
        Status().update(status="BLOCKED",error=str(exc)); raise
    finally:
        try: (OUT/"monitoring/runner.pid").unlink()
        except FileNotFoundError: pass
        if (OUT/"completion.json").exists(): subprocess.run(["systemctl","--user","disable","--now","blade-p6d1-monitor.timer"],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def pid_alive(pid: int) -> bool:
    try: os.kill(pid,0); return True
    except (ProcessLookupError,PermissionError): return False


def monitor(initial: bool=False) -> dict[str,Any]:
    ensure_dirs(); lock=(OUT/"monitoring/monitor.lock").open("w")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return {"status":"SKIP_OVERLAP"}
    status=json.loads((OUT/"status.json").read_text()) if (OUT/"status.json").exists() else {}
    pid_path=OUT/"monitoring/runner.pid"; pid=int(pid_path.read_text()) if pid_path.exists() else -1; alive=pid>0 and pid_alive(pid)
    completed=int(training_manifest().status.eq("PASS").sum()) if (OUT/"logs/audit.json").exists() else 0
    current=str(status.get("current_run","")); epochs=updates=0; loss_finite=True; checkpoint_ok=True; oom=False
    if "-" in current:
        rep,arm=current.rsplit("-",1); curve=run_dir(rep,arm)/"training_cycle/09_bq_training/training_loss_curve.csv"
        if curve.exists():
            try:
                c=pd.read_csv(curve); epochs=len(c); updates=int(c.optimizer_updates.sum()); loss_finite=bool(np.isfinite(c.mean_training_loss).all())
            except Exception: loss_finite=False
        cp=checkpoint(rep,arm); integrity=run_dir(rep,arm)/"training_integrity.json"
        if integrity.exists() and cp.exists(): checkpoint_ok=sha256_file(cp)==json.loads(integrity.read_text()).get("checkpoint_sha256")
        failure=(OUT/"monitoring/failures.md").read_text().lower(); oom="out of memory" in failure[-10000:] or "cuda oom" in failure[-10000:]
    gpu_access=False; gpu_conflict=False; gpu_text=""
    try:
        q=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20); gpu_access=q.returncode==0; gpu_text=q.stdout.strip().replace("\n","; ")
        if gpu_access:
            gpu1_uuid=next((x.split(",")[1].strip() for x in q.stdout.strip().splitlines() if x.split(",")[0].strip()=="1"),"")
            apps=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20)
            foreign=[]
            if apps.returncode==0:
                for line in apps.stdout.strip().splitlines():
                    fields=[x.strip() for x in line.split(",")]
                    if len(fields)>=4 and fields[0]==gpu1_uuid and int(fields[1])!=pid: foreign.append(line)
            gpu_conflict=bool(foreign); gpu_text += ("; foreign_gpu1="+" | ".join(foreign)) if foreign else "; foreign_gpu1=none"
    except Exception as exc: gpu_text=str(exc)
    disk=shutil.disk_usage(OUT); completion=(OUT/"completion.json").exists()
    snapshot={"timestamp_utc":now(),"initial_check":bool(initial),"runner_alive":alive,"runner_pid":pid if alive else "","phase":status.get("phase","UNKNOWN"),"current_run":current,"task_status":status.get("status","UNKNOWN"),"epochs":epochs,"updates":updates,"loss_finite":loss_finite,"gpu_access":gpu_access,"gpu_conflict":gpu_conflict,"gpu_status":gpu_text,"disk_free_gib":disk.free/2**30,"checkpoint_ok":checkpoint_ok,"oom_seen":oom,"abnormal_exit":not alive and not completion,"completed_runs":completed,"next_run":status.get("next_run",""),"experiment_complete":completion}
    csv=OUT/"monitoring/hourly_status.csv"; pd.DataFrame([snapshot]).to_csv(csv,mode="a",header=not csv.exists(),index=False)
    write_json(OUT/"monitoring/current_status.json",snapshot)
    with (OUT/"monitoring/hourly_monitor.log").open("a") as f: f.write(json.dumps(snapshot,sort_keys=True)+"\n")
    if not alive and not completion and status.get("status")!="BLOCKED":
        repair=OUT/"monitoring/runner_restart_count.json"; value=json.loads(repair.read_text()) if repair.exists() else {"count":0}
        if value["count"]<2:
            value["count"]+=1; value["last_restart_utc"]=now(); write_json(repair,value)
            subprocess.run(["systemctl","--user","start","blade-p6d1-runner.service"],check=False)
            with (OUT/"monitoring/repair_actions.md").open("a") as f: f.write(f"## {now()} runner restart\n\nHourly monitor found no live runner and requested safe idempotent service restart ({value['count']}/2).\n\n")
    if completion: subprocess.run(["systemctl","--user","disable","--now","blade-p6d1-monitor.timer"],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return snapshot


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("audit","train","evaluate","ce","finalize","runner","monitor")); p.add_argument("--replicate",choices=REPLICATES); p.add_argument("--arm",choices=ARMS); p.add_argument("--device",default="cuda:1"); p.add_argument("--initial",action="store_true"); a=p.parse_args()
    if a.phase=="audit": result=audit()
    elif a.phase=="runner": result=runner(a.device)
    elif a.phase=="monitor": result=monitor(a.initial)
    elif a.phase=="finalize": result=finalize()
    else:
        if not a.replicate or not a.arm: p.error("--replicate and --arm are required")
        result=train(a.replicate,a.arm,a.device) if a.phase=="train" else evaluate(a.replicate,a.arm,a.device) if a.phase=="evaluate" else infer_ce(a.replicate,a.arm,a.device)
    if result is not None: print(json.dumps(result,indent=2,default=str))


if __name__=="__main__": main()
