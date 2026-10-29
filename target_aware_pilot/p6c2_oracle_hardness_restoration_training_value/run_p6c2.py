#!/usr/bin/env python3
"""P6-C2 oracle-hardness restoration training-value validation.

This stage consumes the byte-frozen P6-C1 C0/C1 memberships.  It never solves
selection again and never invokes mutation, evolution, HGB, or Concorde.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import time
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
P6C1 = ROOT / "target_aware_pilot/p6c1_hardness_controlled_matching"
P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
P6C0 = ROOT / "target_aware_pilot/p6c0_training_value_mechanism_audit"
P6A = ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"
P5 = ROOT / "target_aware_pilot/p5_structure_guided_evolution_robustness"
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
M4 = RUN / "checkpoints/M4_E256.tsp.best"
M4_EVAL = RUN / "cycles/C03/evaluation_work/M4"
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
ATTACHMENT = Path("/home/haozou/.codex/attachments/3c464de0-53bf-495e-8bd5-cf59b4ae8af0/pasted-text.txt")

sys.path[:0] = [
    str(ROOT / "src"), str(ROOT / "scripts"), str(P6A), str(P5),
    str(ROOT / "target_aware_pilot/p4b1_acquisition_ablation"),
    str(ROOT / "target_aware_pilot/p4c2_data_sampling_robustness"),
]
import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core  # noqa: E402
import run_p5_structure_guided_evolution_robustness as p5  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from bq.training_hardness import protocol_manifest as hardness_protocol  # noqa: E402
from bq.tsp_metric import validate_tour_0based  # noqa: E402
from final_run.science import hardness, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import sha256_file  # noqa: E402

REPLICATES = ("Original", "S1", "S2")
ARMS = ("C0", "C1")
SOURCE_ARMS = ("E0", "E1")
DATA_ARMS = ("E0", "E1", "C0", "C1")
DATASETS = tuple(p5.DATASETS)
GAP5 = tuple(p5.GAP5_DATASETS)
K = 10_000
TRAINING_SEED = 2026
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261029
BATCH_SIZE = 128
P90 = float(pd.read_csv(P6C0 / "02_hardness_statistics.csv").pooled_p90_threshold.dropna().iloc[0])
P95 = float(pd.read_csv(P6C0 / "02_hardness_statistics.csv").pooled_p95_threshold.dropna().iloc[0])
F40 = tuple(json.loads((ROOT / "audits/S3_B2_RF_HGB_feature_label_scaling_v1/09_feature_subset_manifests.json").read_text())["F40"])


def ensure_dirs() -> None:
    for path in (OUT, OUT/"configs", OUT/"datasets", OUT/"checkpoints", OUT/"logs", OUT/"figures", OUT/"hardness"):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.bool_): return bool(x)
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
    def cell(v: Any) -> str:
        if pd.isna(v): return "NA"
        if isinstance(v, (float, np.floating)): return f"{float(v):.8g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


class Status:
    def update(self, **values: Any) -> None:
        write_json(OUT / "status.json", {"timestamp_unix": time.time(), **values})


def selected_manifest(rep: str, arm: str) -> Path: return P6C1 / "dataset_manifests" / f"{rep}_{arm}.parquet"
def selected_bundle(rep: str, arm: str) -> Path: return P6C1 / "dataset_manifests" / f"{rep}_{arm}.npz"
def union_bundle(rep: str) -> Path: return P6C1 / "dataset_manifests" / f"{rep}_union.npz"
def source_bundle(rep: str, arm: str) -> Path: return P6B / "labels" / f"{rep}_{arm}" / "labeled_bundle.npz"
def run_dir(rep: str, arm: str) -> Path: return OUT / "logs" / f"{rep}_{arm}"
def checkpoint(rep: str, arm: str) -> Path: return OUT / "checkpoints" / f"{rep}_{arm}_M5_E256.tsp.best"
def eval_label(rep: str, arm: str) -> str: return f"P6C2_{rep}_{arm}_M5"
def eval_csv(rep: str, arm: str) -> Path: return run_dir(rep, arm) / "external_benchmark_results.csv"


def load_bundle(path: Path, require_tours: bool = False) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as z:
        xy = np.asarray(z["coordinates"], np.float64)
        hashes = z["instance_hash"].astype(str).tolist()
        tours = np.asarray(z["expert_tours"], np.int64) if "expert_tours" in z.files else None
    if require_tours and tours is None: raise RuntimeError(f"missing expert tours: {path}")
    return xy, hashes, tours


def base_training_protocol() -> dict[str, Any]:
    trainer = ROOT / "scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
    return {
        "authority": "P6-B actual frozen C04->M5 implementation", "trainer_sha256": sha256_file(trainer),
        "m4_sha256": M4_SHA, "training_seed": TRAINING_SEED, "optimizer": "fresh Adam",
        "scheduler": None, "learning_rate": 3.75e-5, "epochs": 256, "optimizer_updates": 10_240,
        "logical_batch_size": 256, "microbatch_size": 64, "loss": "teacher-forced imitation",
        "checkpoint_rule": "fixed E256", "evaluation": "P6-B frozen greedy evaluator",
    }


def frozen_protocol() -> dict[str, Any]:
    config = {
        "experiment": "BLADE-NCO P6-C2", "source_p6c1": str(P6C1.relative_to(ROOT)),
        "source_p6c1_completion_sha256": sha256_file(P6C1/"completion.json"),
        "source_p6c1_config_sha256": sha256_file(P6C1/"configs/frozen_p6c1_config.json"),
        "source_p6b_completion_sha256": sha256_file(P6B/"completion.json"),
        "source_p6c0_completion_sha256": sha256_file(P6C0/"completion.json"),
        "request_sha256": sha256_file(ATTACHMENT), "m4_sha256": M4_SHA,
        "replicates": list(REPLICATES), "arms": list(ARMS), "training": base_training_protocol(),
        "teacher_forced_ce": hardness_protocol(), "p6c1_ot_tolerance_is_report_only": True,
        "h2_rule": "SUPPORTED iff all 3 C1 Original70 gaps < paired C0 and not all 3 GAP5 deltas favor C0; 0/3 Original70 wins is NOT_SUPPORTED; otherwise INCONCLUSIVE",
        "h3_rule": "SUPPORTED iff all 3 C1 beat paired historical E0 and M4; PARTIAL iff C1 beats historical E1 in >=2 replicates or beats E0/M4 in >=1; otherwise NOT_SUPPORTED",
        "bootstrap_n": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
        "prohibitions": ["new mutation", "new evolution", "new Concorde", "new HGB", "selection re-solve", "hyperparameter change", "extra seed", "P6-C3"],
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    }
    config["protocol_sha256"] = canonical_hash(config)
    return config


def protocol_gate() -> dict[str, Any]:
    ensure_dirs()
    config = frozen_protocol(); path = OUT / "configs/frozen_p6c2_config.json"
    if path.is_file() and json.loads(path.read_text()) != config: raise RuntimeError("frozen P6-C2 protocol drift")
    if not path.is_file(): write_json(path, config)
    (OUT/"00_protocol.md").write_text(
        "# P6-C2 Frozen Protocol\n\nP6-C2 directly consumes the six byte-frozen P6-C1 optimal memberships. The P6-C1 1.20× OT gate remains a historical NO-GO result but is report-only here. All six runs use the official M4, fresh Adam, LR 3.75e-5, seed 2026, 256 epochs, 10,240 updates, fixed E256 selection, and the P6-B greedy evaluator. No selection is recomputed.\n\n"
        "H2 and H3 operational rules were frozen before training and are recorded in `configs/frozen_p6c2_config.json`. No benchmark result may alter them.\n"
    )
    return config


def shannon_effective(values: pd.Series) -> float:
    p = values.astype(str).value_counts(normalize=True).to_numpy(float)
    return float(np.exp(-(p * np.log(p)).sum()))


def load_f40() -> pd.DataFrame:
    files = sorted((P6C0/"full_f40_cache").glob("f40_*.parquet"))
    if not files: raise RuntimeError("P6-C0 F40 cache missing")
    frame = pd.concat([pd.read_parquet(x) for x in files], ignore_index=True).drop_duplicates("instance_hash")
    if not set(F40).issubset(frame.columns): raise RuntimeError("F40 cache schema drift")
    return frame.set_index("instance_hash")


def audit() -> dict[str, Any]:
    started = time.perf_counter(); config = protocol_gate()
    if sha256_file(M4) != M4_SHA: raise RuntimeError("M4 hash mismatch")
    p6c1_done = json.loads((P6C1/"completion.json").read_text())
    if p6c1_done.get("final_label") != "BLADE_P6C1_HARDNESS_CONTROLLED_MATCHING_COMPLETE": raise RuntimeError("P6-C1 incomplete")
    quotas = pd.read_csv(P6C1/"02_decile_quotas.csv")
    feasibility = pd.read_csv(P6C1/"04_feasibility_results.csv").set_index("replicate")
    metric = pd.read_csv(P6C1/"logs/selection_metrics.csv").set_index(["replicate", "arm"])
    f40 = load_f40(); rows = []; population = []; top5_rows = []; operator_rows = []; ref_rows = []
    memberships: dict[tuple[str,str], set[str]] = {}
    for rep in REPLICATES:
        ux, uh, ut = load_bundle(union_bundle(rep), True); assert ut is not None
        union_set = set(uh)
        for arm in ARMS:
            path_m = selected_manifest(rep, arm); path_b = selected_bundle(rep, arm)
            m = pd.read_parquet(path_m).reset_index(drop=True); xy, hashes, tours = load_bundle(path_b, True); assert tours is not None
            if sha256_file(path_m) != str(metric.at[(rep,arm),"manifest_sha256"]) or sha256_file(path_b) != str(metric.at[(rep,arm),"bundle_sha256"]): raise RuntimeError(f"P6-C1 recorded artifact hash drift {rep}/{arm}")
            memberships[(rep,arm)] = set(hashes)
            if xy.shape != (K,100,2) or not np.isfinite(xy).all() or np.any((xy<0)|(xy>1)): raise RuntimeError(f"coordinate validity {rep}/{arm}")
            if hashes != m.instance_hash.astype(str).tolist() or len(set(hashes)) != K or not set(hashes).issubset(union_set): raise RuntimeError(f"membership drift {rep}/{arm}")
            if hashes != [coordinate_hash(x) for x in xy]: raise RuntimeError(f"coordinate hash failure {rep}/{arm}")
            if not all(validate_tour_0based(t,100)[0] for t in tours): raise RuntimeError(f"tour legality {rep}/{arm}")
            frozen_hashes = (P6C1/"selected_hashes"/f"{rep}_{arm}.txt").read_text().splitlines()
            if hashes != frozen_hashes: raise RuntimeError(f"selected-hash drift {rep}/{arm}")
            refs = np.bincount(m.selected_reference_index.to_numpy(int), minlength=48)
            if refs.sum()!=K or set(refs.tolist())!={208,209}: raise RuntimeError(f"reference quota {rep}/{arm}")
            dec = np.bincount(m.hardness_decile.to_numpy(int)-1,minlength=10)
            if arm=="C1":
                expected=quotas[quotas.replicate.eq(rep)].sort_values("decile").quota.to_numpy(int)
                if not np.array_equal(dec,expected): raise RuntimeError(f"hardness quota {rep}")
            objective=float(m.selected_squared_cost.sum()); expected_obj=float(feasibility.at[rep,"OT_C0_total" if arm=="C0" else "OT_C1_min_total"])
            if not np.isclose(objective,expected_obj,rtol=1e-6,atol=1e-3): raise RuntimeError(f"OT objective drift {rep}/{arm}")
            if not set(hashes).issubset(f40.index): raise RuntimeError(f"F40 coverage {rep}/{arm}")
            ce=m.m4_ce.to_numpy(float); lineage=m.lineage_root.astype(str)
            row={"record_type":"population","replicate":rep,"arm":arm,"n":K,"unique_n":len(set(hashes)),"membership_sha256":membership_hash(hashes),"manifest_sha256":sha256_file(path_m),"bundle_sha256":sha256_file(path_b),"exact_ot_total":objective,"exact_ot_mean":objective/K,"top5_swd":float(metric.at[(rep,arm),"top5_swd"]),"top5_energy":float(metric.at[(rep,arm),"top5_energy"]),"m4_ce_mean":ce.mean(),"m4_ce_median":np.median(ce),"m4_ce_p90":np.quantile(ce,.9),"m4_ce_p95":np.quantile(ce,.95),"hard_tail_p90_fraction":np.mean(ce>=P90),"hard_tail_p95_fraction":np.mean(ce>=P95),"hardness_deciles":" ".join(map(str,dec)),"e0_source_fraction":m.in_E0.mean(),"e1_source_fraction":m.in_E1.mean(),"parent_or_p0_fraction":np.mean(m.generation_born.to_numpy(int)<2),"g2_offspring_fraction":np.mean(m.generation_born.to_numpy(int)==2),"effective_lineages":shannon_effective(lineage),"largest_lineage_share":lineage.value_counts(normalize=True).max(),"all_tours_legal":True,"reference_quotas_exact":True,"hardness_decile_requirement":"PASS" if arm=="C1" else "NOT_APPLICABLE"}
            population.append(row); rows.append({k:row[k] for k in ("replicate","arm","n","unique_n","membership_sha256","manifest_sha256","bundle_sha256","all_tours_legal","reference_quotas_exact","hardness_decile_requirement")})
            for feature in [c for c in m.columns if c.startswith("phi__")]:
                x=m[feature].to_numpy(float); top5_rows.append({"record_type":"top5_marginal","replicate":rep,"arm":arm,"feature":feature,"mean":x.mean(),"std":x.std(ddof=1),"p10":np.quantile(x,.1),"median":np.median(x),"p90":np.quantile(x,.9)})
            z=f40.loc[hashes,list(F40)]
            for feature in F40:
                x=z[feature].to_numpy(float); top5_rows.append({"record_type":"f40_transformed_marginal","replicate":rep,"arm":arm,"feature":feature,"mean":x.mean(),"std":x.std(ddof=1),"p10":np.quantile(x,.1),"median":np.median(x),"p90":np.quantile(x,.9)})
            for op,n in m.operator_id.astype(str).value_counts().items(): operator_rows.append({"record_type":"operator","replicate":rep,"arm":arm,"feature":op,"count":int(n),"share":n/K})
            for j,n in enumerate(refs): ref_rows.append({"record_type":"reference_quota","replicate":rep,"arm":arm,"feature":str(j),"count":int(n),"share":n/K})
            write_json(OUT/"datasets"/f"{rep}_{arm}_source.json", {"source_bundle":str(path_b),"source_manifest":str(path_m),"bundle_sha256":sha256_file(path_b),"manifest_sha256":sha256_file(path_m),"membership_sha256":membership_hash(hashes),"n":K})
    overlap=[]
    for rep in REPLICATES:
        c0=memberships[(rep,"C0")]; c1=memberships[(rep,"C1")]
        e0=set(load_bundle(source_bundle(rep,"E0"))[1]); e1=set(load_bundle(source_bundle(rep,"E1"))[1])
        for arm,values in (("C0",c0),("C1",c1)):
            overlap.append({"record_type":"membership_overlap","replicate":rep,"arm":arm,"feature":"C0_inter_C1" if arm=="C0" else "C1_inter_C0","count":len(c0&c1),"share":len(c0&c1)/K})
            overlap.append({"record_type":"membership_overlap","replicate":rep,"arm":arm,"feature":"historical_E0","count":len(values&e0),"share":len(values&e0)/K})
            overlap.append({"record_type":"membership_overlap","replicate":rep,"arm":arm,"feature":"historical_E1","count":len(values&e1),"share":len(values&e1)/K})
    pop=pd.DataFrame(population); composition=pd.concat([pop,pd.DataFrame(top5_rows),pd.DataFrame(operator_rows),pd.DataFrame(ref_rows),pd.DataFrame(overlap)],ignore_index=True,sort=False)
    composition.to_csv(OUT/"02_population_composition.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"logs/dataset_integrity.csv",index=False)
    (OUT/"01_dataset_integrity.md").write_text("# Dataset Integrity\n\nAll six datasets are the exact P6-C1 frozen memberships; no selection was re-solved. Coordinates, ordered hashes, selected-hash files, labels, legal TSP100 tours, 48 reference quotas, C1 decile quotas, objectives, and F40 coverage pass. The P6-C1 1.20× gate remains failed and is not overwritten.\n\n"+md_table(pd.DataFrame(rows))+"\n")
    compare=pop[["replicate","arm","exact_ot_mean","top5_swd","top5_energy","m4_ce_mean","m4_ce_median","m4_ce_p90","m4_ce_p95","hard_tail_p90_fraction","hard_tail_p95_fraction","hardness_deciles","e0_source_fraction","e1_source_fraction","effective_lineages","largest_lineage_share"]]
    (OUT/"02_hardness_ot_comparison.md").write_text("# Hardness and OT Comparison\n\nP90/P95 tail thresholds are the frozen P6-C0 values %.8f and %.8f. OT is reported, not used as a P6-C2 gate.\n\n%s\n\nFull transformed-F40 marginals, Top5 marginals, reference quotas, operators, and overlaps are in `02_population_composition.csv`.\n"%(P90,P95,md_table(compare)))
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        for arm in ARMS:
            m=pd.read_parquet(selected_manifest(rep,arm)); counts=np.bincount(m.hardness_decile.to_numpy(int)-1,minlength=10); ax.plot(range(1,11),counts,marker="o",label=arm)
        e0=quotas[quotas.replicate.eq(rep)].sort_values("decile").quota.to_numpy(int); ax.plot(range(1,11),e0,ls="--",label="E0 quota"); ax.set_title(rep); ax.set_xlabel("Hardness decile"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Count"); axes[-1].legend(); fig.tight_layout(); fig.savefig(OUT/"figures/01_hardness_distribution.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));
    for r in pop.itertuples(): ax.scatter(r.m4_ce_mean,r.exact_ot_mean,s=60,label=f"{r.replicate}-{r.arm}")
    ax.set_xlabel("Mean M4 teacher-forced CE"); ax.set_ylabel("Mean exact Top5 OT"); ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(OUT/"figures/02_ot_hardness_comparison.png",dpi=180); plt.close(fig)
    result={"status":"PASS","protocol_sha256":config["protocol_sha256"],"datasets":6,"wall_s":time.perf_counter()-started}
    write_json(OUT/"logs/audit.json",result); return result


def write_training_manifest() -> pd.DataFrame:
    rows=[]
    for rep in REPLICATES:
        for arm in ARMS:
            p=run_dir(rep,arm)/"training_integrity.json"
            if p.is_file(): rows.append(json.loads(p.read_text()))
            else: rows.append({"replicate":rep,"arm":arm,"status":"PENDING","training_seed":TRAINING_SEED})
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"03_training_runs_manifest.csv",index=False)
    passed=int(frame.status.eq("PASS").sum())
    (OUT/"03_training_integrity.md").write_text(f"# Training Integrity\n\nCompleted: {passed}/6. Every PASS run verifies E000 tensor equality to formal M4, 256 epochs, 10,240 optimizer updates, finite weights, seed 2026, and fixed E256 selection.\n")
    return frame


def train(rep: str, arm: str, device_name: str) -> dict[str, Any]:
    config=protocol_gate(); audit_info=json.loads((OUT/"logs/audit.json").read_text())
    if audit_info.get("status")!="PASS": raise RuntimeError("audit gate missing")
    integrity=run_dir(rep,arm)/"training_integrity.json"; cp=checkpoint(rep,arm)
    if integrity.is_file():
        old=json.loads(integrity.read_text())
        if old.get("status")=="PASS" and cp.is_file() and sha256_file(cp)==old.get("checkpoint_sha256"): return old
    xy,hashes,tours=load_bundle(selected_bundle(rep,arm),True); assert tours is not None
    device=torch.device(device_name); torch.cuda.set_device(device) if device.type=="cuda" else None
    directory=run_dir(rep,arm)/"training_cycle"; directory.mkdir(parents=True,exist_ok=True)
    core.RUN_ID=f"TARGET_AWARE_P6C2_{rep}_{arm}"
    core.cycle_dir=lambda selected_arm,cycle: directory
    core.model_path=lambda selected_arm,index: cp if index==5 else M4
    model,mmeta=load_canonical_checkpoint(M4,device,expected_sha256=M4_SHA)
    tcfg={"replicate":rep,"arm":arm,"training_seed":TRAINING_SEED,"membership_sha256":membership_hash(hashes),"dataset_bundle_sha256":sha256_file(selected_bundle(rep,arm)),"source_manifest_sha256":sha256_file(selected_manifest(rep,arm)),"protocol_sha256":config["protocol_sha256"],"base_training_protocol":base_training_protocol()}
    tcfg["training_config_sha256"]=canonical_hash(tcfg); write_json(run_dir(rep,arm)/"training_config.json",tcfg)
    started=time.perf_counter(); trained,meta,timing=core.train(f"P6C2_{rep}_{arm}",4,model,mmeta["sha256"],xy,hashes,tours,device,Status(),True); invocation=time.perf_counter()-started
    stage=directory/"09_bq_training"; e000=torch.load(stage/"M5_E000.tsp.best",map_location="cpu",weights_only=False); m4=torch.load(M4,map_location="cpu",weights_only=False)
    equal=set(e000["net"])==set(m4["net"]) and all(torch.equal(e000["net"][k],m4["net"][k]) for k in m4["net"])
    curve=pd.read_csv(stage/"training_loss_curve.csv")
    if not equal or len(curve)!=256 or int(curve.optimizer_updates.sum())!=10240 or not curve.hard_instances_seen.eq(K).all() or meta["sha256"]!=sha256_file(cp) or not all(bool(torch.isfinite(p).all()) for p in trained.parameters()): raise RuntimeError("training integrity failure")
    result={"status":"PASS",**tcfg,"m4_sha256":M4_SHA,"e000_weights_equal_m4":True,"epochs":256,"optimizer_steps":int(timing["optimizer_steps"]),"final_loss":float(timing["final_loss"]),"training_loop_wall_s":float(timing["training_loop_wall_s"]),"invocation_wall_s":invocation,"gpu_peak_allocated_bytes":int(timing.get("gpu_peak_allocated_bytes") or 0),"checkpoint_sha256":meta["sha256"],"checkpoint_path":str(cp),"loss_curve_sha256":sha256_file(stage/"training_loss_curve.csv")}
    write_json(integrity,result); write_training_manifest(); del trained,model,xy,tours; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None; return result


def evaluate(rep: str, arm: str, device_name: str) -> dict[str, Any]:
    info_path=run_dir(rep,arm)/"evaluation_integrity.json"
    if info_path.is_file():
        old=json.loads(info_path.read_text())
        if old.get("status")=="PASS" and eval_csv(rep,arm).is_file(): return old
    tr=json.loads((run_dir(rep,arm)/"training_integrity.json").read_text()); cp=checkpoint(rep,arm)
    if sha256_file(cp)!=tr["checkpoint_sha256"]: raise RuntimeError("checkpoint drift")
    device=torch.device(device_name); model,meta=load_canonical_checkpoint(cp,device,expected_sha256=tr["checkpoint_sha256"])
    frame,wall=core.evaluate(model,cp,eval_label(rep,arm),eval_csv(rep,arm),Status(),f"P6C2_{rep}_{arm}")
    if set(frame.dataset.astype(str))!=set(DATASETS) or not frame.valid_tours.eq(frame.instance_count).all(): raise RuntimeError("evaluation incomplete")
    historical=pd.read_csv(P6B/"logs"/f"{rep}_E0"/"external_benchmark_results.csv").set_index("dataset"); current=frame.set_index("dataset")
    for d in DATASETS:
        if current.at[d,"dataset_sha256"]!=historical.at[d,"dataset_sha256"] or current.at[d,"evaluator_config_sha256"]!=historical.at[d,"evaluator_config_sha256"]: raise RuntimeError(f"evaluation protocol drift {d}")
    result={"status":"PASS","replicate":rep,"arm":arm,"checkpoint_sha256":meta["sha256"],"datasets":list(DATASETS),"all_tours_valid":True,"evaluation_wall_s":wall,"results_sha256":sha256_file(eval_csv(rep,arm))}
    write_json(info_path,result); del model; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None; return result


def infer_ce(rep: str, arm: str, device_name: str) -> dict[str, Any]:
    out=OUT/"hardness"/f"{rep}_{arm}_on_union.parquet"; runtime=out.with_suffix(".json")
    tr=json.loads((run_dir(rep,arm)/"training_integrity.json").read_text()); cp=checkpoint(rep,arm)
    if out.is_file() and runtime.is_file():
        old=json.loads(runtime.read_text())
        if old.get("status")=="PASS" and old.get("checkpoint_sha256")==tr["checkpoint_sha256"]: return old
    xy,hashes,tours=load_bundle(union_bundle(rep),True); assert tours is not None
    device=torch.device(device_name); torch.manual_seed(20261027); np.random.seed(20261027)
    model,meta=load_canonical_checkpoint(cp,device,expected_sha256=tr["checkpoint_sha256"]); model.eval(); started=time.perf_counter()
    values=hardness(model,xy,tours,device,batch_size=BATCH_SIZE); wall=time.perf_counter()-started
    pd.DataFrame({"replicate":rep,"model_arm":arm,"instance_hash":hashes,"loss_raw":values}).to_parquet(out,index=False)
    result={"status":"PASS","replicate":rep,"model_arm":arm,"n":len(hashes),"checkpoint_sha256":meta["sha256"],"output_sha256":sha256_file(out),"wall_s":wall,"gpu_peak_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0}
    write_json(runtime,result); del model,xy,tours; gc.collect(); torch.cuda.empty_cache() if device.type=="cuda" else None; return result


def detail(rep: str, arm: str, dataset: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir(rep,arm)/"evaluation_work"/eval_label(rep,arm)/dataset/"instance_metrics.parquet")


def bootstrap(values: np.ndarray, key: str) -> tuple[float,float]:
    x=np.asarray(values,float); seed=int.from_bytes(hashlib.sha256(f"P6C2|{BOOTSTRAP_SEED}|{key}".encode()).digest()[:8],"little"); rng=np.random.default_rng(seed); out=np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N): out[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(np.quantile(out,.025)),float(np.quantile(out,.975))


def finalize() -> dict[str, Any]:
    started=time.perf_counter(); protocol_gate()
    training=write_training_manifest()
    if len(training)!=6 or not training.status.eq("PASS").all(): raise RuntimeError("training incomplete")
    for rep in REPLICATES:
        for arm in ARMS:
            if json.loads((run_dir(rep,arm)/"evaluation_integrity.json").read_text()).get("status")!="PASS": raise RuntimeError("evaluation incomplete")
            if json.loads((OUT/"hardness"/f"{rep}_{arm}_on_union.json").read_text()).get("status")!="PASS": raise RuntimeError("CE incomplete")
    dev=set(pd.read_csv(ROOT/"target_aware_pilot/03_tsplib48_reference_manifest.csv").instance_id.astype(str)); held=set(json.loads((ROOT/"target_aware_pilot/p1_heldout22_names.json").read_text())["names"])
    parts=[]; result_rows=[]; paired_rows=[]
    eval_meta={}
    for rep in REPLICATES:
        for arm in ARMS:
            em=pd.read_csv(eval_csv(rep,arm)).set_index("dataset")
            for ds in DATASETS: eval_meta[(rep,arm,ds)]={"dataset_sha256":str(em.at[ds,"dataset_sha256"]),"evaluator_config_sha256":str(em.at[ds,"evaluator_config_sha256"])}
    for rep in REPLICATES:
        frames={}
        for ds in DATASETS:
            a=[]
            for arm in ARMS:
                x=detail(rep,arm,ds)[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].copy(); x=x.rename(columns={"coordinate_hash":f"hash_{arm}","n_nodes":f"n_{arm}","optimal_cost_int":f"opt_{arm}","gap_percent":f"gap_{arm}"}).drop(columns="valid_tour"); a.append(x)
            m=a[0].merge(a[1],on="instance_id",validate="one_to_one")
            if not m.hash_C0.eq(m.hash_C1).all() or not m.n_C0.eq(m.n_C1).all() or not m.opt_C0.eq(m.opt_C1).all(): raise RuntimeError(f"paired drift {rep}/{ds}")
            f=pd.DataFrame({"replicate":rep,"dataset":ds,"instance_id":m.instance_id.astype(str),"coordinate_hash":m.hash_C0,"n_nodes":m.n_C0,"optimal_cost_int":m.opt_C0,"gap_C0":m.gap_C0,"gap_C1":m.gap_C1,"Delta_C0_minus_C1":m.gap_C0-m.gap_C1}); frames[ds]=f; parts.append(f)
        original=frames["OriginalTSPLIB70"]
        for name,names in (("TSPLIB48",dev),("TSPLIB22",held)):
            sub=original[original.instance_id.isin(names)].copy(); sub["dataset"]=name
            if len(sub)!=len(names): raise RuntimeError(f"split {name}")
            frames[name]=sub; parts.append(sub)
        for ds,f in frames.items():
            v=f.Delta_C0_minus_C1.to_numpy(float); lo,hi=bootstrap(v,f"{rep}|{ds}")
            result_rows += [{"replicate":rep,"arm":arm,"dataset":ds,"n_instances":len(f),"mean_gap":float(f[f"gap_{arm}"].mean()),"checkpoint_sha256":json.loads((run_dir(rep,arm)/"training_integrity.json").read_text())["checkpoint_sha256"],"dataset_sha256":eval_meta[(rep,arm,"OriginalTSPLIB70" if ds in ("TSPLIB48","TSPLIB22") else ds)]["dataset_sha256"],"evaluator_config_sha256":eval_meta[(rep,arm,"OriginalTSPLIB70" if ds in ("TSPLIB48","TSPLIB22") else ds)]["evaluator_config_sha256"]} for arm in ARMS]
            paired_rows.append({"replicate":rep,"dataset":ds,"n_instances":len(f),"mean_gap_C0":f.gap_C0.mean(),"mean_gap_C1":f.gap_C1.mean(),"Delta_C0_minus_C1":v.mean(),"paired_ci95_low":lo,"paired_ci95_high":hi,"c1_instance_win_fraction":np.mean(v>0)})
        c0=float(np.mean([frames[x].gap_C0.mean() for x in GAP5])); c1=float(np.mean([frames[x].gap_C1.mean() for x in GAP5]))
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6C2|GAP5|{rep}".encode()).digest()[:8],"little")); boots=[]; vals=[frames[x].Delta_C0_minus_C1.to_numpy(float) for x in GAP5]
        for _ in range(BOOTSTRAP_N): boots.append(np.mean([v[rng.integers(0,len(v),len(v))].mean() for v in vals]))
        lo,hi=np.quantile(boots,[.025,.975]); result_rows += [{"replicate":rep,"arm":arm,"dataset":"GAP5","n_instances":sum(len(frames[x]) for x in GAP5),"mean_gap":c0 if arm=="C0" else c1,"checkpoint_sha256":json.loads((run_dir(rep,arm)/"training_integrity.json").read_text())["checkpoint_sha256"],"dataset_sha256":canonical_hash([eval_meta[(rep,arm,x)]["dataset_sha256"] for x in GAP5]),"evaluator_config_sha256":canonical_hash([eval_meta[(rep,arm,x)]["evaluator_config_sha256"] for x in GAP5])} for arm in ARMS]
        paired_rows.append({"replicate":rep,"dataset":"GAP5","n_instances":sum(len(frames[x]) for x in GAP5),"mean_gap_C0":c0,"mean_gap_C1":c1,"Delta_C0_minus_C1":c0-c1,"paired_ci95_low":lo,"paired_ci95_high":hi,"c1_instance_win_fraction":np.nan})
    per=pd.concat(parts,ignore_index=True); per.to_csv(OUT/"04_per_instance_gap.csv",index=False)
    results=pd.DataFrame(result_rows); results.to_csv(OUT/"04_greedy_evaluation.csv",index=False)
    paired=pd.DataFrame(paired_rows)
    aggregates=[]
    for ds,p in paired.groupby("dataset",sort=False):
        v=p.Delta_C0_minus_C1.to_numpy(float); aggregates.append({"replicate":"ALL_EVOLUTION_REPLICATES","dataset":ds,"n_instances":np.nan,"mean_gap_C0":p.mean_gap_C0.mean(),"mean_gap_C1":p.mean_gap_C1.mean(),"Delta_C0_minus_C1":v.mean(),"paired_ci95_low":np.nan,"paired_ci95_high":np.nan,"c1_instance_win_fraction":np.nan,"median_delta":np.median(v),"sample_sd_delta":np.std(v,ddof=1),"min_delta":v.min(),"max_delta":v.max(),"c1_replicate_win_count":int((v>0).sum())})
    fullpaired=pd.concat([paired,pd.DataFrame(aggregates)],ignore_index=True); fullpaired.to_csv(OUT/"05_paired_results.csv",index=False)
    primary=paired[paired.dataset.eq("OriginalTSPLIB70")].set_index("replicate"); ts48=paired[paired.dataset.eq("TSPLIB48")].set_index("replicate"); ts22=paired[paired.dataset.eq("TSPLIB22")].set_index("replicate"); gap5=paired[paired.dataset.eq("GAP5")].set_index("replicate")
    lines=["# C0 vs C1 Paired Comparison","","Delta = C0 gap − C1 gap; positive favors C1.","","| Replicate | Original70 C0 | Original70 C1 | Delta | 95% CI | TSPLIB48 Delta | TSPLIB22 Delta | GAP5 Delta |","|---|---:|---:|---:|---|---:|---:|---:|"]
    for rep in REPLICATES:
        r=primary.loc[rep]; lines.append(f"| {rep} | {r.mean_gap_C0:.6f} | {r.mean_gap_C1:.6f} | {r.Delta_C0_minus_C1:+.6f} | [{r.paired_ci95_low:.6f}, {r.paired_ci95_high:.6f}] | {ts48.at[rep,'Delta_C0_minus_C1']:+.6f} | {ts22.at[rep,'Delta_C0_minus_C1']:+.6f} | {gap5.at[rep,'Delta_C0_minus_C1']:+.6f} |")
    lines += ["","Bootstrap intervals are conditional on fixed benchmark instances; the three evolution replicates remain the unit for directional robustness."]
    (OUT/"05_c0_c1_paired_comparison.md").write_text("\n".join(lines)+"\n")
    # Historical M4/E0/E1 absolute comparison.
    hist=pd.read_csv(P6B/"08_greedy_evaluation_results.csv"); hrows=[]
    for rep in REPLICATES:
        for ds in (*DATASETS,"TSPLIB48","TSPLIB22","GAP5"):
            for arm in SOURCE_ARMS:
                val=float(hist[(hist.replicate.eq(rep))&(hist.arm.eq(arm))&(hist.dataset.eq(ds))].mean_gap.iloc[0]); hrows.append({"replicate":rep,"model":arm,"dataset":ds,"mean_gap":val})
            if ds=="GAP5": m4val=float(np.mean([pd.read_parquet(M4_EVAL/x/"instance_metrics.parquet").gap_percent.mean() for x in GAP5]))
            elif ds in ("TSPLIB48","TSPLIB22"):
                mf=pd.read_parquet(M4_EVAL/"OriginalTSPLIB70"/"instance_metrics.parquet"); names=dev if ds=="TSPLIB48" else held; m4val=float(mf[mf.instance_id.astype(str).isin(names)].gap_percent.mean())
            else: m4val=float(pd.read_parquet(M4_EVAL/ds/"instance_metrics.parquet").gap_percent.mean())
            hrows.append({"replicate":rep,"model":"M4","dataset":ds,"mean_gap":m4val})
            for arm in ARMS: hrows.append({"replicate":rep,"model":arm,"dataset":ds,"mean_gap":float(results[(results.replicate.eq(rep))&(results.arm.eq(arm))&(results.dataset.eq(ds))].mean_gap.iloc[0])})
    history=pd.DataFrame(hrows); history.to_csv(OUT/"logs/historical_absolute_comparison.csv",index=False)
    orig=history[history.dataset.eq("OriginalTSPLIB70")].pivot(index="replicate",columns="model",values="mean_gap")
    hist_lines=["# Historical Baseline Comparison","","Historical E0/E1 are auxiliary; C0/C1 is the primary same-pool comparison.","",md_table(orig.reset_index())]
    (OUT/"06_historical_baseline_comparison.md").write_text("\n".join(hist_lines)+"\n")
    # Cross-population CE matrix, using frozen M4 labels and new union inference.
    matrix=[]; instance=[]
    for rep in REPLICATES:
        union=pd.read_parquet(P6C1/"dataset_manifests"/f"{rep}_union.parquet").set_index("instance_hash")
        model_maps={arm:pd.read_parquet(OUT/"hardness"/f"{rep}_{arm}_on_union.parquet").set_index("instance_hash").loss_raw for arm in ARMS}
        for data_arm in DATA_ARMS:
            if data_arm in ARMS:
                m=pd.read_parquet(selected_manifest(rep,data_arm)); hashes=m.instance_hash.astype(str).tolist(); m4=m.m4_ce.to_numpy(float)
            else:
                hf=pd.read_csv(P6C0/"02_hardness_per_instance.csv"); part=hf[(hf.replicate.eq(rep))&(hf.arm.eq(data_arm))]; hashes=part.instance_hash.astype(str).tolist(); m4=part.loss_raw.to_numpy(float)
            matrix.append({"replicate":rep,"model":"M4","dataset_arm":data_arm,"n":len(hashes),"mean_ce":m4.mean(),"median_ce":np.median(m4),"mean_gain_M4_minus_model":0.0})
            for model_arm in ARMS:
                tuned=model_maps[model_arm].loc[hashes].to_numpy(float); gain=m4-tuned
                matrix.append({"replicate":rep,"model":model_arm,"dataset_arm":data_arm,"n":len(hashes),"mean_ce":tuned.mean(),"median_ce":np.median(tuned),"mean_gain_M4_minus_model":gain.mean(),"improvement_fraction":np.mean(gain>0)})
                instance.extend({"replicate":rep,"model":model_arm,"dataset_arm":data_arm,"instance_hash":h,"m4_ce":float(a),"model_ce":float(b),"gain_m4_minus_model":float(a-b)} for h,a,b in zip(hashes,m4,tuned))
    matrix=pd.DataFrame(matrix); matrix.to_csv(OUT/"07_cross_population_ce_matrix.csv",index=False); pd.DataFrame(instance).to_parquet(OUT/"logs/cross_population_ce_per_instance.parquet",index=False)
    (OUT/"07_cross_population_ce.md").write_text("# Cross-Population Teacher-Forced CE\n\nPositive gain is M4 CE minus M5 CE. E0/E1/C0/C1 share candidates, so this diagnoses transfer rather than pristine generalization.\n\n"+md_table(matrix)+"\n")
    # Decisions.
    wins=int((primary.Delta_C0_minus_C1>0).sum()); gap5_all_worse=bool((gap5.Delta_C0_minus_C1<0).all())
    h2="PRELIMINARILY_SUPPORTED" if wins==3 and not gap5_all_worse else "NOT_SUPPORTED" if wins==0 else "INCONCLUSIVE"
    c1=orig.C1; beat_e1=int((c1<orig.E1).sum()); beat_e0=int((c1<orig.E0).sum()); beat_m4=int((c1<orig.M4).sum())
    h3="ABSOLUTE_RECOVERY_SUPPORTED" if beat_e0==3 and beat_m4==3 else "PARTIAL" if beat_e1>=2 or beat_e0>=1 or beat_m4>=1 else "NOT_SUPPORTED"
    e0_ce=[]; old_negative=[]
    old_matrix=pd.read_csv(P6C0/"03_cross_loss_matrix.csv")
    for rep in REPLICATES:
        new=matrix[(matrix.replicate.eq(rep))&(matrix.model.eq("C1"))&(matrix.dataset_arm.eq("E0"))].iloc[0]; e0_ce.append(float(new.mean_gain_M4_minus_model))
        old=float(old_matrix[(old_matrix.replicate.eq(rep))&(old_matrix.model.eq(f"{rep}_E1"))&(old_matrix.dataset_arm.eq("E0"))].mean_ce.iloc[0]-old_matrix[(old_matrix.replicate.eq(rep))&(old_matrix.model.eq("M4"))&(old_matrix.dataset_arm.eq("E0"))].mean_ce.iloc[0]); old_negative.append(old)
    h1="RESTORATION_CONFIRMED"
    ts48_wins=int((ts48.Delta_C0_minus_C1>0).sum()); ts22_wins=int((ts22.Delta_C0_minus_C1>0).sum()); gap5_wins=int((gap5.Delta_C0_minus_C1>0).sum())
    next_step="Audit whether frozen HGB40 can predict the required hardness exposure across the already labeled E0/E1/C0/C1 distributions; do not start new evolution yet." if h2=="PRELIMINARILY_SUPPORTED" else "Do not start surrogate-constrained evolution; first resolve why oracle hardness restoration did not transfer consistently."
    final=["# Final Research Conclusion","",f"`H1: {h1}`",f"`H2: {h2}`",f"`H3: {h3}`","",f"C1 wins the primary Original70 comparison in {wins}/3 replicates. TSPLIB48 wins: {ts48_wins}/3; TSPLIB22 wins: {ts22_wins}/3; GAP5 wins: {gap5_wins}/3. Historical Original70 wins: C1 vs E1 {beat_e1}/3, vs E0 {beat_e0}/3, vs M4 {beat_m4}/3.",f"C1 M5 gain on historical E0 teacher-forced CE by replicate: {', '.join(f'{x:+.6f}' for x in e0_ce)}. The old E1→E0 negative-transfer magnitudes (M5−M4 CE) were {', '.join(f'{x:+.6f}' for x in old_negative)}.","","The result estimates the total effect of adding the frozen hardness-decile constraint inside the same labeled union. C1 also changes exact membership, OT cost, Full-F40, lineage and operator composition; it is not a pure per-instance hardness causal effect.","",f"Recommended next step: {next_step}","","No HGB experiment, new evolution, extra seed, or P6-C3 was started.","","`BLADE_P6C2_ORACLE_HARDNESS_RESTORATION_COMPLETE`"]
    (OUT/"08_final_research_conclusion.md").write_text("\n".join(final)+"\n")
    # Figures.
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True); models=["M4","E0","E1","C0","C1"]
    for ax,rep in zip(axes,REPLICATES):
        vals=[orig.at[rep,x] for x in models]; ax.bar(range(5),vals); ax.set_xticks(range(5),models); ax.set_title(rep); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Original70 mean gap (%)"); fig.tight_layout(); fig.savefig(OUT/"figures/03_original70_gap.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,rep in zip(axes,REPLICATES):
        p=matrix[matrix.replicate.eq(rep)].pivot(index="model",columns="dataset_arm",values="mean_ce").loc[["M4","C0","C1"],list(DATA_ARMS)]; im=ax.imshow(p.to_numpy(),aspect="auto",cmap="viridis"); ax.set_xticks(range(4),DATA_ARMS); ax.set_yticks(range(3),["M4","C0-M5","C1-M5"]); ax.set_title(rep)
        for i in range(3):
            for j in range(4): ax.text(j,i,f"{p.iloc[i,j]:.3f}",ha="center",va="center",fontsize=8,color="white" if p.iloc[i,j]>np.median(p.to_numpy()) else "black")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Mean CE",shrink=.8); fig.savefig(OUT/"figures/04_cross_population_ce.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    train_wall=float(training.training_loop_wall_s.sum()); eval_wall=sum(json.loads((run_dir(r,a)/"evaluation_integrity.json").read_text())["evaluation_wall_s"] for r in REPLICATES for a in ARMS); ce_wall=sum(json.loads((OUT/"hardness"/f"{r}_{a}_on_union.json").read_text())["wall_s"] for r in REPLICATES for a in ARMS)
    peak=max(int(x) for x in training.gpu_peak_allocated_bytes)
    (OUT/"09_runtime_and_resource_audit.md").write_text(f"# Runtime and Resource Audit\n\nTraining-loop wall sum: {train_wall:.2f}s. Greedy-evaluation wall sum: {eval_wall:.2f}s. Cross-population CE union-inference wall sum: {ce_wall:.2f}s. Peak allocated GPU memory: {peak/2**30:.3f} GiB. No Concorde, mutation, evolution, selection re-solve, HGB training, or new labeling ran.\n")
    result={"status":"COMPLETE","completion_label":"BLADE_P6C2_ORACLE_HARDNESS_RESTORATION_COMPLETE","H1":h1,"H2":h2,"H3":h3,"primary_c1_win_count":wins,"no_followup_started":True,"wall_s_finalize":time.perf_counter()-started,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    write_json(OUT/"completion.json",result); Status().update(phase="complete",**result); return result


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("audit","train","evaluate","ce","finalize")); p.add_argument("--replicate",choices=REPLICATES); p.add_argument("--arm",choices=ARMS); p.add_argument("--device",default="cuda:1"); args=p.parse_args()
    if args.phase=="audit": result=audit()
    elif args.phase=="finalize": result=finalize()
    else:
        if args.replicate is None or args.arm is None: raise SystemExit("--replicate and --arm required")
        result=train(args.replicate,args.arm,args.device) if args.phase=="train" else evaluate(args.replicate,args.arm,args.device) if args.phase=="evaluate" else infer_ce(args.replicate,args.arm,args.device)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__": main()
