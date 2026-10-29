#!/usr/bin/env python3
"""P6-G4 structure-augmented initialization training-value experiment.

The runner consumes the six frozen P6-G3 G1 RAW offspring pools.  It applies
the byte-equivalent formal M4-HGB40 + QD4/CVT-1000 survivor rule, labels the
six frozen 10k populations with the canonical Concorde pipeline, then uses the
formal M4->M5 trainer and greedy evaluator.  It never mutates, fits HGB, or
uses benchmark outcomes during selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/home/haozou/blade-nco-mainline")
OUT = Path(__file__).resolve().parent
G3 = ROOT / "target_aware_pilot/p6g3_initial_support_intervention"
G2 = ROOT / "target_aware_pilot/p6g2_structural_similarity_validation"
G1 = ROOT / "target_aware_pilot/p6g1_structural_reachability_analysis"
E0 = ROOT / "target_aware_pilot/p6e0_evoreal_structural_representation_audit"
P6B = ROOT / "target_aware_pilot/p6b_top5_distribution_guided_evolution"
H1 = ROOT / "target_aware_pilot/p6h1_joint_evolution"
RUN = ROOT / "runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE"
M4 = RUN / "checkpoints/M4_E256.tsp.best"
M4_EVAL = RUN / "cycles/C03/evaluation_work/M4"
REQUEST = Path("/home/haozou/.codex/attachments/c6a18aff-81b8-424d-9542-3c8eaaf1b12e/pasted-text.txt")

sys.path[:0] = [
    str(ROOT / "src"), str(ROOT / "scripts"), str(H1), str(P6B),
    str(ROOT / "target_aware_pilot/p6a_top5_target_matching_feasibility"),
    str(ROOT / "target_aware_pilot/p6g2_structural_similarity_validation"),
]
import run_p6h1 as h1  # noqa: E402
import run_p6b as p6b  # noqa: E402
from bq.checkpoint import load_canonical_checkpoint  # noqa: E402
from final_run.science import label_with_cache, membership_hash  # noqa: E402
from mutation.result import coordinate_hash  # noqa: E402
from oracle.concorde import CANONICAL_CONCORDE_SHA256, ORACLE_LEDGER_COLUMNS, sha256_file  # noqa: E402

REPLICATES = ("Original", "S1", "S2")
ARMS = ("A", "B")
JOBS = tuple((rep, arm) for rep in REPLICATES for arm in ARMS)
K = 10_000
M4_SHA = "c749311bd2a461084d0cbcfd383a4c0a1fc1268762b72e5f7220952f108f8190"
TRAINING_SEED = 2026
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261031
F40 = tuple(h1.F40)
QD4 = tuple(h1.QD4)
FFT_RULES = E0 / "configs/reference_rules_frozen.json"
REF_FFT = E0 / "features/TSPLIB48.parquet"
REF_MANIFEST = ROOT / "target_aware_pilot/03_tsplib48_reference_manifest.csv"
HELD22 = ROOT / "target_aware_pilot/p1_heldout22_names.json"
FORMAL_TRAINER = ROOT / "scripts/run_blade_core_ablation_5c_noanchor_foga11base_v1.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)): return int(value)
        if isinstance(value, (np.floating,)): return float(value)
        if isinstance(value, (np.bool_,)): return bool(value)
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Path): return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n")
    os.replace(temporary, path)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def md_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)): return json.dumps(value, sort_keys=True, default=str).replace("|", "\\|")
        if pd.isna(value): return "NA"
        if isinstance(value, (float, np.floating)): return f"{float(value):.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    rows += ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


class Status:
    def update(self, **values: Any) -> None:
        path = OUT / "status.json"
        old = json.loads(path.read_text()) if path.is_file() else {}
        write_json(path, {**old, **values, "updated_utc": now()})


def ensure_dirs() -> None:
    for name in ("configs", "features", "datasets", "labels", "checkpoints", "logs", "diagnostics"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def raw_bundle(rep: str, arm: str) -> Path:
    return G3 / "artifacts" / rep / f"{arm}_G1_RAW" / "offspring_coordinates.npz"


def raw_runtime(rep: str, arm: str) -> Path:
    return G3 / "artifacts" / rep / f"{arm}_G1_RAW" / "runtime.json"


def mutation_manifest(rep: str, arm: str) -> Path:
    if arm == "B": return G3 / "artifacts" / rep / "B_G1_RAW" / "mutation_manifest.parquet"
    return P6B / "evolution" / rep / "shared_G1" / "mutation" / "mutation_manifest.parquet"


def parent_manifest(arm: str) -> Path:
    return G3 / "artifacts" / ("A_original_parents.parquet" if arm == "A" else "B_augmented_parents.parquet")


def selected_stem(rep: str, arm: str) -> Path:
    return OUT / "datasets" / f"{rep}_{arm}_structure10k"


def label_dir(rep: str, arm: str) -> Path:
    return OUT / "labels" / f"{rep}_{arm}"


def run_dir(rep: str, arm: str) -> Path:
    return OUT / "logs" / f"{rep}_{arm}"


def checkpoint(rep: str, arm: str) -> Path:
    return OUT / "checkpoints" / f"M5_{arm}_{rep}_E256.tsp.best"


def detail(rep: str, arm: str, dataset: str) -> pd.DataFrame:
    label = f"P6G4_{rep}_{arm}_M5"
    return pd.read_parquet(run_dir(rep, arm) / "evaluation_work" / label / dataset / "instance_metrics.parquet")


def formal_select(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact formal A0 QD-HGB survivor ordering."""
    hashes = frame.instance_hash.astype(str).to_numpy()
    scores = frame.predicted_m4_ce.to_numpy(float)
    niches = frame.niche_id.to_numpy(int)
    work = pd.DataFrame({"index": np.arange(len(frame)), "niche": niches, "score": scores, "hash": hashes})
    elite = work.sort_values(["niche", "score", "hash"], ascending=[True, False, True], kind="stable").drop_duplicates("niche")
    chosen = elite["index"].astype(int).tolist()
    chosen_set = set(chosen)
    fill = [int(index) for index in np.lexsort((hashes, -scores)) if int(index) not in chosen_set][:K - len(chosen)]
    indices = np.asarray(chosen + fill, int)
    selected = frame.iloc[indices].copy().reset_index(drop=True)
    selected["component"] = ["qd_hgb_elite"] * len(chosen) + ["global_hgb_fill"] * len(fill)
    selected["selection_rank"] = np.arange(1, K + 1)
    selected["source_position"] = indices
    if len(selected) != K or selected.instance_hash.nunique() != K:
        raise RuntimeError("formal selection size/uniqueness failure")
    return selected


def formal_select_source_sha() -> str:
    return hashlib.sha256(inspect.getsource(formal_select).encode()).hexdigest()


def protocol() -> dict[str, Any]:
    return {
        "experiment": "BLADE-NCO P6-G4", "request_sha256": sha256_file(REQUEST),
        "g3_completion_sha256": sha256_file(G3 / "completion.json"), "m4_sha256": M4_SHA,
        "candidate_pool": "P6-G3 valid nonidentity unique G1 RAW offspring only; no parents and no E0/E1/H1 pools",
        "selection": "one max M4-HGB40 per occupied frozen QD4/CVT-1000 niche, then global HGB40 fill",
        "tie_break": "score descending then instance_hash ascending", "lineage_cap": False,
        "selection_source_sha256": formal_select_source_sha(), "K": K,
        "replicates": list(REPLICATES), "arms": {"A": "Original P03", "B": "Augmented P03"},
        "training": p6b.base_training_protocol(), "training_seed": TRAINING_SEED,
        "bootstrap_n": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
        "decision_rules": {
            "structure_retained_YES": "B FFT coverage > A in all 3 seeds and median advantage >=3 references",
            "structure_retained_PARTIAL": "B > A in >=2 seeds or median advantage >0",
            "hardness_preserved_YES": "B mean predicted CE >= A in >=2 seeds and B/A grand mean >=0.98",
            "hardness_preserved_PARTIAL": "one of those two conditions holds",
            "stable_gain_SUPPORTED": "B Original70 mean gap < A in all 3 seeds",
            "stable_gain_NOT_SUPPORTED": "B Original70 mean gap >= A in all 3 seeds",
        },
        "prohibitions": ["mutation", "new evolution", "HGB fitting", "structure-aware survivor repair", "benchmark-guided selection", "extra training seed"],
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    }


def preflight() -> dict[str, Any]:
    ensure_dirs()
    cfg = protocol(); cfg["protocol_sha256"] = canonical_hash(cfg)
    frozen = OUT / "configs/frozen_p6g4_config.json"
    if frozen.is_file() and json.loads(frozen.read_text()) != cfg:
        raise RuntimeError("frozen P6-G4 protocol drift")
    if not frozen.is_file(): write_json(frozen, cfg)
    if sha256_file(M4) != M4_SHA: raise RuntimeError("formal M4 drift")
    if json.loads((G3 / "completion.json").read_text()).get("status") != "COMPLETE": raise RuntimeError("P6-G3 incomplete")
    rows = []
    for rep, arm in JOBS:
        xy, hashes, _ = p6b.load_bundle(raw_bundle(rep, arm))
        runtime = json.loads(raw_runtime(rep, arm).read_text())
        if runtime.get("status") != "PASS" or runtime.get("valid_nonidentity_unique_n") != len(hashes): raise RuntimeError(f"RAW runtime mismatch {rep}/{arm}")
        if xy.shape != (len(hashes), 100, 2) or len(set(hashes)) != len(hashes) or not np.isfinite(xy).all(): raise RuntimeError(f"RAW validity {rep}/{arm}")
        if not np.all((xy >= 0) & (xy <= 1)): raise RuntimeError(f"RAW coordinate range {rep}/{arm}")
        for index in np.linspace(0, len(hashes)-1, 128, dtype=int):
            if coordinate_hash(xy[index]) != hashes[index]: raise RuntimeError(f"RAW coordinate hash {rep}/{arm}")
        manifest = pd.read_parquet(mutation_manifest(rep, arm), columns=["attempt_index", "parent_hash", "operator_id", "proposal_status", "valid_after_repair", "within_offspring_duplicate", "child_hash"])
        if len(manifest) != 110_000: raise RuntimeError(f"mutation attempts {rep}/{arm}")
        rows.append({"replicate": rep, "arm": arm, "raw_n": len(hashes), "raw_unique_n": len(set(hashes)), "mutation_attempts": len(manifest),
                     "bundle_sha256": sha256_file(raw_bundle(rep, arm)), "runtime_sha256": sha256_file(raw_runtime(rep, arm)),
                     "mutation_manifest_sha256": sha256_file(mutation_manifest(rep, arm)), "parent_manifest_sha256": sha256_file(parent_manifest(arm))})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "diagnostics/preflight_inputs.csv", index=False)
    hgb_manifest = RUN / "cycles/C04/HGB_model_manifest.json"
    assets = pd.DataFrame([
        {"asset": "M4", "path": str(M4), "sha256": sha256_file(M4)},
        {"asset": "HGB manifest", "path": str(hgb_manifest), "sha256": sha256_file(hgb_manifest)},
        {"asset": "QD normalization", "path": str(ROOT/'assets/qd/normalization/normalization_A_P1P99.json'), "sha256": sha256_file(ROOT/'assets/qd/normalization/normalization_A_P1P99.json')},
        {"asset": "CVT1000", "path": str(ROOT/'assets/qd/cvt/centroids.csv'), "sha256": sha256_file(ROOT/'assets/qd/cvt/centroids.csv')},
        {"asset": "formal trainer", "path": str(FORMAL_TRAINER), "sha256": sha256_file(FORMAL_TRAINER)},
    ])
    assets.to_csv(OUT / "diagnostics/frozen_assets.csv", index=False)
    (OUT / "01_preflight.md").write_text(
        "# P6-G4 Preflight\n\n状态：**PASS**。六组输入均为P6-G3冻结的G1 RAW，实际各有10,000父代×11算子=110,000 attempts；"
        "bundle只含有效、非identity、unique offspring。候选池不含父代，也不使用P6-B E0/E1或P6-H1。正式M4、C04 M4-conditioned HGB40、"
        "QD4归一化、CVT-1000及trainer均已哈希冻结。正式选择无lineage cap。\n\n## RAW输入\n\n" + md_table(frame) +
        "\n\n## 冻结资产\n\n" + md_table(assets) + f"\n\nProtocol SHA256: `{cfg['protocol_sha256']}`。\n")
    Status().update(phase="preflight", status="PASS")
    return cfg


def load_all_f40(required: set[str]) -> pd.DataFrame:
    columns = ["instance_hash", *F40]
    formal = pd.read_parquet(G1 / "cache/formal_f40_all.parquet", columns=columns)
    parts = [formal[formal.instance_hash.astype(str).isin(required)]]
    for path in sorted((G3 / "f40_cache").glob("f40_*.parquet")):
        parts.append(pd.read_parquet(path, columns=columns))
    frame = pd.concat(parts, ignore_index=True).drop_duplicates("instance_hash", keep="last")
    frame["instance_hash"] = frame.instance_hash.astype(str)
    frame = frame[frame.instance_hash.isin(required)].set_index("instance_hash", drop=False)
    missing = required - set(frame.index)
    if missing: raise RuntimeError(f"P6-G3 F40 cache missing {len(missing)} required RAW hashes")
    return frame


def raw_metadata(rep: str, arm: str, hashes: list[str]) -> pd.DataFrame:
    manifest = pd.read_parquet(mutation_manifest(rep, arm))
    valid = manifest.proposal_status.eq("accepted") & manifest.valid_after_repair.astype(bool) & ~manifest.within_offspring_duplicate.astype(bool)
    manifest = manifest.loc[valid].copy()
    manifest["child_hash"] = manifest.child_hash.astype(str)
    manifest = manifest[manifest.child_hash.ne(manifest.parent_hash.astype(str))].drop_duplicates("child_hash", keep="first").set_index("child_hash", drop=False)
    missing = set(hashes) - set(manifest.index)
    if missing: raise RuntimeError(f"manifest does not explain {len(missing)} RAW hashes: {rep}/{arm}")
    selected = manifest.loc[hashes].reset_index(drop=True)
    parents = pd.read_parquet(parent_manifest(arm)).copy()
    parents["instance_hash"] = parents.instance_hash.astype(str)
    psource = dict(zip(parents.instance_hash, parents.parent_source.fillna("P03_original").astype(str)))
    generator = dict(zip(parents.instance_hash, parents.generator_type.fillna("P03").astype(str)))
    roots = dict(zip(parents.instance_hash, parents.lineage_root.fillna(parents.instance_hash).astype(str)))
    out = pd.DataFrame({
        "instance_hash": hashes, "parent_hash": selected.parent_hash.astype(str), "operator_id": selected.operator_id.astype(str),
        "attempt_index": selected.attempt_index.astype(int), "parent_source": selected.parent_hash.astype(str).map(psource),
        "generator_type": selected.parent_hash.astype(str).map(generator), "lineage_root": selected.parent_hash.astype(str).map(roots),
    })
    if out[["parent_source", "generator_type", "lineage_root"]].isna().any().any(): raise RuntimeError(f"lineage/provenance missing {rep}/{arm}")
    return out


def fft_features(coordinates: np.ndarray) -> np.ndarray:
    from evoreal_structure_alignment_audit_corrected_v2 import fft_energy, nn_ratio_stats
    values = np.empty((len(coordinates), 2), np.float64)
    for index, xy in enumerate(coordinates):
        values[index] = (fft_energy(xy), nn_ratio_stats(xy)["nn_ratio"])
    return values


def structure_diagnostics(rep: str, arm: str, xy: np.ndarray, selected: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    rules = json.loads(FFT_RULES.read_text())
    refs = pd.read_csv(REF_MANIFEST)[["instance_id", "n_nodes", "source_file"]].merge(
        pd.read_parquet(REF_FFT)[["instance_id", "fft_energy", "nn_ratio"]], on="instance_id", validate="one_to_one")
    fft = fft_features(xy)
    candidate = (fft - np.asarray(rules["center"])) / np.asarray(rules["scale_iqr"])
    reference = (refs[["fft_energy", "nn_ratio"]].to_numpy(float) - np.asarray(rules["center"])) / np.asarray(rules["scale_iqr"])
    from scipy.spatial.distance import cdist
    fft_dist = cdist(candidate, reference)
    radius = float(rules["radius"])
    coverage = int(np.sum(fft_dist.min(axis=0) <= radius))
    config = h1.load_distance_config(); _rm, ref_values = h1.reference_f40(); ref_z, eff = h1.standardized_references(config, ref_values)
    f40_dist, nearest = h1.min_f40_distance(features[list(F40)].to_numpy(float), config, ref_z, eff)
    reference_best = []
    matrix = features[list(F40)].to_numpy(float)
    for index in range(len(refs)):
        distance, _ = h1.min_f40_distance(matrix, config, ref_z[index:index+1], eff)
        reference_best.append(float(distance.min()))
    lineage_p = selected.lineage_root.astype(str).value_counts(normalize=True).to_numpy(float)
    niche_p = selected.niche_id.astype(int).value_counts(normalize=True).to_numpy(float)
    supplement = selected.parent_source.eq("supplement")
    return {
        "replicate": rep, "arm": arm, "n": len(selected), "unique_n": selected.instance_hash.nunique(),
        "mean_predicted_m4_ce": float(selected.predicted_m4_ce.mean()), "median_predicted_m4_ce": float(selected.predicted_m4_ce.median()),
        "p90_predicted_m4_ce": float(selected.predicted_m4_ce.quantile(.9)), "occupied_qd_niches": int(selected.niche_id.nunique()),
        "effective_qd_niches": float(1 / np.sum(niche_p*niche_p)), "effective_lineages": float(np.exp(-(lineage_p*np.log(lineage_p)).sum())),
        "largest_lineage_share": float(lineage_p.max()), "supplement_parent_offspring_n": int(supplement.sum()),
        "supplement_parent_offspring_fraction": float(supplement.mean()), "p03_parent_offspring_n": int((~supplement).sum()),
        "fft_nn_coverage48": coverage, "fft_nn_candidate_inside_fraction": float(np.mean(fft_dist.min(axis=1) <= radius)),
        "f40_candidate_nearest_median": float(np.median(f40_dist)), "f40_reference_best_mean": float(np.mean(reference_best)),
        "f40_reference_best_p90": float(np.quantile(reference_best, .9)),
    }


def select_all() -> pd.DataFrame:
    preflight(); Status().update(phase="selection")
    pools: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}; required: set[str] = set()
    for rep, arm in JOBS:
        xy, hashes, _ = p6b.load_bundle(raw_bundle(rep, arm)); pools[(rep, arm)] = (xy, hashes); required.update(hashes)
    all_f40 = load_all_f40(required); models = h1.hgb_models()
    summaries = []; manifests = []
    external_hashes = set()
    for dataset in p6b.DATASETS:
        detail_path = M4_EVAL / dataset / "instance_metrics.parquet"
        if detail_path.is_file(): external_hashes.update(pd.read_parquet(detail_path, columns=["coordinate_hash"]).coordinate_hash.astype(str))
    for rep, arm in JOBS:
        xy, hashes = pools[(rep, arm)]; meta = raw_metadata(rep, arm, hashes)
        features = all_f40.loc[hashes].reset_index(drop=True)
        scored = meta.copy(); scored["predicted_m4_ce"] = h1.predict_hardness(models, features); scored["niche_id"] = h1.niche_from_features(features)
        for name in QD4: scored[name] = features[name].to_numpy(float)
        chosen = formal_select(scored); positions = chosen.source_position.to_numpy(int); selected_xy = xy[positions]
        selected_hashes = chosen.instance_hash.astype(str).tolist(); selected_features = features.iloc[positions].reset_index(drop=True)
        if set(selected_hashes) & external_hashes: raise RuntimeError(f"training/evaluation exact coordinate overlap {rep}/{arm}")
        if [coordinate_hash(row) for row in selected_xy] != selected_hashes: raise RuntimeError(f"selected coordinate alignment {rep}/{arm}")
        p6b.save_bundle(selected_stem(rep, arm).with_suffix(".npz"), selected_xy, selected_hashes)
        chosen.insert(0, "arm", arm); chosen.insert(0, "replicate", rep)
        chosen.to_parquet(selected_stem(rep, arm).with_suffix(".parquet"), index=False)
        selected_features.insert(0, "instance_hash", selected_hashes) if "instance_hash" not in selected_features else None
        selected_features.to_parquet(OUT / "features" / f"{rep}_{arm}_selected_f40.parquet", index=False)
        summary = structure_diagnostics(rep, arm, selected_xy, chosen, selected_features)
        summary.update({"raw_n": len(hashes), "elite_n": int(chosen.component.eq("qd_hgb_elite").sum()),
                        "global_fill_n": int(chosen.component.eq("global_hgb_fill").sum()), "membership_sha256": membership_hash(selected_hashes),
                        "dataset_sha256": sha256_file(selected_stem(rep, arm).with_suffix(".npz")), "selection_protocol_match": True})
        summaries.append(summary); manifests.append(chosen)
        print(f"[P6-G4] selected {rep}/{arm}: coverage={summary['fft_nn_coverage48']} supplement={summary['supplement_parent_offspring_n']}", flush=True)
    manifest = pd.concat(manifests, ignore_index=True); manifest.to_csv(OUT / "02_selection_manifest.csv", index=False)
    comparison = pd.DataFrame(summaries); comparison.to_csv(OUT / "03_population_comparison.csv", index=False)
    write_json(OUT / "diagnostics/selection_integrity.json", {"status": "PASS", "datasets": len(summaries), "rows": len(manifest),
               "formal_rule_source_sha256": formal_select_source_sha(), "m4_hgb_models": {str(seed): sha256_file(h1.HGB_DIR/f'HGB40_seed_{seed}.joblib') for seed in h1.HGB_SEEDS}})
    Status().update(phase="selection", status="PASS")
    return comparison


def label_all() -> dict[str, Any]:
    if not (OUT / "03_population_comparison.csv").is_file(): select_all()
    prior = p6b.verified_label_cache(); known = set(prior.instance_hash.astype(str))
    coordinate_map: dict[str, np.ndarray] = {}; order: list[str] = []
    for rep, arm in JOBS:
        xy, hashes, _ = p6b.load_bundle(selected_stem(rep, arm).with_suffix(".npz"))
        for value, row in zip(hashes, xy, strict=True):
            if value not in coordinate_map: coordinate_map[value] = row; order.append(value)
            elif not np.array_equal(coordinate_map[value], row): raise RuntimeError(f"coordinate collision {value}")
    union_xy = np.stack([coordinate_map[value] for value in order]); missing_n = sum(value not in known for value in order)
    directory = OUT / "labels/global_unique_union"; integrity = directory / "labeling_integrity.json"; started = time.perf_counter()
    if integrity.is_file():
        info = json.loads(integrity.read_text()); ledger = pd.read_parquet(directory / "ledger.parquet")
        if info.get("union_membership_sha256") != membership_hash(order): raise RuntimeError("label union drift")
    else:
        Status().update(phase="label", unique_n=len(order), missing_n=missing_n)
        ledger = label_with_cache(union_xy, order, directory, prior); ledger.to_parquet(directory / "ledger.parquet", index=False)
        info = {"status": "PASS", "total_memberships": 60_000, "unique_union_n": len(order), "reused_labels": len(order)-missing_n,
                "physical_new_concorde_calls": missing_n, "failures": 0, "pipeline_wall_seconds": time.perf_counter()-started,
                "union_membership_sha256": membership_hash(order), "ledger_sha256": sha256_file(directory/"ledger.parquet"),
                "concorde_sha256": CANONICAL_CONCORDE_SHA256}
        write_json(integrity, info)
    lookup = ledger.set_index("instance_hash", drop=False); rows = []
    for rep, arm in JOBS:
        xy, hashes, _ = p6b.load_bundle(selected_stem(rep, arm).with_suffix(".npz")); selected = lookup.loc[hashes].reset_index(drop=True)
        tours = p6b.validate_ledger(xy, hashes, selected[list(ORACLE_LEDGER_COLUMNS)])
        directory = label_dir(rep, arm); directory.mkdir(parents=True, exist_ok=True); selected.to_parquet(directory/"ledger.parquet", index=False)
        p6b.save_bundle(directory/"labeled_bundle.npz", xy, hashes, tours)
        record = {"status": "PASS", "replicate": rep, "arm": arm, "verified": K, "cache_reuse_n": sum(value in known for value in hashes),
                  "missing_before_global_solve_n": sum(value not in known for value in hashes), "failures": 0,
                  "membership_sha256": membership_hash(hashes), "labeled_bundle_sha256": sha256_file(directory/"labeled_bundle.npz"),
                  "all_tours_legal": True, "all_costs_reproduced": True, "concorde_sha256": CANONICAL_CONCORDE_SHA256}
        write_json(directory/"labeling_integrity.json", record); rows.append(record)
    lines = ["# Concorde Labeling Audit", "", md_table(pd.DataFrame(rows)), "",
             f"六组共60,000 memberships；global unique={len(order):,}；复用={len(order)-missing_n:,}；实际新Concorde调用={missing_n:,}；失败=0。",
             f"标注wall-clock={float(info['pipeline_wall_seconds']):.3f}s；canonical Concorde SHA256=`{CANONICAL_CONCORDE_SHA256}`。",
             "全部tour均为合法0-based TSP100 Hamiltonian permutation，整数目标与训练闭环约定已由正式validator复核；没有替换实例。"]
    (OUT / "04_concorde_label_audit.md").write_text("\n".join(lines) + "\n")
    Status().update(phase="label", status="PASS"); return info


def train_one(rep: str, arm: str, device_name: str) -> dict[str, Any]:
    import torch
    integrity = run_dir(rep, arm) / "training_integrity.json"; cp = checkpoint(rep, arm)
    if integrity.is_file():
        old = json.loads(integrity.read_text())
        if old.get("status") == "PASS" and cp.is_file() and sha256_file(cp) == old.get("checkpoint_sha256"): return old
    if json.loads((label_dir(rep, arm)/"labeling_integrity.json").read_text()).get("verified") != K: raise RuntimeError("label gate")
    xy, hashes, tours = p6b.load_bundle(label_dir(rep, arm)/"labeled_bundle.npz", True); assert tours is not None
    frozen_xy, frozen_hashes, _ = p6b.load_bundle(selected_stem(rep, arm).with_suffix(".npz"))
    if hashes != frozen_hashes or not np.array_equal(xy, frozen_xy): raise RuntimeError("training membership drift")
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    device = torch.device(device_name)
    if device.type == "cuda": torch.cuda.set_device(device)
    directory = run_dir(rep, arm)/"training_cycle"; directory.mkdir(parents=True, exist_ok=True)
    core.RUN_ID = f"TARGET_AWARE_P6G4_{rep}_{arm}"; core.cycle_dir = lambda selected_arm, cycle: directory; core.model_path = lambda selected_arm, index: cp if index == 5 else M4
    model, meta0 = load_canonical_checkpoint(M4, device, expected_sha256=M4_SHA)
    Status().update(phase="training", current_run=f"{rep}-{arm}", device=device_name); started = time.perf_counter()
    trained, meta, timing = core.train(f"P6G4_{rep}_{arm}", 4, model, meta0["sha256"], xy, hashes, tours, device, Status(), True)
    stage = directory/"09_bq_training"; e000 = torch.load(stage/"M5_E000.tsp.best", map_location="cpu", weights_only=False); m4 = torch.load(M4, map_location="cpu", weights_only=False)
    equal = set(e000["net"]) == set(m4["net"]) and all(torch.equal(e000["net"][key], m4["net"][key]) for key in m4["net"])
    curve = pd.read_csv(stage/"training_loss_curve.csv")
    if not equal or len(curve) != 256 or int(curve.optimizer_updates.sum()) != 10_240 or not curve.hard_instances_seen.eq(K).all() or meta["sha256"] != sha256_file(cp): raise RuntimeError("training integrity")
    result = {"status": "PASS", "replicate": rep, "arm": arm, "training_seed": TRAINING_SEED, "m4_sha256": M4_SHA,
              "e000_weights_equal_m4": True, "epochs": 256, "optimizer_steps": int(timing["optimizer_steps"]), "final_loss": float(timing["final_loss"]),
              "training_loop_wall_s": float(timing["training_loop_wall_s"]), "invocation_wall_s": time.perf_counter()-started,
              "gpu_peak_allocated_bytes": int(timing.get("gpu_peak_allocated_bytes") or 0), "checkpoint_sha256": meta["sha256"], "checkpoint_path": str(cp),
              "membership_sha256": membership_hash(hashes), "label_bundle_sha256": sha256_file(label_dir(rep, arm)/"labeled_bundle.npz"),
              "loss_curve_sha256": sha256_file(stage/"training_loss_curve.csv")}
    write_json(integrity, result); del trained, model, xy, tours; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    training_manifest(); return result


def training_manifest() -> pd.DataFrame:
    rows = []
    for rep, arm in JOBS:
        path = run_dir(rep, arm)/"training_integrity.json"
        rows.append(json.loads(path.read_text()) if path.is_file() else {"replicate": rep, "arm": arm, "status": "PENDING", "training_seed": TRAINING_SEED})
    frame = pd.DataFrame(rows); frame.to_csv(OUT/"05_training_runs.csv", index=False); return frame


def evaluate_one(rep: str, arm: str, device_name: str) -> dict[str, Any]:
    import torch
    integrity = run_dir(rep, arm)/"evaluation_integrity.json"; output = run_dir(rep, arm)/"external_benchmark_results.csv"
    training = json.loads((run_dir(rep, arm)/"training_integrity.json").read_text()); cp = checkpoint(rep, arm)
    if integrity.is_file() and output.is_file():
        old = json.loads(integrity.read_text())
        if old.get("status") == "PASS" and old.get("checkpoint_sha256") == training["checkpoint_sha256"]: return old
    device = torch.device(device_name); model, meta = load_canonical_checkpoint(cp, device, expected_sha256=training["checkpoint_sha256"])
    import run_blade_core_ablation_5c_noanchor_foga11base_v1 as core
    label = f"P6G4_{rep}_{arm}_M5"; Status().update(phase="evaluation", current_run=f"{rep}-{arm}")
    frame, wall = core.evaluate(model, cp, label, output, Status(), f"P6G4_{rep}_{arm}")
    historical = pd.read_csv(P6B/"logs"/f"{rep}_E0"/"external_benchmark_results.csv").set_index("dataset"); current = frame.set_index("dataset")
    for dataset in p6b.DATASETS:
        if current.at[dataset,"dataset_sha256"] != historical.at[dataset,"dataset_sha256"] or current.at[dataset,"evaluator_config_sha256"] != historical.at[dataset,"evaluator_config_sha256"]: raise RuntimeError(f"evaluation drift {dataset}")
    result = {"status": "PASS", "replicate": rep, "arm": arm, "checkpoint_sha256": meta["sha256"], "evaluation_wall_s": wall, "results_sha256": sha256_file(output)}
    write_json(integrity, result); del model; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None; return result


def bootstrap(values: np.ndarray, key: str) -> tuple[float, float]:
    x = np.asarray(values, float); rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"P6G4|{BOOTSTRAP_SEED}|{key}".encode()).digest()[:8], "little")); samples = np.empty(BOOTSTRAP_N)
    for index in range(BOOTSTRAP_N): samples[index] = x[rng.integers(0, len(x), len(x))].mean()
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def finalize() -> dict[str, Any]:
    training = training_manifest()
    if len(training) != 6 or not training.status.eq("PASS").all(): raise RuntimeError("training incomplete")
    for rep, arm in JOBS:
        if json.loads((run_dir(rep, arm)/"evaluation_integrity.json").read_text()).get("status") != "PASS": raise RuntimeError("evaluation incomplete")
    dev = set(pd.read_csv(REF_MANIFEST).instance_id.astype(str)); held = set(json.loads(HELD22.read_text())["names"])
    evaluation_rows = []; pair_rows = []; per_rows = []
    for rep in REPLICATES:
        frames = {}
        for dataset in p6b.DATASETS:
            a = detail(rep, "A", dataset)[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].rename(columns={"gap_percent":"gap_A","valid_tour":"valid_A"})
            b = detail(rep, "B", dataset)[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_percent","valid_tour"]].rename(columns={"gap_percent":"gap_B","valid_tour":"valid_B"})
            merged = a.merge(b, on=["instance_id","coordinate_hash","n_nodes","optimal_cost_int"], validate="one_to_one")
            m4 = pd.read_parquet(M4_EVAL/dataset/"instance_metrics.parquet")[["instance_id","coordinate_hash","gap_percent"]].rename(columns={"gap_percent":"gap_M4"})
            merged = merged.merge(m4, on=["instance_id","coordinate_hash"], validate="one_to_one")
            if not merged.valid_A.all() or not merged.valid_B.all(): raise RuntimeError("invalid greedy tour")
            frame = merged[["instance_id","coordinate_hash","n_nodes","optimal_cost_int","gap_M4","gap_A","gap_B"]].copy(); frames[dataset] = frame
        original = frames["OriginalTSPLIB70"]
        for name, names in (("TSPLIB48", dev), ("TSPLIB22", held)):
            part = original[original.instance_id.astype(str).isin(names)].copy()
            if len(part) != len(names): raise RuntimeError(f"split mismatch {name}")
            frames[name] = part
        for dataset, frame in frames.items():
            for model in ("M4","A","B"):
                evaluation_rows.append({"replicate": rep, "model": model, "dataset": dataset, "n_instances": len(frame), "mean_gap": float(frame[f'gap_{model}'].mean()),
                                        "checkpoint_sha256": M4_SHA if model == "M4" else json.loads((run_dir(rep,model)/"training_integrity.json").read_text())["checkpoint_sha256"]})
            for comparison, left in (("B_vs_A","A"),("B_vs_M4","M4"),("A_vs_M4","M4")):
                right = "B" if comparison.startswith("B_") else "A"; delta = frame[f"gap_{left}"].to_numpy(float)-frame[f"gap_{right}"].to_numpy(float); low, high = bootstrap(delta, f"{rep}|{dataset}|{comparison}")
                pair_rows.append({"replicate": rep, "comparison": comparison, "dataset": dataset, "n": len(delta), "mean_delta_comparator_minus_target": float(delta.mean()),
                                  "median_delta": float(np.median(delta)), "paired_ci95_low": low, "paired_ci95_high": high, "target_win_fraction": float(np.mean(delta>0))})
        frame = original.copy(); frame.insert(0,"replicate",rep); frame["B_minus_A"] = frame.gap_B-frame.gap_A; frame["B_minus_M4"] = frame.gap_B-frame.gap_M4; per_rows.append(frame)
    evaluations = pd.DataFrame(evaluation_rows); evaluations.to_csv(OUT/"06_external_evaluation.csv", index=False)
    pairs = pd.DataFrame(pair_rows); pd.concat(per_rows, ignore_index=True).to_csv(OUT/"diagnostics/original70_per_instance.csv", index=False)
    primary = pairs[(pairs.dataset == "OriginalTSPLIB70") & pairs.comparison.isin(["B_vs_A","B_vs_M4","A_vs_M4"])]
    (OUT/"07_original70_paired_analysis.md").write_text("# Original70 Paired Analysis\n\n正delta表示目标模型更好（gap更低）。逐实例bootstrap只反映固定70实例的不确定性，不能替代三个训练/evolution seed的不确定性。\n\n"+md_table(primary)+"\n")
    population = pd.read_csv(OUT/"03_population_comparison.csv"); pivot = evaluations[evaluations.dataset.eq("OriginalTSPLIB70")].pivot(index="replicate",columns="model",values="mean_gap")
    b_wins_a = int((pivot.B < pivot.A).sum()); b_wins_m4 = int((pivot.B < pivot.M4).sum())
    b_vs_a = "BETTER" if b_wins_a == 3 else "WORSE" if b_wins_a == 0 else "MIXED"; b_vs_m4 = "BETTER" if b_wins_m4 == 3 else "WORSE" if b_wins_m4 == 0 else "MIXED"
    structure_delta = population.pivot(index="replicate",columns="arm",values="fft_nn_coverage48"); sd = structure_delta.B-structure_delta.A
    retained = "YES" if (sd>0).all() and float(sd.median()) >= 3 else "PARTIAL" if int((sd>0).sum()) >= 2 or float(sd.median()) > 0 else "NO"
    hard = population.pivot(index="replicate",columns="arm",values="mean_predicted_m4_ce"); conditions = int((hard.B>=hard.A).sum()) >= 2; ratio = float(hard.B.mean()/hard.A.mean())
    hardness_preserved = "YES" if conditions and ratio >= .98 else "PARTIAL" if conditions or ratio >= .98 else "NO"
    stable = "SUPPORTED" if b_wins_a == 3 else "NOT_SUPPORTED" if b_wins_a == 0 else "INCONCLUSIVE"
    labels = {"P6_G4_DATA_INTEGRITY":"PASS","P6_G4_SELECTION_PROTOCOL_MATCH":"PASS","P6_G4_AUGMENTED_STRUCTURE_RETAINED":retained,
              "P6_G4_HARDNESS_PRESERVED":hardness_preserved,"P6_G4_B_VS_A_ORIGINAL70":b_vs_a,"P6_G4_B_VS_M4_ORIGINAL70":b_vs_m4,
              "P6_G4_STABLE_TRAINING_GAIN":stable,"P6_G4_STATUS":"COMPLETE"}
    table = pivot.reset_index(); table["B-A"] = table.B-table.A
    conclusion = ["# Final Research Conclusion","",md_table(table),"",f"最终10k的B−A FFT–NN覆盖差：{dict(sd)}；结构保留判定 **{retained}**。",
                  f"B/A HGB predicted CE总体均值比={ratio:.6f}；hardness判定 **{hardness_preserved}**。",
                  f"Original70中B优于A {b_wins_a}/3 seed，B优于M4 {b_wins_m4}/3 seed；稳定训练增益 **{stable}**。","",
                  "TSPLIB48是target-informed；TSPLIB22仅为事后切分，不重新声称独立holdout。结构、hardness与gap之间仅报告跨六个冻结population的描述性关系，不作实例级因果解释。","",
                  *[f"`{key} = {value}`" for key,value in labels.items()],"","`BLADE_P6G4_AUGMENTED_INITIALIZATION_FINETUNE_COMPLETE`"]
    (OUT/"08_final_conclusion.md").write_text("\n".join(conclusion)+"\n")
    result = {"status":"COMPLETE","labels":labels,"b_wins_a":b_wins_a,"b_wins_m4":b_wins_m4,"coverage_delta_by_seed":sd.to_dict(),
              "hgb_mean_ratio_B_over_A":ratio,"final_tag":"BLADE_P6G4_AUGMENTED_INITIALIZATION_FINETUNE_COMPLETE","created_utc":now()}
    write_json(OUT/"completion.json", result); Status().update(phase="complete",status="COMPLETE"); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("preflight","select","label","train","evaluate","finalize","all")); parser.add_argument("--rep", choices=REPLICATES); parser.add_argument("--arm", choices=ARMS); parser.add_argument("--device", default="cuda:1"); args = parser.parse_args()
    if args.phase == "preflight": preflight(); return
    if args.phase == "select": select_all(); return
    if args.phase == "label": label_all(); return
    if args.phase == "train":
        if not args.rep or not args.arm: parser.error("train requires --rep and --arm")
        train_one(args.rep,args.arm,args.device); return
    if args.phase == "evaluate":
        if args.rep and args.arm: evaluate_one(args.rep,args.arm,args.device)
        else:
            for rep,arm in JOBS: evaluate_one(rep,arm,args.device)
        return
    if args.phase == "finalize": finalize(); return
    preflight(); select_all(); label_all()
    for rep,arm in JOBS: train_one(rep,arm,args.device); evaluate_one(rep,arm,args.device)
    finalize()


if __name__ == "__main__": main()
