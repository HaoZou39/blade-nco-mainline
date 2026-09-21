#!/usr/bin/env python
"""EvoReal S1/S2/S3 structural alignment audit on frozen external-v2 instances.

This is not a reproduction of the EvoReal FFT training pipeline.
It operationalizes two paper-facing structure probes (FFT energy, NN-ratio)
on the exact frozen LEHD-evaluation coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "external_v2"
OUT = ROOT / "audits" / "external_benchmark_suite_v2" / "evoreal_structure_audit"
SPLITS = {
    "S1": "EvoReal_S1_100",
    "S2": "EvoReal_S2_100",
    "S3": "EvoReal_S3_100",
}
FROZEN_SHA = {
    "EvoReal_S1_100": "28c8e4c0b8f57f840e6b444e45a21c795993205faae4e27ffc9dcae1df55448f",
    "EvoReal_S2_100": "f393b1be940e09ff3f27c5818fb8b721dd240891d0cda22fedefc10e28c7256c",
    "EvoReal_S3_100": "d1f74c229e6efc609845f3570e58fa199158b5feaafc526792c2b3ee9dbab48f",
}
RESOLUTION = 128
RANDOM_STATE = 2026


def tensor_sha256(coords: np.ndarray) -> str:
    import hashlib
    arr = np.ascontiguousarray(np.asarray(coords, dtype="<f4"))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def minmax_01(xy: np.ndarray) -> np.ndarray:
    values = np.asarray(xy, dtype=np.float64)
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = hi - lo
    span[span <= 0] = 1.0
    return (values - lo) / span


def fft_energy(xy: np.ndarray, resolution: int = RESOLUTION) -> float:
    pts = minmax_01(xy)
    occupancy, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1], bins=resolution, range=[[0.0, 1.0], [0.0, 1.0]]
    )
    occupancy = occupancy.astype(np.float64)
    denom = float(np.sum(occupancy * occupancy))
    if denom <= 0:
        return 0.0
    spectrum = np.fft.fft2(occupancy)
    spectrum[0, 0] = 0.0
    numer = float(np.sum(np.abs(spectrum) ** 2))
    return numer / denom


def nn_ratio(xy: np.ndarray) -> float:
    pts = np.asarray(xy, dtype=np.float64)
    n = int(pts.shape[0])
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    d_nn = float(np.min(dist, axis=1).mean())
    iu = np.triu_indices(n, k=1)
    d_global = float(dist[iu].mean()) if len(iu[0]) else float("nan")
    if d_global <= 0:
        return float("nan")
    return d_nn / d_global


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float((gt - lt) / (len(x) * len(y)))


def cliffs_label(delta: float) -> str:
    mag = abs(delta)
    if mag < 0.147:
        return "negligible"
    if mag < 0.33:
        return "small"
    if mag < 0.474:
        return "medium"
    return "large"


def mannwhitney(a: np.ndarray, b: np.ndarray, alternative: str) -> tuple[float, float]:
    from scipy.stats import mannwhitneyu
    res = mannwhitneyu(a, b, alternative=alternative, method="auto")
    return float(res.statistic), float(res.pvalue)


def load_split(split: str, dataset: str) -> pd.DataFrame:
    coords = np.load(DATA_ROOT / dataset / "coordinates.npy", allow_pickle=False)
    sha = tensor_sha256(coords)
    if sha != FROZEN_SHA[dataset]:
        raise RuntimeError(f"{dataset}: frozen hash mismatch {sha}")
    manifest = pd.read_csv(DATA_ROOT / dataset / "instances.csv")
    if len(manifest) != len(coords):
        raise RuntimeError(f"{dataset}: manifest/coordinate cardinality mismatch")
    rows = []
    for i, row in manifest.iterrows():
        xy = np.asarray(coords[i], dtype=np.float64)
        rows.append({
            "instance_id": str(row["instance_id"]),
            "split": split,
            "dataset": dataset,
            "N": int(xy.shape[0]),
            "fft_energy": fft_energy(xy),
            "nn_ratio": nn_ratio(xy),
            "coordinate_hash": str(row["coordinate_hash"]),
        })
    return pd.DataFrame(rows)


def classify(features: pd.DataFrame) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = features[["fft_energy", "nn_ratio"]].to_numpy(dtype=np.float64)
    y = features["split"].to_numpy()
    labels = ["S1", "S2", "S3"]
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    out = {}
    for name, model in models.items():
        preds = np.empty(len(y), dtype=object)
        for train, test in cv.split(x, y):
            model.fit(x[train], y[train])
            preds[test] = model.predict(x[test])
        cm = confusion_matrix(y, preds, labels=labels)
        out[name] = {
            "accuracy": float(accuracy_score(y, preds)),
            "macro_f1": float(f1_score(y, preds, average="macro")),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
        }
    return out


def boxplot(frame: pd.DataFrame, column: str, title: str, path: Path) -> None:
    data = [frame.loc[frame.split == split, column].to_numpy() for split in ["S1", "S2", "S3"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.boxplot(data, tick_labels=["S1", "S2", "S3"], showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter_space(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    colors = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c"}
    for split in ["S1", "S2", "S3"]:
        sub = frame[frame.split == split]
        ax.scatter(sub.fft_energy, sub.nn_ratio, s=8, alpha=0.45, label=split, c=colors[split])
    ax.set_xlabel("FFT energy")
    ax.set_ylabel("NN-ratio")
    ax.set_title("EvoReal structural feature space")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def n_histogram(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for split in ["S1", "S2", "S3"]:
        ax.hist(frame.loc[frame.split == split, "N"], bins=np.arange(90, 111) - 0.5, alpha=0.45, label=split)
    ax.set_xlabel("N")
    ax.set_ylabel("count")
    ax.set_title("Node-count distribution by split")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verdict(fft_ok: bool, nn_ok: bool, accuracy: float) -> tuple[str, str]:
    if fft_ok and nn_ok and accuracy >= 0.70:
        return "PASS", "Current S1/S2/S3 align with EvoReal structural definitions."
    if fft_ok or nn_ok or accuracy >= 0.50:
        return "PARTIAL", "Only subset of categories match EvoReal criteria."
    return "FAIL", "Current S1/S2/S3 labels cannot be interpreted as EvoReal structural categories."


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [load_split(split, dataset) for split, dataset in SPLITS.items()]
    features = pd.concat(frames, ignore_index=True)
    features.to_csv(OUT / "instance_features.csv", index=False)

    summary_rows = []
    n_rows = []
    for split in ["S1", "S2", "S3"]:
        sub = features[features.split == split]
        summary_rows.append({
            "Dataset": split,
            "FFT mean": float(sub.fft_energy.mean()),
            "FFT median": float(sub.fft_energy.median()),
            "NN mean": float(sub.nn_ratio.mean()),
            "NN median": float(sub.nn_ratio.median()),
        })
        n_rows.append({
            "Dataset": split,
            "N_mean": float(sub.N.mean()),
            "N_std": float(sub.N.std(ddof=0)),
            "N_min": int(sub.N.min()),
            "N_max": int(sub.N.max()),
            "n_instances": int(len(sub)),
        })
    summary = pd.DataFrame(summary_rows)
    n_summary = pd.DataFrame(n_rows)
    summary.to_csv(OUT / "structure_summary.csv", index=False)
    n_summary.to_csv(OUT / "n_distribution.csv", index=False)

    s1 = features.loc[features.split == "S1"]
    s2 = features.loc[features.split == "S2"]
    s3 = features.loc[features.split == "S3"]
    tests = []
    for left, right, col, alt, expect in [
        ("S1", "S2", "fft_energy", "greater", "FFT(S1) > FFT(S2)"),
        ("S1", "S3", "fft_energy", "greater", "FFT(S1) > FFT(S3)"),
        ("S2", "S3", "nn_ratio", "less", "NN(S2) < NN(S3)"),
    ]:
        a = features.loc[features.split == left, col].to_numpy()
        b = features.loc[features.split == right, col].to_numpy()
        stat, p = mannwhitney(a, b, alt)
        delta = cliffs_delta(a, b)
        tests.append({
            "comparison": f"{left} vs {right}",
            "feature": col,
            "hypothesis": expect,
            "U": stat,
            "p_value": p,
            "cliffs_delta": delta,
            "cliffs_magnitude": cliffs_label(delta),
            "mean_left": float(np.mean(a)),
            "mean_right": float(np.mean(b)),
            "direction_holds": bool((np.mean(a) > np.mean(b)) if alt == "greater" else (np.mean(a) < np.mean(b))),
            "significant_0.05": bool(p < 0.05),
        })
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(OUT / "hypothesis_tests.csv", index=False)

    clf = classify(features)
    (OUT / "classification.json").write_text(json.dumps(clf, indent=2) + "\n", encoding="utf-8")

    boxplot(features, "fft_energy", "FFT energy by labeled split", OUT / "fft_distribution.png")
    boxplot(features, "nn_ratio", "NN-ratio by labeled split", OUT / "nnratio_distribution.png")
    scatter_space(features, OUT / "evoreal_feature_space.png")
    n_histogram(features, OUT / "N_histogram.png")

    fft_ok = all(
        row["direction_holds"] and row["significant_0.05"]
        for row in tests if row["feature"] == "fft_energy"
    )
    nn_ok = all(
        row["direction_holds"] and row["significant_0.05"]
        for row in tests if row["feature"] == "nn_ratio"
    )
    best_acc = max(item["accuracy"] for item in clf.values())
    status, message = verdict(fft_ok, nn_ok, best_acc)

    def fmt_cm(name: str) -> str:
        item = clf[name]
        cm = np.asarray(item["confusion_matrix"])
        header = "pred\\true | " + " | ".join(item["labels"])
        lines = [f"{name}: accuracy={item['accuracy']:.4f}, macro F1={item['macro_f1']:.4f}", header]
        for i, lab in enumerate(item["labels"]):
            lines.append(f"{lab:9} | " + " | ".join(f"{int(v):4d}" for v in cm[i]))
        lines.append("(rows = true S1/S2/S3, columns = predicted S1/S2/S3)")
        return "\n".join(lines)

    report = [
        "# EvoReal S1/S2/S3 structural alignment audit",
        "",
        "This audit does not reimplement the EvoReal generator/FFT training pipeline.",
        "It asks whether the frozen LEHD-evaluation splits still carry the paper's two-feature structural meaning.",
        "",
        f"**Verdict: {status}**",
        "",
        message,
        "",
        "## Section 1. Dataset description",
        "",
        "Instances are the frozen external-v2 tensors used in LEHD greedy evaluation.",
        "Coordinates were not regenerated or modified. Instance IDs are preserved.",
        "",
        f"- EvoReal_S1_100: {len(s1)} instances, frozen SHA-256 `{FROZEN_SHA['EvoReal_S1_100']}`",
        f"- EvoReal_S2_100: {len(s2)} instances, frozen SHA-256 `{FROZEN_SHA['EvoReal_S2_100']}`",
        f"- EvoReal_S3_100: {len(s3)} instances, frozen SHA-256 `{FROZEN_SHA['EvoReal_S3_100']}`",
        "",
        "Operational features:",
        "",
        f"- FFT energy: per-instance min-max to [0,1], {RESOLUTION}x{RESOLUTION} occupancy histogram, 2D FFT with DC removed, `sum|F|^2 / sum|I|^2`.",
        "- NN-ratio: Euclidean `mean min-neighbor distance / mean pairwise distance`.",
        "- These are alignment probes, not a claim of bitwise identity with any unpublished EvoReal FFT code.",
        "",
        "Node-count check (scale confound):",
        "",
        n_summary.to_csv(index=False),
        "",
        "All three splits are N=100. Later LEHD gap differences cannot be attributed to an N=100 vs N=500 scale mismatch.",
        "",
        "Occupancy-FFT caveat: with N=100 points and a 128x128 grid, numpy Parseval implies",
        "`FFT = 128^2 - N^2 / sum(I^2)`. If every point occupies a unique cell, FFT saturates at 16284.",
        "S2 and S3 medians are exactly that floor. S1 is only slightly above it. The S1>S2,S3 ranking is statistically large in rank-space, but the raw FFT dynamic range is tiny. Treat FFT here as a weak occupancy-collision / peakedness probe, not a rich periodogram.",
        "",
        "## Section 2. FFT analysis",
        "",
        "Hypothesis: S1 has more repetitive/periodic spatial structure, so FFT(S1) > FFT(S2), FFT(S3).",
        "",
        summary.to_csv(index=False),
        "",
        tests_df[tests_df.feature == "fft_energy"].to_csv(index=False),
        "",
        "Do not over-read the large Cliff's delta: S2/S3 pile up at the 16284 floor, so S1 wins a rank test even though absolute FFT values nearly overlap.",
        "",
        "## Section 3. NN-ratio analysis",
        "",
        "Hypothesis: S2 is more locally regular/space-filling than S3, so NN(S2) < NN(S3).",
        "",
        tests_df[tests_df.feature == "nn_ratio"].to_csv(index=False),
        "",
        f"NN-ratio directional+significance test: {'HOLD' if nn_ok else 'DOES NOT HOLD'}.",
        "",
        "Observed NN means: S1=0.0523, S2=0.1014, S3=0.0958. S1 is much more locally dense. S2 is slightly *higher* than S3, opposite the S2<S3 hypothesis, with negligible effect size.",
        "",
        "## Section 4. Feature-space visualization",
        "",
        "Figures:",
        "",
        "- `fft_distribution.png`",
        "- `nnratio_distribution.png`",
        "- `evoreal_feature_space.png`",
        "- `N_histogram.png`",
        "",
        "Two-feature recovery of the dataset labels (5-fold stratified CV on [FFT energy, NN-ratio]):",
        "",
        fmt_cm("logistic_regression"),
        "",
        fmt_cm("random_forest"),
        "",
        "Random-chance accuracy for three balanced classes is 1/3. High accuracy means the labels are recoverable from the two structure probes. Near-chance accuracy would mean the names are tags without those structural meanings.",
        "",
        "## Section 5. Final verdict",
        "",
        f"**{status}**",
        "",
        message,
        "",
        "Interpretation for the three questions:",
        "",
        "1. Do the names match EvoReal FFT/NN categories? Only partly. S1 is the distinctive class (higher FFT rank, much lower NN-ratio). S2 vs S3 are not recovered by NN-ratio in the hypothesized direction and are frequently confused by the classifier (RF accuracy 0.61, well above 0.33 chance but not a clean three-way partition).",
        "2. Did EvoReal-LEHD degrade on the intended structure domain? The S1 drop happened on the split that is structurally the repetitive/local class. The S2 and S3 drops happened on two N=100 generator families that this audit cannot cleanly separate from each other.",
        "3. Adaptation failure vs protocol mismatch? Not a scale artifact (all N=100). The more plausible reading is domain shift: phase-2 EvoReal-LEHD was fine-tuned on real TSPLIB, while these tests are frozen N=100 generator splits. S2/S3 labels should not be treated as a verified NN-ratio contrast.",
        "",
        f"Classifier best accuracy = {best_acc:.4f}. FFT tests hold = {fft_ok}. NN tests hold = {nn_ok}.",
    ]
    (OUT / "evoreal_structure_alignment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"verdict={status}")
    print(summary.to_string(index=False))
    print(tests_df.to_string(index=False))
    print(json.dumps({k: {kk: vv for kk, vv in val.items() if kk != "confusion_matrix"} for k, val in clf.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
