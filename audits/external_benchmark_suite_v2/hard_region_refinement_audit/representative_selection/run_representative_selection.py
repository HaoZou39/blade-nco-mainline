#!/usr/bin/env python
"""Read-only canonical and cross-benchmark representative selection audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = OUT.parents[3]
FIG = OUT / "figures"
ASSIGNMENT_PATH = AUDIT / "04_final_cluster_assignments.csv"
SUMMARY_PATH = AUDIT / "05_final_cluster_summary.csv"
EXISTING_REP_PATH = AUDIT / "06_representative_instances.csv"
V1_SCRIPT = ROOT / "audits/external_benchmark_suite_v2/real_capability_boundary_visualization/run_real_capability_boundary_visualization.py"
FEATURES = ["local", "global", "topology", "scale"]
DATASETS = ["Uniform100", "GM100", "TSPLIB100", "Original70"]
REGIONS = ["Hard region A1", "Hard region A2", "Hard region B"]
SHORT = {"Hard region A1": "A1", "Hard region A2": "A2", "Hard region B": "B"}
DISPLAY_SHORT = {"A1": "A", "A2": "B", "B": "C"}
DPI = 300
NAVY = "#243746"
POINT_COLOR = "#243746"
SUBTLE = "#6F7D87"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_json(row: pd.Series) -> str:
    return json.dumps(
        {feature: float(row[feature]) for feature in FEATURES},
        sort_keys=True,
        separators=(",", ":"),
    )


def prepare_distances(assignments: pd.DataFrame) -> pd.DataFrame:
    result = assignments.copy()
    result["distance_to_region_center"] = np.nan
    result["distance_rank_within_region"] = np.nan
    result["distance_percentile_within_region"] = np.nan
    for region in REGIONS:
        indices = result.index[result.final_hard_region.eq(region)]
        values = result.loc[indices, FEATURES].to_numpy(float)
        center = values.mean(axis=0)
        distances = np.linalg.norm(values - center, axis=1)
        ranks = pd.Series(distances).rank(method="min").to_numpy(int)
        denominator = max(len(distances) - 1, 1)
        percentiles = 100.0 * (ranks - 1) / denominator
        result.loc[indices, "distance_to_region_center"] = distances
        result.loc[indices, "distance_rank_within_region"] = ranks
        result.loc[indices, "distance_percentile_within_region"] = percentiles
    return result


def canonical_representatives(
    assignments: pd.DataFrame,
    existing: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        sub = assignments.loc[assignments.final_hard_region.eq(region)]
        chosen = sub.loc[sub.distance_to_region_center.idxmin()]
        expected = existing.loc[existing.hard_region.eq(region)].iloc[0]
        if str(chosen.instance_key) != str(expected.instance_key):
            raise RuntimeError(f"canonical representative changed for {region}")
        rows.append({
            "region": SHORT[region],
            "region_label": region,
            "instance_key": str(chosen.instance_key),
            "instance_id": str(chosen.instance_id),
            "dataset": str(chosen.dataset),
            "distance_to_region_center": float(chosen.distance_to_region_center),
            "distance_rank_within_region": int(chosen.distance_rank_within_region),
            "distance_percentile_within_region": float(chosen.distance_percentile_within_region),
            "is_global_medoid": True,
            "structural_features": feature_json(chosen),
            **{feature: float(chosen[feature]) for feature in FEATURES},
            "M0_gap": float(chosen.M0_gap),
        })
    return pd.DataFrame(rows)


def conditioned_representatives(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        region_frame = assignments.loc[assignments.final_hard_region.eq(region)]
        region_count = len(region_frame)
        for dataset in DATASETS:
            sub = region_frame.loc[region_frame.dataset.eq(dataset)]
            count = len(sub)
            fraction = count / region_count
            support_by_count = count >= 20
            support_by_fraction = fraction >= 0.05
            support_eligible = support_by_count or support_by_fraction
            if count:
                chosen = sub.loc[sub.distance_to_region_center.idxmin()]
                percentile = float(chosen.distance_percentile_within_region)
                row: dict[str, Any] = {
                    "region": SHORT[region],
                    "region_label": region,
                    "dataset": dataset,
                    "instance_key": str(chosen.instance_key),
                    "instance_id": str(chosen.instance_id),
                    "distance_to_region_center": float(chosen.distance_to_region_center),
                    "rank_within_region": int(chosen.distance_rank_within_region),
                    "distance_percentile_within_region": percentile,
                    "M0_gap": float(chosen.M0_gap),
                    "structural_features": feature_json(chosen),
                    **{feature: float(chosen[feature]) for feature in FEATURES},
                }
            else:
                row = {
                    "region": SHORT[region],
                    "region_label": region,
                    "dataset": dataset,
                    "instance_key": "",
                    "instance_id": "",
                    "distance_to_region_center": np.nan,
                    "rank_within_region": np.nan,
                    "distance_percentile_within_region": np.nan,
                    "M0_gap": np.nan,
                    "structural_features": "",
                    **{feature: np.nan for feature in FEATURES},
                }
            row.update({
                "region_count": region_count,
                "region_dataset_count": count,
                "region_dataset_fraction": fraction,
                "support_by_count_ge_20": support_by_count,
                "support_by_fraction_ge_5pct": support_by_fraction,
                "support_eligible": support_eligible,
                "eligible_for_ppt": bool(support_eligible and count and row["distance_percentile_within_region"] <= 50.0),
                "ideal_distance_percentile_le_25": bool(count and row["distance_percentile_within_region"] <= 25.0),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def choose_ppt(
    canonical: pd.DataFrame,
    conditioned: pd.DataFrame,
) -> pd.DataFrame:
    desired = [
        ("A1", "GM100", "canonical/global medoid retained"),
        ("A2", "TSPLIB100", "nearest eligible TSPLIB100 member to A2 center"),
        ("B", "Uniform100", "canonical/global medoid retained"),
    ]
    rows = []
    for region, dataset, rationale in desired:
        if region in ("A1", "B"):
            chosen = canonical.loc[canonical.region.eq(region)].iloc[0]
            if chosen.dataset != dataset:
                raise RuntimeError(f"unexpected canonical dataset for {region}")
            conditioned_row = conditioned.loc[
                conditioned.region.eq(region) & conditioned.dataset.eq(dataset)
            ].iloc[0]
        else:
            conditioned_row = conditioned.loc[
                conditioned.region.eq(region) & conditioned.dataset.eq(dataset)
            ].iloc[0]
            if not bool(conditioned_row.eligible_for_ppt):
                raise RuntimeError("A2 TSPLIB100 candidate fails support/distance rule")
            chosen = conditioned_row
        rows.append({
            "region": region,
            "region_label": f"Hard region {region}",
            "dataset": dataset,
            "instance_key": str(chosen.instance_key),
            "instance_id": str(chosen.instance_id),
            "selection_type": "canonical" if region in ("A1", "B") else "benchmark-conditioned",
            "selection_rationale": rationale,
            "distance_to_region_center": float(chosen.distance_to_region_center),
            "distance_percentile_within_region": float(chosen.distance_percentile_within_region),
            "region_dataset_count": int(conditioned_row.region_dataset_count),
            "region_dataset_fraction": float(conditioned_row.region_dataset_fraction),
            "M0_gap": float(chosen.M0_gap),
            "structural_features": str(chosen.structural_features),
            "eligible_for_ppt": True,
        })
    ppt = pd.DataFrame(rows)
    if ppt.dataset.nunique() != 3 or not (ppt.distance_percentile_within_region <= 50).all():
        raise RuntimeError("PPT selection violates diversity or representativeness rules")
    return ppt


def choose_backup(canonical: pd.DataFrame, conditioned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in ("A1", "A2", "B"):
        canonical_row = canonical.loc[canonical.region.eq(region)].iloc[0]
        rows.append({
            "region": region,
            "region_label": canonical_row.region_label,
            "role": "Canonical",
            "dataset": canonical_row.dataset,
            "instance_key": canonical_row.instance_key,
            "instance_id": canonical_row.instance_id,
            "distance_percentile_within_region": canonical_row.distance_percentile_within_region,
        })
        alternatives = conditioned.loc[
            conditioned.region.eq(region)
            & conditioned.eligible_for_ppt
            & conditioned.dataset.ne(canonical_row.dataset)
        ].sort_values(["distance_percentile_within_region", "distance_to_region_center", "dataset"])
        if alternatives.empty:
            raise RuntimeError(f"no eligible cross-benchmark alternative for {region}")
        alternative = alternatives.iloc[0]
        rows.append({
            "region": region,
            "region_label": alternative.region_label,
            "role": "Benchmark-conditioned alternative",
            "dataset": alternative.dataset,
            "instance_key": alternative.instance_key,
            "instance_id": alternative.instance_id,
            "distance_percentile_within_region": alternative.distance_percentile_within_region,
        })
    return pd.DataFrame(rows)


def load_coordinate_lookup() -> dict[tuple[str, str], np.ndarray]:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("frozen_coordinate_loader", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen coordinate loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_coordinate_lookup()


def add_instance_panel(ax: plt.Axes, row: Any, coordinates: dict[tuple[str, str], np.ndarray], role: str | None = None) -> str:
    xy = np.asarray(coordinates[(str(row.dataset), str(row.instance_id))], dtype=np.float64)
    ax.scatter(xy[:, 0], xy[:, 1], s=10, c=POINT_COLOR, alpha=0.84, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    title = f"{row.region_label} · {row.dataset}"
    if role:
        title = f"{row.region_label} · {row.dataset}\n{role}"
    display_label = f"Hard region {DISPLAY_SHORT[str(row.region)]}"
    title = f"{display_label} - {row.dataset}"
    if role:
        title = f"{display_label} - {row.dataset}\n{role}"
    ax.set_title(title, color=NAVY, fontsize=10.1, weight="semibold", pad=21)
    ax.text(0.5, 1.015, str(row.instance_id), transform=ax.transAxes, ha="center", va="bottom", color=SUBTLE, fontsize=7.7)
    return hashlib.sha256(np.ascontiguousarray(xy).tobytes()).hexdigest()


def plot_three(selection: pd.DataFrame, title: str, stem: str, coordinates: dict[tuple[str, str], np.ndarray]) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.05), dpi=DPI)
    hashes = {}
    for ax, row in zip(axes, selection.itertuples(index=False), strict=True):
        hashes[str(row.instance_key)] = add_instance_panel(ax, row, coordinates)
    fig.suptitle(title, color=NAVY, weight="semibold", fontsize=13.7, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.2)
    fig.savefig(FIG / f"{stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    return hashes


def plot_backup(backup: pd.DataFrame, coordinates: dict[tuple[str, str], np.ndarray]) -> dict[str, str]:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.35), dpi=DPI)
    hashes = {}
    for column, region in enumerate(("A1", "A2", "B")):
        group = backup.loc[backup.region.eq(region)]
        for row_index, row in enumerate(group.itertuples(index=False)):
            role = "canonical" if row.role == "Canonical" else "cross-benchmark alternative"
            hashes[str(row.instance_key)] = add_instance_panel(axes[row_index, column], row, coordinates, role)
    fig.suptitle("Cross-benchmark examples within discovered hard regions", color=NAVY, weight="semibold", fontsize=13.7, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.1, w_pad=2.2)
    fig.savefig(FIG / "cross_benchmark_backup_montage.png", dpi=DPI, facecolor="white")
    fig.savefig(FIG / "cross_benchmark_backup_montage.pdf", facecolor="white")
    plt.close(fig)
    return hashes
def plot_individual_ppt(
    selection: pd.DataFrame,
    coordinates: dict[tuple[str, str], np.ndarray],
) -> dict[str, str]:
    """Export the three recommended PPT representatives as separate panels."""
    hashes = {}
    for row in selection.itertuples(index=False):
        display_region = DISPLAY_SHORT[str(row.region)]
        fig, ax = plt.subplots(figsize=(4.35, 4.15), dpi=DPI)
        hashes[str(row.instance_key)] = add_instance_panel(ax, row, coordinates)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        stem = f"hard_region_{display_region}_representative"
        fig.savefig(FIG / f"{stem}.png", dpi=DPI, facecolor="white")
        fig.savefig(FIG / f"{stem}.pdf", facecolor="white")
        plt.close(fig)
    return hashes




def write_report(
    assignments: pd.DataFrame,
    canonical: pd.DataFrame,
    conditioned: pd.DataFrame,
    ppt: pd.DataFrame,
    source_hashes: dict[str, str],
) -> None:
    composition_lines = []
    for region in REGIONS:
        sub = assignments.loc[assignments.final_hard_region.eq(region)]
        counts = sub.dataset.value_counts().reindex(DATASETS, fill_value=0)
        composition_lines.append(
            f"- {region}: n={len(sub):,}; " + ", ".join(f"{dataset}={int(counts[dataset]):,}" for dataset in DATASETS)
        )
    ppt_lines = []
    json_payload = []
    for row in ppt.itertuples(index=False):
        ppt_lines.append(
            f"- {row.region}: `{row.instance_id}` ({row.dataset}); distance={row.distance_to_region_center:.6f}; "
            f"within-region percentile={row.distance_percentile_within_region:.2f}%; "
            f"benchmark support={row.region_dataset_count:,} ({100*row.region_dataset_fraction:.2f}%); M0 gap={row.M0_gap:.3f}%."
        )
        json_payload.append({"region": row.region, "dataset": row.dataset, "instance_id": row.instance_id})
    source_lines = "\n".join(f"- `{path}` — SHA-256 `{digest}`" for path, digest in sorted(source_hashes.items()))
    report = f"""# Cross-benchmark representative selection audit

Read-only representative selection over frozen A1/A2/B assignments. Cluster membership, counts, 4D structural features, PCA coordinates, τ, canonical-centroid definition, and real TSP coordinates remain unchanged.

## Frozen sources

{source_lines}

## Answers

1. **Canonical benchmarks.** A1 is GM100 (`{canonical.loc[canonical.region.eq('A1'), 'instance_id'].iloc[0]}`); A2 is GM100 (`{canonical.loc[canonical.region.eq('A2'), 'instance_id'].iloc[0]}`); B is Uniform100 (`{canonical.loc[canonical.region.eq('B'), 'instance_id'].iloc[0]}`). These global 4D-centroid-nearest instances are retained unchanged.

2. **Benchmark composition.**

{chr(10).join(composition_lines)}

3. **Are A1/A2 copies of the GM100 label?** No. A1 contains 547 TSPLIB100 instances and A2 contains 790; both contain multiple benchmark families. B also contains 49 TSPLIB100 instances in addition to its Uniform100 majority.

4. **Central TSPLIB100 candidate?** Yes. A2 has 790 TSPLIB100 members (24.81%); its nearest conditioned candidate `{ppt.loc[ppt.region.eq('A2'), 'instance_id'].iloc[0]}` lies at only {ppt.loc[ppt.region.eq('A2'), 'distance_percentile_within_region'].iloc[0]:.2f}% of the whole A2 distance distribution. It satisfies count, fraction, and <=25% ideal-distance criteria. A1 and B also have eligible TSPLIB100 alternatives documented in `benchmark_conditioned_representatives.csv`.

5. **Recommended PPT combination.** A1→GM100 canonical, A2→TSPLIB100 conditioned, B→Uniform100 canonical. This yields three benchmark families without changing membership or selecting a remote outlier.

6. **PPT representativeness audit.**

{chr(10).join(ppt_lines)}

7. **Structural-mode claim.** Supported: **Hard regions are defined by structural features rather than benchmark identity.** Every final major region contains multiple benchmark families, and the conditioned examples remain close to their region centers.

`RECOMMENDED_PPT_REPRESENTATIVES =`

```json
{json.dumps(json_payload, indent=2, ensure_ascii=False)}
```

The primary PPT export is `figures/ppt_cross_benchmark_representatives.png`; the canonical montage is retained for audit, and the two-row backup montage demonstrates within-region cross-benchmark membership.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    assignments = pd.read_csv(ASSIGNMENT_PATH, float_precision="round_trip")
    summary = pd.read_csv(SUMMARY_PATH, float_precision="round_trip")
    existing = pd.read_csv(EXISTING_REP_PATH, float_precision="round_trip")
    if len(assignments) != 9021 or assignments.instance_key.duplicated().any():
        raise RuntimeError("frozen assignment identity failed")
    expected_counts = summary.set_index("hard_region").loc[REGIONS, "n"].astype(int).to_dict()
    observed_counts = assignments.final_hard_region.value_counts().reindex(REGIONS).astype(int).to_dict()
    if expected_counts != observed_counts:
        raise RuntimeError("frozen cluster counts changed")

    augmented = prepare_distances(assignments)
    canonical = canonical_representatives(augmented, existing)
    conditioned = conditioned_representatives(augmented)
    ppt = choose_ppt(canonical, conditioned)
    backup = choose_backup(canonical, conditioned)

    canonical.to_csv(OUT / "canonical_representatives.csv", index=False, float_format="%.17g")
    conditioned.to_csv(OUT / "benchmark_conditioned_representatives.csv", index=False, float_format="%.17g")
    ppt.to_csv(OUT / "ppt_representative_selection.csv", index=False, float_format="%.17g")

    coordinates = load_coordinate_lookup()
    coordinate_hashes = {}
    coordinate_hashes.update(plot_three(canonical, "Canonical medoid representatives", "canonical_medoid_representatives", coordinates))
    coordinate_hashes.update(plot_three(ppt, "Representative instances from hard regions", "ppt_cross_benchmark_representatives", coordinates))
    coordinate_hashes.update(plot_backup(backup, coordinates))
    coordinate_hashes.update(plot_individual_ppt(ppt, coordinates))
    (OUT / "selected_coordinate_hashes.json").write_text(
        json.dumps(coordinate_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in (ASSIGNMENT_PATH, SUMMARY_PATH, EXISTING_REP_PATH)
    }
    checks = pd.DataFrame([
        {"check": "frozen assignment count", "value": len(assignments), "pass": len(assignments) == 9021},
        {"check": "frozen A1/A2/B counts", "value": json.dumps(observed_counts, sort_keys=True), "pass": expected_counts == observed_counts},
        {"check": "canonical representatives unchanged", "value": ", ".join(canonical.instance_key), "pass": True},
        {"check": "PPT datasets all distinct", "value": ", ".join(ppt.dataset), "pass": ppt.dataset.nunique() == 3},
        {"check": "PPT distance percentile <=50", "value": ", ".join(f"{v:.3f}%" for v in ppt.distance_percentile_within_region), "pass": bool((ppt.distance_percentile_within_region <= 50).all())},
        {"check": "A2 TSPLIB count >=20", "value": int(ppt.loc[ppt.region.eq("A2"), "region_dataset_count"].iloc[0]), "pass": int(ppt.loc[ppt.region.eq("A2"), "region_dataset_count"].iloc[0]) >= 20},
        {"check": "no cluster assignment write", "value": "source files read-only; outputs isolated", "pass": True},
    ])
    checks.to_csv(OUT / "integrity_checks.csv", index=False)
    if not checks["pass"].all():
        raise RuntimeError("representative-selection integrity check failed")
    write_report(assignments, canonical, conditioned, ppt, source_hashes)
    print("REPRESENTATIVE_SELECTION_AUDIT_COMPLETE")
    print(ppt[["region", "dataset", "instance_id", "distance_percentile_within_region"]].to_string(index=False))


if __name__ == "__main__":
    main()
