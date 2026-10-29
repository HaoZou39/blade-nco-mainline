# E0 Structure10k Acquisition Audit

- Frozen E0 bounded archive: 30,000 unique eligible offspring; source bundle SHA256 `fd9bd10e5bd79f20054ea3b78dad62b541c0cae6226a661ffcea2b466c8bcb0a`.
- Exact P2.2 acquisition replay: ordered IDs **identical**, metadata **identical**, and archive-derived coordinates **bitwise identical**.
- Output: 10,000 unique TSP100; bundle SHA256 `987395ae7de3e9a4118ed26ce9cf2c369a6b0498623dadd14668812a4cc3cc69`; membership SHA256 `139ab50f468eb545f378f52372f259d150c84d4e34c94b040e090cbd146d02f7`.
- C00–C03 overlap: 0; C04 Hard10k overlap: 0; E1 Structure10k overlap: 3,406.
- Strict target: 3,016/10,000 (30.16%); reference coverage 45/48; missing neighborhoods: ['d198', 'eil51', 'ts225'].
- Requested quotas: `[2292, 1042, 0, 208, 2083, 1042, 833, 208, 2292]`.
- Capacity-adjusted quotas: `[1451, 136, 0, 1671, 1543, 50, 2296, 1670, 1183]`; actual counts reproduce them exactly. Original cell deficits are recorded in `e0_structure10k_config.json`; no outside data or mutation was added.
- Effective lineages 563.7; maximum lineage fraction 1.20%; lineage cap started at 50 and deterministically relaxed to 200.
- Hardness and HGB predictions are absent from selection.

Status: **PASS; E0 Structure10k frozen before labeling**.
