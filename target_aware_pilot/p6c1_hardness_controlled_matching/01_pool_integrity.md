# Pool Integrity

All six P6-B G2 Population10k and label bundles were checked in original order. Coordinates, hashes, manifests, labels and P6-C0 M4 CE align; all tours are legal TSP100 permutations. Duplicate hashes have identical coordinates and expert tours. Their CE values must agree within a fixed forward-numerical tolerance of 5e-6; canonical union values use deterministic E0-first ordering. Unions preserve unweighted source membership and label-row provenance.

| replicate | union_n | e0_only | e1_only | both | duplicate_ce_nonzero_n | duplicate_ce_max_abs_diff | duplicate_ce_acceptance_atol | canonical_duplicate_rule | union_manifest_sha256 | union_bundle_sha256 | all_labels_legal | all_hardness_consistent_within_atol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | 19929 | 9929 | 9929 | 71 | 0 | 0 | 5e-06 | E0-first | 8f81ff24c59edbf66ddd509328d2075bc770d29dcb7568a4ec1dea629628981e | aead622f9efb511a5b84816812a43e248fb6e4984b6093609f9f1aefd027d65a | True | True |
| S1 | 19937 | 9937 | 9937 | 63 | 1 | 2.1578086e-06 | 5e-06 | E0-first | 10e0d02bab83b720d172f5c93deb0d11dc78ae2b0d6f8487b08a0a28099258d7 | b12cb5cd1d391ff80c6ac53f4c5b08464657bb093e740a397e7c11a6111a4de8 | True | True |
| S2 | 19927 | 9927 | 9927 | 73 | 0 | 0 | 5e-06 | E0-first | 0902036a18e2a865dcaa297bbe69218136ddf07ff5c644e53a5937ee1393591d | 60f75a7985a6695eeeab34f08a252d3ebd11cd799390f0d9436a47cc39aed374 | True | True |
