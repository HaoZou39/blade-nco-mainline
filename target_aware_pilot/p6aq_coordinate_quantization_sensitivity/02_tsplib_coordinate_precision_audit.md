# TSPLIB48 Coordinate Precision Audit

| edge_weight_type | instances |
|---|---|
| EUC_2D | 48 |

## Lexical precision recorded in source files

| max_fraction_digits_in_file | lexical_storage_quantum_q | instances |
|---|---|---|
| 0 | 1.0000000000 | 30 |
| 2 | 0.0100000000 | 4 |
| 3 | 0.0010000000 | 5 |
| 5 | 0.0000100000 | 7 |
| 10 | 0.0000000001 | 2 |

- Numerically integer-valued coordinate files: 33/48; 30/48 use integer tokens with no decimal places. The per-file lexical storage quantum is recorded, but true physical/measurement precision is unknown for all 48 and is not inferred.
- One reference (`a280`) contains a duplicated coordinate row and therefore has raw `distance_min=0`; this is retained exactly as frozen.
- The formal feature extractor passes double coordinates to `salesperson::makeNetwork`; feature distances are continuous Euclidean values. It does not apply TSPLIB `EUC_2D`, ATT, GEO, CEIL_2D, or tour-length rounding rules.
- `D` is the Euclidean diagonal of the coordinate bounding box after aspect-preserving normalization. The dimensionless `n·distance_min/D` is invariant to translation and positive uniform scale.
- Integer coordinate storage is not equivalent to an integer-rounded distance matrix. The perturbations below use lexical q only as a declared hypothetical stress scale, not evidence of measurement error.
