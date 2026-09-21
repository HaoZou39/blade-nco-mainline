# Repair provenance

The raw output is produced by each operator's mutator and returned as `RawMutation`; the shared `execute_operator` function in `src/mutation/base.py` then calls `check_validity` and `repair_coordinates` from `src/mutation/repair_adapter.py`.

The adapter marks every non-finite/out-of-[0,1] node and the later-index node of each pair at Euclidean distance below `1e-5`. It deterministically replaces only those indexed coordinate pairs with i.i.d. Uniform([0,1]^2) draws from the materialized `repair_seed`, repeating for at most 20 passes. The union of touched node indices is recorded. A proposal fails when the touched-node fraction exceeds 0.20 or final validity fails.

This preserves instance cardinality, node ordering, unaffected coordinates, scale convention, and all legal geometry. It performs no clipping, scaling, translation, or normalization. Duplicate/too-close points are handled by resampling the later-index endpoint. There is no operator-specific repair branch: all 11 operators use the same execution adapter.

Historical inspection found the same domain-repair concept in the pre-consolidation development pipeline (`src/blade_nco/operator_bank/validity.py`). The frozen consolidated implementation was introduced by development commit `003afd4a6f13d8faa86e9f3118034eac5b36b233` and describes itself as a BLADE-specific deterministic replay/repair/validity compatibility contract around the Bossek et al. FOGA/tspgen geometry, not as a line-by-line upstream R repair implementation. Mainline's operator modules and YAML are byte-identical to that frozen development implementation, except `parameters.py` only resolves the colocated mainline configuration path.

Raw repaired outputs are finite, so QD4 is numerically calculable as a geometric diagnostic. Because every such raw output violates the admissible unit-square/spacing contract, pre-repair QD4 is explicitly tagged as outside the scientifically admissible candidate domain and is not used for calibration, training, or acquisition.
