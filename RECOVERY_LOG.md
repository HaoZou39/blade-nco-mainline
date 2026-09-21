# BLADE V2 recovery log

- Prelaunch: historical Oracle tour-cache Parquet files exposed a PyArrow repetition-histogram reader incompatibility. Added a read-only DuckDB fallback, verified 167112 unique solved tours, and left all scientific settings unchanged.

- 2026-09-17T15:16:43.947128+00:00: execution failure `OSError: Repetition level histogram size mismatch`. Awaiting protocol-neutral diagnosis.

- 2026-09-17T15:19:20.967327+00:00: resumed by PID 47584 after artifact/hash verification.

- 2026-09-18T10:57:27.081347+00:00: resumed by PID 9156 after artifact/hash verification.

- 2026-09-19T04:55:56.426132+00:00: resumed by PID 44376 after artifact/hash verification.

## E4.5 corrected union sanity

- 2026-09-20: First launch was stopped before BQ after detecting that identical B0 hashes had been bound to different equivalent optimal tours from the global Oracle cache. Teacher-forced M0 hardness changed (max absolute difference 0.10558543), producing a non-frozen GP0 and invalid C00 survivor counts. The incomplete attempt was preserved under `audits/E4_5_corrected_union_full_sanity_v1/failed_attempts/attempt_001_nonfrozen_b0_tour_binding`. Scientific repair: reuse the byte-frozen historical B0 labels/tours and GP0 calibration artifacts. No optimizer, selection, mutation, QD, epoch, LR, or benchmark setting changed.
