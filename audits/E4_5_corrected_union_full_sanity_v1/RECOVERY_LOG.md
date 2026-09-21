# E4.5 Recovery Log
## E4.5 corrected union sanity

- 2026-09-20: First launch was stopped before BQ after detecting that identical B0 hashes had been bound to different equivalent optimal tours from the global Oracle cache. Teacher-forced M0 hardness changed (max absolute difference 0.10558543), producing a non-frozen GP0 and invalid C00 survivor counts. The incomplete attempt was preserved under `audits/E4_5_corrected_union_full_sanity_v1/failed_attempts/attempt_001_nonfrozen_b0_tour_binding`. Scientific repair: reuse the byte-frozen historical B0 labels/tours and GP0 calibration artifacts. No optimizer, selection, mutation, QD, epoch, LR, or benchmark setting changed.
