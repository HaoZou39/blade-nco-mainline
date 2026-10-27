# Oracle cache and compute budget

## Exact availability

| Scope | Evaluable unique instances | M5 labels available | M5 labels missing | Missing optimal tours | Invalid/rejected attempts |
|---|---:|---:|---:|---:|---:|
| 30-parent pilot | 356 (30 parents + 326 children) | 20 (6 + 14) | 336 | 312 | 4 |
| 300-parent panel | 3,581 (300 parents + 3,281 children) | 323 (86 + 237) | 3,258 | 3,044 | 19 |

All 300 parent tours are reusable from exact-hash S6-AE ledgers. Every currently M5-labeled child also has a reusable tour. The 3,044 unlabeled valid children have no exact-hash tour in the C00-C09 S6-AE oracle ledgers and therefore require new Concorde work if the full panel is approved. Another-checkpoint labels exist for 214 parents but do not replace M5 evaluation.

The per-record classification is in `05_current_checkpoint_label_availability.csv`: A=M5 label; B=other-checkpoint label only; C=tour but no label; D=no label/tour; E=invalid/rejected/duplicate. Class precedence intentionally keeps B separate even when its tour is reusable.

## Verified timing basis and estimate

No benchmark was run for this audit. Estimates use frozen C05 logs only:

- `C05/_calibration_info.json`: 10,000 M5 forward labels in 233.3313835 s, or 42.8575 instances/s.
- `C05/07_oracle/T/oracle_summary.json`: 10,000 requested, 5,223 cache hits, 4,777 new Concorde calls, 668.3377722 s combined oracle-stage wall time.
- `C05/07_oracle/T/ledger.parquet`: stored solver runtime mean 0.641950 s, median 0.515407 s per solved tour; summed runtime 6,419.50 CPU-s for 10,000 tours.
- Canonical solver call uses eight workers (`src/final_run/science.py`).

Projected at those verified rates:

| Scope | M5 forward compute | Concorde calls | Concorde CPU-time proxy | Conservative combined wall proxy |
|---|---:|---:|---:|---:|
| Pilot | 336 labels ≈ 7.84 s | 312 | ≈ 200.29 CPU-s; ideal 8-worker floor ≈ 25.0 s | ≈ 43.7 s |
| Full | 3,258 labels ≈ 76.02 s | 3,044 | ≈ 1,954.10 CPU-s; ideal 8-worker floor ≈ 244.3 s | ≈ 426.0 s |

The combined wall proxy scales C05's measured 4,777-new-call stage directly and includes its observed cache/evaluation overhead; it is not a hardware guarantee. The GPU model and concurrent resource contention are not recorded here, so no narrower wall-clock promise is made. There is no measured CPU-only BQ fallback estimate.

## Budget conclusion

The measured-order cost is small relative to one S6-AE cycle (`C05/cycle_costs.json`: 9,142.36 s total), so cost does not justify a C verdict. Execution still requires explicit human approval and must start with only the frozen 30-parent pilot.

Identity: M5 SHA-256 `ef07226bb8c3e8cd3a1404eece474df44d378316fd66509b762eb5f8bc26558c`; sample 300 parents/3,300 attempts; mutation manifest SHA-256 `e21fc9f4e989f14193407b6194b885ab4e5e8ab523ba300bec3550caf8bfd158`.

