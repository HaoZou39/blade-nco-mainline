# Repair Actions

## 2026-10-27T15:37:00+00:00 monitor conflict classification

The initial monitor classified the P6-D1 runner's own expected GPU1 allocation as a resource conflict because the first implementation used a fixed memory threshold. The monitor was changed to flag only compute PIDs on GPU1 other than the recorded runner PID. No scientific code, dataset, training process, or schedule was changed, and no extra health check was triggered.

## 2026-10-28T07:22:00+08:00 final-report duplicate dataset key

All nine trainings, all nine frozen greedy evaluations, and all nine union CE inference jobs passed before finalization. Final report generation failed because `OriginalTSPLIB70` appeared once as a primary dataset and again inside GAP5, producing a duplicate historical comparison key. The report-only dataset list was changed to preserve order while removing duplicate names. No dataset, checkpoint, inference result, statistic definition, or scientific protocol was changed; only CPU finalization is rerun.

## 2026-10-28T07:24:00+08:00 complete M4 CE source for cross-population matrix

The first repaired finalization reached the 60-cell cross-population CE report and found that the P6-D0 selection manifest does not contain every historical E0/E1 member, as expected because it records selected A/B/C/D/E populations rather than the complete union. The M4 baseline CE lookup was changed to the frozen P6-C1 full union manifest. An exact `1e-12` consistency gate against all D0-selected rows was added. This restores only missing baseline lookup coverage and does not recompute CE or alter any model result.
