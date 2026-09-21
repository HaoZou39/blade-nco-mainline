$ErrorActionPreference = 'Continue'

$Root = 'D:\NCO\BLADE-NCO-mainline'
$RunRoot = Join-Path $Root 'runs\blade_final_20cycle_seed2026_v1'
$AuditRoot = Join-Path $Root 'audits\original70_canonical_trajectory_v1'
$Python = 'D:\conda_envs\bq-nco\python.exe'
$Evaluator = Join-Path $AuditRoot 'evaluate_original70_canonical.py'
$Runner = Join-Path $Root 'scripts\run_final_20cycle.py'
$Config = Join-Path $Root 'configs\final_20cycle_protocol.yaml'
$StatusLog = Join-Path $AuditRoot 'urgent_orchestrator.log'
$EvalStdout = Join-Path $AuditRoot 'canonical_stdout.log'
$EvalStderr = Join-Path $AuditRoot 'canonical_stderr.log'
$FormalStdout = Join-Path $RunRoot 'logs\resume_gpu_attempt9_after_original70_stdout.log'
$FormalStderr = Join-Path $RunRoot 'logs\resume_gpu_attempt9_after_original70_stderr.log'

"$(Get-Date -Format o) START_CANONICAL_M0_M16" | Add-Content -LiteralPath $StatusLog
& $Python $Evaluator --through 16 --no-require-formal-complete 1>> $EvalStdout 2>> $EvalStderr
$EvaluationExitCode = $LASTEXITCODE
"$(Get-Date -Format o) CANONICAL_M0_M16_EXIT_CODE=$EvaluationExitCode" | Add-Content -LiteralPath $StatusLog

"$(Get-Date -Format o) RESUME_FORMAL_SAME_RUN_ID" | Add-Content -LiteralPath $StatusLog
& $Python $Runner --config $Config --run-id 'blade_final_20cycle_seed2026_v1' --resume 1>> $FormalStdout 2>> $FormalStderr
$FormalExitCode = $LASTEXITCODE
"$(Get-Date -Format o) FORMAL_RESUME_EXIT_CODE=$FormalExitCode" | Add-Content -LiteralPath $StatusLog

exit $FormalExitCode
