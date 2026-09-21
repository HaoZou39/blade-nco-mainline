$ErrorActionPreference = 'Stop'

$RunRoot = 'D:\NCO\BLADE-NCO-mainline\runs\blade_final_20cycle_seed2026_v1'
$AuditRoot = 'D:\NCO\BLADE-NCO-mainline\audits\original70_canonical_trajectory_v1'
$Marker = Join-Path $RunRoot 'BLADE_FINAL_20CYCLE_EXECUTION_COMPLETE'
$LockPath = Join-Path $RunRoot 'RUN.lock'
$Python = 'D:\conda_envs\bq-nco\python.exe'
$Evaluator = Join-Path $AuditRoot 'evaluate_original70_canonical.py'
$WatcherLog = Join-Path $AuditRoot 'watcher.log'
$StdoutLog = Join-Path $AuditRoot 'canonical_stdout.log'
$StderrLog = Join-Path $AuditRoot 'canonical_stderr.log'

New-Item -ItemType Directory -Path $AuditRoot -Force | Out-Null
Set-Content -LiteralPath (Join-Path $AuditRoot 'watcher.pid') -Value $PID
"$(Get-Date -Format o) WAITING_FOR_FORMAL_COMPLETION" | Add-Content -LiteralPath $WatcherLog

while (-not (Test-Path -LiteralPath $Marker)) {
    Start-Sleep -Seconds 60
}

$FormalPid = $null
if (Test-Path -LiteralPath $LockPath) {
    try {
        $FormalPid = [int]((Get-Content -Raw -LiteralPath $LockPath | ConvertFrom-Json).pid)
    } catch {
        "$(Get-Date -Format o) LOCK_PID_PARSE_FAILED $($_.Exception.Message)" | Add-Content -LiteralPath $WatcherLog
    }
}
if ($null -ne $FormalPid) {
    while (Get-Process -Id $FormalPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }
}

"$(Get-Date -Format o) FORMAL_COMPLETE_AND_PID_EXITED STARTING_CANONICAL_REEVALUATION" | Add-Content -LiteralPath $WatcherLog
& $Python $Evaluator --through 20 1>> $StdoutLog 2>> $StderrLog
$ExitCode = $LASTEXITCODE
"$(Get-Date -Format o) CANONICAL_REEVALUATION_EXIT_CODE=$ExitCode" | Add-Content -LiteralPath $WatcherLog
exit $ExitCode
