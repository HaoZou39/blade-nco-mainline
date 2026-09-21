param(
    [int]$IntervalSeconds = 900,
    [string]$Workspace = "D:\NCO\BLADE-NCO-mainline"
)

$ErrorActionPreference = "Stop"
$runtime = Join-Path $Workspace "runtime"
$heartbeatPath = Join-Path $runtime "heartbeat.json"
$logPath = Join-Path $runtime "e4_5_watchdog.jsonl"
$completePath = Join-Path $Workspace "E4_5_CORRECTED_UNION_FULL_SANITY_COMPLETE"
$runPath = Join-Path $Workspace "runs\blade_v2_corrected_union_sanity_seed2026_v1"
$stderrPath = Join-Path $runPath "logs\runner_stderr.log"

function Write-JsonLine([hashtable]$Record) {
    $Record["timestamp"] = [DateTime]::UtcNow.ToString("o")
    Add-Content -LiteralPath $logPath -Value ($Record | ConvertTo-Json -Compress -Depth 8) -Encoding UTF8
}

Write-JsonLine @{ event = "watchdog_started"; interval_seconds = $IntervalSeconds; pid = $PID }

while ($true) {
    if (Test-Path -LiteralPath $completePath) {
        Write-JsonLine @{ event = "experiment_complete"; action = "watchdog_exit" }
        break
    }

    $record = @{ event = "periodic_check"; status = "unknown"; action = "none" }
    try {
        if (-not (Test-Path -LiteralPath $heartbeatPath)) {
            throw "heartbeat missing"
        }
        $heartbeat = Get-Content -LiteralPath $heartbeatPath -Raw | ConvertFrom-Json
        $record["runner_pid"] = [int]$heartbeat.pid
        $record["arm"] = $heartbeat.arm
        $record["cycle"] = $heartbeat.cycle
        $record["phase"] = $heartbeat.phase
        $record["latest_error"] = $heartbeat.latest_error
        $heartbeatTime = [DateTimeOffset]::Parse($heartbeat.timestamp)
        $age = ([DateTimeOffset]::UtcNow - $heartbeatTime).TotalSeconds
        $record["heartbeat_age_seconds"] = [Math]::Round($age, 1)
        $process = Get-Process -Id ([int]$heartbeat.pid) -ErrorAction SilentlyContinue
        $record["process_alive"] = $null -ne $process
        $record["completed_cycles"] = @(
            Get-ChildItem -LiteralPath $runPath -Directory -Filter "C*" -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "CYCLE_COMPLETE.json") }
        ).Count
        $record["stderr_bytes"] = if (Test-Path -LiteralPath $stderrPath) { (Get-Item -LiteralPath $stderrPath).Length } else { 0 }
        $drive = Get-PSDrive -Name D
        $record["disk_free_gb"] = [Math]::Round($drive.Free / 1GB, 2)
        $os = Get-CimInstance Win32_OperatingSystem
        $record["ram_free_gb"] = [Math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $gpu) { $record["gpu"] = $gpu.Trim() }
        if ($heartbeat.latest_error) {
            $record["status"] = "runner_reported_error"
            $record["action"] = "human_diagnosis_required"
        } elseif ($null -eq $process) {
            $record["status"] = "runner_not_alive"
            $record["action"] = "human_diagnosis_required"
        } elseif ($age -gt 1800) {
            $record["status"] = "heartbeat_stale"
            $record["action"] = "human_diagnosis_required"
        } elseif ($record["stderr_bytes"] -gt 0) {
            $record["status"] = "stderr_nonempty"
            $record["action"] = "inspect_stderr"
        } else {
            $record["status"] = "healthy"
        }
    } catch {
        $record["status"] = "watchdog_check_error"
        $record["action"] = "inspect_watchdog"
        $record["error"] = $_.Exception.Message
    }
    Write-JsonLine $record
    Start-Sleep -Seconds $IntervalSeconds
}
