param(
    [int]$IntervalSeconds = 7200,
    [int]$HeartbeatStaleSeconds = 900,
    [int]$MaxConsecutiveRestarts = 3
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $repo "runtime"
$heartbeatPath = Join-Path $runtime "heartbeat.json"
$completionPaths = @(
    (Join-Path $repo "BLADE_V2_FULL_M15_AND_GP_ONLY_M15_COMPLETE")
)
$statePath = Join-Path $runtime "blade_v2_watchdog_state.json"
$logPath = Join-Path $runtime "blade_v2_watchdog.jsonl"
$python = "D:\conda_envs\bq-nco\python.exe"
$runner = "scripts\run_blade_v2.py"
$config = "configs\blade_v2_protocol.yaml"

New-Item -ItemType Directory -Path $runtime -Force | Out-Null

function Write-WatchdogEvent {
    param([hashtable]$Event)
    $Event.timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -LiteralPath $logPath -Value ($Event | ConvertTo-Json -Compress -Depth 6) -Encoding UTF8
}

function Read-WatchdogState {
    if (-not (Test-Path -LiteralPath $statePath)) {
        return @{ consecutive_restarts = 0 }
    }
    try {
        $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        return @{ consecutive_restarts = [int]$saved.consecutive_restarts }
    }
    catch {
        return @{ consecutive_restarts = 0 }
    }
}

function Write-WatchdogState {
    param([hashtable]$State)
    $temporary = "$statePath.tmp"
    $State.updated_at = (Get-Date).ToUniversalTime().ToString("o")
    Set-Content -LiteralPath $temporary -Value ($State | ConvertTo-Json -Depth 4) -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statePath -Force
}

function Get-BladeProcess {
    return Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object {
            $_.ExecutablePath -eq $python -and
            $_.CommandLine -match "run_blade_v2\.py" -and
            $_.CommandLine -match "blade_v2_protocol\.yaml"
        } |
        Select-Object -First 1
}

function Read-Heartbeat {
    if (-not (Test-Path -LiteralPath $heartbeatPath)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $heartbeatPath -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

Write-WatchdogEvent @{ event = "watchdog_started"; interval_seconds = $IntervalSeconds }

while ($true) {
    Start-Sleep -Seconds $IntervalSeconds

    if ($completionPaths | Where-Object { Test-Path -LiteralPath $_ }) {
        Write-WatchdogEvent @{ event = "experiment_complete"; action = "watchdog_exit" }
        break
    }

    $state = Read-WatchdogState
    $process = Get-BladeProcess
    $heartbeat = Read-Heartbeat
    $heartbeatAge = $null
    if ($null -ne $heartbeat -and $null -ne $heartbeat.timestamp) {
        try {
            $heartbeatAge = [math]::Round(((Get-Date).ToUniversalTime() - [datetimeoffset]::Parse($heartbeat.timestamp).UtcDateTime).TotalSeconds, 1)
        }
        catch {
            $heartbeatAge = $null
        }
    }

    if ($null -ne $process) {
        $state.consecutive_restarts = 0
        Write-WatchdogState $state
        $health = if ($null -ne $heartbeatAge -and $heartbeatAge -le $HeartbeatStaleSeconds) { "healthy" } else { "heartbeat_stale" }
        Write-WatchdogEvent @{
            event = "periodic_check"
            status = $health
            pid = [int]$process.ProcessId
            heartbeat_age_seconds = $heartbeatAge
            arm = $heartbeat.arm
            cycle = $heartbeat.cycle
            phase = $heartbeat.phase
            action = "none"
        }
        continue
    }

    if ($state.consecutive_restarts -ge $MaxConsecutiveRestarts) {
        Write-WatchdogEvent @{
            event = "attention_required"
            status = "process_absent"
            consecutive_restarts = $state.consecutive_restarts
            heartbeat_age_seconds = $heartbeatAge
            latest_error = $heartbeat.latest_error
            action = "restart_limit_reached"
        }
        continue
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $runtime "blade_v2_watchdog_resume_${stamp}_stdout.log"
    $stderrPath = Join-Path $runtime "blade_v2_watchdog_resume_${stamp}_stderr.log"
    $started = Start-Process -FilePath $python `
        -ArgumentList @($runner, "--config", $config, "--resume") `
        -WorkingDirectory $repo `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    $state.consecutive_restarts = [int]$state.consecutive_restarts + 1
    Write-WatchdogState $state
    Write-WatchdogEvent @{
        event = "automatic_resume"
        status = "process_restarted"
        pid = [int]$started.Id
        consecutive_restarts = $state.consecutive_restarts
        heartbeat_age_seconds = $heartbeatAge
        previous_error = $heartbeat.latest_error
        stdout = $stdoutPath
        stderr = $stderrPath
        action = "resume"
    }
}
