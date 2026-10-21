param(
    [int]$IntervalSeconds = 3600,
    [int]$MinimumFreeGpuMiB = 3500
)

$ErrorActionPreference = 'Stop'
$repoRoot = 'D:\NCO\BLADE-NCO-mainline'
$runtimeDir = Join-Path $repoRoot 'runtime'
$experimentDir = Join-Path $repoRoot 'runs\BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11HIGH_V2'
$lockPath = Join-Path $runtimeDir 'blade_core_ablation_5c_foga11high_v2.lock'
$statusPath = Join-Path $runtimeDir 'blade_core_ablation_5c_foga11high_v2_status.json'
$watchPath = Join-Path $runtimeDir 'blade_core_ablation_5c_foga11high_v2_watch.json'
$watchLog = Join-Path $runtimeDir 'blade_core_ablation_5c_foga11high_v2_watch.jsonl'
$stdoutPath = Join-Path $runtimeDir 'blade_core_ablation_5c_foga11high_v2.stdout.log'
$stderrPath = Join-Path $runtimeDir 'blade_core_ablation_5c_foga11high_v2.stderr.log'
$pythonExe = 'D:\conda_envs\bq-nco\python.exe'
$runner = 'scripts\run_blade_core_ablation_5c_noanchor_foga11high_v2.py'
$nvidiaSmi = 'C:\Windows\System32\nvidia-smi.exe'

function Read-JsonSafe([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try { return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json }
    catch { return $null }
}

function Test-ProcessLive([int]$ProcessId) {
    if ($ProcessId -le 0) { return $false }
    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Get-FreeGpuMiB {
    if (-not (Test-Path -LiteralPath $nvidiaSmi)) { return 0 }
    $line = & $nvidiaSmi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($null -eq $line) { return 0 }
    $parsed = 0
    if (-not [int]::TryParse(([string]$line).Trim(), [ref]$parsed)) { return 0 }
    return $parsed
}

function Write-WatchState([string]$State, [string]$Detail, [int]$RunnerPid = 0) {
    $record = [ordered]@{
        timestamp = [DateTime]::UtcNow.ToString('o')
        state = $State
        detail = $Detail
        runner_pid = $RunnerPid
        free_gpu_mib = Get-FreeGpuMiB
        interval_seconds = $IntervalSeconds
    }
    $json = $record | ConvertTo-Json -Compress
    $json | Set-Content -LiteralPath $watchPath -Encoding UTF8
    $json | Add-Content -LiteralPath $watchLog -Encoding UTF8
}

$formalProcess = $null
while ($true) {
    $lock = Read-JsonSafe $lockPath
    $status = Read-JsonSafe $statusPath
    $completeMarker = Join-Path $experimentDir 'BLADE_NCO_CORE_ABLATION_V2_COMPLETE'
    if (Test-Path -LiteralPath $completeMarker) {
        Write-WatchState 'COMPLETE' 'Formal completion marker verified.'
        break
    }
    if ($null -ne $lock -and $lock.state -eq 'FAILED') {
        Write-WatchState 'NEEDS_REPAIR' ([string]$lock.error) ([int]$lock.pid)
        break
    }
    if ($null -ne $formalProcess) {
        if (-not $formalProcess.HasExited) {
            $phase = if ($null -ne $status) { [string]$status.phase } else { 'status unavailable' }
            Write-WatchState 'RUNNING' $phase $formalProcess.Id
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        if ($formalProcess.ExitCode -ne 0) {
            Write-WatchState 'NEEDS_REPAIR' "Formal runner exited with code $($formalProcess.ExitCode)." $formalProcess.Id
            break
        }
        $formalProcess = $null
        Start-Sleep -Seconds 10
        continue
    }
    if ($null -ne $lock -and $lock.state -eq 'ACTIVE' -and (Test-ProcessLive ([int]$lock.pid))) {
        Write-WatchState 'COMMON_PREPARATION_RUNNING' ([string]$status.phase) ([int]$lock.pid)
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    if ($null -eq $lock -or $lock.state -ne 'COMMON_PREPARED') {
        Write-WatchState 'WAITING_FOR_COMMON_PREPARATION' 'Common initialization has not completed.'
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    $freeGpu = Get-FreeGpuMiB
    if ($freeGpu -lt $MinimumFreeGpuMiB) {
        Write-WatchState 'WAITING_FOR_GPU' "GPU free memory ${freeGpu} MiB is below ${MinimumFreeGpuMiB} MiB."
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    $formalProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @('-u', $runner, '--resume') `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    Write-WatchState 'FORMAL_RUN_LAUNCHED' 'GPU gate passed; formal runner started.' $formalProcess.Id
    Start-Sleep -Seconds $IntervalSeconds
}
