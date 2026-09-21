param(
    [int]$TargetPid = 47584,
    [int]$PollSeconds = 10
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $repo 'runtime'
$cycleMarker = Join-Path $repo 'runs\blade_v2_full_seed2026\cycles\C14\CYCLE_COMPLETE.json'
$checkpoint = Join-Path $repo 'runs\blade_v2_full_seed2026\cycles\C14\09_bq_training\M15_E256.tsp.best'
$transitionLog = Join-Path $runtime 'blade_v2_m15_transition.json'
$python = 'D:\conda_envs\bq-nco\python.exe'

function Write-Transition {
    param([hashtable]$Value)
    $Value.timestamp = (Get-Date).ToUniversalTime().ToString('o')
    $temporary = $transitionLog + '.tmp'
    Set-Content -LiteralPath $temporary -Value ($Value | ConvertTo-Json -Depth 5) -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $transitionLog -Force
}

Write-Transition @{ status = 'WAITING_FOR_C14'; target_pid = $TargetPid }

while (-not (Test-Path -LiteralPath $cycleMarker)) {
    Start-Sleep -Seconds $PollSeconds
}

if (-not (Test-Path -LiteralPath $checkpoint)) {
    Write-Transition @{ status = 'FAILED'; reason = 'C14 complete marker exists but M15 checkpoint is missing' }
    exit 1
}

$target = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $TargetPid) -ErrorAction SilentlyContinue
if ($null -ne $target -and $target.CommandLine -match 'run_blade_v2\.py') {
    Stop-Process -Id $TargetPid -Force
    Wait-Process -Id $TargetPid -Timeout 30 -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2
$active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'run_blade_v2\.py' } |
    Select-Object -First 1

if ($null -ne $active) {
    Write-Transition @{
        status = 'NEW_RUNNER_ALREADY_ACTIVE'
        active_pid = [int]$active.ProcessId
        checkpoint = $checkpoint
    }
    exit 0
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $runtime ('blade_v2_m15_resume_' + $stamp + '_stdout.log')
$stderr = Join-Path $runtime ('blade_v2_m15_resume_' + $stamp + '_stderr.log')
$started = Start-Process -FilePath $python -ArgumentList @('scripts\run_blade_v2.py', '--config', 'configs\blade_v2_protocol.yaml', '--resume') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

Write-Transition @{
    status = 'M15_SAVED_GP_ONLY_RESUMED'
    stopped_pid = $TargetPid
    resumed_pid = [int]$started.Id
    checkpoint = $checkpoint
    stdout = $stdout
    stderr = $stderr
}
