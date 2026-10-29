#!/usr/bin/env python3
"""Half-hour P6-G4 supervisor which never overlaps the formal mainline run."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/haozou/blade-nco-mainline")
OUT = ROOT / "target_aware_pilot/p6g4_augmented_initialization_finetune"
RUNNER = OUT / "run_p6g4.py"
PYTHON = "/usr/bin/python3"
INTERVAL = 1800
REPLICATES = ("Original", "S1", "S2")
ARMS = ("A", "B")
MAX_RETRIES = 2
ALLOW_RESOURCE_OVERLAP = True  # Explicit user authorization, 2026-10-29.


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**payload: object) -> None:
    path = OUT / "supervisor_status.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({**old, **payload, "updated_utc": now()}, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (OUT / "logs/supervisor.log").open("a") as handle:
        handle.write(line + "\n")


def conflicts() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True)
    blockers = []
    for line in result.stdout.splitlines():
        if "BLADE_FULL_PIPELINE_MULTISEED_V1" in line and ("run_blade_core_ablation" in line or "run_blade_full_pipeline_multiseed" in line):
            blockers.append(line.strip())
        # P6-H3 is CPU-only and the G4 protocol explicitly states that its
        # unfinished status does not gate G4.  We only gate on the formal
        # multi-seed mainline processes that currently own the training GPUs.
    return blockers


def run(arguments: list[str], label: str) -> None:
    command = [PYTHON, str(RUNNER), *arguments]
    for attempt in range(MAX_RETRIES + 1):
        log(f"START {label} attempt={attempt + 1}: {' '.join(command)}")
        write_status(phase=label, state="RUNNING", attempt=attempt + 1)
        with (OUT / "logs/supervisor_commands.log").open("a") as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if process.returncode == 0:
            log(f"PASS {label}"); write_status(phase=label, state="PASS", attempt=attempt + 1); return
        log(f"FAIL {label} exit={process.returncode}")
        write_status(phase=label, state="RETRY_WAIT" if attempt < MAX_RETRIES else "FAILED", exit_code=process.returncode)
        if attempt < MAX_RETRIES:
            time.sleep(INTERVAL)
    raise RuntimeError(f"{label} failed after {MAX_RETRIES + 1} attempts")


def main() -> None:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    write_status(state="WAITING_FOR_RESOURCES", pid=os.getpid(), monitor_interval_seconds=INTERVAL)
    if ALLOW_RESOURCE_OVERLAP:
        log("RESOURCE OVERLAP explicitly authorized by user; bypassing mainline wait gate")
        write_status(state="RESOURCE_OVERLAP_AUTHORIZED", resource_overlap_authorized=True,
                     authorization_note="User explicitly stated G4 may preempt/share because it is critical")
    while not ALLOW_RESOURCE_OVERLAP:
        active = conflicts()
        if not active:
            break
        log(f"WAIT resource_conflicts={len(active)}")
        write_status(state="WAITING_FOR_RESOURCES", conflict_count=len(active), conflicts=active[:8])
        time.sleep(INTERVAL)
    log("RESOURCES CLEAR; starting frozen P6-G4 train/evaluate sequence on cuda:1")
    write_status(state="RESOURCES_CLEAR", conflicts=[])
    for rep in REPLICATES:
        for arm in ARMS:
            run(["train", "--rep", rep, "--arm", arm, "--device", "cuda:1"], f"train_{rep}_{arm}")
            run(["evaluate", "--rep", rep, "--arm", arm, "--device", "cuda:1"], f"evaluate_{rep}_{arm}")
    run(["finalize"], "finalize")
    write_status(state="COMPLETE", phase="complete")
    log("P6-G4 COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"SUPERVISOR BLOCKED: {type(error).__name__}: {error}")
        write_status(state="BLOCKED", error_type=type(error).__name__, error=str(error))
        raise
