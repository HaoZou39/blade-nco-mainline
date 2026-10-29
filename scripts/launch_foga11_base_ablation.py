#!/usr/bin/env python
"""Schedule the FOGA-11 Base ablation: one shared baseline, then two GPU queues."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1"
LOGS = RUN / "logs"
RUNNER = ROOT / "scripts" / "run_blade_core_ablation_5c_noanchor_foga11base_v1.py"
QUEUES = {
    "0": ("A0_FULL_BASE", "A2_WITHOUT_SURROGATE_BASE"),
    "1": ("A1_WITHOUT_QD_BASE", "A3_RANDOM_BASE"),
}


def arm_done(arm: str) -> bool:
    return all((RUN / arm / "cycles" / f"C{cycle:02d}" / "COMPLETE.json").is_file() for cycle in range(5))


def run(args: list[str], log_name: str, gpu: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    log = LOGS / log_name
    with log.open("ab", buffering=0) as handle:
        handle.write(f"\n=== {' '.join(args)} GPU {gpu} ===\n".encode())
        process = subprocess.Popen(
            [sys.executable, str(RUNNER), *args],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        code = process.wait()
        handle.write(f"\n=== exit {code} ===\n".encode())
        return code


def run_queue(gpu: str, arms: tuple[str, ...]) -> dict[str, int]:
    codes: dict[str, int] = {}
    for arm in arms:
        if arm_done(arm):
            print(f"{arm} already complete", flush=True)
            codes[arm] = 0
            continue
        print(f"starting {arm} on physical GPU {gpu}", flush=True)
        codes[arm] = run(["--resume", "--arms", arm], f"{arm}.log", gpu)
        print(f"{arm} exit {codes[arm]}", flush=True)
    return codes


def arm_process_alive(arm: str) -> int | None:
    lock = ROOT / "runtime" / f"blade_core_ablation_5c_foga11base_v1_{arm}.lock"
    if not lock.is_file():
        return None
    value = json.loads(lock.read_text(encoding="utf-8"))
    pid = int(value.get("pid", -1))
    if value.get("state") == "ACTIVE" and pid > 0 and Path(f"/proc/{pid}").exists():
        return pid
    return None


def resume_hgb() -> int:
    """Restart the two HGB arms after the arm already using that GPU exits."""
    pairs = (
        ("0", "A2_WITHOUT_SURROGATE_BASE", "A0_FULL_BASE"),
        ("1", "A3_RANDOM_BASE", "A1_WITHOUT_QD_BASE"),
    )

    def one(gpu: str, holder: str, arm: str) -> int:
        if arm_done(arm):
            print(f"{arm} already complete", flush=True)
            return 0
        while True:
            pid = arm_process_alive(holder)
            if pid is None:
                break
            print(f"waiting to start {arm}: {holder} still active as PID {pid} on GPU {gpu}", flush=True)
            time.sleep(30)
        if arm_process_alive(arm):
            print(f"{arm} is already active", flush=True)
            return 0
        print(f"starting {arm} on physical GPU {gpu}", flush=True)
        return run(["--resume", "--arms", arm], f"{arm}.log", gpu)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(one, gpu, holder, arm) for gpu, holder, arm in pairs]
        codes = {pairs[i][2]: future.result() for i, future in enumerate(futures)}
    if all(arm_done(arm) for arms in QUEUES.values() for arm in arms):
        return run(["--finalize"], "finalize.log", "0")
    print(f"HGB resume finished: {codes}", flush=True)
    return 0 if all(code == 0 for code in codes.values()) else 1


def main() -> int:
    if not (RUN / "protocol" / "common_initialization" / "M0_external_benchmark_results.csv").is_file():
        code = run(["--prepare-baseline"], "prepare.log", "0")
        if code != 0:
            print(f"prepare-baseline failed: {code}", flush=True)
            return code
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_queue, gpu, arms) for gpu, arms in QUEUES.items()]
        codes: dict[str, int] = {}
        for future in futures:
            codes.update(future.result())
    if all(arm_done(arm) for arms in QUEUES.values() for arm in arms):
        return run(["--finalize"], "finalize.log", "0")
    failed = {arm: code for arm, code in codes.items() if code}
    print(f"stopped with unfinished or failed arms: {failed or codes}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    if "--resume-hgb" in sys.argv:
        raise SystemExit(resume_hgb())
    raise SystemExit(main())
