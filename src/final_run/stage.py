"""Atomic, hash-verifying stage and run-lock primitives."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def output_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths}


def verify_stage(record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "COMPLETE":
        raise RuntimeError(f"stage is not complete: {record_path}")
    for value in record["outputs"].values():
        path = Path(value["path"])
        if not path.is_file() or path.stat().st_size != value["bytes"] or sha256_file(path) != value["sha256"]:
            raise RuntimeError(f"completed stage output hash mismatch: {path}")
    return record


@contextmanager
def atomic_stage(directory: Path, name: str, inputs: dict[str, str], *, resume: bool) -> Iterator[list[Path] | None]:
    directory.mkdir(parents=True, exist_ok=True)
    done = directory / "STAGE_DONE.json"
    if done.exists():
        if not resume:
            raise FileExistsError(f"completed stage exists without --resume: {done}")
        record = verify_stage(done)
        if record.get("input_hashes") != inputs:
            raise RuntimeError(f"stage input hash mismatch: {name}")
        yield None
        return
    started_utc = utc_now()
    started = time.perf_counter()
    outputs: list[Path] = []
    try:
        yield outputs
    except BaseException as exc:
        atomic_json(directory / "STAGE_FAILED.json", {
            "stage": name, "status": "FAILED", "input_hashes": inputs,
            "start_timestamp": started_utc, "end_timestamp": utc_now(),
            "wall_seconds": time.perf_counter() - started, "exception": repr(exc),
        })
        raise
    records = output_records(outputs)
    atomic_json(done, {
        "stage": name, "status": "COMPLETE", "input_hashes": inputs,
        "outputs": {key: {"path": key, **value} for key, value in records.items()},
        "start_timestamp": started_utc, "end_timestamp": utc_now(),
        "wall_seconds": time.perf_counter() - started,
    })


def acquire_run_lock(run_dir: Path, run_id: str, command: str) -> Path:
    # The Windows launcher pre-creates only logs/ so Start-Process can attach
    # stdout/stderr before Python begins.  The exclusive lock remains the
    # authoritative duplicate-writer guard.
    run_dir.mkdir(parents=True, exist_ok=True)
    unexpected=[path.name for path in run_dir.iterdir() if path.name!="logs"]
    if unexpected:
        raise FileExistsError(f"conflicting formal run contents before lock: {unexpected}")
    lock = run_dir / "RUN.lock"
    payload = {"pid": os.getpid(), "launch_timestamp": utc_now(), "command": command, "run_id": run_id, "host": socket.gethostname()}
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    atomic_json(run_dir / "LAUNCH_RECORD.json", payload)
    return lock
