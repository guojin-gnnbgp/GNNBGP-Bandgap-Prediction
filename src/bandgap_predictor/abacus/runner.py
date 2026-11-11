"""Interrupt-safe synchronous local ABACUS execution."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..errors import PreflightError


@dataclass(frozen=True)
class RunProcessResult:
    command: list[str]
    returncode: int
    started_at: str
    finished_at: str
    stdout_path: Path
    interrupted: bool = False


def is_progress_line(line: str) -> bool:
    normalized = line.strip().upper()
    return (
        normalized.startswith("ITER")
        or "SCF IS CONVERGED" in normalized
        or "SCF IS NOT CONVERGED" in normalized
        or "E_GAP(K)" in normalized
    )


def _write_process_record(workdir: Path, result: RunProcessResult) -> None:
    target = workdir / "process.json"
    temporary = target.with_name(target.name + ".tmp")
    payload = asdict(result)
    payload["stdout_path"] = str(result.stdout_path)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _resolve_mpirun(mpirun: Path | None) -> Path:
    candidate = Path(mpirun) if mpirun is not None else None
    if candidate is None:
        discovered = shutil.which("mpirun")
        if discovered is None:
            raise PreflightError("mpirun is required for local ABACUS execution")
        candidate = Path(discovered)
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise PreflightError(f"mpirun is not executable: {candidate}")
    return candidate


def run_abacus(
    executable: Path,
    workdir: Path,
    progress: Callable[[str], None],
    *,
    mpirun: Path | None = None,
) -> RunProcessResult:
    """Run one local MPI rank and retain its complete combined output."""

    executable = Path(executable).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PreflightError(f"ABACUS executable is not executable: {executable}")
    launcher = _resolve_mpirun(mpirun)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    stdout_path = workdir / "stdout.log"
    command = [str(launcher), "-np", "1", str(executable)]
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    started_at = datetime.now(UTC).isoformat()

    with stdout_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if is_progress_line(line):
                    progress(line.rstrip())
            returncode = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            interrupted_result = RunProcessResult(
                command=command,
                returncode=process.returncode if process.returncode is not None else -1,
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                stdout_path=stdout_path,
                interrupted=True,
            )
            _write_process_record(workdir, interrupted_result)
            raise
        finally:
            process.stdout.close()

    result = RunProcessResult(
        command=command,
        returncode=returncode,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        stdout_path=stdout_path,
    )
    _write_process_record(workdir, result)
    return result
