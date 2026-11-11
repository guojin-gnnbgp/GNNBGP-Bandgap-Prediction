import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bandgap_predictor.abacus.runner import run_abacus


def fake_executables(tmp_path: Path) -> tuple[Path, Path]:
    abacus = tmp_path / "fake_abacus.py"
    shutil.copyfile(Path("tests/abacus/fake_abacus.py"), abacus)
    abacus.chmod(0o755)
    mpirun = tmp_path / "fake_mpirun.py"
    mpirun.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "assert sys.argv[1:3] == ['-np', '1']\n"
        "os.execv(sys.argv[3], sys.argv[3:])\n"
    )
    mpirun.chmod(0o755)
    return abacus, mpirun


def test_runner_uses_one_rank_and_one_thread(tmp_path: Path) -> None:
    abacus, mpirun = fake_executables(tmp_path)
    messages: list[str] = []

    result = run_abacus(abacus, tmp_path, progress=messages.append, mpirun=mpirun)

    assert result.command[-3:] == ["-np", "1", str(abacus)]
    assert json.loads((tmp_path / "fake-env.json").read_text())[
        "OMP_NUM_THREADS"
    ] == "1"
    assert (tmp_path / "stdout.log").read_text().startswith("ABACUS fake startup")
    assert messages == ["ITER 1 etot=-1.0", "#SCF IS CONVERGED#"]
    process_record = json.loads((tmp_path / "process.json").read_text())
    assert process_record["returncode"] == 0
    assert process_record["interrupted"] is False


def test_interrupt_terminates_process_group_and_preserves_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abacus, mpirun = fake_executables(tmp_path)
    monkeypatch.setenv("BGP_FAKE_MODE", "hang")
    driver = subprocess.Popen(
        [
            sys.executable,
            "tests/abacus/interrupt_driver.py",
            str(abacus),
            str(mpirun),
            str(tmp_path),
        ],
        env=os.environ.copy(),
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "fake-pid.txt").is_file():
            if driver.poll() is not None:
                pytest.fail(f"interrupt driver exited early with {driver.returncode}")
            if time.monotonic() >= deadline:
                pytest.fail("fake ABACUS process did not start")
            time.sleep(0.01)
        driver.send_signal(signal.SIGINT)
        assert driver.wait(timeout=5) == 130
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()

    assert (tmp_path / "stdout.log").exists()
    process_record = json.loads((tmp_path / "process.json").read_text())
    assert process_record["interrupted"] is True
    pid = int((tmp_path / "fake-pid.txt").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
