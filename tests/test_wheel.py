from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from ase.io import read


def test_wheel_installs_cli_assets_examples_and_source_notices(tmp_path: Path) -> None:
    environment = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"}
    configured = os.environ.get("BGP_WHEEL")
    candidates = [Path(configured)] if configured else list(
        Path("dist").glob("bandgap_predictor-*.whl")
    )
    assert candidates, "build the release wheel with `uv build` before this test"
    wheel = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    installed = tmp_path / "installed"

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        archive.extractall(installed)

    assert "bandgap_predictor/data/manifest.json" in names
    manifest = json.loads(
        (installed / "bandgap_predictor/data/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["assets"]) == 74

    for system in ("InSb", "Si", "GaAs", "MgO"):
        example = installed / "bandgap_predictor/examples" / f"{system}.cif"
        assert len(read(example)) == 8

    notices = installed / "bandgap_predictor/data/abacus-source"
    assert (notices / "UPSTREAM.md").is_file()
    assert (notices / "INTEGRATION_PATCHES.md").is_file()
    assert (notices / "LICENSE").is_file()

    forbidden_suffixes = (".o", ".a", ".so")
    assert not any(name.startswith("vendor/abacus/") for name in names)
    assert not any(name.endswith(forbidden_suffixes) for name in names)
    assert not any("OUT.ABACUS" in name or name.endswith("abacus_pw_gpu") for name in names)

    command_environment = {
        **environment,
        "PYTHONPATH": os.pathsep.join(
            [str(installed), environment.get("PYTHONPATH", "")]
        ),
    }
    help_result = subprocess.run(
        [sys.executable, "-m", "bandgap_predictor.cli", "--help"],
        check=False,
        cwd=tmp_path,
        env=command_environment,
        capture_output=True,
        text=True,
    )
    assets_result = subprocess.run(
        [sys.executable, "-m", "bandgap_predictor.cli", "assets", "verify"],
        check=False,
        cwd=tmp_path,
        env=command_environment,
        capture_output=True,
        text=True,
    )
    entry_points = next(installed.glob("bandgap_predictor-*.dist-info/entry_points.txt"))

    assert help_result.returncode == 0
    assert "predict" in help_result.stdout
    assert assets_result.returncode == 0
    assert "Verified 74 bundled assets" in assets_result.stdout
    assert "bgp = bandgap_predictor.cli:main" in entry_points.read_text(
        encoding="utf-8"
    )
