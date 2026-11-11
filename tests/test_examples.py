import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from ase.io import read


def test_four_example_files_match_manifest() -> None:
    root = Path("examples")
    manifest = json.loads((root / "manifest.json").read_text())

    assert set(manifest) == {"InSb", "Si", "GaAs", "MgO"}
    for name, record in manifest.items():
        path = root / f"{name}.cif"
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
        atoms = read(path)
        assert len(atoms) == record["atom_count"]
        assert atoms.cell.rank == 3
        assert all(atoms.pbc)


def test_rebuilding_examples_is_byte_reproducible(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "tools/build_examples.py", "--output", str(tmp_path)],
        check=True,
    )
    generated_manifest = json.loads((tmp_path / "manifest.json").read_text())
    committed_root = Path("examples")
    committed_manifest = json.loads((committed_root / "manifest.json").read_text())

    assert generated_manifest == committed_manifest
    for name in committed_manifest:
        assert (tmp_path / f"{name}.cif").read_bytes() == (
            committed_root / f"{name}.cif"
        ).read_bytes()
