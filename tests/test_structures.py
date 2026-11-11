from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from bandgap_predictor.errors import InputError
from bandgap_predictor.structures import load_structure, write_site_map


def test_load_structure_preserves_cell_positions_and_order(tmp_path: Path) -> None:
    atoms = Atoms(
        "SbIn",
        positions=[[0.0, 0.0, 0.0], [1.2, 1.3, 1.4]],
        cell=np.diag([6.4, 6.5, 6.6]),
        pbc=True,
    )
    path = tmp_path / "input.extxyz"
    write(path, atoms)

    loaded = load_structure(path)

    assert loaded.atoms.get_chemical_symbols() == ["Sb", "In"]
    np.testing.assert_allclose(loaded.atoms.cell.array, atoms.cell.array)
    np.testing.assert_allclose(loaded.atoms.positions, atoms.positions)
    assert loaded.original_sha256 == loaded.sha256_bytes(loaded.original_bytes)
    assert len(loaded.structure_sha256) == 64


def test_equivalent_structure_bytes_have_same_structure_hash(tmp_path: Path) -> None:
    atoms = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]], cell=[5, 5, 5], pbc=True)
    xyz_path = tmp_path / "Si.extxyz"
    cif_path = tmp_path / "Si.cif"
    write(xyz_path, atoms)
    write(cif_path, atoms)

    xyz = load_structure(xyz_path)
    cif = load_structure(cif_path)

    assert xyz.original_sha256 != cif.original_sha256
    assert xyz.structure_sha256 == cif.structure_sha256


def test_write_site_map_retains_input_indices(tmp_path: Path) -> None:
    atoms = Atoms("SbInSb", positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], cell=[6, 6, 6], pbc=True)
    source = tmp_path / "ordered.extxyz"
    write(source, atoms)
    loaded = load_structure(source)

    target = tmp_path / "site-map.json"
    write_site_map(loaded, target)

    assert target.read_text().count('"input_index"') == 3
    assert '"symbol": "In"' in target.read_text()


def test_rejects_nonperiodic_structure(tmp_path: Path) -> None:
    path = tmp_path / "molecule.xyz"
    write(path, Atoms("Si", positions=[[0, 0, 0]], pbc=False))

    with pytest.raises(InputError, match="three-dimensional periodic"):
        load_structure(path)


def test_rejects_magnetic_initialization(tmp_path: Path) -> None:
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    atoms.set_initial_magnetic_moments([2.0])
    path = tmp_path / "Fe.extxyz"
    write(path, atoms)

    with pytest.raises(InputError, match="non-spin-polarized"):
        load_structure(path)
