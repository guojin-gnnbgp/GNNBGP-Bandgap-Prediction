"""Lossless structure ingestion and deterministic provenance hashes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ase import Atoms
from ase.io import read

from .errors import InputError


@dataclass(frozen=True)
class LoadedStructure:
    """A validated structure together with its immutable input evidence."""

    source_path: Path
    original_bytes: bytes
    atoms: Atoms
    original_sha256: str
    structure_sha256: str
    formula: str

    @staticmethod
    def sha256_bytes(data: bytes) -> str:
        return sha256(data).hexdigest()


def _rounded_rows(values: object) -> list[list[float]]:
    return [[round(float(value), 12) for value in row] for row in values]  # type: ignore[union-attr]


def _structure_digest(atoms: Atoms) -> str:
    canonical = json.dumps(
        {
            "cell_angstrom": _rounded_rows(atoms.cell.array),
            "pbc": [bool(value) for value in atoms.pbc],
            "positions_angstrom": _rounded_rows(atoms.positions),
            "symbols": atoms.get_chemical_symbols(),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def load_structure(path: Path) -> LoadedStructure:
    """Load exactly one 3-D periodic, nonmagnetic structure without changing it."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise InputError(f"structure file does not exist: {source_path}")

    original_bytes = source_path.read_bytes()
    try:
        images = read(source_path, index=":")
    except Exception as error:
        raise InputError(f"could not read crystal structure {source_path}: {error}") from error
    if len(images) != 1:
        raise InputError(f"expected exactly one crystal structure, found {len(images)}")

    atoms = images[0]
    if len(atoms) == 0:
        raise InputError("structure contains no atoms")
    if not all(bool(value) for value in atoms.pbc) or atoms.cell.rank != 3:
        raise InputError("structure must have a complete three-dimensional periodic cell")
    if any(abs(float(value)) > 1e-12 for value in atoms.get_initial_magnetic_moments()):
        raise InputError("version 1 supports only non-spin-polarized structures")

    return LoadedStructure(
        source_path=source_path,
        original_bytes=original_bytes,
        atoms=atoms.copy(),
        original_sha256=sha256(original_bytes).hexdigest(),
        structure_sha256=_structure_digest(atoms),
        formula=atoms.get_chemical_formula(mode="hill"),
    )


def write_site_map(loaded: LoadedStructure, path: Path) -> None:
    """Persist original logical site order for later ABACUS index mapping."""

    records = [
        {
            "input_index": index,
            "position_angstrom": [float(value) for value in loaded.atoms.positions[index]],
            "symbol": symbol,
        }
        for index, symbol in enumerate(loaded.atoms.get_chemical_symbols())
    ]
    Path(path).write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
