"""Build the package's deterministic crystal-structure examples with ASE."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from ase.build import bulk
from ase.io import write

CASE_SPECS = {
    "GaAs": ("zincblende", 5.6533),
    "InSb": ("zincblende", 6.479),
    "MgO": ("rocksalt", 4.212),
    "Si": ("diamond", 5.431),
}


def build_examples(root: Path) -> dict[str, dict[str, object]]:
    """Write all examples under *root* and return their byte manifest."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}

    for name, (prototype, lattice_constant) in CASE_SPECS.items():
        atoms = bulk(name, prototype, a=lattice_constant, cubic=True)
        path = root / f"{name}.cif"
        write(path, atoms)
        manifest[name] = {
            "atom_count": len(atoms),
            "prototype": prototype,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples",
        help="directory in which to write the CIF files and manifest",
    )
    arguments = parser.parse_args()
    build_examples(arguments.output)


if __name__ == "__main__":
    main()
