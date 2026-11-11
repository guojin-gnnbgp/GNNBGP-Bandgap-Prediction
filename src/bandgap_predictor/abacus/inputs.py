"""Deterministic ABACUS input preparation from the locked v1 preset."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ase.data import atomic_masses, atomic_numbers

from ..assets import AssetRegistry, VerifiedAsset
from ..structures import LoadedStructure


class AbacusMethod(StrEnum):
    HSE06 = "hse06"
    PBE0 = "pbe0"
    CIDER_PBE0 = "cider-pbe0"


RYDBERG_TO_EV = 13.605693122994
SMEARING_SIGMA_EV = 0.001
SMEARING_SIGMA_RY = SMEARING_SIGMA_EV / RYDBERG_TO_EV


COMMON: Mapping[str, object] = MappingProxyType(
    {
        "basis_type": "pw",
        "calculation": "scf",
        "pseudo_dir": "./pseudos",
        "ecutwfc": 60,
        "ecutexx": 60,
        "kspacing": 0.15,
        "device": "gpu",
        "precision": "single",
        "nspin": 1,
        "lspinorb": 0,
        "symmetry": 1,
        "gamma_only": 0,
        "ks_solver": "dav",
        "scf_thr": 1e-8,
        "scf_nmax": 60,
        "smearing_method": "gaussian",
        "smearing_sigma": SMEARING_SIGMA_RY,
        "mixing_type": "broyden",
        "mixing_beta": 0.7,
        "out_band": 1,
        "out_bgp_result": 1,
        "kpar": 1,
        "bndpar": 1,
        "restart_save": "false",
        "restart_load": "false",
    }
)
HSE06: Mapping[str, object] = MappingProxyType(
    {
        **COMMON,
        "dft_functional": "HSE",
        "exxace": 1,
        "exx_symmetry_realspace": 0,
        "exx_auto_tiling": 1,
        "exx_tile_memory_budget_mb": 0,
    }
)
PBE0: Mapping[str, object] = MappingProxyType(
    {
        **COMMON,
        "dft_functional": "PBE0",
        "exxace": 1,
        "exx_symmetry_realspace": 0,
        "exx_auto_tiling": 1,
        "exx_tile_memory_budget_mb": 0,
    }
)
CIDER_PBE0: Mapping[str, object] = MappingProxyType(
    {
        **COMMON,
        "dft_functional": "PBE",
        "cider_scf": 1,
        "cider_xmix": 0.15,
        "cider_model": "./models/nldf_mgga_cand00_n96_noise0p01.yaml",
    }
)


@dataclass(frozen=True)
class PreparedAbacusInput:
    workdir: Path
    input_path: Path
    stru_path: Path
    settings: dict[str, Any]
    pseudopotentials: list[VerifiedAsset]
    model: VerifiedAsset | None


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render_input(values: Mapping[str, object]) -> str:
    return "INPUT_PARAMETERS\n" + "".join(
        f"{key} {_render_value(value)}\n" for key, value in values.items()
    )


def _unique_symbols(loaded: LoadedStructure) -> list[str]:
    return list(dict.fromkeys(loaded.atoms.get_chemical_symbols()))


def _format_number(value: float) -> str:
    return format(float(value), ".16g")


def _write_stru(
    path: Path,
    loaded: LoadedStructure,
    pseudopotentials: Mapping[str, VerifiedAsset],
) -> list[dict[str, object]]:
    symbols = loaded.atoms.get_chemical_symbols()
    unique_symbols = _unique_symbols(loaded)
    lines = ["ATOMIC_SPECIES"]
    for symbol in unique_symbols:
        mass = atomic_masses[atomic_numbers[symbol]]
        lines.append(
            f"{symbol} {_format_number(mass)} {pseudopotentials[symbol].path.name}"
        )

    lines.extend(["", "LATTICE_CONSTANT", "1.8897261254578281", "", "LATTICE_VECTORS"])
    lines.extend(
        " ".join(_format_number(value) for value in vector)
        for vector in loaded.atoms.cell.array
    )
    lines.extend(["", "ATOMIC_POSITIONS", "Cartesian_angstrom", ""])

    site_map: list[dict[str, object]] = []
    abacus_index = 0
    for symbol in unique_symbols:
        input_indices = [
            index for index, input_symbol in enumerate(symbols) if input_symbol == symbol
        ]
        lines.extend([symbol, "0.0", str(len(input_indices))])
        for input_index in input_indices:
            position = [float(value) for value in loaded.atoms.positions[input_index]]
            lines.append(" ".join(_format_number(value) for value in position))
            site_map.append(
                {
                    "abacus_index": abacus_index,
                    "input_index": input_index,
                    "position_angstrom": position,
                    "symbol": symbol,
                }
            )
            abacus_index += 1
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return site_map


def prepare_abacus_input(
    workdir: Path,
    structure: LoadedStructure,
    assets: AssetRegistry,
    method: AbacusMethod,
) -> PreparedAbacusInput:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    pseudo_dir = workdir / "pseudos"
    pseudo_dir.mkdir(exist_ok=True)

    pseudopotentials = [
        assets.pseudopotential_for(symbol) for symbol in _unique_symbols(structure)
    ]
    pseudo_by_symbol = {
        asset.element: asset for asset in pseudopotentials if asset.element is not None
    }
    for asset in pseudopotentials:
        shutil.copyfile(asset.path, pseudo_dir / asset.path.name)

    if method in {AbacusMethod.HSE06, AbacusMethod.PBE0}:
        values = dict(HSE06 if method is AbacusMethod.HSE06 else PBE0)
        model = None
        settings: dict[str, Any] = {
            "preset_version": "1.1",
            "smearing_sigma_ev": SMEARING_SIGMA_EV,
            **values,
        }
    elif method is AbacusMethod.CIDER_PBE0:
        values = dict(CIDER_PBE0)
        model = assets.verify("cider-pbe0-model")
        model_dir = workdir / "models"
        model_dir.mkdir(exist_ok=True)
        shutil.copyfile(model.path, model_dir / model.path.name)
        settings = {
            "preset_version": "1.2",
            "smearing_sigma_ev": SMEARING_SIGMA_EV,
            "cider_feature_density": "valence-only",
            **values,
        }
    else:  # pragma: no cover - StrEnum exhaustiveness guard
        raise ValueError(f"unsupported ABACUS method: {method}")

    input_path = workdir / "INPUT"
    input_path.write_text(render_input(values), encoding="utf-8")
    stru_path = workdir / "STRU"
    site_map = _write_stru(stru_path, structure, pseudo_by_symbol)
    (workdir / "site-map.json").write_text(
        json.dumps(site_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PreparedAbacusInput(
        workdir=workdir,
        input_path=input_path,
        stru_path=stru_path,
        settings=settings,
        pseudopotentials=pseudopotentials,
        model=model,
    )
