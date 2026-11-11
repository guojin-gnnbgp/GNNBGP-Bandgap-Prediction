import json
from pathlib import Path

import numpy as np
import pytest

from bandgap_predictor.abacus.inputs import AbacusMethod, prepare_abacus_input
from bandgap_predictor.assets import AssetRegistry
from bandgap_predictor.structures import load_structure


def test_hse06_input_is_locked_and_gpu_ace(tmp_path: Path) -> None:
    loaded = load_structure(Path("examples/InSb.cif"))
    prepared = prepare_abacus_input(
        tmp_path,
        loaded,
        AssetRegistry.default(),
        AbacusMethod.HSE06,
    )

    assert prepared.input_path.read_text() == Path(
        "tests/abacus/golden/hse06_INPUT"
    ).read_text()
    assert prepared.settings["preset_version"] == "1.1"
    assert prepared.settings["smearing_sigma_ev"] == 0.001
    assert prepared.settings["smearing_sigma"] == pytest.approx(
        0.001 / 13.605693122994,
        rel=0.0,
        abs=1e-18,
    )
    assert (
        "smearing_sigma 7.349864435130999e-05\n"
        in prepared.input_path.read_text()
    )
    assert prepared.settings["ecutwfc"] == 100
    assert prepared.settings["kspacing"] == 0.15
    assert prepared.settings["precision"] == "double"
    assert prepared.settings["nspin"] == 1
    assert prepared.settings["lspinorb"] == 0
    assert prepared.settings["exxace"] == 1
    assert prepared.settings["exx_auto_tiling"] == 1
    assert prepared.model is None


def test_pbe0_input_is_locked_and_gpu_ace(tmp_path: Path) -> None:
    loaded = load_structure(Path("examples/InSb.cif"))
    prepared = prepare_abacus_input(
        tmp_path,
        loaded,
        AssetRegistry.default(),
        AbacusMethod.PBE0,
    )

    assert prepared.input_path.read_text() == Path(
        "tests/abacus/golden/pbe0_INPUT"
    ).read_text()
    assert prepared.settings["preset_version"] == "1.1"
    assert prepared.settings["smearing_sigma_ev"] == 0.001
    assert prepared.settings["dft_functional"] == "PBE0"
    assert prepared.settings["exxace"] == 1
    assert prepared.settings["exx_symmetry_realspace"] == 0
    assert prepared.settings["exx_auto_tiling"] == 1
    assert prepared.settings["exx_tile_memory_budget_mb"] == 0
    assert prepared.model is None


def test_cider_input_uses_valence_only_pseudodojo_model(tmp_path: Path) -> None:
    loaded = load_structure(Path("examples/InSb.cif"))
    prepared = prepare_abacus_input(
        tmp_path,
        loaded,
        AssetRegistry.default(),
        AbacusMethod.CIDER_PBE0,
    )

    assert prepared.input_path.read_text() == Path(
        "tests/abacus/golden/cider_INPUT"
    ).read_text()
    assert prepared.settings["preset_version"] == "1.2"
    assert prepared.settings["smearing_sigma_ev"] == 0.001
    assert prepared.settings["smearing_sigma"] == pytest.approx(
        0.001 / 13.605693122994,
        rel=0.0,
        abs=1e-18,
    )
    assert (
        "smearing_sigma 7.349864435130999e-05\n"
        in prepared.input_path.read_text()
    )
    assert prepared.settings["cider_feature_density"] == "valence-only"
    assert prepared.settings["cider_xmix"] == 0.15
    assert "cider_xmix 0.15\n" in prepared.input_path.read_text()
    assert prepared.model is not None
    assert prepared.model.sha256 == (
        "0bdadb33599a4f5d26baa743b1a5849a1fb63e0ce6b0fdf7ad9479feb2e28734"
    )
    assert (tmp_path / "models" / prepared.model.path.name).is_file()


def test_stru_grouping_is_exactly_reversible_to_input_order(tmp_path: Path) -> None:
    loaded = load_structure(Path("examples/InSb.cif"))
    prepared = prepare_abacus_input(
        tmp_path,
        loaded,
        AssetRegistry.default(),
        AbacusMethod.HSE06,
    )
    site_map = json.loads((tmp_path / "site-map.json").read_text())
    restored = sorted(site_map, key=lambda record: record["input_index"])

    assert [record["symbol"] for record in restored] == (
        loaded.atoms.get_chemical_symbols()
    )
    np.testing.assert_allclose(
        [record["position_angstrom"] for record in restored],
        loaded.atoms.positions,
        atol=1e-12,
        rtol=0.0,
    )
    assert [record["abacus_index"] for record in site_map] == list(range(len(loaded.atoms)))
    assert len(prepared.pseudopotentials) == 2
    assert {asset.element for asset in prepared.pseudopotentials} == {"In", "Sb"}
    assert all((tmp_path / "pseudos" / asset.path.name).is_file() for asset in prepared.pseudopotentials)
    assert "Cartesian_angstrom" in prepared.stru_path.read_text()
