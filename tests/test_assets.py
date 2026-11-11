from pathlib import Path

import pytest

from bandgap_predictor.assets import AssetRegistry, VerifiedAsset
from bandgap_predictor.errors import PreflightError


def test_default_registry_verifies_authoritative_models() -> None:
    registry = AssetRegistry.default()

    assert (
        registry.verify("gnn-hse06-model").sha256
        == "de1e759d1c7d6d15d9b317010f11f270e31769bab0172d5d2507ed5160697317"
    )
    assert (
        registry.verify("gnn-feature-map").sha256
        == "cb90bd2257dbf8cfc4e6b2b6d4348d4616e8e9680f1339b1461137032e0f1894"
    )
    assert (
        registry.verify("cider-pbe0-model").sha256
        == "0bdadb33599a4f5d26baa743b1a5849a1fb63e0ce6b0fdf7ad9479feb2e28734"
    )


def test_registry_contains_complete_pseudodojo_collection() -> None:
    registry = AssetRegistry.default()
    pseudos = [asset for asset in registry.assets if asset.kind == "pseudopotential"]

    assert len(pseudos) == 71
    assert {
        registry.pseudopotential_for(element).element
        for element in ("In", "Sb", "Si", "Ga", "As", "Mg", "O")
    } == {"In", "Sb", "Si", "Ga", "As", "Mg", "O"}


def test_registry_rejects_tampered_asset(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text("changed")
    registry = AssetRegistry(
        tmp_path,
        [
            VerifiedAsset(
                logical_id="model",
                path=Path("model.yaml"),
                kind="model",
                sha256="0" * 64,
                size=7,
            )
        ],
    )

    with pytest.raises(PreflightError, match="integrity check failed"):
        registry.verify("model")
