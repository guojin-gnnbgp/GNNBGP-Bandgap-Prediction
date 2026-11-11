import json
from pathlib import Path

import pytest

from bandgap_predictor.backends.gnn import GNNHSE06Backend
from bandgap_predictor.schemas import BackendId, GapType, PredictionRequest
from bandgap_predictor.service import PredictionService


def test_gnn_backend_declares_locked_legacy_contract() -> None:
    backend = GNNHSE06Backend()

    assert backend.backend_id is BackendId.GNN_HSE06
    assert backend.method == (
        "HSE06-like ScalarPropertyMACE with frozen direct/indirect head "
        "(20250421 backbone low-LR fine-tuned to the InSb VASP HSE06+SOC "
        "(aexx-tuned) reference 0.241 eV; head trained on "
        "semiconductor-only labels)"
    )
    assert "legacy/provisional" in backend.warning
    assert "pickle" in backend.warning


@pytest.mark.gnn
@pytest.mark.parametrize("name", ["InSb", "Si", "GaAs", "MgO"])
def test_gnn_regression(name: str, tmp_path: Path) -> None:
    # The scalar head is bit-identical to the mid-term 20250421 model, so the
    # pinned-environment gap baselines are unchanged by the classifier swap.
    # Values are reproducible to 1e-6 on the locked uv stack
    # (mace-torch 0.3.9 / torch 2.10); other stacks differ in the last digits.
    baselines = json.loads(Path("tests/gnn/baselines.json").read_text())
    expected = baselines["predictions_ev"][name]
    expected_type = baselines["predictions_gap_type"][name]
    source = Path("examples") / f"{name}.cif"
    original = source.read_bytes()
    service = PredictionService({BackendId.GNN_HSE06: GNNHSE06Backend()})

    result = service.predict(
        PredictionRequest(
            structure_path=source,
            backend=BackendId.GNN_HSE06,
            work_root=tmp_path,
        )
    )

    assert result.prediction.gap_ev == pytest.approx(expected, abs=1e-6)
    assert result.prediction.gap_type is GapType(expected_type)
    assert result.prediction.gap_type_probability is not None
    assert 0.0 <= result.prediction.gap_type_probability <= 1.0
    assert "legacy/provisional" in " ".join(result.warnings)
    assert result.provenance["model_sha256"] == (
        "299aa263144a237fef2ecc1463287819130ea1ebb5474ab771b54292827ef748"
    )
    assert source.read_bytes() == original


@pytest.mark.gnn
def test_gnn_reports_contract_material_as_direct(tmp_path: Path) -> None:
    """The contract test material InSb is a Gamma-point direct-gap system."""
    service = PredictionService({BackendId.GNN_HSE06: GNNHSE06Backend()})
    result = service.predict(
        PredictionRequest(
            structure_path=Path("examples/InSb.cif"),
            backend=BackendId.GNN_HSE06,
            work_root=tmp_path,
        )
    )
    assert result.prediction.gap_type is GapType.DIRECT
