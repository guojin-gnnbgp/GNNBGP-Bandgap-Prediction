from pathlib import Path

from bandgap_predictor.schemas import (
    BackendId,
    ElectronicClassification,
    PredictionRequest,
    PredictionValue,
)


def test_request_uses_explicit_backend_id(tmp_path: Path) -> None:
    request = PredictionRequest(
        structure_path=tmp_path / "Si.cif",
        backend=BackendId.ABACUS_HSE06,
    )

    assert request.backend.value == "abacus-hse06"
    assert request.work_root == Path("bgp-runs")


def test_zero_gap_retains_partial_occupation_classification() -> None:
    value = PredictionValue(
        gap_ev=0.0,
        classification=ElectronicClassification.PARTIAL_OCCUPANCY,
    )

    assert value.model_dump(mode="json") == {
        "gap_ev": 0.0,
        "classification": "partial_occupancy",
        "gap_type": None,
        "gap_type_probability": None,
    }
