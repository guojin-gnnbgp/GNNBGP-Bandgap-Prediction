import json
from pathlib import Path

import pytest

from bandgap_predictor.backends.base import BackendOutcome
from bandgap_predictor.errors import PreflightError
from bandgap_predictor.schemas import (
    BackendId,
    ElectronicClassification,
    PredictionRequest,
    PredictionValue,
    RunStatus,
)
from bandgap_predictor.service import PredictionService


class FakeBackend:
    backend_id = BackendId.GNN_HSE06
    method = "HSE06-like GNN"

    def predict(self, context) -> BackendOutcome:
        assert context.structure.source_path.name == "Si.cif"
        context.progress("predicting")
        return BackendOutcome(
            prediction=PredictionValue(
                gap_ev=1.25,
                classification=ElectronicClassification.INSULATOR,
            ),
            converged=None,
            occupation_summary={},
            settings={"device": "cpu"},
            provenance={"fake": True},
            diagnostics={},
        )


class FailingBackend(FakeBackend):
    def predict(self, context) -> BackendOutcome:
        raise PreflightError("missing backend dependency")


class InterruptedBackend(FakeBackend):
    def predict(self, context) -> BackendOutcome:
        raise KeyboardInterrupt


def request(tmp_path: Path) -> PredictionRequest:
    return PredictionRequest(
        structure_path=Path("examples/Si.cif"),
        backend=BackendId.GNN_HSE06,
        work_root=tmp_path,
    )


def test_service_returns_typed_result_and_retains_artifacts(tmp_path: Path) -> None:
    messages: list[str] = []
    service = PredictionService(
        {BackendId.GNN_HSE06: FakeBackend()},
        progress=messages.append,
    )

    result = service.predict(request(tmp_path))

    assert result.status is RunStatus.SUCCESS
    assert result.prediction.gap_ev == 1.25
    assert result.artifact_path.joinpath("result.json").is_file()
    assert result.artifact_path.joinpath("request.json").is_file()
    assert messages == ["predicting"]


def test_service_retains_failed_result_and_attaches_artifact_path(
    tmp_path: Path,
) -> None:
    service = PredictionService({BackendId.GNN_HSE06: FailingBackend()})

    with pytest.raises(PreflightError) as captured:
        service.predict(request(tmp_path))

    artifact_path = captured.value.artifact_path
    assert artifact_path is not None
    result = json.loads((artifact_path / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["diagnostics"]["error_type"] == "PreflightError"


def test_service_retains_interrupted_result(tmp_path: Path) -> None:
    service = PredictionService({BackendId.GNN_HSE06: InterruptedBackend()})

    with pytest.raises(KeyboardInterrupt):
        service.predict(request(tmp_path))

    result_paths = list(tmp_path.glob("*/result.json"))
    assert len(result_paths) == 1
    result = json.loads(result_paths[0].read_text())
    assert result["status"] == "interrupted"
