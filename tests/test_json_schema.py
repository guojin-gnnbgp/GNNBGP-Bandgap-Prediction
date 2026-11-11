import json
from pathlib import Path

from bandgap_predictor.backends.base import BackendOutcome
from bandgap_predictor.schemas import (
    BackendId,
    ElectronicClassification,
    PredictionRequest,
    PredictionResult,
    PredictionValue,
)
from bandgap_predictor.service import PredictionService


class _Backend:
    backend_id = BackendId.GNN_HSE06
    method = "schema-test"

    def predict(self, context) -> BackendOutcome:
        return BackendOutcome(
            prediction=PredictionValue(
                gap_ev=1.0,
                classification=ElectronicClassification.INSULATOR,
            ),
            converged=None,
            occupation_summary={},
            settings={},
            provenance={},
            diagnostics={},
        )


def test_retained_json_round_trips_through_versioned_result_schema(
    tmp_path: Path,
) -> None:
    result = PredictionService({_Backend.backend_id: _Backend()}).predict(
        PredictionRequest(
            structure_path=Path("examples/Si.cif"),
            backend=BackendId.GNN_HSE06,
            work_root=tmp_path,
        )
    )
    retained = result.artifact_path / "result.json"
    payload = json.loads(retained.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert PredictionResult.model_validate(payload) == result
