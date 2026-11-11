import json
from pathlib import Path

from bandgap_predictor.runs import RunStore
from bandgap_predictor.schemas import (
    BackendId,
    ElectronicClassification,
    PredictionRequest,
    PredictionResult,
    PredictionValue,
    RunStatus,
)
from bandgap_predictor.structures import load_structure


def test_allocate_retains_original_and_writes_atomic_json(tmp_path: Path) -> None:
    loaded = load_structure(Path("examples/Si.cif"))
    request = PredictionRequest(
        structure_path=loaded.source_path,
        backend=BackendId.GNN_HSE06,
        work_root=tmp_path,
    )
    run = RunStore(tmp_path).allocate(loaded, request.backend)
    result = PredictionResult(
        run_id=run.run_id,
        status=RunStatus.SUCCESS,
        backend=request.backend,
        method="HSE06-like GNN",
        prediction=PredictionValue(
            gap_ev=1.25,
            classification=ElectronicClassification.INSULATOR,
        ),
        converged=None,
        structure={"formula": loaded.formula},
        started_at="2026-08-19T00:00:00+00:00",
        finished_at="2026-08-19T00:00:01+00:00",
        elapsed_seconds=1.0,
        artifact_path=run.path,
    )

    copied = run.copy_original(loaded)
    run.write_request(request)
    run.write_result(result)

    assert copied.read_bytes() == loaded.original_bytes
    assert json.loads((run.path / "request.json").read_text())["backend"] == "gnn-hse06"
    assert json.loads((run.path / "result.json").read_text())["status"] == "success"
    assert not list(run.path.rglob("*.tmp"))


def test_allocations_are_unique_and_use_absolute_paths(tmp_path: Path) -> None:
    loaded = load_structure(Path("examples/InSb.cif"))
    store = RunStore(tmp_path)

    first = store.allocate(loaded, BackendId.CIDER_PBE0)
    second = store.allocate(loaded, BackendId.CIDER_PBE0)

    assert first.run_id != second.run_id
    assert first.path != second.path
    assert first.path.is_absolute() and second.path.is_absolute()
