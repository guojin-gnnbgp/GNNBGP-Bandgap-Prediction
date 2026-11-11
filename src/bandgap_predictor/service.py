"""Backend-neutral band-gap prediction orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from .assets import AssetRegistry
from .backends.base import (
    BackendOutcome,
    PredictionBackend,
    PredictionContext,
)
from .errors import BGPError, PreflightError
from .runs import RunDirectory, RunStore
from .schemas import (
    BackendId,
    PredictionRequest,
    PredictionResult,
    PredictionValue,
    RunStatus,
)
from .structures import LoadedStructure, load_structure


class PredictionService:
    """Validate, dispatch, retain, and serialize one prediction request."""

    def __init__(
        self,
        backends: Mapping[BackendId, PredictionBackend],
        *,
        assets: AssetRegistry | None = None,
        config_path: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.backends = dict(backends)
        self.assets = assets if assets is not None else AssetRegistry.default()
        self.config_path = config_path
        self.progress = progress if progress is not None else lambda _message: None

    def predict(self, request: PredictionRequest) -> PredictionResult:
        backend = self.backends.get(request.backend)
        if backend is None:
            raise PreflightError(f"prediction backend is unavailable: {request.backend.value}")

        structure = load_structure(request.structure_path)
        run = RunStore(request.work_root).allocate(structure, request.backend)
        run.copy_original(structure)
        run.write_request(request)
        started_at = datetime.now(UTC)
        started_clock = monotonic()
        context = PredictionContext(
            request=request,
            structure=structure,
            run=run,
            assets=self.assets,
            config_path=self.config_path,
            progress=self.progress,
        )

        try:
            outcome = backend.predict(context)
        except BGPError as error:
            result = self._terminal_result(
                request=request,
                backend=backend,
                structure=structure,
                run=run,
                started_at=started_at,
                started_clock=started_clock,
                status=RunStatus.FAILED,
                diagnostics={
                    "error_message": str(error),
                    "error_type": type(error).__name__,
                },
            )
            run.write_result(result)
            error.artifact_path = run.path
            raise
        except KeyboardInterrupt:
            result = self._terminal_result(
                request=request,
                backend=backend,
                structure=structure,
                run=run,
                started_at=started_at,
                started_clock=started_clock,
                status=RunStatus.INTERRUPTED,
                diagnostics={"error_type": "KeyboardInterrupt"},
            )
            run.write_result(result)
            raise

        result = self._success_result(
            request=request,
            backend=backend,
            structure=structure,
            run=run,
            started_at=started_at,
            started_clock=started_clock,
            outcome=outcome,
        )
        run.write_result(result)
        return result

    @staticmethod
    def _structure_record(structure: LoadedStructure) -> dict[str, object]:
        return {
            "atom_count": len(structure.atoms),
            "formula": structure.formula,
            "original_sha256": structure.original_sha256,
            "source_path": str(structure.source_path),
            "structure_sha256": structure.structure_sha256,
        }

    def _success_result(
        self,
        *,
        request: PredictionRequest,
        backend: PredictionBackend,
        structure: LoadedStructure,
        run: RunDirectory,
        started_at: datetime,
        started_clock: float,
        outcome: BackendOutcome,
    ) -> PredictionResult:
        finished_at = datetime.now(UTC)
        return PredictionResult(
            run_id=run.run_id,
            status=RunStatus.SUCCESS,
            backend=request.backend,
            method=backend.method,
            prediction=outcome.prediction,
            converged=outcome.converged,
            occupation_summary=outcome.occupation_summary,
            structure=self._structure_record(structure),
            settings=outcome.settings,
            provenance=outcome.provenance,
            diagnostics=outcome.diagnostics,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            elapsed_seconds=monotonic() - started_clock,
            artifact_path=run.path,
            warnings=outcome.warnings,
        )

    def _terminal_result(
        self,
        *,
        request: PredictionRequest,
        backend: PredictionBackend,
        structure: LoadedStructure,
        run: RunDirectory,
        started_at: datetime,
        started_clock: float,
        status: RunStatus,
        diagnostics: dict[str, object],
    ) -> PredictionResult:
        finished_at = datetime.now(UTC)
        return PredictionResult(
            run_id=run.run_id,
            status=status,
            backend=request.backend,
            method=backend.method,
            prediction=PredictionValue(),
            converged=None,
            structure=self._structure_record(structure),
            diagnostics=diagnostics,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            elapsed_seconds=monotonic() - started_clock,
            artifact_path=run.path,
        )
