"""Locked ABACUS hybrid-functional and native-CIDER prediction backends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..abacus.inputs import AbacusMethod, PreparedAbacusInput, prepare_abacus_input
from ..abacus.parser import parse_abacus_result
from ..abacus.runner import RunProcessResult, run_abacus
from ..config import resolve_abacus_path, verify_abacus_executable
from ..errors import PreflightError
from ..schemas import BackendId
from .base import BackendOutcome, PredictionContext

Runner = Callable[[Path, Path, Callable[[str], None]], RunProcessResult]


class AbacusBackend:
    """Shared local execution template for the two locked ABACUS methods."""

    backend_id: BackendId
    method: str
    method_kind: AbacusMethod

    def __init__(self, runner: Runner = run_abacus) -> None:
        self.runner = runner

    def predict(self, context: PredictionContext) -> BackendOutcome:
        configured = resolve_abacus_path(
            context.request.abacus_path,
            context.config_path,
        )
        if configured is None:
            raise PreflightError("ABACUS executable is not configured")
        executable = Path(configured).expanduser().resolve()
        build = verify_abacus_executable(executable)

        context.progress(f"Preparing locked {self.method} ABACUS calculation")
        prepared = prepare_abacus_input(
            context.run.path,
            context.structure,
            context.assets,
            self.method_kind,
        )
        context.progress(f"Running local {self.method} ABACUS calculation")
        process = self.runner(executable, prepared.workdir, context.progress)
        parsed = parse_abacus_result(prepared.workdir, process.returncode)

        return BackendOutcome(
            prediction=parsed.prediction,
            converged=parsed.converged,
            occupation_summary=parsed.occupation_summary,
            settings=prepared.settings,
            provenance=self._provenance(executable, build.model_dump(), prepared),
            diagnostics={
                **parsed.diagnostics,
                "command": process.command,
                "process_finished_at": process.finished_at,
                "process_started_at": process.started_at,
                "stdout_log": str(process.stdout_path),
            },
        )

    def _provenance(
        self,
        executable: Path,
        build: dict[str, Any],
        prepared: PreparedAbacusInput,
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {
            "cmake_options": build["cmake_options"],
            "executable_path": str(executable),
            "executable_sha256": build["executable_sha256"],
            "features": build["features"],
            "method": self.method,
            "pseudopotential_sha256": {
                asset.element: asset.sha256 for asset in prepared.pseudopotentials
            },
            "source_revisions": build["source_revisions"],
            "toolchain": build["toolchain"],
        }
        if prepared.model is not None:
            provenance["cider_model_sha256"] = prepared.model.sha256
        return provenance


class AbacusHSE06Backend(AbacusBackend):
    backend_id = BackendId.ABACUS_HSE06
    method = "HSE06"
    method_kind = AbacusMethod.HSE06


class AbacusPBE0Backend(AbacusBackend):
    backend_id = BackendId.ABACUS_PBE0
    method = "PBE0"
    method_kind = AbacusMethod.PBE0


class CiderPBE0Backend(AbacusBackend):
    backend_id = BackendId.CIDER_PBE0
    method = "CIDER-PBE0"
    method_kind = AbacusMethod.CIDER_PBE0
