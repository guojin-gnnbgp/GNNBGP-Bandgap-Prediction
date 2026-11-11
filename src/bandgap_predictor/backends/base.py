"""Backend-neutral prediction interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..assets import AssetRegistry
from ..runs import RunDirectory
from ..schemas import BackendId, PredictionRequest, PredictionValue
from ..structures import LoadedStructure


@dataclass(frozen=True)
class PredictionContext:
    request: PredictionRequest
    structure: LoadedStructure
    run: RunDirectory
    assets: AssetRegistry
    config_path: Path | None = None
    progress: Callable[[str], None] = lambda _message: None


@dataclass(frozen=True)
class BackendOutcome:
    prediction: PredictionValue
    converged: bool | None
    occupation_summary: dict[str, Any]
    settings: dict[str, Any]
    provenance: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class PredictionBackend(Protocol):
    backend_id: BackendId
    method: str

    def predict(self, context: PredictionContext) -> BackendOutcome: ...
