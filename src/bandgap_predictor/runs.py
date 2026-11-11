"""Persistent, atomic prediction-run storage."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .schemas import BackendId, PredictionRequest, PredictionResult
from .structures import LoadedStructure


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class RunDirectory:
    run_id: str
    path: Path

    def copy_original(self, loaded: LoadedStructure) -> Path:
        target = self.path / "input" / loaded.source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(loaded.original_bytes)
        return target

    def write_request(self, request: PredictionRequest) -> None:
        _atomic_json(
            self.path / "request.json",
            request.model_dump(mode="json"),
        )

    def write_result(self, result: PredictionResult) -> None:
        _atomic_json(
            self.path / "result.json",
            result.model_dump(mode="json"),
        )


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def allocate(
        self,
        loaded: LoadedStructure,
        backend: BackendId,
    ) -> RunDirectory:
        run_id = uuid.uuid4().hex
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = self.root / f"{stamp}-{loaded.formula}-{backend.value}-{run_id[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return RunDirectory(run_id=run_id, path=path.resolve())
