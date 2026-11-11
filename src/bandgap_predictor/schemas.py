"""Versioned request and result models shared by all frontends."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BackendId(StrEnum):
    """Stable identifiers for supported scientific backends."""

    GNN_HSE06 = "gnn-hse06"
    ABACUS_HSE06 = "abacus-hse06"
    ABACUS_PBE0 = "abacus-pbe0"
    CIDER_PBE0 = "cider-pbe0"


class RunStatus(StrEnum):
    """Terminal status of an allocated prediction run."""

    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ElectronicClassification(StrEnum):
    """Occupation-aware interpretation of a numerical gap."""

    INSULATOR = "insulator"
    ZERO_GAP = "zero_gap"
    PARTIAL_OCCUPANCY = "partial_occupancy"
    METAL = "metal"
    UNKNOWN = "unknown"


class GapType(StrEnum):
    """Whether the gap is direct or indirect, from the bundled classifier."""

    DIRECT = "direct"
    INDIRECT = "indirect"


class PredictionRequest(BaseModel):
    """A single-backend prediction request."""

    structure_path: Path
    backend: BackendId
    abacus_path: Path | None = None
    work_root: Path = Path("bgp-runs")


class PredictionValue(BaseModel):
    """Numerical band gap and its electronic interpretation."""

    gap_ev: float | None = Field(default=None, ge=0.0)
    classification: ElectronicClassification = ElectronicClassification.UNKNOWN
    # Filled by backends whose model carries a direct/indirect head
    # (currently only ``gnn-hse06``); ``None`` for first-principles backends.
    gap_type: GapType | None = None
    gap_type_probability: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Versioned machine-readable result returned by every backend."""

    schema_version: str = "1.0"
    run_id: str
    status: RunStatus
    backend: BackendId
    method: str
    prediction: PredictionValue
    converged: bool | None
    occupation_summary: dict[str, Any] = Field(default_factory=dict)
    structure: dict[str, Any]
    settings: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    finished_at: str
    elapsed_seconds: float = Field(ge=0.0)
    artifact_path: Path
    warnings: list[str] = Field(default_factory=list)
