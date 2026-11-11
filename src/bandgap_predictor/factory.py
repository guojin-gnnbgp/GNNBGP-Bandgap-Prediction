"""Application composition root shared by current and future frontends."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .backends.abacus import AbacusHSE06Backend, AbacusPBE0Backend, CiderPBE0Backend
from .backends.gnn import GNNHSE06Backend
from .schemas import BackendId
from .service import PredictionService


def create_service(
    *,
    config_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PredictionService:
    """Create the complete local v1 prediction service."""

    return PredictionService(
        {
            BackendId.GNN_HSE06: GNNHSE06Backend(),
            BackendId.ABACUS_HSE06: AbacusHSE06Backend(),
            BackendId.ABACUS_PBE0: AbacusPBE0Backend(),
            BackendId.CIDER_PBE0: CiderPBE0Backend(),
        },
        config_path=config_path,
        progress=progress,
    )
