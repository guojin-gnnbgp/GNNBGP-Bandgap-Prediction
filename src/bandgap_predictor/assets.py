"""Hash-verified access to bundled models and pseudopotentials."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .errors import PreflightError


class VerifiedAsset(BaseModel):
    logical_id: str
    path: Path
    kind: str
    sha256: str
    size: int
    element: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssetRegistry:
    """A fail-closed registry of immutable package-owned scientific files."""

    def __init__(self, root: Path, records: list[VerifiedAsset]):
        self.root = Path(root).resolve()
        self.assets = records
        self._by_id = {asset.logical_id: asset for asset in records}
        if len(self._by_id) != len(records):
            raise PreflightError("asset manifest contains duplicate logical identifiers")

    def verify(self, logical_id: str) -> VerifiedAsset:
        try:
            asset = self._by_id[logical_id]
        except KeyError as error:
            raise PreflightError(f"asset is not registered: {logical_id}") from error

        path = self.root / asset.path
        if not path.is_file():
            raise PreflightError(f"asset integrity check failed: {logical_id}: missing")
        data = path.read_bytes()
        if len(data) != asset.size or sha256(data).hexdigest() != asset.sha256:
            raise PreflightError(f"asset integrity check failed: {logical_id}")
        return asset.model_copy(update={"path": path.resolve()})

    def pseudopotential_for(self, symbol: str) -> VerifiedAsset:
        matches = [
            asset
            for asset in self.assets
            if asset.kind == "pseudopotential" and asset.element == symbol
        ]
        if len(matches) != 1:
            raise PreflightError(
                f"no unique bundled PseudoDojo potential for {symbol}"
            )
        return self.verify(matches[0].logical_id)

    @classmethod
    def default(cls) -> AssetRegistry:
        package_root = Path(__file__).resolve().parent / "data"
        source_root = Path(__file__).resolve().parents[2] / "assets"
        root = package_root if (package_root / "manifest.json").is_file() else source_root
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise PreflightError(f"bundled asset manifest is missing: {manifest_path}")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "1.0":
                raise ValueError("unsupported schema version")
            records = [
                VerifiedAsset.model_validate(item) for item in payload["assets"]
            ]
        except (OSError, KeyError, TypeError, ValueError, ValidationError) as error:
            raise PreflightError(f"bundled asset manifest is invalid: {error}") from error
        return cls(root, records)
