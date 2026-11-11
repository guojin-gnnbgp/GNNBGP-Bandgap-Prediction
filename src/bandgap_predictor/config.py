"""ABACUS executable configuration and build-manifest verification."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from platformdirs import user_config_path
from pydantic import BaseModel, Field, ValidationError

from .errors import PreflightError

REQUIRED_ABACUS_FEATURES = frozenset(
    {
        "bgp-result-json",
        "cuda",
        "gpu-exx",
        "libxc",
        "mpi",
        "native-cider",
        "pw",
    }
)


class BuildManifest(BaseModel):
    """Provenance contract emitted beside a package-owned ABACUS binary."""

    schema_version: str
    executable_sha256: str
    executable_name: str
    source_revisions: list[str]
    cmake_options: dict[str, bool | str]
    features: list[str]
    toolchain: dict[str, str] = Field(default_factory=dict)
    built_at: str | None = None


def resolve_abacus_path(
    cli_path: Path | None = None,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve ABACUS using CLI, config, then environment precedence."""

    if cli_path is not None:
        return Path(cli_path)

    resolved_config = (
        Path(config_path)
        if config_path is not None
        else user_config_path("bandgap-predictor") / "config.toml"
    )
    if resolved_config.is_file():
        try:
            payload = tomllib.loads(resolved_config.read_text(encoding="utf-8"))
            configured = payload.get("abacus", {}).get("executable")
        except (OSError, tomllib.TOMLDecodeError, AttributeError) as error:
            raise PreflightError(
                f"could not read ABACUS configuration {resolved_config}: {error}"
            ) from error
        if configured:
            return Path(configured)

    environment = os.environ if environ is None else environ
    configured = environment.get("BGP_ABACUS")
    return Path(configured) if configured else None


def verify_abacus_executable(path: Path) -> BuildManifest:
    """Verify a compiled executable and its package build provenance."""

    executable = Path(path).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PreflightError(f"ABACUS executable is not executable: {executable}")

    manifest_path = executable.with_name(executable.name + ".bgp-build.json")
    if not manifest_path.is_file():
        raise PreflightError(f"ABACUS build manifest is missing: {manifest_path}")
    try:
        manifest = BuildManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise PreflightError(
            f"ABACUS build manifest is invalid: {manifest_path}: {error}"
        ) from error

    digest = sha256(executable.read_bytes()).hexdigest()
    if digest != manifest.executable_sha256:
        raise PreflightError(
            "ABACUS executable SHA-256 does not match its build manifest"
        )
    if manifest.executable_name != executable.name:
        raise PreflightError(
            "ABACUS executable name does not match its build manifest"
        )

    missing = REQUIRED_ABACUS_FEATURES.difference(manifest.features)
    if missing:
        raise PreflightError(
            f"ABACUS build is missing required features: {sorted(missing)}"
        )
    return manifest
