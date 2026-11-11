"""Command-line presentation layer for local band-gap prediction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import click

from . import __version__
from .assets import AssetRegistry
from .config import resolve_abacus_path, verify_abacus_executable
from .errors import BGPError, PreflightError
from .factory import create_service
from .schemas import BackendId, PredictionRequest, PredictionResult


def _path_option(**kwargs):
    return click.Path(path_type=Path, **kwargs)


def _emit_error(error: BGPError) -> None:
    click.echo(f"Error: {error}", err=True)
    if error.artifact_path is not None:
        click.echo(f"Artifacts: {error.artifact_path}", err=True)


def _emit_result(result: PredictionResult) -> None:
    click.echo(f"Backend: {result.backend.value}")
    click.echo(f"Method: {result.method}")
    if result.prediction.gap_ev is None:
        click.echo("Band gap: unavailable")
    else:
        click.echo(f"Band gap: {result.prediction.gap_ev:.7f} eV")
    click.echo(f"Classification: {result.prediction.classification.value}")
    if result.prediction.gap_type is not None:
        click.echo(f"Gap type: {result.prediction.gap_type.value}")
    for warning in result.warnings:
        click.echo(f"Warning: {warning}")
    click.echo(f"Artifacts: {result.artifact_path}")


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Predict crystal band gaps with method-specific provenance."""


@main.command()
@click.argument(
    "structure",
    type=_path_option(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--backend",
    type=click.Choice([item.value for item in BackendId]),
    required=True,
)
@click.option("--abacus", type=_path_option(file_okay=True, dir_okay=False))
@click.option("--config", type=_path_option(file_okay=True, dir_okay=False))
@click.option(
    "--work-root",
    type=_path_option(file_okay=False, dir_okay=True),
    default=Path("bgp-runs"),
    show_default=True,
)
def predict(
    structure: Path,
    backend: str,
    abacus: Path | None,
    config: Path | None,
    work_root: Path,
) -> None:
    """Run one synchronous local prediction for STRUCTURE."""

    try:
        result = create_service(config_path=config, progress=click.echo).predict(
            PredictionRequest(
                structure_path=structure,
                backend=BackendId(backend),
                abacus_path=abacus,
                work_root=work_root,
            )
        )
    except BGPError as error:
        _emit_error(error)
        raise click.exceptions.Exit(error.exit_code) from error
    except KeyboardInterrupt as error:
        click.echo("Interrupted; calculation artifacts were retained.", err=True)
        raise click.exceptions.Exit(130) from error
    _emit_result(result)


@main.command()
@click.option("--abacus", type=_path_option(file_okay=True, dir_okay=False))
@click.option("--config", type=_path_option(file_okay=True, dir_okay=False))
def backends(abacus: Path | None, config: Path | None) -> None:
    """Report backend availability without loading the legacy model."""

    gnn_modules = ("torch", "mace", "e3nn", "monty", "matscipy")
    missing = [name for name in gnn_modules if importlib.util.find_spec(name) is None]
    if missing:
        click.echo(f"{BackendId.GNN_HSE06.value}: unavailable ({', '.join(missing)})")
    else:
        click.echo(f"{BackendId.GNN_HSE06.value}: available")

    try:
        executable = resolve_abacus_path(abacus, config)
        if executable is None:
            raise PreflightError("ABACUS executable is not configured")
        verify_abacus_executable(executable)
    except BGPError as error:
        status = f"unavailable ({error})"
    else:
        status = "available"
    click.echo(f"{BackendId.ABACUS_HSE06.value}: {status}")
    click.echo(f"{BackendId.ABACUS_PBE0.value}: {status}")
    click.echo(f"{BackendId.CIDER_PBE0.value}: {status}")


@main.group(name="assets")
def assets_group() -> None:
    """Inspect immutable package-owned scientific assets."""


@assets_group.command(name="verify")
def verify_assets() -> None:
    """Hash and verify every bundled model and pseudopotential."""

    try:
        registry = AssetRegistry.default()
        for asset in registry.assets:
            registry.verify(asset.logical_id)
    except BGPError as error:
        _emit_error(error)
        raise click.exceptions.Exit(error.exit_code) from error
    click.echo(f"Verified {len(registry.assets)} bundled assets")


@main.command()
@click.option("--abacus", type=_path_option(file_okay=True, dir_okay=False))
@click.option("--config", type=_path_option(file_okay=True, dir_okay=False))
def doctor(abacus: Path | None, config: Path | None) -> None:
    """Verify a separately compiled combined ABACUS executable."""

    try:
        executable = resolve_abacus_path(abacus, config)
        if executable is None:
            raise PreflightError("ABACUS executable is not configured")
        manifest = verify_abacus_executable(executable)
    except BGPError as error:
        _emit_error(error)
        raise click.exceptions.Exit(error.exit_code) from error
    click.echo(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
