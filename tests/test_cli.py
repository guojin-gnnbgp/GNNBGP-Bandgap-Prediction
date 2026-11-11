import json
from hashlib import sha256
from pathlib import Path

from click.testing import CliRunner

from bandgap_predictor.config import REQUIRED_ABACUS_FEATURES
from bandgap_predictor.errors import PreflightError
from bandgap_predictor.schemas import (
    BackendId,
    ElectronicClassification,
    PredictionResult,
    PredictionValue,
    RunStatus,
)


def _result(tmp_path: Path) -> PredictionResult:
    artifact = tmp_path / "run"
    artifact.mkdir()
    return PredictionResult(
        run_id="run-1",
        status=RunStatus.SUCCESS,
        backend=BackendId.GNN_HSE06,
        method="HSE06-like legacy ScalarPropertyMACE",
        prediction=PredictionValue(
            gap_ev=1.3439248,
            classification=ElectronicClassification.INSULATOR,
        ),
        converged=None,
        structure={"formula": "Si8"},
        provenance={"model_sha256": "abc"},
        started_at="2026-08-19T00:00:00+00:00",
        finished_at="2026-08-19T00:00:01+00:00",
        elapsed_seconds=1.0,
        artifact_path=artifact,
        warnings=["legacy/provisional"],
    )


class _SuccessfulService:
    def __init__(self, result: PredictionResult) -> None:
        self.result = result

    def predict(self, request):
        assert request.structure_path.name == "Si.cif"
        assert request.backend is BackendId.GNN_HSE06
        return self.result


class _FailingService:
    def __init__(self, artifact: Path) -> None:
        self.artifact = artifact

    def predict(self, request):
        raise PreflightError("ABACUS executable is not configured", self.artifact)


def _executable(tmp_path: Path) -> Path:
    executable = tmp_path / "abacus_pw_gpu"
    executable.write_bytes(b"combined abacus")
    executable.chmod(0o755)
    manifest = {
        "schema_version": "1.0",
        "executable_name": executable.name,
        "executable_sha256": sha256(executable.read_bytes()).hexdigest(),
        "source_revisions": ["integration"],
        "cmake_options": {"USE_CUDA": True},
        "features": sorted(REQUIRED_ABACUS_FEATURES),
        "toolchain": {"cuda": "test"},
    }
    executable.with_name(executable.name + ".bgp-build.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return executable


def test_help_lists_gui_independent_commands() -> None:
    from bandgap_predictor.cli import main

    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    for command in ["predict", "backends", "assets", "doctor"]:
        assert command in result.output


def test_predict_prints_gap_warning_and_artifact_path(
    monkeypatch, tmp_path: Path
) -> None:
    from bandgap_predictor import cli

    expected = _result(tmp_path)
    monkeypatch.setattr(
        cli,
        "create_service",
        lambda **_kwargs: _SuccessfulService(expected),
    )

    result = CliRunner().invoke(
        cli.main,
        ["predict", "examples/Si.cif", "--backend", "gnn-hse06"],
    )

    assert result.exit_code == 0
    assert "Band gap: 1.3439248 eV" in result.output
    assert "legacy/provisional" in result.output
    assert f"Artifacts: {expected.artifact_path}" in result.output


def test_locked_cli_rejects_abacus_overrides() -> None:
    from bandgap_predictor.cli import main

    result = CliRunner().invoke(
        main,
        [
            "predict",
            "examples/Si.cif",
            "--backend",
            "abacus-hse06",
            "--set",
            "ecutwfc=50",
        ],
    )

    assert result.exit_code == 2
    assert "No such option" in result.output
    assert "--set" in result.output


def test_preflight_error_uses_exit_three_and_retained_path(
    monkeypatch, tmp_path: Path
) -> None:
    from bandgap_predictor import cli

    artifact = tmp_path / "retained-run"
    monkeypatch.setattr(
        cli,
        "create_service",
        lambda **_kwargs: _FailingService(artifact),
    )

    result = CliRunner().invoke(
        cli.main,
        ["predict", "examples/Si.cif", "--backend", "abacus-hse06"],
    )

    assert result.exit_code == 3
    assert "ABACUS executable is not configured" in result.output
    assert str(artifact) in result.output


def test_assets_verify_and_doctor_do_not_load_gnn_model(tmp_path: Path) -> None:
    from bandgap_predictor.cli import main

    runner = CliRunner()
    assets = runner.invoke(main, ["assets", "verify"])
    executable = _executable(tmp_path)
    doctor = runner.invoke(main, ["doctor", "--abacus", str(executable)])

    assert assets.exit_code == 0
    assert "Verified 74 bundled assets" in assets.output
    assert doctor.exit_code == 0
    assert "bgp-result-json" in doctor.output
    assert sha256(executable.read_bytes()).hexdigest() in doctor.output


def test_backends_lists_all_four_stable_backend_ids(tmp_path: Path) -> None:
    from bandgap_predictor.cli import main

    result = CliRunner().invoke(
        main,
        ["backends", "--abacus", str(_executable(tmp_path))],
    )

    assert result.exit_code == 0
    for backend in BackendId:
        assert f"{backend.value}:" in result.output
