import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest

from bandgap_predictor.abacus.runner import RunProcessResult
from bandgap_predictor.assets import AssetRegistry
from bandgap_predictor.backends.abacus import (
    AbacusHSE06Backend,
    AbacusPBE0Backend,
    CiderPBE0Backend,
)
from bandgap_predictor.backends.base import PredictionContext
from bandgap_predictor.config import REQUIRED_ABACUS_FEATURES
from bandgap_predictor.runs import RunStore
from bandgap_predictor.schemas import BackendId, PredictionRequest
from bandgap_predictor.structures import load_structure


def _configured_executable(tmp_path: Path) -> Path:
    executable = tmp_path / "abacus_pw_gpu"
    executable.write_bytes(b"verified combined abacus")
    executable.chmod(0o755)
    manifest = {
        "schema_version": "1.0",
        "executable_name": executable.name,
        "executable_sha256": sha256(executable.read_bytes()).hexdigest(),
        "source_revisions": ["cider-base", "gpu-exx-head", "bgp-integration"],
        "cmake_options": {"ENABLE_CIDER": True, "USE_CUDA": True},
        "features": sorted(REQUIRED_ABACUS_FEATURES),
        "toolchain": {"cuda": "test"},
        "built_at": "2026-08-19T00:00:00+00:00",
    }
    executable.with_name(executable.name + ".bgp-build.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return executable


def _runner(calls: list[tuple[Path, Path]]) -> Callable[..., RunProcessResult]:
    def fake_runner(
        executable: Path,
        workdir: Path,
        progress: Callable[[str], None],
    ) -> RunProcessResult:
        calls.append((executable, workdir))
        progress("#SCF IS CONVERGED#")
        output = workdir / "OUT.BGP"
        output.mkdir()
        (output / "running_scf.log").write_text(
            "#SCF IS CONVERGED#\nE_gap(k) 1.250000 1.250000\n",
            encoding="utf-8",
        )
        (output / "bgp_electronic_state.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "converged": True,
                    "nspin": 1,
                    "occupation_weighting": "abacus_k_weighted",
                    "spin_capacity": 2.0,
                    "energy_unit": "eV",
                    "fermi_energy_ev": 0.0,
                    "printed_gap_ev": 1.25,
                    "k_weights": [1.0],
                    "eigenvalues_ev": [[-1.0, 1.0]],
                    "occupations": [[1.0, 0.0]],
                }
            ),
            encoding="utf-8",
        )
        return RunProcessResult(
            command=["mpirun", "-np", "1", str(executable)],
            returncode=0,
            started_at="2026-08-19T00:00:00+00:00",
            finished_at="2026-08-19T00:01:00+00:00",
            stdout_path=workdir / "stdout.log",
        )

    return fake_runner


@pytest.mark.parametrize(
    ("backend_cls", "backend_id", "method", "functional"),
    [
        (AbacusHSE06Backend, BackendId.ABACUS_HSE06, "HSE06", "HSE"),
        (AbacusPBE0Backend, BackendId.ABACUS_PBE0, "PBE0", "PBE0"),
        (CiderPBE0Backend, BackendId.CIDER_PBE0, "CIDER-PBE0", "PBE"),
    ],
)
def test_adapter_returns_locked_result_and_full_provenance(
    tmp_path: Path,
    backend_cls: (
        type[AbacusHSE06Backend]
        | type[AbacusPBE0Backend]
        | type[CiderPBE0Backend]
    ),
    backend_id: BackendId,
    method: str,
    functional: str,
) -> None:
    executable = _configured_executable(tmp_path)
    structure = load_structure(Path("examples/Si.cif"))
    request = PredictionRequest(
        structure_path=structure.source_path,
        backend=backend_id,
        abacus_path=executable,
        work_root=tmp_path / "runs",
    )
    run = RunStore(request.work_root).allocate(structure, backend_id)
    calls: list[tuple[Path, Path]] = []
    outcome = backend_cls(runner=_runner(calls)).predict(
        PredictionContext(
            request=request,
            structure=structure,
            run=run,
            assets=AssetRegistry.default(),
        )
    )

    assert calls == [(executable.resolve(), run.path)]
    assert outcome.prediction.gap_ev == 1.25
    assert outcome.converged is True
    assert outcome.settings["dft_functional"] == functional
    assert outcome.provenance["method"] == method
    assert outcome.provenance["executable_sha256"] == sha256(
        executable.read_bytes()
    ).hexdigest()
    assert set(outcome.provenance["pseudopotential_sha256"]) == {"Si"}
    assert outcome.provenance["source_revisions"][-1] == "bgp-integration"
    if backend_id is BackendId.CIDER_PBE0:
        assert outcome.provenance["cider_model_sha256"]
        assert outcome.settings["cider_feature_density"] == "valence-only"
    else:
        assert "cider_model_sha256" not in outcome.provenance


def test_adapter_uses_configured_executable_when_cli_path_is_absent(
    tmp_path: Path,
) -> None:
    executable = _configured_executable(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text(
        f'[abacus]\nexecutable = "{executable}"\n',
        encoding="utf-8",
    )
    structure = load_structure(Path("examples/Si.cif"))
    request = PredictionRequest(
        structure_path=structure.source_path,
        backend=BackendId.ABACUS_HSE06,
        work_root=tmp_path / "runs",
    )
    run = RunStore(request.work_root).allocate(structure, request.backend)
    calls: list[tuple[Path, Path]] = []

    AbacusHSE06Backend(runner=_runner(calls)).predict(
        PredictionContext(
            request=request,
            structure=structure,
            run=run,
            assets=AssetRegistry.default(),
            config_path=config,
        )
    )

    assert calls[0][0] == executable.resolve()
