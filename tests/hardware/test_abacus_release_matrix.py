from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from bandgap_predictor.config import verify_abacus_executable
from bandgap_predictor.factory import create_service
from bandgap_predictor.schemas import BackendId, PredictionRequest, RunStatus

REQUIRED = ("InSb", "Si")
EXTENDED = ("GaAs", "MgO")
ABACUS_BACKENDS = (
    BackendId.ABACUS_HSE06,
    BackendId.ABACUS_PBE0,
    BackendId.CIDER_PBE0,
)
BASELINE_PATH = Path(__file__).with_name("baselines.json")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _baselines() -> dict[str, object]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _case(baselines: dict[str, object], backend: BackendId, system: str) -> dict:
    return baselines["cases"][backend.value][system]


def test_required_baseline_contract_is_complete_and_provenanced() -> None:
    baselines = _baselines()

    assert baselines["schema_version"] == "1.0"
    for backend in ABACUS_BACKENDS:
        for system in REQUIRED:
            case = _case(baselines, backend, system)
            assert case["status"] in {"success", "not_baselined"}
            assert len(case["structure_sha256"]) == 64
            if case["status"] == "success":
                assert case["classification"]
                assert case["gap_ev"] >= 0.0
                assert case["abs_tolerance_ev"] > 0.0
                assert len(case["executable_sha256"]) == 64
                assert len(case["settings_sha256"]) == 64
                assert case["pseudopotential_sha256"]
                if backend is BackendId.CIDER_PBE0:
                    assert len(case["cider_model_sha256"]) == 64
            else:
                assert case["reason"]


@pytest.fixture(scope="session")
def executable() -> Path:
    configured = os.environ.get("BGP_ABACUS")
    if not configured:
        pytest.skip("set BGP_ABACUS to the package-built abacus_pw_gpu executable")
    path = Path(configured).expanduser().resolve()
    verify_abacus_executable(path)
    return path


def _assert_release_case(system: str, backend: BackendId, executable: Path) -> None:
    baselines = _baselines()
    expected = _case(baselines, backend, system)
    if expected["status"] != "success":
        pytest.xfail(expected["reason"])

    work_root = Path(os.environ.get("BGP_HARDWARE_RUN_ROOT", "hardware-runs"))
    result = create_service().predict(
        PredictionRequest(
            structure_path=Path("examples") / f"{system}.cif",
            backend=backend,
            abacus_path=executable,
            work_root=work_root,
        )
    )

    assert result.status is RunStatus.SUCCESS
    assert result.converged is True
    assert result.prediction.classification.value == expected["classification"]
    assert result.prediction.gap_ev == pytest.approx(
        expected["gap_ev"], abs=expected["abs_tolerance_ev"]
    )
    assert result.provenance["executable_sha256"] == expected["executable_sha256"]
    assert _sha256_json(result.settings) == expected["settings_sha256"]
    assert result.structure["structure_sha256"] == expected["structure_sha256"]
    assert (
        result.provenance["pseudopotential_sha256"]
        == expected["pseudopotential_sha256"]
    )
    if backend is BackendId.CIDER_PBE0:
        assert result.provenance["cider_model_sha256"] == expected["cider_model_sha256"]


@pytest.mark.hardware
@pytest.mark.parametrize("system", REQUIRED)
@pytest.mark.parametrize("backend", ABACUS_BACKENDS)
def test_required_release_matrix(
    system: str, backend: BackendId, executable: Path
) -> None:
    _assert_release_case(system, backend, executable)


@pytest.mark.hardware
@pytest.mark.extended
@pytest.mark.parametrize("system", EXTENDED)
@pytest.mark.parametrize("backend", ABACUS_BACKENDS)
def test_extended_release_matrix(
    system: str, backend: BackendId, executable: Path
) -> None:
    _assert_release_case(system, backend, executable)
