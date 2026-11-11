import json
from hashlib import sha256
from pathlib import Path

import pytest

from bandgap_predictor.config import verify_abacus_executable
from bandgap_predictor.errors import PreflightError

REQUIRED_FEATURES = [
    "pw",
    "cuda",
    "mpi",
    "libxc",
    "native-cider",
    "gpu-exx",
    "bgp-result-json",
]


def write_executable_and_manifest(tmp_path: Path) -> Path:
    executable = tmp_path / "abacus_pw_gpu"
    executable.write_bytes(b"verified executable\n")
    executable.chmod(0o755)
    manifest = {
        "schema_version": "1.0",
        "executable_sha256": sha256(executable.read_bytes()).hexdigest(),
        "executable_name": executable.name,
        "source_revisions": ["cider-base", "gpu-exx-head", "bgp-integration"],
        "cmake_options": {"ENABLE_MPI": True, "USE_CUDA": True},
        "features": REQUIRED_FEATURES,
        "toolchain": {"cmake": "3.22.1"},
        "built_at": "2026-08-19T00:00:00+00:00",
    }
    executable.with_name(executable.name + ".bgp-build.json").write_text(
        json.dumps(manifest)
    )
    return executable


def test_verified_executable_returns_typed_manifest(tmp_path: Path) -> None:
    executable = write_executable_and_manifest(tmp_path)

    manifest = verify_abacus_executable(executable)

    assert manifest.executable_name == "abacus_pw_gpu"
    assert set(manifest.features) == set(REQUIRED_FEATURES)


def test_manifest_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    executable = write_executable_and_manifest(tmp_path)
    executable.write_text("changed")

    with pytest.raises(PreflightError, match="SHA-256"):
        verify_abacus_executable(executable)


def test_missing_required_feature_fails_closed(tmp_path: Path) -> None:
    executable = write_executable_and_manifest(tmp_path)
    manifest_path = executable.with_name(executable.name + ".bgp-build.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["features"].remove("native-cider")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PreflightError, match="native-cider"):
        verify_abacus_executable(executable)
