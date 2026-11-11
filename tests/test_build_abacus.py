from __future__ import annotations

import json
from pathlib import Path

from bandgap_predictor.config import REQUIRED_ABACUS_FEATURES, verify_abacus_executable
from tools.build_abacus import (
    cmake_build_command,
    cmake_configure_command,
    discover_cuda_compiler,
    install_executable,
    locate_built_executable,
    select_generator,
)


def test_configure_command_locks_combined_pw_cuda_profile(tmp_path: Path) -> None:
    source = Path("vendor/abacus")
    build = tmp_path / "build"

    command = cmake_configure_command(
        source,
        build,
        ["86", "89", "120"],
        install_prefix=tmp_path / "install",
        cmake_prefix_paths=[Path("/opt/deps"), Path("/opt/more-deps")],
        cuda_compiler=Path("/opt/cuda/bin/nvcc"),
    )

    assert command[:6] == [
        "cmake",
        "-S",
        str(source.resolve()),
        "-B",
        str(build.resolve()),
        "-G",
    ]
    assert "Ninja" in command
    for option in (
        "-DENABLE_LCAO=OFF",
        "-DENABLE_MPI=ON",
        "-DUSE_OPENMP=ON",
        "-DUSE_CUDA=ON",
        "-DENABLE_LIBXC=ON",
        "-DENABLE_CIDER=ON",
        "-DENABLE_CIDER_EXTERNAL=OFF",
        "-DENABLE_HDF5=OFF",
        "-DCOMMIT_INFO=OFF",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=86;89;120",
        "-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc",
        "-DCMAKE_PREFIX_PATH=/opt/deps;/opt/more-deps",
    ):
        assert option in command


def test_build_command_targets_only_owned_pw_executable(tmp_path: Path) -> None:
    command = cmake_build_command(tmp_path / "build", jobs=7)

    assert command == [
        "cmake",
        "--build",
        str((tmp_path / "build").resolve()),
        "--target",
        "abacus_pw_gpu",
        "--parallel",
        "7",
    ]


def test_generator_falls_back_when_ninja_is_unavailable() -> None:
    assert select_generator(which=lambda _name: None) == "Unix Makefiles"


def test_generator_prefers_available_ninja() -> None:
    assert select_generator(which=lambda _name: "/usr/bin/ninja") == "Ninja"


def test_cuda_discovery_prefers_explicit_compiler(tmp_path: Path) -> None:
    explicit = tmp_path / "custom" / "nvcc"
    explicit.parent.mkdir()
    explicit.touch()

    assert discover_cuda_compiler(explicit, which=lambda _name: "/path/nvcc") == (
        explicit.resolve()
    )


def test_cuda_discovery_uses_existing_fallback(tmp_path: Path) -> None:
    fallback = tmp_path / "cuda" / "bin" / "nvcc"
    fallback.parent.mkdir(parents=True)
    fallback.touch()

    assert discover_cuda_compiler(
        None, which=lambda _name: None, candidates=[fallback]
    ) == fallback.resolve()


def test_built_executable_is_located_for_make_and_ninja_layouts(
    tmp_path: Path,
) -> None:
    root_binary = tmp_path / "make" / "abacus_pw_gpu"
    root_binary.parent.mkdir()
    root_binary.touch()
    nested_binary = tmp_path / "ninja" / "source" / "abacus_pw_gpu"
    nested_binary.parent.mkdir(parents=True)
    nested_binary.touch()

    assert locate_built_executable(root_binary.parent) == root_binary.resolve()
    assert locate_built_executable(tmp_path / "ninja") == nested_binary.resolve()


def test_install_writes_manifest_accepted_by_runtime_verifier(tmp_path: Path) -> None:
    built = tmp_path / "build" / "source" / "abacus_pw_gpu"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"combined abacus executable\n")
    built.chmod(0o755)

    installed = install_executable(
        built,
        tmp_path / "prefix",
        source_revisions=["native-cider@7828d7cc2", "integration@681fd5c0b"],
        cmake_options={"ENABLE_MPI": True, "USE_CUDA": True},
        toolchain={"cmake": "4.1.0", "cuda": "13.0", "mpi": "4.1.8"},
    )

    manifest_path = installed.with_name(installed.name + ".bgp-build.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert installed == tmp_path / "prefix" / "bin" / "abacus_pw_gpu"
    assert set(payload["features"]) == REQUIRED_ABACUS_FEATURES
    assert payload["source_revisions"][-1] == "integration@681fd5c0b"
    assert verify_abacus_executable(installed).executable_name == "abacus_pw_gpu"
