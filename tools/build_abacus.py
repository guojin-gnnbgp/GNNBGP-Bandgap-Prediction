"""Explicitly build and install the package-owned combined ABACUS executable."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

FEATURES = [
    "pw",
    "cuda",
    "mpi",
    "libxc",
    "native-cider",
    "gpu-exx",
    "bgp-result-json",
]
SOURCE_REVISIONS = [
    "abacus-v3.11.0-beta.3@d33c34f3fa880349a6577ef1700c6ab1bfe8356a",
    "native-cider@7828d7cc28c8702ced99e88fd3cad51b58cf7fee",
    "integration@bbfa7d67c0476ae4a586719bd4a9713d81df8536",
]


def _cmake_options(
    architectures: Sequence[str],
    install_prefix: Path,
    cmake_prefix_paths: Sequence[Path] = (),
    cuda_compiler: Path | None = None,
) -> dict[str, bool | str]:
    options: dict[str, bool | str] = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_INSTALL_PREFIX": str(install_prefix.resolve()),
        "ENABLE_LCAO": False,
        "ENABLE_MPI": True,
        "USE_OPENMP": True,
        "USE_CUDA": True,
        "USE_CUDA_MPI": False,
        "USE_ROCM": False,
        "USE_ELPA": False,
        "ENABLE_LIBXC": True,
        "ENABLE_CIDER": True,
        "ENABLE_CIDER_EXTERNAL": False,
        "ENABLE_HDF5": False,
        # A git archive nested in this repository would otherwise inherit the
        # package repository's revision, which is not the ABACUS source revision.
        "COMMIT_INFO": False,
        "BUILD_TESTING": False,
        "CMAKE_CUDA_ARCHITECTURES": ";".join(architectures),
    }
    if cmake_prefix_paths:
        options["CMAKE_PREFIX_PATH"] = ";".join(
            str(path.expanduser().resolve()) for path in cmake_prefix_paths
        )
    if cuda_compiler is not None:
        options["CMAKE_CUDA_COMPILER"] = str(cuda_compiler.expanduser().resolve())
    return options


def cmake_configure_command(
    source: Path,
    build: Path,
    architectures: Sequence[str],
    *,
    install_prefix: Path | None = None,
    cmake_prefix_paths: Sequence[Path] = (),
    cuda_compiler: Path | None = None,
    generator: str = "Ninja",
) -> list[str]:
    """Return the locked PW CUDA+MPI native-CIDER configure command."""

    if not architectures or any(not value.strip() for value in architectures):
        raise ValueError("at least one non-empty CUDA architecture is required")
    prefix = install_prefix or build / "install"
    options = _cmake_options(
        [value.strip() for value in architectures],
        prefix,
        cmake_prefix_paths,
        cuda_compiler,
    )
    return [
        "cmake",
        "-S",
        str(source.expanduser().resolve()),
        "-B",
        str(build.expanduser().resolve()),
        "-G",
        generator,
        *(f"-D{name}={'ON' if value is True else 'OFF' if value is False else value}" for name, value in options.items()),
    ]


def select_generator(
    *, which: Callable[[str], str | None] = shutil.which
) -> str:
    """Prefer Ninja when installed and otherwise use portable Makefiles."""

    return "Ninja" if which("ninja") is not None else "Unix Makefiles"


def discover_cuda_compiler(
    explicit: Path | None,
    *,
    which: Callable[[str], str | None] = shutil.which,
    candidates: Sequence[Path] = (Path("/usr/local/cuda/bin/nvcc"),),
) -> Path:
    """Resolve nvcc from an explicit path, PATH, then standard local prefixes."""

    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"CUDA compiler not found: {resolved}")
        return resolved
    discovered = which("nvcc")
    if discovered is not None:
        return Path(discovered).resolve()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "CUDA compiler not found; pass --cuda-compiler /path/to/nvcc"
    )


def cmake_build_command(build: Path, *, jobs: int) -> list[str]:
    """Return the command that builds only the owned production target."""

    if jobs < 1:
        raise ValueError("jobs must be positive")
    return [
        "cmake",
        "--build",
        str(build.expanduser().resolve()),
        "--target",
        "abacus_pw_gpu",
        "--parallel",
        str(jobs),
    ]


def locate_built_executable(build: Path) -> Path:
    """Locate ABACUS across CMake generator output layouts."""

    build = build.expanduser().resolve()
    candidates = (build / "abacus_pw_gpu", build / "source" / "abacus_pw_gpu")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "built ABACUS executable not found; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def install_executable(
    built_executable: Path,
    install_prefix: Path,
    *,
    source_revisions: Sequence[str],
    cmake_options: dict[str, bool | str],
    toolchain: dict[str, str],
) -> Path:
    """Copy the compiled target and create its runtime-verifiable manifest."""

    built_executable = built_executable.expanduser().resolve()
    if not built_executable.is_file():
        raise FileNotFoundError(f"built ABACUS executable not found: {built_executable}")
    destination_dir = install_prefix.expanduser().resolve() / "bin"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "abacus_pw_gpu"
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(built_executable, temporary)
    temporary.chmod(temporary.stat().st_mode | 0o111)
    temporary.replace(destination)

    payload = {
        "schema_version": "1.0",
        "executable_name": destination.name,
        "executable_sha256": sha256(destination.read_bytes()).hexdigest(),
        "source_revisions": list(source_revisions),
        "cmake_options": cmake_options,
        "features": FEATURES,
        "toolchain": toolchain,
        "built_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(destination.with_name(destination.name + ".bgp-build.json"), payload)
    return destination


def _version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    return output[0].strip() if output else "unknown"


def _parse_architectures(value: str) -> list[str]:
    architectures = [part.strip() for part in value.split(",") if part.strip()]
    if not architectures or any(not part.isdigit() for part in architectures):
        raise argparse.ArgumentTypeError(
            "CUDA architectures must be a comma-separated list of integers"
        )
    return architectures


def build(args: argparse.Namespace) -> Path:
    source = args.source.expanduser().resolve()
    if not (source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"vendored ABACUS source not found: {source}")
    build_dir = args.build_dir.expanduser().resolve()
    install_prefix = args.install_prefix.expanduser().resolve()
    cuda_compiler = discover_cuda_compiler(args.cuda_compiler)
    options = _cmake_options(
        args.cuda_architectures,
        install_prefix,
        args.cmake_prefix_path,
        cuda_compiler,
    )
    configure = cmake_configure_command(
        source,
        build_dir,
        args.cuda_architectures,
        install_prefix=install_prefix,
        cmake_prefix_paths=args.cmake_prefix_path,
        cuda_compiler=cuda_compiler,
        generator=select_generator(),
    )
    subprocess.run(configure, check=True)
    subprocess.run(cmake_build_command(build_dir, jobs=args.jobs), check=True)

    executable = locate_built_executable(build_dir)
    cuda_command = [str(cuda_compiler), "--version"]
    toolchain = {
        "cmake": _version(["cmake", "--version"]),
        "cuda": _version(cuda_command),
        "mpi": _version(["mpicxx", "--version"]),
    }
    return install_executable(
        executable,
        install_prefix,
        source_revisions=SOURCE_REVISIONS,
        cmake_options=options,
        toolchain=toolchain,
    )


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(
        description="Build the package-owned PW CUDA+MPI CIDER/EXX ABACUS target"
    )
    result.add_argument("--source", type=Path, default=root / "vendor" / "abacus")
    result.add_argument("--build-dir", type=Path, required=True)
    result.add_argument("--install-prefix", type=Path, required=True)
    result.add_argument(
        "--cuda-architectures", type=_parse_architectures, required=True
    )
    result.add_argument(
        "--cmake-prefix-path", type=Path, action="append", default=[]
    )
    result.add_argument("--cuda-compiler", type=Path)
    result.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    return result


def main() -> int:
    args = parser().parse_args()
    installed = build(args)
    print(installed)
    print(installed.with_name(installed.name + ".bgp-build.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
