# Installation and local ABACUS build

## Python environment

Python 3.11 or newer and `uv` are recommended:

```bash
uv sync --group dev
```

Add the legacy GNN dependencies only when `gnn-hse06` is needed:

```bash
uv sync --extra gnn --group dev
```

The GNN environment pins the proven-compatible MACE and CPU PyTorch versions.
The model is a trusted pickle and is loaded only after its bundled SHA-256 has
been verified.

## Build the owned ABACUS executable

The source checkout owns the combined source at `vendor/abacus`. Building is a
separate installation step and is never triggered by Python packaging, import,
or CLI startup.

The target machine needs a C++ toolchain, CMake, MPI, CUDA, OpenMP, FFTW,
BLAS/LAPACK/ScaLAPACK, LibXC, and the native-CIDER YAML dependency. Dependency
prefixes can be repeated when they are not on standard search paths:

```bash
uv run python tools/build_abacus.py \
  --build-dir build/abacus \
  --install-prefix build/abacus-install \
  --cuda-architectures 86,89,120 \
  --cmake-prefix-path /path/to/dependencies
```

The helper builds only the PW CUDA+MPI target `abacus_pw_gpu`. It writes the
executable and `abacus_pw_gpu.bgp-build.json` under the installation prefix.
The manifest contains the executable hash, source lineage, required features,
locked CMake options, and toolchain versions. It is mandatory at runtime.

Verify the pair:

```bash
uv run bgp doctor --abacus build/abacus-install/bin/abacus_pw_gpu
```

## Configure the executable path

Resolution order is:

1. `--abacus /absolute/path/to/abacus_pw_gpu` on the command line;
2. a TOML configuration file supplied with `--config`, or the platform user
   configuration file for `bandgap-predictor`;
3. the `BGP_ABACUS` environment variable.

Configuration files use:

```toml
[abacus]
executable = "/absolute/path/to/abacus_pw_gpu"
```

The CLI rejects an executable whose name, SHA-256, or required feature set does
not match its companion manifest.

## Local runtime

ABACUS runs use the local MPI launcher with one rank. The runner enforces
`OMP_NUM_THREADS=1`, streams progress, captures combined output, and terminates
the entire process group on interruption. No scheduler or remote execution is
implemented in version 0.1.

```bash
uv run bgp predict structure.cif \
  --backend abacus-pbe0 \
  --abacus /absolute/path/to/abacus_pw_gpu \
  --work-root bgp-runs
```

The supplied structure must be three-dimensionally periodic and nonmagnetic.
It is evaluated exactly as provided.
