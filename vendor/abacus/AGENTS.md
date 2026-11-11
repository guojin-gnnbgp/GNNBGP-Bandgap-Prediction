# AGENTS.md

These instructions apply inside `abacus-cider/`.

## Project

- This directory is a vendored ABACUS tree inside the CiderPress repository.
- It is not a plain upstream checkout: this copy includes CIDER NLDF exchange-correlation integration.
- Keep ABACUS-style C++/CMake conventions when editing this tree, and avoid broad upstream refactors unless the task requires them.
- Online ABACUS documentation lives at https://abacus.deepmodeling.com/.

## Environment

- Native CIDER ABACUS builds and runs do not use a Python virtual environment,
  the CiderPress package, or `libciderbridge`.
- The parent CiderPress virtual environment is managed by `uv` and is useful
  only for external oracle regeneration or training tooling. Prefer `uv` for
  those Python tasks and run `uv pip` outside the sandbox so its cache is
  available.
- CIDER-enabled builds require LibXC, yaml-cpp 0.6 or newer, and the normal
  ABACUS FFTW3 or MKL backend.

## Build

For CIDER-specific ABACUS builds, follow `BUILD_CIDER.md`. Keep that file in sync
when changing CIDER build flags, dependency lookup behavior, executable names, or
runtime environment requirements.

ABACUS is built with CMake. Common options in this checkout include:

- `ENABLE_MPI=ON` by default
- `USE_OPENMP=ON` by default
- `ENABLE_LCAO=ON` by default
- `USE_ELPA=ON` by default when MPI is enabled
- `ENABLE_LIBXC=OFF` by default
- `ENABLE_CIDER=OFF` by default; requires `ENABLE_LIBXC=ON`
- `ENABLE_MLALGO=OFF` by default
- `ENABLE_LIBRI=OFF` by default
- `USE_CUDA=OFF`, `USE_ROCM=OFF`, `USE_DSP=OFF` by default
- `BUILD_TESTING=OFF` by default
- `ENABLE_ASAN=OFF`, `ENABLE_COVERAGE=OFF` by default

Basic non-CIDER CMake build:

```bash
cmake -B build -DENABLE_MPI=ON -DENABLE_LCAO=ON
cmake --build build -j$(nproc)
```

CIDER-enabled PW CPU build from `abacus-cider/`:

```bash
cmake -S . -B build-cider-pw-cpu \
  -DENABLE_MPI=ON \
  -DENABLE_LCAO=OFF \
  -DUSE_CUDA=OFF \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -Dyaml-cpp_DIR=/path/to/lib/cmake/yaml-cpp \
  -DCMAKE_CXX_STANDARD=14
cmake --build build-cider-pw-cpu --target abacus_pw_para -j$(nproc)
```

CIDER-enabled PW CUDA build:

```bash
cmake -S . -B build-cider-pw-gpu \
  -DENABLE_MPI=ON \
  -DENABLE_LCAO=OFF \
  -DUSE_CUDA=ON \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -Dyaml-cpp_DIR=/path/to/lib/cmake/yaml-cpp \
  -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_CUDA_ARCHITECTURES="<target-architectures>"
cmake --build build-cider-pw-gpu --target abacus_pw_gpu -j$(nproc)
```

For CUDA builds intended for heterogeneous GPU nodes such as `cata`, always set
`CMAKE_CUDA_ARCHITECTURES` to cover every GPU generation that Slurm may allocate
for the binary. For the current cata GPU mix, use
`-DCMAKE_CUDA_ARCHITECTURES="86;89;120"` so A6000, RTX 4090, and RTX 5090 jobs
each get a compatible/optimized code path. Avoid single-architecture builds
such as `89` for shared campaign binaries, because they can fail on A6000
(`sm_86`) allocations.

The CIDER evaluator remains host-side in CUDA builds. At runtime no CIDER
virtual environment, `PYTHONPATH`, CiderPress checkout, or CIDER-specific
`LD_LIBRARY_PATH` is required.

The toolchain flow is still available for dependency-heavy builds:

```bash
cd toolchain
./toolchain_gnu.sh
./build_abacus_gnu.sh
source install/setup
source abacus_env.sh
```

Other toolchain variants present here include `toolchain_intel.sh`,
`toolchain_gcc-aocl.sh`, and `toolchain_aocc-aocl.sh`.

## Executables

This CMake configuration names ABACUS binaries by enabled feature class, not by
the simple upstream aliases. Examples:

- `abacus_pw_para`: PW-only MPI CPU build
- `abacus_pw_ser`: PW-only serial CPU build
- `abacus_pw_gpu`: PW-only MPI CUDA build
- `abacus_basic_para`: LCAO/PW MPI CPU build without LibRI or MLALGO
- `abacus_basic_ser`: LCAO/PW serial CPU build without LibRI or MLALGO
- `abacus_basic_gpu`: LCAO/PW MPI CUDA build without LibRI or MLALGO
- `abacus_std_*`, `abacus_ml_*`, `abacus_max_*`: LibRI and/or MLALGO variants

Confirm the exact target with `cmake --build <build-dir> --target help`.

## CIDER Integration

CIDER support is wired through:

- CMake option: `ENABLE_CIDER`
- Native static targets: `cider_native` and `cider_pot_native`
- Potential implementation: `source/source_estate/module_pot/pot_cider_xc.*`
- Model/evaluator implementation: `source/source_hamilt/module_xc/cider/`
- PW Hamiltonian registration: `source/source_pw/module_pwdft/hamilt_pw.cpp`
- Input parameters: `cider_model`, `cider_xmix`, `cider_scf`, `cider_tf_tau`,
  and `cider_feature_density`

Current CIDER restrictions enforced by input conversion:

- `basis_type` must be `pw`
- `nspin` must be `1` or `2`
- `cider_xmix` must be in `[0, 1]`
- `cider_model` must be an accepted `.yaml` or `.yml` artifact
- `cider_feature_density` must be `auto`, `valence`, or
  `valence_pseudo_core`
- Do not combine `cider_model` with native hybrid/EXX `dft_functional` settings
- MGGA CIDER models request kinetic-energy density and may need NLCC-aware handling

CIDER feature-density policy:

- Use the default `cider_feature_density auto` so authoritative model metadata
  selects the tuple.
- Newly trained NLDF models should carry explicit `feature_density_policy`
  metadata. The narrow legacy `valence_pseudo_core` inference applies to any
  otherwise accepted NLDF artifact whose basename contains `CIDER23X`
  case-insensitively; it is not hash-gated to stock files.
- The old spelling `total` is accepted only as a deprecated compatibility alias for `valence_pseudo_core`.

CIDER examples are under:

- `examples/02_scf/pw_Si2-cider*`
- `examples/bandgap/pw_*`

The parent repository also documents a quick smoke test:

```bash
cd /tmp/copied-cider-example
OMP_NUM_THREADS=1 /path/to/abacus_pw_ser
```

Copy runtime inputs to a scratch directory before testing when the source
example must remain unchanged. Use the executable or wrapper that exists in
the build you are testing.

## Testing

Always run ABACUS internal tests with `OMP_NUM_THREADS=1` so MPI test cases do not oversubscribe the machine and results are comparable to the reference data.
Run ABACUS runtime tests, MPI tests, integration tests, and any test where process visibility matters outside the sandbox using an escalated command. This avoids sandbox interference with MPI launch, local sockets, process inspection, and runtime behavior.

Unit tests are CMake/CTest-based when configured with `BUILD_TESTING=ON`:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
cd build
export OMP_NUM_THREADS=1
ctest -N
ctest -R cell -V
```

Integration tests are organized in `tests/`:

- `01_PW`: PW basis
- `02_NAO_Gamma`: NAO gamma-only
- `03_NAO_multik`: NAO k-point
- `04_FF`: force fields
- `05_rtTDDFT`: real-time TDDFT
- `06_SDFT`: stochastic DFT
- `07_OFDFT`: orbital-free DFT
- `08_EXX`: hybrid and LR-TDDFT
- `09_DeePKS`: DeePKS
- `10_others`: miscellaneous
- `11_PW_GPU`, `12_NAO_Gamma_GPU`, `13_NAO_multik_GPU`, `15_rtTDDFT_GPU`, `16_SDFT_GPU`: GPU tests
- `PP_ORB`: pseudopotentials and numerical orbitals used by tests
- `integrate`: integration-test scripts
- `libxc`, `performance`: specialized examples/tests

Common integration commands:

```bash
cd tests/integrate
export OMP_NUM_THREADS=1
bash Autotest.sh
```

```bash
cd tests/01_PW/101_PW_MD_1O
export OMP_NUM_THREADS=1
bash ../../integrate/Single.sh
```

Check `tests/README` before changing test layout or naming.

## Source Layout

Important directories in `source/`:

- `source_base`: math, containers, device abstraction, FFT/grid/mixing utilities
- `source_basis`: AO, NAO, and PW basis modules
- `source_cell`: cell, symmetry, and neighbor handling
- `source_estate`: charge density, electronic state, and potentials
- `source_hamilt`: Hamiltonian, XC, VDW, and surface chemistry modules
- `source_hsolver`: eigensolvers including genelpa and PEXSI hooks
- `source_pw`: PW DFT, OFDFT, and stochastic DFT
- `source_lcao`: LCAO modules including gint, hcontainer, RI, LR/RT, DeePKS, DFT+U, and RDMFT
- `source_io`: input/output and parameter handling
- `source_md`, `source_relax`, `source_esolver`, `source_psi`, `source_main`

GPU/device kernels follow the local pattern:

- CPU kernels: `kernels/*.cpp`
- CUDA kernels: `kernels/cuda/*.cu`
- ROCm kernels: `kernels/rocm/*.hip.cu`

## Development Workflow

- Use existing module boundaries and CMake target patterns.
- Keep CIDER changes focused on the bridge, input validation, potential registration, and PW paths unless broader behavior is explicitly requested.
- Formatting configuration is present in `.clang-format`, `.clang-tidy`, and `.pre-commit-config.yaml`.
- CI workflow files live in `.github/workflows/`, including CMake build tests, integration tests, CUDA, coverage, pytest, Doxygen, and performance workflows.
- The Python package under `python/pyabacus/` uses `scikit-build-core`, `pybind11`, and pytest settings in its own `pyproject.toml`.
