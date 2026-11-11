# Building ABACUS with CIDER support

ABACUS always builds the native C++/C evaluator when `ENABLE_CIDER=ON`.
`ENABLE_CIDER_EXTERNAL=ON` optionally adds the existing CiderPress Python
bridge to the same executable. A dual build can select native or external
evaluation at runtime, so development and parity testing do not require two
ABACUS branches or executables.

Related documentation:

- [architecture and migration](docs/developers_guide/cider_native_architecture.md);
- [native legacy-YAML compatibility](docs/developers_guide/cider_legacy_yaml_compatibility.md);
- [model authoring and troubleshooting](docs/developers_guide/cider_model_authoring_and_troubleshooting.md); and
- [validation evidence](docs/developers_guide/cider_native_validation.md).

## Build profiles

Both profiles require LibXC, yaml-cpp 0.6 or newer, and the FFT backend already
selected by ABACUS. The native-only profile has no Python, NumPy, CiderPress,
or `libciderbridge` build/runtime dependency.

The dual profile additionally requires:

- a Python development installation with NumPy headers;
- a CiderPress checkout available through `CIDERPRESS_DIR`; and
- `libciderbridge` available through `CIDERPRESS_LIB_DIR`.

Automatic yaml-cpp fetching is disabled by default. Set `yaml-cpp_DIR` or
`CMAKE_PREFIX_PATH`, or explicitly use `CIDER_FETCH_YAML_CPP=ON` when network
fetching is intended.

### Native-only serial CPU

```bash
cmake -S . -B build-cider-native \
  -DENABLE_MPI=OFF \
  -DENABLE_LCAO=OFF \
  -DUSE_CUDA=OFF \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=OFF \
  -Dyaml-cpp_DIR=/path/to/lib/cmake/yaml-cpp \
  -DBUILD_TESTING=OFF

cmake --build build-cider-native --target abacus_pw_ser -j"$(nproc)"
```

### Dual serial CPU

```bash
cmake -S . -B build-cider-dual \
  -DENABLE_MPI=OFF \
  -DENABLE_LCAO=OFF \
  -DUSE_CUDA=OFF \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=ON \
  -DENABLE_HDF5=OFF \
  -DCIDERPRESS_DIR=/path/to/CiderPress \
  -DCIDERPRESS_LIB_DIR=/path/to/CiderPress/ciderpress/lib \
  -Dyaml-cpp_DIR=/path/to/lib/cmake/yaml-cpp \
  -DBUILD_TESTING=OFF

cmake --build build-cider-dual --target abacus_pw_ser -j"$(nproc)"
```

`ENABLE_HDF5=OFF` is recommended when the embedded environment's `h5py` uses a
different HDF5 ABI from ABACUS. Loading both HDF5 versions into one process can
break Python extension initialization. It is not required when the ABACUS and
Python HDF5 stacks are ABI-compatible.

### MPI CPU

Use a separate build directory, change `ENABLE_MPI` to `ON`, and build
`abacus_pw_para`. Add the three external options from the dual command when a
dual MPI executable is required.

```bash
cmake -S . -B build-cider-dual-mpi \
  -DENABLE_MPI=ON -DENABLE_LCAO=OFF -DUSE_CUDA=OFF \
  -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=ON -DENABLE_HDF5=OFF \
  -DCIDERPRESS_DIR=/path/to/CiderPress \
  -DCIDERPRESS_LIB_DIR=/path/to/CiderPress/ciderpress/lib \
  -Dyaml-cpp_DIR=/path/to/lib/cmake/yaml-cpp \
  -DBUILD_TESTING=OFF
cmake --build build-cider-dual-mpi --target abacus_pw_para -j"$(nproc)"
```

Run with one OpenMP thread per rank unless separately benchmarked:

```bash
OMP_NUM_THREADS=1 mpirun -np 2 build-cider-dual-mpi/abacus_pw_para
```

### CUDA PW

Both CIDER backends execute on the host while CUDA accelerates the surrounding
ABACUS PW calculation. Configure every architecture used by the deployment
nodes; for the current cata mix use `86;89;120`.

```bash
cmake -S . -B build-cider-dual-cuda \
  -DENABLE_MPI=ON -DENABLE_LCAO=OFF -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
  -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=ON -DENABLE_HDF5=OFF \
  -DCIDERPRESS_DIR=/path/to/CiderPress \
  -DCIDERPRESS_LIB_DIR=/path/to/CiderPress/ciderpress/lib \
  -Dyaml-cpp_DIR=/path/to/lib/cmake/yaml-cpp \
  -DBUILD_TESTING=OFF
cmake --build build-cider-dual-cuda --target abacus_pw_gpu -j"$(nproc)"
```

## Runtime backend selection

Set `cider_model` to the YAML artifact and choose `cider_backend`:

| Value | Behavior |
| --- | --- |
| `auto` | Selects external when compiled, otherwise native. This is the default. |
| `external` | Requires `ENABLE_CIDER_EXTERNAL=ON`; uses CiderPress and the bridge. |
| `native` | Uses the ABACUS-owned safe YAML loader and evaluator. |

Example:

```text
cider_model    /path/to/model.yaml
cider_backend  native
```

Selection is strict: ABACUS never silently falls back from an explicitly
requested backend. The runtime log records `runtime ready backend=native` or
`runtime ready backend=external`.

External remains the default in a dual build for compatibility during
development. It retains the model coverage of the installed CiderPress bridge,
including models outside the native whitelist. Explicit native selection
reports the structured native schema error for unsupported YAML.

A native-only executable needs no activated virtual environment. External
execution must make the CiderPress Python modules and bridge shared libraries
available, for example:

```bash
export PYTHONPATH=/path/to/CiderPress/.venv/lib/python3.X/site-packages:/path/to/CiderPress
export LD_LIBRARY_PATH=/path/to/CiderPress/ciderpress/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
```

## Install and inspect

```bash
cmake --install build-cider-dual --prefix /path/to/abacus-install
/path/to/abacus-install/bin/abacus -i
```

Build information reports native and external capabilities separately and the
conditional `auto` default. For native-only release artifacts, `ldd` and
`readelf -d` must contain neither Python nor `libciderbridge`. Dual artifacts
must contain both.

```bash
cmake \
  -DCIDER_CACHE="$PWD/build-cider-native/CMakeCache.txt" \
  -DCIDER_LINK="$PWD/build-cider-native/CMakeFiles/abacus_pw_ser.dir/link.txt" \
  -DCIDER_BINARY="$PWD/build-cider-native/abacus_pw_ser" \
  -P cmake/tests/CheckNativeCiderReleaseArtifact.cmake

cmake \
  -DCIDER_CACHE="$PWD/build-cider-dual/CMakeCache.txt" \
  -DCIDER_LINK="$PWD/build-cider-dual/CMakeFiles/abacus_pw_ser.dir/link.txt" \
  -DCIDER_BINARY="$PWD/build-cider-dual/abacus_pw_ser" \
  -P cmake/tests/CheckDualCiderReleaseArtifact.cmake
```

## Development parity workflow

Copy one calculation to two scratch directories and change only
`cider_backend`. Run the same dual executable in each directory and compare the
backend log, energy components, and potential diagnostics. The five-model unit
parity suite also includes the project-trained NumPy-2 YAML artifact, not only
the four stock paper models.

## Common failures

- `ENABLE_CIDER requires ENABLE_LIBXC`: add `-DENABLE_LIBXC=ON`.
- `ENABLE_CIDER_EXTERNAL requires ENABLE_CIDER`: enable native CIDER as the
  common runtime foundation.
- Python, NumPy, CiderPress, or `libciderbridge` not found: supply the dual
  profile paths or build native-only.
- `external backend was requested but is not compiled`: reconfigure with
  `ENABLE_CIDER_EXTERNAL=ON` or select `native`.
- `native backend was requested but is not compiled`: use a CIDER-enabled
  executable.
- `ValueError: Not a datatype` during external startup: an HDF5 ABI collision
  is likely; use compatible HDF5/h5py builds or configure ABACUS with
  `ENABLE_HDF5=OFF`.
- `yaml-cpp is required`: install yaml-cpp and set `yaml-cpp_DIR` or
  `CMAKE_PREFIX_PATH`.
- `DENSITY_POLICY_REQUIRED`: add explicit `feature_density_policy` metadata to
  a newly trained NLDF model.
- `JOBLIB_UNSUPPORTED`, `ARTIFACT_UNSUPPORTED`, or `UNSUPPORTED_*` under native
  selection means the artifact is outside the native boundary. Use external
  only when that model is supported by the installed CiderPress bridge.

Each MPI rank currently owns a complete model, gathered grid, output packets,
and NLDF workspace. Increasing the rank count does not distribute CIDER
evaluation or reduce its per-rank memory footprint.
