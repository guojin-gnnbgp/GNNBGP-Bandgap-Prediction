# Native CIDER architecture and migration

ABACUS owns the complete native inference path for the supported CIDER YAML
profile. A native-only binary neither imports nor links CiderPress. A dual
binary can additionally compile the existing CiderPress bridge and choose the
native or external runtime without changing executable or repository branch.
See the [legacy YAML compatibility profile](cider_legacy_yaml_compatibility.md)
for the normative artifact boundary, [the build guide](cider_build.md) for
build and runtime instructions, and the
[validation record](cider_native_validation.md) for measured acceptance
evidence.

## Ownership and data flow

The common path is split into five boundaries:

1. Input conversion resolves `cider_backend=auto|external|native` against the
   compiled capabilities. `auto` selects external in a dual build and native in
   a native-only build. Explicit requests never fall back silently.
2. For native selection, `cider_model_input.cpp` checks the `.yaml`/`.yml`
   suffix and asks `ModelLoader` for authoritative metadata. In MPI builds rank
   zero performs the inspection and broadcasts either metadata or the complete
   failure. External selection retains the legacy bridge-compatible early
   MGGA setup before the external runtime loads authoritative bridge metadata.
3. `source/source_hamilt/module_xc/cider/` safely parses the whitelisted legacy
   YAML into an immutable typed `Model`. It owns feature normalization,
   U/V/Z transforms, one- through three-dimensional spline evaluation,
   semilocal descriptors, the migrated version-J NLDF kernels, reverse
   derivatives, and the role-separated dual-density evaluator. yaml-cpp is
   private to the loader; the model/evaluator interfaces expose no Python or
   third-party parser types.
4. `PotCiderXC` asks `pot_cider_xc_runtime_factory` for a common
   `PotCiderRuntime`. The native adapter owns one native model/evaluator per
   rank; the external adapter owns one bridge context per rank. `PotCiderXC`
   builds a
   baseline density tuple and a model-selected feature density tuple, gathers
   each tuple to the complete real-space grid, evaluates it on the host, then
   scatters derivatives to the local slabs. ABACUS's existing gradient helpers
   reconstruct the local potential and the MGGA path supplies `vtau`.
5. ABACUS owns the final energy components, potential, diagnostics, post-SCF
   label, and training-dump schema for either runtime. `cider_scf` selects SCF
   use versus a post-SCF label; `cider_backend` selects evaluation ownership.

The baseline tuple follows the normal NLCC convention: valence plus pseudo-core
for rho and sigma. The feature tuple is either valence or
valence-plus-pseudo-core according to model metadata. Pseudo-core tau is added
only when `cider_tf_tau` is enabled. `cider_feature_density auto` is the normal
runtime setting. Input conversion validates the request against root-inspected
metadata before potential allocation; `PotCiderXC` then loads the same model
and independently resolves `auto` for evaluation.

## Energy conventions

The evaluator deliberately returns separate baseline and learned-feature
packets:

- LibXC baseline `exc` is energy per particle and is integrated with the
  baseline density.
- Mapped CIDER feature `exc` is energy density and is integrated over volume.
- Derivatives and diagnostics remain separated until ABACUS constructs the
  total XC potential and `EnergyComponents`.

This is the accepted role-separated convention and is part of the native
runtime contract. The historical ABACUS bridge returned
`baseline_per_particle + feature_energy_density` in one array and the adapter
then rho-weighted the whole array. Historical final CIDER energies therefore
are not parity targets for the native convention. Fixed-density replay
reproduced that old first-call behavior and explains the divergence, but later
SCF states have different densities and cannot be corrected by one scalar.
The [validation record](cider_native_validation.md#historical-bridge-controls)
contains the complete evidence and explicitly does not claim historical
final-energy parity.

## Build and runtime boundary

`ENABLE_CIDER=ON` requires `ENABLE_LIBXC=ON`, yaml-cpp 0.6 or newer, and the
ABACUS-selected FFTW3 or MKL target. `cider_native` and `cider_pot_native` are
static ABACUS-owned targets linked into the selected executable. No CIDER
shared runtime is installed for native execution.

With `ENABLE_CIDER_EXTERNAL=OFF`, the CIDER path performs no Python or NumPy
discovery and has no CiderPress or bridge link. Running accepted native YAML
artifacts requires no virtual environment or CIDER-specific library path. The
portable release guard in
`cmake/tests/CheckNativeCiderReleaseArtifact.cmake` checks this boundary in a
release cache, final link command, and executable dependency closure.

With `ENABLE_CIDER_EXTERNAL=ON`, CMake discovers Python development/NumPy and
the CiderPress bridge. The resulting dual artifact intentionally links Python
and `libciderbridge`; `CheckDualCiderReleaseArtifact.cmake` checks that positive
dependency boundary. Both adapters implement the same role-separated runtime
interface, and the factory chooses exactly one adapter during construction.

CUDA-enabled PW builds still execute native CIDER evaluation on the host. CUDA
accelerates the surrounding ABACUS path; this release does not contain a GPU
CIDER evaluator. Each rank also retains a full model, complete gathered grid,
output packets, and NLDF FFT workspace. CIDER evaluation and its memory are
therefore replicated per rank. Distributed NLDF FFTs, local-grid evaluation,
and GPU kernels are future optimizations, not current behavior.

## Supported boundary and exclusions

The release boundary is the five-model compatibility corpus: four stock
CIDER23X SL/NL GGA/MGGA models plus one accepted NumPy-2-era trained NLDF MGGA
model. It covers `numpy.core` and `numpy._core`, one and two spins, both density
policies, and one- through three-dimensional spline terms. The normative
profile records immutable model hashes and the exact tag, dtype, metadata, and
cross-field rules.

The native runtime does not support joblib/pickle, `MappedXC2`, multiple
kernels or evaluators, RBF/linear/neural evaluators, NLOF, SDMX, hybrid
descriptor settings, non-version-J NLDF, unknown transforms, object dtype,
Fortran-order arrays, or four-dimensional splines. Only PW calculations with
`nspin` 1 or 2 are accepted, and CIDER cannot be combined with native
hybrid/EXX functional selection. These are native whitelist limits, not a claim
about the external bridge: external selection retains whatever models the
installed CiderPress bridge supports.

## Migration from the bridge path

For an existing ABACUS+CiderPress workflow:

1. Choose a native-only distribution build or a dual development build using
   `BUILD_CIDER.md`.
2. Point `cider_model` directly at an accepted `.yaml` or `.yml` artifact.
   Joblib/pickle is not an alternative input path.
3. Prefer `cider_feature_density auto`. A newly trained NLDF model must carry
   explicit density-policy metadata. The narrowly defined legacy inference
   applies to any otherwise accepted NLDF artifact whose basename contains
   `CIDER23X` case-insensitively; it is not hash-gated to the stock corpus.
4. Set `cider_backend native` for dependency-free execution. Keep the Python
   environment and select `external` when checking CiderPress compatibility.
5. Compare both backends with the same dual executable, model, and input. Do
   not use historical combined-convention final energies as a native
   acceptance gate.

The dual validation build avoided an external HDF5/h5py ABI collision by
configuring ABACUS with `ENABLE_HDF5=OFF`. It then ran native and external
role-separated evaluation through the same serial, MPI, and CUDA executables.
The validation record contains the measured numerical results. Historical
combined-convention logs remain an incompatible substitute.
