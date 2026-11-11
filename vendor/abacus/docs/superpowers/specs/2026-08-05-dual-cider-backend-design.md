# Dual native/external CIDER backend design

## Status and scope

This specification extends the ABACUS-native CIDER integration so one ABACUS
source revision can build an executable containing both the native evaluator
and the existing CiderPress bridge. A dual-backend executable selects its
backend at runtime. The external backend is the default when it is compiled;
an explicitly native-only build remains Python-free.

All work described here is restricted to the isolated worktree at
`/home/bonan/appdir/CiderPress/.planning/worktrees/abacus-cider-native-evaluation`.
The existing `/home/bonan/appdir/CiderPress/abacus-cider` worktree must not be
modified.

The target is one ABACUS source revision containing both implementations, not
two maintained repositories or long-lived source variants. The development
branch may retain normal implementation commits; it can be squash-merged later
if a single history commit is desired.

## Goals

- Preserve the current external CiderPress-backed execution path.
- Preserve the standalone native evaluator and its Python-free build.
- Allow an external-enabled executable to select `external` or `native` from
  `INPUT` without rebuilding.
- Default to external execution for external-enabled builds during the current
  development period.
- Keep model loading, density-role conventions, energy components, training
  dumps, MPI behavior, and ABACUS potential reconstruction testable through a
  shared contract.
- Make same-binary native/external parity runs straightforward.

## Non-goals

- Switching backend during an SCF calculation.
- Evaluating both backends on every SCF call in production.
- Making the external backend optional at dynamic-link time in a dual build.
- Extending the native YAML compatibility profile as part of this change.
- Reproducing the obsolete April combined-energy bridge convention.
- Removing the need for Python and CiderPress from an external-enabled build.

## Build configuration

`ENABLE_CIDER=ON` enables the common CIDER integration and always compiles the
native backend. A new `ENABLE_CIDER_EXTERNAL` CMake option, defaulting to
`OFF`, adds the external bridge to the same executable:

```bash
# Standalone native-only build
cmake .. -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=OFF

# Dual-backend build
cmake .. -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=ON \
  -DCIDERPRESS_DIR=/path/to/CiderPress
```

`ENABLE_CIDER_EXTERNAL=ON` requires `ENABLE_CIDER=ON`. Only that option enables
Python/NumPy discovery, locates `cider_bridge.h` and `libciderbridge`, and links
the external dependencies. Native yaml-cpp, LibXC, and FFT dependencies remain
available in both profiles.

The build summary reports these capabilities independently:

- native CIDER: enabled/disabled;
- external CIDER bridge: enabled/disabled;
- automatic runtime default: native/external.

The native release guard continues to reject Python, NumPy, CiderPress, and
`libciderbridge` from an external-disabled artifact. A separate dual-build
audit requires the expected bridge/Python dependencies and verifies that the
native evaluator is also present.

## Runtime selection

ABACUS gains a `cider_backend` input parameter with three accepted values:

- `auto` (the input default);
- `external`;
- `native`.

Resolution occurs during input conversion and is immutable for the run:

| Compiled capabilities | `auto` resolves to | Explicit `external` | Explicit `native` |
| --- | --- | --- | --- |
| Native only | native | fail before SCF | native |
| Native plus external | external | external | native |

Unknown values and unavailable backends fail before model or SCF setup. The
diagnostic includes the requested value and both compiled capabilities. The
resolved backend is printed in the running log, build/runtime summary, and
training metadata where the schema permits adding non-breaking metadata.

A dual executable is linked to the external dependencies, but selecting
`native` must not initialize Python or construct a bridge context. A native-only
executable has no Python or bridge link/runtime dependency.

## Backend boundary

`PotCiderXC` remains the ABACUS-facing potential component and owns the shared
work:

1. gather rank-local ABACUS density fields into the established global
   spin-major layout;
2. select valence or valence-plus-pseudo-core tuples according to model and
   input policy;
3. invoke one backend instance;
4. reconstruct local scalar and kinetic-energy-density potentials;
5. reduce and publish energy components, diagnostics, and training outputs;
6. scatter rank-local results.

Backend-specific model ownership and evaluation move behind a common internal
interface. The interface exposes:

- immutable model metadata (`is_mgga`, feature-density policy and source,
  energy convention, and model family);
- construction from model path, grid, cell, spin count, and exchange mixing;
- role-separated baseline and feature evaluation;
- energy-density, `vrho`, `vsigma`, and optional `vtau` outputs;
- destruction through backend-owned lifetime management.

`NativeCiderBackend` wraps the existing native model and
`DualDensityEvaluator`. `ExternalCiderBackend` restores and contains the
existing `cider_bridge_create`, `cider_bridge_evaluate_dual`, metadata-query,
and destruction calls. `PotCiderXC` contains no backend-specific handles.

Both adapters must populate the same role-separated result structure. This is
the accepted comparison boundary. The historical bridge implementation that
combined per-particle baseline exchange with feature energy density is not
restored as a parity target.

## Model inspection and compatibility

Backend selection precedes model inspection.

- Native selection uses the safe native YAML metadata inspector and therefore
  accepts only the documented native compatibility profile.
- External selection retains the existing bridge-compatible behavior and can
  continue to run external-only families such as `NNEvaluator` or other YAML
  variants supported by the installed CiderPress version.

The native inspector remains authoritative for early MGGA and density-policy
setup in native mode. External mode preserves the legacy preallocation
behavior used by the existing bridge path, including its existing MGGA model
identification, and validates authoritative bridge metadata when the external
context is created. This change does not silently route a model rejected by
the native loader to the external backend; fallback occurs only when the user
selected `external` or `auto` resolved to external.

Each MPI rank resolves the same backend. Backend kind and model metadata are
checked collectively before SCF evaluation so rank-local failures cannot leave
other ranks inside a different initialization path.

## Error handling

- An unavailable requested backend fails during input conversion.
- Native YAML/schema failures keep the existing structured native error code,
  YAML path, model path, and reason.
- External initialization and evaluation failures keep the bridge error and
  identify the selected external backend and model path.
- Metadata disagreement across MPI ranks is fatal before SCF.
- Backend exceptions must not partially mutate ABACUS output potentials or
  published energy components.
- Logs identify `backend=native` or `backend=external` on construction and on
  diagnostic evaluation records.

## Testing and acceptance

### Build profiles

- Configure and build serial and MPI native-only executables.
- Configure and build serial and MPI dual-backend executables.
- Build the CUDA PW target for both relevant profiles when the installed
  toolkit permits; CIDER evaluation may remain host-side.
- Confirm the native-only binary has no Python, NumPy, CiderPress, or
  `libciderbridge` dependency.
- Confirm the dual binary contains both backend capabilities and the expected
  external dependency closure.

### Selection behavior

- In a native-only binary, verify `auto` and `native` select native and
  `external` fails before SCF.
- In a dual binary, verify omitted/`auto` and explicit `external` select the
  bridge while explicit `native` selects the native evaluator.
- Verify invalid values fail consistently on every MPI rank.
- Verify selecting native in a dual binary does not initialize a Python
  interpreter or create a bridge context.

### Numerical parity

Run one dual executable twice with byte-identical `INPUT`, structure,
pseudopotential, YAML model, MPI layout, and initial density, changing only
`cider_backend`. Compare:

- role-separated feature energy density;
- `vrho`, `vsigma`, and `vtau` where applicable;
- baseline, feature, and total integrated energies;
- reconstructed scalar and kinetic-energy-density potentials;
- one- and two-rank MPI fields and reductions;
- selected one-iteration and converged SCF results.

The five existing frozen Python-oracle models remain the minimum matrix:
SL-GGA, SL-MGGA, NLDF-GGA, NLDF-MGGA, and the trained NumPy-2-era NLDF-MGGA
artifact. At least one external-only YAML model verifies that external
compatibility is preserved and that explicit native selection produces the
documented rejection rather than fallback.

Parity tolerances are applied to the current role-separated bridge contract.
The obsolete April combined-energy final totals are documented historical
controls, not acceptance targets.

### Regression coverage

- Existing native loader, evaluator, derivative, MPI, and release tests remain
  green.
- Existing external backend examples remain runnable in a dual build.
- Training-dump schemas and energy-component meanings remain unchanged; a
  backend-identification metadata field may be added only where readers ignore
  unknown metadata safely.

## Documentation and migration

Build documentation shows both profiles and their dependencies. Input help
documents the conditional `auto` resolution and the temporary external default
for dual builds. Native compatibility documentation remains explicit that
unsupported artifacts are rejected rather than automatically delegated.

Developers can use one external-enabled executable for native/external parity
work. Distribution and CI can continue to produce a Python-free native-only
artifact by leaving `ENABLE_CIDER_EXTERNAL=OFF`.

## Acceptance criteria

- One ABACUS source revision builds both native-only and dual-backend profiles.
- One dual executable switches between external and native execution solely
  through `cider_backend` in `INPUT`.
- A dual build defaults to external; a native-only build defaults to native.
- External-only model support remains available in external mode.
- Native mode never falls back silently to external execution.
- Same-binary fixed-density parity passes for the five-model supported native
  corpus under the current role-separated convention.
- Native-only release artifacts remain free of Python and bridge dependencies.
- The original `abacus-cider` worktree remains unmodified.
