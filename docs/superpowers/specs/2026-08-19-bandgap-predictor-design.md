# Bandgap Predictor Integration Design

Date: 2026-08-19

## Purpose

`bandgap-predictor` provides one local Python interface and the `bgp` command for predicting the electronic band gap of a periodic crystal structure with one of three explicitly named backends:

- `gnn-hse06`: the existing structure-only GNN trained against HSE06-like targets.
- `abacus-hse06`: ABACUS plane-wave HSE06 using the accelerated GPU EXX implementation.
- `cider-pbe0`: native CIDER in ABACUS using the original PseudoDojo, PBE0-reference model convention.

The package accepts a structure exactly as supplied, runs one selected backend synchronously on the local machine, and returns a provenance-rich result. It owns the ABACUS source, model artifacts, and PseudoDojo pseudopotentials needed by the supported workflows. ABACUS compilation is an explicit installation step separate from Python package installation.

## Scope

Version 1 supports three-dimensional periodic, non-spin-polarized calculations without spin-orbit coupling. It does not relax, standardize, symmetrize, wrap, sort, or otherwise alter the supplied structure. Converting the structure to an ABACUS input representation preserves the lattice vectors, Cartesian positions, chemical identities, and site ordering at sufficient precision for a numerical round trip.

Version 1 is local and synchronous. It does not include a scheduler, daemon, remote execution, workflow engine, graphical interface, magnetic initialization, SOC, geometry optimization, training interface, arbitrary ABACUS parameter overrides, or simultaneous multi-backend execution.

## Scientific Contract

The three backends represent different methods and are not required to agree. Every result identifies the backend, method, model, pseudopotentials, numerical settings, executable, and source revision.

The GNN and accelerated ABACUS backends are identified as HSE06. Native CIDER is identified as `cider-pbe0`, uses `cider_xmix=0.25`, and retains the initial PseudoDojo valence-only feature convention. CIDER descriptors use valence-only `rho`, `sigma`, and `tau`; nonlinear core charge present in a PseudoDojo UPF is excluded from CIDER feature inputs. The later SG15/no-NLCC model and later NLCC-inclusive retraining are not part of this package.

A converged calculation with fractional occupation or metallic band crossing returns a numerical gap of `0.0 eV` and an explicit classification such as `partial_occupancy` or `metal`. It is not described as an insulating gap. A non-converged or unparseable calculation returns no numerical gap. This allows the occupancy-aware zero-gap CIDER result for InSb to coexist with an HSE-like GNN prediction without hiding their different provenance.

## Repository Layout

The integration repository has these principal areas:

```text
pyproject.toml
src/bandgap_predictor/
  api.py
  cli.py
  config.py
  schemas.py
  structures.py
  assets.py
  runs.py
  backends/
    base.py
    gnn_hse06.py
    abacus_hse06.py
    cider_pbe0.py
  abacus/
    inputs.py
    runner.py
    parser.py
    progress.py
assets/
  manifest.json
  models/gnn/
  models/cider/
  pseudopotentials/pseudo-dojo/
vendor/abacus/
tools/build_abacus.py
examples/
tests/
```

Modules may be split further when a focused unit grows, but the boundaries remain stable: presentation, application API, backend adapters, ABACUS execution, structure conversion, assets, and result storage do not absorb each other's responsibilities.

## Public API and CLI

The application boundary consists of a typed `PredictionRequest`, `PredictionResult`, and `PredictionService.predict(request)` operation. Backend implementations conform to a common protocol with metadata, request validation, availability checks, and prediction execution. The CLI is a thin client of this API. A future GUI can call the same service without invoking a shell command or parsing terminal output.

The primary command is:

```text
bgp predict STRUCTURE --backend {gnn-hse06,abacus-hse06,cider-pbe0}
                    [--abacus PATH] [--work-root DIRECTORY]
```

Supporting read-only commands list backend availability, validate assets, and diagnose the ABACUS executable:

```text
bgp backends
bgp assets verify
bgp doctor [--abacus PATH]
```

One backend runs per `predict` invocation. Human-readable progress and a final summary go to the terminal; the authoritative machine result is `result.json` in the retained run directory.

## Executable Configuration and Build

ABACUS is compiled separately after the Python environment is installed. `tools/build_abacus.py` configures and builds the vendored source as one PW-only CUDA+MPI executable with native CIDER, LibXC, and GPU EXX enabled. The helper produces the executable plus a machine-readable build manifest containing the source revision, integration patch identity, CMake options, compiler and library versions, CUDA architectures, build time, and executable SHA-256.

The application never compiles ABACUS implicitly during `pip`, `uv`, import, or `bgp predict`. The compiled executable is resolved in this order:

1. `bgp ... --abacus PATH`
2. `abacus.executable` in the user configuration file at `$XDG_CONFIG_HOME/bandgap-predictor/config.toml`, defaulting to `~/.config/bandgap-predictor/config.toml`
3. `BGP_ABACUS`

ABACUS backends fail preflight if the path is absent, not executable, lacks the companion build manifest, or does not match the manifest hash and required feature set. `mpirun` must be available on `PATH`. Local execution uses one MPI rank and sets `OMP_NUM_THREADS=1`.

## ABACUS Source Ownership

No repository outside this workspace is modified. Source integration is performed on a workspace-local ABACUS branch derived from native CIDER revision `7828d7cc2` and updated with the current GPU EXX work represented by `73a8bd8a8`, including reduced-k fixes, automatic tiling, hardened execution context, and compatible upstream changes. The older combined revision `1ace47911` is evidence and a comparison point, not the final vendored source.

After source-level and scientific regressions pass, a clean tracked snapshot is exported to `vendor/abacus`. Build products, run artifacts, and the source repository object database are excluded. An upstream-revision document and integration patch record make the snapshot reproducible. ABACUS LGPL-3.0 notices and the GPL-3-or-later notices retained by migrated native CIDER files remain adjacent to their sources. The distribution documents its composite licensing rather than assigning an inaccurate single license to all bundled material.

## Asset Registry

The package bundles immutable, hash-addressed assets:

- GNN HSE model SHA-256 `4afc7500a467cba1f8a00786ef25887b76b02105e09e41e1d644f85d38aaecf9`.
- GNN feature map SHA-256 `cb90bd2257dbf8cfc4e6b2b6d4348d4616e8e9680f1339b1461137032e0f1894`.
- CIDER YAML SHA-256 `0bdadb33599a4f5d26baa743b1a5849a1fb63e0ce6b0fdf7ad9479feb2e28734`.
- The 71-file scalar-relativistic PseudoDojo collection, with an individual SHA-256 for each UPF.

The asset manifest records logical ID, filename, type, size, hash, origin, citations, applicable notice, supported elements, and backend consumers. Runtime verification fails closed on missing or changed assets. Only the pseudopotentials required by a calculation are copied or linked into its run directory, while the result records their exact hashes.

The current serialized MACE artifact is a trusted-source PyTorch pickle with incomplete training provenance. It is labeled `legacy/provisional`; loading it produces a terminal warning and records the same warning in `result.json`. No untrusted user-supplied pickle loading is added.

## Locked Numerical Preset

The ABACUS adapters generate input from a versioned, package-owned preset. Its common requirements are a 100 Ry plane-wave cutoff, `kspacing=0.15`, no SOC, non-spin-polarized execution, the validated smearing and SCF policy, and double precision. HSE06 enables ACE and the validated automatic EXX tiling/grid policy. CIDER uses the original PseudoDojo model and its PBE0-reference mixing convention.

All generated values are serialized into `result.json`. Version 1 exposes no raw ABACUS overrides and rejects attempts to supply them. Changes to any numerical value require a new preset version and updated regression baselines.

## Request Lifecycle

1. Parse the command and resolve the requested backend.
2. Read the structure with ASE and retain the original input bytes.
3. Validate periodicity, cell rank, supported species, no-SOC/nonmagnetic scope, and the absence of forbidden overrides.
4. Compute the original-file hash and a deterministic parsed-structure hash without changing the structure.
5. Verify backend-specific model and pseudopotential assets.
6. For an ABACUS backend, resolve and verify the executable and build manifest.
7. Allocate a unique persistent run directory beneath `--work-root` or `./bgp-runs`.
8. Write an initial request record atomically and copy the original structure.
9. Execute the selected backend synchronously.
10. Parse output, classify occupations, assemble provenance, and atomically write the final `result.json`.
11. Render a concise terminal summary containing status, method, gap or failure, elapsed time, and artifact path.

The GNN adapter verifies its assets, converts the unchanged parsed structure to the expected MACE graph, and evaluates the scalar model in process.

The ABACUS adapters select PseudoDojo UPFs by element and generate `STRU`, `INPUT`, and any required k-point files. The runner launches `mpirun -np 1` with `OMP_NUM_THREADS=1`, streams concise milestone messages, and saves complete stdout and stderr. The CIDER adapter additionally stages the verified YAML model. Generated input and staged assets become immutable run evidence once execution begins.

On an interrupt, the runner forwards termination to the complete local process group, allows a bounded graceful shutdown, and escalates locally if needed. It always preserves the run directory and writes an interrupted result if the directory has been allocated.

## Result Schema

`result.json` uses schema version `1.0` and contains these logical sections:

- Identity: run ID, status (`success`, `failed`, or `interrupted`), timestamps, elapsed seconds, and artifact path.
- Request: backend ID, method, original structure name, formula, original-file hash, and parsed-structure hash.
- Prediction: nullable `gap_ev`, electronic classification (`insulator`, `zero_gap`, `partial_occupancy`, `metal`, or `unknown`), convergence state, and occupation summary.
- Settings: preset ID and the complete effective numerical configuration.
- Provenance: package version, backend implementation version, executable hash and build metadata, ABACUS source revisions, model hashes, and pseudopotential hashes.
- Diagnostics: warnings, failure category, process return code, parser evidence, and relevant artifact filenames.

Paths in the persisted result are absolute. The Python API returns a typed representation of the same schema. Schema additions remain backward-compatible within major version 1; incompatible changes increment the schema major version.

## Errors and Exit Status

Failures are categorized at the layer that can explain them:

- Exit `0`: successful prediction, including a scientifically valid zero-gap or metallic result.
- Exit `2`: command usage, structure validation, or unsupported-request error before execution.
- Exit `3`: configuration, executable capability, build-manifest, or asset-integrity preflight failure.
- Exit `4`: calculation launch failure, abnormal process exit, or non-convergence.
- Exit `5`: output parsing or result-serialization failure.
- Exit `130`: user interruption.

Validation and preflight errors occurring before run allocation produce a concise diagnostic without fabricating a run result. After a run directory is allocated, every termination path attempts to write `result.json` atomically and reports the artifact path. Raw ABACUS logs remain available for diagnosis. A non-converged calculation never exposes a provisional gap as a successful result.

## Run-Directory Policy

The default root is `./bgp-runs`; `--work-root` changes it for one invocation. Run directories use a sortable timestamp, formula, backend ID, and collision-resistant run ID. They are retained by default in all statuses and are never automatically deleted. Temporary writes use same-directory atomic replacement so interruption cannot leave a valid-looking partial JSON document.

## Testing Strategy

Common example structures are constructed reproducibly with ASE and pinned by lattice, coordinates, species ordering, and hashes:

- Zincblende InSb, the primary narrow-gap and partial-occupancy-sensitive system.
- Diamond Si.
- Zincblende GaAs.
- Rocksalt MgO.

Fast tests cover request/result schemas, configuration precedence, structure round trips, atom-order preservation, asset verification, input generation, backend registration, run-directory handling, atomic result writes, signal handling, and stable CLI exit codes. Parser fixtures cover a converged insulator, zero gap, fractional occupations, non-convergence, abnormal termination, missing files, and malformed output.

The GNN regression suite evaluates all four systems and pins outputs within explicit tolerances. ABACUS runner tests use controlled fixture processes in ordinary CI, while real MPI/GPU jobs are separately marked and never run implicitly.

The required hardware release gate runs InSb and diamond Si through both `abacus-hse06` and `cider-pbe0` with one MPI rank and one OpenMP thread. The extended hardware matrix runs GaAs and MgO through both ABACUS backends. Tests compare each backend only with its own versioned baseline and verify convergence classification, provenance, inputs, hashes, and retained artifacts. Cross-backend numerical agreement is not an acceptance criterion.

ABACUS source integration additionally runs native CIDER parity/regression tests and the GPU EXX reduced-k, symmetry-grid, automatic-tiling, and execution-context regressions before the source snapshot is accepted.

## Acceptance Criteria

The design is implemented when:

1. A clean environment can install the Python package with `uv`, build the owned ABACUS source in the documented separate step, and configure `bgp` with the resulting executable.
2. `bgp predict` accepts an unchanged supported crystal structure and selects any of the three backends.
3. Both ABACUS adapters use the same bundled PseudoDojo family, while CIDER applies the specified valence-only feature convention.
4. Locked inputs, local synchronous execution, interrupt-safe artifact retention, and executable precedence behave as specified.
5. Successful, zero-gap, failed, non-converged, and interrupted outcomes are distinguishable through terminal output, exit code, and schema-valid JSON.
6. Every numerical result contains enough hashes and source/build metadata to identify the executable, method, model, pseudopotentials, settings, and supplied structure.
7. Fast automated tests pass, and the InSb/Si GPU+MPI release matrix passes on the target machine before a release is declared.
8. Existing repositories and data sources outside this workspace remain unmodified.
