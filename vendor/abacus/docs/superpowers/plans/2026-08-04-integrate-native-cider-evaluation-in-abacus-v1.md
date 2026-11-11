# Integrate Native CIDER Functional Evaluation in ABACUS

## Objective

Implement CIDER functional inference inside ABACUS without embedding Python and without requiring the `ciderpress` Python package at runtime. ABACUS must directly load the currently deployed Python-tagged YAML functional artifacts, evaluate the accepted `MappedXC` spline-model subset in native C++ and C, and preserve the existing CIDER energy, potential, density-policy, spin, MGGA, diagnostic, and training-label behavior.

The initial release is deliberately bounded to the functionals already exercised by ABACUS: semilocal GGA and MGGA models plus version-J NLDF GGA and MGGA models, one or two spin channels, `MappedDFTKernel` in separated-spin mode, `SplineSetEvaluator`, U/V/Z feature maps, identity and constant normalizers, and the observed LDA/Chachiyo/PBE baseline functions. Joblib, arbitrary Python object reconstruction, `MappedXC2`, neural evaluators, NLOF, SDMX runtime operators, LCAO, noncollinear spin, distributed NLDF FFT, and GPU-native CIDER kernels are outside the first release.

## Source-Grounded Architecture

- `abacus-cider/source/source_estate/module_pot/pot_cider_xc.cpp:890` is the stable ABACUS adapter. It already owns model-path input, density-role selection, grid/layout conversion, MPI global-grid assembly, energy integration, potential reconstruction, MGGA operator application, diagnostics, and training-label state.
- `ciderpress/lib/bridge/cider_bridge_py.py:449` is the numerical reference pipeline to reproduce: regularize density ingredients, construct semilocal and NLDF features, normalize features, evaluate mapped splines, reverse feature normalization, apply semilocal and NLDF adjoints, and return role-separated derivatives.
- `abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml:1` is the primary accepted trained-artifact fixture. Stock GGA/MGGA and NLDF models under `functionals/` provide backward-compatibility fixtures.
- `ciderpress/lib/pwutil/nldf_fft_core.h:101` and `ciderpress/lib/mod_cider/cider_coefs.c:183` expose reusable compiled NLDF convolution and interpolation primitives. They should be migrated as ordinary ABACUS-native targets rather than retained as Python extension libraries.
- `abacus-cider/source/source_hamilt/module_xc/CMakeLists.txt:1` is the planned ownership boundary for a new focused `cider/` subtree containing the loader, typed model, mapped evaluator, semilocal plan, NLDF engine, and tests. `PotCiderXC` remains a thin electronic-state adapter.

## Implementation Plan

### Phase 1: Freeze the compatibility contract and oracle corpus

- [ ] 1. Define and document the exact legacy YAML compatibility profile that the native loader will accept, using the current stock and accepted trained models as the normative corpus.

  Affected files: `functionals/CIDER23X_SL_GGA.yaml:1`, `functionals/CIDER23X_NL_GGA.yaml:1`, `functionals/CIDER23X_SL_MGGA.yaml:1`, `functionals/CIDER23X_NL_MGGA_DTR.yaml:1`, and `abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml:1`.

  Work: Record the whitelisted root, kernel, evaluator, settings, feature-map, normalizer, NumPy reconstruction, scalar, and named-function tags; required and optional attributes; supported dtypes, byte order, array order, shapes, metadata defaults, and NumPy 1 versus NumPy 2 tag aliases. Define deterministic errors for joblib, unknown tags, unsupported evaluators, unsupported descriptor families, malformed arrays, inconsistent feature counts, and missing density-policy metadata.

  Rationale: Existing YAML files are Python object graphs rather than a stable cross-language schema. A strict compatibility profile prevents a native parser from becoming an unsafe or incomplete Python deserializer while guaranteeing that accepted deployed artifacts remain usable.

- [ ] 2. Generate and commit a compact, versioned Python-reference fixture suite before changing runtime behavior.

  Affected files: `ciderpress/lib/bridge/cider_bridge_py.py:156`, `ciderpress/lib/bridge/cider_bridge_py.py:352`, `abacus_training_docs/compare_abacus_cider_dump.py:47`, and new fixture manifests under the native CIDER test subtree.

  Work: Capture representative loader metadata, decoded model arrays, semilocal raw features, NLDF interpolation arguments and convolution features, normalized features, transformed descriptors, spline energies and derivatives, reverse feature derivatives, role-separated baseline and feature outputs, and integrated ABACUS results. Include GGA, MGGA, NLDF, non-NLDF, spin-unpolarized, spin-polarized, valence, and valence-plus-pseudo-core cases, with checksums and provenance for the source model and Python reference version.

  Rationale: Once Python is removed from ABACUS, these fixtures become the independent numerical oracle and prevent implementation tests from merely comparing native code with itself.

### Phase 2: Establish the native module and safe model loader

- [ ] 3. Create a focused native CIDER module and CMake target under ABACUS XC ownership, with public interfaces separated from implementation details.

  Affected files: `abacus-cider/source/source_hamilt/module_xc/CMakeLists.txt:1`, `abacus-cider/source/source_estate/CMakeLists.txt:62`, `abacus-cider/CMakeLists.txt:692`, and new files under `abacus-cider/source/source_hamilt/module_xc/cider/`.

  Work: Define independently testable loader, model, mapped evaluator, semilocal plan, NLDF plan/backend, and top-level evaluator units. Add a private YAML parser dependency using the repository's existing dependency-discovery and FetchContent conventions, preserve third-party license notices for migrated CiderPress sources, and keep Python headers and extension-module macros out of all native targets.

  Rationale: A layered native module prevents model parsing and scientific kernels from becoming entangled with `PotCiderXC`, and it permits focused unit tests without constructing an ABACUS SCF calculation.

- [ ] 4. Implement a safe legacy-YAML and NumPy-payload reader for the whitelisted artifact profile.

  Affected files: new loader sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, with reference behavior from `ciderpress/dft/model_utils.py:28` and artifact encodings beginning at `abacus_training_docs/artifacts/combined_diverse_bandedge_nldf_mgga_20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml:50`.

  Work: Parse YAML nodes without constructing arbitrary objects; normalize `numpy.core` and `numpy._core` tag spellings; decode NumPy scalar and ndarray reconstruction sequences, base64 payloads, dtype descriptors, dimensions, C/Fortran order, and byte order; reject object dtypes and unbounded allocations; and report the YAML path plus node context for every validation failure.

  Rationale: Direct compatibility with current artifacts is the user-facing goal, and parser validation is a security and correctness boundary because current PyYAML tags encode executable Python type names.

- [ ] 5. Build a typed, immutable native model representation with authoritative metadata inspection and structural validation.

  Affected files: new model/schema sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, `ciderpress/dft/settings.py:1278`, `ciderpress/dft/settings.py:1681`, and `ciderpress/dft/xc_evaluator.py:506`.

  Work: Represent semilocal and version-J NLDF settings, normalizers, feature maps, spline grids and coefficients, baseline identifiers, output scales, feature-density policy, energy convention, and exchange-label scale. Validate cross-field invariants such as feature counts, transform indexes, spline dimensions, grid extents, coefficient shapes, supported separated-spin mode, and MGGA tau requirements. Preserve documented legacy inference only where stock artifacts lack explicit metadata, and expose metadata inspection before the SCF potential is allocated.

  Rationale: Typed validation localizes compatibility decisions, removes runtime string dispatch from numerical loops, and replaces ABACUS's current filename/text heuristic for detecting MGGA requirements.

### Phase 3: Port mapped functional mathematics

- [ ] 6. Implement native feature normalization and U/V/Z descriptor transforms with exact reverse derivatives.

  Affected files: new mapped-feature sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, with reference formulas at `ciderpress/dft/feat_normalizer.py:245`, `ciderpress/dft/transform_data.py:88`, `ciderpress/dft/transform_data.py:172`, `ciderpress/dft/transform_data.py:599`, and `ciderpress/dft/transform_data.py:1322`.

  Work: Support identity and constant normalization across the observed semilocal modes, compute normalized feature vectors, propagate derivatives back to raw features, evaluate the accepted descriptor maps, and apply their transposed Jacobians. Keep unsupported transform tags as load-time errors rather than silent approximations.

  Rationale: These operations are small but derivative-sensitive; isolating them enables strict scalar and vector golden tests before adding FFT and SCF complexity.

- [ ] 7. Implement the native tensor-product spline evaluator and derivatives for stored `SplineSetEvaluator` tables.

  Affected files: new spline sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, with reference orchestration at `ciderpress/dft/xc_evaluator.py:378`.

  Work: Reproduce the stored uniform-grid coefficient convention used by the Python `interpolation` package, including boundaries, one- through four-dimensional index sets, per-term scales, constants, coefficient tensor layout, accumulated energy, and derivatives with respect to transformed descriptors. Optimize only after scalar, batched, boundary, and out-of-domain behavior matches the oracle.

  Rationale: Accepted production models use mapped spline tables rather than runtime Gaussian-process kernels, so spline parity is the central model-evaluation requirement.

- [ ] 8. Implement the native `MappedDFTKernel` and `MappedXC` execution path for separated-spin spline models.

  Affected files: new mapped-model sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, with reference flow at `ciderpress/dft/xc_evaluator.py:506` and `ciderpress/dft/xc_evaluator.py:623`.

  Work: Assemble transformed descriptors per spin, evaluate and sum spline terms, propagate derivatives through feature maps, apply LDA multiplicative and Chachiyo/PBE additive baselines, honor optional output and exchange-label scaling, apply the low-density cutoff, and combine separated-spin energy contributions exactly as the Python reference does.

  Rationale: This completes model inference independently of density-feature generation and allows mapped models to be tested using stored raw-feature fixtures.

### Phase 4: Port semilocal and version-J NLDF feature evaluation

- [ ] 9. Implement semilocal GGA/MGGA feature construction and reverse potentials for the supported modes.

  Affected files: new semilocal-plan sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, with reference behavior at `ciderpress/dft/plans.py:428` and settings at `ciderpress/dft/settings.py:1630`.

  Work: Port density, reduced-gradient, and iso-orbital/kinetic-energy-density feature formulas; spin scaling; density floors; same-spin sigma handling; and reverse derivatives into `vrho`, `vsigma`, and `vtau`. Cover the observed GGA `np` and MGGA `npa` modes first and reject unimplemented modes during model validation.

  Rationale: Semilocal models provide the lowest-complexity end-to-end native milestone and the semilocal portion is also required by every accepted NLDF+MGGA model.

- [ ] 10. Port version-J NLDF alpha-grid preparation, interpolation, convolution, and adjoint kernels into ABACUS-native targets.

  Affected files: new NLDF sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, selected migrated sources from `ciderpress/lib/mod_cider/cider_coefs.c:183`, `ciderpress/lib/pwutil/nldf_fft_core.c:1`, `ciderpress/lib/pwutil/nldf_fft_serial.c:1`, and reference plan setup at `ciderpress/dft/plans.py:2016`.

  Work: Reproduce alpha-grid selection, Gaussian normalization/exponent matrices, stable interpolation-table construction, density-dependent exponent values and derivatives, coefficient interpolation, FFT-kernel setup, forward feature convolution, and backward potential convolution. Compile the reused C kernels as ordinary ABACUS libraries against ABACUS-selected FFTW or MKL/OpenMP dependencies, with no Python extension API. Preserve the current serial full-grid backend for the first release.

  Rationale: Reusing proven compiled kernels reduces numerical risk, while moving setup and orchestration to typed native code removes the embedded interpreter.

- [ ] 11. Assemble the complete native dual-density evaluator, including LibXC baselines, regularization, runtime mixing, diagnostics, and finite-output checks.

  Affected files: new top-level evaluator sources under `abacus-cider/source/source_hamilt/module_xc/cider/`, with reference behavior at `ciderpress/lib/bridge/cider_bridge_py.py:324`, `ciderpress/lib/bridge/cider_bridge_py.py:403`, `ciderpress/lib/bridge/cider_bridge_py.py:411`, and `ciderpress/lib/bridge/cider_bridge_py.py:641`.

  Work: Accept role-separated baseline and feature density tuples in the existing spin-major layout; validate and regularize inputs; evaluate baseline exchange/correlation through ABACUS-linked LibXC; evaluate native CIDER exchange; scale the trained exchange contribution to `cider_xmix`; return separate energy, density, gradient, and tau derivatives; retain feature-density and energy-convention metadata; and produce structured errors for non-finite inputs or outputs.

  Rationale: Preserving the existing dual-density contract lets ABACUS retain its validated NLCC, energy-integration, and potential-reconstruction behavior while replacing only the Python-owned inference core.

### Phase 5: Integrate with ABACUS and remove Python linkage

- [ ] 12. Replace the bridge context in `PotCiderXC` with the native evaluator while preserving ABACUS-facing behavior and diagnostics.

  Affected files: `abacus-cider/source/source_estate/module_pot/pot_cider_xc.h:14`, `abacus-cider/source/source_estate/module_pot/pot_cider_xc.cpp:890`, `abacus-cider/source/source_estate/module_pot/potential_types.cpp:73`, and `abacus-cider/source/source_io/module_training/training_dump.cpp:1030`.

  Work: Construct the native model/evaluator from `cider_model`, query authoritative metadata, replace `cider_bridge_evaluate_dual`, preserve density-role selection, global gather/scatter, energy convention, native ABACUS potential reconstruction, MGGA `vofk`, diagnostic dumps, potential clipping switches, timers, and `EnergyComponents`. Rename bridge-oriented messages without changing input semantics or training-dump fields.

  Rationale: Restricting integration changes to the evaluator seam protects the already validated ABACUS density and Hamiltonian conventions.

- [ ] 13. Replace heuristic model inspection during input conversion and update the supported input contract.

  Affected files: `abacus-cider/source/source_io/module_parameter/input_conv.cpp:173`, `abacus-cider/source/source_io/module_parameter/input_conv.cpp:542`, `abacus-cider/source/source_io/module_parameter/input_parameter.h:260`, `abacus-cider/source/source_io/module_parameter/read_input_item_ofdft.cpp:831`, and associated input tests.

  Work: Inspect native model metadata before potential allocation to set the kinetic-energy-density flag reliably; restrict `cider_model` documentation to supported YAML artifacts; reject joblib and unsupported YAML families with actionable messages; retain PW, spin, mixing, feature-density, and native-EXX compatibility checks; and ensure MPI ranks report identical model metadata and failures.

  Rationale: Filename-based MGGA detection can fail for renamed models and is incompatible with a typed native loader.

- [ ] 14. Remove Python and external CiderPress build/runtime requirements from `ENABLE_CIDER` and make the native runtime install with ABACUS.

  Affected files: `abacus-cider/CMakeLists.txt:31`, `abacus-cider/CMakeLists.txt:692`, `abacus-cider/source/source_hamilt/module_xc/CMakeLists.txt:1`, `abacus-cider/source/source_estate/CMakeLists.txt:62`, install rules, and CIDER build documentation.

  Work: Delete `find_package` requirements for Python and NumPy from the CIDER path; remove `CIDERPRESS_DIR`, `CIDERPRESS_LIB_DIR`, `libciderbridge`, and Python library linkage; build the native loader/evaluator and minimal migrated C kernels as ABACUS-owned targets; link LibXC and the selected FFT backend through existing ABACUS dependencies; install all required native libraries or link them into the executable; and expose native-CIDER capability in build information.

  Rationale: The completion condition is an ABACUS executable that loads and evaluates CIDER YAML without a Python installation or CiderPress checkout.

### Phase 6: Verification, performance, and release transition

- [ ] 15. Add layered native unit and parity tests for parsing, model validation, transforms, splines, semilocal features, NLDF features, adjoints, and dual-density outputs.

  Affected files: new tests under `abacus-cider/source/source_hamilt/module_xc/cider/test/`, `abacus-cider/cmake/Testing.cmake:32`, and compact fixtures from task 2.

  Work: Test every accepted YAML tag and metadata variant; malformed and unsupported artifacts; scalar and batched transform derivatives; spline values and derivatives at interior and boundary points; GGA/MGGA semilocal outputs; NLDF forward intermediates and backward potentials; spin behavior; exchange scaling; density policies; and all role-separated output arrays against the frozen Python oracle.

  Rationale: Layered parity tests identify the first divergent mathematical stage and make later Python removal auditable.

- [ ] 16. Validate functional derivatives with finite differences and verify ABACUS serial/MPI end-to-end behavior on the established CIDER examples and benchmark controls.

  Affected files: `abacus_training_docs/compare_abacus_cider_dump.py:137`, `abacus-cider/examples/02_scf/pw_Si2-cider/INPUT:1`, `abacus-cider-notes/implementation/abacus-cider-runtime-testing.md:1`, and ABACUS integration-test manifests.

  Work: Compare analytic `vrho`, `vsigma`, and `vtau` with central finite differences of native feature energy; run semilocal GGA, NLDF GGA, semilocal MGGA, and accepted NLDF+MGGA models; cover one and two spins; compare one-rank and multi-rank replicated-global-grid results; and confirm post-SCF CIDER labels and training dumps retain their schema and numerical meaning.

  Rationale: Matching intermediate arrays is necessary but not sufficient; the native path must reproduce variational derivatives, SCF convergence, total energies, potentials, and downstream data products.

- [ ] 17. Verify clean Python-free CPU and CUDA builds, runtime independence, performance, memory, and failure diagnostics before changing the default path.

  Affected files: CIDER build documentation, CI/build profiles, `abacus-cider/BUILD_CIDER.md:1`, and benchmark result notes under `abacus-cider-notes/`.

  Work: Build serial PW CPU, MPI PW CPU, and PW CUDA executables with `ENABLE_CIDER`; confirm no Python, NumPy, or `libciderbridge` dependency appears in configure caches or dynamic linkage; run with an empty Python environment and no CiderPress checkout; measure model-load time, per-SCF-call time, peak memory, and MPI replication cost; and verify unsupported artifacts fail before SCF with file, tag, and reason.

  Rationale: A numerically correct implementation does not satisfy the goal if the installed executable still depends on Python or introduces unacceptable runtime regressions.

- [ ] 18. Update durable architecture, compatibility, model-authoring, migration, and troubleshooting documentation after all release gates pass.

  Affected files: `abacus-cider/BUILD_CIDER.md:1`, relevant ABACUS input documentation, and durable notes under `abacus-cider-notes/`.

  Work: Document the supported YAML compatibility profile, native build flags and dependencies, removal of virtual-environment and `LD_LIBRARY_PATH` setup, known exclusions, model metadata requirements, oracle provenance, parity results, performance results, and a future schema-version/export recommendation. Preserve the Python bridge only as an external reference tool until the native parity suite is accepted, then remove or archive its ABACUS linkage.

  Rationale: Users and future model trainers need a precise contract for which artifacts ABACUS can consume and how compatibility evolves without reintroducing Python.

## Verification Criteria

- `ENABLE_CIDER=ON` configures and builds without finding or linking Python, NumPy, `libciderbridge`, or an external CiderPress checkout.
- The installed CPU and CUDA ABACUS executables load accepted YAML artifacts with an empty `PYTHONPATH`, no activated virtual environment, and no CiderPress libraries in `LD_LIBRARY_PATH`.
- Loader tests recover the expected metadata, scalar values, dimensions, dtypes, and array values from all five compatibility-corpus models; malformed or unsupported artifacts fail deterministically before SCF.
- NumPy 1 `numpy.core` and NumPy 2 `numpy._core` YAML reconstruction tags produce identical typed native arrays.
- Native feature normalization, U/V/Z transforms, and spline values/derivatives agree with frozen Python fixtures to maximum absolute error no greater than `1e-12` for deterministic scalar/batched cases.
- Semilocal raw features and reverse derivatives agree with Python fixtures to relative error no greater than `1e-11` and absolute error no greater than `1e-12` away from documented density cutoffs.
- Version-J NLDF interpolation arguments, convolution features, normalized features, mapped energies, and returned `vrho`, `vsigma`, and `vtau` agree with Python fixtures to relative error no greater than `1e-10` and absolute error no greater than `1e-12`, with separately documented tolerances at masking boundaries.
- Finite-difference feature-energy checks pass for rho, same-spin sigma, and tau channels with the same or tighter thresholds recorded by the Python oracle suite.
- One-rank and multi-rank replicated-global-grid native runs agree within `1e-10` for returned grid fields and within `1e-10 Ha` for integrated CIDER energy on fixed-density fixtures.
- Established Si and GaAs SCF controls converge through the native path with total energies within `1e-8 Ha`, reported gaps within `1e-6 eV`, and no unexplained change in convergence status relative to the frozen Python-bridge baselines.
- Post-SCF CIDER labels and training dumps retain the existing schema fields, density-policy meaning, exchange scaling, and energy conventions.
- Native model-load plus evaluation peak memory and steady-state evaluation time do not regress by more than 20 percent relative to the current Python bridge on the fixed benchmark grid; any exception is explicitly accepted and documented.
- Unsupported joblib, `MappedXC2`, NLOF, SDMX runtime, neural evaluator, unknown transform, object dtype, and malformed coefficient cases fail with actionable messages that identify the model path and unsupported construct.

## Potential Risks and Mitigations

1. **The current YAML format is an unstable Python serialization rather than a formal interchange schema.**
   Mitigation: Freeze a strict whitelisted compatibility profile, support both observed NumPy tag families, reject unknown constructs, commit representative artifacts and checksums, and introduce explicit schema/version metadata for future CiderPress exports without dropping legacy fixtures.

2. **Spline coefficient ordering or boundary behavior differs from the Python `interpolation` package.**
   Mitigation: Capture scalar and batched oracle cases for every observed spline dimensionality, boundary, grid size, and coefficient layout before implementation; gate mapped evaluation on exact low-level spline parity.

3. **Small derivative differences destabilize self-consistent NLDF+MGGA calculations.**
   Mitigation: Require intermediate-field parity, central finite-difference validation, fixed-density ABACUS dump replay, and SCF controls before promotion; keep the current Python oracle executable available outside ABACUS during migration.

4. **Migrated CiderPress FFT code conflicts with ABACUS FFTW, MKL, MPI, or OpenMP configuration.**
   Mitigation: Compile only the minimal C sources as private ABACUS targets, consume ABACUS-selected dependency targets, use separate symbols/namespaces where needed, and test serial CPU, MPI CPU, and CUDA-host configurations in clean build directories.

5. **The current replicated global grid limits memory and MPI scaling.**
   Mitigation: Preserve it for initial parity and measure it explicitly. Plan distributed NLDF FFT and local-grid evaluation as a separate optimization only after the native serial/global evaluator is accepted.

6. **MGGA state is allocated incorrectly because model requirements are discovered too late.**
   Mitigation: Replace filename/text heuristics with native metadata inspection during input conversion and test renamed MGGA/GGA artifacts before constructing `Potential`.

7. **Direct YAML parsing permits excessive allocations or unsafe tag handling.**
   Mitigation: Use an established parser, never instantiate types from tag names, impose file/node/dimension/element limits, validate multiplication overflow before allocation, accept only numeric dtypes, and fail closed on aliases or tags outside the compatibility profile.

8. **Vendoring numerical sources creates maintenance and license drift from CiderPress.**
   Mitigation: Preserve source provenance and GPL notices, record the upstream commit, isolate migrated kernels, add oracle tests, and establish an explicit process for syncing scientific fixes without copying unrelated Python code.

9. **Legacy artifact metadata relies on filenames or implicit defaults.**
   Mitigation: Preserve narrowly documented inference for stock CIDER23X fixtures, require explicit metadata for newly trained artifacts, log every inferred value, and add future exporter validation that writes all required native fields.

10. **The plan expands into support for every CiderPress model family.**
    Mitigation: Treat the five-model corpus and whitelisted tag table as the release boundary. Add new evaluators or descriptor families only through separate plans with their own fixtures, mathematics, and acceptance gates.

## Alternative Approaches

1. **Require CiderPress to export a new clean native schema and do not parse current YAML.**
   This would simplify and harden ABACUS, but it would not meet the direct-compatibility goal for existing accepted YAML artifacts. It remains the recommended long-term authoring format after the legacy loader is operational.

2. **Translate existing YAML to a native artifact with a standalone conversion command.**
   This keeps Python out of ABACUS and reduces parser complexity, but users would still need CiderPress for every existing model conversion. It is useful as a validation/reference tool, not as the only supported deployment path.

3. **Embed a minimal Python interpreter or bundle an inference-only Python package.**
   This preserves Python-object YAML behavior with less scientific porting, but it contradicts the final requirement and retains Python ABI, packaging, and shared-library conflicts.

4. **Rewrite NLDF immediately on ABACUS's distributed FFT infrastructure.**
   This could improve scaling, but it combines scientific-porting and parallel-layout changes, making parity failures difficult to localize. Preserve the global serial backend first, then optimize under a separate performance plan.

5. **Implement only the accepted NLDF+MGGA production model as generated C++ constants.**
   This is the shortest demonstration, but it cannot consume arbitrary compatible YAML artifacts and would require recompilation for every trained functional. It is unsuitable as the production architecture.

## Assumptions

- CiderPress remains the authoritative training and Python-reference implementation; only ABACUS inference becomes native.
- Direct legacy YAML compatibility means the documented `MappedXC` spline subset, not arbitrary PyYAML object graphs.
- The primary production target is plane-wave ABACUS with one or two collinear spin channels.
- CIDER evaluation remains host-side in the initial CUDA build and continues to use the current all-ranks global-grid replication.
- ABACUS's existing LibXC dependency remains mandatory for CIDER surrogate-hybrid baseline evaluation.
- The accepted trained NLDF+MGGA artifact and four stock functional families remain available as immutable compatibility fixtures.
- Implementation is expected to take approximately 9–16 engineer-weeks for one experienced developer through native parity and SCF acceptance; distributed MPI/GPU optimization is additional work.

## Deliverables

- A Python-free native CIDER loader and evaluator under ABACUS XC ownership.
- Direct support for the whitelisted current YAML artifacts, including NumPy 1 and NumPy 2 encodings.
- Migrated minimal version-J NLDF compiled kernels with preserved provenance and licensing.
- Layered loader, mathematical, derivative, MPI, and SCF parity tests backed by frozen Python oracle fixtures.
- Updated ABACUS build and runtime behavior with no CiderPress checkout, virtual environment, `PYTHONPATH`, or CIDER-specific `LD_LIBRARY_PATH` requirement.
- Durable compatibility, build, migration, validation, and troubleshooting documentation.
