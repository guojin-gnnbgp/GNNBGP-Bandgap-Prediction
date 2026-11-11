# Dual Native/External CIDER Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one ABACUS executable that contains the native and optional CiderPress bridge backends and selects `external` or `native` at runtime, while preserving a Python-free native-only build.

**Architecture:** `PotCiderXC` keeps all ABACUS grid, MPI, potential-reconstruction, energy, and training-output responsibilities. Backend-specific model ownership and role-separated evaluation move behind a small `PotCiderRuntime` interface implemented by native and external adapters; a factory selects the adapter once from the resolved `cider_backend` input.

**Tech Stack:** C++11/14, CMake, ABACUS PW/MPI infrastructure, native CIDER/yaml-cpp/LibXC/FFTW, optional CPython/NumPy/CiderPress `libciderbridge`, GoogleTest, CTest.

## Global Constraints

- Make every file change and commit only in `/home/bonan/appdir/CiderPress/.planning/worktrees/abacus-cider-native-evaluation`.
- Do not modify `/home/bonan/appdir/CiderPress/abacus-cider`; verify its tracked and staged state after every task.
- `ENABLE_CIDER=ON` always supplies the native backend.
- `ENABLE_CIDER_EXTERNAL=ON`, default `OFF`, adds the external backend and requires `ENABLE_CIDER=ON`.
- In a dual build, `cider_backend=auto` resolves to `external`; in a native-only build it resolves to `native`.
- Backend selection is immutable after input conversion; there is no mid-SCF switch and no silent fallback.
- Selecting `native` in a dual binary must not initialize Python or construct a bridge context.
- Native-only release artifacts must have no Python, NumPy, CiderPress, or `libciderbridge` dependency.
- Preserve the current role-separated baseline/feature energy convention, MPI gather/scatter layout, energy components, potential reconstruction, and training schemas.
- Do not treat the obsolete April combined-energy bridge totals as parity targets.
- Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` for CiderPress and SCF validation.
- Run MPI and GPU commands outside the sandbox with one OpenMP thread.

---

### Task 1: Add build capabilities and runtime backend resolution

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_ofdft.cpp`
- Modify: `source/source_io/module_parameter/cider_model_input.h`
- Modify: `source/source_io/module_parameter/cider_model_input.cpp`
- Modify: `source/source_io/module_parameter/test/cider_model_input_test.cpp`
- Modify: `source/source_io/module_parameter/input_conv.cpp`
- Modify: `cmake/CollectBuildInfoVars.cmake`

**Interfaces:**
- Consumes: existing `ENABLE_CIDER`, `Input_para`, and native model metadata inspection.
- Produces: `ENABLE_CIDER_EXTERNAL`; compile definition `ENABLE_CIDER_EXTERNAL`; `Input_para::cider_backend`; `ModuleIO::CiderBackendCapabilities`; `ModuleIO::resolve_cider_backend(const std::string&, const CiderBackendCapabilities&)`; a concrete `PARAM.inp.cider_backend` value before model inspection.

- [ ] **Step 1: Write failing resolver and input-registration tests**

Add pure resolver coverage to `cider_model_input_test.cpp` so it does not depend on the build profile:

```cpp
TEST(CiderModelInput, ResolvesBackendAgainstCompiledCapabilities)
{
    const ModuleIO::CiderBackendCapabilities native_only(true, false);
    EXPECT_EQ("native", ModuleIO::resolve_cider_backend("auto", native_only));
    EXPECT_EQ("native", ModuleIO::resolve_cider_backend("NATIVE", native_only));
    EXPECT_THROW(ModuleIO::resolve_cider_backend("external", native_only),
                 std::runtime_error);

    const ModuleIO::CiderBackendCapabilities dual(true, true);
    EXPECT_EQ("external", ModuleIO::resolve_cider_backend("auto", dual));
    EXPECT_EQ("external", ModuleIO::resolve_cider_backend("EXTERNAL", dual));
    EXPECT_EQ("native", ModuleIO::resolve_cider_backend("native", dual));
    EXPECT_THROW(ModuleIO::resolve_cider_backend("fallback", dual),
                 std::runtime_error);
}
```

Extend the existing input-item test to assert that omitted `cider_backend`
remains `"auto"` and an explicit value is synchronized into
`Input_para::cider_backend`.

- [ ] **Step 2: Run the focused tests and verify the new API is missing**

Run:

```bash
cmake --build /tmp/abacus-cider-task12-focused \
  --target MODULE_HAMILT_CIDER_model_loader_test MODULE_IO_CIDER_model_input_test -j2
```

Expected: compilation fails because `CiderBackendCapabilities`,
`resolve_cider_backend`, and `Input_para::cider_backend` are not defined.

- [ ] **Step 3: Implement the resolver and input field**

Add to `cider_model_input.h`:

```cpp
struct CiderBackendCapabilities
{
    CiderBackendCapabilities(bool native_in, bool external_in)
        : native(native_in), external(external_in) {}
    bool native;
    bool external;
};

std::string resolve_cider_backend(
    const std::string& requested,
    const CiderBackendCapabilities& capabilities);
```

Implement lowercase normalization and these exact rules in
`cider_model_input.cpp`: `auto` chooses external when available and native
otherwise; explicit choices require the corresponding capability; every error
includes `requested=`, `native=enabled|disabled`, and
`external=enabled|disabled`.

Add `std::string cider_backend = "auto";` beside `cider_model` in
`Input_para`. Register an `Input_Item("cider_backend")` whose documented values
are `auto`, `external`, and `native`.

Add at the top-level CMake option block:

```cmake
option(ENABLE_CIDER_EXTERNAL
       "Add the external CiderPress bridge backend to an ENABLE_CIDER build"
       OFF)
if(ENABLE_CIDER_EXTERNAL AND NOT ENABLE_CIDER)
  message(FATAL_ERROR "ENABLE_CIDER_EXTERNAL requires ENABLE_CIDER=ON")
endif()
```

Inside `if(ENABLE_CIDER)`, add `ENABLE_CIDER_EXTERNAL` as a compile definition
only when enabled. In `input_conv.cpp`, resolve `PARAM.inp.cider_backend` before
metadata inspection using compile-time capabilities:

```cpp
const ModuleIO::CiderBackendCapabilities capabilities(
    true,
#ifdef ENABLE_CIDER_EXTERNAL
    true
#else
    false
#endif
);
PARAM.inp.cider_backend = ModuleIO::resolve_cider_backend(
    PARAM.inp.cider_backend, capabilities);
```

Keep native metadata inspection only when the resolved value is `native`.
For `external`, initially preserve the legacy filename-based MGGA allocation
helper from base commit `46ca593c25793b33a6cae25838ab344dafd3be31` until Task 4
validates bridge metadata at runtime.

- [ ] **Step 4: Report both capabilities in build information**

Extend `CollectBuildInfoVars.cmake` so the summary prints native CIDER and
external CIDER separately. A native-only `abacus -i` must contain:

```text
Native CIDER Support:          yes
External CIDER Bridge:         no
Default CIDER Backend:         native
```

A dual build must report `yes` and `external` for the latter two fields.

- [ ] **Step 5: Run focused tests and configure-negative checks**

Run:

```bash
cmake --build /tmp/abacus-cider-task12-focused \
  --target MODULE_HAMILT_CIDER_model_loader_test MODULE_IO_CIDER_model_input_test -j2
ctest --test-dir /tmp/abacus-cider-task12-focused --output-on-failure \
  -R 'MODULE_(IO|HAMILT)_CIDER_(model_input|model_loader)_test'
cmake -S . -B /tmp/abacus-cider-invalid-external \
  -DENABLE_CIDER=OFF -DENABLE_CIDER_EXTERNAL=ON
```

Expected: focused tests pass; the final configure command fails with
`ENABLE_CIDER_EXTERNAL requires ENABLE_CIDER=ON`.

- [ ] **Step 6: Commit the selection contract**

```bash
git add CMakeLists.txt cmake/CollectBuildInfoVars.cmake \
  source/source_io/module_parameter/input_parameter.h \
  source/source_io/module_parameter/read_input_item_ofdft.cpp \
  source/source_io/module_parameter/cider_model_input.h \
  source/source_io/module_parameter/cider_model_input.cpp \
  source/source_io/module_parameter/test/cider_model_input_test.cpp \
  source/source_io/module_parameter/input_conv.cpp
git commit -m "feat: add CIDER runtime backend selection"
```

---

### Task 2: Introduce a backend-neutral PotCider runtime boundary

**Files:**
- Create: `source/source_estate/module_pot/pot_cider_xc_runtime.h`
- Create: `source/source_estate/module_pot/pot_cider_xc_runtime.cpp`
- Modify: `source/source_estate/module_pot/pot_cider_xc_native.h`
- Modify: `source/source_estate/module_pot/pot_cider_xc_native.cpp`
- Modify: `source/source_estate/module_pot/test/pot_cider_xc_native_test.cpp`
- Modify: `source/source_estate/CMakeLists.txt`

**Interfaces:**
- Consumes: `hamilt::cider::EvaluationInput`, `ComponentOutput`, native
  `ModelMetadata`, and `PotCiderNativeRuntime`.
- Produces: `PotCiderRuntimeMetadata`; abstract `PotCiderRuntime`; native
  implementation of that interface; no behavior change in native evaluation.

- [ ] **Step 1: Write the interface-conformance test**

Add to `pot_cider_xc_native_test.cpp`:

```cpp
static_assert(std::is_base_of<elecstate::PotCiderRuntime,
                              elecstate::PotCiderNativeRuntime>::value,
              "native CIDER runtime must implement the common backend");

TEST(PotCiderNativeRuntime, ReportsBackendNeutralMetadata)
{
    const char* root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
    ASSERT_NE(nullptr, root);
    const std::size_t shape[3] = {2, 2, 2};
    const double cell[9] = {4.0, 0.0, 0.0,
                            0.0, 4.0, 0.0,
                            0.0, 0.0, 4.0};
    elecstate::PotCiderNativeRuntime runtime(
        std::string(root) + "/functionals/CIDER23X_SL_MGGA.yaml",
        shape, cell, 2, 0.25);
    EXPECT_STREQ("native", runtime.backend_name());
    EXPECT_TRUE(runtime.metadata().requires_tau);
    EXPECT_EQ("energy_density", runtime.metadata().feature_exc_convention);
}
```

Keep the test's existing fixture-path and frozen-array helpers rather than
creating another fixture format.

- [ ] **Step 2: Run the test and verify the common type is absent**

Run:

```bash
cmake --build /tmp/abacus-cider-task12-focused \
  --target MODULE_ESTATE_CIDER_pot_native_test -j2
```

Expected: compilation fails because `PotCiderRuntime` and
`PotCiderRuntimeMetadata` do not exist.

- [ ] **Step 3: Define the backend-neutral interface**

Create `pot_cider_xc_runtime.h` with this public contract:

```cpp
struct PotCiderRuntimeMetadata
{
    bool requires_tau = false;
    std::string feature_density_policy;
    std::string feature_density_policy_source;
    std::string feature_exc_convention;
};

class PotCiderRuntime
{
  public:
    virtual ~PotCiderRuntime();
    virtual const char* backend_name() const = 0;
    virtual const PotCiderRuntimeMetadata& metadata() const = 0;
    virtual void evaluate(
        const hamilt::cider::EvaluationInput& input,
        const hamilt::cider::ComponentOutput& baseline,
        const hamilt::cider::ComponentOutput& feature) const = 0;
  protected:
    PotCiderRuntime() = default;
};
```

Define the out-of-line destructor in `pot_cider_xc_runtime.cpp`. Make
`PotCiderNativeRuntime final : public PotCiderRuntime`, translate immutable
native model metadata into a stored `PotCiderRuntimeMetadata`, return
`"native"`, and keep the existing evaluator call unchanged.

- [ ] **Step 4: Link the common runtime into the estate target**

Add `pot_cider_xc_runtime.cpp` to `cider_pot_native` in
`source/source_estate/CMakeLists.txt`. Do not add Python or bridge headers to
this target.

- [ ] **Step 5: Run the native runtime and evaluator suites**

Run:

```bash
cmake --build /tmp/abacus-cider-task12-focused \
  --target MODULE_ESTATE_CIDER_pot_native_test MODULE_HAMILT_CIDER_evaluator_test -j2
CIDER_NATIVE_CORPUS_ROOT=/home/bonan/appdir/CiderPress \
  ctest --test-dir /tmp/abacus-cider-task12-focused --output-on-failure \
  -R 'MODULE_(ESTATE_CIDER_pot_native|HAMILT_CIDER_evaluator)_test'
```

Expected: both tests pass with the same frozen-oracle tolerances as before.

- [ ] **Step 6: Commit the common runtime interface**

```bash
git add source/source_estate/module_pot/pot_cider_xc_runtime.h \
  source/source_estate/module_pot/pot_cider_xc_runtime.cpp \
  source/source_estate/module_pot/pot_cider_xc_native.h \
  source/source_estate/module_pot/pot_cider_xc_native.cpp \
  source/source_estate/module_pot/test/pot_cider_xc_native_test.cpp \
  source/source_estate/CMakeLists.txt
git commit -m "refactor: add common CIDER runtime interface"
```

---

### Task 3: Restore the external bridge as an optional runtime adapter

**Files:**
- Create: `source/source_estate/module_pot/pot_cider_xc_external.h`
- Create: `source/source_estate/module_pot/pot_cider_xc_external.cpp`
- Create: `source/source_estate/module_pot/test/pot_cider_xc_external_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `source/source_estate/CMakeLists.txt`

**Interfaces:**
- Consumes: `PotCiderRuntime`; C API `cider_bridge_create`,
  `cider_bridge_evaluate_dual`, metadata queries, and `cider_bridge_destroy`.
- Produces: `PotCiderExternalRuntime final : public PotCiderRuntime`, compiled
  only with `ENABLE_CIDER_EXTERNAL`.

- [ ] **Step 1: Write the real-bridge adapter test**

Create `pot_cider_xc_external_test.cpp`, guarded by
`ENABLE_CIDER_EXTERNAL`, that constructs the stock SL-GGA model on a `2x2x2`
grid and asserts:

```cpp
EXPECT_EQ("external", runtime.backend_name());
EXPECT_FALSE(runtime.metadata().requires_tau);
EXPECT_EQ("valence", runtime.metadata().feature_density_policy);
EXPECT_EQ("energy_density", runtime.metadata().feature_exc_convention);
```

Evaluate the frozen spin-one input and compare every baseline/feature `exc`,
`vrho`, and `vsigma` entry to the v1 fixture using its declared tolerance.

- [ ] **Step 2: Configure the dual build and verify the adapter is absent**

Run outside the sandbox if dependency discovery needs the shared cache:

```bash
cmake -S . -B /tmp/abacus-cider-dual \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -DENABLE_CIDER_EXTERNAL=ON -DENABLE_LCAO=OFF -DUSE_ELPA=OFF \
  -DCIDERPRESS_DIR=/home/bonan/appdir/CiderPress/ciderpress \
  -Dyaml-cpp_DIR=/tmp/yaml-cpp-local/lib/cmake/yaml-cpp \
  -DCIDER_NATIVE_CORPUS_ROOT=/home/bonan/appdir/CiderPress \
  -DBUILD_TESTING=ON
cmake --build /tmp/abacus-cider-dual \
  --target MODULE_ESTATE_CIDER_pot_external_test -j2
```

Expected: target/source failure because the external adapter has not been
implemented.

- [ ] **Step 3: Restore external dependency discovery under the new option**

Inside `if(ENABLE_CIDER_EXTERNAL)` in top-level `CMakeLists.txt`, restore the
base revision's exact Python and bridge discovery:

```cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_library(CIDERBRIDGE_LIBRARY ciderbridge
  HINTS ${CIDERPRESS_DIR}/lib ${CIDERPRESS_DIR}/ciderpress/lib
        ${CIDERPRESS_LIB_DIR} ENV CIDERPRESS_LIB_DIR)
find_path(CIDERBRIDGE_INCLUDE_DIR cider_bridge.h
  HINTS ${CIDERPRESS_DIR}/bridge ${CIDERPRESS_DIR}/lib/bridge
        ${CIDERPRESS_LIB_DIR}/bridge ${CIDERPRESS_LIB_DIR}/../bridge)
```

Fail with actionable `CIDERPRESS_DIR` guidance when either library or header is
missing. Link these dependencies only into the external adapter/final ABACUS
target, never into `cider_native` or native-only test targets.

- [ ] **Step 4: Implement `PotCiderExternalRuntime`**

The class stores `cider_bridge_ctx* ctx_` and a
`PotCiderRuntimeMetadata metadata_`. Its constructor calls
`cider_bridge_create(model_path.c_str(), grid, cell, nspin, cider_xmix)`, then
copies `is_mgga`, feature-density policy/source, and energy convention before
returning. Its destructor calls `cider_bridge_destroy(&ctx_)`.

Implement `evaluate` as a direct role-separated call:

```cpp
const int status = cider_bridge_evaluate_dual(
    ctx_, static_cast<int>(input.nspin()), static_cast<int>(input.ngrids()),
    input.baseline().rho(), input.baseline().sigma(), input.baseline().tau(),
    input.feature().rho(), input.feature().sigma(), input.feature().tau(),
    baseline.exc(), baseline.vrho(), baseline.vsigma(), baseline.vtau(),
    feature.exc(), feature.vrho(), feature.vsigma(), feature.vtau());
if (status != 0)
{
    throw std::runtime_error("external CIDER bridge evaluate_dual failed");
}
```

Validate sizes and integer narrowing before the call. Reject unknown policy or
energy-convention strings during construction. Do not initialize the external
adapter from native code paths.

- [ ] **Step 5: Build and run the external adapter test**

Run:

```bash
cmake --build /tmp/abacus-cider-dual \
  --target MODULE_ESTATE_CIDER_pot_external_test -j2
CIDER_NATIVE_CORPUS_ROOT=/home/bonan/appdir/CiderPress \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
ctest --test-dir /tmp/abacus-cider-dual --output-on-failure \
  -R MODULE_ESTATE_CIDER_pot_external_test
```

Expected: test passes and reports bridge/native-oracle agreement within the
fixture tolerance.

- [ ] **Step 6: Commit the optional external adapter**

```bash
git add CMakeLists.txt source/source_estate/CMakeLists.txt \
  source/source_estate/module_pot/pot_cider_xc_external.h \
  source/source_estate/module_pot/pot_cider_xc_external.cpp \
  source/source_estate/module_pot/test/pot_cider_xc_external_test.cpp
git commit -m "feat: restore optional external CIDER runtime"
```

---

### Task 4: Select the runtime through a factory and preserve metadata behavior

**Files:**
- Create: `source/source_estate/module_pot/pot_cider_xc_runtime_factory.h`
- Create: `source/source_estate/module_pot/pot_cider_xc_runtime_factory.cpp`
- Create: `source/source_estate/module_pot/test/pot_cider_xc_runtime_factory_test.cpp`
- Modify: `source/source_estate/module_pot/pot_cider_xc.h`
- Modify: `source/source_estate/module_pot/pot_cider_xc.cpp`
- Modify: `source/source_estate/CMakeLists.txt`
- Modify: `source/source_io/module_parameter/input_conv.cpp`

**Interfaces:**
- Consumes: resolved `PARAM.inp.cider_backend`, native/external runtime
  constructors, grid/cell/spin/xmix configuration.
- Produces: `make_pot_cider_runtime(...) -> std::unique_ptr<PotCiderRuntime>`;
  backend-neutral ownership in `PotCiderXC`.

- [ ] **Step 1: Write factory selection tests**

Create a test covering native construction and unavailable external selection:

```cpp
const char* root = std::getenv("CIDER_NATIVE_CORPUS_ROOT");
ASSERT_NE(nullptr, root);
const std::size_t shape[3] = {2, 2, 2};
const double cell[9] = {4.0, 0.0, 0.0,
                        0.0, 4.0, 0.0,
                        0.0, 0.0, 4.0};
const auto make_fixture_runtime = [&](const std::string& backend) {
    return elecstate::make_pot_cider_runtime(
        backend,
        std::string(root) + "/functionals/CIDER23X_SL_GGA.yaml",
        shape, cell, 1, 0.25);
};
EXPECT_EQ("native",
          make_fixture_runtime("native")->backend_name());
#ifndef ENABLE_CIDER_EXTERNAL
EXPECT_THROW(make_fixture_runtime("external"), std::runtime_error);
#else
EXPECT_EQ("external",
          make_fixture_runtime("external")->backend_name());
#endif
EXPECT_THROW(make_fixture_runtime("auto"), std::invalid_argument);
```

The factory accepts only already-resolved backend names; this prevents a second
default-resolution policy from appearing in estate code.

- [ ] **Step 2: Run the test and verify factory symbols are absent**

Run:

```bash
cmake --build /tmp/abacus-cider-task12-focused \
  --target MODULE_ESTATE_CIDER_pot_runtime_factory_test -j2
```

Expected: missing target/header/factory failure.

- [ ] **Step 3: Implement the factory**

Expose this signature:

```cpp
std::unique_ptr<PotCiderRuntime> make_pot_cider_runtime(
    const std::string& backend,
    const std::string& model_path,
    const std::size_t grid_shape[3],
    const double cell[9],
    std::size_t nspin,
    double cider_xmix);
```

Construct native unconditionally for `native`; compile the `external` branch
only under `ENABLE_CIDER_EXTERNAL`; otherwise throw an unavailable-capability
error. Reject `auto` and unknown strings because input conversion must already
have resolved them.

- [ ] **Step 4: Convert `PotCiderXC` to the common runtime**

Replace `std::unique_ptr<PotCiderNativeRuntime>` with
`std::unique_ptr<PotCiderRuntime>`. In the constructor call the factory with
`PARAM.inp.cider_backend`. Read only common metadata:

```cpp
const PotCiderRuntimeMetadata& metadata = runtime_->metadata();
is_mgga_ = metadata.requires_tau;
feature_exc_convention_ = metadata.feature_exc_convention;
```

Retain the existing explicit `cider_feature_density` conflict checks and all
evaluation, gather/scatter, reconstruction, diagnostics, clipping, energy, and
training behavior. Change backend-specific log text to include
`backend=<runtime_->backend_name()>`.

- [ ] **Step 5: Preserve external early MGGA allocation and native inspection**

In `input_conv.cpp`, use the native safe inspector only for native selection.
For external selection, restore the base revision's
`cider_model_requests_ked(model_path)` behavior so established external MGGA
files allocate `vofk` before bridge construction. Add an input conversion test
using misleading filenames to prove native selection remains metadata-driven
and an external MGGA filename still requests tau. At construction, the external
runtime's authoritative `is_mgga` must match the preallocated `vofk`; report a
fatal metadata/allocation mismatch rather than writing through a null buffer.

- [ ] **Step 6: Run native and dual factory/potential tests**

Run:

```bash
cmake --build /tmp/abacus-cider-task12-focused \
  --target MODULE_ESTATE_CIDER_pot_runtime_factory_test \
           MODULE_ESTATE_CIDER_pot_native_test abacus_pw_ser -j2
ctest --test-dir /tmp/abacus-cider-task12-focused --output-on-failure \
  -R 'MODULE_ESTATE_CIDER_pot_(runtime_factory|native)_test'
cmake --build /tmp/abacus-cider-dual \
  --target MODULE_ESTATE_CIDER_pot_runtime_factory_test \
           MODULE_ESTATE_CIDER_pot_external_test abacus_pw_ser -j2
ctest --test-dir /tmp/abacus-cider-dual --output-on-failure \
  -R 'MODULE_ESTATE_CIDER_pot_(runtime_factory|external)_test'
```

Expected: all tests and both serial executables build successfully.

- [ ] **Step 7: Commit runtime selection integration**

```bash
git add source/source_estate/module_pot/pot_cider_xc_runtime_factory.h \
  source/source_estate/module_pot/pot_cider_xc_runtime_factory.cpp \
  source/source_estate/module_pot/test/pot_cider_xc_runtime_factory_test.cpp \
  source/source_estate/module_pot/pot_cider_xc.h \
  source/source_estate/module_pot/pot_cider_xc.cpp \
  source/source_estate/CMakeLists.txt \
  source/source_io/module_parameter/input_conv.cpp
git commit -m "feat: select CIDER backend at runtime"
```

---

### Task 5: Add same-binary native/external parity coverage

**Files:**
- Create: `source/source_estate/module_pot/test/pot_cider_xc_dual_backend_test.cpp`
- Modify: `source/source_estate/CMakeLists.txt`
- Reuse read-only: `source/source_hamilt/module_xc/cider/test/fixtures/v1/manifest.json`
- Reuse read-only: `source/source_hamilt/module_xc/cider/test/fixtures/v1/arrays.bin`

**Interfaces:**
- Consumes: runtime factory, five-model frozen oracle, both compiled backends.
- Produces: deterministic fixed-density parity evidence from one dual test
  executable and explicit preservation of an external-only model.

- [ ] **Step 1: Write the dual-backend parity test**

For each of the five v1 cases, construct `native` and `external` runtimes with
the same model, grid, cell, spin, and `xmix`. Feed byte-identical baseline and
feature tuples. Compare every baseline/feature `exc`, `vrho`, `vsigma`, and
`vtau` element and record maxima:

```cpp
EXPECT_NEAR(native_value, external_value,
            2.0e-11 + 2.0e-10 * std::abs(external_value));
RecordProperty("dual_backend_max_absolute_error", max_absolute);
RecordProperty("dual_backend_max_relative_error", max_relative);
```

Also compare volume-integrated baseline/feature energies and potential-density
contractions at the existing v1 tolerances.

- [ ] **Step 2: Add the external-only compatibility test**

Load `/home/bonan/appdir/CiderPress/functionals/CIDER24Xe.yaml` with the
external runtime and assert construction succeeds. Constructing the native
runtime for the same file must fail with `UNSUPPORTED_EVALUATOR` and must not
fall back to external.

- [ ] **Step 3: Run the test and diagnose any parity mismatch systematically**

Run:

```bash
CIDER_NATIVE_CORPUS_ROOT=/home/bonan/appdir/CiderPress \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
ctest --test-dir /tmp/abacus-cider-dual --output-on-failure \
  -R MODULE_ESTATE_CIDER_dual_backend_test
```

Expected: all five role-separated cases and the external-only compatibility
case pass. If a numerical case fails, stop feature edits and use the
systematic-debugging workflow to identify array convention, density role,
exchange scaling, or derivative layout before changing tolerances.

- [ ] **Step 4: Verify deterministic oracle regeneration**

Run:

```bash
LD_LIBRARY_PATH=/home/bonan/appdir/CiderPress/ciderpress/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
/home/bonan/appdir/CiderPress/.venv/bin/python3 \
  source/source_hamilt/module_xc/cider/test/fixtures/v1/generate_oracle.py \
  --parent /home/bonan/appdir/CiderPress --check
```

Expected: `PASS: deterministic regeneration matches 2 generated files`.

- [ ] **Step 5: Commit the same-binary parity suite**

```bash
git add source/source_estate/module_pot/test/pot_cider_xc_dual_backend_test.cpp \
  source/source_estate/CMakeLists.txt
git commit -m "test: compare native and external CIDER backends"
```

---

### Task 6: Harden build and release guards for native-only and dual artifacts

**Files:**
- Modify: `cmake/tests/CheckNativeCiderBuildBoundary.cmake`
- Modify: `cmake/tests/CheckNativeCiderBuildBoundarySelfTest.cmake`
- Modify: `cmake/tests/CheckNativeCiderReleaseArtifact.cmake`
- Modify: `cmake/tests/CheckNativeCiderReleaseArtifactSelfTest.cmake`
- Create: `cmake/tests/CheckDualCiderReleaseArtifact.cmake`
- Create: `cmake/tests/CheckDualCiderReleaseArtifactSelfTest.cmake`
- Modify: `source/source_hamilt/module_xc/cider/test/CMakeLists.txt`

**Interfaces:**
- Consumes: CMake caches, link commands, built binaries, compile definitions.
- Produces: profile-aware, fail-closed source and release artifact audits.

- [ ] **Step 1: Add negative self-test fixtures for option leakage**

Extend the boundary self-test so Python/bridge tokens are accepted only inside
an `if(ENABLE_CIDER_EXTERNAL)` block and remain rejected in unconditional or
native-only `ENABLE_CIDER` blocks. Add release self-test cases where a
native-only cache or link command contains `Python3`, NumPy, CiderPress, or
`libciderbridge`; each must be rejected.

- [ ] **Step 2: Run the guards and verify the restored external code is rejected**

Run:

```bash
cmake -DABACUS_SOURCE_DIR=$PWD \
  -P cmake/tests/CheckNativeCiderBuildBoundary.cmake
cmake -DABACUS_SOURCE_DIR=$PWD \
  -DCIDER_BOUNDARY_SELF_TEST_DIR=/tmp/cider-dual-boundary-red \
  -P cmake/tests/CheckNativeCiderBuildBoundarySelfTest.cmake
```

Expected: at least the source boundary fails until it understands the external
option boundary.

- [ ] **Step 3: Make the source guard profile-aware**

Teach `CheckNativeCiderBuildBoundary.cmake` to permit external dependency tokens
only inside syntactically active `ENABLE_CIDER_EXTERNAL` blocks. It must still
reject those tokens in native CIDER targets/files and confirm that
`cider_native` never links the bridge.

- [ ] **Step 4: Add the dual release artifact guard**

`CheckDualCiderReleaseArtifact.cmake` requires:

```text
ENABLE_CIDER:BOOL=ON
ENABLE_CIDER_EXTERNAL:BOOL=ON
```

It rejects unresolved dependencies, verifies the link/runtime closure contains
the configured `libciderbridge` and Python library, rejects duplicate/conflicting
bridge libraries, and verifies native CIDER capability is present through the
build cache and build-info output. The existing native artifact guard must
require `ENABLE_CIDER_EXTERNAL:BOOL=OFF` or an unset false value before applying
its prohibited-dependency scan.

- [ ] **Step 5: Run source, self-test, and both artifact guards**

Run:

```bash
cmake -DABACUS_SOURCE_DIR=$PWD \
  -P cmake/tests/CheckNativeCiderBuildBoundary.cmake
cmake -DABACUS_SOURCE_DIR=$PWD \
  -DCIDER_BOUNDARY_SELF_TEST_DIR=/tmp/cider-dual-boundary-green \
  -P cmake/tests/CheckNativeCiderBuildBoundarySelfTest.cmake
cmake -DCIDER_CACHE=/tmp/abacus-cider-task17-serial/CMakeCache.txt \
  -DCIDER_LINK=/tmp/abacus-cider-task17-serial/CMakeFiles/abacus_pw_ser.dir/link.txt \
  -DCIDER_BINARY=/tmp/abacus-cider-task17-serial/abacus_pw_ser \
  -P cmake/tests/CheckNativeCiderReleaseArtifact.cmake
cmake -DCIDER_CACHE=/tmp/abacus-cider-dual/CMakeCache.txt \
  -DCIDER_LINK=/tmp/abacus-cider-dual/CMakeFiles/abacus_pw_ser.dir/link.txt \
  -DCIDER_BINARY=/tmp/abacus-cider-dual/abacus_pw_ser \
  -P cmake/tests/CheckDualCiderReleaseArtifact.cmake
```

Expected: all four commands pass and print profile-specific success messages.

- [ ] **Step 6: Commit the profile-aware guards**

```bash
git add cmake/tests/CheckNativeCiderBuildBoundary.cmake \
  cmake/tests/CheckNativeCiderBuildBoundarySelfTest.cmake \
  cmake/tests/CheckNativeCiderReleaseArtifact.cmake \
  cmake/tests/CheckNativeCiderReleaseArtifactSelfTest.cmake \
  cmake/tests/CheckDualCiderReleaseArtifact.cmake \
  cmake/tests/CheckDualCiderReleaseArtifactSelfTest.cmake \
  source/source_hamilt/module_xc/cider/test/CMakeLists.txt
git commit -m "test: audit native and dual CIDER artifacts"
```

---

### Task 7: Validate runtime switching in serial, MPI, and CUDA builds

**Files:**
- Verify only: copy the established read-only Si CIDER example from
  `/home/bonan/appdir/CiderPress/abacus-cider/examples/02_scf/pw_Si2-cider-gga-k12-oncv-nxyz60`
  into `/tmp/abacus-cider-dual-runtime`.
- Record measured results later in:
  `docs/developers_guide/cider_native_validation.md`.

**Interfaces:**
- Consumes: one dual executable, runtime input selector, same model/system.
- Produces: end-to-end evidence that one executable selects both backends.

- [ ] **Step 1: Prepare paired transient runtime inputs**

Copy the example to two `/tmp` directories outside both worktrees. The two
inputs must be byte-identical except:

```text
cider_backend native
```

versus:

```text
cider_backend external
```

Use one SCF iteration, one OpenMP thread, the same model and ONCV Si fixture,
and identical random seed/starting data where ABACUS exposes one. Capture
selected backend, first-iteration total energy, component energies, and finite
potential diagnostics. Keep these runtime copies under `/tmp`; do not add the
original worktree's untracked examples to this branch.

- [ ] **Step 2: Run the serial same-binary switch test**

Run the two cases with `/tmp/abacus-cider-dual/abacus_pw_ser`. Expected:
external is selected when `cider_backend` is omitted/auto; explicit native and
external logs identify the requested backend; both complete and satisfy the
fixed-density/component tolerances established in Task 5.

- [ ] **Step 3: Configure/build and run dual MPI outside the sandbox**

Configure `/tmp/abacus-cider-dual-mpi` with `ENABLE_MPI=ON`,
`ENABLE_CIDER_EXTERNAL=ON`, and the same dependency paths. Build
`abacus_pw_para`, then run each backend with `mpirun -n 1` and `mpirun -n 2`,
all with one OpenMP thread. Expected: rank counts agree within `1e-10 Ha` after
tight convergence or within the fixed-field tolerances for the one-iteration
debug dumps.

- [ ] **Step 4: Configure/build and run the dual CUDA target outside the sandbox**

Configure `/tmp/abacus-cider-dual-cuda` using the installed CUDA toolkit and
the architecture list required by the host. Build `abacus_pw_gpu`; run at least
the SL-GGA native and external inputs with `device gpu`. Expected: both use the
same GPU-enabled ABACUS binary while the selected CIDER backend executes on the
host and returns finite matching components.

- [ ] **Step 5: Verify native selection does not initialize the bridge**

Run the dual binary with `cider_backend native` and bridge debug logging enabled.
Assert the log contains `backend=native` and contains neither
`cider_bridge_create` nor Python initialization output. Run the external input
and assert the bridge construction log is present.

- [ ] **Step 6: Preserve runtime evidence for the documentation task**

Record the executable SHA, CMake capabilities, exact model SHA-256, selected
backend log line, component energies, field maxima, MPI rank counts, CUDA
device, and command lines in the worktree-local `.planning/sdd/` execution
report. Task 8 migrates the stable results into
`cider_native_validation.md`; no runtime fixture commit is made in this task.

---

### Task 8: Update build, input, architecture, and validation documentation

**Files:**
- Modify: `BUILD_CIDER.md`
- Modify: `docs/developers_guide/cider_build.md`
- Modify: `docs/developers_guide/cider_native_architecture.md`
- Modify: `docs/developers_guide/cider_native_validation.md`
- Modify: `docs/developers_guide/cider_model_authoring_and_troubleshooting.md`
- Modify: `docs/developers_guide/cider_legacy_yaml_compatibility.md`
- Modify: `source/source_io/module_parameter/read_input_item_ofdft.cpp`

**Interfaces:**
- Consumes: final option names, runtime selection table, test results, artifact
  dependency closures.
- Produces: user-facing instructions for native-only and dual builds and a
  durable same-binary parity record.

- [ ] **Step 1: Document both build profiles and runtime selection**

Add exact native-only and dual CMake commands from the approved specification.
Document `cider_backend=auto|external|native`, including conditional defaults,
external dependency requirements, and the lack of silent fallback.

- [ ] **Step 2: Document compatibility and development workflow**

State that external mode retains CiderPress-supported models, including models
outside the native whitelist, while explicit native selection reports native
schema errors. Explain that developers can run the same executable twice for
parity and that distribution builds can leave `ENABLE_CIDER_EXTERNAL=OFF`.

- [ ] **Step 3: Record measured validation evidence**

Record actual native/external maximum array and integrated errors, serial/MPI
results, CUDA status, selected backend logs, and dependency audits. Preserve the
existing warning that April combined-convention totals are not acceptance
targets.

- [ ] **Step 4: Run documentation and link checks**

Run available Sphinx/MyST checks or explicitly record that Sphinx is unavailable.
Use `rg` to confirm every CIDER document links to the correct build,
compatibility, architecture, and validation pages. Run `git diff --check`.

- [ ] **Step 5: Commit documentation**

```bash
git add BUILD_CIDER.md docs/developers_guide \
  source/source_io/module_parameter/read_input_item_ofdft.cpp
git commit -m "docs: describe dual CIDER backend builds"
```

---

### Task 9: Perform final branch-wide verification and review

**Files:**
- Verify only; modify implementation files only if a failing check identifies a
  real defect, using a focused fix commit.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: a reviewable dual-backend branch with fresh build, test, parity,
  dependency, and isolation evidence.

- [ ] **Step 1: Run all focused native and dual CIDER tests**

Run all tests matching `MODULE_(HAMILT|IO|ESTATE)_CIDER` in the native-only and
dual build trees with `CIDER_NATIVE_CORPUS_ROOT=/home/bonan/appdir/CiderPress`.
Expected: zero failures and zero corpus skips.

- [ ] **Step 2: Rebuild release executables**

Incrementally rebuild native-only serial, MPI, and CUDA targets and dual serial,
MPI, and CUDA targets. Run the native and dual artifact guards on every produced
binary.

- [ ] **Step 3: Rerun same-binary parity and isolated runtime tests**

Run the five-model fixed-density parity suite. Run native-only ABACUS with an
empty `PYTHONPATH`, no virtual environment, and no CiderPress-specific
`LD_LIBRARY_PATH`. Run the dual binary twice, selecting native and external,
and verify only external selection emits bridge/Python initialization output.

- [ ] **Step 4: Check formatting, tracked state, and original isolation**

Run:

```bash
git diff --check 55e0b4518..HEAD
git status --short
git -C /home/bonan/appdir/CiderPress/abacus-cider diff --quiet
git -C /home/bonan/appdir/CiderPress/abacus-cider diff --cached --quiet
```

Expected: no formatting errors; only intentionally untracked worktree-local
`.planning` records; original `abacus-cider` tracked and staged states clean.

- [ ] **Step 5: Request final code review**

Review the complete diff from `55e0b4518` to final head. Treat backend-selection
errors, accidental Python linkage in native-only builds, Python initialization
under native selection, convention/scaling mismatches, MPI divergence, and
external-model regressions as blocking.

- [ ] **Step 6: Prepare integration handoff**

Report final commits, build matrices, exact parity maxima, known exclusions,
and the isolated worktree path. Do not merge, push, squash, or modify the
original `abacus-cider` worktree without a separate explicit user instruction.
