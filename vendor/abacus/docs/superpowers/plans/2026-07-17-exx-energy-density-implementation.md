# ABACUS EXX Energy Density Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export an opt-in conventional-Fock-gauge `exx_energy_density_g` whose charge-grid integral reproduces the existing post-SCF scalar full-range EXX training label.

**Architecture:** Extend the existing q-tiled scalar EXX traversal with an optional complex real-space grid accumulator. The scalar contraction remains authoritative and unchanged; when requested, the same pair density and cached Coulomb potential are inverse transformed and accumulated pairwise, then carried through the existing optional-label lifecycle into the HDF5 writer.

**Tech Stack:** C++17, ABACUS PW/FFT/device abstractions, MPI, CUDA, GoogleTest/CTest, HDF5, Python/h5py.

## Global Constraints

- Default `out_exx_energy_density = false` preserves scalar-only behavior and allocations.
- Output is real double `[nxyz]`, Ha/Bohr^3, spin summed, full range, unscaled, conventional Fock/exchange-hole gauge.
- CPU integral tolerance is `max(1e-8 Ha, 1e-8 * abs(E_scalar))`; never rescale the field.
- Require `out_training_data`, `out_exx_label`, PW, `nspin` 1/2, `kpar=1`, and `bndpar=1`.
- Reject EXX/charge FFT-grid mismatch.
- Preserve pair-map/full-active weights, singular correction, FFT normalization, and tiling.
- Every ABACUS runtime uses `OMP_NUM_THREADS=1` and runs outside the sandbox.
- Do not modify training/normalization code, datasets, unrelated files, or existing references for flag-off cases.
- Do not commit or push.

---

### Task 1: Input Flag and Validation

**Files:**
- Modify: `source/source_io/test_serial/read_input_item_test.cpp`
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_output.cpp`

**Interfaces:**
- Produces: `Input_para::out_exx_energy_density` (boolean, default false).
- Consumes: existing `out_training_data`, `out_exx_label`, `basis_type`, `nspin`, `kpar`, and `bndpar` values.

- [ ] Add a failing parser test block that obtains `out_exx_energy_density`, verifies the default is false, accepts the complete valid configuration, and death-tests each missing/unsupported prerequisite independently.

```cpp
auto it = find_label("out_exx_energy_density", readinput.input_lists);
ASSERT_NE(it, readinput.input_lists.end());
EXPECT_FALSE(param.input.out_exx_energy_density);
param.input.out_training_data = true;
param.input.out_exx_label = true;
param.input.basis_type = "pw";
param.input.nspin = 1;
param.input.kpar = 1;
param.input.bndpar = 1;
param.input.out_exx_energy_density = true;
EXPECT_NO_FATAL_FAILURE(it->second.check_value(it->second, param));
```

- [ ] Configure/build `MODULE_IO_read_item_serial` in a fresh test build and verify the new test fails because the input item is absent.
- [ ] Add `bool out_exx_energy_density = false;` adjacent to `out_exx_label`.
- [ ] Register an `Input_Item("out_exx_energy_density")` after `out_exx_label`, call `read_sync_bool`, and implement precise `WARNING_QUIT` checks for every prerequisite.
- [ ] Rebuild and run `MODULE_IO_read_item_serial`; verify all parser tests pass.

### Task 2: Complex Pair-Field Accumulation Kernel

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/exx_cal_energy_op.h`
- Modify: `source/source_pw/module_pwdft/kernels/exx_cal_energy_op.cpp`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/exx_cal_energy_op.cu`
- Modify: `source/source_pw/module_pwdft/kernels/test/exx_q_state_op_test.cpp`

**Interfaces:**
- Produces:

```cpp
template <typename T, typename Device>
struct exx_energy_density_accumulate_op
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const T* psi_n,
                    const T* psi_m,
                    const T* pair_potential,
                    Real weighted_half_over_omega,
                    T* energy_density,
                    int nrxx);
    void operator_batch(const T* psi_n,
                        const T* psi_m_batch,
                        const T* pair_potential_batch,
                        const Real* pair_weights,
                        Real half_over_omega,
                        T* energy_density,
                        int nrxx,
                        int batch_size);
};
```

- [ ] Add a failing CPU unit test using complex `psi_n`, two complex `psi_m`/potential contributions, and conjugate-related weights. Assert the final complex grid equals a hand calculation, its integrated real part equals the matching pair scalar, and the final imaginary residual cancels while individual terms have nonzero imaginary components.
- [ ] Run `MODULE_PW_Hamilt_Kernels_UTs` and verify the failure is an undefined `exx_energy_density_accumulate_op`.
- [ ] Implement CPU scalar and batch accumulation as one grid loop computing `conj(psi_n[g]) * psi_m[g] * pair_potential[g] * weighted_half_over_omega`; batch loops over active sources inside each grid element so no atomic operation is needed.
- [ ] Implement the equivalent CUDA kernels with one thread per grid point and source-batch accumulation inside the thread. Instantiate float/double CPU and GPU templates consistently with existing EXX kernels.
- [ ] Rebuild/run the kernel unit target and verify the complex identity/cancellation test passes.

### Task 3: Operator Result and Single-Traversal Accumulation

**Files:**
- Modify: `source/source_pw/module_pwdft/op_pw_exx.h`
- Modify: `source/source_pw/module_pwdft/op_pw_exx.cpp`
- Modify: `source/source_esolver/esolver_ks_pw.cpp`

**Interfaces:**
- Produces:

```cpp
struct ExxEnergyResult
{
    double energy_ry = 0.0;
    std::vector<double> energy_density_ry_bohr3;
    double max_imaginary_abs_ry_bohr3 = 0.0;
    std::string k_policy;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nrxx = 0;
};

ExxEnergyResult cal_exx_energy_with_density(psi::Psi<T, Device>* psi_) const;
```

- Consumes: Task 2 accumulation kernel and the existing scalar `process_qtile_energy_tile` traversal.

- [ ] Add compile-facing tests/usages first: update `TrainingExxLabel` to carry the optional field/diagnostics and call the wished-for result API when the new flag is set. Build and verify failure because the API is absent.
- [ ] Refactor `cal_exx_energy_op_qtile` into an internal overload accepting an optional result/accumulator request. Keep `cal_exx_energy()` routed through the null/disabled path.
- [ ] At density-enabled entry, verify `rhopw_dev` and the original charge `rhopw` have identical `nx`, `ny`, `nz`, `nxyz`, `nrxx`, and compatible distribution; fail with both mesh descriptions otherwise.
- [ ] Allocate and zero one device `T[nrxx]` accumulator only for the density-enabled call.
- [ ] In scalar pair mode, retain `exx_cal_energy_op` before mutating `density_recip`; then multiply by the cached potential, inverse FFT, and call Task 2 with `0.5 * k_weight * q_weight / omega`.
- [ ] In GPU batch mode, retain `exx_density_potential_mul_op`; then apply `mul_potential_op::operator_batch`, inverse-transform the batch once, and call the batch accumulator with the exact already-scaled device weights.
- [ ] After the traversal, leave scalar `0.5 * omega` and its MPI reduction unchanged. Copy the local complex grid to host once, compute real values and local max imaginary only after all pairs, and reduce the max diagnostic across the pool/world as required.
- [ ] Convert the operator payload into `TrainingExxLabel`: scalar remains Ry until the existing handoff; density remains Ry/Bohr^3 until one explicit Ha conversion in the training-label layer.
- [ ] Build CPU PW and CUDA PW targets to catch template/device errors before writer changes.

### Task 4: Optional Label Lifecycle and HDF5 Metadata

**Files:**
- Modify: `source/source_io/module_training/training_dump.h`
- Modify: `source/source_io/module_training/training_dump.cpp`
- Modify: `source/source_esolver/esolver_ks_pw.cpp`

**Interfaces:**
- Produces:

```cpp
struct TrainingExxEnergyDensity
{
    bool present = false;
    std::vector<double> values_local_Ha_bohr3;
    double integrated_energy_Ha = 0.0;
    double scalar_energy_Ha = 0.0;
    double integration_abs_residual_Ha = 0.0;
    double integration_rel_residual = 0.0;
    double max_imaginary_abs_Ha_bohr3 = 0.0;
    std::string k_policy;
};

void set_last_training_exx_energy_density(const TrainingExxEnergyDensity&);
void clear_last_training_exx_energy_density();
```

- [ ] Add writer/lifecycle expectations first: requesting output with no present payload must fail; flag-off ignores/omits payload; a payload with wrong local size must fail. Use the closest available unit boundary or a focused executable test if `training_dump.cpp` lacks a direct unit target.
- [ ] Add the structure and explicit last-label clear/set lifecycle. Clear at the beginning of every PW `after_scf`, set only after successful requested evaluation, and clear after the output phase.
- [ ] Convert the local field from Ry to Ha exactly once (`0.5`), integrate it with `dV`, calculate absolute/relative residuals against the scalar, and reject residuals beyond the CPU tolerance without rescaling.
- [ ] In `write_training_dump`, require a present payload when requested, gather it through `gather_spin_major_to_global(..., 1)`, validate `[nxyz]`, and append `TrainingArray{"exx_energy_density_g", {nxyz}, values}`.
- [ ] Add the optional `exx_label.energy_density` object with the approved keys and add `exx_energy_density_g` to the arrays map only when present. Keep schema version 2.
- [ ] Rebuild CPU PW and focused unit targets; verify flag-off code paths compile and remain unchanged.

### Task 5: HDF5 Validator and Opt-In Integration Cases

**Files:**
- Create: `tests/integrate/tools/validate_exx_energy_density_h5.py`
- Create/modify: focused cases under `tests/18_PW_EXX/`
- Preserve: existing `PW_PBE_training_exx_label_auto*` inputs/references.

**Interfaces:**
- Validator arguments: HDF5 path, expected-present boolean, optional tolerance.
- Validator output: scalar Ha, integrated Ha, absolute/relative residual, max imaginary residual, shape, dtype, and k policy.

- [ ] Write the validator before generating output. It reads `/record_json`, `/arrays/exx_energy_density_g`, checks presence/absence, `float64`, shape `(nxyz,)`, finiteness, metadata, negativity sanity, and independently computes `sum(field) * dV_bohr3`.
- [ ] Run it against an existing flag-off scalar HDF5 dump and verify absence passes; run with expected-present and verify it fails.
- [ ] Add a small Gamma `nspin=1` opt-in case derived from the scalar label case with `out_exx_energy_density 1` and explicit modest tiles.
- [ ] Add tractable `nspin=2`, multi-k, and symmetry-on opt-in cases by reusing existing PW EXX structures/pseudopotentials; keep reference outputs separate from flag-off cases.
- [ ] Add two tile configurations for one physical input and run pair-map/full-active diagnostics to compare both scalar and full grid arrays.
- [ ] Run every case outside the sandbox with `OMP_NUM_THREADS=1`, then run the HDF5 validator directly on each generated `step_0.h5`.
- [ ] Record scalar/integrated values and tiling/policy comparisons in `.planning/2026-07-17-exx-energy-density/progress.md`.

### Task 6: Full Verification and Compatibility Audit

**Files:**
- Modify if required by observed results: `BUILD_CIDER.md` only if build/runtime requirements changed.
- Update: `.planning/2026-07-17-exx-energy-density/{task_plan.md,findings.md,progress.md}`.

**Interfaces:**
- Consumes all prior tasks; produces verification evidence only.

- [ ] Configure a fresh CPU/MPI PW build with `BUILD_TESTING=ON`, `ENABLE_HDF5=ON`, `ENABLE_LIBXC=ON`, and the repository-required CIDER/Python paths. Inspect and record relevant `CMakeCache.txt` entries and target names.
- [ ] Run focused parser/kernel unit tests, then relevant broader CTest groups.
- [ ] Build `abacus_pw_para` and run all new/existing scalar EXX integration cases outside the sandbox with `OMP_NUM_THREADS=1`.
- [ ] Inspect each HDF5 directly with the validator and record exact scalar/integrated values.
- [ ] Configure a separate portable CUDA build with `CMAKE_CUDA_ARCHITECTURES="86;89;120"`, build `abacus_pw_gpu`, and record compiler/profile evidence.
- [ ] If a compatible GPU is visible, run one opt-in numerical case outside the sandbox and derive CPU/GPU tolerances from observed differences. Otherwise report compilation-only CUDA evidence and the unavailable runtime reason.
- [ ] Compare flag-off scalar results to existing references and confirm no new array/extra EXX traversal.
- [ ] Run `git diff --check`, `git diff --stat`, and a scoped `git status --short`; distinguish all pre-existing untracked user files from task changes.
- [ ] Re-read the approved design and acceptance criteria line by line, update all plan phases, and report remaining gaps honestly.
