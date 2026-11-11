# PW EXX Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the reviewed PW EXX KPAR, label-validation, auto-tiling, allocation-safety, diagnostics, and regression-coverage issues without changing the validated reduced-k numerical formulation.

**Architecture:** Correct ownership at the k-map source, reject unsupported label parallelism during input validation, and extend existing policy helpers for memory and diagnostics. Keep the ordered qk energy loop and real-space symmetry kernels unchanged. Each behavior is introduced through a failing focused test and committed in one of two coherent phases.

**Tech Stack:** C++17, MPI/OpenMPI, CUDA, CMake/GoogleTest, ABACUS shell integration harness, LibXC HSE/PBE.

## Global Constraints

- Preserve the ordered qk map, full-point occupation scaling, ACE construction, real-space symmetry rotation, and q-tile cache lifetime.
- `out_exx_label` and `out_bandgap_label` require `kpar = 1`; do not add KPAR label support.
- Preserve explicit unsupported ROCm behavior.
- Run all ABACUS, MPI, and GPU tests with `OMP_NUM_THREADS=1`; MPI/GPU execution must run outside the sandbox.
- Do not touch pre-existing unrelated tracked or untracked files.
- Use `apply_patch` for source edits and commit correctness separately from robustness/performance.

---

### Task 1: Correct global EXX identity-map weights under KPAR

**Files:**
- Modify: `source/source_cell/test/klist_test_para.cpp`
- Modify: `source/source_cell/klist.cpp:181-207`
- Test: `source/source_cell/test/klist_test_para.cpp`

**Interfaces:**
- Consumes: `Parallel_Kpoints::{nkstot_np,nks_np,startk_pool,my_pool,rank_in_pool}` and local `K_Vectors::wk`.
- Produces: identical complete `exx_full_q_map` and `exx_full_k_map` weights on every MPI rank.

- [ ] **Step 1: Add a failing four-rank identity-map test**

Add an MPI test to `klist_test_para.cpp` that initializes two k pools over eight full k-points, assigns four local weights per pool, calls `build_exx_identity_full_q_map()`, and checks all eight no-spin weights:

```cpp
TEST_F(KlistParaTest, ExxIdentityWeightsCoverEveryKPool)
{
    if (GlobalV::NPROC != 4)
    {
        GTEST_SKIP() << "requires the existing four-rank k-list test launcher";
    }
    int nkstot = 8;
    GlobalV::KPAR = 2;
    Parallel_Global::init_pools(GlobalV::NPROC,
                                GlobalV::MY_RANK,
                                1,
                                GlobalV::KPAR,
                                GlobalV::NPROC_IN_BNDGROUP,
                                GlobalV::RANK_IN_BPGROUP,
                                GlobalV::MY_BNDGROUP,
                                GlobalV::NPROC_IN_POOL,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::MY_POOL);
    kv->nspin = 1;
    kv->para_k.kinfo(nkstot,
                     GlobalV::KPAR,
                     GlobalV::MY_POOL,
                     GlobalV::RANK_IN_POOL,
                     GlobalV::NPROC,
                     1);
    kv->set_nkstot(nkstot);
    kv->set_nks(kv->para_k.nks_np);
    kv->kvec_d.resize(kv->para_k.nks_np);
    kv->kvec_c.resize(kv->para_k.nks_np);
    kv->wk.assign(kv->para_k.nks_np, 0.25);
    for (int ik = 0; ik < kv->para_k.nks_np; ++ik)
    {
        const int iglobal = kv->para_k.startk_pool[kv->para_k.my_pool] + ik;
        kv->kvec_d[ik] = ModuleBase::Vector3<double>(0.125 * iglobal, 0.0, 0.0);
        kv->kvec_c[ik] = kv->kvec_d[ik];
    }

    kv->build_exx_identity_full_q_map();

    ASSERT_EQ(kv->exx_full_q_map.size(), 8);
    double weight_sum = 0.0;
    for (const auto& point: kv->exx_full_q_map)
    {
        EXPECT_DOUBLE_EQ(point.weight, 0.125);
        weight_sum += point.weight;
    }
    EXPECT_DOUBLE_EQ(weight_sum, 1.0);
}
```

- [ ] **Step 2: Build and run the RED test**

Run outside the sandbox:

```bash
cmake --build build-merge-test --target MODULE_CELL_klist_test_para1 -j1
cd build-merge-test/source/source_cell/test
OMP_NUM_THREADS=1 mpirun -np 4 ./MODULE_CELL_klist_test_para1 \
  --gtest_filter=KlistParaTest.ExxIdentityWeightsCoverEveryKPool
```

Expected: FAIL because entries owned by pool 1 have weight `0` instead of `0.125`.

- [ ] **Step 3: Apply the minimal global-bound fix**

In `build_exx_identity_full_q_map()`, retain the local `ik` bounds check but validate `global_ik` against `nk_no_spin`:

```cpp
if (global_ik >= 0 && global_ik < nk_no_spin && ik < static_cast<int>(this->wk.size()))
{
    wk_global[global_ik] = this->wk[ik] * spin_weight_scale;
}
```

- [ ] **Step 4: Run the GREEN test and complete k-list suites**

```bash
cd build-merge-test/source/source_cell/test
OMP_NUM_THREADS=1 mpirun -np 4 ./MODULE_CELL_klist_test_para1 \
  --gtest_filter=KlistParaTest.ExxIdentityWeightsCoverEveryKPool
OMP_NUM_THREADS=1 ./MODULE_CELL_klist_test
OMP_NUM_THREADS=1 bash ./klist_test_para.sh
```

Expected: new test passes on all ranks; serial and parallel k-list suites report zero failures.

---

### Task 2: Guard EXX labels and make label regression numerical

**Files:**
- Modify: `source/source_io/module_parameter/read_input_item_output.cpp:361-397`
- Modify: `source/source_io/test_serial/read_input_item_test.cpp:450-480`
- Modify: `source/source_esolver/esolver_ks_pw.cpp:837-877`
- Modify: `tests/integrate/tools/catch_properties.sh`
- Modify: `tests/18_PW_EXX/PW_PBE_training_exx_label_auto/assert_log_contains`
- Modify: `tests/18_PW_EXX/PW_PBE_training_exx_label_auto/result.ref`
- Test: `source/source_io/test_serial/read_input_item_test.cpp`
- Test: `tests/18_PW_EXX/PW_PBE_training_exx_label_auto`

**Interfaces:**
- Consumes: `Parameter::input.{out_exx_label,out_bandgap_label,kpar}`.
- Produces: early input rejection for label KPAR, high-precision `training_exx_label` log value, and `exxlabelref` integration result.

- [ ] **Step 1: Add RED input-validation tests**

Extend the existing output-item tests with:

```cpp
param.input.out_exx_label = true;
param.input.kpar = 2;
EXPECT_EXIT(item.check_value(item, param),
            testing::ExitedWithCode(1),
            "out_exx_label currently requires kpar = 1");
```

Also verify `out_bandgap_label=true`, `out_training_data=true`, and
`out_exx_label=true` rejects `kpar=2` through the same rule.

- [ ] **Step 2: Run the RED validation tests**

```bash
cmake --build build-merge-test --target read_input_item_test -j1
OMP_NUM_THREADS=1 ./build-merge-test/source/source_io/test_serial/read_input_item_test \
  --gtest_filter='*ExxLabel*:*BandgapLabel*'
```

Expected: FAIL because `out_exx_label` currently has no `check_value` callback.

- [ ] **Step 3: Add the KPAR=1 input guard**

Set `out_exx_label.check_value` to:

```cpp
item.check_value = [](const Input_Item&, const Parameter& para) {
    if (para.input.out_exx_label && para.input.kpar != 1)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "out_exx_label currently requires kpar = 1");
    }
};
```

Keep the existing band-gap prerequisite checks; because band-gap labels require
`out_exx_label`, the same KPAR guard applies without duplicate logic.

- [ ] **Step 4: Add stable numeric EXX-label output and extraction**

In `after_scf()`, preserve and restore the stream precision while emitting the
label:

```cpp
const std::streamsize saved_precision = GlobalV::ofs_running.precision();
GlobalV::ofs_running << "training_exx_label: full EXX energy = "
                     << std::setprecision(16) << 0.5 * this->pelec->f_en.exx << " Ha\n";
GlobalV::ofs_running.precision(saved_precision);
```

In `catch_properties.sh`, add:

```bash
out_exx_label=$(get_input_key_value "out_exx_label" "INPUT")
if [ "$out_exx_label" == "1" ]; then
    exx_label=$(grep "training_exx_label: full EXX energy" "$running_path" | tail -1 | awk '{print $(NF-1)}')
    echo "exxlabelref $exx_label" >>$1
fi
```

Update the stale assertion to:

```text
training_exx_label: requested one-shot full EXX tiles:
```

- [ ] **Step 5: Verify the integration test fails numerically before updating its reference**

Run the existing case outside the sandbox with the current CPU MPI binary and
`OMP_NUM_THREADS=1`. Generate `result.out` through `Single.sh` or
`catch_properties.sh` and confirm it contains `exxlabelref` while the old
`result.ref` comparison fails because that key is absent.

- [ ] **Step 6: Add the validated full-k label reference and rerun GREEN**

Use the symmetry-off calculation from this test case as the reference source;
append its high-precision `exxlabelref` value to `result.ref`. Do not derive it
from a reduced-k or KPAR calculation.

Run:

```bash
cd tests/18_PW_EXX/PW_PBE_training_exx_label_auto
OMP_NUM_THREADS=1 bash ../../integrate/Single.sh
```

Expected: the PBE total and `exxlabelref` both pass, and every log assertion is found.

---

### Task 3: Add matched KPAR and symmetry integration coverage

**Files:**
- Create: `tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1/INPUT`
- Create: `tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1/KPT`
- Create: `tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1/STRU`
- Create: `tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1/result.ref`
- Modify: `tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR/result.ref`
- Create: `tests/18_PW_EXX/PW_HSE_ACE_MgO_primitive_symm_off_stress/{INPUT,KPT,STRU,result.ref}`
- Modify: `tests/18_PW_EXX/CASES_CPU.txt`

**Interfaces:**
- Consumes: existing `PW_HSE_ACE_MPI_KPAR` and `PW_HSE_ACE_MgO_primitive_symm_on_stress` fixtures.
- Produces: paired integration cases whose references come from equivalent KPAR=1/full-k calculations.

- [ ] **Step 1: Create matched case copies**

Create fresh directories and copy only input fixtures and references:

```bash
mkdir -p tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1
cp tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR/{INPUT,KPT,STRU,result.ref} \
  tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1/
```

Set `kpar 1` and use a distinct suffix. Likewise create a fresh MgO directory,
copy only `INPUT`, `KPT`, `STRU`, and `result.ref` from the symmetry-on stress
case, change only `suffix` and `symmetry -1`, and preserve every other
physical/control setting.

- [ ] **Step 2: Demonstrate KPAR RED against the KPAR=1 reference**

Run the KPAR=1 case first and save its `etotref`. Put that same reference in
both KPAR case `result.ref` files, then run the unfixed KPAR=2 case and confirm
its energy comparison fails.

- [ ] **Step 3: Run both KPAR cases after Task 1 GREEN**

```bash
cd tests/18_PW_EXX/PW_HSE_ACE_MPI_KPAR1
OMP_NUM_THREADS=1 bash ../../integrate/Single.sh
cd ../PW_HSE_ACE_MPI_KPAR
OMP_NUM_THREADS=1 bash ../../integrate/Single.sh
```

Expected: both cases pass the shared reference within the existing threshold.

- [ ] **Step 4: Run matched symmetry cases**

Run the symmetry-off and symmetry-on MgO cases. Use the converged full-k result
as the common physical reference and confirm both pass the same threshold. Add
the two new case names to `CASES_CPU.txt` in the EXX group.

- [ ] **Step 5: Commit correctness phase**

```bash
git add source/source_cell/klist.cpp \
  source/source_cell/test/klist_test_para.cpp \
  source/source_io/module_parameter/read_input_item_output.cpp \
  source/source_io/test_serial/read_input_item_test.cpp \
  source/source_esolver/esolver_ks_pw.cpp \
  tests/integrate/tools/catch_properties.sh \
  tests/18_PW_EXX
git commit -m "Fix PW EXX KPAR mapping and label validation"
```

---

### Task 4: Make auto-tiling account for full-k stars

**Files:**
- Modify: `source/source_pw/module_pwdft/op_pw_exx.h:63-286`
- Modify: `source/source_pw/module_pwdft/op_pw_exx.cpp:170-215,1960-2010`
- Modify: `source/source_pw/module_pwdft/test/radial_proj_test.cpp:20-180`

**Interfaces:**
- Consumes: `ExxTilePolicyInput::{active_q_count,npw,requested_q_tile_size}` and `K_Vectors::exx_full_k_map`.
- Produces: `ExxTilePolicyInput::max_star_size` and a cache estimate covering both operator and exact-energy paths.

- [ ] **Step 1: Add a RED star-cache estimate test**

Add a policy test with `active_q_count=8`, `max_star_size=4`,
`requested_q_tile_size=8`, and a budget that fits 8 potentials but not 32. The
test must expect auto-tiling to reduce q tile size or reject the oversized
candidate. It fails before `max_star_size` participates in the estimate.

- [ ] **Step 2: Add the star-size policy field and helper**

Add:

```cpp
std::size_t max_star_size = 1;
```

Compute it by counting active `exx_full_k_map` entries grouped by
`(rep_pool, rep_local_index)`. Keep a minimum of one for full-k maps.

- [ ] **Step 3: Correct the potential-cache estimate**

For each candidate q tile, calculate:

```cpp
std::size_t direct_entries = 0;
if (!checked_exx_size_product(input.max_star_size, q, direct_entries))
{
    ModuleBase::WARNING_QUIT("OperatorEXXPW", "EXX potential-cache size overflow");
}
const std::size_t potential_entries = std::max(static_cast<std::size_t>(input.active_q_count),
                                               direct_entries);
bytes += static_cast<double>(potential_entries) * npw * real_bytes;
```

Use the same populated `max_star_size` in normal and copied EXX operators.

- [ ] **Step 4: Run policy GREEN and focused suite**

```bash
cmake --build build-merge-test --target MODULE_PW_radial_proj_test -j1
OMP_NUM_THREADS=1 ./build-merge-test/source/source_pw/module_pwdft/test/MODULE_PW_radial_proj_test
```

Expected: all policy tests pass and the star-cache case selects a budget-safe tile.

---

### Task 5: Check workspace products and gate diagnostics

**Files:**
- Modify: `source/source_pw/module_pwdft/op_pw_exx.h`
- Modify: `source/source_pw/module_pwdft/op_pw_exx.cpp:44-60,1015-1040,1370-1450,2168-2190`
- Modify: `source/source_pw/module_pwdft/test/radial_proj_test.cpp`

**Interfaces:**
- Consumes: existing `checked_exx_size_product(std::size_t,std::size_t,std::size_t&)`.
- Produces: checked workspace-size helper and default-off `ABACUS_EXX_ENERGY_DIAGNOSTICS` policy.

- [ ] **Step 1: Add RED checked-workspace tests**

Expose a small policy helper that folds an initializer list through
`checked_exx_size_product` and returns false on overflow:

```cpp
bool checked_exx_size_product(std::initializer_list<std::size_t> factors,
                              std::size_t& result);
```

Test `{8, 4, 64000}` succeeds and
`{std::numeric_limits<std::size_t>::max(), 2, 2}` fails.

- [ ] **Step 2: Add RED diagnostics-policy tests**

Add pure helper tests requiring:

```text
nullptr, "", "0" -> supported and disabled
"1"              -> supported and enabled
"yes", "2"       -> unsupported
```

- [ ] **Step 3: Implement checked workspace construction**

Use the initializer-list helper for FFT dimensions, `target_size`, `q_size`,
and `q_weight_count`. On failure call `WARNING_QUIT` with the relevant
workspace name before allocation.

- [ ] **Step 4: Gate exact-energy diagnostics**

Parse `ABACUS_EXX_ENERGY_DIAGNOSTICS` once in the operator constructor. Reject
unsupported values. Store the enabled state on the operator and make
`log_exx_energy_memory()` and `log_exx_energy_progress()` return immediately
when disabled. Avoid CUDA-memory queries and `std::cout.flush()` in default
execution.

- [ ] **Step 5: Remove dead conflict-resolution code**

Delete the anonymous-namespace `consecutive_integers()` helper; verify no call
site remains with:

```bash
rg 'consecutive_integers' source/source_pw/module_pwdft
```

Expected: no matches.

- [ ] **Step 6: Run focused robustness suites**

```bash
cmake --build build-merge-test --target MODULE_PW_radial_proj_test MODULE_PW_pw_test -j1
OMP_NUM_THREADS=1 ./build-merge-test/source/source_pw/module_pwdft/test/MODULE_PW_radial_proj_test
OMP_NUM_THREADS=1 ./build-merge-test/source/source_basis/module_pw/test/MODULE_PW_pw_test \
  --gtest_filter='*Exx*'
```

Expected: all EXX policy and CPU symmetry tests pass.

- [ ] **Step 7: Commit robustness phase**

```bash
git add source/source_pw/module_pwdft/op_pw_exx.h \
  source/source_pw/module_pwdft/op_pw_exx.cpp \
  source/source_pw/module_pwdft/test/radial_proj_test.cpp
git commit -m "Harden PW EXX tiling and diagnostics"
```

---

### Task 6: Final CPU, MPI, CUDA, and GaAs verification

**Files:**
- Verify only; update `docs/developers_guide/pw_exx_reduced_k_implementation.md` only if behavior descriptions changed.

**Interfaces:**
- Consumes: all preceding fixes and tests.
- Produces: fresh evidence that the complete implementation meets the design success criteria.

- [ ] **Step 1: Rebuild production and focused targets**

```bash
cmake --build build-merge-test --target abacus_pw_para MODULE_CELL_klist_test \
  MODULE_CELL_klist_test_para1 MODULE_PW_radial_proj_test MODULE_PW_pw_test -j1
cmake --build build-cuda-pw-gpu-cider --target abacus_pw_gpu pw_test_gpu -j1
```

Expected: both builds exit zero.

- [ ] **Step 2: Run focused CPU/MPI/CUDA suites**

Run outside the sandbox with `OMP_NUM_THREADS=1`:

```bash
./build-merge-test/source/source_cell/test/MODULE_CELL_klist_test
cd build-merge-test/source/source_cell/test && bash ./klist_test_para.sh
./build-merge-test/source/source_pw/module_pwdft/test/MODULE_PW_radial_proj_test
./build-merge-test/source/source_basis/module_pw/test/MODULE_PW_pw_test --gtest_filter='*Exx*'
./build-cuda-pw-gpu-cider/source/source_basis/module_pw/test_gpu/pw_test_gpu \
  --gtest_filter='ExxRealspaceSymmetryGpuTest.*'
```

Expected: zero focused failures.

- [ ] **Step 3: Run EXX integration cases**

Run the PBE label, KPAR=1, KPAR=2, MgO symmetry-off, and MgO symmetry-on cases
through `Single.sh`. Confirm every case reports pass and the label result
contains `exxlabelref`.

- [ ] **Step 4: Rerun GaAs GPU symmetry verification**

Reuse the committed GaAs `2x2x2`, 60 Ry inputs with fresh suffixes and the
rebuilt commit. Require:

```text
|Delta E_total(symmetry 1 - symmetry -1)| < 1e-5 eV
|Delta E_exx(symmetry 1 - symmetry -1)|   < 1e-5 eV
```

Also rerun PBE `out_exx_label` symmetry on/off and require pair-map versus
full-active agreement below `1e-12 Ha` on the same reduced PBE state.

- [ ] **Step 5: Final hygiene and review**

```bash
git diff --check
git status --short
```

Confirm no raw OUT directories, copied pseudopotentials, build products, or
pre-existing unrelated files are staged. Request an independent final code
review before committing any remaining documentation-only adjustment.
