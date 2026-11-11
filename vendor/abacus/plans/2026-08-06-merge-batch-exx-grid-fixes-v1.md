# Merge batch-exx-pr Symmetry-Compatible FFT-Grid Fixes

## Objective

Integrate the behavior of local `batch-exx-pr` commit `375c7e3ce` into `abacus-cider` without merging the divergent branch wholesale. The result must select a bounded symmetry-compatible private EXX FFT embedding before reduced-k allocation, fall back to the full-k mesh when the embedding is unsafe, preserve CIDER's PBE-based one-shot EXX labels and selected-band evaluations, and retain the charge-grid contract required by EXX energy-density output. Success means the focused policy tests, the existing PW EXX CPU/MPI/GPU suite, CIDER label regressions, and the ZnO reduced/full comparison all meet their stated criteria.

## Implementation Plan

- [x] 1. Create an isolated integration branch from current `abacus-cider`, record `375c7e3ce` as the behavioral source, and exclude unrelated untracked artifacts.

- [x] 2. Preserve the current EXX entry points at `source/source_esolver/esolver_fp.cpp:66`, `source/source_esolver/esolver_ks_pw.cpp:167`, and `source/source_pw/module_pwdft/op_pw_exx.cpp:88` as comparison anchors.

- [x] 3. Move overflow-safe size arithmetic from `source/source_pw/module_pwdft/kernels/exx_q_state_op.cpp:193` into a shared `source/source_pw/module_pwdft/exx_size_utils.h`.

- [x] 4. Port the fixed-scratch and tile-fit estimates needed by the grid gate into standalone tile-policy files while retaining CIDER fields from `source/source_pw/module_pwdft/op_pw_exx.h:98`.

- [x] 5. Preserve the `exx_auto_tiling = false` default at `source/source_io/module_parameter/input_parameter.h:697` and verify both explicit manual tiles and automatic bounded selection.

- [x] 6. Port the deterministic selector into `exx_fft_grid_policy.h` and `exx_fft_grid_policy.cpp`, including translation checks, smallest-first search, 16-trial limit, and diagnostics.

- [x] 7. Port `exx_fft_grid_policy_test.cpp` with coverage for ZnO-like selection, compatible controls, overflow, memory limits, trial limits, diagnostics, and G-vector preservation.

- [x] 8. Introduce a reduced-mesh acceptance hook around k-point initialization at `source/source_esolver/esolver_fp.cpp:66`, with the PW decision implemented near `source/source_esolver/esolver_ks_pw.cpp:200`.

- [x] 9. Define PW EXX demand from native hybrid execution plus `out_exx_label` and `out_bandgap_label`, because the PBE one-shot path at `source/source_esolver/esolver_ks_pw.cpp:772` constructs EXX later.

- [x] 10. Require full-k fallback for an incompatible `out_exx_energy_density` mesh so the charge-grid equality at `source/source_pw/module_pwdft/op_pw_exx.cpp:2272` remains intact.

- [x] 11. Apply the selected grid in direct and copied EXX operator constructors before basis parameterization and FFT planning at `source/source_pw/module_pwdft/op_pw_exx.cpp:147`.

- [x] 12. Replay the same deterministic grid choice and stress-specific memory gate before private stress transforms at `source/source_pw/module_pwdft/stress_exx.cpp:115`.

- [x] 13. Replace the duplicate predicate at `source/source_pw/module_pwdft/kernels/exx_q_state_op.cpp:140` with shared policy diagnostics while retaining CPU, GPU, and MPI remap behavior.

- [x] 14. Register production and test sources at `source/CMakeLists.txt:504` and `source/source_pw/module_pwdft/test/CMakeLists.txt:25` without removing `MODULE_PW_op_pw_exx_selected`.

- [x] 15. Extend regressions for hybrid SCF, PBE scalar and selected labels, energy density, compatible grids, memory fallback, stress gating, symmetry-off, nspin, and manual tiles.

- [x] 16. Run focused CTest, CIDER CPU, MPI/KPAR, CUDA, existing `tests/18_PW_EXX` cases, and ZnO reduced/full plus constrained-budget validation outside the sandbox with one OpenMP thread.

- [x] 17. Review the final diff against `375c7e3ce`, confirm CIDER and selected-target preservation, record numerical evidence, and leave execution-context cleanup as a separate follow-up.

## Verification Criteria

- The focused FFT-grid tests select 36x36x50 from the incompatible ZnO-like 32x32x50 grid, leave compatible grids unchanged, stop after at most 16 enlarged candidates, and report memory, trial, invalid-grid, and exhaustion failures deterministically.
- Enlarging the private EXX embedding preserves the requested physical cutoff and the exact G-vector set; the reference 240 Ry case retains 20,185 physical EXX plane waves.
- The reduced ZnO HSE/ACE run stores 12 of 64 k points, converges, and agrees with the independent full-k run within 0.1 meV/atom for total and EXX energy and within 1e-4 eV for the band gap.
- A deliberately constrained memory budget rebuilds the full-k mesh before wavefunction and k-dependent EXX allocation, with a diagnostic containing the initial grid, attempted grid, required bytes, budget bytes, and `ecutexx` guidance.
- PBE one-shot `out_exx_label` and selected-band requests participate in the pre-allocation decision even when native hybrid EXX is disabled.
- Symmetry-on `out_exx_energy_density` never reaches evaluation with a private EXX grid different from the charge grid; it either uses the unchanged compatible grid or falls back to full-k.
- Existing scalar EXX labels, energy-density labels, selected-band shifts, hybrid forces and stress, MPI/KPAR, symmetry-off, nspin 1 and 2, and CUDA smoke cases retain their reference results within existing tolerances.
- `exx_auto_tiling = false` remains the default and preserves explicit manual tiles; automatic mode remains memory-defensive and produces valid bounded tile sizes.
- CPU, MPI, and CUDA builds complete for CIDER-enabled PW targets, focused CTest targets report zero failures, and all runtime tests are executed outside the sandbox with one OpenMP thread per rank.
- The final diff retains `MODULE_PW_op_pw_exx_selected`, all CIDER runtime backends, one-shot EXX energy and density output, band-target logic, and existing diagnostics.

## Potential Risks and Mitigations

1. **Semilocal CIDER training runs bypass the source commit's hybrid-only detector.**
   - Impact: Reduced-k PBE label calculations could reach the operator with an incompatible grid or miss the intended full-k fallback.
   - Likelihood: High without adaptation.
   - Mitigation: Base the solver decision on all planned EXX consumers, including scalar and selected-band post-SCF requests, and cover them with focused tests.
   - Contingency: Disable symmetry reduction for affected one-shot requests until the expanded predicate and policy tests pass.

2. **An enlarged private EXX grid violates the energy-density charge-grid contract.**
   - Impact: `out_exx_energy_density` aborts or produces data on an unusable layout.
   - Likelihood: High for nonsymmorphic reduced meshes.
   - Mitigation: Force pre-allocation full-k fallback when energy density requires an incompatible private embedding.
   - Contingency: Keep energy-density remapping as a separate design rather than weakening the current equality check.

3. **Porting the newer memory policy changes established tiling behavior.**
   - Impact: Memory use, performance, or explicit manual tile semantics could regress.
   - Likelihood: Medium.
   - Mitigation: Preserve the current default, test manual and automatic modes separately, and port only the shared estimates and fit decisions required by the grid gate.
   - Contingency: Retain the existing inline selector behind the same interface while using the enhanced estimator only for reduced-mesh acceptance.

4. **Solver, operator, and stress repeat grid selection independently.**
   - Impact: A mismatch could accept one grid early and construct another later.
   - Likelihood: Low if inputs remain identical.
   - Mitigation: Use one pure deterministic policy, centralize operation extraction, and assert or log the selected dimensions at each boundary.
   - Contingency: Introduce an immutable selected-grid value in existing options only if deterministic replay cannot be proven.

5. **Whole-branch conflict resolution overwrites CIDER-specific code.**
   - Impact: Selected-target tests, one-shot labels, energy-density output, or CIDER backend support could disappear silently.
   - Likelihood: High for a direct merge or cherry-pick.
   - Mitigation: Manually port the behavioral changes and perform a preservation-oriented diff review against current HEAD.
   - Contingency: Split the work into smaller commits for shared utilities, grid policy, solver fallback, operator/stress wiring, and tests so any regression can be isolated.

6. **GPU, MPI, and stress layouts expose paths not covered by pure unit tests.**
   - Impact: Compilation succeeds while runtime mapping, ownership, or memory behavior fails.
   - Likelihood: Medium.
   - Mitigation: Run the existing CPU MPI/KPAR and CUDA suites plus the ZnO reduced/full and constrained-budget cases outside the sandbox.
   - Contingency: Gate unsupported combinations with explicit diagnostics while retaining full-k CPU behavior until the path is validated.

## Alternative Approaches

1. **Merge `batch-exx-pr` wholesale.** This would preserve branch history but import extensive upstream divergence and overwrite CIDER-specific EXX code and tests. It is not recommended.
2. **Cherry-pick only `375c7e3ce`.** This is superficially simple but the commit assumes standalone tile-policy and execution-context APIs absent from current HEAD, creating large semantic conflicts. It is not recommended.
3. **Port only the grid search and ignore memory-fit prerequisites.** This minimizes files changed but loses the source fix's pre-allocation defensive fallback and can select embeddings that exceed the intended budget. It is acceptable only as a temporary diagnostic prototype.
4. **Manual CIDER-aware behavioral port.** This adds the shared utility and policy pieces needed for correctness, adapts EXX-demand and energy-density rules, and preserves current interfaces. This is the recommended approach.
5. **Disable reduced-k EXX for every CIDER or training-label request.** This is safe and small but gives up the validated symmetry speedup, including the roughly fivefold ZnO result. It is a fallback, not the target design.

## Assumptions

- Local commit `375c7e3ce` is the authoritative version of the newly committed fix even though `origin/batch-exx-pr` still points to `73a8bd8a8`.
- The scope is the symmetry-compatible FFT-grid correctness fix and its minimal memory-policy prerequisites, not the entire `batch-exx-pr` history.
- The broad `73a8bd8a8` execution-context refactor and changing automatic tiling to default-on remain separate follow-up decisions.
- Full-k fallback is acceptable for energy-density output when the charge grid is incompatible; density-grid remapping is out of scope.
- Existing untracked validation artifacts may be copied to scratch for manual verification but are not implicitly added to the repository.

## Dependencies

- A CIDER-enabled ABACUS CPU build with LibXC and yaml-cpp, following `BUILD_CIDER.md:72`.
- An MPI runtime for KPAR and parallel integration tests, executed outside the sandbox with one OpenMP thread per rank.
- A CUDA-capable node and a CIDER-enabled `abacus_pw_gpu` build covering architectures 86, 89, and 120, following `BUILD_CIDER.md:96`.
- Access to the existing ZnO validation inputs and pseudopotentials for the manual reduced/full comparison, or an equivalent reproducible nonsymmorphic fixture.

## Notes

- Keep the implementation as a sequence of reviewable commits: shared utilities and tile estimates, FFT-grid policy and unit tests, solver fallback, operator and stress wiring, CIDER-specific regressions, then verification evidence.
- Do not stage or modify the user's existing untracked examples, test outputs, pseudopotentials, or validation artifacts during implementation.
