# PW EXX Review Remediation Design

## Objective

Resolve the actionable findings from the parallel review of upstream commit
`7e24ed1be` and its CIDER integration `0b7dfb0ab`, without changing the
validated reduced-k EXX numerical formulation.

## Scope

The work is split into two independently testable phases.

### 1. Correctness and regression coverage

- Correct `K_Vectors::build_exx_identity_full_q_map()` so a global k-point
  index is checked against the global no-spin k-point count, not the current
  pool's local count.
- Add a regression that distributes a full mesh over multiple pools and proves
  every owning pool contributes the correct global weight.
- Compare a small PW HSE/ACE `KPAR=2` calculation against the equivalent
  `KPAR=1` calculation. The two cases must share physical settings and expected
  EXX/total energies; references must not be regenerated from the buggy KPAR=2
  result.
- Guard `out_exx_label` and `out_bandgap_label` at input validation when
  `kpar != 1`. Scalar label KPAR support is intentionally deferred.
- Update the stale PBE EXX-label log assertion to match the current full-range
  label terminology.
- Extend integration-property extraction so the generated EXX-label energy is
  checked numerically rather than only checking the converged PBE total energy.
- Add an executable symmetry-on/off reduced-k EXX regression using matched
  inputs. Existing manual GaAs validation remains supporting evidence, not the
  sole regression.

### 2. Robustness and performance

- Extend EXX auto-tiling input with the maximum active full-k star size.
- Estimate potential-cache entries as the larger of:
  - operator reuse cache: `active_q_count`; and
  - direct reduced-energy cache: `max_star_size * q_tile_size`.
- Use checked `size_t` multiplication for FFT-grid sizes and q-tile workspace
  sizes before allocation or indexing.
- Make per-k EXX progress, forced output flushes, and CUDA-memory queries opt-in
  through `ABACUS_EXX_ENERGY_DIAGNOSTICS`. Default execution performs none of
  those diagnostics.
- Remove the unused `consecutive_integers()` helper left by conflict
  resolution.

## Non-goals

- No new GPU or band-edge-label KPAR support.
- No change to the ordered qk map, full-point occupation scaling, ACE
  construction, real-space symmetry rotation, or q-tile cache lifetime.
- No broad refactor of the EXX operator or ABACUS integration harness.
- No implementation of ROCm EXX; the existing explicit unsupported behavior is
  preserved.

## Data flow and invariants

### KPAR identity map

Each pool contributes its local `wk[ik]` to
`wk_global[startk_pool[my_pool] + ik]`. The destination index is valid when it
is in `[0, nkstot_np)`. After the world reduction, every rank holds the same
complete full-q weights.

### Label validation

Input conversion rejects EXX-label requests with `kpar != 1` before the PBE SCF
starts. This avoids completing an expensive base-state calculation only to
fail during post-SCF label construction.

### Auto-tiling memory

The estimator must cover both operator application and exact-energy evaluation.
The potential-cache contribution is therefore:

```text
max(active_q_count, max_star_size * q_tile_size) * npw * sizeof(Real)
```

The estimate remains conservative for full-k meshes, where `max_star_size=1`.

### Diagnostics

`ABACUS_EXX_ENERGY_DIAGNOSTICS` is disabled when absent, empty, or `0`, and
enabled when set to `1`. Unsupported values fail early with a clear message.
Only the exact-energy progress/memory diagnostics are gated; normal ABACUS
logging is unchanged.

## Testing strategy

All production changes follow red-green-refactor.

1. K-list unit tests reproduce nonzero-pool weight loss before the bound fix.
2. Input tests reproduce acceptance of label requests with `kpar > 1` before
   adding the guard.
3. Integration assertions fail on the stale label string and missing numerical
   label reference before harness updates.
4. Auto-tiling policy tests reproduce the star-size underestimation.
5. Checked-product tests exercise overflow without allocating large buffers.
6. Diagnostic policy tests verify default-off, explicit-on, and invalid values.
7. Existing k-list, PW EXX policy, CPU symmetry, and CUDA symmetry suites remain
   green.
8. Small MPI CPU KPAR cases run outside the sandbox with
   `OMP_NUM_THREADS=1`.
9. The validated GaAs `2x2x2`, 60 Ry GPU symmetry comparison is rerun after all
   changes.

## Commit structure

1. `Fix PW EXX KPAR mapping and label validation`
2. `Harden PW EXX tiling and diagnostics`

Documentation and test-harness changes are committed with the phase whose
behavior they validate.

## Success criteria

- Full-q weights are complete and identical on all pools.
- KPAR=1 and KPAR=2 small HSE/ACE results agree within the configured numerical
  tolerance.
- EXX-label requests with `kpar > 1` fail during input validation.
- The PBE EXX-label integration test checks the label energy numerically.
- Auto-tiling never omits the reduced-energy star multiplier from its cache
  estimate.
- Workspace size overflow is rejected before allocation.
- Exact-energy diagnostics have no default per-k synchronization overhead.
- All focused CPU, MPI, and CUDA verification passes with no regression in the
  GaAs symmetry result.
