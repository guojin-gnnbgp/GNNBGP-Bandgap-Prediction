# ABACUS Conventional-Gauge EXX Energy Density Design

## Objective

Add an opt-in PW training-data output, `arrays/exx_energy_density_g`, containing the spin-summed, full-range exact-exchange energy density evaluated on the converged PBE orbitals used by the existing scalar EXX label. The field uses the conventional Fock/exchange-hole gauge and is stored as real `float64` in Ha/Bohr^3 on the charge-density FFT grid.

The central numerical invariant is

\[
\sum_g e_x(\mathbf r_g)\,\Delta V
= E_x^{\mathrm{scalar}},
\qquad
\Delta V = \Omega/N_{xyz},
\]

within `max(1e-8 Ha, 1e-8 * abs(E_x))` for CPU double precision. The grid field must not be normalized after accumulation to force this identity.

## User Interface and Validation

Add `out_exx_energy_density`, a boolean input parameter defaulting to `false`.

When `true`, input validation requires all of the following:

- `out_training_data = true`;
- `out_exx_label = true`;
- `basis_type = pw`;
- `nspin = 1` or `2`;
- `kpar = 1`;
- `bndpar = 1`.

The existing training-data validation continues to require PBE, LibXC, and norm-conserving pseudopotentials. Runtime validation additionally requires the EXX real-space mesh and distribution to be identical to the charge-density FFT mesh. This is normally true when `ecutexx = 0`, because the EXX operator then uses `ecutrho`. A nonmatching explicit `ecutexx` is rejected for density output rather than interpolated.

The implementation supports both internal k policies currently accepted by the scalar label, `pair-map` and `full-active`, provided their full physical k/q pairs are reconstructed on the common real-space mesh. Any policy other than those two already fails in the scalar operator and remains unsupported.

## Mathematical Normalization

The current operator forms, for each physical spin/k/q/band pair,

\[
\rho_{nm}^{kq}(\mathbf r_g)
= \frac{1}{\Omega}\psi_{nk}(\mathbf r_g)
  \psi_{mq}^*(\mathbf r_g),
\]

using `cal_density_real_op`. It Fourier transforms this pair density and contracts it with the cached full-range Fock kernel `P_{kq}(G)`, including the current singularity correction:

\[
E_x^{\mathrm{Ry}}
= \frac{\Omega}{2}
  \sum_{\sigma,kq,nm} W_{\sigma kn,qm}
  \sum_G |\rho_{nm}^{kq}(G)|^2 P_{kq}(G).
\]

`W` denotes exactly the scalar traversal's combined spin degeneracy, representative/full-star k weight, occupation, q weight, and pair-map multiplier. `P` already carries the negative Fock sign. The scalar code applies `0.5 * omega` after the traversal and performs the existing MPI reduction.

For density output, the same reciprocal pair density is multiplied in place by the same cached potential and inverse transformed:

\[
v_{nm}^{kq}(\mathbf r_g)
= \mathcal F^{-1}\left[P_{kq}(G)\rho_{nm}^{kq}(G)\right](\mathbf r_g).
\]

The complex conventional-gauge accumulator in Ry/Bohr^3 is

\[
\tilde e_x(\mathbf r_g)
= \frac{1}{2}
  \sum_{\sigma,kq,nm} W_{\sigma kn,qm}
  \rho_{nm}^{kq}(\mathbf r_g)^*
  v_{nm}^{kq}(\mathbf r_g).
\]

ABACUS's existing forward/inverse FFT normalization then gives the same discrete Parseval contraction:

\[
\sum_g \tilde e_x(\mathbf r_g)\frac{\Omega}{N_{xyz}}
= E_x^{\mathrm{Ry}}.
\]

The implementation accumulates `tilde e_x` as complex, performs all required pair and MPI sums, records `max_g |Im tilde e_x(r_g)|`, and only then exports `0.5 * Re tilde e_x` in Ha/Bohr^3. The scalar label is likewise converted from Ry to Ha with the existing factor `0.5`.

No independent textbook occupation factor, sign, or spin factor is introduced: the scalar traversal's `k_weight * q_weight` is authoritative.

## Operator API and Data Flow

Introduce an operator result type that contains:

- scalar energy in Ry;
- optional complex local-grid energy density in Ry/Bohr^3;
- maximum residual imaginary magnitude after global accumulation;
- the selected k-policy name and local grid dimensions.

Keep `cal_exx_energy(psi)` as the compatibility scalar-only entry point. Add a result-returning entry point used only by the post-SCF training-label path when `out_exx_energy_density` is enabled. Both entry points call one internal q-tile implementation with an optional accumulator pointer, so there is one k/q/band traversal.

Within each scalar pair contribution:

1. Form `density_real` and `density_recip` exactly as today.
2. Evaluate the existing scalar contraction before mutating `density_recip`.
3. If density output is enabled, multiply `density_recip` by the cached `pot_ik_iq`, inverse FFT to real space, and accumulate `0.5 * weight * conj(pair_density_real) * pair_potential_real` into one complex local-grid buffer.
4. Reuse existing single/batched workspaces. A dedicated kernel performs the complex pointwise accumulation on CPU and CUDA without a device-to-host copy inside the pair loop.

The disabled path allocates no grid accumulator, performs no extra inverse FFT, and follows the current scalar operations unchanged. The enabled path adds one grid-sized complex accumulator and reuses existing pair scratch; it never allocates storage proportional to the number of orbital pairs.

For CUDA, accumulation stays on device throughout the traversal. There is one grid-sized device-to-host transfer after the traversal and reductions, not one per pair.

## MPI, Grid Ordering, and Lifecycle

The EXX accumulator is local in the same slab distribution as the charge-density FFT basis. It is reduced over the same communicators needed to combine the scalar physical-pair ownership. The result passed to the dump remains in local charge-grid order; `write_training_dump` uses its existing `gather_spin_major_to_global` ordering mechanism with one component, exactly as for `rho_core_g`.

Because `write_training_dump` is reached indirectly through `ctrl_scf_pw`, use the existing optional-label lifecycle pattern:

- define `TrainingExxEnergyDensity` in `training_dump.h`;
- add explicit `clear_last_training_exx_energy_density()` and `set_last_training_exx_energy_density(...)` functions;
- clear it at the start of every `after_scf` call;
- set it only after a successful requested evaluation;
- require a present, size-consistent payload when `out_exx_energy_density` is true;
- clear it after output or at the next lifecycle start so data cannot leak between steps/calculations.

The payload stores the local real Ha/Bohr^3 field plus scalar/integrated energies, absolute/relative residuals, imaginary residual, k policy, and mesh metadata. The dump recomputes and checks the global integral before writing.

## HDF5 and Metadata Contract

When requested, write the double dataset:

```text
/arrays/exx_energy_density_g    shape [nxyz]
```

Add `exx_energy_density_g` to the JSON `arrays` map. Extend `exx_label` with:

```json
"energy_density": {
  "present": true,
  "array": "exx_energy_density_g",
  "unit": "Ha Bohr^-3",
  "gauge": "conventional_fock_exchange_hole",
  "spin": "summed",
  "grid": "charge_density_fft_real_space",
  "integrated_energy_Ha": 0.0,
  "scalar_energy_Ha": 0.0,
  "integration_abs_residual_Ha": 0.0,
  "integration_rel_residual": 0.0,
  "max_imaginary_abs_Ha_bohr3": 0.0,
  "full_range": true,
  "hybrid_scaled": false,
  "base_state": "pbe_scf_density_and_pw_wavefunctions",
  "k_policy": "pair-map"
}
```

When the flag is false, omit the optional `energy_density` object and array mapping entirely. This preserves byte-level behavior as closely as practical for existing scalar dumps. The record remains schema version 2 because this is an optional backward-compatible extension.

## Error Handling

Fail with a precise `WARNING_QUIT` message for:

- any invalid input combination;
- missing wavefunctions, occupations, or EXX density payload;
- EXX/charge FFT mesh or local-layout mismatch;
- unsupported k policy;
- non-finite scalar, field, or diagnostic values;
- field size inconsistent with the local or global grid;
- scalar/grid integration residual beyond the CPU acceptance tolerance.

The integration residual is diagnostic metadata but also a correctness guard. It is never repaired by rescaling the field.

## Test Strategy

Development follows red-green-refactor cycles.

1. Parser tests establish default `false`, parsing, valid configuration, and separate death tests for every invalid dependency.
2. A pure complex-accumulation kernel test establishes the scalar/grid identity with complex data and verifies imaginary cancellation occurs only after summing conjugate-related contributions.
3. Writer/validator tests establish optional absence, required presence, array shape/type, metadata values, and stale-payload rejection.
4. Separate integration cases cover Gamma `nspin=1`, `nspin=2`, multi-k, symmetry on, and at least two tile configurations. Pair-map and diagnostic full-active runs compare scalar and grid outputs.
5. A Python HDF5 validator reads `/record_json` and `/arrays/exx_energy_density_g` directly and prints scalar, integrated value, absolute/relative residual, imaginary residual, shape, dtype, and k policy.
6. Existing scalar cases run unchanged with the new flag off.
7. CPU double builds/runs are mandatory. CUDA compilation is mandatory; a numerical CUDA comparison runs when a compatible GPU is available, with its tolerance derived from observed CPU/GPU results.

## Alternatives Considered

### 1. Direct pair-field accumulation in the scalar traversal — selected

This directly represents the conventional gauge, shares all authoritative scalar weights and Coulomb handling, uses bounded grid memory, and avoids a second EXX traversal.

### 2. Accumulate occupied-orbital Fock actions, then contract with orbitals

This is mathematically valid and resembles `process_qtile_apply_tile`, but requires storing or repeatedly managing Fock-action grids for occupied orbitals. It increases memory and complicates symmetry/star bookkeeping without improving the normalization proof.

### 3. Inverse FFT a reciprocal scalar-energy integrand

This produces a field with the correct integral but does not directly establish the requested conventional Fock/exchange-hole gauge. It is rejected.

## Scope Boundaries

This design does not change CiderPress training, target normalization, datasets, model fitting, covariance calculation, or unrelated ABACUS EXX behavior. It does not commit, push, regenerate references for existing flag-off cases, or silently support incompatible grids.
