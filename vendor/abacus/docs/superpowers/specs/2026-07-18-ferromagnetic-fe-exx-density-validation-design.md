# Ferromagnetic Fe EXX Energy-Density Validation Design

## Objective

Verify that `exx_energy_density_g` works for a genuinely spin-asymmetric collinear calculation, not merely an `nspin=2` calculation whose two spin channels remain equal.

## System

Use the conventional two-atom bcc Fe cell because it is a direct physical ferromagnetic model and matches existing ABACUS Fe examples:

- cubic lattice parameter: 2.86 Angstrom;
- Fe positions: `(0,0,0)` and `(0.5,0.5,0.5)`;
- pseudopotential: `Fe_ONCV_PBE-1.0.upf`, norm-conserving, 16 valence electrons per atom;
- collinear `nspin=2` with initial moment `3.0 Bohr magnetons` per Fe atom;
- no fixed `nupdown`: the final moment is determined self-consistently.

## Numerical Profile

- PBE plane waves, double precision, CPU;
- `ecutwfc=60 Ry`, `nbands=22` per spin channel;
- Gaussian smearing `0.02 Ry`;
- Gamma-centered 2x2x2 k mesh;
- symmetry enabled, `kpar=1`, `bndpar=1`;
- deterministic random wavefunctions with `pw_seed=1`;
- one MPI rank and one OpenMP thread for this first stage.

## Acceptance Checks

1. SCF converges.
2. Final total magnetism is finite and clearly nonzero.
3. Integrated `rho_valence_sg[0]` and `rho_valence_sg[1]` are unequal and sum to 32 valence electrons within numerical tolerance.
4. HDF5 metadata reports `nspin=2` and `exx_label.energy_density.spin="summed"`.
5. `arrays/exx_energy_density_g` is finite `float64`, shape `[nxyz]`, and normally negative.
6. `sum_g exx_energy_density_g[g] * dV_bohr3` matches the scalar EXX label within `max(1e-8 Ha, 1e-8*abs(E_scalar))`.
7. Report spin electron counts, magnetic moment, scalar/integrated EXX energies, residual, and imaginary diagnostic.

## Scope

All generated Fe inputs and outputs remain under `/tmp/abacus-fe-fm-spin-exx`. This stage does not add a repository fixture, modify source code, constrain `nupdown`, run multiple MPI ranks, commit, or push.
