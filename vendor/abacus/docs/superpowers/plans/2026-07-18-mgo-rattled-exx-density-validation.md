# Rattled MgO EXX Energy-Density Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate `exx_energy_density_g` for a deterministic two-species, cell-and-position-rattled MgO primitive cell on a Gamma-centered 2x2x2 k mesh.

**Architecture:** Construct rocksalt MgO with ASE, apply a reproducible symmetric strain whose largest component is 3% and opposite 0.05 Angstrom Mg/O displacements, and write only scratch ABACUS inputs. Run the existing CPU/MPI `abacus_pw_para` binary under pair-map and full-active EXX-label policies, then independently validate and compare the HDF5 fields.

**Tech Stack:** ASE 3.28, ABACUS PW/MPI CPU double precision, ONCV-PBE Mg/O pseudopotentials, h5py, NumPy.

## Global Constraints

- Work only under `/tmp/abacus-mgo-rattled-222` for generated inputs and outputs.
- Use `OMP_NUM_THREADS=1` for every ABACUS runtime and launch MPI outside the sandbox.
- Use `Mg_ONCV_PBE-1.0.upf` and `O_ONCV_PBE-1.0.upf` from `tests/PP_ORB`.
- Use `ecutwfc=80 Ry`, `nspin=1`, `symmetry=1`, `kpar=1`, `bndpar=1`, and a Gamma-centered 2x2x2 k mesh.
- Do not modify datasets, repository fixtures, source code, or git state.
- Do not rescale the completed EXX energy-density field.

---

### Task 1: Construct and validate the distorted MgO structure

**Files:**
- Create in scratch: `/tmp/abacus-mgo-rattled-222/pair-map/STRU`
- Create in scratch: `/tmp/abacus-mgo-rattled-222/pair-map/structure.json`

**Interfaces:**
- Consumes: ASE rocksalt primitive MgO with cubic lattice parameter `4.21 Angstrom`.
- Produces: ABACUS direct-coordinate structure, exact strain/displacement record, volume, and minimum-image Mg-O distance.

- [ ] **Step 1: Generate a deterministic distorted cell and positions with ASE**

Use NumPy RNG seed `20260718`. Symmetrize a normally distributed 3x3 matrix and scale its largest absolute component to `0.03`; apply `F = I + strain` to the primitive rocksalt cell. Generate one random unit direction and move Mg by `+0.05 Angstrom` and O by `-0.05 Angstrom`, then wrap fractional coordinates.

- [ ] **Step 2: Verify the structure numerically**

Confirm the composition is `MgO`, both atoms remain in the periodic cell, the cell determinant is positive, and the minimum-image Mg-O distance exceeds `1.5 Angstrom`. Record cell, scaled positions, strain, displacements, volume, and distance in `structure.json`.

### Task 2: Run pair-map EXX-density validation

**Files:**
- Create in scratch: `/tmp/abacus-mgo-rattled-222/pair-map/INPUT`
- Create in scratch: `/tmp/abacus-mgo-rattled-222/pair-map/KPT`
- Produce: `/tmp/abacus-mgo-rattled-222/pair-map/OUT.mgo_rattled_222/training_dump/step_0.h5`

**Interfaces:**
- Consumes: Task 1 structure and `build-exx-energy-density-cpu-mpi/abacus_pw_para`.
- Produces: Pair-map scalar EXX label and conventional-gauge grid field.

- [ ] **Step 1: Run ABACUS outside the sandbox**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun -np 1 build-exx-energy-density-cpu-mpi/abacus_pw_para
```

- [ ] **Step 2: Validate the HDF5 contract and integral**

```bash
../.venv/bin/python3 tests/integrate/tools/validate_exx_energy_density_h5.py \
  /tmp/abacus-mgo-rattled-222/pair-map/OUT.mgo_rattled_222/training_dump/step_0.h5 \
  --expect-present
```

Expected: shape `[nxyz]`, dtype `float64`, finite negative integral, and residual below `max(1e-8 Ha, 1e-8*abs(E_scalar))`.

### Task 3: Run full-active comparison and report

**Files:**
- Copy scratch inputs to: `/tmp/abacus-mgo-rattled-222/full-active/`
- Produce: `/tmp/abacus-mgo-rattled-222/full-active/OUT.mgo_rattled_222/training_dump/step_0.h5`

**Interfaces:**
- Consumes: Pair-map inputs and output from Tasks 1-2.
- Produces: Full-active field plus scalar/grid comparison metrics.

- [ ] **Step 1: Run with full-active policy**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ABACUS_EXX_LABEL_K_POLICY=full-active
mpirun -np 1 build-exx-energy-density-cpu-mpi/abacus_pw_para
```

- [ ] **Step 2: Independently compare outputs**

Read both HDF5 files with h5py. Report scalar and integrated energies, residuals, maximum imaginary magnitudes, field minimum/maximum/mean, scalar absolute difference, grid maximum absolute difference, and grid RMS difference.

- [ ] **Step 3: Report supported behavior and any failure**

Include the exact distorted structure, SCF convergence, k-policy names, numerical results, and paths to retained scratch artifacts. Do not commit or stage anything.
