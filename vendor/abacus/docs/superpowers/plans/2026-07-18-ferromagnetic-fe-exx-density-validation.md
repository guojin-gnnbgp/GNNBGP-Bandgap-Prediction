# Ferromagnetic Fe EXX Energy-Density Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and independently validate a genuinely spin-polarized, single-rank bcc Fe EXX energy-density calculation.

**Architecture:** Create a scratch conventional bcc Fe input with unconstrained self-consistent magnetization, run the existing MPI PW CPU binary with one rank, and inspect both the ABACUS magnetic diagnostics and HDF5 spin-resolved density arrays. The EXX output remains one spin-summed grid whose integral must equal the scalar label.

**Tech Stack:** ABACUS PW/MPI CPU double precision, ONCV-PBE Fe pseudopotential, h5py, NumPy.

## Global Constraints

- Generate files only under `/tmp/abacus-fe-fm-spin-exx`.
- Use the conventional two-atom bcc cell with `a=2.86 Angstrom` and positions `(0,0,0)`, `(0.5,0.5,0.5)`.
- Use self-consistent `nspin=2` with initial `3.0` Bohr magnetons per atom and no fixed `nupdown`.
- Use `Fe_ONCV_PBE-1.0.upf`, `ecutwfc=60 Ry`, `nbands=22`, Gaussian smearing `0.02 Ry`, and a Gamma-centered 2x2x2 k mesh.
- Run `mpirun -np 1` outside the sandbox with `OMP_NUM_THREADS=1`.
- Do not modify source, datasets, repository fixtures, index, HEAD, branch, or commits.

---

### Task 1: Create the physical Fe input

**Files:**
- Create: `/tmp/abacus-fe-fm-spin-exx/INPUT`
- Create: `/tmp/abacus-fe-fm-spin-exx/KPT`
- Create: `/tmp/abacus-fe-fm-spin-exx/STRU`

**Interfaces:**
- Consumes: `tests/PP_ORB/Fe_ONCV_PBE-1.0.upf`.
- Produces: deterministic single-rank ferromagnetic Fe SCF input.

- [ ] **Step 1: Write the exact ABACUS inputs**

Set PBE/PW, `nspin=2`, `ecutwfc=60`, `nbands=22`, `smearing_method=gaussian`, `smearing_sigma=0.02`, `symmetry=1`, `pw_seed=1`, `out_training_data=1`, `out_exx_label=1`, and `out_exx_energy_density=1`. Put `mag 3.0` on both Fe atoms and omit `nupdown`.

- [ ] **Step 2: Check the input contract**

Confirm the Fe UPF is norm-conserving with 16 valence electrons and that the cell contains exactly two Fe atoms, for 32 total valence electrons.

### Task 2: Run the single-rank calculation

**Files:**
- Produce: `/tmp/abacus-fe-fm-spin-exx/OUT.fe_fm_spin_exx/running_scf.log`
- Produce: `/tmp/abacus-fe-fm-spin-exx/OUT.fe_fm_spin_exx/training_dump/step_0.h5`

**Interfaces:**
- Consumes: Task 1 input and `build-exx-energy-density-cpu-mpi/abacus_pw_para`.
- Produces: converged magnetic PBE state and one-shot full-range EXX label/grid.

- [ ] **Step 1: Run outside the sandbox**

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mpirun -np 1 /home/bonan/appdir/CiderPress/abacus-cider/build-exx-energy-density-cpu-mpi/abacus_pw_para
```

- [ ] **Step 2: Check convergence and magnetism**

Require `#SCF IS CONVERGED#` and extract the final total/absolute magnetism and spin-up/spin-down electron counts from `running_scf.log`.

### Task 3: Validate spin asymmetry and EXX parity

**Files:**
- Read: `/tmp/abacus-fe-fm-spin-exx/OUT.fe_fm_spin_exx/training_dump/step_0.h5`
- Create: `/tmp/abacus-fe-fm-spin-exx/validation_summary.json`

**Interfaces:**
- Consumes: Task 2 output.
- Produces: independent spin-density and EXX-energy validation metrics.

- [ ] **Step 1: Validate the generic EXX HDF5 contract**

```bash
/home/bonan/appdir/CiderPress/.venv/bin/python3 \
  /home/bonan/appdir/CiderPress/abacus-cider/tests/integrate/tools/validate_exx_energy_density_h5.py \
  /tmp/abacus-fe-fm-spin-exx/OUT.fe_fm_spin_exx/training_dump/step_0.h5 \
  --expect-present
```

- [ ] **Step 2: Integrate the spin-resolved valence densities**

Read `arrays/rho_valence_sg`, require shape `[2,nxyz]`, and calculate `N_up` and `N_down` with `dV_bohr3`. Require `N_up+N_down` to match 32 within `1e-8` electrons and `abs(N_up-N_down)>0.1` electrons.

- [ ] **Step 3: Report**

Record the converged moment, spin electron counts, scalar and integrated EXX energies, residuals, maximum imaginary magnitude, grid shape/range, and retained scratch paths. Do not stage, commit, or push.
