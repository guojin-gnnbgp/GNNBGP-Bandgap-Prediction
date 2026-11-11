# InSb CIDER Mixing Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a converged InSb band-gap curve over `cider_xmix = 0.00–0.25` and locate the mixing value that matches HSE06 within `0.05 eV`.

**Architecture:** Use existing package input preparation, local ABACUS runner, and parser APIs in isolated build artifacts. Change only `cider_xmix` in each generated INPUT, run points sequentially on one GPU, and analyze the retained JSON results without extending the public CLI.

**Tech Stack:** Python 3.12, NumPy, ASE, bandgap-predictor, ABACUS PW CUDA+MPI, D3 visualization.

## Global Constraints

- Use the supplied `examples/InSb.cif` exactly as provided.
- Use `build/abacus-install-fixed/bin/abacus_pw_gpu` with `OMP_NUM_THREADS=1` outside the sandbox.
- Use the bundled PseudoDojo files and `nldf_mgga_cand00_n96_noise0p01.yaml` model.
- Keep all locked preset-v1.1 settings unchanged except `cider_xmix`.
- Do not modify historical run directories or the production backend default.
- Require convergence and finite occupation-aware gaps before fitting.

---

### Task 1: Prepare the isolated coarse sweep

**Files:**
- Create: `build/cider-xmix-insb-v1.1/xmix-*/INPUT`
- Create: `build/cider-xmix-insb-v1.1/xmix-*/STRU`
- Create: `build/cider-xmix-insb-v1.1/xmix-*/point.json`

**Interfaces:**
- Consumes: `prepare_abacus_input(..., AbacusMethod.CIDER_PBE0)` and `examples/InSb.cif`.
- Produces: six independent ABACUS work directories and point metadata.

- [ ] Verify the executable with `uv run bgp doctor --abacus build/abacus-install-fixed/bin/abacus_pw_gpu`.
- [ ] Generate each work directory using the existing input preparation API, replace exactly `cider_xmix 0.25` with its assigned value, and record the structure/executable/model/pseudopotential hashes in `point.json`.
- [ ] Assert every INPUT differs only in the `cider_xmix` line and contains `smearing_sigma 7.349864435130999e-05`.

### Task 2: Run and validate the coarse calculations

**Files:**
- Create: `build/cider-xmix-insb-v1.1/xmix-*/stdout.log`
- Create: `build/cider-xmix-insb-v1.1/xmix-*/process.json`
- Create: `build/cider-xmix-insb-v1.1/xmix-*/OUT.ABACUS/`

**Interfaces:**
- Consumes: six prepared work directories.
- Produces: ABACUS electronic-state artifacts and parsed gaps.

- [ ] Run each point sequentially through `run_abacus()` with one MPI rank and `OMP_NUM_THREADS=1`.
- [ ] Parse each point with `parse_abacus_result(workdir, returncode)` and require `converged is True`, a finite non-negative gap, and a successful process exit.
- [ ] Preserve failed work directories and stop before fitting if any coarse point is invalid.

### Task 3: Fit and confirm the HSE06 crossing

**Files:**
- Create: `build/cider-xmix-insb-v1.1/sweep-results.json`
- Create: `build/cider-xmix-insb-v1.1/sweep-results.csv`

**Interfaces:**
- Consumes: converged `(cider_xmix, gap_ev)` points.
- Produces: linear coefficients, residuals, fitted crossing, and nearest calculated match.

- [ ] Fit `numpy.polyfit(xmix, gap_ev, 1)` and solve `(HSE06_GAP_EV - intercept) / slope`.
- [ ] Identify the calculated point minimizing `abs(gap_ev - HSE06_GAP_EV)`.
- [ ] If that error exceeds `0.05 eV`, prepare and run one confirmation point at the clipped fitted crossing and include it in the retained results.
- [ ] Write JSON and CSV atomically and verify that their values reproduce the fit.

### Task 4: Create and verify the requested curve

**Files:**
- Create outside repository: `/home/bonan/.codex/visualizations/2026/08/19/01a01894-2676-7812-8cda-e788164db84a/insb-cider-xmix-curve.html`

**Interfaces:**
- Consumes: `sweep-results.json`.
- Produces: responsive curve with calculated points, fit, HSE06 target, and tolerance band.

- [ ] Render the measured points and fitted line with labeled `cider_xmix` and band-gap axes.
- [ ] Add the HSE06 horizontal reference and ±`0.05 eV` band; label the fitted crossing and nearest calculated match.
- [ ] Verify marks, labels, and tooltip behavior at 736 px and 360 px, then report retained calculation and summary paths.

