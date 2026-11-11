# InSb CIDER Mixing Sweep Design

## Objective

Determine the `cider_xmix` in the interval `[0.00, 0.25]` whose converged InSb CIDER band gap matches the locked HSE06 reference gap of `0.244276540450355 eV` to within `0.05 eV`.

## Scientific controls

The sweep uses the supplied `examples/InSb.cif` without structural transformation. Every point uses the bundled `nldf_mgga_cand00_n96_noise0p01.yaml` model and PseudoDojo In/Sb pseudopotentials, 100 Ry, `kspacing=0.15`, Gaussian smearing of exactly `0.001 eV`, `nspin=1`, no SOC, one MPI rank, and one OpenMP thread. Only `cider_xmix` changes. The package's public `cider-pbe0` preset remains locked at `0.25`.

## Sweep and fit

Run fresh independent calculations at `cider_xmix = 0.00, 0.05, 0.10, 0.15, 0.20, 0.25`. Fit the converged numerical gaps with ordinary least squares,

`gap_eV = intercept_eV + slope_eV_per_xmix * cider_xmix`.

Solve the line for the HSE06 gap. If no calculated coarse point is within `0.05 eV`, run one fresh confirmation calculation at the fitted mixing value clipped to `[0.00, 0.25]`. Report both the continuous fitted crossing and the nearest actually calculated point; do not present an extrapolated value outside the interval as a match.

## Artifacts and acceptance

Retain every input, output, log, process record, parsed electronic-state artifact, and point metadata below `build/cider-xmix-insb-v1.1`. Write versioned JSON and CSV summaries containing convergence, gap, classification, executable/model/pseudopotential hashes, structure hash, settings, elapsed time, fit coefficients, fitted crossing, and residuals. The requested curve shows calculated points, the linear fit, the HSE06 reference, and the ±0.05 eV acceptance band.

Success requires all coarse points used in the fit to be converged and finite, provenance hashes to match across the sweep, and at least one calculated point within `0.05 eV` of HSE06.

