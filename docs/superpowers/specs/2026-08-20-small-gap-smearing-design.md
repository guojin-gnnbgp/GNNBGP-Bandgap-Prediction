# Small-gap smearing preset v1.1

## Goal

Change both ABACUS backends from the original Gaussian smearing width of
`0.015 Ry` to exactly `0.001 eV`, then regenerate the required InSb and Si
release baselines with the final package-built executable.

## Unit convention

ABACUS reads `smearing_sigma` in Rydberg. The package will keep the requested
value as `SMEARING_SIGMA_EV = 0.001` and convert it with
`1 Ry = 13.605693122994 eV`, producing
`smearing_sigma 7.349864435130999e-05` in every generated ABACUS `INPUT`.
The public settings record will retain both the ABACUS value and the intended
electron-volt value.

## Preset and provenance

The locked ABACUS preset version changes from `1.0` to `1.1`. No other
scientific or execution parameter changes: 100 Ry cutoff, `kspacing=0.15`,
Gaussian smearing, no SOC, `nspin=1`, one MPI rank, one OpenMP thread, HSE06
ACE/automatic tiling, and CIDER-PBE0 model/mixing remain fixed.

Existing v1.0 run directories and result JSON files are immutable historical
artifacts. They will not be edited or relabeled. The tracked release baseline
file will be replaced only after four new v1.1 calculations converge and their
executable, preset, structure, model, and pseudopotential hashes are checked.

## Testing and release evidence

Golden-input tests must fail against the old width before implementation and
pass after the preset change. Tests will assert the exact electron-volt intent,
Rydberg conversion, and preset version. Documentation will state both units.

The required local GPU/MPI matrix is InSb and Si through `abacus-hse06` and
`cider-pbe0`, always with `OMP_NUM_THREADS=1`. Occupation-aware classification
remains authoritative for narrow-gap results. Fast tests, GNN regressions,
linting, package build, clean-wheel validation, asset verification, backend
discovery, and `bgp doctor` will be rerun before completion.

GaAs and MgO remain GNN-covered extended examples; no new extended ABACUS
calculation is required by this change.
