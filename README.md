# Bandgap Predictor

`bandgap-predictor` provides the `bgp` command for local, provenance-rich
band-gap predictions from a periodic crystal structure. It keeps the submitted
cell, positions, species, and site order unchanged and exposes four explicit
methods:

- `gnn-hse06`: a fast legacy/provisional HSE-like ScalarPropertyMACE model
  with a frozen direct/indirect classification head (the scalar output is
  bit-identical to the mid-term 20250421 model; the head adds
  `prediction.gap_type` and `prediction.gap_type_probability`);
- `abacus-hse06`: GPU-accelerated HSE06 in the package-owned ABACUS source;
- `abacus-pbe0`: direct PBE0 with the standard 25% exact-exchange fraction;
- `cider-pbe0`: the native-CIDER model fitted to PBE0 reference data.

The three electronic-structure backends use the same bundled, hash-verified
PseudoDojo pseudopotentials. Results are method-specific: the software records
their provenance and occupation interpretation instead of forcing the methods
to agree.

## Quick start

From a source checkout:

```bash
uv sync --extra gnn --group dev
uv run python tools/build_abacus.py \
  --build-dir build/abacus \
  --install-prefix build/abacus-install \
  --cuda-architectures 86,89,120
uv run bgp doctor --abacus build/abacus-install/bin/abacus_pw_gpu
uv run bgp predict examples/InSb.cif \
  --backend abacus-pbe0 \
  --abacus build/abacus-install/bin/abacus_pw_gpu
```

The ABACUS build is an explicit, separate step. Installing the Python package
never compiles ABACUS and no ABACUS executable is included in the wheel.

For structure-only inference:

```bash
uv run bgp predict examples/Si.cif --backend gnn-hse06
```

Useful diagnostics are:

```bash
uv run bgp backends --abacus /absolute/path/to/abacus_pw_gpu
uv run bgp assets verify
uv run bgp doctor --abacus /absolute/path/to/abacus_pw_gpu
```

## Scope

Version 0.1 runs synchronously on the local machine using one MPI rank and one
OpenMP thread. ABACUS calculations are nonmagnetic (`nspin=1`), have no SOC,
and use a locked 100 Ry, `kspacing=0.15` preset. There is no relaxation,
standardization, scheduler submission, or arbitrary ABACUS override. Every run
directory is retained with its input, logs, process metadata, and versioned
`result.json`, including failed and interrupted runs.

See [installation](docs/installation.md), [method definitions](docs/methods.md),
and [result semantics](docs/results.md) for the full contract.

## Source and asset notices

The integrated ABACUS source is under `vendor/abacus`; its exact lineage is in
`vendor/abacus/UPSTREAM.md`. Scientific asset hashes and provenance are in
`assets/manifest.json` and `assets/NOTICE.md`. Applicable notices are collected
under `LICENSES/`. The supplied PseudoDojo files retain their embedded notices,
but their public redistribution status still requires a separate review.
