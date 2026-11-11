# Methods and scientific limitations

## Backend definitions

`gnn-hse06` returns a scalar from the supplied legacy ScalarPropertyMACE model.
It was trained for an HSE-like band-gap target and does not run ABACUS. The
model has incomplete training provenance, so every result carries a
`legacy/provisional` and trusted-pickle warning. Its hash is recorded with the
prediction.

`abacus-hse06` runs self-consistent HSE06 with the combined PW CUDA exact-
exchange implementation. ACE and automatic EXX tiling are enabled. This is the
computationally expensive first-principles backend.

`abacus-pbe0` runs direct self-consistent PBE0 with the standard 25% exact-
exchange fraction through the same accelerated GPU EXX implementation. It is
the first-principles PBE0 reference backend and stages no CIDER model.

`cider-pbe0` runs the native-CIDER surrogate fitted to the original PBE0
reference campaign. ABACUS uses a PBE base and mixes the learned CIDER
correction with `cider_xmix=0.15`, calibrated against the no-SOC InSb HSE06
target. The CIDER descriptors use valence-only
`rho`, `sigma`, and `tau`: nonlinear core-correction density present in a UPF
may participate in the ordinary PBE/orbital calculation but is excluded from
the learned feature inputs.

## Shared locked ABACUS preset

All three ABACUS backends use:

- the bundled scalar-relativistic PseudoDojo norm-conserving potentials;
- 100 Ry plane-wave cutoff and `kspacing=0.15`;
- double precision, Gaussian smearing of exactly 0.001 eV
  (7.349864435130999e-05 Ry), Davidson diagonalization, and Broyden mixing;
- `nspin=1`, no SOC, and one MPI rank with one OpenMP thread;
- the exact input cell and Cartesian positions, without relaxation, wrapping,
  reduction, primitive-cell conversion, or standardization.

Ordinary CLI users cannot override this preset. This makes results comparable
and ensures that a recorded preset version has a definite meaning.
The direct HSE06 and PBE0 backends retain preset `1.1`; the calibrated CIDER
backend uses preset `1.2`.

## Interpreting different answers

The backends do not estimate an interchangeable observable. Direct PBE0 and
CIDER-PBE0 are retained as separate methods: the former uses a standard 25%
exact Fock-exchange fraction, while the latter uses a calibrated 0.15 mixing
coefficient for a learned CIDER correction. In particular,
InSb is a narrow-gap, SOC-sensitive material while version 0.1 deliberately
omits SOC. The legacy GNN produces an HSE-like scalar, whereas a converged CIDER
calculation may have fractional occupations and therefore reports a zero gap.
The package keeps both results with their method, occupation evidence, asset
hashes, and settings. It does not average them or relabel one method as another.

## Current limitations

- no spin polarization, magnetic initialization, noncollinear magnetism, or SOC;
- no molecular or partially periodic cells;
- no geometry or cell optimization;
- no scheduler, remote host, multi-rank, or asynchronous execution;
- no arbitrary ABACUS input overrides;
- the GNN model is provisional and a trusted pickle;
- PseudoDojo public redistribution remains subject to a separate license review.
