# CIDER Mixing Preset v1.2 Design

## Objective

Promote the experimentally validated InSb CIDER mixing value to the production
`cider-pbe0` backend. The locked backend will use `cider_xmix=0.15` for band-gap
prediction instead of `0.25`.

## Preset boundary

Only the native-CIDER backend changes. Its preset version advances from `1.1`
to `1.2` so that result metadata and settings hashes distinguish the two
scientific configurations. The direct `abacus-hse06` and `abacus-pbe0`
backends remain preset `1.1` and retain their existing inputs and baselines.

CIDER preset `1.2` otherwise preserves the complete locked configuration:
the original `nldf_mgga_cand00_n96_noise0p01.yaml` model, bundled PseudoDojo
pseudopotentials, valence-only CIDER features, 100 Ry, `kspacing=0.15`, exactly
0.001 eV Gaussian smearing, `nspin=1`, no SOC, one MPI rank, and one OpenMP
thread. The public CLI does not gain an arbitrary `cider_xmix` override.

## Input and metadata changes

The immutable CIDER input mapping and golden ABACUS INPUT change only from
`cider_xmix 0.25` to `cider_xmix 0.15`. Prepared CIDER settings report
`preset_version=1.2`; other ABACUS methods continue to report `1.1`. Current
methods documentation will explain that `0.15` is a calibrated learned-
correction mixing coefficient and remains scientifically distinct from the
25% exact Fock-exchange fraction used by direct PBE0.

Historical design documents, retained preset-v1.1 calculation directories,
and the completed 0.00–0.25 sweep remain immutable evidence and are not
rewritten.

## Baseline regeneration

The final package-built executable will run production `cider-pbe0` predictions
for the exact supplied InSb and diamond-Si structures under preset `1.2`.
Both calculations run locally and synchronously with one MPI rank and
`OMP_NUM_THREADS=1`, and retain their complete work directories.

The hardware baseline file and results documentation will replace only the
CIDER InSb and Si entries with the new converged gaps, classifications, and
settings hashes. Executable, model, pseudopotential, and structure hashes must
match the retained artifacts. No GaAs or MgO values will be inferred.

## Testing and acceptance

Implementation follows a red-green cycle: first change the CIDER input test to
require `cider_xmix=0.15` and preset `1.2`, verify that it fails against the
current implementation, then minimally update the mapping and golden input.

Acceptance requires:

- focused input tests pass while HSE06 and direct-PBE0 golden inputs remain
  unchanged;
- the full non-hardware test suite passes;
- fresh InSb and Si CIDER calculations converge and return finite,
  occupation-aware results;
- hardware baselines reproduce the retained result settings and provenance;
- documentation identifies the calibrated `0.15` value and the preset-v1.2
  transition without altering historical records.
