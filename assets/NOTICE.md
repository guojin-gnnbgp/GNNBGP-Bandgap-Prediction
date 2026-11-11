# Scientific asset provenance

This directory contains immutable copies of scientific files supplied for the
band-gap integration project. `manifest.json` records the byte size and SHA-256
of every file.

## Legacy GNN assets

The ScalarPropertyMACE pickle and `atom_init.json` were copied from the supplied
GNNBGP InSb example. GNNBGP source is MIT licensed; its license is retained when
the compatible inference namespace is integrated. The pickle has incomplete
training provenance and is intentionally labelled **legacy/provisional**. It is
loaded only after its approved SHA-256 is verified and must be treated as trusted
executable Python/PyTorch data.

## CIDER model

`nldf_mgga_cand00_n96_noise0p01.yaml` is the authoritative balanced band-edge
model supplied from the original PseudoDojo/PBE0-reference fitting campaign.
Runtime CIDER descriptors use valence-only `rho`, `sigma`, and `tau`; nonlinear
core correction density is excluded from CIDER features.

## PseudoDojo pseudopotentials

The 71 UPF files are an unchanged copy of the supplied scalar-relativistic,
norm-conserving PseudoDojo collection. Their embedded headers identify ONCVPSP
3.3.0 by D. R. Hamann and suggest citing D. R. Hamann, *Phys. Rev. B* **88**,
085117 (2013). Some files contain a model core charge; this remains available to
the ordinary electronic-structure calculation but is excluded from CIDER feature
inputs.

No standalone redistribution license accompanied the supplied UPF directory.
The embedded notices are preserved exactly in every UPF, but public release of
the collection requires a separate redistribution review; this notice makes no
unsupported licensing claim.
