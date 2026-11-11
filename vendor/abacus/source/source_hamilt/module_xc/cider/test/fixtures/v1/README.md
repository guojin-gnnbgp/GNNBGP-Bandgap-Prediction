# Native CIDER Python-reference oracle v1

This directory freezes a compact numerical oracle produced by the current
Python CiderPress bridge before ABACUS switches to native model loading and
evaluation.  Native tests may read `manifest.json` and the packed raw
`arrays.bin` payload; native runtime code must not depend on Python, NumPy,
CiderPress, or the generator.

The five deterministic 2×2×2 cases jointly cover GGA and MGGA, NLDF and
non-NLDF, one and two spin channels, and valence versus
valence-plus-pseudo-core feature-density policies.  They include all four stock
models in the v1 compatibility corpus plus the accepted NumPy-2-era trained
model, preserving both `numpy.core` and `numpy._core` reconstruction spellings.
Each case records loader
metadata, representative decoded model arrays, raw and normalized features,
transformed descriptors, spline values and derivatives, reverse derivatives,
role-separated dual-density outputs, and volume-integrated results.  Full
decoded coefficient arrays are identified by shape, byte count, and SHA-256;
nine deterministic values per term are committed to keep the suite compact.

`arrays.bin` is a headerless concatenation of C-order raw arrays.  Every
array's explicit byte offset, little-endian dtype, shape, byte count, checksum,
stage, description, and tolerances are given in the manifest.  It is safe to
read without enabling pickle.

Generate from the ABACUS worktree root:

```bash
LD_LIBRARY_PATH=/home/bonan/appdir/CiderPress/ciderpress/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
/home/bonan/appdir/CiderPress/.venv/bin/python3 \
source/source_hamilt/module_xc/cider/test/fixtures/v1/generate_oracle.py \
--parent /home/bonan/appdir/CiderPress
```

Validate using only the Python standard library:

```bash
python3 source/source_hamilt/module_xc/cider/test/fixtures/v1/validate_oracle.py
```

Check byte-for-byte reproducibility against a temporary regeneration:

```bash
LD_LIBRARY_PATH=/home/bonan/appdir/CiderPress/ciderpress/lib \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
/home/bonan/appdir/CiderPress/.venv/bin/python3 \
source/source_hamilt/module_xc/cider/test/fixtures/v1/generate_oracle.py \
--parent /home/bonan/appdir/CiderPress --check
```
