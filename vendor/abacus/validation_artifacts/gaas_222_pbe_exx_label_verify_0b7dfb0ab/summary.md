# GaAs 2x2x2 PBE-to-EXX-label symmetry verification

## Context

- ABACUS commit: `0b7dfb0ab`
- NVIDIA GeForce RTX 5090, one MPI process, `OMP_NUM_THREADS=1`
- Primitive GaAs, Gamma-centered `2x2x2`, `ecutwfc=60`, `nbands=10`
- PBE SCF with `scf_thr=1e-10` and `pw_diag_thr=1e-10`
- `out_training_data=1`, `out_exx_label=1`
- One-shot full-range, unscreened EXX label on the converged PBE state

The symmetry-off and symmetry-on inputs are identical after removing `suffix`
and `symmetry`.

## Results

| Case | PBE k-points | Full q | EXX-label policy | EXX label (Ha) | PBE total energy (eV) |
| --- | ---: | ---: | --- | ---: | ---: |
| symmetry `-1` | 8 | 8 | pair-map | -8.283294844593076 | -1956.685528744308 |
| symmetry `1` | 3 | 8 | pair-map | -8.283295274625969 | -1956.685528744551 |
| symmetry `1` diagnostic | 3 | 8 | full-active | -8.283295274625971 | -1956.685528744551 |

- Reduced pair-map minus full-k label: `-4.30032893e-7 Ha`, or
  `-1.17017911e-5 eV`.
- Reduced full-active minus reduced pair-map: about `-2e-15 Ha`.
- PBE total-energy difference, symmetry on minus off: `-2.43e-10 eV`.

All calculations converged. The generated HDF5 records identify the label as
`pbe_base_state_one_shot_full_exx_label`, with source
`one_shot_full_range_exx_on_pbe_state`. No fatal, CUDA, NaN, or nonconvergence
diagnostic was present.

This run validates the scalar `out_exx_label` energy. It does not separately
validate the optional `out_bandgap_label` VBM/CBM diagonal labels.
