# GaAs 2x2x2 HSE symmetry verification

## Context

- ABACUS commit: `0b7dfb0ab`
- Executable: `build-cuda-pw-gpu-cider/abacus_pw_gpu`
- Device: NVIDIA GeForce RTX 5090, one MPI process, `OMP_NUM_THREADS=1`
- Primitive GaAs cell, Gamma-centered `2x2x2` mesh
- `ecutwfc=60`, `nbands=10`, LibXC HSE
- `exxace=1`, `exx_separate_loop=1`, `exx_thr_type=energy`, `exx_ene_thr=1e-6`
- `exx_batch_fft_size=1`, `exx_band_tile_size=4`, `exx_q_tile_size=4`

The three inputs are identical after removing `suffix` and `symmetry`.

## Results

| Symmetry | Reduced k | Full q | E_exx (eV) | Total energy (eV) | External wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `-1` | 8 | 8 | -48.3696954400 | -1944.990717627722 | 10.25 |
| `0` | 8 | 8 | -48.3696954400 | -1944.990717627710 | 9.63 |
| `1` | 3 | 8 | -48.3696956751 | -1944.990717307951 | 5.21 |

Relative to `symmetry=-1`:

- `symmetry=0`: total-energy delta `+1.21e-11 eV`; EXX delta `0 eV`.
- `symmetry=1`: total-energy delta `+3.20e-7 eV`; EXX delta `-2.35e-7 eV`.

All three calculations printed `#SCF IS CONVERGED#`. No fatal, segmentation,
CUDA, invalid-argument, NaN, or nonconvergence diagnostic was present. The GPU
returned to 41 MiB used and 0% utilization after completion.
