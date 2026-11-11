# Native version-J NLDF provenance

The first native backend is the accepted serial, replicated full-grid
version-J implementation.  It was ported against CiderPress commit
`6b2945ef9aea1494396e15b6f61389821d0e8549`, which is also the commit recorded
by the frozen v1 oracle manifest.

The following minimal C routines retain the upstream CiderPress copyright and
GPL-3-or-later notice and use the `abacus_cider_nldf_` prefix to coexist with
the legacy bridge libraries:

- `nldf_interpolation.c`: `cider_ind_etb`, `cider_ind_clip`, and
  `cider_coefs_spline_gq`, migrated from
  `ciderpress/lib/mod_cider/cider_coefs.c`.
- `nldf_convolution.c`: the Gaussian reciprocal-kernel evaluation from
  `ciderpw_compute_kernels_helper`, migrated from
  `ciderpress/lib/pwutil/nldf_fft_core.c`.

The typed C++ orchestration in `nldf_plan.cpp` follows
`ciderpress/dft/plans.py` (`NLDFAuxiliaryPlan` and `NLDFSplinePlan`) and
`ciderpress/dft/settings.py` (`get_cider_exponent` and
`get_cider_exponent_gga`).  The serial FFT storage, reciprocal-vector
association, unnormalized FFT scaling, and symmetric forward/backward
convolution in `nldf_backend.cpp` follow
`ciderpress/lib/pwutil/nldf_fft_serial.c` and
`ciderpress/lib/pwutil/nldf_fft_core.c`.

Only the squared-exponential (`se`), density-multiplier-one subset accepted by
the native model loader is included.  No GPAW, Python extension, NumPy,
ctypes, MPI, GPU, l=0/l=1, or other historical NLDF machinery is migrated.
