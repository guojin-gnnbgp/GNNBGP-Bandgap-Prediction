# Native CIDER validation record

This record describes the CPU numerical validation performed on 2026-08-05,
the native-only release validation completed through commit `05c8d1295`, and
the same-binary native/external validation implemented through dual-backend
commit `8378776de`.
Runtime cases were copied to `/tmp`; none of the
examples or logs in the original `abacus-cider` checkout were modified.

The [native architecture and migration record](cider_native_architecture.md),
[normative YAML profile](cider_legacy_yaml_compatibility.md),
[model-authoring and troubleshooting guide](cider_model_authoring_and_troubleshooting.md),
and [build guide](cider_build.md) define the corresponding user-facing
contracts. This file records evidence and does not broaden model support.

## Dual-backend same-binary gate

A dual build with `ENABLE_CIDER_EXTERNAL=ON` was tested with the same copied Si
NLDF GGA input and model, changing only `cider_backend`. The model SHA-256 was
`b4e29d9530eaa7c94a3c61cc5e942a0cf7cfd0dd3c4a265dc35395123ae26c86`.
Omitting the selector (`auto`) chose external, while explicit native execution
produced no bridge or Python initialization output.

| Profile | Native first-iteration total (eV) | External first-iteration total (eV) | Absolute difference (eV) |
| --- | ---: | ---: | ---: |
| Serial CPU | `-215.4821214746584` | `-215.4821214746583` | `1e-13` |
| MPI, 1 rank | `-215.4821214746584` | `-215.4821214746583` | `1e-13` |
| MPI, 2 ranks | `-215.4821919554440` | `-215.4821919554439` | `1e-13` |
| CUDA, 1 rank | `-215.4821214746584` | `-215.4821214746584` | `0` at printed precision |

The one- and two-rank values differ because the intentionally unconverged
one-iteration cases use rank-dependent random wavefunction initialization.
Backend parity is compared within each rank count. The CUDA binary was built
for architectures `86;89;120` with CUDA 12.8.93 and ran on an RTX 5090.

The direct dual-runtime test evaluated all five native-compatibility artifacts:
the four stock SL/NL GGA/MGGA models and the project-trained NumPy-2 NLDF MGGA
artifact. Across role-separated output arrays, the maximum absolute
native/external error was `8.0491169285323849e-16` and the maximum relative
error was `2.0347501035505874e-14`. External construction also accepted the
out-of-native-profile `CIDER24Xe` model, while explicit native construction
failed deterministically with `UNSUPPORTED_EVALUATOR`.

The dual serial, MPI, and CUDA artifacts intentionally linked
`libciderbridge.so` and `libpython3.10.so.1.0`; the CUDA artifact additionally
linked the CUDA 12 runtime libraries. Native-only artifact guards continued to
reject Python and bridge dependencies. The dual builds used
`ENABLE_HDF5=OFF`: with ABACUS system HDF5 1.10 preloaded, the environment's
`h5py` wheel built for HDF5 2.0 failed during PySCF import. Disabling ABACUS
HDF5 removed that process-wide ABI collision without changing CIDER results.

## Native-only release build and runtime gate

Fresh native-only Release PW serial CPU, MPI CPU, and CUDA builds enabled LibXC
and native CIDER at commit `e1022440b`. The CPU targets were `abacus_pw_ser` and
`abacus_pw_para`. CUDA 12.8.93 compiled `abacus_pw_gpu` for local architecture
120; the installed executable then completed the semilocal-GGA smoke on an
RTX 5090. This local binary is not a heterogeneous deployment binary: the
project's cata deployment profile must use architectures `86;89;120`.

All three caches, final link commands, `ldd` output, and ELF dynamic sections
were checked. The built and installed executables have no Python, NumPy,
CiderPress, or `libciderbridge` dependency. The site-specific yaml-cpp library
was linked statically. Serial and MPI install prefixes contain only the
executable plus the relative `bin/abacus` symlink; native CIDER requires no
separate installed runtime library.

The installed serial executable was run from copied `/tmp` cases using
`env -i`, empty `PYTHONPATH`, empty `VIRTUAL_ENV`, empty `LD_LIBRARY_PATH`, and
one OpenMP/BLAS thread. It completed one SCF iteration for both the stock SL
GGA and NLDF GGA artifacts. Peak RSS was 84,352 KiB and 139,276 KiB,
respectively in the initial run. A separately retained SL audit run recorded
85,376 KiB in `/tmp/task17-native-sl-retained.time` and its native runtime log
under `/tmp/task17-isolated/sl-native-retained`. The NLDF ABACUS timer reported
0.27 s per `PotCiderXC::cal_v_eff` call; the direct timings below isolate both
SL and NLDF evaluation more precisely.

A direct native evaluator harness on the same `36^3` grid measured SL model
load at 0.0022 s, ten evaluations at 0.1467 s (0.0147 s mean), and 17,860 KiB
peak RSS. NLDF model/evaluator construction took 0.2209 s, ten evaluations took
2.4762 s (0.2476 s mean), and peak RSS was 71,832 KiB. These harness numbers
isolate CIDER from diagonalization and other ABACUS allocations.

For NLDF GGA, one MPI rank used 155,244 KiB peak RSS and 0.27 s per CIDER call.
Two ranks used 128,336 and 127,260 KiB peak RSS (255,596 KiB total) and 0.28 s
per call. Total application wall time improved from 4.88 s to about 3.40 s due
to the PW workload, but replicated CIDER evaluation did not speed up and total
resident memory increased by about 65 percent. For this spin-one GGA `36^3`
case, adapter-owned global packets account for about 3.56 MiB per rank, before
the model, evaluator, FFT workspace, and transient gather storage.

The original HDF5-enabled role-separated bridge executable could not create a
model because of the HDF5/h5py ABI collision described above. The later
HDF5-disabled dual executables establish numerical and runtime parity, but no
controlled native-versus-external wall-time or peak-memory benchmark was
recorded. The 20 percent performance gate therefore remains **unavailable**,
not passed or failed. The April combined-convention logs documented below are
not a compatible performance or numerical comparator.

Fresh installed-binary failures for `.joblib`, a `MappedXC2` YAML root, and
malformed YAML all exited with status 1 during `Input_Conv`, before
`PotCiderXC` construction or `INIT SCF`. Each diagnostic included model path,
native error code, YAML path, and reason (`JOBLIB_UNSUPPORTED`,
`UNSUPPORTED_ROOT`, and `YAML_SYNTAX`, respectively).

## Build profiles

Fresh PW-only serial and MPI builds used the installed LibXC 5.1.7, FFTW, and
an already available static yaml-cpp package.  No dependency was downloaded.
The site yaml-cpp package uses the `YAML_PACE` namespace, hence the namespace
definition below; a standard yaml-cpp installation does not need it.

```bash
cmake -S . -B /tmp/abacus-cider-task16-serial \
  -DENABLE_MPI=OFF -DENABLE_LCAO=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF \
  -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -Dyaml-cpp_DIR=/tmp/yaml-cpp-local -DBUILD_TESTING=OFF \
  -DGIT_SUBMODULE=OFF -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_CXX_FLAGS=-DCIDER_YAML_CPP_NAMESPACE=YAML_PACE \
  -DCMAKE_PROJECT_INCLUDE=/tmp/abacus-task3-serial-test-workaround.cmake
cmake --build /tmp/abacus-cider-task16-serial --target abacus_pw_ser -j8

cmake -S . -B /tmp/abacus-cider-task16-mpi-out \
  -DENABLE_MPI=ON -DENABLE_LCAO=OFF -DUSE_CUDA=OFF -DUSE_ROCM=OFF \
  -DENABLE_LIBXC=ON -DENABLE_CIDER=ON \
  -Dyaml-cpp_DIR=/tmp/yaml-cpp-local -DBUILD_TESTING=OFF \
  -DGIT_SUBMODULE=OFF -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_CXX_FLAGS=-DCIDER_YAML_CPP_NAMESPACE=YAML_PACE \
  -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx
cmake --build /tmp/abacus-cider-task16-mpi-out --target abacus_pw_para -j8
```

The resulting targets were `abacus_pw_ser` and `abacus_pw_para` (OpenMPI
4.1.2).  Their caches contained `ENABLE_CIDER=ON`, `ENABLE_LIBXC=ON`, and the
intended MPI settings.  Searching both caches for `CIDERPRESS`,
`CIDERBRIDGE`, `ciderbridge`, `Python3`, and `NumPy` returned no match.  Their
dynamic dependency lists likewise contained no Python, NumPy, CiderPress, or
`libciderbridge` library.

## Analytic derivatives and frozen-oracle parity

All commands used one OpenMP and one BLAS thread.  The complete evaluator test
was run with the read-only five-model corpus enabled:

```bash
CIDER_NATIVE_CORPUS_ROOT=/home/bonan/appdir/CiderPress \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
/tmp/abacus-cider-task12-focused/source/source_hamilt/module_xc/cider/test/\
MODULE_HAMILT_CIDER_evaluator_test
```

All 11 tests passed with zero skips.  This covers spin one and two, GGA and
MGGA, semilocal and NLDF, all accepted density policies, and all five frozen
models.  The maximum absolute errors reported by the tests were:

| Check | Maximum absolute error |
| --- | ---: |
| Full physical feature tuple (`vrho`, same-spin `vsigma`, `vtau`) | `2.904532517278291e-11` |
| Semilocal GGA reverse | `1.5947750064970734e-11` |
| Semilocal MGGA reverse | `4.7508497136306005e-11` |
| NLDF reverse finite difference | `9.237166587183765e-12` |
| NLDF adjoint dot product | `3.2845413888704655e-11` |
| Mapped raw-feature reverse | `3.886491128923808e-11` |
| Five-model dual evaluator versus frozen oracle | `8.326672684688674e-16` |
| Five-model integrated quantities versus frozen oracle | `3.552713678800501e-15` |

The physical-tuple check is a spin-two MGGA calculation and perturbs every
density, all three sigma channels, and every tau entry.  It separately verifies
that the two cross-spin sigma derivatives are exactly zero.

## ABACUS model matrix

The established Si structure, 12x12x12 K-point mesh, 50 Ry cutoff, and ONCV
pseudopotential were copied to `/tmp`.  Absolute read-only model and
pseudopotential paths were used.  One SCF iteration was deliberately requested
for the broad smoke matrix; these cases validate model loading, metadata-driven
MGGA setup, evaluation, potential construction, and finite output, but are not
convergence claims.

| Model | Native metadata | First-iteration total (eV) | Result |
| --- | --- | ---: | --- |
| `CIDER23X_SL_GGA.yaml` | GGA, valence | `-214.4432227910937` | completed, finite |
| `CIDER23X_NL_GGA.yaml` | GGA, valence+pseudo-core | `-215.4821214746584` | completed, finite |
| `CIDER23X_SL_MGGA.yaml` | MGGA, valence | `-215.2017944604347` | completed, finite `vtau` path |
| `CIDER23X_NL_MGGA_DTR.yaml` | NLDF+MGGA, valence+pseudo-core | `-214.6041954065961` | completed, finite `vtau` path |

Spin-one end-to-end coverage is provided by these ABACUS cases.  Spin-one and
spin-two coverage for all four functional families is provided by the frozen
evaluator suite above.

## One-rank versus two-rank MPI

The Si NLDF GGA case was run outside the sandbox with
`OMP_NUM_THREADS=1`.  At the example's original `scf_thr=1e-7`, both ranks
converged in five iterations, but their final energies differed by
`3.87447e-8 eV` (`1.42e-9 Ha`).  That run therefore does **not** meet a
`1e-10 Ha` comparison threshold; rank-dependent random wavefunction
initialization remained visible at the loose SCF threshold.

The same copied input was repeated with `scf_thr=1e-11`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 mpirun -n 1 \
  /tmp/abacus-cider-task16-mpi-out/abacus_pw_para
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 mpirun -n 2 \
  /tmp/abacus-cider-task16-mpi-out/abacus_pw_para
```

Both runs used a `36x36x36` grid and converged in eight iterations.  The final
energies were `-215.5397905425412 eV` and `-215.5397905432052 eV`.  Their
absolute difference is `6.64016e-10 eV`, or `2.44022e-11 Ha`, satisfying the
`1e-10 Ha` threshold.  The final printed baseline, feature, total CIDER energy,
and potential diagnostics also agree at their printed precision.

First-call binary debug dumps used rank-qualified filenames to avoid a
multi-rank file collision.  The two `36x36x18` rank-local slabs were
reconstructed along the distributed z dimension and compared with the
one-rank `36x36x36` fields.  Maximum absolute differences were `2.23e-15` for
rho, `3.30e-16` for sigma, `2.94e-15` for baseline energy density,
`1.26e-15` for feature energy density, `1.75e-13` for feature `vrho`,
`3.66e-12` for feature `vsigma`, and `4.38e-13` for the final total local
potential.  The globally reduced baseline and feature energies differed by at
most `1.89e-13 Ry` and `8.08e-14 Ry`, respectively.  Every returned field and
global scalar therefore satisfies the `1e-10` comparison threshold.

## Current Python-oracle replay on an ABACUS grid

Python is used here only as an external validation oracle; the native-only
ABACUS binaries in this section neither load nor link it. A first-call debug
dump from the native Si semilocal-GGA calculation was loaded by the existing read-only
`abacus_training_docs/compare_abacus_cider_dump.py` helper.  The exact density
and sigma arrays were replayed through the current CiderPress evaluator.

| Array | Maximum absolute native/Python difference |
| --- | ---: |
| feature energy density | `3.469446951953614e-18` |
| feature `vrho` | `1.6653345369377348e-16` |
| feature `vsigma` | `1.1511624986582092e-14` |

The native and Python feature sums were `-87.53010354384654` and
`-87.53010354384656`.  The native stored feature energy and the independently
reintegrated dump differed by `2.8e-15 Ry`.  This is a like-for-like current
oracle check on an actual ABACUS `36x36x36` density grid.

## Historical bridge controls

The April bridge controls and current native calculations use byte-identical
YAML models and pseudopotentials.  The model SHA-256 values are
`fdd60d58ae0f981dcdf64ec24437454bdc334e65b162e379e3630bb99dc82bba`
(SL MGGA),
`93312ddde97a8b8e88a9f875df221a44f65a1a1b7c955a55d9214bd449e363ea`
(NLDF MGGA), and
`b4e29d9530eaa7c94a3c61cc5e942a0cf7cfd0dd3c4a265dc35395123ae26c86`
(NLDF GGA).  With the same copied `INPUT_PBE`, `STRU`, `KPT`, and
pseudopotential bytes, the current plain-PBE GaAs result
(`-4967.825660012794 eV`) reproduces the April plain-PBE result
(`-4967.825660012468 eV`) to `3.2651e-10 eV` (`1.20e-11 Ha`).  This excludes a
generic ABACUS, input, or pseudopotential drift.

The CIDER totals nevertheless differ substantially:

| Control | April bridge (eV) | Current native (eV) | Native minus bridge (Ha) |
| --- | ---: | ---: | ---: |
| GaAs SL MGGA | `-6360.048242657146` | `-4775.188433010001` | `+58.24252374797` |
| GaAs NLDF MGGA | `-6490.710572634332` | `-5047.041173662813` | `+53.05387185794` |
| Si NLDF GGA | `-200.824657400291` | `-215.5397884554181` | `-0.54077109200` |

This historical threshold is **not passed**.  Source inspection corrects an
initial diagnostic hypothesis: ABACUS commit `49debf3ca` already used
`MPI_Allgatherv` to construct full global rho, sigma, and tau buffers and
called `cider_bridge_evaluate` with `nxyz`.  The logged `nrxx=46575` is only
rank-local storage for scattering and potential construction, not the bridge
evaluation extent.

The demonstrated difference is the old combined-energy convention.  The
April bridge implementation formed one array as
`exc_total = baseline_per_particle + feature_energy_density`, while the C++
adapter treated the entire combined array as per-particle and integrated
`2 * dV * sum(exc_total * rho_total)`.  It also supplied the same
valence+pseudo-core tuple to both baseline and feature evaluation, including
for the SL model.  The native adapter instead integrates baseline and feature
according to their separate conventions and uses valence features for the SL
model.

This was checked quantitatively using the exact current first-call GaAs rho,
sigma, and tau arrays and the last committed bridge implementation before the
April logs (`CiderPress` commit `42f0d99`).  Replaying the historical combined
bridge and its C++ rho-weighted integral produced `-187.79639239282648 Ry`
(SL MGGA) and `-194.09406772237293 Ry` (NLDF MGGA), reproducing the April
call-1 log values `-187.796 Ry` and `-194.094 Ry` at their printed precision.
The role-separated native values on the same first-call state are
`-77.50205163548571 Ry` and `-97.18581149502602 Ry`.  This proves the source of
the initial SCF-path divergence; after that first potential, the densities
and final SCF energies are no longer fixed-density quantities that can be
decomposed into one convention-only scalar correction.

The current-grid Python replay above independently verifies the native feature
arrays.  The April totals therefore remain useful records of the historical
combined convention, but they are not a strict parity target for the accepted
role-separated convention.  Promotion evidence must use the frozen v1 oracle
and current-grid replay rather than claim agreement with those totals.

## Post-SCF CIDER label and training schema

A converged Si PBE calculation used `cider_scf 0`, `out_cider_label 1`, and
`out_training_data 1`.  After SCF, native CIDER evaluated the label and wrote
`training_dump/step_0.h5`.  Validation with h5py found:

- HDF5 container schema `abacus_pw_nldx_training_dump_hdf5`, version 1;
- record schema `abacus_pw_nldx_training_dump`, version 2;
- label mode `pbe_base_state_post_scf_cider_label`;
- 13 finite datasets and `converged=true`;
- baseline, scaled feature, unmixed feature, and total energies of
  `-1.881610792546196`, `-0.5261198262443983`, `-2.104479304977593`, and
  `-2.407730618790594 Ha`.

The exact identities `feature_scaled = xmix * feature_unmixed` and
`total = baseline + feature_scaled` both passed at `1e-14 Ha`.  This preserves
the post-SCF label meaning, exchange scaling, and training-dump schema without
embedding Python in ABACUS.
