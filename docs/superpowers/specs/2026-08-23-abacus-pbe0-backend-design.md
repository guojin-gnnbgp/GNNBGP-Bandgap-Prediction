# Direct ABACUS PBE0 Backend Design

## Objective

Add `abacus-pbe0` as a public fourth backend and generate reproducible direct-PBE0 reference targets for the supplied InSb and diamond-Si structures. This backend is the first-principles PBE0 reference and remains scientifically distinct from `cider-pbe0`.

## Scientific contract

Direct PBE0 uses ABACUS `dft_functional PBE0`, whose exact-exchange fraction is the standard 25%. It uses the same locked preset v1.1 as the existing ABACUS methods: bundled PseudoDojo pseudopotentials, 100 Ry, `kspacing=0.15`, double precision, Gaussian `smearing_sigma=0.001 eV` (`7.349864435130999e-05 Ry`), `nspin=1`, no SOC, one MPI rank, and one OpenMP thread. ACE, disabled real-space EXX symmetry, and automatic GPU EXX tiling match the HSE06 GPU path.

`cider-pbe0` remains unchanged. Its PBE base and `cider_xmix=0.25` describe the learned CIDER correction; its results are not relabelled as direct PBE0.

The submitted CIF cell, Cartesian positions, chemical identities, and site order remain exactly supplied. The package does not relax, wrap, reduce, standardize, or otherwise transform the structures.

## Package integration

Add `ABACUS_PBE0 = "abacus-pbe0"` to the stable backend schema and `PBE0 = "pbe0"` to the internal ABACUS method enum. Input preparation selects a locked direct-PBE0 mapping and does not stage a CIDER model. A new `AbacusPBE0Backend` reuses the shared synchronous runner, parser, retained-artifact behavior, executable resolution, and provenance collection. The service factory and `bgp backends` expose the backend alongside the existing three.

## Inputs and provenance

The golden `INPUT` contract includes `dft_functional PBE0`, `exxace 1`, `exx_symmetry_realspace 0`, `exx_auto_tiling 1`, and `exx_tile_memory_budget_mb 0`. Results record method `PBE0`, the complete locked settings, executable/source/toolchain metadata, and PseudoDojo hashes. They do not contain a CIDER model hash.

## Verification and reference targets

Tests first establish the missing backend, preset, adapter, factory/CLI visibility, and hardware-baseline contract. After implementation passes non-hardware tests and lint, run synchronous local GPU/MPI calculations with `OMP_NUM_THREADS=1` using the existing package-built `abacus_pw_gpu`. Retain new calculation directories under `build/release-hardware-runs-pbe0-v1.1` and record only converged parsed InSb and Si results in `tests/hardware/baselines.json`. Update user documentation to compare direct PBE0 with HSE06 and CIDER-PBE0 without implying interchangeability.

