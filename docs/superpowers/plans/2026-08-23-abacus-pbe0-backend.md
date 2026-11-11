# Direct ABACUS PBE0 Backend Implementation Plan

**Goal:** Expose direct 25%-EXX PBE0 as `abacus-pbe0` and record converged local InSb and Si targets.

**Architecture:** Extend the existing typed backend/method enums and reuse the common ABACUS input, execution, parsing, and provenance template. Direct PBE0 receives its own immutable input mapping and backend class; no CIDER model is staged. The existing release matrix becomes three ABACUS backends.

**Tech stack:** Python 3.11+, Pydantic, Click, ASE, pytest, Ruff, uv, package-owned CUDA/MPI ABACUS.

---

## Task 1: Lock the public and input contracts with failing tests

**Files:**
- Modify: `tests/abacus/test_inputs.py`
- Modify: `tests/abacus/test_backends.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/hardware/test_abacus_release_matrix.py`
- Create: `tests/abacus/golden/pbe0_INPUT`

- [ ] Add a direct-PBE0 input test that requests `AbacusMethod.PBE0`, compares the rendered file with `pbe0_INPUT`, asserts preset v1.1 and 0.001 eV smearing, asserts `dft_functional == "PBE0"` plus ACE/tiling settings, and asserts no model is staged.
- [ ] Add `AbacusPBE0Backend`, `BackendId.ABACUS_PBE0`, method `PBE0`, and functional `PBE0` to the adapter parametrization; widen the test class annotation.
- [ ] Add a CLI test asserting `bgp backends` prints all four stable backend IDs.
- [ ] Extend `ABACUS_BACKENDS` with `BackendId.ABACUS_PBE0` so the baseline contract requires InSb and Si entries.
- [ ] Run `uv run pytest tests/abacus/test_inputs.py tests/abacus/test_backends.py tests/test_cli.py tests/hardware/test_abacus_release_matrix.py -m 'not hardware' -q` and confirm collection/import fails because direct PBE0 does not yet exist.

## Task 2: Implement the minimal direct-PBE0 backend

**Files:**
- Modify: `src/bandgap_predictor/schemas.py`
- Modify: `src/bandgap_predictor/abacus/inputs.py`
- Modify: `src/bandgap_predictor/backends/abacus.py`
- Modify: `src/bandgap_predictor/factory.py`
- Modify: `src/bandgap_predictor/cli.py`

- [ ] Add `ABACUS_PBE0 = "abacus-pbe0"` and `PBE0 = "pbe0"` to the corresponding enums.
- [ ] Add immutable `PBE0 = {**COMMON, "dft_functional": "PBE0", "exxace": 1, "exx_symmetry_realspace": 0, "exx_auto_tiling": 1, "exx_tile_memory_budget_mb": 0}` and prepare it without a model.
- [ ] Add `AbacusPBE0Backend` with backend ID `ABACUS_PBE0`, public method `PBE0`, and method kind `PBE0`.
- [ ] Register `AbacusPBE0Backend()` in `create_service()` and print its availability in `bgp backends`.
- [ ] Run the focused command from Task 1 and confirm all selected non-hardware tests pass.
- [ ] Run `uv run pytest -m 'not hardware' -q` and `uv run ruff check .`.
- [ ] Commit the backend implementation.

## Task 3: Generate direct-PBE0 InSb and Si targets

**Files:**
- Create retained artifacts below: `build/release-hardware-runs-pbe0-v1.1/`
- Modify: `tests/hardware/baselines.json`

- [ ] Verify `build/abacus-install-fixed/bin/abacus_pw_gpu` and its build manifest with `uv run bgp doctor --abacus build/abacus-install-fixed/bin/abacus_pw_gpu`.
- [ ] Outside the sandbox, run `OMP_NUM_THREADS=1 uv run bgp predict examples/InSb.cif --backend abacus-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-pbe0-v1.1` and retain the completed run.
- [ ] Run the corresponding Si command with the same executable and work root.
- [ ] Inspect each `result.json`; require status `success`, `converged=true`, method `PBE0`, the locked settings, and matching executable/pseudopotential hashes.
- [ ] Add exact gaps, classifications, tolerances, hashes, and settings hashes to the `abacus-pbe0` baseline section; mark GaAs and MgO explicitly `not_baselined`.

## Task 4: Document and release-verify the PBE0 targets

**Files:**
- Modify: `README.md`
- Modify: `docs/methods.md`
- Modify: `docs/results.md`

- [ ] Describe four public methods and state that direct PBE0 uses the standard 25% exact-exchange fraction while CIDER-PBE0 uses `cider_xmix=0.25` as a separate learned approximation.
- [ ] Add the converged InSb and Si direct-PBE0 values to the results table and preserve the no-SOC caveat.
- [ ] Run `uv run pytest -m 'not hardware' -q`, `uv run ruff check .`, and `uv run pytest tests/hardware/test_abacus_release_matrix.py -m 'not hardware' -q`.
- [ ] Run `uv build`, install the wheel into a clean temporary virtual environment, verify `bgp backends`, `bgp assets verify`, and `bgp doctor`, and confirm the source tree is clean except for intended changes.
- [ ] Commit the baseline and documentation update on `main`.

