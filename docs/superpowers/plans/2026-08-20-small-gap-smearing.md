# Small-Gap Smearing v1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate ABACUS inputs with exactly `0.001 eV` Gaussian smearing and replace required hardware baselines with converged v1.1 calculations.

**Architecture:** The package keeps the requested width in electron volts, converts it once to the Rydberg value required by ABACUS, and records both values with preset version `1.1`. Existing v1.0 artifacts remain immutable; only newly converged v1.1 results may update tracked baselines.

**Tech Stack:** Python 3.12, pytest, uv, Click, Pydantic, ABACUS PW CUDA+MPI, JSON release baselines.

## Global Constraints

- The requested Gaussian width is exactly `0.001 eV`.
- Use `1 Ry = 13.605693122994 eV`; generated ABACUS input is `7.349864435130999e-05 Ry`.
- Both `abacus-hse06` and `cider-pbe0` use the same width and preset version `1.1`.
- Do not alter existing v1.0 calculation directories or result JSON files.
- Run all GPU/MPI calculations outside the sandbox with one MPI rank and `OMP_NUM_THREADS=1`.
- Do not change any other locked scientific parameter.

---

### Task 1: Convert and version the locked preset

**Files:**
- Modify: `tests/abacus/test_inputs.py`
- Modify: `tests/abacus/golden/hse06_INPUT`
- Modify: `tests/abacus/golden/cider_INPUT`
- Modify: `src/bandgap_predictor/abacus/inputs.py`
- Modify: `docs/methods.md`

**Interfaces:**
- Consumes: `prepare_abacus_input(...) -> PreparedAbacusInput`
- Produces: `SMEARING_SIGMA_EV`, `SMEARING_SIGMA_RY`, and preset `1.1` settings.

- [x] **Step 1: Write the failing exact-unit tests**

Add assertions for both methods:

```python
assert prepared.settings["preset_version"] == "1.1"
assert prepared.settings["smearing_sigma_ev"] == 0.001
assert prepared.settings["smearing_sigma"] == pytest.approx(
    0.001 / 13.605693122994, rel=0.0, abs=1e-18
)
assert "smearing_sigma 7.349864435130999e-05\n" in prepared.input_path.read_text()
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py -q`

Expected: FAIL because the current preset is `1.0`, has no `smearing_sigma_ev`, and renders `0.015`.

- [x] **Step 3: Implement the minimal conversion and provenance bump**

In `inputs.py`, define:

```python
RYDBERG_TO_EV = 13.605693122994
SMEARING_SIGMA_EV = 0.001
SMEARING_SIGMA_RY = SMEARING_SIGMA_EV / RYDBERG_TO_EV
```

Use `SMEARING_SIGMA_RY` in `COMMON`, set both method settings to preset `1.1`, and add `smearing_sigma_ev` only to the public settings record so it is not emitted as an unknown ABACUS keyword. Update both golden files to the exact rendered Rydberg value and document both units.

- [x] **Step 4: Verify GREEN and the broader fast suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py -q`

Expected: all focused tests PASS.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m 'not hardware'`

Expected: all non-hardware tests PASS.

- [x] **Step 5: Commit the preset change**

```bash
git add src/bandgap_predictor/abacus/inputs.py tests/abacus docs/methods.md
git commit -m "fix: use narrow-gap abacus smearing"
```

### Task 2: Regenerate required ABACUS release evidence

**Files:**
- Modify: `tests/hardware/baselines.json`
- Modify: `docs/results.md`

**Interfaces:**
- Consumes: final package-built `build/abacus-install-fixed/bin/abacus_pw_gpu` and preset `1.1`.
- Produces: four retained result directories and exact v1.1 numerical baselines.

- [x] **Step 1: Verify the executable before computation**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bgp doctor --abacus build/abacus-install-fixed/bin/abacus_pw_gpu`

Expected: executable SHA-256 `9ad2d03bbb4f6fa8a56384a654f744fd23bd563872555b105b44f7efe544fb2e` with PW, CUDA, MPI, LibXC, native-CIDER, GPU-EXX, and result-JSON features.

- [x] **Step 2: Run InSb and Si HSE06 outside the sandbox**

```bash
env OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/InSb.cif --backend abacus-hse06 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-smearing-v1.1
env OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/Si.cif --backend abacus-hse06 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-smearing-v1.1
```

Expected: both return status `success`, `converged=true`, preset `1.1`, and the final executable hash.

- [x] **Step 3: Run InSb and Si CIDER-PBE0 outside the sandbox**

```bash
env OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/InSb.cif --backend cider-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-smearing-v1.1
env OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/Si.cif --backend cider-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-smearing-v1.1
```

Expected: both return status `success`, `converged=true`, preset `1.1`, and exact CIDER/model/pseudopotential provenance.

- [x] **Step 4: Replace baselines only from the four converged JSON records**

For each result, copy `gap_ev`, `classification`, structure, executable, model, pseudopotential, and settings hashes into `tests/hardware/baselines.json`. Update `docs/results.md` with the new values and state that the prior v1.0 values are historical.

- [x] **Step 5: Verify the recorded hardware contract and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/hardware/test_abacus_release_matrix.py::test_required_baseline_contract_is_complete_and_provenanced -q`

Expected: PASS.

```bash
git add tests/hardware/baselines.json docs/results.md
git commit -m "test: refresh narrow-gap hardware baselines"
```

### Task 3: Complete post-change release verification

**Files:**
- Modify: `docs/superpowers/plans/2026-08-20-small-gap-smearing.md`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: clean, verified main branch.

- [x] **Step 1: Run tests and lint**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m 'not hardware'`

Expected: all tests PASS.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check .`

Expected: `All checks passed!`

- [x] **Step 2: Build and validate distribution artifacts**

Run outside the network sandbox: `env UV_CACHE_DIR=/tmp/uv-cache uv build`

Expected: wheel and sdist build successfully.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_wheel.py -q`

Expected: PASS.

- [x] **Step 3: Verify public diagnostics**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bgp assets verify
UV_CACHE_DIR=/tmp/uv-cache BGP_ABACUS=build/abacus-install-fixed/bin/abacus_pw_gpu uv run bgp backends
UV_CACHE_DIR=/tmp/uv-cache uv run bgp doctor --abacus build/abacus-install-fixed/bin/abacus_pw_gpu
```

Expected: 74 assets verified and all three backends available.

- [x] **Step 4: Review and commit the completed plan**

Run: `git diff --check && git status --short`

Expected: no whitespace error and only the checked-off plan remains modified.

```bash
git add docs/superpowers/plans/2026-08-20-small-gap-smearing.md
git commit -m "docs: complete narrow-gap smearing rollout"
```
