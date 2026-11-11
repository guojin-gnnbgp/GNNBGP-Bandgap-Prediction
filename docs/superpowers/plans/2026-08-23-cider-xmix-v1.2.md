# CIDER Mixing Preset v1.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cider-pbe0` use the validated locked `cider_xmix=0.15` preset and publish reproducible InSb and Si preset-v1.2 baselines.

**Architecture:** Change only the immutable native-CIDER input mapping and its preset metadata; direct HSE06 and PBE0 remain byte-for-byte unchanged at preset v1.1. Exercise the production CLI with the final executable for InSb and Si, then record only values and hashes extracted from retained successful results.

**Tech Stack:** Python 3.12, pytest, uv, ASE, NumPy, ABACUS PW CUDA+MPI, JSON/Markdown release records.

## Global Constraints

- `cider-pbe0` uses exactly `cider_xmix=0.15` and reports `preset_version=1.2`.
- `abacus-hse06` and `abacus-pbe0` stay at `preset_version=1.1` with unchanged golden INPUT files and baselines.
- Keep the original `nldf_mgga_cand00_n96_noise0p01.yaml`, bundled PseudoDojo potentials, valence-only CIDER features, 100 Ry, `kspacing=0.15`, 0.001 eV Gaussian smearing, `nspin=1`, and no SOC.
- Keep the CLI preset locked; do not add a `cider_xmix` override.
- Run the exact supplied `examples/InSb.cif` and `examples/Si.cif` locally, synchronously, with one MPI rank and `OMP_NUM_THREADS=1` outside the sandbox.
- Preserve historical preset-v1.1 directories and historical design/plan documents.
- Work directly on `main`, as explicitly requested for this dedicated workspace.

---

### Task 1: Promote the locked CIDER input preset

**Files:**
- Modify: `tests/abacus/test_inputs.py`
- Modify: `src/bandgap_predictor/abacus/inputs.py`
- Modify: `tests/abacus/golden/cider_INPUT`

**Interfaces:**
- Consumes: `prepare_abacus_input(workdir, structure, assets, AbacusMethod.CIDER_PBE0)`.
- Produces: a prepared CIDER INPUT containing `cider_xmix 0.15` and settings containing `preset_version="1.2"`.

- [ ] **Step 1: Change the CIDER expectations first**

In `test_cider_input_uses_valence_only_pseudodojo_model`, require:

```python
assert prepared.settings["preset_version"] == "1.2"
assert prepared.settings["cider_xmix"] == 0.15
assert "cider_xmix 0.15\n" in prepared.input_path.read_text()
```

Keep the existing HSE06 and direct-PBE0 assertions at preset `1.1`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py::test_cider_input_uses_valence_only_pseudodojo_model -q
```

Expected: FAIL because the current prepared settings report preset `1.1` and `cider_xmix=0.25`.

- [ ] **Step 3: Make the minimal production and golden-input change**

In the immutable CIDER mapping, change only:

```python
"cider_xmix": 0.15,
```

In the CIDER branch of `prepare_abacus_input`, set:

```python
"preset_version": "1.2",
```

Change the golden INPUT line to:

```text
cider_xmix 0.15
```

- [ ] **Step 4: Verify GREEN and unchanged sibling methods**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py -q
```

Expected: all input tests pass; HSE06 and PBE0 retain preset `1.1` and their existing golden files.

- [ ] **Step 5: Commit the locked preset**

```bash
git add src/bandgap_predictor/abacus/inputs.py tests/abacus/test_inputs.py tests/abacus/golden/cider_INPUT
git commit -m "feat: calibrate CIDER mixing preset"
```

---

### Task 2: Generate final InSb and Si CIDER evidence

**Files:**
- Create: `build/release-hardware-runs-cider-v1.2/*/result.json`
- Create: `build/release-hardware-runs-cider-v1.2/*/INPUT`
- Create: `build/release-hardware-runs-cider-v1.2/*/OUT.ABACUS/`

**Interfaces:**
- Consumes: `bgp predict STRUCTURE --backend cider-pbe0 --abacus EXECUTABLE --work-root ROOT`.
- Produces: two retained public result records generated through the production service.

- [ ] **Step 1: Verify the final executable and production input preview**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bgp doctor --abacus build/abacus-install-fixed/bin/abacus_pw_gpu
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py -q
```

Expected: doctor reports the required PW/CUDA/MPI/native-CIDER features, and the focused suite passes.

- [ ] **Step 2: Run InSb outside the sandbox**

Run with GPU/MPI permission:

```bash
OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/InSb.cif --backend cider-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-cider-v1.2
```

Expected: exit code 0, classification `insulator`, finite non-negative gap, retained `result.json`, and generated INPUT containing `cider_xmix 0.15`.

- [ ] **Step 3: Run diamond Si outside the sandbox**

Run with GPU/MPI permission:

```bash
OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/Si.cif --backend cider-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --work-root build/release-hardware-runs-cider-v1.2
```

Expected: exit code 0, classification `insulator`, finite non-negative gap, retained `result.json`, and generated INPUT containing `cider_xmix 0.15`.

- [ ] **Step 4: Validate both public results and provenance**

For each of the exactly two `result.json` files, require:

```python
assert result["status"] == "success"
assert result["converged"] is True
assert result["backend"] == "cider-pbe0"
assert result["settings"]["preset_version"] == "1.2"
assert result["settings"]["cider_xmix"] == 0.15
assert result["prediction"]["classification"] == "insulator"
assert result["prediction"]["gap_ev"] >= 0.0
assert result["provenance"]["executable_sha256"] == "9ad2d03bbb4f6fa8a56384a654f744fd23bd563872555b105b44f7efe544fb2e"
assert result["provenance"]["cider_model_sha256"] == "0bdadb33599a4f5d26baa743b1a5849a1fb63e0ce6b0fdf7ad9479feb2e28734"
```

Also compare each structure and pseudopotential hash with the existing audited system entry and verify `INPUT`, `process.json`, `stdout.log`, and `OUT.ABACUS/bgp_electronic_state.json` exist.

---

### Task 3: Publish preset-v1.2 baselines and documentation

**Files:**
- Modify: `tests/hardware/baselines.json`
- Modify: `docs/methods.md`
- Modify: `docs/results.md`

**Interfaces:**
- Consumes: the two retained Task 2 `result.json` records.
- Produces: release cases whose numerical values and hashes reproduce those records exactly.

- [ ] **Step 1: Extract exact release fields**

For each result, extract its system, gap, classification, structure hash,
executable hash, CIDER model hash, pseudopotential hashes, and the settings
hash computed by the hardware test contract:

```python
sha256(json.dumps(result["settings"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
```

- [ ] **Step 2: Replace only CIDER InSb and Si baseline fields**

Keep each entry `status="success"`; replace `gap_ev`, `classification`, and
`settings_sha256` with the exact Task 2 values. Assert the unchanged executable,
model, pseudopotential, and structure hashes equal the fresh result before
retaining them. Update the notes to state that CIDER uses calibrated
`cider_xmix=0.15` at preset `1.2`, while direct PBE0 retains 25% exact exchange.

- [ ] **Step 3: Update current scientific documentation**

In `docs/methods.md`, describe `cider_xmix=0.15`, remove the statement that
direct PBE0 and CIDER share a 0.25 coefficient, and identify the CIDER value as
an InSb-HSE06 calibration of a learned correction. State the CIDER preset is
`1.2` while the direct hybrid presets remain `1.1`.

In `docs/results.md`, replace the CIDER InSb and Si values with the two fresh
results and describe preset `1.1` CIDER (`cider_xmix=0.25`) as superseded
historical evidence. Leave all direct HSE06/PBE0 and unbaselined GaAs/MgO
records unchanged.

- [ ] **Step 4: Verify baseline structure and focused documentation claims**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/hardware/test_abacus_release_matrix.py::test_required_baseline_contract_is_complete_and_provenanced -q
rg -n "cider_xmix=0.15|preset.*1.2|25% exact" docs/methods.md docs/results.md tests/hardware/baselines.json
```

Expected: contract test passes; current files consistently identify the new
CIDER coefficient/version and retain the direct-PBE0 25% distinction.

- [ ] **Step 5: Commit the baselines and documentation**

```bash
git add tests/hardware/baselines.json docs/methods.md docs/results.md
git commit -m "test: record CIDER preset v1.2 baselines"
```

---

### Task 4: Final verification

**Files:**
- Verify: repository tracked files and retained Task 2 artifacts.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: completion evidence for the calibrated production backend.

- [ ] **Step 1: Run the complete test suite**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected: all non-hardware tests and baseline-contract tests pass; hardware
tests without `BGP_ABACUS` remain skipped.

- [ ] **Step 2: Run both required CIDER hardware cases against baselines**

Use a separate verification root so the baseline test performs fresh runs:

```bash
BGP_ABACUS=build/abacus-install-fixed/bin/abacus_pw_gpu BGP_HARDWARE_RUN_ROOT=build/release-hardware-verification-cider-v1.2 OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/hardware/test_abacus_release_matrix.py -m hardware -k "cider_pbe0 and (InSb or Si)" -q
```

Expected: the InSb and Si `cider-pbe0` hardware cases pass against the new
baselines. If pytest's generated parameter IDs do not match that expression,
run the two exact collected CIDER node IDs shown by `--collect-only`.

- [ ] **Step 3: Audit the final diff and retained evidence**

```bash
git diff --check
git status --short
```

Confirm no historical result directory or historical design document changed,
all committed source/test/docs changes are in scope, and all four fresh Task 2
and verification calculations remain under ignored `build/` paths.
