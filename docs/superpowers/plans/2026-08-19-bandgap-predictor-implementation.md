# Bandgap Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the `bandgap-predictor` distribution and `bgp` CLI with GNN-HSE06, GPU ABACUS-HSE06, and native CIDER-PBE0 backends operating on unchanged local crystal structures.

**Architecture:** A typed `PredictionService` owns validation, backend dispatch, persistent run records, and result serialization. Backend adapters share immutable asset verification and, for ABACUS, a locked input generator, local MPI runner, and occupation-aware parser; the CLI is only a presentation layer. ABACUS source and scientific assets are copied into this repository with hashes and provenance, while compilation remains an explicit separate installation step.

**Tech Stack:** Python 3.11+, uv, hatchling, Click, Pydantic 2, ASE, NumPy, pytest; optional GNN inference dependencies PyTorch, MACE, e3nn, and Monty; CMake/C++17, CUDA, OpenMPI, LibXC, yaml-cpp, FFTW/BLAS/LAPACK for ABACUS.

## Global Constraints

- Distribution name is `bandgap-predictor`; command name is `bgp`.
- Backend IDs are exactly `gnn-hse06`, `abacus-hse06`, and `cider-pbe0`.
- Supplied cell, positions, species, and logical site ordering are preserved; no relaxation, standardization, symmetry conversion, wrapping, or sorting is allowed.
- Version 1 is three-dimensional periodic, scalar-relativistic, no-SOC, non-spin-polarized, local, synchronous, and one backend per command.
- Both ABACUS backends use the bundled scalar-relativistic PseudoDojo family; CIDER features exclude NLCC and use the original YAML with `cider_xmix=0.25`.
- The locked ABACUS preset includes 100 Ry, `kspacing=0.15`, double precision, one MPI rank, one OpenMP thread, ACE for HSE06, and automatic EXX tiling; arbitrary overrides are rejected.
- ABACUS is built separately as one PW-only CUDA+MPI executable with LibXC, native CIDER, and GPU EXX.
- Executable resolution precedence is CLI `--abacus`, config file, then `BGP_ABACUS`.
- Calculation directories and logs are retained in success, failure, non-convergence, and interruption cases.
- Do not modify `gnnbgp/gnnbgp` or any repository outside `/home/bonan/work/band-gap-prediction-software`; copy required tracked source/data into this repository.
- Use ASE to build InSb, diamond Si, GaAs, and MgO examples.
- Run all MPI and GPU commands outside the sandbox with `OMP_NUM_THREADS=1`.
- Preserve ABACUS, migrated native-CIDER, GNNBGP, model, and PseudoDojo provenance and notices.

## File Map

- `pyproject.toml`: package metadata, `bgp` entry point, dependencies, uv groups, package-data rules, pytest configuration.
- `src/bandgap_predictor/schemas.py`: stable request/result/configuration models and enums.
- `src/bandgap_predictor/errors.py`: categorized public errors with stable CLI exit codes.
- `src/bandgap_predictor/structures.py`: ASE loading, validation, immutable hashes, and site-map metadata.
- `src/bandgap_predictor/config.py`: executable path precedence and user configuration loading.
- `src/bandgap_predictor/assets.py`: manifest models, hash verification, model lookup, and element-to-UPF lookup.
- `src/bandgap_predictor/runs.py`: run allocation and atomic request/result persistence.
- `src/bandgap_predictor/backends/base.py`: backend protocol, execution context, and outcome type.
- `src/bandgap_predictor/service.py`: application orchestration independent of CLI or GUI.
- `src/bandgap_predictor/abacus/inputs.py`: locked presets and deterministic INPUT/STRU generation.
- `src/bandgap_predictor/abacus/parser.py`: convergence, occupation, energy, and band-gap parsing from the owned ABACUS result artifact.
- `src/bandgap_predictor/abacus/runner.py`: synchronous local MPI process lifecycle and signal forwarding.
- `src/bandgap_predictor/backends/abacus.py`: shared ABACUS adapter plus HSE06 and CIDER-PBE0 specializations.
- `src/bandgap_predictor/backends/gnn.py`: legacy/provisional HSE GNN adapter.
- `src/bandgap_predictor/cli.py`: Click commands, progress, terminal summary, and exit translation.
- `src/gnnbgp/`: copied minimal compatibility namespace required by the trusted legacy pickle.
- `assets/`: models, 71 PseudoDojo UPFs, citations/notices, and generated hash manifest.
- `tools/build_asset_manifest.py`: deterministic manifest generator.
- `tools/build_abacus.py`: explicit CMake build/install wrapper and build-manifest writer.
- `vendor/abacus/`: clean combined ABACUS source snapshot and provenance.
- `examples/`: ASE-built four-system structures and metadata.
- `tests/`: fast unit/contract/CLI/parser tests plus explicitly marked GNN and hardware suites.

---

### Task 1: Package Foundation, Schemas, and Stable Errors

**Files:**
- Create: `pyproject.toml`
- Create: `src/bandgap_predictor/__init__.py`
- Create: `src/bandgap_predictor/schemas.py`
- Create: `src/bandgap_predictor/errors.py`
- Create: `tests/test_schemas.py`
- Create: `tests/test_errors.py`

**Interfaces:**
- Produces: `BackendId`, `RunStatus`, `ElectronicClassification`, `PredictionRequest`, `PredictionValue`, `PredictionResult`, and `BGPError.exit_code`.
- Consumes: no project interfaces.

- [x] **Step 1: Write schema and error tests first**

```python
# tests/test_schemas.py
from pathlib import Path
from bandgap_predictor.schemas import BackendId, ElectronicClassification, PredictionRequest, PredictionValue

def test_request_uses_explicit_backend_id(tmp_path: Path) -> None:
    request = PredictionRequest(structure_path=tmp_path / "Si.cif", backend=BackendId.ABACUS_HSE06)
    assert request.backend.value == "abacus-hse06"
    assert request.work_root == Path("bgp-runs")

def test_zero_gap_retains_partial_occupation_classification() -> None:
    value = PredictionValue(gap_ev=0.0, classification=ElectronicClassification.PARTIAL_OCCUPANCY)
    assert value.model_dump(mode="json") == {"gap_ev": 0.0, "classification": "partial_occupancy"}
```

```python
# tests/test_errors.py
from bandgap_predictor.errors import InputError, PreflightError, RunError, ParseError

def test_public_error_exit_codes_are_stable() -> None:
    assert [InputError.exit_code, PreflightError.exit_code, RunError.exit_code, ParseError.exit_code] == [2, 3, 4, 5]
```

- [x] **Step 2: Run tests and confirm collection fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_schemas.py tests/test_errors.py -q`

Expected: FAIL because `bandgap_predictor` does not exist.

- [x] **Step 3: Add project metadata and minimal models**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "bandgap-predictor"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["ase>=3.23", "click>=8.1", "numpy>=1.26", "platformdirs>=4.2", "pydantic>=2.8"]

[project.optional-dependencies]
gnn = ["torch>=2.2", "mace-torch>=0.3.10", "e3nn>=0.4.4", "monty>=2024.7"]

[project.scripts]
bgp = "bandgap_predictor.cli:main"

[dependency-groups]
dev = ["pytest>=8.3", "pytest-cov>=5", "ruff>=0.6"]

[tool.hatch.build.targets.wheel]
packages = ["src/bandgap_predictor", "src/gnnbgp"]

[tool.hatch.build.targets.wheel.force-include]
"assets" = "bandgap_predictor/data"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gnn: requires legacy GNN dependencies", "hardware: requires local MPI and GPU"]
```

```python
# src/bandgap_predictor/schemas.py
from enum import StrEnum
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

class BackendId(StrEnum):
    GNN_HSE06 = "gnn-hse06"
    ABACUS_HSE06 = "abacus-hse06"
    CIDER_PBE0 = "cider-pbe0"

class RunStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

class ElectronicClassification(StrEnum):
    INSULATOR = "insulator"
    ZERO_GAP = "zero_gap"
    PARTIAL_OCCUPANCY = "partial_occupancy"
    METAL = "metal"
    UNKNOWN = "unknown"

class PredictionRequest(BaseModel):
    structure_path: Path
    backend: BackendId
    abacus_path: Path | None = None
    work_root: Path = Path("bgp-runs")

class PredictionValue(BaseModel):
    gap_ev: float | None = Field(default=None, ge=0.0)
    classification: ElectronicClassification = ElectronicClassification.UNKNOWN

class PredictionResult(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    status: RunStatus
    backend: BackendId
    method: str
    prediction: PredictionValue
    converged: bool | None
    occupation_summary: dict[str, Any] = Field(default_factory=dict)
    structure: dict[str, Any]
    settings: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    finished_at: str
    elapsed_seconds: float = Field(ge=0.0)
    artifact_path: Path
    warnings: list[str] = Field(default_factory=list)
```

```python
# src/bandgap_predictor/errors.py
class BGPError(RuntimeError):
    exit_code = 1
    def __init__(self, message: str, artifact_path=None):
        super().__init__(message)
        self.artifact_path = artifact_path

class InputError(BGPError):
    exit_code = 2

class PreflightError(BGPError):
    exit_code = 3

class RunError(BGPError):
    exit_code = 4

class ParseError(BGPError):
    exit_code = 5
```

- [x] **Step 4: Sync and run tests**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv sync --group dev && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_schemas.py tests/test_errors.py -q`

Expected: `3 passed`.

- [x] **Step 5: Commit the foundation**

```bash
git add pyproject.toml src/bandgap_predictor tests/test_schemas.py tests/test_errors.py
git commit -m "feat: establish bandgap predictor schemas"
```

### Task 2: Exact Structure Loading, Hashing, and ASE Examples

**Files:**
- Create: `src/bandgap_predictor/structures.py`
- Create: `tools/build_examples.py`
- Create: `examples/manifest.json`
- Create: `examples/InSb.cif`
- Create: `examples/Si.cif`
- Create: `examples/GaAs.cif`
- Create: `examples/MgO.cif`
- Create: `tests/test_structures.py`
- Create: `tests/test_examples.py`

**Interfaces:**
- Produces: `LoadedStructure`, `load_structure(path: Path) -> LoadedStructure`, and `write_site_map(loaded, path)`.
- Consumes: `InputError`.

- [x] **Step 1: Write validation and preservation tests**

```python
# tests/test_structures.py
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from bandgap_predictor.errors import InputError
from bandgap_predictor.structures import load_structure

def test_load_structure_preserves_cell_positions_and_order(tmp_path: Path) -> None:
    atoms = Atoms("SbIn", positions=[[0, 0, 0], [1.2, 1.3, 1.4]], cell=np.diag([6.4, 6.5, 6.6]), pbc=True)
    path = tmp_path / "input.extxyz"
    write(path, atoms)
    loaded = load_structure(path)
    assert loaded.atoms.get_chemical_symbols() == ["Sb", "In"]
    np.testing.assert_allclose(loaded.atoms.cell.array, atoms.cell.array)
    np.testing.assert_allclose(loaded.atoms.positions, atoms.positions)
    assert loaded.original_sha256 != loaded.structure_sha256

def test_rejects_nonperiodic_structure(tmp_path: Path) -> None:
    path = tmp_path / "molecule.xyz"
    write(path, Atoms("Si", positions=[[0, 0, 0]], pbc=False))
    with pytest.raises(InputError, match="three-dimensional periodic"):
        load_structure(path)
```

- [x] **Step 2: Verify tests fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_structures.py -q`

Expected: FAIL because `structures.py` is absent.

- [x] **Step 3: Implement immutable structure loading**

```python
# src/bandgap_predictor/structures.py
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from ase import Atoms
from ase.io import read
from .errors import InputError

@dataclass(frozen=True)
class LoadedStructure:
    source_path: Path
    original_bytes: bytes
    atoms: Atoms
    original_sha256: str
    structure_sha256: str
    formula: str

def load_structure(path: Path) -> LoadedStructure:
    path = Path(path).resolve()
    if not path.is_file():
        raise InputError(f"structure file does not exist: {path}")
    raw = path.read_bytes()
    atoms = read(path, index=0)
    if not all(bool(v) for v in atoms.pbc) or atoms.cell.rank != 3:
        raise InputError("structure must have a complete three-dimensional periodic cell")
    if len(atoms) == 0:
        raise InputError("structure contains no atoms")
    if any(abs(float(v)) > 1e-12 for v in atoms.get_initial_magnetic_moments()):
        raise InputError("version 1 supports only non-spin-polarized structures")
    canonical = json.dumps(
        {"symbols": atoms.get_chemical_symbols(), "cell": atoms.cell.array.tolist(),
         "positions": atoms.positions.tolist(), "pbc": [bool(v) for v in atoms.pbc]},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return LoadedStructure(path, raw, atoms.copy(), sha256(raw).hexdigest(),
                           sha256(canonical).hexdigest(), atoms.get_chemical_formula(mode="hill"))

def write_site_map(loaded: LoadedStructure, path: Path) -> None:
    records = [{"input_index": i, "symbol": symbol, "position_angstrom": loaded.atoms.positions[i].tolist()}
               for i, symbol in enumerate(loaded.atoms.get_chemical_symbols())]
    Path(path).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
```

- [x] **Step 4: Build four deterministic examples with ASE**

```python
# tools/build_examples.py
import json
from hashlib import sha256
from pathlib import Path
from ase.build import bulk
from ase.io import write

ROOT = Path(__file__).resolve().parents[1] / "examples"
CASES = {
    "InSb": bulk("InSb", "zincblende", a=6.479, cubic=True),
    "Si": bulk("Si", "diamond", a=5.431, cubic=True),
    "GaAs": bulk("GaAs", "zincblende", a=5.6533, cubic=True),
    "MgO": bulk("MgO", "rocksalt", a=4.212, cubic=True),
}
ROOT.mkdir(exist_ok=True)
manifest = {}
for name, atoms in CASES.items():
    path = ROOT / f"{name}.cif"
    write(path, atoms)
    manifest[name] = {"prototype": "diamond" if name == "Si" else "zincblende" if name in {"InSb", "GaAs"} else "rocksalt",
                      "sha256": sha256(path.read_bytes()).hexdigest(), "atom_count": len(atoms)}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
```

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python tools/build_examples.py`

Expected: four CIFs and a manifest whose hashes match their bytes.

- [x] **Step 5: Add and run example reproducibility tests**

```python
# tests/test_examples.py
import json
from hashlib import sha256
from pathlib import Path
from ase.io import read

def test_four_example_files_match_manifest() -> None:
    root = Path("examples")
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest) == {"InSb", "Si", "GaAs", "MgO"}
    for name, record in manifest.items():
        path = root / f"{name}.cif"
        assert sha256(path.read_bytes()).hexdigest() == record["sha256"]
        atoms = read(path)
        assert len(atoms) == record["atom_count"]
        assert atoms.cell.rank == 3 and all(atoms.pbc)
```

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_structures.py tests/test_examples.py -q`

Expected: all tests PASS, including rebuilding examples into a temporary directory and comparing atoms/cell/hashes.

- [x] **Step 6: Commit structures and examples**

```bash
git add src/bandgap_predictor/structures.py tools/build_examples.py examples tests/test_structures.py tests/test_examples.py
git commit -m "feat: preserve supplied crystal structures"
```

### Task 3: Executable Configuration and Build-Manifest Verification

**Files:**
- Create: `src/bandgap_predictor/config.py`
- Create: `tests/test_config.py`
- Create: `tests/test_executable.py`

**Interfaces:**
- Produces: `resolve_abacus_path(cli_path, config_path, environ) -> Path | None`, `BuildManifest`, and `verify_abacus_executable(path) -> BuildManifest`.
- Consumes: `PreflightError`.

- [x] **Step 1: Write precedence and integrity tests**

```python
def test_cli_path_precedes_config_and_environment(tmp_path):
    from bandgap_predictor.config import resolve_abacus_path
    config = tmp_path / "config.toml"
    config.write_text('[abacus]\nexecutable = "/config/abacus"\n')
    assert resolve_abacus_path(Path("/cli/abacus"), config, {"BGP_ABACUS": "/env/abacus"}) == Path("/cli/abacus")

def test_manifest_hash_mismatch_fails_closed(fake_executable):
    from bandgap_predictor.config import verify_abacus_executable
    from bandgap_predictor.errors import PreflightError
    fake_executable.write_text("changed")
    with pytest.raises(PreflightError, match="SHA-256"):
        verify_abacus_executable(fake_executable)
```

- [x] **Step 2: Run tests to confirm failure**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_config.py tests/test_executable.py -q`

Expected: FAIL because configuration interfaces are absent.

- [x] **Step 3: Implement precedence and verification**

```python
# src/bandgap_predictor/config.py
from hashlib import sha256
import os, tomllib
from pathlib import Path
from pydantic import BaseModel
from platformdirs import user_config_path
from .errors import PreflightError

class BuildManifest(BaseModel):
    schema_version: str
    executable_sha256: str
    executable_name: str
    source_revisions: list[str]
    cmake_options: dict[str, bool | str]
    features: list[str]
    toolchain: dict[str, str] = Field(default_factory=dict)
    built_at: str | None = None

def resolve_abacus_path(cli_path=None, config_path=None, environ=None):
    if cli_path is not None:
        return Path(cli_path)
    config_path = Path(config_path) if config_path else user_config_path("bandgap-predictor") / "config.toml"
    if config_path.is_file():
        value = tomllib.loads(config_path.read_text())["abacus"].get("executable")
        if value:
            return Path(value)
    value = (environ or os.environ).get("BGP_ABACUS")
    return Path(value) if value else None

def verify_abacus_executable(path: Path) -> BuildManifest:
    path = Path(path).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise PreflightError(f"ABACUS executable is not executable: {path}")
    manifest_path = path.with_name(path.name + ".bgp-build.json")
    if not manifest_path.is_file():
        raise PreflightError(f"ABACUS build manifest is missing: {manifest_path}")
    manifest = BuildManifest.model_validate_json(manifest_path.read_text())
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != manifest.executable_sha256:
        raise PreflightError("ABACUS executable SHA-256 does not match its build manifest")
    required = {"pw", "cuda", "mpi", "libxc", "native-cider", "gpu-exx", "bgp-result-json"}
    if not required.issubset(manifest.features):
        raise PreflightError(f"ABACUS build is missing required features: {sorted(required - set(manifest.features))}")
    return manifest
```

- [x] **Step 4: Run tests and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_config.py tests/test_executable.py -q`

Expected: all tests PASS.

```bash
git add src/bandgap_predictor/config.py tests/test_config.py tests/test_executable.py
git commit -m "feat: verify configured abacus executable"
```

### Task 4: Immutable Model and PseudoDojo Asset Registry

**Files:**
- Create: `assets/manifest.json`
- Create: `assets/NOTICE.md`
- Create: `assets/models/gnn/*`
- Create: `assets/models/cider/nldf_mgga_cand00_n96_noise0p01.yaml`
- Create: `assets/pseudopotentials/pseudo-dojo/*.upf`
- Create: `assets/pseudopotentials/pseudo-dojo/README.md`
- Create: `tools/build_asset_manifest.py`
- Create: `src/bandgap_predictor/assets.py`
- Create: `tests/test_assets.py`

**Interfaces:**
- Produces: `AssetRegistry.default()`, `verify(logical_id) -> VerifiedAsset`, and `pseudopotential_for(symbol) -> VerifiedAsset`.
- Consumes: `PreflightError`.

- [x] **Step 1: Write tests for immutable assets and element lookup**

```python
def test_default_registry_verifies_authoritative_models():
    registry = AssetRegistry.default()
    assert registry.verify("gnn-hse06-model").sha256 == "4afc7500a467cba1f8a00786ef25887b76b02105e09e41e1d644f85d38aaecf9"
    assert registry.verify("cider-pbe0-model").sha256 == "0bdadb33599a4f5d26baa743b1a5849a1fb63e0ce6b0fdf7ad9479feb2e28734"

def test_registry_contains_complete_pseudodojo_collection():
    registry = AssetRegistry.default()
    pseudos = [asset for asset in registry.assets if asset.kind == "pseudopotential"]
    assert len(pseudos) == 71
    assert {registry.pseudopotential_for(e).element for e in ("In", "Sb", "Si", "Ga", "As", "Mg", "O")} == {"In", "Sb", "Si", "Ga", "As", "Mg", "O"}
```

- [x] **Step 2: Copy only the approved assets into this repository**

Run the following mechanical copies without modifying their sources:

```bash
mkdir -p assets/models/gnn assets/models/cider assets/pseudopotentials/pseudo-dojo
cp gnnbgp/gnnbgp/InSb-example/20250421-baseline-band_gap_hse-batch-64-r_max-6-L-1-channels-32-node-feat-mlp-64-knn-24.model assets/models/gnn/
cp gnnbgp/gnnbgp/InSb-example/atom_init.json assets/models/gnn/
cp /mnt/hdd2/bonan/cider-combined-diverse-bandedge-training-20260726/full/nldf_mgga_cand00_n96_noise0p01.yaml assets/models/cider/
cp /home/bonan/appdir/ABACUS-orbitals/Dojo-NC-SR/Pseudopotential/*.upf assets/pseudopotentials/pseudo-dojo/
```

- [x] **Step 3: Implement deterministic manifest generation**

```python
# tools/build_asset_manifest.py (core logic)
ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
gnn_model = next((ASSET_ROOT / "models/gnn").glob("*.model"))
gnn_map = ASSET_ROOT / "models/gnn/atom_init.json"
cider_model = ASSET_ROOT / "models/cider/nldf_mgga_cand00_n96_noise0p01.yaml"
pseudo_root = ASSET_ROOT / "pseudopotentials/pseudo-dojo"
manifest = ASSET_ROOT / "manifest.json"

def record(logical_id: str, path: Path, kind: str, **metadata) -> dict:
    data = path.read_bytes()
    element = metadata.pop("element", None)
    return {"logical_id": logical_id, "path": path.relative_to(ASSET_ROOT).as_posix(), "kind": kind,
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "element": element, "metadata": metadata}

def discover_upfs(root: Path) -> list[tuple[str, Path]]:
    files = sorted(root.glob("*.upf"))
    if len(files) != 71:
        raise RuntimeError(f"expected 71 PseudoDojo UPFs, found {len(files)}")
    return [(path.stem, path) for path in files]

def main() -> None:
    entries = [
        record("gnn-hse06-model", gnn_model, "model", method="HSE06", trust="legacy-provisional-pickle"),
        record("gnn-feature-map", gnn_map, "feature-map", method="HSE06"),
        record("cider-pbe0-model", cider_model, "model", method="CIDER-PBE0", nlcc_features=False),
    ]
    entries.extend(record(f"pseudo-{element}", path, "pseudopotential", element=element,
                          family="PseudoDojo-NC-SR") for element, path in discover_upfs(pseudo_root))
    manifest.write_text(json.dumps({"schema_version": "1.0", "assets": entries}, indent=2, sort_keys=True) + "\n")
```

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python tools/build_asset_manifest.py`

Expected: 74 entries (71 UPFs, two GNN assets, one CIDER model) with the three approved hashes.

- [x] **Step 4: Implement runtime verification**

```python
# src/bandgap_predictor/assets.py
class VerifiedAsset(BaseModel):
    logical_id: str
    path: Path
    kind: str
    sha256: str
    size: int
    element: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class AssetRegistry:
    def __init__(self, root: Path, records: list[VerifiedAsset]):
        self.root, self.assets = root, records
        self._by_id = {a.logical_id: a for a in records}

    def verify(self, logical_id: str) -> VerifiedAsset:
        asset = self._by_id[logical_id]
        path = self.root / asset.path
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != asset.sha256:
            raise PreflightError(f"asset integrity check failed: {logical_id}")
        return asset.model_copy(update={"path": path})

    def pseudopotential_for(self, symbol: str) -> VerifiedAsset:
        matches = [a for a in self.assets if a.kind == "pseudopotential" and a.element == symbol]
        if len(matches) != 1:
            raise PreflightError(f"no unique bundled PseudoDojo potential for {symbol}")
        return self.verify(matches[0].logical_id)

    @classmethod
    def default(cls) -> "AssetRegistry":
        root = Path(resources.files("bandgap_predictor").joinpath("data"))
        payload = json.loads((root / "manifest.json").read_text())
        return cls(root, [VerifiedAsset.model_validate(item) for item in payload["assets"]])
```

- [x] **Step 5: Add provenance notices and run verification**

Document original paths, hashes, citations, UPF header provenance, GNN MIT source notice, trusted-pickle status, CIDER model convention, and the fact that NLCC is excluded from CIDER features. Do not invent a redistribution license; reproduce source headers/notices and flag redistribution review in release documentation if no standalone PseudoDojo license text is present.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_assets.py -q`

Expected: all tests PASS and `bgp` package builds with assets present in the wheel.

- [x] **Step 6: Commit immutable assets and registry**

```bash
git add assets tools/build_asset_manifest.py src/bandgap_predictor/assets.py tests/test_assets.py
git commit -m "feat: bundle verified bandgap assets"
```

### Task 5: Persistent Run Store and Atomic Results

**Files:**
- Create: `src/bandgap_predictor/runs.py`
- Create: `tests/test_runs.py`

**Interfaces:**
- Produces: `RunStore.allocate(loaded, backend) -> RunDirectory`, `RunDirectory.write_request`, `write_result`, and `copy_original`.
- Consumes: `LoadedStructure`, `PredictionRequest`, `PredictionResult`.

- [x] **Step 1: Write run retention and atomicity tests**

```python
def test_allocate_retains_original_and_writes_atomic_json(tmp_path, loaded_si, successful_result):
    run = RunStore(tmp_path).allocate(loaded_si, BackendId.GNN_HSE06)
    run.copy_original(loaded_si)
    run.write_result(successful_result)
    assert (run.path / "input" / loaded_si.source_path.name).read_bytes() == loaded_si.original_bytes
    assert json.loads((run.path / "result.json").read_text())["status"] == "success"
    assert not list(run.path.glob("*.tmp"))
```

- [x] **Step 2: Verify failure, implement, and retest**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runs.py -q`

Expected before implementation: import failure.

```python
# src/bandgap_predictor/runs.py (essential implementation)
def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)

@dataclass(frozen=True)
class RunDirectory:
    run_id: str
    path: Path
    def copy_original(self, loaded: LoadedStructure) -> Path:
        target = self.path / "input" / loaded.source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(loaded.original_bytes)
        return target
    def write_request(self, request: PredictionRequest) -> None:
        _atomic_json(self.path / "request.json", request.model_dump(mode="json"))
    def write_result(self, result: PredictionResult) -> None:
        _atomic_json(self.path / "result.json", result.model_dump(mode="json"))

class RunStore:
    def __init__(self, root: Path): self.root = Path(root).resolve()
    def allocate(self, loaded: LoadedStructure, backend: BackendId) -> RunDirectory:
        run_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.root / f"{stamp}-{loaded.formula}-{backend.value}-{run_id[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return RunDirectory(run_id, path.resolve())
```

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_runs.py -q`

Expected: all tests PASS.

- [x] **Step 3: Commit run persistence**

```bash
git add src/bandgap_predictor/runs.py tests/test_runs.py
git commit -m "feat: retain atomic prediction run records"
```

### Task 6: Backend Protocol and GUI-Ready Prediction Service

**Files:**
- Create: `src/bandgap_predictor/backends/__init__.py`
- Create: `src/bandgap_predictor/backends/base.py`
- Create: `src/bandgap_predictor/service.py`
- Create: `tests/test_service.py`

**Interfaces:**
- Produces: `BackendOutcome`, `PredictionContext`, `PredictionBackend` protocol, `PredictionService.predict(request) -> PredictionResult`.
- Consumes: schemas, structure loader, assets, and run store.

- [x] **Step 1: Write a service contract test with a fake backend**

```python
class FakeBackend:
    backend_id = BackendId.GNN_HSE06
    method = "HSE06-like GNN"
    def predict(self, context):
        return BackendOutcome(PredictionValue(gap_ev=1.25, classification=ElectronicClassification.INSULATOR),
                              converged=None, occupation_summary={}, settings={}, provenance={"fake": True}, diagnostics={})

def test_service_returns_typed_result_and_retains_artifacts(tmp_path, si_path):
    result = PredictionService({BackendId.GNN_HSE06: FakeBackend()}).predict(
        PredictionRequest(structure_path=si_path, backend=BackendId.GNN_HSE06, work_root=tmp_path))
    assert result.prediction.gap_ev == 1.25
    assert result.artifact_path.joinpath("result.json").is_file()
```

- [x] **Step 2: Implement the protocol and orchestration**

```python
# src/bandgap_predictor/backends/base.py
@dataclass(frozen=True)
class PredictionContext:
    request: PredictionRequest
    structure: LoadedStructure
    run: RunDirectory
    assets: AssetRegistry
    config_path: Path | None = None
    progress: Callable[[str], None] = lambda _message: None

@dataclass(frozen=True)
class BackendOutcome:
    prediction: PredictionValue
    converged: bool | None
    occupation_summary: dict[str, Any]
    settings: dict[str, Any]
    provenance: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

class PredictionBackend(Protocol):
    backend_id: BackendId
    method: str
    def predict(self, context: PredictionContext) -> BackendOutcome: ...
```

Implement `PredictionService.predict` to validate before run allocation, copy the original input, write `request.json`, time backend execution, translate successful output to `PredictionResult`, and on `BGPError` after allocation atomically write a failed result before re-raising an error carrying the artifact path.

- [x] **Step 3: Run service tests and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_service.py tests/test_runs.py -q`

Expected: all tests PASS.

```bash
git add src/bandgap_predictor/backends src/bandgap_predictor/service.py tests/test_service.py
git commit -m "feat: add backend-neutral prediction service"
```

### Task 7: Locked ABACUS Input and Structure Generation

**Files:**
- Create: `src/bandgap_predictor/abacus/__init__.py`
- Create: `src/bandgap_predictor/abacus/inputs.py`
- Create: `tests/abacus/test_inputs.py`
- Create: `tests/abacus/golden/hse06_INPUT`
- Create: `tests/abacus/golden/cider_INPUT`

**Interfaces:**
- Produces: `AbacusMethod`, immutable `COMMON`/`HSE06`/`CIDER_PBE0` mappings, and `prepare_abacus_input(workdir, structure, assets, method) -> PreparedAbacusInput`.
- Consumes: `LoadedStructure`, `AssetRegistry`, `RunDirectory`.

- [x] **Step 1: Write golden-input and site-map tests**

```python
def test_hse06_input_is_locked_and_gpu_ace(tmp_path, loaded_insb, registry):
    prepared = prepare_abacus_input(tmp_path, loaded_insb, registry, AbacusMethod.HSE06)
    text = prepared.input_path.read_text()
    assert "ecutwfc 100" in text
    assert "kspacing 0.15" in text
    assert "dft_functional HSE" in text
    assert "exxace 1" in text
    assert "exx_symmetry_realspace 0" in text
    assert "precision double" in text
    assert "lspinorb 0" in text and "nspin 1" in text
    assert "out_bgp_result 1" in text
    assert "kpar 1" in text and "bndpar 1" in text

def test_cider_input_uses_valence_only_pseudodojo_model(tmp_path, loaded_insb, registry):
    prepared = prepare_abacus_input(tmp_path, loaded_insb, registry, AbacusMethod.CIDER_PBE0)
    text = prepared.input_path.read_text()
    assert "dft_functional PBE" in text
    assert "cider_xmix 0.25" in text
    assert "cider_scf 1" in text
    assert "cider_model ./models/nldf_mgga_cand00_n96_noise0p01.yaml" in text
    assert prepared.settings["cider_feature_density"] == "valence-only"
```

- [x] **Step 2: Run tests and confirm missing implementation**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py -q`

Expected: FAIL importing `bandgap_predictor.abacus.inputs`.

- [x] **Step 3: Implement versioned presets and deterministic writers**

```python
# src/bandgap_predictor/abacus/inputs.py
class AbacusMethod(StrEnum):
    HSE06 = "hse06"
    CIDER_PBE0 = "cider-pbe0"

COMMON = {
    "basis_type": "pw", "calculation": "scf", "ecutwfc": 100,
    "kspacing": 0.15, "device": "gpu", "precision": "double",
    "nspin": 1, "lspinorb": 0, "out_band": 1, "out_bgp_result": 1,
    "kpar": 1, "bndpar": 1,
    "restart_save": "false", "restart_load": "false",
}
HSE06 = {**COMMON, "dft_functional": "HSE", "exxace": 1,
         "exx_symmetry_realspace": 0, "exx_auto_tiling": 1}
CIDER_PBE0 = {**COMMON, "dft_functional": "PBE", "cider_scf": 1,
              "cider_xmix": 0.25,
              "cider_model": "./models/nldf_mgga_cand00_n96_noise0p01.yaml"}

@dataclass(frozen=True)
class PreparedAbacusInput:
    workdir: Path
    input_path: Path
    stru_path: Path
    settings: dict[str, object]
    pseudopotentials: list[VerifiedAsset]
    model: VerifiedAsset | None

def render_input(values: dict[str, object]) -> str:
    return "INPUT_PARAMETERS\n" + "".join(f"{key} {value}\n" for key, value in values.items())
```

Implement `write_stru` with species sections required by ABACUS, Cartesian coordinates with 16 significant digits, one UPF per unique symbol, and `site-map.json` mapping grouped ABACUS indices back to original indices. Assert in tests that reversing the site map reproduces the original symbol/position sequence exactly within `1e-12 Å`.

- [x] **Step 4: Reject all override entry points**

Do not add an `extra_parameters` field to `PredictionRequest`. Add a CLI test later proving `--set`, `--input`, and unknown options exit `2`. The only preset selection is implied by backend ID and every value is recorded in `PreparedAbacusInput.settings`.

- [x] **Step 5: Run tests and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_inputs.py -q`

Expected: golden files match and all tests PASS.

```bash
git add src/bandgap_predictor/abacus/inputs.py tests/abacus/test_inputs.py tests/abacus/golden
git commit -m "feat: generate locked abacus inputs"
```

### Task 8: Occupation-Aware ABACUS Result Parser

**Files:**
- Create: `src/bandgap_predictor/abacus/parser.py`
- Create: `tests/abacus/test_parser.py`
- Create: `tests/abacus/fixtures/{insulator,partial,unconverged,malformed}/`

**Interfaces:**
- Produces: `ParsedAbacusResult`, `parse_abacus_result(run_dir: Path, returncode: int) -> ParsedAbacusResult`, consuming `OUT.*/bgp_electronic_state.json`.
- Consumes: `PredictionValue`, `ElectronicClassification`, `ParseError`, `RunError`.

- [x] **Step 1: Add four explicit parser fixtures and tests**

```python
def test_converged_insulator_uses_final_printed_gap(fixture_dir):
    parsed = parse_abacus_result(fixture_dir / "insulator", returncode=0)
    assert parsed.converged is True
    assert parsed.prediction.gap_ev == pytest.approx(1.2345)
    assert parsed.prediction.classification is ElectronicClassification.INSULATOR

def test_fractional_frontier_forces_zero_gap(fixture_dir):
    parsed = parse_abacus_result(fixture_dir / "partial", returncode=0)
    assert parsed.prediction.gap_ev == 0.0
    assert parsed.prediction.classification is ElectronicClassification.PARTIAL_OCCUPANCY
    assert parsed.occupation_summary["materially_fractional_count"] > 0

def test_nonconvergence_never_returns_gap(fixture_dir):
    with pytest.raises(RunError, match="did not converge"):
        parse_abacus_result(fixture_dir / "unconverged", returncode=0)

def test_missing_gap_evidence_is_parse_error(fixture_dir):
    with pytest.raises(ParseError, match="band-gap evidence"):
        parse_abacus_result(fixture_dir / "malformed", returncode=0)
```

- [x] **Step 2: Confirm failures, then implement evidence priority**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_parser.py -q`

Expected before implementation: import failure.

```python
# src/bandgap_predictor/abacus/parser.py (decision core)
CONVERGED = re.compile(r"#SCF IS CONVERGED#", re.I)
UNCONVERGED = re.compile(r"SCF IS NOT CONVERGED", re.I)
GAP = re.compile(r"E_gap\(k\)\s+(?:[-+0-9.Ee]+\s+)?([-+0-9.Ee]+)")

@dataclass(frozen=True)
class ParsedAbacusResult:
    prediction: PredictionValue
    converged: bool
    occupation_summary: dict[str, object]
    diagnostics: dict[str, object]

def classify(gap_ev: float, normalized_occupations: list[float]) -> PredictionValue:
    fractional = [x for x in normalized_occupations if 0.01 < x < 0.99]
    if fractional:
        return PredictionValue(gap_ev=0.0, classification=ElectronicClassification.PARTIAL_OCCUPANCY)
    if gap_ev <= 1e-8:
        return PredictionValue(gap_ev=0.0, classification=ElectronicClassification.ZERO_GAP)
    return PredictionValue(gap_ev=gap_ev, classification=ElectronicClassification.INSULATOR)
```

Read the final ABACUS log only after rejecting nonzero return codes and explicit non-convergence. Require `OUT.*/bgp_electronic_state.json` with schema `1.0`, final `converged=true`, eigenvalues in eV, k weights, and raw ABACUS weighted occupations. Divide each occupation row by its corresponding k weight and the nonmagnetic spin capacity recorded by the artifact before applying the `[0.01, 0.99]` material-fraction bounds. Record the source filename, Fermi energy, printed gap, normalized frontier occupations, and thresholds in `occupation_summary`/diagnostics. Missing or inconsistent JSON is a `ParseError`; the printed log gap alone is never sufficient for success.

- [x] **Step 3: Run parser tests and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_parser.py -q`

Expected: all four fixture classes PASS.

```bash
git add src/bandgap_predictor/abacus/parser.py tests/abacus
git commit -m "feat: parse occupation-aware abacus gaps"
```

### Task 9: Interrupt-Safe Local MPI Runner

**Files:**
- Create: `src/bandgap_predictor/abacus/runner.py`
- Create: `tests/abacus/fake_abacus.py`
- Create: `tests/abacus/test_runner.py`

**Interfaces:**
- Produces: `RunProcessResult` and `run_abacus(executable, workdir, progress) -> RunProcessResult`.
- Consumes: verified executable path and `RunError`.

- [x] **Step 1: Write launch, environment, logging, and interrupt tests**

```python
def test_runner_uses_one_rank_and_one_thread(tmp_path, fake_mpirun, fake_abacus):
    result = run_abacus(fake_abacus, tmp_path, progress=lambda line: None, mpirun=fake_mpirun)
    assert result.command[-3:] == ["-np", "1", str(fake_abacus)]
    assert json.loads((tmp_path / "fake-env.json").read_text())["OMP_NUM_THREADS"] == "1"
    assert (tmp_path / "stdout.log").is_file()

def test_interrupt_terminates_process_group_and_preserves_logs(tmp_path, hanging_abacus):
    with pytest.raises(KeyboardInterrupt):
        run_and_interrupt(hanging_abacus, tmp_path)
    assert (tmp_path / "stdout.log").exists()
```

- [x] **Step 2: Implement process-group lifecycle**

```python
# src/bandgap_predictor/abacus/runner.py (launch contract)
@dataclass(frozen=True)
class RunProcessResult:
    command: list[str]
    returncode: int
    started_at: str
    finished_at: str
    stdout_path: Path

command = [str(mpirun), "-np", "1", str(executable)]
environment = {**os.environ, "OMP_NUM_THREADS": "1"}
process = subprocess.Popen(command, cwd=workdir, env=environment,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
try:
    for line in process.stdout:
        log.write(line)
        if is_progress_line(line):
            progress(line.rstrip())
except KeyboardInterrupt:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    raise
```

Use a configurable `mpirun` parameter only for tests; production resolves `shutil.which("mpirun")` and raises `PreflightError` if absent. Record command, timestamps, return code, and log paths. Do not print a fabricated completion percentage.

- [x] **Step 3: Run tests and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_runner.py -q`

Expected: runner tests PASS without invoking real MPI.

```bash
git add src/bandgap_predictor/abacus/runner.py tests/abacus/fake_abacus.py tests/abacus/test_runner.py
git commit -m "feat: run local abacus jobs safely"
```

### Task 10: Legacy/Provisional GNN-HSE06 Backend

**Files:**
- Create: `src/bandgap_predictor/backends/gnn.py`
- Create: `src/gnnbgp/` by copying the minimal inference compatibility namespace
- Create: `src/gnnbgp/LICENSE.txt`
- Create: `tests/gnn/test_backend.py`
- Create: `tests/gnn/baselines.json`

**Interfaces:**
- Produces: `GNNHSE06Backend.predict(context) -> BackendOutcome`.
- Consumes: asset registry, `LoadedStructure`, copied `ScalarPropertyPredictor` inference path.

- [x] **Step 1: Copy the minimal trusted-pickle compatibility code**

Copy `gnnbgp/gnnbgp/src/gnnbgp/macegap/{data,modules,tools}` and the MIT license into `src/gnnbgp`, excluding training-only modules, CLI, plots, outputs, tests, and dirty working-tree artifacts. Use `git -C gnnbgp/gnnbgp show HEAD:<path>` or a clean archive so uncommitted nested-repository changes are not copied. Preserve module paths required by the pickle.

- [x] **Step 2: Add a direct single-Atoms inference test**

```python
@pytest.mark.gnn
@pytest.mark.parametrize("name", ["InSb", "Si", "GaAs", "MgO"])
def test_gnn_regression(name, example_paths, expected_gnn_gaps):
    request = PredictionRequest(structure_path=example_paths[name], backend=BackendId.GNN_HSE06)
    result = make_service_with_gnn().predict(request)
    assert result.prediction.gap_ev == pytest.approx(expected_gnn_gaps[name], abs=1e-6)
    assert "legacy/provisional" in " ".join(result.warnings)
    assert result.provenance["model_sha256"] == "4afc7500a467cba1f8a00786ef25887b76b02105e09e41e1d644f85d38aaecf9"
```

Generate `tests/gnn/baselines.json` once from the verified original CLI/model in a clean uv environment and record the environment versions and model hash beside the four values. Do not use experimental gaps as expected GNN outputs.

- [x] **Step 3: Implement trusted loading and direct prediction**

```python
class GNNHSE06Backend:
    backend_id = BackendId.GNN_HSE06
    method = "HSE06-like legacy ScalarPropertyMACE"
    warning = "Bundled GNN model is a trusted-source legacy/provisional PyTorch pickle."

    def predict(self, context: PredictionContext) -> BackendOutcome:
        model = context.assets.verify("gnn-hse06-model")
        feature_map = context.assets.verify("gnn-feature-map")
        predictor = ScalarPropertyPredictor(str(model.path), device="cpu", feat_map_file=str(feature_map.path))
        configuration = config_from_atoms(context.structure.atoms, scalar_property_key=None)
        atomic_data = AtomicData.from_config(configuration, predictor.z_table, predictor.r_max,
                                             extra_node_attrs_map=predictor.extra_node_attrs_map, knn=24)
        values, _ = predictor.predict([atomic_data], include_ref=False, batch_size=1, knn=24)
        gap = max(0.0, float(values[0]))
        return BackendOutcome(PredictionValue(gap_ev=gap, classification=ElectronicClassification.INSULATOR if gap > 1e-8 else ElectronicClassification.ZERO_GAP),
                              None, {}, {"knn": 24, "device": "cpu"},
                              {"model_sha256": model.sha256, "feature_map_sha256": feature_map.sha256,
                               "trust": "legacy-provisional-pickle"}, {}, [self.warning])
```

Call `torch.load(..., weights_only=False, map_location=device)` explicitly and only for the bundled hash-verified model. Do not accept a user model path.

- [x] **Step 4: Run GNN tests outside the sandbox if using CUDA is requested**

Run CPU regression: `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra gnn --group dev && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/gnn -q -m gnn`

Expected: four numerical regressions PASS and no structure file is rewritten.

- [x] **Step 5: Commit GNN integration**

```bash
git add src/gnnbgp src/bandgap_predictor/backends/gnn.py tests/gnn
git commit -m "feat: integrate provisional hse gnn backend"
```

### Task 11: ABACUS HSE06 and CIDER-PBE0 Backend Adapters

**Files:**
- Create: `src/bandgap_predictor/backends/abacus.py`
- Create: `tests/abacus/test_backends.py`

**Interfaces:**
- Produces: `AbacusHSE06Backend` and `CiderPBE0Backend`.
- Consumes: executable resolution/verification, input preparation, runner, parser, and `BackendOutcome`.

- [x] **Step 1: Write adapter tests with a fake verified executable and runner**

```python
@pytest.mark.parametrize("backend_cls,method", [(AbacusHSE06Backend, "HSE06"), (CiderPBE0Backend, "CIDER-PBE0")])
def test_adapter_returns_full_provenance(backend_cls, method, context, fake_runner, fake_build_manifest):
    outcome = backend_cls(runner=fake_runner).predict(context)
    assert outcome.provenance["method"] == method
    assert outcome.provenance["executable_sha256"] == fake_build_manifest.executable_sha256
    assert set(outcome.provenance["pseudopotential_sha256"]) == set(context.structure.atoms.get_chemical_symbols())
```

- [x] **Step 2: Implement a shared template with two locked specializations**

```python
class AbacusBackend:
    method_kind: AbacusMethod
    def predict(self, context: PredictionContext) -> BackendOutcome:
        executable = resolve_abacus_path(context.request.abacus_path, context.config_path, os.environ)
        if executable is None:
            raise PreflightError("ABACUS executable is not configured")
        build = verify_abacus_executable(executable)
        prepared = prepare_abacus_input(context.run.path, context.structure, context.assets, self.method_kind)
        process = self.runner(executable, prepared.workdir, context.progress)
        parsed = parse_abacus_result(prepared.workdir, process.returncode)
        return BackendOutcome(parsed.prediction, parsed.converged, parsed.occupation_summary,
                              prepared.settings, build_provenance(build, prepared), parsed.diagnostics)

class AbacusHSE06Backend(AbacusBackend):
    backend_id, method, method_kind = BackendId.ABACUS_HSE06, "HSE06", AbacusMethod.HSE06

class CiderPBE0Backend(AbacusBackend):
    backend_id, method, method_kind = BackendId.CIDER_PBE0, "CIDER-PBE0", AbacusMethod.CIDER_PBE0
```

- [x] **Step 3: Run adapter tests and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/abacus/test_backends.py -q`

Expected: both specializations PASS with no real subprocess.

```bash
git add src/bandgap_predictor/backends/abacus.py tests/abacus/test_backends.py
git commit -m "feat: add abacus prediction backends"
```

### Task 12: `bgp` CLI, Diagnostics, and JSON Contract

**Files:**
- Create: `src/bandgap_predictor/cli.py`
- Create: `src/bandgap_predictor/factory.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_json_schema.py`

**Interfaces:**
- Produces: Click `main`, `predict`, `backends`, `assets verify`, and `doctor` commands.
- Consumes: `PredictionService`, backend factory, config/assets, and stable errors.

- [x] **Step 1: Write CLI tests for help, success, precedence, and failures**

```python
def test_predict_prints_gap_and_artifact_path(runner, fake_service, si_path):
    result = runner.invoke(main, ["predict", str(si_path), "--backend", "gnn-hse06"])
    assert result.exit_code == 0
    assert "Band gap" in result.output and "Artifacts" in result.output

def test_locked_cli_rejects_abacus_overrides(runner, si_path):
    result = runner.invoke(main, ["predict", str(si_path), "--backend", "abacus-hse06", "--set", "ecutwfc=50"])
    assert result.exit_code == 2

def test_preflight_error_uses_exit_three(runner, si_path):
    result = runner.invoke(main, ["predict", str(si_path), "--backend", "abacus-hse06"])
    assert result.exit_code == 3
```

- [x] **Step 2: Implement thin CLI commands**

```python
@click.group()
@click.version_option()
def main() -> None:
    """Predict crystal band gaps with provenance."""

@main.command()
@click.argument("structure", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--backend", type=click.Choice([item.value for item in BackendId]), required=True)
@click.option("--abacus", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--work-root", type=click.Path(path_type=Path, file_okay=False), default=Path("bgp-runs"))
def predict(structure: Path, backend: str, abacus: Path | None, work_root: Path) -> None:
    try:
        result = create_service().predict(PredictionRequest(structure_path=structure, backend=BackendId(backend),
                                                            abacus_path=abacus, work_root=work_root))
    except BGPError as error:
        raise click.exceptions.Exit(error.exit_code) from error
    click.echo(f"Backend: {result.backend.value}\nMethod: {result.method}")
    click.echo(f"Band gap: {result.prediction.gap_ev:.7f} eV" if result.prediction.gap_ev is not None else "Band gap: unavailable")
    click.echo(f"Classification: {result.prediction.classification.value}\nArtifacts: {result.artifact_path}")
```

Implement `factory.create_service()` as the only composition root. `backends` reports available/unavailable plus reasons without loading the GNN pickle. `assets verify` hashes every asset. `doctor` resolves and verifies ABACUS and prints its manifest fields. Error messages include retained artifact paths when available and never suppress the JSON diagnostics.

- [x] **Step 3: Validate versioned JSON and CLI behavior**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli.py tests/test_json_schema.py -q`

Expected: all CLI exit-code, output, and schema round-trip tests PASS.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bgp --help`

Expected: help lists `predict`, `backends`, `assets`, and `doctor`.

- [x] **Step 4: Commit CLI**

```bash
git add src/bandgap_predictor/cli.py src/bandgap_predictor/factory.py tests/test_cli.py tests/test_json_schema.py
git commit -m "feat: expose unified bgp command"
```

### Task 13: Workspace-Local ABACUS Integration Branch, Result Artifact, Clean Snapshot, and Build Helper

**Files:**
- Create: `vendor/abacus/**`
- Create: `vendor/abacus/UPSTREAM.md`
- Create: `vendor/abacus/INTEGRATION_PATCHES.md`
- Create in integrated source: `source/source_io/module_bgp/bgp_result.h`
- Create in integrated source: `source/source_io/module_bgp/bgp_result.cpp`
- Modify in integrated source: `source/source_io/CMakeLists.txt`
- Modify in integrated source: `source/source_io/module_parameter/input_parameter.h`
- Modify in integrated source: `source/source_io/module_parameter/read_input_item_output.cpp`
- Modify in integrated source: `source/source_esolver/esolver_ks_pw.cpp`
- Create in integrated source: `source/source_io/test/bgp_result_test.cpp`
- Modify in integrated source: `source/source_io/test/CMakeLists.txt`
- Create: `tools/build_abacus.py`
- Create: `tests/test_build_abacus.py`
- Create: `tests/vendor/test_abacus_source_contract.py`

**Interfaces:**
- Produces: clean combined source and a build helper that creates `abacus_pw_gpu` plus `abacus_pw_gpu.bgp-build.json` accepted by `verify_abacus_executable`.
- Consumes: CMake options and build-manifest schema from Task 3.

- [x] **Step 1: Create a workspace-local source-integration clone and branch**

Use only paths inside this repository for writes:

```bash
git clone --no-checkout /home/bonan/appdir/CiderPress/abacus-cider .integration/abacus
git -C .integration/abacus switch --detach 7828d7cc2
git -C .integration/abacus switch -c integration/cider-gpu-exx
git -C .integration/abacus cherry-pick 1ace47911
```

Apply the six relevant newer GPU EXX commits in their original order:

```bash
git -C .integration/abacus cherry-pick 20b5024a9 e96a73f7e 36789e304 7e24ed1be d67df55bb 73a8bd8a8
```

Resolve conflicts by retaining native CIDER hooks from `7828d7cc2` and the newer symmetry-remap cutoff, GPU real-space symmetry default, reduced-k energy/boundary fixes, automatic tiling, and hardened execution context. Do not resolve conflicts by wholesale choosing one side. Record each cherry-pick and manual resolution in `INTEGRATION_PATCHES.md`.

- [x] **Step 2: Run source-level ABACUS regressions before export**

Configure a test build with `BUILD_TESTING=ON`, then run native CIDER evaluator/parity tests and GPU EXX unit tests covering reduced-k mapping, symmetry-compatible FFT grids, automatic tiling, and execution context. Each configure/build/test command runs outside the sandbox if it initializes MPI or CUDA. Expected: all selected tests PASS; otherwise do not export.

- [x] **Step 3: Add a backend-neutral final electronic-state JSON artifact**

On the workspace-local integration branch, add the `out_bgp_result` boolean input, default false. After the final PW SCF state is available, rank zero writes `OUT.<suffix>/bgp_electronic_state.json` atomically with:

```json
{
  "schema_version": "1.0",
  "converged": true,
  "nspin": 1,
  "occupation_weighting": "abacus_k_weighted",
  "spin_capacity": 2.0,
  "energy_unit": "eV",
  "fermi_energy_ev": 0.0,
  "printed_gap_ev": 0.0,
  "k_weights": [1.0],
  "eigenvalues_ev": [[-1.0, 1.0]],
  "occupations": [[2.0, 0.0]]
}
```

The real arrays are the final `ekb` and `wg` for every irreducible k point and band. Register `out_bgp_result` in `read_input_item_output.cpp`, synchronize it like other boolean output flags, and reject non-PW or `nspin != 1` use. Add the writer to `source_io`, invoke it from the final PW SCF output path, and use a same-directory temporary file followed by rename. Add C++ unit tests for schema keys, dimensions, finite values, rank-zero-only writing, atomic replacement, converged and non-converged flags, and nspin=1 normalization. Add one HSE06 and one native-CIDER integration fixture proving the artifact is emitted without enabling training-data output. This output is package integration functionality, not a training interface.

- [x] **Step 4: Re-run source regressions and export only clean tracked source**

```bash
mkdir -p vendor/abacus
git -C .integration/abacus archive --format=tar --output=../abacus-integration.tar integration/cider-gpu-exx
tar -xf .integration/abacus-integration.tar -C vendor/abacus
```

Verify `vendor/abacus` contains no `.git`, build directory, object file, executable, run output, Python cache, or external training artifact. Write `UPSTREAM.md` with native CIDER base, GPU branch head, final integration commit, ABACUS upstream base, licenses, and the exact export command.

- [x] **Step 5: Write failing build-helper command tests**

```python
def test_build_command_is_pw_cuda_mpi_native_cider(tmp_path):
    command = cmake_configure_command(Path("vendor/abacus"), tmp_path / "build", ["86", "89", "120"])
    assert "-DENABLE_LCAO=OFF" in command
    assert "-DENABLE_MPI=ON" in command
    assert "-DUSE_CUDA=ON" in command
    assert "-DENABLE_LIBXC=ON" in command
    assert "-DENABLE_CIDER=ON" in command
    assert "-DENABLE_CIDER_EXTERNAL=OFF" in command
    assert "-DCMAKE_CUDA_ARCHITECTURES=86;89;120" in command
```

- [x] **Step 6: Implement explicit build and manifest creation**

```python
# tools/build_abacus.py (public command construction)
def cmake_configure_command(source: Path, build: Path, architectures: list[str]) -> list[str]:
    return ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
            "-DENABLE_LCAO=OFF", "-DENABLE_MPI=ON", "-DUSE_OPENMP=ON", "-DUSE_CUDA=ON",
            "-DENABLE_LIBXC=ON", "-DENABLE_CIDER=ON", "-DENABLE_CIDER_EXTERNAL=OFF",
            "-DBUILD_TESTING=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(architectures)}"]

def write_manifest(executable: Path, source_revisions: list[str], options: dict, toolchain: dict) -> Path:
    payload = {"schema_version": "1.0", "executable_name": executable.name,
               "executable_sha256": sha256(executable.read_bytes()).hexdigest(),
               "source_revisions": source_revisions, "cmake_options": options,
               "features": ["pw", "cuda", "mpi", "libxc", "native-cider", "gpu-exx", "bgp-result-json"],
               "toolchain": toolchain, "built_at": datetime.now(timezone.utc).isoformat()}
    target = executable.with_name(executable.name + ".bgp-build.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target
```

The tool accepts explicit `--build-dir`, `--install-prefix`, `--cuda-architectures`, and dependency prefix options; it never builds during package installation/import. It runs configure, `cmake --build ... --target abacus_pw_gpu`, installs/copies the executable, hashes it, and writes the companion manifest.

- [x] **Step 7: Test helper and perform target-machine build**

Run fast tests: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_build_abacus.py tests/vendor/test_abacus_source_contract.py -q`

Expected: command, manifest, source cleanliness, CIDER boundary, and provenance tests PASS.

Run outside the sandbox:

```bash
OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run python tools/build_abacus.py --build-dir build/abacus-cuda --install-prefix build/abacus-install --cuda-architectures 86,89,120
```

Expected: `build/abacus-install/bin/abacus_pw_gpu` and matching `.bgp-build.json`; `bgp doctor --abacus .../abacus_pw_gpu` reports all seven required features.

- [x] **Step 8: Commit source snapshot and build tooling**

```bash
git add vendor/abacus tools/build_abacus.py tests/test_build_abacus.py tests/vendor
git commit -m "feat: own combined cuda cider abacus source"
```

### Task 14: End-to-End Matrix, Packaging, and User Documentation

**Files:**
- Create: `tests/hardware/test_abacus_release_matrix.py`
- Create: `tests/hardware/baselines.json`
- Create: `tests/test_wheel.py`
- Create: `README.md`
- Create: `docs/installation.md`
- Create: `docs/methods.md`
- Create: `docs/results.md`
- Create: `LICENSES/`

**Interfaces:**
- Produces: reproducible installation instructions, method limitations, full release evidence, and buildable wheel/sdist.
- Consumes: all package interfaces and the compiled ABACUS executable.

- [x] **Step 1: Write required and extended hardware matrix tests**

```python
REQUIRED = ["InSb", "Si"]
EXTENDED = ["GaAs", "MgO"]
ABACUS_BACKENDS = [BackendId.ABACUS_HSE06, BackendId.CIDER_PBE0]

@pytest.mark.hardware
@pytest.mark.parametrize("system", REQUIRED)
@pytest.mark.parametrize("backend", ABACUS_BACKENDS)
def test_required_release_matrix(system, backend, executable, example_paths, hardware_baselines):
    result = create_service().predict(PredictionRequest(structure_path=example_paths[system], backend=backend,
                                                        abacus_path=executable, work_root=Path("hardware-runs")))
    expected = hardware_baselines[backend.value][system]
    assert result.status is RunStatus.SUCCESS
    assert result.prediction.classification.value == expected["classification"]
    assert result.prediction.gap_ev == pytest.approx(expected["gap_ev"], abs=expected["abs_tolerance_ev"])
    assert result.provenance["executable_sha256"]
```

Add the same test for `EXTENDED`, marked `hardware` and `extended`. Generate baselines only from converged runs of the final hashed executable and store executable/model/pseudopotential/preset hashes beside every value. InSb CIDER may validly baseline as `0.0` with `partial_occupancy`; do not force agreement with HSE06 or GNN.

- [x] **Step 2: Run the fast release suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m 'not hardware and not gnn'`

Expected: all fast tests PASS.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m gnn`

Expected: four GNN regressions PASS.

- [x] **Step 3: Run the required GPU/MPI release suite outside the sandbox**

```bash
OMP_NUM_THREADS=1 BGP_ABACUS=/absolute/path/to/abacus_pw_gpu UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/hardware/test_abacus_release_matrix.py -q -m 'hardware and not extended'
```

Expected: four required calculations PASS: InSb and Si through HSE06 and CIDER-PBE0. Run the four GaAs/MgO extended cases separately and record their status.

- [x] **Step 4: Document installation and method limitations**

`README.md` must show:

```bash
uv sync --extra gnn --group dev
uv run python tools/build_abacus.py --build-dir build/abacus --install-prefix build/abacus-install --cuda-architectures 86,89,120
uv run bgp doctor --abacus build/abacus-install/bin/abacus_pw_gpu
uv run bgp predict examples/InSb.cif --backend cider-pbe0 --abacus build/abacus-install/bin/abacus_pw_gpu
```

Document no SOC, nonmagnetic-only, no relaxation/standardization, local synchronous execution, locked preset, retained directories, executable precedence, CIDER valence-only/NLCC exclusion, method-specific InSb interpretation, and the trusted legacy pickle warning. Copy applicable upstream notices into `LICENSES/` and include the PseudoDojo redistribution-review status without making unsupported claims.

- [x] **Step 5: Verify wheel contents and installed CLI**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv build`

Expected: wheel and sdist build successfully.

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_wheel.py -q`

Expected: a clean temporary installation can run `bgp --help`, verify all 74 assets, load all four examples, and locate vendored source documentation; the wheel contains no ABACUS binary or run artifacts.

- [x] **Step 6: Run final verification and commit**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m 'not hardware'`

Expected: all non-hardware tests PASS.

Run: `git status --short`

Expected: only intentionally retained hardware run directories or documented baseline updates are untracked; do not add calculation outputs to Git.

```bash
git add README.md docs LICENSES tests/hardware tests/test_wheel.py tests/hardware/baselines.json
git commit -m "docs: complete bandgap predictor release workflow"
```

## Final Verification Checklist

- [x] `uv build` succeeds and the installed command is `bgp`.
- [x] `bgp backends`, `bgp assets verify`, and `bgp doctor` work without loading the legacy pickle unnecessarily.
- [x] All 74 bundled assets match their recorded hashes; all required elements map to exactly one PseudoDojo UPF.
- [x] Structure round trips preserve cell, positions, species, and logical input ordering through `site-map.json`.
- [x] Raw ABACUS overrides, SOC, magnetism, non-periodic cells, missing assets, and mismatched executables fail with documented exit codes.
- [x] Converged partial occupation returns `0.0 eV` with provenance; non-convergence and parse failure never return a successful gap.
- [x] Interrupts return exit `130` and retain logs plus an interrupted `result.json`.
- [x] GNN regressions pass for InSb, Si, GaAs, and MgO and always carry the legacy/provisional warning.
- [x] Required InSb/Si HSE06 and CIDER GPU/MPI runs pass with one rank and `OMP_NUM_THREADS=1`; extended GaAs/MgO status is recorded.
- [x] `vendor/abacus` is a clean, licensed, provenance-recorded source snapshot with no build products or nested Git database.
- [x] No repository or data source outside the integration workspace was modified.
