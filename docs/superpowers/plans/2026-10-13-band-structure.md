# Band Structure Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `bgp bands`, a two-stage band-structure workflow that writes a
machine-readable `bands.json` (k-path, eigenvalues, Fermi level, VBM/CBM/gap)
for `abacus-hse06`, `abacus-pbe0` and `cider-pbe0`, with an optional PNG/PDF
plot, without changing any vendored ABACUS code or rebuilding the GPU binary.

**Architecture:** Stage 1 runs the existing locked SCF preset on the
seekpath-standardized primitive cell; stage 2 reuses that mesh and charge
density and solves the k-path either through the native `K_POINTS_BAND` hybrid
band targets (`calculation scf`, hybrids) or through `calculation nscf` with a
line-mode main KPT (CIDER). `OUT.ABACUS/band.txt` supplies eigenvalues;
k-points, labels and distances come from our own seekpath-derived plan; the
combination is written as `bands.json` and embedded in the CLI JSON envelope.

**Tech Stack:** Python 3.12, uv, click, pydantic, pytest, numpy, ASE,
`seekpath>=2.1.0`, `spglib>=2.5.0`, matplotlib (plot only), ABACUS PW CUDA+MPI.

Design authority: [`../specs/2026-10-13-band-structure-design.md`](../specs/2026-10-13-band-structure-design.md).

## Global Constraints

- Locked gap presets stay byte-for-byte unchanged: `abacus-hse06`/`abacus-pbe0`
  preset `1.1`, `cider-pbe0` preset `1.2` (`cider_xmix=0.15`), all with
  `kspacing 0.15`, `smearing_sigma 0.001 eV`, `nspin 1`, no SOC, one MPI rank,
  one OpenMP thread. Band mode adds a **separate** stage-2 mapping only.
- No vendored ABACUS source change and no GPU rebuild in Tasks 1–7 and 9; the
  released executable sha256 `9ad2d03bbb4f6fa8a56384a654f744fd23bd563872555b105b44f7efe544fb2e`
  (`build/abacus-install-fixed/bin/abacus_pw_gpu`) must stay the binary used by
  every hardware run. Task 8 is the only conditional place a rebuild may be
  proposed, and it must not be executed without an explicit user decision.
- All hardware runs use `build/abacus-install-fixed/bin/abacus_pw_gpu` with
  `OMP_NUM_THREADS=1`.
- The submitted structure is never modified; the band cell is a **separate,
  recorded** seekpath primitive cell.
- `bgp bands` supports exactly `abacus-hse06`, `abacus-pbe0`, `cider-pbe0`;
  `gnn-hse06` raises `InputError` with `error.kind = "unsupported"`.
- One JSON object on stdout with `--json`; progress on stderr (contract from
  `docs/frontends.md`).
- Existing tests must keep passing: `uv run pytest -q` → 66 passed, 12 skipped
  before this plan's first task; only additive counts afterwards.

---

### Task 1: Dependencies and k-path planning

**Files:**
- Modify: `pyproject.toml` (add `seekpath>=2.1.0`, `spglib>=2.5.0` to `dependencies`)
- Create: `src/bandgap_predictor/bands/__init__.py`
- Create: `src/bandgap_predictor/bands/path.py`
- Test: `tests/bands/test_path.py`

**Interfaces:**
- Produces: `BandPathPlan` (frozen dataclass) with fields `labels` (one name per special point), `special_kpoints_direct`, `segment_counts`, `kpoints_direct`, `kpoints_cartesian`, `cumulative_distance`, `segment_breaks`, `label_positions`, `reference_distance`, `spacegroup_number`, `spacegroup_international`, `bravais_lattice`, `primitive_transformation_matrix`, `inverse_primitive_transformation_matrix`, `volume_original_wrt_prim`, `seekpath_version`, `spglib_version`.
- Produces: `build_band_path(cell_angstrom, scaled_positions, numbers, *, reference_distance=DEFAULT_REFERENCE_DISTANCE, max_kpoints=DEFAULT_MAX_KPOINTS, allow_many_kpoints=False, symprec=1e-5, angle_tolerance=-1.0) -> BandPathPlan`.
- Consumes: `bandgap_predictor.errors.InputError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/bands/test_path.py
import numpy as np
import pytest
from ase.io import read

from bandgap_predictor.bands.path import build_band_path
from bandgap_predictor.errors import InputError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _structure(name):
    atoms = read(EXAMPLES / name)
    return atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers()


def test_si_path_uses_standard_fcc_labels():
    plan = build_band_path(*_structure("Si.cif"))
    assert plan.spacegroup_number == 227
    assert plan.bravais_lattice.startswith("cF")
    assert plan.labels[0] == "GAMMA"
    assert set(plan.labels) <= {"GAMMA", "X", "U", "K", "L", "W", "W_2"}
    assert plan.segment_counts[-1] == 1
    assert len(plan.kpoints_direct) == sum(plan.segment_counts)
    assert plan.cumulative_distance[0] == 0.0
    assert all(b >= a for a, b in zip(plan.cumulative_distance, plan.cumulative_distance[1:]))
    assert len(plan.label_positions) == len(plan.labels)
    assert plan.segment_breaks[0] > 0


def test_path_is_deterministic_and_guarded():
    first = build_band_path(*_structure("Si.cif"))
    second = build_band_path(*_structure("Si.cif"))
    assert first.kpoints_direct == second.kpoints_direct
    with pytest.raises(InputError, match="max-kpoints"):
        build_band_path(*_structure("Si.cif"), reference_distance=0.005, max_kpoints=100)
    dense = build_band_path(*_structure("Si.cif"), reference_distance=0.005, max_kpoints=100, allow_many_kpoints=True)
    assert len(dense.kpoints_direct) > 100


def test_indirect_reference_distance_controls_point_count():
    coarse = build_band_path(*_structure("Si.cif"), reference_distance=0.05)
    fine = build_band_path(*_structure("Si.cif"), reference_distance=0.0125)
    assert len(coarse.kpoints_direct) < len(fine.kpoints_direct)


def test_insb_path_is_f43m():
    plan = build_band_path(*_structure("InSb.cif"))
    assert plan.spacegroup_number == 216
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_path.py -q`
Expected: FAIL — `ModuleNotFoundError: bandgap_predictor.bands`.

- [ ] **Step 3: Add the dependencies**

```toml
dependencies = [
  "ase>=3.23",
  "click>=8.1",
  "numpy>=1.26,<2.5",
  "platformdirs>=4.2",
  "pydantic>=2.8",
  "seekpath>=2.1.0",
  "spglib>=2.5.0",
]
```

Run: `UV_CACHE_DIR=/tmp/uv-cache uv lock && UV_CACHE_DIR=/tmp/uv-cache uv sync --extra gnn --group dev`

- [ ] **Step 4: Implement `bands/path.py`**

```python
"""seekpath-driven high-symmetry k-path planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import seekpath
import spglib

from ..errors import InputError

DEFAULT_REFERENCE_DISTANCE = 0.025
DEFAULT_MAX_KPOINTS = 400


@dataclass(frozen=True)
class BandPathPlan:
    labels: list[str]
    special_kpoints_direct: list[tuple[float, float, float]]
    segment_counts: list[int]
    kpoints_direct: list[tuple[float, float, float]]
    kpoints_cartesian: list[tuple[float, float, float]]
    cumulative_distance: list[float]
    segment_breaks: list[int]
    label_positions: list[float]
    reference_distance: float
    spacegroup_number: int
    spacegroup_international: str
    bravais_lattice: str
    primitive_transformation_matrix: list[list[float]]
    inverse_primitive_transformation_matrix: list[list[float]]
    volume_original_wrt_prim: float
    seekpath_version: str
    spglib_version: str


def _float3(vector) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _chains(path_segments) -> list[list[str]]:
    """Merge consecutive seekpath segments that share an endpoint."""

    chains: list[list[str]] = []
    for start, end in path_segments:
        if chains and chains[-1][-1] == start:
            chains[-1].append(end)
        else:
            chains.append([start, end])
    return chains


def build_band_path(
    cell_angstrom,
    scaled_positions,
    numbers,
    *,
    reference_distance: float = DEFAULT_REFERENCE_DISTANCE,
    max_kpoints: int = DEFAULT_MAX_KPOINTS,
    allow_many_kpoints: bool = False,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> BandPathPlan:
    """Plan a labelled k-path for the given cell using seekpath."""

    structure = (
        np.asarray(cell_angstrom, dtype=float),
        np.asarray(scaled_positions, dtype=float),
        np.asarray(numbers, dtype=int),
    )
    try:
        result = seekpath.get_explicit_k_path(
            structure,
            with_time_reversal=True,
            reference_distance=reference_distance,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
    except Exception as error:  # seekpath raises assorted exceptions
        raise InputError(f"seekpath could not determine a band path: {error}") from error

    reciprocal = np.asarray(result["reciprocal_primitive_lattice"], dtype=float)
    coordinates = {
        name: np.asarray(value, dtype=float) for name, value in result["point_coords"].items()
    }
    chains = _chains(result["path"])

    labels: list[str] = []
    special: list[tuple[float, float, float]] = []
    counts: list[int] = []
    for chain in chains:
        for index, name in enumerate(chain):
            labels.append(name)
            special.append(_float3(coordinates[name]))
            if index == len(chain) - 1:
                counts.append(1)
                continue
            delta = coordinates[chain[index + 1]] - coordinates[name]
            length = float(np.linalg.norm(delta @ reciprocal))
            counts.append(max(1, int(round(length / reference_distance))))

    direct: list[tuple[float, float, float]] = []
    cartesian: list[tuple[float, float, float]] = []
    distance: list[float] = []
    breaks: list[int] = []
    label_positions: list[float] = []
    special_index = 0
    for chain in chains:
        if direct:
            breaks.append(len(direct))
        total = 0.0
        for index in range(len(chain)):
            start = np.asarray(special[special_index], dtype=float)
            count = counts[special_index]
            step = (
                (np.asarray(special[special_index + 1], dtype=float) - start) / count
                if index < len(chain) - 1
                else np.zeros(3)
            )
            for offset in range(count):
                point = start + offset * step
                if direct and not (index == 0 and offset == 0):
                    previous = np.asarray(direct[-1], dtype=float)
                    total += float(np.linalg.norm((point - previous) @ reciprocal))
                direct.append(_float3(point))
                cartesian.append(_float3(point @ reciprocal))
                distance.append(total)
            label_positions.append(total)
            special_index += 1

    if not allow_many_kpoints and len(direct) > max_kpoints:
        raise InputError(
            f"band path requires {len(direct)} k-points, above max-kpoints={max_kpoints}; "
            "raise --reference-distance or pass --allow-many-kpoints"
        )

    return BandPathPlan(
        labels=labels,
        special_kpoints_direct=special,
        segment_counts=counts,
        kpoints_direct=direct,
        kpoints_cartesian=cartesian,
        cumulative_distance=distance,
        segment_breaks=breaks,
        label_positions=label_positions,
        reference_distance=reference_distance,
        spacegroup_number=int(result["spacegroup_number"]),
        spacegroup_international=str(result["spacegroup_international"]),
        bravais_lattice=str(result["bravais_lattice_extended"]),
        primitive_transformation_matrix=np.asarray(
            result["primitive_transformation_matrix"], dtype=float
        ).tolist(),
        inverse_primitive_transformation_matrix=np.asarray(
            result["inverse_primitive_transformation_matrix"], dtype=float
        ).tolist(),
        volume_original_wrt_prim=float(result["volume_original_wrt_prim"]),
        seekpath_version=str(seekpath.__version__),
        spglib_version=str(spglib.__version__),
    )
```

Invariants this implementation must satisfy (already asserted by the tests):
`cumulative_distance[0] == 0.0`; distances non-decreasing; exactly
`sum(segment_counts)` k-points; `segment_counts[-1] == 1`; one
`segment_breaks` entry per chain after the first; `label_positions` has one
entry per special point and equals the accumulated distance at that point.

- [ ] **Step 5: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_path.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/bandgap_predictor/bands tests/bands
git commit -m "feat: add seekpath k-path planning for band mode"
```

---

### Task 2: Primitive band cell

**Files:**
- Create: `src/bandgap_predictor/bands/cell.py`
- Test: `tests/bands/test_cell.py`

**Interfaces:**
- Produces: `BandCell` (frozen dataclass) with `atoms: ase.Atoms` (primitive cell, `pbc=True`, positions wrapped into `[0, 1)`), `structure_sha256: str`, `formula: str`, `symmetry: dict[str, object]` (seekpath/spglib versions, spacegroup number/symbol, `bravais_lattice`, `primitive_transformation_matrix`, `inverse_primitive_transformation_matrix`, `volume_original_wrt_prim`).
- Produces: `standardize_cell(loaded: LoadedStructure, *, symprec: float = 1e-5) -> BandCell` reusing `structures._structure_digest` for the hash.
- Consumes: `structures.LoadedStructure`, `structures.load_structure`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/bands/test_cell.py
from bandgap_predictor.bands.cell import standardize_cell
from bandgap_predictor.structures import load_structure

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_si_standardizes_to_two_atom_primitive_cell():
    loaded = load_structure(EXAMPLES / "Si.cif")
    band_cell = standardize_cell(loaded)
    assert len(band_cell.atoms) == 2
    assert band_cell.formula == "Si2"
    assert band_cell.symmetry["spacegroup_number"] == 227
    assert abs(float(band_cell.atoms.cell.volume) - 40.06) < 0.5
    assert all(0.0 <= value < 1.0 for value in band_cell.atoms.get_scaled_positions().ravel())
    assert band_cell.structure_sha256 != loaded.structure_sha256


def test_standardization_is_deterministic():
    loaded = load_structure(EXAMPLES / "InSb.cif")
    first = standardize_cell(loaded)
    second = standardize_cell(loaded)
    assert first.structure_sha256 == second.structure_sha256
    assert first.atoms.get_chemical_symbols() == second.atoms.get_chemical_symbols()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_cell.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `bands/cell.py`**

```python
"""seekpath-standardized primitive cell used for band calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import seekpath
import spglib
from ase import Atoms

from ..errors import InputError
from ..structures import LoadedStructure, _structure_digest


@dataclass(frozen=True)
class BandCell:
    atoms: Atoms
    structure_sha256: str
    formula: str
    symmetry: dict[str, object]


def standardize_cell(loaded: LoadedStructure, *, symprec: float = 1e-5) -> BandCell:
    """Return the seekpath primitive cell derived from a loaded structure."""

    structure = (
        np.asarray(loaded.atoms.cell.array, dtype=float),
        np.asarray(loaded.atoms.get_scaled_positions(), dtype=float),
        np.asarray(loaded.atoms.get_atomic_numbers(), dtype=int),
    )
    try:
        result = seekpath.get_explicit_k_path(structure, with_time_reversal=True, symprec=symprec)
    except Exception as error:
        raise InputError(f"seekpath could not standardize the structure: {error}") from error

    lattice = np.asarray(result["primitive_lattice"], dtype=float)
    positions = np.asarray(result["primitive_positions"], dtype=float) % 1.0
    numbers = [int(value) for value in result["primitive_types"]]
    atoms = Atoms(numbers=numbers, scaled_positions=positions, cell=lattice, pbc=True)

    return BandCell(
        atoms=atoms,
        structure_sha256=_structure_digest(atoms),
        formula=atoms.get_chemical_formula(mode="hill"),
        symmetry={
            "seekpath_version": str(seekpath.__version__),
            "spglib_version": str(spglib.__version__),
            "spacegroup_number": int(result["spacegroup_number"]),
            "spacegroup_international": str(result["spacegroup_international"]),
            "bravais_lattice": str(result["bravais_lattice_extended"]),
            "primitive_transformation_matrix": np.asarray(
                result["primitive_transformation_matrix"], dtype=float
            ).tolist(),
            "inverse_primitive_transformation_matrix": np.asarray(
                result["inverse_primitive_transformation_matrix"], dtype=float
            ).tolist(),
            "volume_original_wrt_prim": float(result["volume_original_wrt_prim"]),
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_cell.py -q`
Expected: PASS. If the Si volume assertion differs, print the measured volume
and pin the measured value with a comment instead of loosening the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/bandgap_predictor/bands/cell.py tests/bands/test_cell.py
git commit -m "feat: add seekpath primitive band cell"
```

---

### Task 3: Stage-2 INPUT and KPT rendering

**Files:**
- Create: `src/bandgap_predictor/bands/inputs.py`
- Create: `tests/bands/golden/bands_hse06_KPT`, `tests/bands/golden/bands_cider_INPUT`, `tests/bands/golden/bands_cider_KPT`
- Test: `tests/bands/test_band_inputs.py`

**Interfaces:**
- Produces: `BandMechanism` (StrEnum) with `HYBRID_BAND_TARGETS = "hybrid_band_targets"` and `NSCF_LINE = "nscf_line"`.
- Produces: `read_mesh_block(scf_kpt: Path) -> str` returning the auto-generated 4-line mesh block verbatim (raises `ParseError` if the first token is not `K_POINTS`).
- Produces: `render_band_kpt(plan: BandPathPlan, mechanism: BandMechanism, mesh_block: str | None) -> str`.
- Produces: `prepare_band_input(stage2_dir: Path, atoms: Atoms, assets: AssetRegistry, method: AbacusMethod, mechanism: BandMechanism, plan: BandPathPlan, scf_kpt: Path) -> PreparedAbacusInput` — stages STRU/pseudopotentials/model through the existing `prepare_abacus_input`, then rewrites `INPUT` with the stage-2 overrides and writes the band `KPT`.
- Consumes: `abacus.inputs.AbacusMethod`, `abacus.inputs.prepare_abacus_input`, `abacus.inputs.render_input`, `assets.AssetRegistry`, `bands.path.BandPathPlan`, `bands.cell.BandCell`.

Stage-2 override table (mechanism → values layered on top of the existing
method mapping):

| key | `hybrid_band_targets` | `nscf_line` |
| --- | --- | --- |
| `calculation` | `scf` | `nscf` |
| `kspacing` | `0` | `0` |
| `symmetry` | keep preset (`1`) | `0` |
| `init_chg` | `file` | `file` |
| `read_file_dir` | `../scf/OUT.ABACUS` | `../scf/OUT.ABACUS` |
| `exx_separate_loop` | `1` | absent |
| `out_app_flag` | `false` | `false` |

- [ ] **Step 1: Write the failing tests**

```python
# tests/bands/test_band_inputs.py
def test_nscf_line_kpt_is_the_main_block(tmp_path, si_plan):
    text = render_band_kpt(si_plan, BandMechanism.NSCF_LINE, mesh_block=None)
    lines = text.splitlines()
    assert lines[0] == "K_POINTS"
    assert lines[1] == str(len(si_plan.special_kpoints_direct))
    assert lines[2] == "Line_Direct"
    assert len(lines) == 3 + len(si_plan.special_kpoints_direct)
    assert lines[-1].split()[-1] == "1"


def test_hybrid_kpt_keeps_the_stage_one_mesh(tmp_path, si_plan):
    mesh = "K_POINTS\n0\nGamma\n5 5 5 0 0 0\n"
    text = render_band_kpt(si_plan, BandMechanism.HYBRID_BAND_TARGETS, mesh_block=mesh)
    assert text.startswith(mesh)
    assert "\nK_POINTS_BAND\n" in text
    assert text.split("K_POINTS_BAND")[1].splitlines()[1] == str(len(si_plan.special_kpoints_direct))


def test_stage_two_input_overrides(tmp_path, si_band_cell):
    prepared = prepare_band_input(
        tmp_path / "bands",
        si_band_cell.atoms,
        AssetRegistry.default(),
        AbacusMethod.HSE06,
        BandMechanism.HYBRID_BAND_TARGETS,
        si_plan,
        scf_kpt=write_mesh_kpt(tmp_path),
    )
    text = prepared.input_path.read_text()
    assert "calculation scf\n" in text
    assert "kspacing 0\n" in text
    assert "exx_separate_loop 1\n" in text
    assert "out_app_flag false\n" in text
    assert "read_file_dir ../scf/OUT.ABACUS\n" in text
    assert "dft_functional HSE\n" in text          # method mapping preserved
    assert prepared.settings["mechanism"] == "hybrid_band_targets"


def test_nscf_line_forces_symmetry_zero(tmp_path, si_band_cell):
    prepared = prepare_band_input(
        tmp_path / "bands",
        si_band_cell.atoms,
        AssetRegistry.default(),
        AbacusMethod.CIDER_PBE0,
        BandMechanism.NSCF_LINE,
        si_plan,
        scf_kpt=write_mesh_kpt(tmp_path),
    )
    text = prepared.input_path.read_text()
    assert "calculation nscf\n" in text
    assert "symmetry 0\n" in text
    assert "exx_separate_loop" not in text
    assert "cider_xmix 0.15\n" in text
```

Golden files pin the complete rendered text for one HSE06 KPT and one CIDER
INPUT/KPT pair (`assert prepared.input_path.read_text() == golden.read_text()`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_band_inputs.py -q`
Expected: FAIL — missing module.

- [ ] **Step 3: Implement `bands/inputs.py`**

```python
"""Stage-2 ABACUS input preparation for band runs."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from ase import Atoms

from ..abacus.inputs import (
    AbacusMethod,
    PreparedAbacusInput,
    method_values,
    prepare_abacus_input,
    render_input,
)
from ..assets import AssetRegistry
from ..errors import ParseError
from .cell import BandCell
from .path import BandPathPlan

HYBRID_OVERRIDES: dict[str, object] = {
    "calculation": "scf",
    "kspacing": 0,
    "init_chg": "file",
    "read_file_dir": "../scf/OUT.ABACUS",
    "exx_separate_loop": 1,
    "out_app_flag": False,
}
NSCF_OVERRIDES: dict[str, object] = {
    "calculation": "nscf",
    "kspacing": 0,
    "symmetry": 0,
    "init_chg": "file",
    "read_file_dir": "../scf/OUT.ABACUS",
    "out_app_flag": False,
}


class BandMechanism(StrEnum):
    HYBRID_BAND_TARGETS = "hybrid_band_targets"
    NSCF_LINE = "nscf_line"


def overrides_for(mechanism: BandMechanism) -> dict[str, object]:
    """Return the stage-2 INPUT overrides for one mechanism."""

    if mechanism is BandMechanism.HYBRID_BAND_TARGETS:
        return dict(HYBRID_OVERRIDES)
    return dict(NSCF_OVERRIDES)


def read_mesh_block(scf_kpt: Path) -> str:
    """Return the auto-generated SCF mesh block verbatim."""

    text = Path(scf_kpt).read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip().upper() not in {"K_POINTS", "KPOINTS", "K"}:
        raise ParseError(f"SCF KPT file does not start with a K_POINTS block: {scf_kpt}")
    block = "\n".join(lines[:4]).rstrip() + "\n"
    return block


def _special_point_lines(plan: BandPathPlan) -> str:
    return "".join(
        f"{point[0]:.10f} {point[1]:.10f} {point[2]:.10f} {count}\n"
        for point, count in zip(plan.special_kpoints_direct, plan.segment_counts, strict=True)
    )


def render_band_kpt(plan: BandPathPlan, mechanism: BandMechanism, mesh_block: str | None) -> str:
    special = _special_point_lines(plan)
    if mechanism is BandMechanism.NSCF_LINE:
        return f"K_POINTS\n{len(plan.special_kpoints_direct)}\nLine_Direct\n{special}"
    if mesh_block is None:
        raise ParseError("hybrid band targets require the stage-1 mesh block")
    return f"{mesh_block}\nK_POINTS_BAND\n{len(plan.special_kpoints_direct)}\nLine_Direct\n{special}"


def prepare_band_input(
    stage2_dir: Path,
    atoms: Atoms,
    assets: AssetRegistry,
    method: AbacusMethod,
    mechanism: BandMechanism,
    plan: BandPathPlan,
    scf_kpt: Path,
) -> PreparedAbacusInput:
    """Prepare the stage-2 INPUT/KPT on top of the method's locked mapping."""

    prepared = prepare_abacus_input(Path(stage2_dir), atoms, assets, method)
    overrides = overrides_for(mechanism)
    prepared.input_path.write_text(
        render_input({**dict(method_values(method)), **overrides}),
        encoding="utf-8",
    )
    prepared.workdir.joinpath("KPT").write_text(
        render_band_kpt(plan, mechanism, read_mesh_block(scf_kpt)),
        encoding="utf-8",
    )
    prepared.settings.update(
        {
            **overrides,
            "mechanism": mechanism.value,
            "stage_two": True,
            "band_kpoint_count": len(plan.kpoints_direct),
            "reference_distance": plan.reference_distance,
        }
    )
    return prepared
```

This task also performs the three prerequisite refactors from the design spec:
add `method_values(method: AbacusMethod) -> Mapping[str, object]` to
`abacus/inputs.py` (returning the `HSE06`/`PBE0`/`CIDER_PBE0` mapping that
`prepare_abacus_input` currently selects inline), change
`prepare_abacus_input(workdir, atoms, assets, method)` to take `atoms`
(`backends/abacus.py` then passes `context.structure.atoms`), and keep the
existing golden gap INPUT files byte-identical — `tests/abacus/test_inputs.py`
is the regression gate for that. `PreparedAbacusInput` stays frozen; only its
`settings` dict is updated.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_band_inputs.py tests/abacus/test_inputs.py -q`
Expected: PASS; existing golden gap INPUT files unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/bandgap_predictor/bands/inputs.py tests/bands
git commit -m "feat: add stage-2 band INPUT and KPT rendering"
```

---

### Task 4: `band.txt` parsing and frontier analysis

**Files:**
- Create: `src/bandgap_predictor/bands/parser.py`
- Create: `tests/bands/fixtures/band_appended.txt`
- Test: `tests/bands/test_band_parser.py`

**Interfaces:**
- Produces: `BandBlock` (frozen dataclass) with `eigenvalues_ev: list[list[float]]`, `klength: list[float]`, `blocks_in_file: int`.
- Produces: `parse_band_txt(path: Path) -> BandBlock` — keeps the last block whose first klength is `0.0`, requires a constant band count, finite values, and equal row count across blocks.
- Produces: `Frontier` (frozen dataclass) with `vbm_ev`, `cbm_ev`, `gap_ev`, `gap_kind` (`"direct"|"indirect"`), `classification` (`ElectronicClassification`), `vbm_kpoint`, `cbm_kpoint`, `vbm_index`, `cbm_index`.
- Produces: `find_frontier(eigenvalues_ev, occupations, plan, *, direct_tolerance=1e-3) -> Frontier` using the same `0.01/0.99` fractional thresholds as `abacus.parser.classify`.
- Consumes: `abacus.parser.classify` semantics, `schemas.ElectronicClassification`, `bands.path.BandPathPlan`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/bands/test_band_parser.py
def test_appended_mesh_block_is_ignored(fixture_plan):
    block = parse_band_txt(FIXTURES / "band_appended.txt")
    assert block.blocks_in_file == 2
    assert len(block.eigenvalues_ev) == len(fixture_plan.kpoints_direct)
    assert block.klength[0] == 0.0


def test_frontier_is_direct_when_extrema_share_a_kpoint(fixture_plan):
    eigenvalues = [[0.0, 1.0], [0.2, 1.2], [0.0, 1.0]]
    frontier = find_frontier(eigenvalues, None, fixture_plan)
    assert frontier.gap_ev == pytest.approx(1.0)
    assert frontier.gap_kind == "direct"


def test_frontier_is_indirect_when_extrema_differ(fixture_plan):
    eigenvalues = [[0.0, 2.0], [0.5, 1.0]]
    frontier = find_frontier(eigenvalues, None, fixture_plan)
    assert frontier.gap_kind == "indirect"
    assert frontier.vbm_index == 0 and frontier.cbm_index == 1


def test_partial_occupancy_forces_zero_gap(fixture_plan):
    eigenvalues = [[0.0, 1.0], [0.0, 1.0]]
    occupations = [[0.5, 0.5], [0.5, 0.5]]
    frontier = find_frontier(eigenvalues, occupations, fixture_plan)
    assert frontier.classification == "partial_occupancy"
    assert frontier.gap_ev == 0.0


def test_truncated_file_raises_parse_error(tmp_path):
    bad = tmp_path / "band.txt"
    bad.write_text("1 0.0 1.0\n2 0.5\n")
    with pytest.raises(ParseError):
        parse_band_txt(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_band_parser.py -q`
Expected: FAIL — missing module.

- [ ] **Step 3: Implement `bands/parser.py`**

```python
"""Parsing of ABACUS band.txt and path frontier analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..errors import ParseError
from ..schemas import ElectronicClassification
from .path import BandPathPlan

FRACTIONAL_LOWER = 0.01
FRACTIONAL_UPPER = 0.99


@dataclass(frozen=True)
class BandBlock:
    eigenvalues_ev: list[list[float]]
    klength: list[float]
    blocks_in_file: int


@dataclass(frozen=True)
class Frontier:
    vbm_ev: float
    cbm_ev: float
    gap_ev: float
    gap_kind: str
    classification: ElectronicClassification
    vbm_kpoint: tuple[float, float, float]
    cbm_kpoint: tuple[float, float, float]
    vbm_index: int
    cbm_index: int


def _rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values = [float(token) for token in stripped.split()]
        except ValueError as error:
            raise ParseError(f"band.txt line {number} is not numeric: {stripped!r}") from error
        if len(values) < 3:
            raise ParseError(f"band.txt line {number} has fewer than three columns")
        rows.append(values)
    if not rows:
        raise ParseError(f"band.txt is empty: {path}")
    return rows


def parse_band_txt(path: Path) -> BandBlock:
    rows = _rows(path)
    band_counts = {len(row) - 2 for row in rows}
    if len(band_counts) != 1:
        raise ParseError("band.txt rows disagree on the number of bands")
    starts = [index for index, row in enumerate(rows) if row[1] == 0.0]
    if not starts:
        raise ParseError("band.txt has no k-line block starting at klength 0")
    first = starts[-1]
    block_rows = rows[first:]
    eigenvalues = [[value for value in row[2:]] for row in block_rows]
    klength = [row[1] for row in block_rows]
    for value in klength + [item for row in eigenvalues for item in row]:
        if not math.isfinite(value):
            raise ParseError("band.txt contains non-finite values")
    return BandBlock(eigenvalues_ev=eigenvalues, klength=klength, blocks_in_file=len(starts))


def find_frontier(
    eigenvalues_ev: list[list[float]],
    occupations: list[list[float]] | None,
    plan: BandPathPlan,
    *,
    direct_tolerance: float = 1e-3,
) -> Frontier:
    """Locate VBM/CBM on the path and classify the gap."""

    if not eigenvalues_ev or not eigenvalues_ev[0]:
        raise ParseError("band path has no eigenvalues")
    if len(eigenvalues_ev) != len(plan.kpoints_direct):
        raise ParseError("band path eigenvalues do not match the planned k-point count")
    bands = len(eigenvalues_ev[0])

    if occupations is None:
        # Without occupations, split at the largest gap across the whole path.
        flat = sorted(value for row in eigenvalues_ev for value in row)
        gaps = [(flat[index + 1] - flat[index], index) for index in range(len(flat) - 1)]
        _, split = max(gaps)
        occupied_bands = split + 1
    else:
        if len(occupations) != len(eigenvalues_ev):
            raise ParseError("band occupations do not match the k-point count")
        fractional = [
            value
            for row in occupations
            for value in row
            if FRACTIONAL_LOWER < value < FRACTIONAL_UPPER
        ]
        if fractional:
            index = max(
                range(len(eigenvalues_ev)),
                key=lambda ik: max(eigenvalues_ev[ik][: len(occupations[ik])]),
            )
            return Frontier(
                vbm_ev=0.0,
                cbm_ev=0.0,
                gap_ev=0.0,
                gap_kind="indirect",
                classification=ElectronicClassification.PARTIAL_OCCUPANCY,
                vbm_kpoint=plan.kpoints_direct[index],
                cbm_kpoint=plan.kpoints_direct[index],
                vbm_index=index,
                cbm_index=index,
            )
        occupied_bands = max(
            1,
            max(
                sum(1 for value in row if value > 0.5)
                for row in occupations
            ),
        )

    if occupied_bands >= bands:
        raise ParseError("band path has no unoccupied band to use as a conduction state")

    vbm_index = max(range(len(eigenvalues_ev)), key=lambda ik: eigenvalues_ev[ik][occupied_bands - 1])
    cbm_index = min(range(len(eigenvalues_ev)), key=lambda ik: eigenvalues_ev[ik][occupied_bands])
    vbm = eigenvalues_ev[vbm_index][occupied_bands - 1]
    cbm = eigenvalues_ev[cbm_index][occupied_bands]
    gap = max(0.0, cbm - vbm)
    same_kpoint = all(
        abs(a - b) <= direct_tolerance
        for a, b in zip(plan.kpoints_direct[vbm_index], plan.kpoints_direct[cbm_index], strict=True)
    )
    return Frontier(
        vbm_ev=vbm,
        cbm_ev=cbm,
        gap_ev=gap,
        gap_kind="direct" if same_kpoint else "indirect",
        classification=(
            ElectronicClassification.INSULATOR if gap > 0.0 else ElectronicClassification.ZERO_GAP
        ),
        vbm_kpoint=plan.kpoints_direct[vbm_index],
        cbm_kpoint=plan.kpoints_direct[cbm_index],
        vbm_index=vbm_index,
        cbm_index=cbm_index,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_band_parser.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/bandgap_predictor/bands/parser.py tests/bands
git commit -m "feat: parse ABACUS band.txt and locate path frontier"
```

---

### Task 5: Band run orchestration and `bands.json`

**Files:**
- Create: `src/bandgap_predictor/bands/artifact.py`
- Create: `src/bandgap_predictor/bands/runner.py`
- Test: `tests/bands/test_band_runner.py`
- Create: `tests/bands/support/fake_abacus.py`

**Interfaces:**
- Produces: `BandRequest` (pydantic) with `structure_path`, `backend`, `abacus_path`, `work_root`, `reference_distance`, `max_kpoints`, `allow_many_kpoints`, `mechanism` (optional override), `symprec`.
- Produces: `BandResult` (pydantic) mirroring the design spec's `bands.json` with `status: RunStatus`.
- Produces: `BandRunner(config_path=None, progress=None)` with `run(request: BandRequest) -> BandResult`, and `write_bands_artifact(run_dir: Path, result: BandResult) -> Path` using `runs._atomic_json`.
- Produces: `select_mechanism(backend: BackendId) -> BandMechanism` (hybrids → `HYBRID_BAND_TARGETS`, `cider-pbe0` → `NSCF_LINE`), the single place the measured Task 7 decision is pinned.
- Consumes: `bands.cell.standardize_cell`, `bands.path.build_band_path`, `bands.inputs.prepare_band_input`, `abacus.inputs.prepare_abacus_input`/`AbacusMethod`, `abacus.runner.run_abacus`, `abacus.parser.parse_abacus_result`, `runs.RunStore`, `assets.AssetRegistry`, `factory.create_service` for the stage-1 result.

Run-directory layout: `<work_root>/<stamp>-<formula>-<backend>-<run8>/` containing
`scf/`, `bands/`, `result.json` (stage 1, written through the existing
`PredictionService`), and `bands.json`.

- [ ] **Step 1: Write the failing tests**

Use `tests/bands/support/fake_abacus.py`, an executable Python script that
mimics the ABACUS process boundary: it writes `OUT.ABACUS/KPT.info`,
`OUT.ABACUS/running_scf.log` (or `running_nscf.log` when the INPUT says
`calculation nscf`), `OUT.ABACUS/band.txt` (deterministic parabolas for the
k-points found in `KPT`), `OUT.ABACUS/bgp_electronic_state.json`, and an
auto-generated `KPT` when the INPUT says `kspacing` is non-zero. It is invoked
through `run_abacus`'s `mpirun`/executable arguments by pointing
`--abacus` at a wrapper that ignores the MPI prefix.

```python
# tests/bands/test_band_runner.py
def test_runner_writes_bands_json_with_frontier(tmp_path, fake_abacus, si_cif):
    result = BandRunner(progress=lambda _: None).run(
        BandRequest(
            structure_path=si_cif,
            backend=BackendId.CIDER_PBE0,
            abacus_path=fake_abacus,
            work_root=tmp_path / "runs",
        )
    )
    assert result.status == "success"
    assert result.mechanism == "nscf_line"
    assert len(result.eigenvalues_ev) == len(result.path.kpoints_direct)
    assert result.frontier.gap_ev > 0.0
    artifact = write_bands_artifact(Path(result.artifact_path), result)
    payload = json.loads(artifact.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["path"]["label_positions"][0] == 0.0
    assert payload["band_cell"]["source"] == "seekpath_primitive"


def test_runner_uses_stage_one_mesh_block(tmp_path, fake_abacus, si_cif):
    result = BandRunner(progress=lambda _: None).run(...)   # hse06
    bands_kpt = (Path(result.artifact_path) / "bands" / "KPT").read_text()
    assert bands_kpt.startswith("K_POINTS\n0\n")
    assert "K_POINTS_BAND" in bands_kpt
    assert "kspacing 0\n" in (Path(result.artifact_path) / "bands" / "INPUT").read_text()


def test_stage_two_failure_is_reported(tmp_path, failing_fake_abacus, si_cif):
    with pytest.raises(RunError) as error:
        BandRunner().run(...)
    assert Path(error.value.artifact_path).is_dir()
    payload = json.loads((Path(error.value.artifact_path) / "bands.json").read_text())
    assert payload["status"] == "failed"


def test_gnn_backend_is_rejected_before_allocating_a_run(tmp_path, si_cif):
    with pytest.raises(InputError, match="gnn-hse06"):
        BandRunner().run(BandRequest(structure_path=si_cif, backend=BackendId.GNN_HSE06, ...))
    assert not (tmp_path / "runs").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_band_runner.py -q`
Expected: FAIL — missing modules.

- [ ] **Step 3: Implement `bands/artifact.py` and `bands/runner.py`**

`artifact.py` holds the pydantic models (`BandRequest`, `BandPathPayload`,
`BandFrontierPayload`, `BandCellPayload`, `BandSymmetryPayload`,
`BandSettingsPayload`, `BandResult`) exactly matching the spec's schema, plus:

```python
def write_bands_artifact(run_dir: Path, result: BandResult) -> Path:
    from ..runs import _atomic_json

    path = Path(run_dir) / "bands.json"
    _atomic_json(path, result.model_dump(mode="json"))
    return path
```

`runner.py` implements the orchestration:

```python
"""Two-stage band-structure orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import numpy as np
from ase.io import write as ase_write

from ..abacus.inputs import AbacusMethod
from ..abacus.parser import load_electronic_state
from ..abacus.runner import run_abacus
from ..assets import AssetRegistry
from ..config import resolve_abacus_path, verify_abacus_executable
from ..errors import BGPError, InputError, ParseError, PreflightError, RunError
from ..factory import create_service
from ..runs import RunDirectory, RunStore, _atomic_json
from ..schemas import BackendId, PredictionRequest, RunStatus
from ..structures import load_structure
from .artifact import BandRequest, BandResult, write_bands_artifact
from .cell import standardize_cell
from .inputs import BandMechanism, prepare_band_input
from .parser import find_frontier, parse_band_txt
from .path import build_band_path

METHOD_FOR_BACKEND: dict[BackendId, AbacusMethod] = {
    BackendId.ABACUS_HSE06: AbacusMethod.HSE06,
    BackendId.ABACUS_PBE0: AbacusMethod.PBE0,
    BackendId.CIDER_PBE0: AbacusMethod.CIDER_PBE0,
}
MECHANISM_BY_BACKEND: dict[BackendId, BandMechanism] = {
    BackendId.ABACUS_HSE06: BandMechanism.HYBRID_BAND_TARGETS,
    BackendId.ABACUS_PBE0: BandMechanism.HYBRID_BAND_TARGETS,
    BackendId.CIDER_PBE0: BandMechanism.NSCF_LINE,
}


def select_mechanism(backend: BackendId) -> BandMechanism:
    """Return the measured band mechanism for a backend."""

    try:
        return MECHANISM_BY_BACKEND[backend]
    except KeyError as error:
        raise InputError(f"{backend.value} does not provide a band structure") from error


class BandRunner:
    """Run stage-1 SCF then stage-2 band targets and publish bands.json."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config_path = config_path
        self.progress = progress if progress is not None else (lambda _message: None)
        self.assets = AssetRegistry.default()
        self.service = create_service(config_path=config_path, progress=self.progress)

    def run(self, request: BandRequest) -> BandResult:
        method = METHOD_FOR_BACKEND.get(request.backend)
        if method is None:
            raise InputError(f"{request.backend.value} does not provide a band structure")
        mechanism = request.mechanism or select_mechanism(request.backend)

        loaded = load_structure(request.structure_path)
        band_cell = standardize_cell(loaded, symprec=request.symprec)
        plan = build_band_path(
            band_cell.atoms.cell.array,
            band_cell.atoms.get_scaled_positions(),
            band_cell.atoms.get_atomic_numbers(),
            reference_distance=request.reference_distance,
            max_kpoints=request.max_kpoints,
            allow_many_kpoints=request.allow_many_kpoints,
            symprec=request.symprec,
        )

        run = RunStore(request.work_root).allocate(loaded, request.backend)
        run.copy_original(loaded)
        _atomic_json(run.path / "request.json", request.model_dump(mode="json"))
        cell_path = run.path / "band-cell.cif"
        ase_write(cell_path, band_cell.atoms)

        started_at = datetime.now(UTC)
        started_clock = monotonic()
        scf_dir = run.path / "scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        scf_result = self.service.predict(
            PredictionRequest(
                structure_path=cell_path,
                backend=request.backend,
                abacus_path=request.abacus_path,
                work_root=run.path,
            ),
            run=RunDirectory(run_id=run.run_id, path=scf_dir),
        )

        executable = resolve_abacus_path(request.abacus_path, self.config_path)
        if executable is None:
            raise PreflightError("ABACUS executable is not configured")
        executable = Path(executable).expanduser().resolve()
        build = verify_abacus_executable(executable)
        backend = self.service.backends[request.backend]

        prepared = prepare_band_input(
            run.path / "bands",
            band_cell.atoms,
            self.assets,
            method,
            mechanism,
            plan,
            scf_kpt=scf_dir / "KPT",
        )
        self.progress(f"Running {method.value} band targets ({mechanism.value}, {len(plan.kpoints_direct)} k-points)")
        try:
            process = run_abacus(executable, prepared.workdir, self.progress)
            if process.returncode != 0:
                raise RunError(f"ABACUS band stage exited with code {process.returncode}")
            block = parse_band_txt(prepared.workdir / "OUT.ABACUS" / "band.txt")
        except BGPError as error:
            failed = self._result(
                request, run, scf_result, plan, band_cell, mechanism,
                status=RunStatus.FAILED, started_at=started_at, started_clock=started_clock,
                error_message=str(error),
            )
            write_bands_artifact(run.path, failed)
            error.artifact_path = run.path
            raise

        if len(block.eigenvalues_ev) != len(plan.kpoints_direct):
            raise ParseError(
                f"band.txt has {len(block.eigenvalues_ev)} k-points, expected {len(plan.kpoints_direct)}"
            )

        state = load_electronic_state(prepared.workdir)
        occupations = None
        fermi = None
        fermi_source = "scf_artifact"
        if state is not None and len(state.get("occupations", [])) == len(plan.kpoints_direct):
            occupations = [[float(value) for value in row] for row in state["occupations"]]
            fermi = float(state["fermi_energy_ev"])
            fermi_source = "bands_artifact"
        if fermi is None:
            fermi = float(scf_result.diagnostics.get("fermi_energy_ev", 0.0))

        frontier = find_frontier(block.eigenvalues_ev, occupations, plan)
        result = self._result(
            request, run, scf_result, plan, band_cell, mechanism,
            status=RunStatus.SUCCESS, started_at=started_at, started_clock=started_clock,
            block=block, occupations=occupations, frontier=frontier,
            fermi=fermi, fermi_source=fermi_source, process=process, build=build,
            prepared=prepared, backend=backend,
        )
        write_bands_artifact(run.path, result)
        return result
```

`_result(...)` assembles the pydantic `BandResult` from the design spec's schema
(frontier block, `path` block straight from `plan`, `scf` block embedding
`scf_result.model_dump(mode="json")`, `settings` from
`{**prepared.settings, "fermi_source": …, "band_txt_blocks": block.blocks_in_file}`,
`provenance` from `backend.provenance(executable, build.model_dump(), prepared)`),
with `error` populated for the failed case.

Two small refactors this task needs, both behaviour-preserving:

1. `PredictionService.predict(self, request, run: RunDirectory | None = None)`
   — use the supplied run directory instead of allocating one
   (`run = run if run is not None else RunStore(request.work_root).allocate(structure, request.backend)`).
2. `abacus/parser.py` gains `load_electronic_state(run_dir: Path) -> dict[str, Any] | None`
   (glob `OUT.*/bgp_electronic_state.json`, return the payload when the schema
   version is `1.0`, else `None`); `parse_abacus_result` calls it, so the band
   stage gets a tolerant reader that does not require `running_scf.log`. Also
   rename `AbacusBackend._provenance` to `AbacusBackend.provenance` so stage 2
   records the identical provenance block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands -q`
Expected: PASS (all band tests, no hardware required).

- [ ] **Step 5: Commit**

```bash
git add src/bandgap_predictor/bands tests/bands
git commit -m "feat: orchestrate two-stage band runs and bands.json"
```

---

### Task 6: `bgp bands` CLI and envelope

**Files:**
- Modify: `src/bandgap_predictor/cli.py`
- Modify: `docs/frontends.md`
- Test: `tests/test_cli_bands.py`

**Interfaces:**
- Produces: `bgp bands STRUCTURE --backend … [--reference-distance F] [--max-kpoints N] [--allow-many-kpoints] [--mechanism {hybrid_band_targets,nscf_line}] [--plot PATH] [--abacus P] [--config P] [--work-root R] [--json]`; `--plot` is accepted here and its renderer is implemented in Task 9.
- Consumes: `bands.runner.BandRunner`, `bands.artifact.BandRequest`, the Task 1 envelope helper.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_bands.py
def test_bands_json_envelope_embeds_payload(monkeypatch, si_cif):
    monkeypatch.setattr(cli, "BandRunner", StubBandRunner)
    result = CliRunner().invoke(main, ["bands", str(si_cif), "--backend", "cider-pbe0", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True and payload["command"] == "bands"
    assert payload["bands"]["schema_version"] == "1.0"
    assert payload["result"]["backend"] == "cider-pbe0"


def test_bands_rejects_gnn_with_unsupported_kind(si_cif):
    result = CliRunner().invoke(main, ["bands", str(si_cif), "--backend", "gnn-hse06", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"]["kind"] == "unsupported"


def test_bands_human_output_lists_frontier(si_cif, monkeypatch):
    monkeypatch.setattr(cli, "BandRunner", StubBandRunner)
    result = CliRunner().invoke(main, ["bands", str(si_cif), "--backend", "abacus-hse06"])
    assert result.exit_code == 0
    assert "Gap kind: indirect" in result.stdout
    assert "Artifacts:" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_bands.py -q`
Expected: FAIL — no `bands` command.

- [ ] **Step 3: Implement the command**

Add the `bands` click command next to `predict`, with `_emit_bands_result`
(human) and `--json` reusing the Task 1 envelope builder with
`command="bands"`, `result=stage_one_result`, `bands=payload`. Map
`InputError` from the GNN rejection to `error.kind = "unsupported"` (extend the
envelope's kind mapping with an `UnsupportedBackendError(InputError)` class so
the mapping stays declarative).

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_cli_bands.py tests/test_cli.py tests/test_envelope.py -q`
Expected: PASS.

- [ ] **Step 5: Document and commit**

Extend `docs/frontends.md` with the `bands` command, its options, the embedded
`bands` payload, and the exit-code/kind table addition. Commit
`feat: add bgp bands command`.

---

### Task 7: Measure mechanisms on Si and pin the decision

**Files:**
- Create: `tests/bands/test_band_mechanisms.py` (hardware-marked)
- Modify: `src/bandgap_predictor/bands/runner.py` (pin `select_mechanism`)
- Modify: `docs/methods.md`, `docs/results.md`
- Create: `build/band-runs/*` (retained evidence, ignored by git)

- [ ] **Step 1: Run the CIDER path end to end (hardware)**

```bash
OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp bands examples/Si.cif \
  --backend cider-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu \
  --work-root build/band-runs --json > /tmp/bands-cider-si.json
```

Expected: exit 0; `bands.json` with `status=success`, `mechanism=nscf_line`,
`len(eigenvalues_ev) == len(path.kpoints_direct)`, finite `fermi_energy_ev`,
`frontier.gap_ev >= 0`, `settings.band_txt_blocks == 1`, and stage-1
`result.json` gap equal to the CIDER Si baseline for the primitive cell
(record the number; it is expected to differ from the conventional-cell
`1.3119157 eV` and that difference must be stated in `docs/results.md`).

If CIDER `nscf` fails or produces no CIDER-corrected eigenvalues, stop and go to
Task 8.

- [ ] **Step 2: Cross-check our k-point list against ABACUS**

```bash
python - <<'PY'
import json, re, pathlib
run = pathlib.Path(json.load(open('/tmp/bands-cider-si.json'))['artifact_path'])
kpt_info = (run / 'bands' / 'OUT.ABACUS' / 'KPT.info').read_text().splitlines()
abl = [tuple(round(float(v), 6) for v in line.split()[1:4]) for line in kpt_info if re.match(r'^\s+\d+\s+-?\d', line)]
payload = json.loads((run / 'bands.json').read_text())
ours = [tuple(round(float(v), 6) for v in k) for k in payload['path']['kpoints_direct']]
print('abacus', len(abl), 'ours', len(ours), 'match', abl == ours)
PY
```

Expected: `match True` (if the wrap convention differs, normalise both sides
into `[-0.5, 0.5)` and record the convention in `bands.json.settings`).

- [ ] **Step 3: Measure the hybrid mechanism on Si (hardware)**

```bash
OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp bands examples/Si.cif \
  --backend abacus-hse06 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu \
  --work-root build/band-runs --mechanism hybrid_band_targets --json > /tmp/bands-hse06-si.json
```

Expected: exit 0 and `settings.band_txt_blocks == 1`. Record elapsed time from
`bands.json`. Then repeat once with `--mechanism nscf_line` and record both
elapsed times.

- [ ] **Step 4: Pin the measured decision**

Set `select_mechanism` to the measured winner per backend (hybrids →
`hybrid_band_targets` unless Step 3 failed, then `nscf_line`), and add a
hardware test asserting the pinned mechanism and `band_txt_blocks == 1` for one
Si run per backend. Record in `docs/methods.md`: mechanism per backend, the
measured Si elapsed times, the k-point count, and the explicit statement that
hybrid path solves run at `kpar=1` with no symmetry reduction of the path.

- [ ] **Step 5: Commit**

```bash
git add src/bandgap_predictor/bands/runner.py tests/bands docs/methods.md docs/results.md
git commit -m "feat: pin measured band mechanisms and record Si evidence"
```

---

### Task 8: CIDER fallback investigation (only if Task 7 Step 1 failed)

**Files:**
- Read: `vendor/abacus/source/source_esolver/esolver_ks_pw.cpp:1310-1520`,
  `vendor/abacus/source/source_pw/module_pwdft/hamilt_pw.h`,
  `vendor/abacus/source/source_pw/module_pwdft/op_pw_exx.h`
- Modify (only on approval): the above + `tools/build_abacus.py` invocation,
  `tests/hardware/baselines.json`, `docs/results.md`, release manifests

- [ ] **Step 1: Determine whether a non-EXX band-target path is feasible**

Read how `HamiltPW` uses `source_exx` and `exx_options` when
`exx_options.enabled == false`, and whether a null/disabled EXX operator can be
passed. Write the finding (with file:line evidence) into
`docs/superpowers/specs/2026-10-13-band-structure-design.md` under a new
"CIDER mechanism decision" section.

- [ ] **Step 2: Decide and record**

If a null-EXX `HamiltPW` is valid, propose the concrete diff (new
`solve_band_targets_plain` mirroring `solve_mixed_band_targets_full` without
`source_exx`), the rebuild command, and the re-baselining cost; **do not
execute** without an explicit user decision, because it invalidates the
released executable hash. Otherwise declare `cider-pbe0` bands unsupported in
v1: `bgp bands --backend cider-pbe0` returns `error.kind = "unsupported"`, and
`docs/methods.md` states why (EXX-gated band targets, non-EXX path not
available without a vendor change).

- [ ] **Step 3: Commit the decision** (documentation and, if applicable, the
  `unsupported` mapping and its tests).

---

### Task 9: Plots, documentation, and release record

**Files:**
- Create: `src/bandgap_predictor/bands/plot.py`
- Modify: `src/bandgap_predictor/cli.py` (`--plot`)
- Modify: `docs/methods.md`, `docs/results.md`, `README.md`
- Create: `../releases/2026-10-13-bgp-bands-v0.2.0.yaml`, modify `../progress.md` (in `/home/bonan/work/HC-Report`)
- Verify: `docs/superpowers/plans/2026-10-13-bandgap-web.md` already points its
  frontend tasks at this plan (no band work is duplicated there)

**Interfaces:**
- Produces: `plot_bands(payload: BandResult, target: Path) -> Path` writing PNG
  or PDF by suffix, x = `path.cumulative_distance` with special-point ticks,
  y = energy relative to `frontier.vbm_ev`, Fermi line, gap annotation, one
  line break per `segment_breaks` entry.
- Consumes: `BandResult`, matplotlib Agg.

- [ ] **Step 1: Write the failing test**

```python
def test_plot_writes_png_with_expected_axes(tmp_path, si_bands_payload):
    target = plot_bands(si_bands_payload, tmp_path / "bands.png")
    assert target.is_file() and target.stat().st_size > 10_000
    image = plt.imread(target)
    assert image.ndim == 3
```

- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands/test_plot.py -q`.

- [ ] **Step 3: Implement `bands/plot.py`** and wire `--plot PATH` into the CLI
  (documented: the plot is a convenience rendering of the same data the web
  frontend draws from `bands.json`; it is never the source of truth).

- [ ] **Step 4: Verify GREEN and real artifacts**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands -q
OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run bgp bands examples/Si.cif \
  --backend cider-pbe0 --abacus build/abacus-install-fixed/bin/abacus_pw_gpu \
  --work-root build/band-runs --plot /tmp/si-bands.png
```

Expected: PNG written; visually inspected once by opening the file; frontier
label matches `bands.json`.

- [ ] **Step 5: Update docs and the coordination repo, then commit**

`docs/methods.md`: the band workflow, the primitive-cell consequence, mechanism
per backend, cost, `kpar=1`, no SOC/spin, `reference_distance` semantics, the
"SCF mesh eigenvalues are not a band structure" statement.
`docs/results.md`: Si (and any additional measured systems) band evidence with
run directories and gaps, clearly separated from the gap baselines.
`README.md`: `bgp bands` quick-start.
HC-Report release manifest: package commit, `bands.json` schema version,
mechanism decisions, evidence run directories, executable hash, test counts.
Web plan: its Task 7 consumes this artifact through `GET /api/runs/{id}/bands.svg`
and never re-derives k-paths or frontiers.

```bash
git add src tests docs README.md
git commit -m "feat: add band plots and document band mode"
```

---

### Task 10: Final verification

**Files:** verify tracked sources and retained artifacts.

- [ ] **Step 1: Full suite and lint**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: the pre-existing 66 passed / 12 skipped plus all new band, path, cell,
parser, runner and CLI tests; hardware tests skipped without `BGP_ABACUS`.

- [ ] **Step 2: Hardware band suite against the pinned mechanisms**

```bash
BGP_ABACUS=build/abacus-install-fixed/bin/abacus_pw_gpu BGP_HARDWARE_RUN_ROOT=build/band-verification \
  OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/bands -m hardware -q
```

Expected: one passing case per supported backend (or `cider-pbe0` unsupported if
Task 8 concluded that), each asserting `band_txt_blocks == 1`, a finite
frontier, and the planned k-point count.

- [ ] **Step 3: Contract audit**

```bash
grep -rn "kspacing" src/bandgap_predictor/abacus/inputs.py src/bandgap_predictor/bands/inputs.py
git status --short
git diff --stat HEAD~10
```

Expected: the locked preset still says `kspacing 0.15`; only stage-2 mappings
say `kspacing 0`; no change to `tests/abacus/golden/*_INPUT`, no change to
`vendor/abacus` (unless Task 8 was explicitly approved), no change to historical
result directories, and the executable hash used by all evidence runs equals
`9ad2d03bbb4f6fa8a56384a654f744fd23bd563872555b105b44f7efe544fb2e`.
