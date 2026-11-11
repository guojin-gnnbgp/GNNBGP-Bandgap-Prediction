# Band Structure Mode Design (`bgp bands`)

## Objective

Add a first-class band-structure capability to the `bgp` CLI: given a periodic
crystal structure and one ABACUS backend, compute eigenvalues along a
high-symmetry k-path and write a machine-readable `bands.json` artifact
(k-points, labels, cumulative distance, eigenvalues, occupations, Fermi level,
frontier/VBM/CBM/gap) plus optional PNG/PDF plots. `gnn-hse06` does not support
bands.

The web frontend consumes `bands.json` and never re-derives crystallography. No vendored
ABACUS change and no GPU rebuild are required for the primary design, so the
released executable hash `9ad2d03b…` and the existing gap baselines stay valid.

## Verified mechanism facts

All verified in this checkout on 2026-10-13 (`vendor/abacus` of commit
`2fbfd26`):

| Fact | Evidence |
| --- | --- |
| Native hybrid band path exists: one `calculation scf` process runs SCF, then solves the `K_POINTS_BAND` targets reusing the converged EXX operator, writing `band.txt` | `vendor/abacus/docs/advanced/elec_properties/band.md`; `esolver_ks_pw.cpp:1320-1420` (`solve_mixed_band_targets_full`), `:1503` (`write_bands`) |
| Its gates: `calculation scf`, `basis_type pw`, `out_band 1`, EXX enabled **and** `exxace 1` **and** `exx_separate_loop 1`, `kpar <= 1` | `esolver_ks_pw.cpp:800-850` (`validate_mixed_band_targets`) |
| EXX is enabled only for `hf/pbe0/b3lyp/hse/scan0/…`; `dft_functional PBE` (CIDER) → `cal_exx = false` | `input_conv.cpp:429-537`; `make_exx_operator_options` at `esolver_ks_pw.cpp:66-91` |
| The band-target solve is EXX-specific: it requires `exx_helper->get_op_exx()` and passes `source_exx` + `exx_options_` into `HamiltPW` | `esolver_ks_pw.cpp:1310-1340` |
| `exx_separate_loop` default is `True`; `out_app_flag` default is `True` (band writes **append** to `band.txt`) | `read_input_item_exx_dftu.cpp:166-176`; `read_input_item_output.cpp:1085-1093` |
| `kspacing > 0` makes rank 0 **overwrite** the `KPT` file with an auto mesh before reading it | `klist.cpp:681-725` |
| Line-mode main KPT (`Line_Direct`) sets `wk = 1.0` for every k-point and requires `symmetry 0` or `-1` | `klist.cpp:860-880` |
| `K_POINTS_BAND` block accepts `Line`/`Line_Direct`/`Line_Cartesian`/`Direct`/`Cartesian`; its band-target weights are `1/nk` | `band.md`; `klist.cpp:350-400` |
| Line-mode special-point format: first number = number of special points, then `kx ky kz n` per point, interpolation `A + i·(B−A)/n` for `i = 0…n−1`, last point `n = 1` | `klist.cpp:904-960` |
| `nbands` auto rule for `nspin=1`: `max(occupied + 10, int(1.2·occupied) + 1)` | `cal_nelec_nband.cpp:62-92` (Si8: 16 occupied → 26 bands, matches the retained run) |
| `calculation nscf` requires a charge density and defaults `init_chg file`; `nscf` logs are not `running_scf.log` | `read_input_item_system.cpp:84, 442-455`; `bgp_electronic_state.json` writer lives in `esolver_ks_pw.cpp:1259-1281` |
| `seekpath` 2.2.2 + `spglib` 2.8.0 installed and working: `examples/Si.cif` → Fd-3m #227, primitive a=3.84 Å, path `GAMMA-X-W-K-GAMMA-L-U-W-L-K|U-X`, 202 k-points at `reference_distance=0.025` | `/tmp/sp-check` probe run 2026-10-13 |
| `explicit_kpoints_linearcoord` / `reciprocal_primitive_lattice` are self-consistent in Å⁻¹ (2π convention) | same probe (step 0.0257 Å⁻¹ over `Γ→X`, |ΓX| ≈ 1.03 Å⁻¹) |

## Scope

In scope: `seekpath`-driven path planning, primitive-cell standardization,
two-stage band workflow for `abacus-hse06` / `abacus-pbe0` / `cider-pbe0`,
`band.txt` parsing, `bands.json`, `bgp bands` CLI + envelope, optional
`--plot` PNG/PDF, tests, documentation, web-frontend handoff.

Out of scope: SOC, spin polarization, DOS/projected bands (`PBAND`), Wannier,
relaxation, refit, arbitrary ABACUS overrides, and frontend code (see
[`2026-10-13-bandgap-web-design.md`](2026-10-13-bandgap-web-design.md)).

## Cell handling

`bgp bands` computes on the **seekpath-standardized primitive cell**
(`primitive_lattice` + `primitive_positions`/`primitive_types` from
`seekpath.get_explicit_k_path`), because seekpath's path coordinates and
`linearcoord` are only valid in that cell's reciprocal basis. The submitted
structure is never modified on disk.

Provenance must record both: the submitted structure (`source_path`,
`original_sha256`, `structure_sha256`, `formula`) and the band cell
(`cell_angstrom`, `symbols`, `positions_angstrom`, `structure_sha256` of the
generated `STRU`), plus `seekpath`/`spglib` versions, spacegroup number and
symbol, `bravais_lattice`, `primitive_transformation_matrix`,
`inverse_primitive_transformation_matrix` and `volume_original_wrt_prim` (the
keys `get_explicit_k_path` actually returns; it does not return
`transformation_matrix`/`origin_shift`). The primitive cell is written to
`<run>/band-cell.cif` and its fractional positions are wrapped into `[0, 1)`
(seekpath can return values such as `1.25`); wrapping is a coordinate change of
an equivalent cell and is recorded in `bands.json.band_cell`.

Documented consequence: the stage-1 SCF gap on the primitive cell is not
bit-identical to `bgp predict`'s gap on the submitted cell (same physics,
different cell and k-mesh). `bands.json` records both the stage-1 gap and the
path gap; neither is presented as a replacement for the other.

## k-path planning

`bands/path.py` produces a `BandPathPlan`:

```python
@dataclass(frozen=True)
class BandPathPlan:
    labels: list[str]            # e.g. ["GAMMA", "X", "W", ...] one per special point
    special_kpoints_direct: list[tuple[float, float, float]]  # primitive reciprocal basis
    segment_counts: list[int]    # n per special point, last == 1 (ABACUS line semantics)
    kpoints_direct: list[tuple[float, float, float]]          # fully expanded, ABACUS order
    cumulative_distance: list[float]   # Å⁻¹, 2π convention, resets at segment breaks
    label_positions: list[float]       # cumulative_distance at each labelled special point
    label_names: list[str]             # parallel to label_positions (Γ rendered as "GAMMA")
    cartesian_kpoints: list[tuple[float, float, float]]       # Å⁻¹
```

Rules:

- Source: `seekpath.get_explicit_k_path(cell, with_time_reversal=True,
  reference_distance=REFERENCE_DISTANCE, symprec=…, angle_tolerance=…)`.
- `REFERENCE_DISTANCE` default `0.025` Å⁻¹ (seekpath's own default); CLI
  `--reference-distance` overrides it. Coarse `0.05`, dense `0.0125` are the
  documented frontend presets.
- Segment counts are derived from each segment's Cartesian length divided by
  the reference distance, clamped to `>= 1`, with the last special point
  count forced to `1`. The expanded k-point list replicates ABACUS
  interpolation exactly, so it can be cross-checked against
  `OUT.ABACUS/KPT.info` from a real run.
- `cumulative_distance` is computed from the expanded Cartesian k-points
  (`|Δk|` accumulated), resetting at segment boundaries (never carrying the
  jump between two disconnected path pieces).
- `label_positions` marks each special point; seekpath's `GAMMA` is rendered as
  `GAMMA` in JSON and as `Γ` only by frontends.
- Cost guard: `--max-kpoints` (default `400`); exceeding it raises
  `InputError` unless `--allow-many-kpoints` is passed.

## Workflow

Two stages inside one run directory allocated by `RunStore`:

```
<run>/                    band run directory (RunStore.allocate)
  request.json            band request
  input/<original>        byte-exact submitted structure
  band-cell.cif           the seekpath primitive cell that is actually computed
  scf/                    stage-1 run directory AND ABACUS workdir
    INPUT KPT STRU pseudos/ [models/] OUT.ABACUS/ process.json stdout.log
    result.json           stage-1 public result, written by PredictionService
  bands/                  stage-2 ABACUS workdir
    INPUT KPT STRU pseudos/ [models/] OUT.ABACUS/ process.json stdout.log
  bands.json              the band artifact (references scf/ and bands/)
```

There is no second top-level `result.json`: the gap result stays the stage-1
`PredictionResult` at `scf/result.json`, produced by the existing, tested
service path. `bands.json.scf` embeds the same payload for frontends.

### Package refactors (small, prerequisite)

1. `PredictionService.predict(request, run: RunDirectory | None = None)` — use
   the provided `RunDirectory` when given instead of allocating one, so stage 1
   can write into `<run>/scf` while keeping all existing behaviour and tests.
2. `abacus.inputs.prepare_abacus_input(atoms: Atoms, workdir, assets, method)` —
   the function only ever reads `structure.atoms`, so taking `atoms` directly
   lets stage 2 stage the primitive cell without a fake `LoadedStructure`.
   `backends/abacus.py` passes `context.structure.atoms`.
3. `abacus.inputs.method_values(method: AbacusMethod) -> Mapping[str, object]` —
   exposes the locked per-method INPUT mapping so stage 2 can layer its
   overrides on the same values instead of re-deriving them.

### Stage 1 — locked SCF (unchanged)

Reuses `prepare_abacus_input` + `run_abacus` + `parse_abacus_result` on the
primitive cell, exactly as `bgp predict` does today (preset `1.1` for the
hybrids, `1.2` for CIDER, `kspacing 0.15`, `symmetry 1`, `out_bgp_result 1`).
ABACUS auto-writes the mesh `KPT`; stage 2 copies that mesh block verbatim so
the package never re-implements ABACUS's `kspacing` mesh formula
(`klist.cpp:693-720`).

### Stage 2 — mechanism A (hybrids, preferred)

```
INPUT_PARAMETERS            # hybrid preset, plus:
calculation scf             # required by validate_mixed_band_targets
kspacing 0                  # otherwise ABACUS overwrites our KPT
init_chg file
read_file_dir ../scf/OUT.ABACUS
exx_separate_loop 1         # pinned; required by the band-target gate
out_app_flag false          # band.txt must contain only the path block
out_band 1                  # already in the preset

KPT                         # mesh block copied from scf/KPT + band block
K_POINTS
0
Gamma
n1 n2 n3 0 0 0

K_POINTS_BAND
<n_special>
Line_Direct
kx ky kz n
...
```

Cost: one SCF starting from the stage-1 density (few iterations) plus the path
diagonalization. The path solve reuses the converged EXX operator.

### Stage 2 — mechanism B (CIDER, and hybrid fallback)

```
INPUT_PARAMETERS            # CIDER preset, plus:
calculation nscf
kspacing 0
symmetry 0                  # required for line mode
init_chg file
read_file_dir ../scf/OUT.ABACUS
out_app_flag false

KPT                         # main block IS the path
K_POINTS
<n_special>
Line_Direct
kx ky kz n
...
```

Mechanism B is required for CIDER because the band-target path is EXX-gated
(`validate_mixed_band_targets` quits when `exx_options.enabled` is false).

Mechanism selection is **measured, then pinned**: Task 7 of the implementation
plan runs both mechanisms on Si and records which one is used per backend in
`docs/methods.md`, `bands.json.settings["mechanism"]`, and a test constant. If
mechanism A fails for hybrids, hybrids fall back to B (allowed by the design,
with the cost recorded).

### Failure handling

- Stage 1 failure → `RunError`/`ParseError` as today; `scf/result.json`
  retained; no `bands.json`.
- Stage 2 process failure → `RunError`, `bands.json` with
  `status="failed"`, `error` message, and the retained stage-2 directory.
- Stage 2 convergence/marker failure → `RunError` with the log excerpt.
- Missing/short/NaN `band.txt` → `ParseError`.
- `band.txt` containing an appended mesh block (mechanism A with
  `out_app_flag true`, or a future ABACUS change) → the parser takes the **last
  block whose klength restarts at 0** and records
  `settings["band_txt_blocks"]` so the choice is visible.
- Unsupported backend (`gnn-hse06`) → `InputError`-class `unsupported`
  envelope, exit code 2, no run directory allocated.

## `band.txt` parsing

Format (`write_bands.cpp:100-140`): `index klength E_1 … E_nband`, energies in
eV with `eshift = 0` (absolute), `klength` computed from Cartesian k-vectors and
reset at k-line segment boundaries.

Parser rules:

1. Read the file; keep the rows of the last klength-reset block.
2. Require: row count == `len(plan.kpoints_direct)`, constant band count,
   all energies finite.
3. Occupations come from stage-2 `OUT.*/bgp_electronic_state.json` when its
   k-point count matches the path (line mode: `k_weights = 1.0`; band targets:
   `1/nk`; both positive, so the existing normalization contract holds), else
   occupations are absent and the frontier is derived from the gap structure of
   the eigenvalue matrix.
4. Fermi level: use the artifact's `fermi_energy_ev` when finite and
   non-zero, else the stage-1 SCF Fermi, recording
   `settings["fermi_source"] ∈ {"bands_artifact", "scf_artifact"}`.
5. x-axis: `cumulative_distance` from the plan (authoritative). `band.txt`
   klength is only checked for monotonicity within segments; the
   Å⁻¹↔ABACUS-unit factor is measured once in Task 7 and recorded.

## Frontier and gap

`frontier` is computed from the path data with the existing classification
vocabulary:

- `vbm_ev` = max over the path of the highest occupied eigenvalue;
- `cbm_ev` = min over the path of the lowest unoccupied eigenvalue;
- `gap_ev` = `cbm_ev − vbm_ev` (clamped at 0);
- `gap_kind` = `"direct"` when VBM and CBM occur at the same k-point (within a
  tolerance) else `"indirect"`;
- `vbm_kpoint` / `cbm_kpoint` = the direct coordinates, plus the label of the
  nearest labelled special point when within tolerance;
- occupations (when present) decide occupied/unoccupied with the same
  `0.01/0.99` fractional thresholds as `parser.classify`; `classification` uses
  the same enum values, and a materially fractional occupation set yields
  `partial_occupancy` with `gap_ev = 0.0`, exactly like the SCF contract.

## `bands.json`

```json
{
  "schema_version": "1.0",
  "run_id": "…",
  "status": "success",
  "backend": "cider-pbe0",
  "method": "CIDER-PBE0",
  "mechanism": "nscf_line",
  "converged": true,
  "submitted_structure": {"source_path": "…", "formula": "Si8",
                          "original_sha256": "…", "structure_sha256": "…"},
  "band_cell": {"cell_angstrom": [[…]], "symbols": ["Si", "Si"],
                "positions_angstrom": [[…]], "structure_sha256": "…",
                "source": "seekpath_primitive"},
  "symmetry": {"seekpath_version": "2.2.2", "spglib_version": "2.8.0",
               "spacegroup_number": 227, "spacegroup_international": "Fd-3m",
               "bravais_lattice": "cF2",
               "primitive_transformation_matrix": [[…]],
               "inverse_primitive_transformation_matrix": [[…]],
               "volume_original_wrt_prim": 4.0},
  "path": {"reference_distance": 0.025, "labels": ["GAMMA", "X", "…"],
           "label_positions": [0.0, 1.03, …], "segment_counts": [41, 41, …],
           "segment_breaks": [83], "kpoints_direct": [[0, 0, 0], …],
           "cumulative_distance": [0.0, …], "kpoints_cartesian": [[0, 0, 0], …]},
  "fermi_energy_ev": 6.36438486430133,
  "frontier": {"vbm_ev": 6.24, "cbm_ev": 6.48, "gap_ev": 0.24,
               "gap_kind": "indirect", "classification": "insulator",
               "vbm_kpoint": [0.0, 0.0, 0.0], "cbm_kpoint": [0.4, 0.0, 0.0],
               "vbm_label": "GAMMA", "cbm_label": null},
  "eigenvalues_ev": [[…]],
  "occupations": [[…]],
  "scf": {"artifact_path": "…", "gap_ev": 1.3119157, "converged": true,
          "prediction": {"gap_ev": 1.3119157, "classification": "insulator"}},
  "settings": {"preset_version": "1.2", "mechanism": "nscf_line",
               "fermi_source": "bands_artifact", "kpoint_count": 202,
               "band_count": 14, "nbands_rule": "auto",
               "band_txt_blocks": 1, "kspacing": 0.0,
               "symmetry": 0, "calculation": "nscf", "out_app_flag": 0},
  "provenance": {"executable_sha256": "…", "source_revisions": ["…"],
                 "pseudopotentials": {"Si": "…"}, "cider_model_sha256": null},
  "warnings": [],
  "started_at": "…", "finished_at": "…", "elapsed_seconds": 0.0,
  "artifact_path": "…"
}
```

Written atomically like `result.json` (`runs._atomic_json`).

## CLI contract

```
bgp bands STRUCTURE --backend {abacus-hse06,abacus-pbe0,cider-pbe0}
    [--reference-distance F] [--max-kpoints N] [--allow-many-kpoints]
    [--plot PATH] [--abacus P] [--config P] [--work-root R] [--json]
```

- Human output mirrors `predict` plus a frontier line and the artifact path.
- `--json` prints one envelope with `command: "bands"`, the stage-1
  `result` payload, and `bands` embedded (small enough: 202 k × 14 bands ≈ 60 KB).
- `--plot PATH` writes PNG or PDF by suffix via matplotlib (Agg); the plotting
  code lives in `bands/plot.py` and is reused by the web frontend's SVG endpoint.
- Exit codes: `InputError` 2 (bad structure, bad labels, k-point guard,
  unsupported backend), `PreflightError` 3, `RunError` 4, `ParseError` 5,
  interrupt 130 — unchanged vocabulary.
- Hybrid cost warning on stderr: the number of path k-points, the mechanism,
  and the note that hybrid path solves are expensive at `kpar=1`.

## Testing strategy

- Unit: path planning (labels, counts, distances, guard), primitive-cell
  standardization (Si/InSb examples → Fd-3m/F-43m), KPT/INPUT rendering
  (golden files for both mechanisms), `band.txt` parsing (fixture with an
  appended mesh block), frontier/gap/direct-vs-indirect, `bands.json` schema.
- Contract: `bgp bands --json` envelope for success, unsupported backend, bad
  structure, and k-point guard; no stdout pollution.
- Hardware (marked `hardware`, skipped without `BGP_ABACUS`): one Si run per
  backend; assert `len(eigenvalues_ev) == len(kpoints_direct)`, finite
  frontier, `band.txt` block count 1, and that the stage-1 gap reproduces the
  retained SCF baseline for the same cell (documented tolerance because the
  cell differs).
- Evidence: retained run directories under `build/band-runs/` plus the measured
  mechanism/cost record in `docs/methods.md`.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Mechanism A needs `calculation scf` + `init_chg file`, so a second SCF runs; behaviour with a pre-converged density is unverified | Task 7 measures it on Si; fallback is mechanism B |
| CIDER `nscf` may not apply the CIDER model to the read density, or may be rejected | Task 7 measures it; if it fails, Task 8 investigates a non-EXX band-target path in `ESolver_KS_PW` (mirroring `solve_mixed_band_targets_full` without `source_exx`) and otherwise declares `cider-pbe0` bands unsupported in v1 |
| `band.txt` append semantics change (`out_app_flag` is a general flag) | `out_app_flag false` in stage 2 **and** last-block parsing, with `settings["band_txt_blocks"]` recorded |
| ABACUS interpolation mismatch vs our expanded k list | our list is cross-checked against `OUT.ABACUS/KPT.info` in Task 7; mismatch fails the run |
| Hybrid path solves are expensive at `kpar=1` | `--max-kpoints` guard, stderr cost warning, `reference_distance` control, measured cost in docs |
| Primitive-cell standardization changes the reported gap vs `bgp predict` | both gaps recorded; documented, never merged |
| seekpath/spglib add weight to the base install | `seekpath>=2.1.0`, `spglib>=2.5.0` added to base dependencies (small, pure/C wheels) since `bgp bands` is a first-class command |
| MPI rank-0-only `KPT` rewriting means stage 2 must run after stage 1 in the same cell | stage 2 reuses stage 1's mesh block verbatim; `kspacing 0` asserted in golden INPUT |
