# Bandgap Web Frontend Implementation Plan (`bgp serve`)

> **状态：已暂存（PARKED，2026-10-15）。** 本计划不在当前工作范围内执行。
> 前端工作暂时搁置，优先交付 GUI 团队的交接文档：见 `docs/handover/`。
> 恢复执行时，取消本横幅并按其前置依赖（`2026-10-13-band-structure.md`）推进。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a browser frontend for the bandgap predictor: `bgp serve` hosts a
FastAPI bridge plus a single-page UI that loads a structure, picks a method,
renders the cell in 3D (3Dmol.js), runs `bgp predict` / `bgp bands` as
subprocesses, streams progress over SSE, and draws results and bands from the
CLI's JSON artifacts.

**Architecture:** The CLI gains the machine-readable contract this frontend
needs (`--json` envelope, `structure show`, `runs list/show`); `bgp serve` is a
thin bridge in the main package (optional `web` extra) that spawns the CLI with
argv lists, keeps a one-job-at-a-time GPU queue, fans stderr out as SSE, and
serves vendored static assets. The band computation is owned by
[`2026-10-13-band-structure.md`](2026-10-13-band-structure.md).

**Tech Stack:** Python 3.12, uv, click, pydantic, pytest, FastAPI + uvicorn
(optional extra), vendored 3Dmol.js, matplotlib (band SVG), ABACUS PW CUDA+MPI.

Design authority: [`../specs/2026-10-13-bandgap-web-design.md`](../specs/2026-10-13-bandgap-web-design.md).

## Global Constraints

- The bridge MUST NOT import `bandgap_predictor.service/backends/abacus/bands`
  for computation; it spawns the CLI and parses one JSON envelope from stdout.
- argv lists only; never a shell string. Uploaded names are sanitized; uploads
  land under `<work_root>/uploads`.
- Default bind `127.0.0.1`; `--host` to expose; `--token` enforced when set;
  `/api/fs` disabled unless `--allow-fs`.
- One GPU job at a time (presets lock one MPI rank and one OpenMP thread).
- Base install keeps `ase/click/numpy/platformdirs/pydantic` (+ band deps);
  FastAPI/uvicorn live in the `web` extra and `bgp serve` fails with an
  actionable `PreflightError` when they are missing.
- Frontend assets are vendored (no CDN at runtime); 3Dmol.js is BSD-3-Clause
  and its notice is recorded under `LICENSES/` and `assets/NOTICE.md`.
- Locked scientific presets stay byte-for-byte unchanged; band stage-2 mappings
  are owned by the band plan.
- All documented hardware runs use
  `build/abacus-install-fixed/bin/abacus_pw_gpu` (sha256 `9ad2d03b…`) with
  `OMP_NUM_THREADS=1`.
- `uv run pytest -q` baseline before this plan: 66 passed, 12 skipped.

---

### Task 1: Machine-readable CLI envelope

**Files:**
- Create: `src/bandgap_predictor/envelope.py`
- Modify: `src/bandgap_predictor/cli.py`
- Create: `tests/test_envelope.py`
- Modify: `tests/test_cli.py`
- Create: `docs/frontends.md`

**Interfaces:**
- Produces: `envelope.build(*, command, argv, result=None, bands=None, error=None, interrupted=False, artifact_path=None) -> dict[str, object]` and `envelope.error_kind(error: BaseException) -> str` (`input|preflight|run|parse|interrupted|unsupported`).
- Produces: `--json` on `predict`, `backends`, `doctor`, `assets verify`: exactly one JSON object on stdout, human text and progress on stderr, exit codes unchanged.
- Consumes: existing `BGPError.exit_code`, `_emit_result`, `_emit_error`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_envelope.py
import json
from click.testing import CliRunner

from bandgap_predictor.cli import main

SI = "examples/Si.cif"


def test_predict_json_is_one_object():
    result = CliRunner().invoke(main, ["predict", SI, "--backend", "gnn-hse06", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["command"] == "predict"
    assert payload["result"]["backend"] == "gnn-hse06"
    assert payload["error"] is None


def test_predict_failure_keeps_kind_and_exit_code():
    result = CliRunner().invoke(main, ["predict", "missing.cif", "--backend", "gnn-hse06", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["kind"] == "input"
    assert "missing.cif" in payload["error"]["message"]


def test_json_commands_emit_parseable_stdout():
    for argv in (["backends", "--json"], ["assets", "verify", "--json"]):
        result = CliRunner().invoke(main, argv)
        json.loads(result.stdout)  # raises if anything else is on stdout
```

Also assert that the same commands **without** `--json` still print the current
human lines (regression guard for `tests/test_cli.py`).

- [ ] **Step 2: Verify RED**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_envelope.py -q`
Expected: FAIL — `--json` is not a recognized option.

- [ ] **Step 3: Implement `envelope.py` and wire `--json`**

```python
"""Versioned JSON envelope for machine frontends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import BGPError, InputError, ParseError, PreflightError, RunError
from .schemas import PredictionResult

KIND_BY_ERROR: dict[type[BaseException], str] = {
    InputError: "input",
    PreflightError: "preflight",
    RunError: "run",
    ParseError: "parse",
}


class UnsupportedBackendError(InputError):
    """Requested capability is not provided by the selected backend."""


def error_kind(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, UnsupportedBackendError):
        return "unsupported"
    for error_type, kind in KIND_BY_ERROR.items():
        if isinstance(error, error_type):
            return kind
    return "internal"


def build(
    *,
    command: str,
    argv: list[str],
    result: PredictionResult | None = None,
    bands: dict[str, Any] | None = None,
    error: BaseException | None = None,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    exit_code = 0
    if error is not None:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else getattr(error, "exit_code", 1)
    return {
        "schema_version": "1.0",
        "command": command,
        "ok": error is None and result is not None and result.status.value == "success",
        "argv": argv,
        "result": None if result is None else result.model_dump(mode="json"),
        "bands": bands,
        "error": (
            None
            if error is None
            else {
                "kind": error_kind(error),
                "message": str(error),
                "exit_code": exit_code,
            }
        ),
        "artifact_path": None if artifact_path is None else str(artifact_path),
        "exit_code": exit_code,
    }
```

In `cli.py`: add `--json` to the four commands, route progress to stderr when
set (`progress=lambda message: click.echo(message, err=True)`), and print
`json.dumps(envelope.build(...), sort_keys=True)` as the only stdout write.
`UnsupportedBackendError` is raised by the band command (Task 6 of the band
plan) so `kind` stays declarative.

- [ ] **Step 4: Verify GREEN and real output**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_envelope.py tests/test_cli.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run bgp predict examples/Si.cif --backend gnn-hse06 --json | python -m json.tool | head -12
```

Expected: tests pass; the real command prints exactly one JSON object whose
`result` matches the retained `result.json` of that run.

- [ ] **Step 5: Document and commit**

Write `docs/frontends.md`: envelope schema, `kind` ↔ exit-code table, the
stdout/stderr rule, artifact layout, and the rule that frontends must not parse
human output. Commit `feat: add machine-readable CLI envelope`.

---

### Task 2: `bgp structure show`

**Files:**
- Create: `src/bandgap_predictor/structure_report.py`
- Modify: `src/bandgap_predictor/cli.py`
- Create: `tests/test_structure_report.py`
- Modify: `docs/frontends.md`

**Interfaces:**
- Produces: `structure_report.describe_structure(path: Path) -> dict[str, object]` with `formula`, `cell_angstrom`, `symbols`, `positions_angstrom`, `pbc`, `site_order`, `original_sha256`, `structure_sha256`, `source_path`, and `lattice` = `{"bravais": type(identify_lattice(cell)[0]).__name__, "default_path": cell.bandpath().path, "labels": sorted(cell.bandpath().special_points)}` (ASE 3.29; for `examples/Si.cif`: `CUB` / `GXMGRX,MR` / `["G","M","R","X"]`); raises `InputError` for anything `load_structure` rejects.
- Produces: `bgp structure show STRUCTURE --json` (envelope `command: "structure show"`, payload under `result`).
- Consumes: `structures.load_structure`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_structure_report.py
def test_structure_show_reports_cell_and_hashes():
    payload = json.loads(CliRunner().invoke(main, ["structure", "show", "examples/Si.cif", "--json"]).stdout)
    assert payload["ok"] is True
    data = payload["result"]
    assert data["formula"] == "Si8"
    assert len(data["symbols"]) == 8 and len(data["cell_angstrom"]) == 3
    assert data["structure_sha256"] == load_structure(Path("examples/Si.cif")).structure_sha256
    assert data["lattice"]["bravais"] == "CUB"
    assert data["lattice"]["labels"] == ["G", "M", "R", "X"]


def test_structure_show_rejects_non_periodic(tmp_path):
    molecule = tmp_path / "mol.xyz"
    molecule.write_text("2\nwater\nO 0 0 0\nH 0 0 1\n")
    payload = json.loads(CliRunner().invoke(main, ["structure", "show", str(molecule), "--json"]).stdout)
    assert payload["ok"] is False and payload["error"]["kind"] == "input"
```

- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_structure_report.py -q` → FAIL (no `structure` group).
- [ ] **Step 3: Implement** the report from `load_structure` plus `ase.lattice`/`ase.dft.kpoints` label discovery (no spglib, no seekpath; the band plan owns standardization), and the `structure show` command with `--json` and a human fallback.
- [ ] **Step 4: Verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_structure_report.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run bgp structure show examples/InSb.cif --json | python -m json.tool | head -20
```

- [ ] **Step 5: Commit** `feat: add machine-readable structure report`.

---

### Task 3: `bgp runs list` / `bgp runs show`

**Files:**
- Create: `src/bandgap_predictor/history.py`
- Modify: `src/bandgap_predictor/cli.py`
- Create: `tests/test_history.py`
- Modify: `docs/frontends.md`

**Interfaces:**
- Produces: `history.list_runs(work_root: Path, limit: int | None = None) -> list[dict]` (newest first: `run_id`, `artifact_path`, `backend`, `method`, `status`, `gap_ev`, `classification`, `started_at`, `finished_at`, `elapsed_seconds`, `has_bands`) and `history.read_run(run_dir: Path) -> dict` (`result`, `bands` or `None`, `request`, `artifact_path`).
- Produces: `bgp runs list --work-root R [--limit N] --json`, `bgp runs show RUN_DIR --json`.
- Consumes: run-directory layout (`result.json`, `request.json`, `bands.json`, `scf/result.json`); unreadable directories are skipped by `list` and raise `parse` in `show`.

- [ ] **Step 1: Write the failing tests** over a synthetic tree in `tmp_path` (two valid runs, one truncated `result.json`, one with `bands.json`): newest-first order, truncated skipped by `list`, `show` raising `parse` for it, and `show` embedding both payloads.
- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_history.py -q`.
- [ ] **Step 3: Implement** `history.py` and the `runs` click group (band run directories expose the stage-1 result from `scf/result.json` when no top-level `result.json` exists).
- [ ] **Step 4: Verify GREEN** against a real root: `uv run bgp runs list --work-root build/smoke-runs-fixed --json` and `uv run bgp runs show build/smoke-runs-fixed/20260819T112438Z-Si8-cider-pbe0-f4bd7420 --json`.
- [ ] **Step 5: Commit** `feat: add run history commands`.

---

### Task 4: `bgp serve` skeleton, settings, and read-only API

**Files:**
- Modify: `pyproject.toml` (add `web` extra + static force-include)
- Create: `src/bandgap_predictor/web/__init__.py`, `settings.py`, `cli.py`, `app.py`, `api.py`
- Create: `src/bandgap_predictor/web/static/index.html`, `app.js`, `style.css`
- Create: `tests/web/__init__.py`, `tests/web/support/stub_bgp.py`, `tests/web/conftest.py`, `tests/web/test_cli_bridge.py`, `tests/web/test_api_readonly.py`
- Modify: `src/bandgap_predictor/cli.py` (add `serve`)
- Modify: `docs/installation.md`

**Interfaces:**
- Produces: `ServeSettings` (frozen dataclass) with `host="127.0.0.1"`, `port=8321`, `token: str | None = None`, `work_root: Path = Path("bgp-runs")`, `bgp: list[str]` (default `[sys.executable, "-m", "bandgap_predictor.cli"]`), `abacus: Path | None = None`, `allow_fs: bool = False`, `fs_roots: tuple[Path, ...] = ()`, `upload_dir: Path | None = None`.
- Produces: `web.cli.run_cli(argv: list[str], *, on_stderr: Callable[[str], None]) -> Envelope` and `web.cli.build_argv(settings, kind, structure_path, backend, **options) -> list[str]`.
- Produces: `web.app.create_app(settings: ServeSettings) -> FastAPI` and `bgp serve [--host H] [--port P] [--token T] [--work-root R] [--abacus P] [--bgp CMD] [--allow-fs] [--fs-root DIR]`.
- Endpoints in this task: `GET /api/health`, `GET /api/backends`, `POST /api/structure`, `GET /` + static assets.
- Consumes: `bgp --json` commands from Tasks 1–3.

- [ ] **Step 1: Write the failing tests**

```python
# tests/web/test_cli_bridge.py
def test_build_argv_is_a_list_with_json_and_work_root(tmp_path, settings):
    argv = build_argv(settings, "bands", tmp_path / "Si.cif", "cider-pbe0", reference_distance=0.05, max_kpoints=200)
    assert isinstance(argv, list)
    assert argv[:3] == settings.bgp
    assert "bands" in argv and "--json" in argv
    assert argv[argv.index("--reference-distance") + 1] == "0.05"
    assert argv[argv.index("--work-root") + 1] == str(settings.work_root)


def test_run_cli_keeps_stdout_and_stderr_separate(settings):
    lines: list[str] = []
    envelope = run_cli([*settings.bgp, "backends", "--json"], on_stderr=lines.append)
    assert envelope.ok in (True, False)
    assert envelope.payload["command"] == "backends"
    assert all("{" not in line for line in lines)


# tests/web/test_api_readonly.py
def test_health_and_backends(client):
    health = client.get("/api/health").json()
    assert health["ok"] is True and "queue_depth" in health
    backends = client.get("/api/backends").json()["backends"]
    assert {item["id"] for item in backends} == {
        "gnn-hse06", "abacus-hse06", "abacus-pbe0", "cider-pbe0"
    }
    assert all("bands" in item["capabilities"] or item["id"] == "gnn-hse06" for item in backends)


def test_structure_endpoint_returns_report(client):
    payload = client.post("/api/structure", json={"path": "examples/Si.cif"}).json()
    assert payload["structure"]["formula"] == "Si8"


def test_structure_endpoint_rejects_bad_path(client):
    response = client.post("/api/structure", json={"path": "nope.cif"})
    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "input"


def test_fs_is_disabled_by_default(client):
    assert client.get("/api/fs", params={"path": "."}).status_code == 403


def test_token_is_enforced_when_configured(token_client):
    assert token_client.get("/api/health").status_code == 403
    assert token_client.get("/api/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert token_client.get("/api/health", params={"token": "s3cret"}).status_code == 200
```

`tests/web/conftest.py` builds `client`/`token_client` from
`create_app(settings)` with `settings.bgp = [sys.executable, str(STUB_BGP)]`.

- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web -q` → FAIL (no `web` package).

- [ ] **Step 3: Add the extra and implement settings/bridge/app**

```toml
[project.optional-dependencies]
web = ["fastapi>=0.115", "uvicorn>=0.34"]
```

```python
# web/cli.py
"""The subprocess boundary: argv in, JSON envelope out."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..errors import PreflightError
from .settings import ServeSettings


@dataclass(frozen=True)
class Envelope:
    ok: bool
    payload: dict[str, Any]
    artifact_path: Path | None

    @property
    def kind(self) -> str | None:
        error = self.payload.get("error")
        return None if error is None else str(error["kind"])

    @property
    def message(self) -> str:
        error = self.payload.get("error")
        return "" if error is None else str(error["message"])


def build_argv(
    settings: ServeSettings,
    kind: str,
    structure_path: Path,
    backend: str,
    *,
    reference_distance: float | None = None,
    max_kpoints: int | None = None,
    mechanism: str | None = None,
) -> list[str]:
    """Build the CLI argv list for one frontend run request."""

    if kind == "predict":
        argv = ["predict", str(structure_path), "--backend", backend]
    elif kind == "bands":
        argv = ["bands", str(structure_path), "--backend", backend]
        if reference_distance is not None:
            argv += ["--reference-distance", f"{reference_distance:.6g}"]
        if max_kpoints is not None:
            argv += ["--max-kpoints", str(int(max_kpoints))]
        if mechanism is not None:
            argv += ["--mechanism", mechanism]
    else:
        raise ValueError(f"unsupported run kind: {kind}")
    if settings.abacus is not None:
        argv += ["--abacus", str(settings.abacus)]
    return [*settings.bgp, *argv, "--work-root", str(settings.work_root), "--json"]


def parse_envelope(stdout: str, returncode: int) -> Envelope:
    """Parse the single JSON object printed by a ``--json`` command."""

    text = stdout.strip()
    if not text:
        return Envelope(
            ok=False,
            payload={
                "schema_version": "1.0",
                "command": "unknown",
                "ok": False,
                "error": {"kind": "internal", "message": "CLI produced no output", "exit_code": returncode},
                "artifact_path": None,
                "exit_code": returncode,
            },
            artifact_path=None,
        )
    payload = json.loads(text.splitlines()[-1])
    artifact = payload.get("artifact_path")
    return Envelope(ok=bool(payload.get("ok")), payload=payload, artifact_path=None if artifact is None else Path(artifact))


def run_cli(argv: list[str], *, on_stderr: Callable[[str], None]) -> Envelope:
    """Run one CLI command synchronously, streaming stderr to a callback."""

    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        raise PreflightError(f"could not start the bgp CLI: {error}") from error
    assert process.stdout is not None and process.stderr is not None
    for line in process.stderr:
        on_stderr(line.rstrip("\n"))
    stdout = process.stdout.read()
    process.wait()
    return parse_envelope(stdout, process.returncode)
```

`web/app.py` builds the FastAPI app, mounts `static/`, and installs a token
dependency:

```python
# web/app.py
def create_app(settings: ServeSettings) -> FastAPI:
    app = FastAPI(title="bandgap-predictor", version=__version__)
    app.state.settings = settings

    def require_token(authorization: str | None = Header(default=None), token: str | None = None) -> None:
        if settings.token is None:
            return
        supplied = token or (authorization or "").removeprefix("Bearer ").strip()
        if supplied != settings.token:
            raise HTTPException(status_code=403, detail={"error": {"kind": "forbidden", "message": "invalid token"}})

    app.include_router(build_router(), dependencies=[Depends(require_token)])
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app
```

`bgp serve` (in `cli.py`) imports `uvicorn` lazily, raising
`PreflightError("bgp serve requires the 'web' extra: uv sync --extra web")`
when the import fails, then runs
`uvicorn.run(create_app(settings), host=settings.host, port=settings.port)`.

- [ ] **Step 4: Verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web -q
UV_CACHE_DIR=/tmp/uv-cache uv run bgp serve --port 8321 --work-root /tmp/bgp-serve-smoke &
curl -sS http://127.0.0.1:8321/api/health | python -m json.tool
curl -sS -X POST http://127.0.0.1:8321/api/structure -H 'content-type: application/json' -d '{"path":"examples/Si.cif"}' | head -c 300
kill %1
```

Expected: tests pass; health reports `ok: true`; the structure endpoint returns
the Si8 report.

- [ ] **Step 5: Commit** `feat: add bgp serve with read-only JSON API`.

---

### Task 5: Job registry, SSE progress, cancel, and the GPU queue

**Files:**
- Create: `src/bandgap_predictor/web/jobs.py`
- Modify: `src/bandgap_predictor/web/api.py`, `app.py`
- Create: `tests/web/test_jobs.py`
- Modify: `tests/web/support/stub_bgp.py`

**Interfaces:**
- Produces: `Job` (`id`, `kind`, `argv`, `status`, `exit_code`, `artifact_path`, `envelope`, `log_tail`, `subscribers`, `process`, `started_at`, `finished_at`) and `JobRegistry(settings)` with `submit(...) -> Job`, `get(job_id)`, `cancel(job_id)`, `list()`, `queue_depth`, `active`, `start()`, `shutdown()`.
- Produces: `POST /api/runs`, `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/events` (SSE), `POST /api/runs/{id}/cancel`.
- Consumes: `web.cli.build_argv`/`parse_envelope`, `bgp runs list/show --json` for the history view.

- [ ] **Step 1: Write the failing tests**

```python
# tests/web/test_jobs.py
def test_run_streams_logs_and_finishes_with_envelope(client):
    started = client.post("/api/runs", json={"kind": "predict", "structure_path": "examples/Si.cif", "backend": "gnn-hse06"})
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        events = [line for line in stream.iter_lines() if line.startswith("event:")]
    assert "event: status" in events and "event: done" in events
    detail = client.get(f"/api/runs/{run_id}").json()["run"]
    assert detail["status"] == "success"
    assert detail["result"]["backend"] == "gnn-hse06"
    assert any("progress" in line for line in detail["log_tail"])


def test_only_one_job_runs_at_a_time(client):
    first = client.post("/api/runs", json={...}).json()["run_id"]
    second = client.post("/api/runs", json={...}).json()
    assert second["status"] in {"queued", "running"}
    if second["status"] == "queued":
        assert client.get("/api/health").json()["queue_depth"] >= 1
    client.post(f"/api/runs/{first}/cancel")
    client.post(f"/api/runs/{second['run_id']}/cancel")


def test_cancel_marks_interrupted(client):
    run_id = client.post("/api/runs", json={"kind": "predict", "structure_path": "examples/Si.cif", "backend": "gnn-hse06"}).json()["run_id"]
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
    assert client.get(f"/api/runs/{run_id}").json()["run"]["status"] in {"interrupted", "failed"}


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/runs/nope/cancel").status_code == 404
```

The stub CLI gains a `--sleep` mode (writes `progress N` lines to stderr, sleeps,
prints an envelope, and honours SIGINT by exiting 130) so these tests exercise
real subprocess, queue, SSE and signal behaviour.

- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web/test_jobs.py -q`.

- [ ] **Step 3: Implement `jobs.py` and the run endpoints**

```python
# web/jobs.py (core)
class Job:
    def __init__(self, job_id: str, kind: str, argv: list[str], request: dict[str, Any]) -> None:
        self.id = job_id
        self.kind = kind
        self.argv = argv
        self.request = request
        self.status = "queued"
        self.exit_code: int | None = None
        self.artifact_path: Path | None = None
        self.envelope: dict[str, Any] | None = None
        self.log_tail: list[str] = []
        self.subscribers: set[asyncio.Queue[tuple[str, str]]] = set()
        self.process: asyncio.subprocess.Process | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None

    def publish(self, event: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        for queue in list(self.subscribers):
            queue.put_nowait((event, payload))


class JobRegistry:
    """One GPU job at a time; additional jobs wait in FIFO order."""

    def __init__(self, settings: ServeSettings) -> None:
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self.active: Job | None = None
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._worker = asyncio.create_task(self._consume())

    async def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    async def submit(self, kind: str, structure_path: Path, backend: str, **options: Any) -> Job:
        argv = build_argv(self.settings, kind, structure_path, backend, **options)
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, argv=argv, request={...})
        self.jobs[job.id] = job
        await self._pending.put(job.id)
        return job

    async def cancel(self, job_id: str) -> Job:
        job = self._require(job_id)
        if job.process is not None and job.process.returncode is None:
            os.killpg(os.getpgid(job.process.pid), signal.SIGINT)
        return job

    async def _consume(self) -> None:
        while True:
            job = self.jobs[await self._pending.get()]
            self.active = job
            try:
                await self._execute(job)
            finally:
                self.active = None

    async def _execute(self, job: Job) -> None:
        job.status = "running"
        job.started_at = datetime.now(UTC).isoformat()
        job.publish("status", {"status": "running", "argv": job.argv})
        process = await asyncio.create_subprocess_exec(
            *job.argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        job.process = process
        pump = asyncio.create_task(self._pump(job, process))
        stdout = (await process.stdout.read()).decode("utf-8", "replace")
        await process.wait()
        await pump
        job.exit_code = process.returncode
        envelope = parse_envelope(stdout, process.returncode)
        job.envelope = envelope.payload
        job.artifact_path = envelope.artifact_path
        job.status = "success" if envelope.ok else ("interrupted" if process.returncode == 130 else "failed")
        job.finished_at = datetime.now(UTC).isoformat()
        job.publish("done", {"status": job.status, "exit_code": job.exit_code, "artifact_path": str(job.artifact_path)})

    async def _pump(self, job: Job, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        async for raw in process.stderr:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            job.log_tail.append(line)
            del job.log_tail[:-500]
            job.publish("log", {"line": line})
```

SSE endpoint (heartbeat + replay):

```python
@router.get("/runs/{job_id}/events")
async def events(job_id: str, request: Request, since: int = 0) -> StreamingResponse:
    job = registry.require(job_id)

    async def stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        for line in job.log_tail[since:]:
            yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"
        job.subscribers.add(queue)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event, payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {event}\ndata: {payload}\n\n"
                if event == "done":
                    break
        finally:
            job.subscribers.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})
```

`GET /api/runs` merges live registry jobs with `history.list_runs` output;
`GET /api/runs/{id}` returns `{run: {..., result, bands}}` where `result`/`bands`
come from the live envelope or `history.read_run` for finished/previous runs.

- [ ] **Step 4: Verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web -q
```

Expected: PASS, including SSE ordering, queue, cancel and 404 cases.

- [ ] **Step 5: Commit** `feat: add serve job registry with SSE and cancel`.

---

### Task 6: Frontend shell — structure, 3D view, method, run, result

**Files:**
- Create: `src/bandgap_predictor/web/static/index.html`, `app.js`, `style.css`
- Create: `src/bandgap_predictor/web/static/3Dmol-min.js` (vendored, BSD-3-Clause)
- Create: `LICENSES/3Dmol.js`, modify `assets/NOTICE.md`
- Modify: `src/bandgap_predictor/web/api.py` (`GET /api/fs`, `POST /api/structures`)
- Create: `tests/web/test_api_uploads.py`, `tests/web/test_static_assets.py`

**Interfaces:**
- Produces: `GET /api/fs?path=` (disabled unless `allow_fs`; entries never follow symlinks outside `fs_roots`), `POST /api/structures` (multipart, sanitized name, stored under `<work_root>/uploads`, then the same report as `POST /api/structure`).
- Produces: the page: structure panel (path + browser + upload), 3D view (3Dmol.js with a server PNG fallback), method panel from `/api/backends`, run panel (SSE log + cancel), result panel (gap, classification, converged, warnings, provenance, artifact path).
- Consumes: Tasks 4–5 endpoints, `structure show` payload.

- [ ] **Step 1: Write the failing tests**

```python
# tests/web/test_api_uploads.py
def test_upload_stores_file_and_returns_report(client, tmp_path):
    response = client.post(
        "/api/structures",
        files={"file": ("Si.cif", (EXAMPLES / "Si.cif").read_bytes(), "chemical/x-cif")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["structure"]["formula"] == "Si8"
    assert Path(payload["path"]).is_file()
    assert Path(payload["path"]).parent == settings.work_root / "uploads"


def test_upload_rejects_traversal_names(client):
    response = client.post("/api/structures", files={"file": ("../../etc/passwd", b"x", "text/plain")})
    assert response.status_code == 400


def test_fs_listing_is_gated_and_confined(fs_client, tmp_path):
    assert fs_client.get("/api/fs", params={"path": str(tmp_path)}).status_code == 200
    assert fs_client.get("/api/fs", params={"path": "/etc"}).status_code == 403


# tests/web/test_static_assets.py
def test_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "3Dmol" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/3Dmol-min.js").status_code == 200
```

- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web/test_api_uploads.py tests/web/test_static_assets.py -q`.

- [ ] **Step 3: Vendor 3Dmol.js and write the page**

```bash
mkdir -p src/bandgap_predictor/web/static LICENSES
curl -sSLo src/bandgap_predictor/web/static/3Dmol-min.js https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js
sha256sum src/bandgap_predictor/web/static/3Dmol-min.js
curl -sSL https://raw.githubusercontent.com/3dmol/3Dmol.js/master/LICENSE | head -5 > LICENSES/3Dmol.js
```

Record the sha256 and BSD-3-Clause notice in `assets/NOTICE.md` and in the
`tests/web/test_static_assets.py` assertions so the vendored file is pinned.

`index.html` (structure): one page, no framework, `fetch` + `EventSource`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Bandgap Predictor</title>
  <link rel="stylesheet" href="/static/style.css">
  <script src="/static/3Dmol-min.js"></script>
</head>
<body>
  <header><h1>Bandgap Predictor</h1><span id="health"></span></header>
  <main>
    <section id="structure-panel">
      <h2>Structure</h2>
      <input id="structure-path" placeholder="/abs/path/Si.cif">
      <button id="load-path">Load</button>
      <input id="structure-file" type="file">
      <button id="upload">Upload</button>
      <dl id="structure-info"></dl>
      <div id="viewer" style="width:100%;height:360px"></div>
    </section>
    <section id="method-panel">
      <h2>Method</h2>
      <select id="backend"></select>
      <p id="backend-reason"></p>
      <label>reference distance <input id="reference-distance" type="number" step="0.005" value="0.025"></label>
      <button id="run-predict">Predict gap</button>
      <button id="run-bands">Compute bands</button>
    </section>
    <section id="run-panel">
      <h2>Run</h2>
      <div id="run-status"></div>
      <button id="cancel">Cancel</button>
      <pre id="log"></pre>
    </section>
    <section id="result-panel">
      <h2>Result</h2>
      <dl id="result"></dl>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
```

`app.js` (structure + run + log + result; bands added in Task 7):

```javascript
const state = { structure: null, runId: null, events: null, token: new URLSearchParams(location.search).get('token') };

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers['Authorization'] = `Bearer ${state.token}`;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function renderStructure(payload) {
  state.structure = payload.structure;
  const info = payload.structure;
  document.getElementById('structure-info').innerHTML =
    `<dt>Formula</dt><dd>${info.formula}</dd>
     <dt>Atoms</dt><dd>${info.symbols.length}</dd>
     <dt>Bravais</dt><dd>${info.lattice.bravais} (${info.lattice.default_path})</dd>
     <dt>Structure sha256</dt><dd>${info.structure_sha256.slice(0, 16)}…</dd>`;
  renderViewer(info);
}

function renderViewer(info) {
  const viewer = $3Dmol.createViewer(document.getElementById('viewer'), { backgroundColor: 'white' });
  const [a, b, c] = info.cell_angstrom;
  let cif = `data_bandgap\n_cell_length_a ${norm(a)}\n_cell_length_b ${norm(b)}\n_cell_length_c ${norm(c)}\n`;
  cif += `_cell_angle_alpha ${angle(b, c)}\n_cell_angle_beta ${angle(a, c)}\n_cell_angle_gamma ${angle(a, b)}\n`;
  cif += `loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n`;
  const frac = fractional(info);
  info.symbols.forEach((symbol, index) => {
    cif += `${symbol}${index + 1} ${symbol} ${frac[index].map((v) => v.toFixed(6)).join(' ')}\n`;
  });
  viewer.addModel(cif, 'cif');
  viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.3 } });
  viewer.addUnitCell();
  viewer.zoomTo();
  viewer.render();
}

async function startRun(kind) {
  const backend = document.getElementById('backend').value;
  const body = { kind, structure_path: state.structure.source_path, backend };
  if (kind === 'bands') body.reference_distance = Number(document.getElementById('reference-distance').value);
  const started = await api('/api/runs', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  state.runId = started.run_id;
  document.getElementById('log').textContent = '';
  if (state.events) state.events.close();
  const query = new URLSearchParams({ since: 0 });
  if (state.token) query.set('token', state.token);
  state.events = new EventSource(`/api/runs/${state.runId}/events?${query}`);
  state.events.addEventListener('log', (event) => {
    const pre = document.getElementById('log');
    pre.textContent += `${JSON.parse(event.data).line}\n`;
    pre.scrollTop = pre.scrollHeight;
  });
  state.events.addEventListener('done', async () => {
    state.events.close();
    const detail = (await api(`/api/runs/${state.runId}`)).run;
    renderResult(detail);
  });
}

function renderResult(detail) {
  const result = detail.result || {};
  document.getElementById('run-status').textContent = `${detail.status} (exit ${detail.exit_code ?? '—'})`;
  document.getElementById('result').innerHTML =
    `<dt>Gap</dt><dd>${result.prediction?.gap_ev ?? 'unavailable'} eV</dd>
     <dt>Classification</dt><dd>${result.prediction?.classification ?? '—'}</dd>
     <dt>Converged</dt><dd>${result.converged ?? '—'}</dd>
     <dt>Artifacts</dt><dd>${detail.artifact_path ?? '—'}</dd>
     <dt>Warnings</dt><dd>${(result.warnings || []).join('<br>') || 'none'}</dd>`;
}
```

Bootstrap: `GET /api/backends` → fill `#backend`, disable the bands action when
`capabilities` lacks `bands` and show `reason`; `GET /api/health` → header.
`fractional()`/`norm()`/`angle()` are small local helpers converting the cell
matrix and Cartesian positions to CIF cell parameters and fractional
coordinates; the viewer is display-only and never feeds data back into a run.

- [ ] **Step 4: Verify GREEN and real browser render**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web -q
UV_CACHE_DIR=/tmp/uv-cache uv run bgp serve --port 8321 --work-root /tmp/bgp-serve-smoke &
```

Then drive the real browser: open `http://127.0.0.1:8321/`, load
`examples/Si.cif`, assert the 3D canvas has non-zero size and WebGL context,
run `gnn-hse06`, and screenshot the page. Expected: 3D cell visible, log lines
streaming, result panel showing the gap; `browser_logs` free of errors.

- [ ] **Step 5: Commit** `feat: add web UI shell with 3D structure view`.

---

### Task 7: Bands tab

**Files:**
- Modify: `src/bandgap_predictor/web/static/index.html`, `app.js`
- Modify: `src/bandgap_predictor/web/api.py` (`GET /api/runs/{id}/bands.svg`, `GET /api/runs/{id}/bands.csv`)
- Create: `tests/web/test_bands_endpoints.py`
- Create: `tests/web/fixtures/bands.json`

**Interfaces:**
- Produces: `GET /api/runs/{id}/bands.svg` rendering `bands.json` through
  `bandgap_predictor.bands.plot.plot_bands` (from the band plan) into an SVG
  response, and `GET /api/runs/{id}/bands.csv` with columns
  `k_index,kx,ky,kz,cumulative_distance,band,eigenvalue_ev`.
- Produces: the bands tab: `--reference-distance` / `--max-kpoints` controls, run
  action, SVG `<img>`, frontier annotation text, CSV download link.
- Consumes: `bands.json` payload from the band plan; `frontier` block.

- [ ] **Step 1: Write the failing tests**

```python
# tests/web/test_bands_endpoints.py
def test_bands_svg_is_rendered_from_the_artifact(client, bands_run):
    response = client.get(f"/api/runs/{bands_run}/bands.svg")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg")
    assert "<svg" in response.text


def test_bands_csv_has_one_row_per_band_and_kpoint(client, bands_run, bands_payload):
    rows = client.get(f"/api/runs/{bands_run}/bands.csv").text.strip().splitlines()
    assert rows[0] == "k_index,kx,ky,kz,cumulative_distance,band,eigenvalue_ev"
    assert len(rows) - 1 == len(bands_payload["path"]["kpoints_direct"]) * len(bands_payload["eigenvalues_ev"][0])


def test_bands_endpoint_is_404_without_bands(client, predict_run):
    assert client.get(f"/api/runs/{predict_run}/bands.svg").status_code == 404
```

- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web/test_bands_endpoints.py -q`.

- [ ] **Step 3: Implement the endpoints and the tab**

The SVG endpoint calls the band plan's `plot_bands(payload, tmp_path / "bands.svg")`
via `run_in_threadpool` (matplotlib is not thread-safe on the event loop) and
returns the file bytes; the CSV endpoint streams rows built from
`path.kpoints_direct`, `path.cumulative_distance` and `eigenvalues_ev`. The page
adds a `#bands-panel` with the two controls, the run button
(`startRun('bands')`), the `<img src="/api/runs/{id}/bands.svg?token=…">` and a
frontier summary (`gap_ev`, `gap_kind`, `vbm_ev`, `cbm_ev`).

- [ ] **Step 4: Verify GREEN and one real band run through the browser**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web -q
UV_CACHE_DIR=/tmp/uv-cache uv run bgp serve --port 8321 --work-root build/band-runs \
  --abacus build/abacus-install-fixed/bin/abacus_pw_gpu &
```

In the browser: select `cider-pbe0`, compute bands on `examples/Si.cif`, watch
the SSE log, assert the SVG appears, screenshot the bands tab, and check that
the frontier text matches `bands.json`. Expected: real two-stage run visible in
the UI, no console errors.

- [ ] **Step 5: Commit** `feat: add web bands tab with server-rendered SVG`.

---

### Task 8: History, provenance, and mesh-eigenvalue panel

**Files:**
- Modify: `src/bandgap_predictor/web/static/app.js`, `index.html`
- Modify: `src/bandgap_predictor/web/api.py` (history merge + startup scan)
- Create: `tests/web/test_history_api.py`

**Interfaces:**
- Produces: `GET /api/runs` merging live registry jobs with `bgp runs list --json`; selecting a run in the UI calls `GET /api/runs/{id}` and repopulates the result/bands panels without recomputation; startup marks stale `running` jobs `interrupted`.
- Produces: provenance panel (executable hash, source revisions, pseudopotential/model hashes) and an optional mesh-eigenvalue panel from `OUT.ABACUS/bgp_electronic_state.json`, labelled "SCF mesh eigenvalues — not a band structure".

- [ ] **Step 1: Write the failing tests**: history ordering with a live job first; a previous run directory opened through `GET /api/runs/{id}` returning `result` and `bands`; a run without bands leaving the bands panel empty; the mesh panel refusing to plot when the artifact lacks a finite Fermi level; a stale `running` job marked `interrupted` on startup.
- [ ] **Step 2: Verify RED**: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/web/test_history_api.py -q`.
- [ ] **Step 3: Implement** the history merge, the startup scan, and the two panels.
- [ ] **Step 4: Verify GREEN** plus a browser check that opens a retained run (`build/smoke-runs-fixed/20260819T112438Z-Si8-cider-pbe0-f4bd7420`) from the history list.
- [ ] **Step 5: Commit** `feat: add web history and provenance panels`.

---

### Task 9: Documentation, licensing, release record, and end-to-end browser verification

**Files:**
- Modify: `README.md`, `docs/installation.md`, `docs/frontends.md`, `docs/methods.md`
- Modify: `assets/NOTICE.md`, `LICENSES/3Dmol.js`
- Create: `../releases/2026-10-13-bgp-web-v0.2.0.yaml`, modify `../progress.md`, `../task_plan.md` (in `/home/bonan/work/HC-Report`)

- [ ] **Step 1: Document** `bgp serve` (install extra, flags, token/host semantics, `--allow-fs`), the JSON API table, the security model, and the "web UI never computes science" rule; state that the frontend is a separate process from the CLI.
- [ ] **Step 2: Record licensing**: 3Dmol.js version + sha256 + BSD-3-Clause in `assets/NOTICE.md` and `LICENSES/3Dmol.js`; assert the hash in `tests/web/test_static_assets.py`.
- [ ] **Step 3: End-to-end browser evidence** using the Paseo browser host on a fresh `bgp serve` instance:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bgp serve --port 8321 --work-root build/web-verification \
  --abacus build/abacus-install-fixed/bin/abacus_pw_gpu --token e2e-token &
```

Drive: load `examples/Si.cif` → 3D render → `gnn-hse06` predict → result panel
→ `cider-pbe0` bands → SVG + frontier → open a history entry → cancel a long
run. Capture screenshots for each step and check `browser_logs` for errors.
Expected: every step observable, screenshots retained under `build/web-verification/`.

- [ ] **Step 4: Release record**: write the HC-Report manifest (package commit,
  `bgp serve` version, envelope `schema_version`, `bands.json` schema version,
  API version, evidence screenshots and run directories, executable hash, test
  counts) and append one progress line; update `task_plan.md` phases.
- [ ] **Step 5: Commit** both repositories.

---

### Task 10: Final verification

- [ ] **Step 1: Full suite and lint**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Expected: the 66 passed / 12 skipped baseline plus envelope, structure, history,
`tests/web`, and band tests; hardware tests skipped without `BGP_ABACUS`.

- [ ] **Step 2: Base install stays lean**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import importlib.util as u; print(u.find_spec('fastapi'))"
UV_CACHE_DIR=/tmp/uv-cache uv run bgp serve --port 8399 || true
```

Expected: `fastapi` absent from the base sync and `bgp serve` printing the
actionable `PreflightError` message; with `--extra web` it starts.

- [ ] **Step 3: Contract audit**

```bash
grep -rn "bandgap_predictor" src/bandgap_predictor/web/*.py | grep -v "^.*:.*#"
grep -rn "shell=True\|os.system\|subprocess.run(.*str(" src/bandgap_predictor/web || echo "no shell strings"
git status --short
```

Expected: the web package imports only `errors`, `schemas`, `history`,
`structure_report`, `bands.plot` and its own modules — never
`service`/`backends`/`abacus`/`bands.runner`; no shell strings; locked presets,
golden gap INPUT files, historical result directories and the executable hash
`9ad2d03b…` unchanged.
