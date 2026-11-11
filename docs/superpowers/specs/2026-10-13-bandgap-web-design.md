# Bandgap Predictor Web Frontend Design (`bgp serve`)

> **状态：已暂存（PARKED，2026-10-15）。** 本计划不在当前工作范围内执行。
> 前端工作暂时搁置，优先交付 GUI 团队的交接文档：见 `docs/handover/`。
> 恢复执行时，取消本横幅并按其前置依赖（`2026-10-13-band-structure.md`）推进。

## Objective

Replace the desktop-GUI plan with a **browser frontend**: a single-page app
served by a new `bgp serve` command in the main package, which drives the `bgp`
CLI as a subprocess and streams its output to the page. The frontend must be
able to target a remote GPU host simply by running the server there, and must
never import scientific internals: argv in, one JSON envelope out, artifacts on
disk.

Required user-facing features (unchanged from the original GUI request):

1. read a structure (server-side path or upload);
2. choose the calculation method;
3. visualize the structure in 3D;
4. visualize bands for the computed model.

Band computation itself is owned by
[`2026-10-13-band-structure-design.md`](2026-10-13-band-structure-design.md)
and its plan; this document covers only the web frontend and the bridge.

## Why web over Qt

- The frontend/backend split already isolates the frontend behind a transport;
  a browser is just another transport client, and remote use becomes "run the
  server on the GPU box, open the page from a laptop" — no SSH plumbing.
- Verified in this environment (2026-10-13): the Paseo browser host is a real
  Chromium (`Chrome/152`, `Electron/44`) with `webgl: true`, `EventSource` and
  `fetch` available, so the frontend can be **verified end-to-end in a real
  browser** (click, run a real calculation, render 3D, screenshot). The Qt plan
  could only be smoke-tested headless (`DISPLAY` is empty on this workstation).
- `fastapi`/`uvicorn`/`starlette` resolve from the configured index; `3Dmol.js`
  2.4.0 (513 KB, BSD-3-Clause) downloads cleanly and can be vendored so the UI
  works offline.

## Architecture

```
┌──────────────────────────── browser (single page, no build step) ───────────┐
│ index.html + app.js + style.css + vendored 3Dmol-min.js + band SVG <img>    │
│ panels: structure | method | run log | result | bands | history             │
└───────────────┬─────────────────────────────── HTTP/JSON + SSE ─────────────┘
                │
┌───────────────▼─────────────── bgp serve (main package, extra "web") ───────┐
│ FastAPI app:  /api/health /api/backends /api/structure /api/structures      │
│               /api/runs (POST/GET) /api/runs/{id} /api/runs/{id}/events     │
│               /api/runs/{id}/cancel /api/runs/{id}/bands.svg                │
│ job registry: one GPU job at a time, SSE fan-out, cancel via SIGINT         │
└───────────────┬─────────────────────────────── argv lists only ─────────────┘
                │
┌───────────────▼────────────────────── bgp CLI (unchanged contract) ─────────┐
│ predict --json | bands --json | structure show --json | runs list/show      │
│ backends --json | doctor --json | assets verify --json                      │
└───────────────┬────────────────────────────────────────────────────────────┘
                ▼
   run directories: request.json, scf/, bands/, result.json, bands.json
```

Invariants:

1. `bgp serve` never imports backend/service/parser modules for computation. It
   spawns the CLI (`sys.executable -m bandgap_predictor.cli …` by default,
   overridable with `--bgp`) and reads the JSON envelope from stdout.
2. argv lists only; no shell strings. Uploaded file names are sanitized and the
   path is passed as a separate argv element.
3. Exactly one JSON object per CLI call on stdout; stderr lines become SSE
   `log` events verbatim.
4. The page never computes science: gap, classification, frontier, k-path,
   distances and 3D coordinates all come from CLI JSON.
5. Default bind is `127.0.0.1`; remote exposure requires `--host` and is
   documented as unsafe without `--token`.

## Package layout

```
src/bandgap_predictor/web/
  __init__.py
  app.py            create_app(settings) -> FastAPI
  settings.py       ServeSettings (host, port, token, work_root, bgp argv, upload_dir)
  jobs.py           JobRegistry, Job, single-GPU queue, SSE broadcaster
  cli.py            run_cli(argv, on_stderr, cancel) -> Envelope   (subprocess boundary)
  api.py            route handlers (thin: call cli.py, map envelope -> HTTP)
  static/index.html
  static/app.js
  static/style.css
  static/3Dmol-min.js      vendored, BSD-3-Clause (LICENSES/3Dmol.js)
tests/web/
  test_api.py       TestClient contracts (health, backends, structure, runs, errors)
  test_jobs.py      job lifecycle, queueing, cancel, SSE replay
  test_cli_bridge.py argv construction, envelope parsing, stdout/stderr separation
  support/stub_bgp.py  a stub CLI used as the process boundary in tests
```

`pyproject.toml`:

```toml
[project.optional-dependencies]
web = ["fastapi>=0.115", "uvicorn>=0.34"]

[tool.hatch.build.targets.wheel.force-include]
"src/bandgap_predictor/web/static" = "bandgap_predictor/web/static"
```

`bgp serve` raises `PreflightError` with an actionable message when the `web`
extra is missing, so the base install stays lean while the command stays
first-class.

## CLI surface (frontend-agnostic, kept from the previous plan)

| Command | Role |
| --- | --- |
| `bgp predict STRUCTURE --backend B --json` | one gap prediction |
| `bgp bands STRUCTURE --backend B [--reference-distance F] [--max-kpoints N] --json` | two-stage band run (see the band plan) |
| `bgp structure show STRUCTURE --json` | validated structure: cell, symbols, positions, hashes, detected Bravais lattice and suggested path labels |
| `bgp backends --abacus P --json` | method availability + capability flags |
| `bgp runs list --work-root R --json` / `bgp runs show RUN_DIR --json` | history and re-open, embedding `result.json`/`bands.json` |
| `bgp doctor --abacus P --json`, `bgp assets verify --json` | environment panel |

## HTTP API

All responses are JSON except `/api/runs/{id}/events` (SSE) and
`/api/runs/{id}/bands.svg` (image). Errors use `{"error": {"kind", "message",
"artifact_path"}}` with `kind` copied from the CLI envelope.

| Method + path | Request | Response |
| --- | --- | --- |
| `GET /api/health` | – | `{ok, version, work_root, abacus, queue_depth, active_run}` |
| `GET /api/backends` | `?abacus=` | `{backends: [{id, available, capabilities: ["gap","bands"], reason}]}` |
| `POST /api/structure` | `{"path": "/abs/x.cif"}` | `{ok, structure: <structure show payload>}` |
| `POST /api/structures` | multipart file | `{ok, path, structure: {…}}` (stored under `<work_root>/uploads/<run-ish name>`) |
| `GET /api/fs` | `?path=/dir` | `{path, entries: [{name, is_dir, size}]}` — server-side browsing for HPC-style layouts |
| `POST /api/runs` | `{"kind": "predict"\|"bands", "structure_path", "backend", "reference_distance"?, "max_kpoints"?, "mechanism"?, "abacus"?}` | `202 {run_id, status: "queued"\|"running", artifact_path}` |
| `GET /api/runs` | `?limit=` | `{runs: [<runs list payload>]}` |
| `GET /api/runs/{id}` | – | `{run: {id, status, exit_code, argv, artifact_path, result, bands}}` |
| `GET /api/runs/{id}/events` | `?since=` | SSE: `event: log` / `event: status` / `event: done` (`data:` JSON) |
| `POST /api/runs/{id}/cancel` | – | `{ok, status}` — SIGINT to the process group |
| `GET /api/runs/{id}/bands.svg` | `?relative=1` | matplotlib SVG rendered from `bands.json` |

Status codes: `400` bad request/`input`, `403` missing/invalid token, `404`
unknown run, `409` queue full or run already finished, `422` `preflight`/`parse`
from the envelope, `500` unexpected. Successful CLI runs are `202`/`200`.

## Job model

- `JobRegistry` holds `Job` records: `id` (= CLI run directory name when
  available), `kind`, `argv`, `status` (`queued|running|success|failed|
  interrupted`), `exit_code`, `artifact_path`, `started_at`, `finished_at`,
  `result`, `bands`, `log_tail` (ring buffer, last 500 lines), `subscribers`.
- **One GPU job at a time**: `kpar=1`, one MPI rank and one OpenMP thread are
  locked by the CLI presets, so concurrent runs would oversubscribe the single
  RTX 5090. Additional `POST /api/runs` requests are queued (`status:
  queued`), and `/api/health` exposes `queue_depth`.
- Cancel sends `SIGINT` to the child process group (the CLI's own interrupt
  path: artifacts retained, exit 130, `interrupted` result when possible).
- Startup scans `work_root` with `bgp runs list --json` so history survives a
  server restart; live jobs are marked `interrupted` if their process is gone.
- SSE subscribers get a replay of `log_tail` on connect (`?since=` index) then
  live events; disconnects never kill the job.

## Frontend

Single page, no framework, no build step, no CDN at runtime (all assets
vendored).

- **Structure panel**: path input with a server-side browser (`/api/fs`),
  upload button, then formula, cell parameters, atom count, site order and both
  hashes from `structure show`.
- **3D view**: `3Dmol.js` renders the cell box + atoms from the structure JSON.
  Coordinates are converted client-side to a minimal XYZ/CIF string with the
  cell vectors as `Lattice=` (3Dmol supports a CIF payload); periodic images
  are drawn by repeating the cell in the viewer, not by changing the data.
  If WebGL is unavailable the panel falls back to a server-rendered PNG
  (`matplotlib`, same cell/atoms data) so the page still works.
- **Method panel**: populated from `/api/backends`; the bands action is
  disabled with the server-provided reason for `gnn-hse06`; `--abacus` path
  comes from the server settings, not from the browser.
- **Run panel**: run button, live log via SSE, cancel button, elapsed time,
  queue position.
- **Result panel**: gap, classification, converged flag, occupation summary,
  warnings verbatim, provenance hashes, artifact path, and a link to the run
  directory.
- **Bands tab**: `--reference-distance` / `--max-kpoints` controls, run action,
  server-rendered SVG with special-point ticks, Fermi line and frontier
  annotation; CSV download of `eigenvalues_ev` + `cumulative_distance`.
- **History panel**: `/api/runs` list; selecting a run calls `/api/runs/{id}`
  and repopulates the result and bands panels without recomputation.
- **Mesh-eigenvalue panel** (optional, hidden by default): SCF eigenvalues vs
  irreducible-k index with k-weight markers, labelled explicitly as "SCF mesh
  eigenvalues — not a band structure".

## Security

- Default `--host 127.0.0.1`. Exposing the server requires `--host`; without
  `--token` the UI prints a persistent warning banner.
- `--token T` requires `Authorization: Bearer T` (or `?token=T` for SSE and
  `<img>` tags, since browsers cannot set headers on those requests).
- `/api/fs` and `/api/structures` are the sensitive endpoints: they read and
  write server-side paths. They are disabled unless `--allow-fs` is passed, and
  uploads are restricted to `<work_root>/uploads` with sanitized names.
- No shell interpolation anywhere; structure paths are argv elements.
- The run directory is served only through the JSON envelope and the SVG
  endpoint; there is no generic file server.

## Testing and verification

- API contracts with `fastapi.testclient.TestClient`: health, backends,
  structure (path + upload + rejection), runs (queue, duplicate, unknown id),
  cancel, token enforcement, `/api/fs` gating, error-kind → status mapping.
- Job semantics against `tests/web/support/stub_bgp.py`: argv construction,
  stdout/stderr separation, SSE event sequence (`status`→`log`*→`done`), replay
  on reconnect, cancel producing `interrupted`, and the single-job queue.
- Band SVG endpoint from a recorded `bands.json` fixture: status 200, non-empty
  SVG, and the frontier annotation text present.
- **Real browser end-to-end** using the Paseo browser host: start
  `bgp serve` on a free port, `browser_navigate` to it, load
  `examples/Si.cif`, assert the 3D canvas renders (WebGL), run `gnn-hse06`
  (fast), watch SSE log lines appear, assert the result panel shows the gap,
  then run `bgp bands` on `cider-pbe0` and screenshot the bands tab; capture
  the screenshots as evidence and check `browser_logs` for console errors.
- A cancel test through the browser (long GNN run → cancel → `interrupted`).
- Existing suite unchanged: `uv run pytest -q` (66 passed, 12 skipped baseline)
  plus the new `tests/web` and band tests.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Main package gains web dependencies | `web` optional extra; `bgp serve` fails with an actionable `PreflightError`; base install unchanged |
| Single RTX 5090 shared with the CLI | one job at a time + queue + `/api/health.queue_depth`; presets already lock 1 MPI rank / 1 OMP thread |
| LAN exposure without auth | localhost default, `--token`, `--allow-fs` gate, warning banner, documented |
| `/api/fs` path traversal | disabled by default, resolved paths must stay under allowed roots, entries listed without following symlinks out of the root |
| SSE buffering behind proxies | `text/event-stream` + `X-Accel-Buffering: no` + heartbeat comments every 15 s |
| Large `bands.json` in JSON responses | payload is ~60 KB for a 202-point path; optional gzip via uvicorn; the SVG endpoint keeps the heavy plot server-side |
| 3Dmol.js licensing/offline | vendored build + `LICENSES/3Dmol.js` (BSD-3-Clause) recorded in `assets/NOTICE.md` |
| Browser cannot set headers for `<img>`/SSE | token accepted as a query parameter for those two endpoints only |
| Server restart loses live jobs | startup scan via `bgp runs list`, stale `running` jobs marked `interrupted` |
