"""GNNBGP Industrial Workbench v0.6 -- local GUI/CLI bridge.

The scientific engine is the user's NEW D:\\program\\gnnbgp checkout.
No scientific backend logic is reimplemented; run "bgp predict" as a subprocess.
Python >=3.11. Python standard library + ASE (already an engine dependency).
"""
from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
DEFAULT_HOME = Path.home() / "BandGap-Studio"
SOURCE = Path(os.environ.get("BGP_SOURCE", "/mnt/d/program/gnnbgp")).expanduser().resolve()
DATA = Path(os.environ.get("BGP_UI_DATA", str(DEFAULT_HOME / "workspace"))).expanduser().resolve()
BGP_BIN = Path(os.environ.get("BGP_BIN", str(DEFAULT_HOME / ".venv" / "bin" / "bgp"))).expanduser().resolve()
DEFAULT_ABACUS = Path(os.environ.get(
    "BGP_ABACUS", str(DEFAULT_HOME / "build" / "abacus-install" / "bin" / "abacus_pw_gpu")
)).expanduser().resolve()
BGP_DEPS_PREFIX = Path(os.environ.get(
    "BGP_DEPS_PREFIX", str(Path.home() / "bgp-deps")
)).expanduser().resolve()
LISTEN = "127.0.0.1"
PORT = int(os.environ.get("BGP_UI_PORT", "8780"))
MAX_UPLOAD = 12 * 1024 * 1024
METHODS = ("gnn-hse06", "abacus-hse06", "abacus-pbe0", "cider-pbe0")
TERMINAL = ("success", "failed", "interrupted")
FILE_SUFFIX = {".cif", ".vasp", ".poscar", ".contcar", ".xyz"}
QUEUE: queue.Queue[str] = queue.Queue()
LOCK = threading.RLock()
RUNNING: dict[str, subprocess.Popen] = {}
# Observations are tied to a GUI task ID. Aggregate GPU readings are not
# represented as per-process utilization unless the driver reports PID evidence.
GPU_LAST: dict[str, dict] = {}
GPU_LOCK = threading.Lock()
CONN: sqlite3.Connection | None = None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db():
    assert CONN is not None, "Database not initialized"
    return CONN


def one(sql, params=()):
    with LOCK:
        row = db().execute(sql, params).fetchone()
    return dict(row) if row else None


def many(sql, params=()):
    with LOCK:
        rows = db().execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def execute(sql, params=()):
    with LOCK:
        db().execute(sql, params)
        db().commit()


def update_job(jid, **fields):
    if not fields:
        return
    fields["updated_at"] = now()
    assigns = ", ".join(f"{key}=?" for key in fields)
    with LOCK:
        db().execute(f"UPDATE jobs SET {assigns} WHERE id=?", (*fields.values(), jid))
        db().commit()


def new_id():
    return uuid.uuid4().hex[:16]


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def init_store():
    global CONN
    DATA.mkdir(parents=True, exist_ok=True)
    for folder in ("materials", "jobs"):
        (DATA / folder).mkdir(exist_ok=True)
    CONN = sqlite3.connect(DATA / "studio.sqlite3", check_same_thread=False)
    CONN.row_factory = sqlite3.Row
    CONN.execute("PRAGMA journal_mode=WAL")
    CONN.executescript("""
      CREATE TABLE IF NOT EXISTS materials (
        id TEXT PRIMARY KEY, label TEXT NOT NULL, filename TEXT NOT NULL,
        path TEXT NOT NULL, origin TEXT NOT NULL, formula TEXT NOT NULL,
        atom_count INTEGER NOT NULL, sha256 TEXT NOT NULL,
        preview TEXT NOT NULL, created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, material_id TEXT NOT NULL, material_label TEXT NOT NULL,
        material_formula TEXT NOT NULL, backend TEXT NOT NULL,
        state TEXT NOT NULL, submitted_at TEXT NOT NULL,
        started_at TEXT, finished_at TEXT, updated_at TEXT NOT NULL,
        exit_code INTEGER, abacus_path TEXT, device TEXT NOT NULL,
        input_path TEXT NOT NULL, run_root TEXT NOT NULL,
        run_dir TEXT, result_path TEXT, log_path TEXT NOT NULL,
        error TEXT NOT NULL DEFAULT '', note TEXT NOT NULL DEFAULT ''
      );
      CREATE INDEX IF NOT EXISTS idx_jobs_submitted ON jobs(submitted_at DESC);
    """)
    CONN.commit()
    # Upgrade legacy v0.4 databases in place. No task/result rows are erased.
    with LOCK:
        columns = {r["name"] for r in db().execute("PRAGMA table_info(materials)").fetchall()}
        if "archived_at" not in columns:
            db().execute("ALTER TABLE materials ADD COLUMN archived_at TEXT")
            db().commit()
    # The restarted GUI cannot reliably reclaim a crashed process tree:
    # explicitly mark unfinished jobs as interrupted rather than fabricate success.
    with LOCK:
        db().execute("""UPDATE jobs SET state='interrupted', finished_at=?,
            error='GUI 服务意外重启：计算状态无法确认，请检查原始日志与系统进程。',
            updated_at=? WHERE state IN ('queued','running','stopping')""", (now(), now()))
        db().commit()


def clean_basename(name):
    raw = unquote(name).replace("\\", "/").split("/")[-1].strip()
    if not raw or len(raw) > 180 or raw in {".", ".."}:
        raise ValueError("无效文件名")
    clean = re.sub(r"[^A-Za-z0-9._+-]", "_", raw)
    if not clean or clean.startswith("."):
        clean = "structure_" + clean.lstrip(".")
    if clean.lower() not in ("poscar", "contcar") and Path(clean).suffix.lower() not in FILE_SUFFIX:
        raise ValueError("只支持 CIF / VASP / POSCAR / CONTCAR / XYZ 文件")
    return clean


def parse_crystal(path):
    # ASE can read the input without introducing any scientific alterations.
    from ase.io import read
    try:
        images = read(str(path), index=":")
    except Exception as exc:
        raise ValueError(f"无法解析晶体结构: {exc}") from exc
    if len(images) != 1:
        raise ValueError(f"当前版本要求一个晶体结构，实际发现 {len(images)} 个")
    atoms = images[0]
    if len(atoms) == 0:
        raise ValueError("晶体没有原子")
    if not all(bool(v) for v in atoms.pbc) or atoms.cell.rank != 3:
        raise ValueError("甲方计算引擎要求完整三维周期性晶胞")
    if any(abs(float(v)) > 1e-12 for v in atoms.get_initial_magnetic_moments()):
        raise ValueError("甲方当前版本不支持自旋极化初始磁矩")
    preview = {
        "formula": atoms.get_chemical_formula(mode="hill"),
        "atom_count": len(atoms),
        "cell": atoms.cell.array.tolist(),
        "pbc": [bool(x) for x in atoms.pbc],
        "symbols": atoms.get_chemical_symbols()[:450],
        "positions": atoms.positions.tolist()[:450],
        "preview_truncated": len(atoms) > 450,
        "lengths": [round(float(x), 6) for x in atoms.cell.lengths()],
        "angles": [round(float(x), 4) for x in atoms.cell.angles()],
    }
    return preview


def store_material(data: bytes, filename: str, origin: str):
    filename = clean_basename(filename)
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError("文件为空或超过 12 MB")
    mid = new_id()
    folder = DATA / "materials" / mid
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / filename
    path.write_bytes(data)
    try:
        preview = parse_crystal(path)
    except Exception:
        path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    digest = hashlib.sha256(data).hexdigest()
    with LOCK:
        db().execute("""INSERT INTO materials
            (id,label,filename,path,origin,formula,atom_count,sha256,preview,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (mid, filename, filename, str(path), origin, preview["formula"],
             preview["atom_count"], digest, json.dumps(preview), now()))
        db().commit()
    return get_material(mid)


def rename_material(mid, label):
    value = str(label or "").strip()
    if not value or len(value) > 100 or any(ord(ch) < 32 for ch in value):
        raise ValueError("材料名称不能为空，最长 100 个字符")
    if not one("SELECT id FROM materials WHERE id=?", (mid,)):
        raise ValueError("材料不存在")
    execute("UPDATE materials SET label=? WHERE id=?", (value, mid))
    return get_material(mid)


def remove_material(mid):
    """Remove from active tree without losing existing job provenance.

    A material without jobs is deleted from workspace storage. If it has jobs,
    archive the library entry but keep its input for reproducibility / reruns.
    Never touch SOURCE/examples or the customer's original code.
    """
    entry = one("SELECT * FROM materials WHERE id=?", (mid,))
    if not entry or entry.get("archived_at"):
        raise ValueError("材料不存在或已移除")
    count = one("SELECT COUNT(*) AS n FROM jobs WHERE material_id=?", (mid,))["n"]
    if count:
        execute("UPDATE materials SET archived_at=? WHERE id=?", (now(), mid))
        return {"ok": True, "archived": True, "jobs_preserved": count}
    execute("DELETE FROM materials WHERE id=?", (mid,))
    target = Path(entry["path"]).resolve()
    material_root = (DATA / "materials").resolve()
    if target.is_relative_to(material_root):
        if target.is_file():
            target.unlink()
        try:
            target.parent.rmdir()
        except OSError:
            pass
    return {"ok": True, "archived": False, "jobs_preserved": 0}


def get_material(mid, include_archived=False):
    material = one("SELECT * FROM materials WHERE id=?", (mid,))
    if material and (include_archived or not material.get("archived_at")):
        material["preview"] = json.loads(material["preview"])
        return material
    return None


def list_materials():
    rows = many("SELECT * FROM materials WHERE archived_at IS NULL ORDER BY created_at DESC")
    for row in rows:
        row["preview"] = json.loads(row["preview"])
    return rows


def samples():
    folder = SOURCE / "examples"
    if not folder.is_dir():
        return []
    result = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in FILE_SUFFIX and path.stat().st_size <= MAX_UPLOAD:
            result.append({"filename": path.name, "bytes": path.stat().st_size})
    return result


def validate_abacus(value):
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError("ABACUS 可执行文件需填写 WSL 绝对路径")
    return str(path.resolve())


def start_job(material_id, backend, abacus, device, note, allow_archived=False):
    if backend not in METHODS:
        raise ValueError("不支持的计算方法")
    if device not in ("cpu", "cuda"):
        raise ValueError("GNN 设备只能选 CPU 或 CUDA")
    material = get_material(str(material_id), include_archived=allow_archived)
    if material is None:
        raise ValueError("请选择已导入的晶体材料")
    if not Path(material["path"]).is_file():
        raise ValueError("晶体源文件已不存在")
    # All GPU/functional methods use the same verified installed GPU ABACUS.
    # An explicit path in the GUI takes priority; otherwise use launcher config.
    # GNN remains independent of the ABACUS binary and library environment.
    if backend != "gnn-hse06":
        abacus = validate_abacus(abacus or str(DEFAULT_ABACUS))
        abacus_bin = Path(abacus)
        if not abacus_bin.is_file() or not os.access(abacus_bin, os.X_OK):
            raise ValueError(
                f"ABACUS 不存在或不可执行：{abacus}。"
                "请确认编译路径，或设置 BGP_ABACUS 后重新启动 GUI。"
            )
    else:
        abacus = ""
    jid = new_id()
    jobdir = DATA / "jobs" / jid
    (jobdir / "input").mkdir(parents=True, exist_ok=False)
    source_input = Path(material["path"])
    input_path = jobdir / "input" / material["filename"]
    shutil.copyfile(source_input, input_path)
    run_root = jobdir / "engine-runs"
    run_root.mkdir()
    log_path = jobdir / "console.log"
    log_path.touch()
    record = (
        jid, material["id"], material["label"], material["formula"], backend,
        "queued", now(), now(), abacus, device, str(input_path),
        str(run_root), str(log_path), str(note or "")[:1000]
    )
    with LOCK:
        db().execute("""INSERT INTO jobs
           (id,material_id,material_label,material_formula,backend,state,
            submitted_at,updated_at,abacus_path,device,input_path,run_root,log_path,note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", record)
        db().commit()
        QUEUE.put(jid)
    return job_public(one("SELECT * FROM jobs WHERE id=?", (jid,)))


def job_public(row):
    if not row:
        return None
    row = dict(row)
    row.pop("input_path", None)
    row.pop("run_root", None)
    row["result"] = read_json(row.get("result_path")) if row.get("result_path") else None
    return row


def find_result(run_root):
    base = Path(run_root)
    if not base.is_dir():
        return None, ""
    dirs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        if (d / "result.json").is_file():
            return d / "result.json", str(d)
    return None, str(dirs[0]) if dirs else ""



def _gpu_smi_command():
    """WSL may install nvidia-smi outside the normal PATH."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    fallback = Path("/usr/lib/wsl/lib/nvidia-smi")
    return str(fallback) if fallback.is_file() else None


def _gpu_number(value):
    value = str(value or "").strip()
    if not value or value.lower() in ("n/a", "not supported", "[not supported]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _gpu_device_reading():
    """Read the physical GPU, not an assumed utilization of the ABACUS process."""
    command = _gpu_smi_command()
    if not command:
        return {"available": False, "error": "未找到nvidia-smi",
                "index": None, "name": None, "utilization_percent": None,
                "memory_used_mb": None, "memory_total_mb": None}
    try:
        result = subprocess.run(
            [command, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
            encoding="utf-8", errors="replace",
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[:250])
        records = list(csv.reader(result.stdout.splitlines()))
        if not records or len(records[0]) < 5:
            raise ValueError("GPU监控数据格式不完整")
        index, name, utilization, used, total = [x.strip() for x in records[0][:5]]
        return {"available": True, "error": "",
                "index": int(index) if index.isdecimal() else None,
                "name": name, "utilization_percent": _gpu_number(utilization),
                "memory_used_mb": _gpu_number(used),
                "memory_total_mb": _gpu_number(total)}
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        return {"available": False, "error": str(exc)[:250],
                "index": None, "name": None, "utilization_percent": None,
                "memory_used_mb": None, "memory_total_mb": None}


def _abacus_descendant_pids(root_pid):
    """Read live child processes, excluding the bgp parent whose argv contains --abacus."""
    try:
        result = subprocess.run(
            ["ps", "-e", "-o", "pid=", "-o", "ppid=", "-o", "comm=", "-o", "args="],
            capture_output=True, text=True, timeout=3, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    children = {}
    command_names = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        command_names[pid] = parts[2].lower()
    descendants = set()
    frontier = [int(root_pid)]
    while frontier and len(descendants) < 2048:
        parent = frontier.pop()
        for child in children.get(parent, []):
            if child not in descendants and child != root_pid:
                descendants.add(child)
                frontier.append(child)
    return sorted(pid for pid in descendants if command_names.get(pid, "").startswith("abacus"))


def _nvidia_compute_pids():
    """Return None if the driver cannot expose a compute-app PID list (common in WSL2)."""
    command = _gpu_smi_command()
    if not command:
        return None
    try:
        result = subprocess.run(
            [command, "--query-compute-apps=pid,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            encoding="utf-8", errors="replace",
        )
        if result.returncode:
            return None
        pids = set()
        for row in csv.reader(result.stdout.splitlines()):
            if row and row[0].strip().isdigit():
                pids.add(int(row[0].strip()))
        return pids
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _sample_gpu_job(jid, backend, engine_pid):
    sample = {"timestamp_utc": now(), "job_id": jid, "backend": backend,
              "engine_pid": engine_pid}
    sample.update(_gpu_device_reading())
    abacus_pids = _abacus_descendant_pids(engine_pid)
    apps = _nvidia_compute_pids() if abacus_pids else None
    sample["abacus_pids"] = abacus_pids
    sample["compute_app_pid_available"] = apps is not None
    sample["process_gpu_match"] = bool(apps is not None and set(abacus_pids) & apps)
    sample["evidence_scope"] = (
        "ABACUS进程级GPU证据" if sample["process_gpu_match"]
        else "整卡GPU活动与ABACUS子进程分别记录；未获得进程级GPU关联"
    )
    return sample


def monitor_gpu_job(jid, backend, engine_pid, stop_event):
    """Sampling must never change the scientific job's success/failure state."""
    log_file = DATA / "jobs" / jid / "gpu_monitor.csv"
    fields = ["timestamp_utc", "job_id", "backend", "engine_pid",
              "abacus_pids", "index", "name", "utilization_percent",
              "memory_used_mb", "memory_total_mb", "process_gpu_match",
              "compute_app_pid_available", "evidence_scope", "available", "error"]
    try:
        with log_file.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            while not stop_event.is_set():
                try:
                    sample = _sample_gpu_job(jid, backend, engine_pid)
                except Exception as exc:
                    sample = {"timestamp_utc": now(), "job_id": jid, "backend": backend,
                              "engine_pid": engine_pid, "available": False,
                              "error": "采样异常：" + str(exc)[:200],
                              "abacus_pids": [], "process_gpu_match": False,
                              "compute_app_pid_available": False,
                              "evidence_scope": "无法取得当前采样"}
                with GPU_LOCK:
                    GPU_LAST[jid] = sample
                row = dict(sample)
                row["abacus_pids"] = ";".join(map(str, sample.get("abacus_pids", [])))
                writer.writerow({k: row.get(k, "") for k in fields})
                fp.flush()
                stop_event.wait(2)
    except OSError:
        # Monitoring is supporting evidence, not part of the ABACUS calculation.
        pass


def get_gpu_job_status(jid):
    job = one("SELECT id, backend, state FROM jobs WHERE id=?", (jid,))
    if not job:
        return None
    with GPU_LOCK:
        sample = dict(GPU_LAST[jid]) if jid in GPU_LAST else None
    monitor_csv = DATA / "jobs" / jid / "gpu_monitor.csv"
    # Re-open the last stored observation after a server restart.
    if sample is None and monitor_csv.is_file():
        try:
            with monitor_csv.open("r", encoding="utf-8", newline="") as fp:
                last = None
                for item in csv.DictReader(fp):
                    last = item
            if last is not None:
                sample = dict(last)
                sample["abacus_pids"] = [int(x) for x in last.get("abacus_pids", "").split(";") if x.isdigit()]
                for field in ("utilization_percent", "memory_used_mb", "memory_total_mb"):
                    sample[field] = _gpu_number(last.get(field))
                for field in ("available", "process_gpu_match", "compute_app_pid_available"):
                    sample[field] = last.get(field, "").lower() == "true"
        except (OSError, ValueError):
            sample = None
    return {"job_id": jid, "backend": job["backend"], "state": job["state"],
            "sample": sample, "monitor_csv_exists": monitor_csv.is_file(),
            "monitor_file": "gpu_monitor.csv" if monitor_csv.is_file() else None}


def worker():
    while True:
        jid = QUEUE.get()
        try:
            job = one("SELECT * FROM jobs WHERE id=?", (jid,))
            if not job or job["state"] != "queued":
                continue
            if not BGP_BIN.is_file() or not os.access(BGP_BIN, os.X_OK):
                update_job(jid, state="failed", error=f"找不到 bgp：{BGP_BIN}",
                           finished_at=now(), exit_code=127)
                continue
            argv = [str(BGP_BIN), "predict", job["input_path"], "--backend",
                    job["backend"], "--work-root", job["run_root"]]
            if job["backend"] != "gnn-hse06":
                # Explicitly pin the executable for reproducible CLI/GUI results.
                argv += ["--abacus", job["abacus_path"]]
            env = dict(os.environ)
            if job["backend"] == "gnn-hse06":
                env["BGP_GNN_DEVICE"] = job["device"]
            # State transition is atomic with queued-cancel operations.
            with LOCK:
                current = one("SELECT state FROM jobs WHERE id=?", (jid,))
                if not current or current["state"] != "queued":
                    continue
                update_job(jid, state="running", started_at=now())
            code = None
            gpu_stop = threading.Event()
            gpu_thread = None
            try:
                with open(job["log_path"], "w", encoding="utf-8") as logfile:
                    logfile.write(f"[GUI] Started: {now()}\n[GUI] Backend: {job['backend']}\n")
                    logfile.write(f"[GUI] Work root: {job['run_root']}\n")
                    if job["backend"] != "gnn-hse06":
                        logfile.write(f"[GUI] ABACUS binary: {job['abacus_path']}\n")
                        logfile.write(f"[GUI] Runtime deps: {BGP_DEPS_PREFIX / 'lib'}\n")
                    logfile.flush()
                    proc = subprocess.Popen(
                        argv, cwd=str(DATA / "jobs" / jid), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", bufsize=1,
                        start_new_session=True
                    )
                    with LOCK:
                        RUNNING[jid] = proc
                        cancel_pending = one("SELECT state FROM jobs WHERE id=?", (jid,))["state"] == "stopping"
                    if job["backend"] != "gnn-hse06":
                        gpu_thread = threading.Thread(
                            target=monitor_gpu_job,
                            args=(jid, job["backend"], proc.pid, gpu_stop),
                            daemon=True, name="gpu-monitor-" + jid,
                        )
                        gpu_thread.start()
                    if cancel_pending:
                        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    assert proc.stdout is not None
                    try:
                        for line in proc.stdout:
                            logfile.write(line)
                            logfile.flush()
                    finally:
                        proc.stdout.close()
                    code = proc.wait()
            except Exception as exc:
                update_job(jid, error=f"进程调用失败：{exc}")
            finally:
                gpu_stop.set()
                if gpu_thread is not None:
                    gpu_thread.join(timeout=3)
                with LOCK:
                    RUNNING.pop(jid, None)

            result_path, run_dir = find_result(job["run_root"])
            result = read_json(result_path) if result_path else None
            prev = one("SELECT state FROM jobs WHERE id=?", (jid,))
            old_state = prev["state"] if prev else ""
            if result:
                st = result.get("status", "failed")
                state = st if st in TERMINAL else "failed"
                if state == "success" and code != 0:
                    state = "failed"
                if old_state == "stopping" and state != "success":
                    state = "interrupted"
                errmsg = ""
                if state != "success":
                    errmsg = str(result.get("diagnostics", {}).get("error_message") or f"计算未成功，退出码：{code}")
            else:
                state = "interrupted" if old_state == "stopping" or code in (130, -signal.SIGINT) else "failed"
                errmsg = f"未找到有效 result.json，bgp 返回码：{code}；请检查日志。"
            update_job(
                jid, state=state, finished_at=now(), exit_code=code,
                run_dir=run_dir, result_path=str(result_path) if result_path else "",
                error=errmsg
            )
        except Exception as exc:
            update_job(jid, state="failed", error=f"GUI 调度器错误：{exc}", finished_at=now())
        finally:
            QUEUE.task_done()


def cancel_job(jid):
    with LOCK:
        row = one("SELECT state FROM jobs WHERE id=?", (jid,))
        if not row:
            raise ValueError("任务不存在")
        st = row["state"]
        if st in TERMINAL:
            raise ValueError("任务已经结束")
        if st == "queued":
            update_job(jid, state="interrupted", finished_at=now(), exit_code=130, error="用户取消排队任务")
        else:
            update_job(jid, state="stopping")
            proc = RUNNING.get(jid)
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass


def environment():
    abacus_ok = DEFAULT_ABACUS.is_file() and os.access(DEFAULT_ABACUS, os.X_OK)
    dep_lib = BGP_DEPS_PREFIX / "lib"
    info = {"bgp": str(BGP_BIN), "installed": BGP_BIN.is_file(),
            "source": str(SOURCE), "data": str(DATA),
            "abacus": {
                "path": str(DEFAULT_ABACUS),
                "executable": bool(abacus_ok),
                "deps_lib": str(dep_lib),
                "deps_lib_present": dep_lib.is_dir(),
                "runtime_lib_inherited": str(dep_lib) in os.environ.get("LD_LIBRARY_PATH", "").split(":")
            },
            "python": sys.executable, "methods": [], "version": "", "error": ""}
    if not info["installed"]:
        info["error"] = "没有找到甲方 bgp。请使用 ~/BandGap-Studio/.venv 的已安装环境启动。"
        return info
    python_bin = BGP_BIN.parent / "python"
    info["engine_module"] = "未检查"
    info["source_match"] = None
    if python_bin.is_file():
        try:
            check = subprocess.run(
                [str(python_bin), "-c",
                 "import bandgap_predictor; print(bandgap_predictor.__file__)"],
                capture_output=True, text=True, timeout=18
            )
            module = check.stdout.strip().splitlines()[-1] if check.returncode == 0 and check.stdout.strip() else ""
            info["engine_module"] = module or check.stderr[-600:]
            if module:
                info["source_match"] = Path(module).resolve().is_relative_to(SOURCE)
                if not info["source_match"]:
                    info["error"] += "当前 bgp 可能指向非本次甲方源码，请核对可编辑安装路径；"
        except Exception as exc:
            info["engine_module"] = f"无法检查：{exc}"
    for kind, argv in (("version", ["--version"]), ("methods", ["backends"])):
        try:
            result = subprocess.run([str(BGP_BIN), *argv], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=20)
            output = (result.stdout + "\n" + result.stderr).strip()
            if kind == "version":
                info["version"] = output[:200]
            else:
                for line in output.splitlines():
                    m = re.match(r"^(gnn-hse06|abacus-hse06|abacus-pbe0|cider-pbe0):\s*(.*)$", line)
                    if m:
                        info["methods"].append({"id": m.group(1), "ready": m.group(2).strip() == "available",
                                                "reason": m.group(2).strip()})
                info["raw"] = output[-5000:]
            if result.returncode != 0:
                info["error"] += f"{kind} 命令返回码 {result.returncode}；"
        except Exception as exc:
            info["error"] += f"{kind}: {exc}；"
    return info


def doctor(abacus_path=""):
    args = ["doctor"]
    if abacus_path:
        args += ["--abacus", validate_abacus(abacus_path)]
    return run_diagnostic(args, timeout=35)


def run_diagnostic(args, timeout=60):
    if not BGP_BIN.is_file():
        return {"ok": False, "output": "未安装 bgp", "code": 127}
    try:
        p = subprocess.run([str(BGP_BIN), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return {"ok": p.returncode == 0, "code": p.returncode,
                "output": (p.stdout + "\n" + p.stderr)[-16000:]}
    except Exception as exc:
        return {"ok": False, "code": 1, "output": str(exc)}


def files_in_job(jid):
    j = one("SELECT * FROM jobs WHERE id=?", (jid,))
    if not j:
        return []
    root = DATA / "jobs" / jid
    if not root.is_dir():
        return []
    result = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if len(result) >= 350:
            break
        result.append({"path": rel, "bytes": path.stat().st_size})
    return sorted(result, key=lambda x: x["path"])


class App(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{now()}] {fmt % args}", flush=True)

    def _headers(self, ctype, length, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()

    def raw(self, data: bytes, ctype="application/octet-stream", status=200, filename=None):
        self._headers(ctype, len(data), status, filename)
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def out(self, obj, status=200):
        self.raw(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8", status)

    def err(self, message, status=400):
        self.out({"error": str(message)}, status)

    def body(self, maxsize=MAX_UPLOAD):
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > maxsize:
            raise ValueError(f"请求内容长度不符合限制（上限 {maxsize} 字节）")
        data = self.rfile.read(n)
        if len(data) != n:
            raise ValueError("数据上传不完整")
        return data

    def _check_local(self, write=False):
        hostname = self.headers.get("Host", "").split(":")[0].lower()
        if hostname not in ("localhost", "127.0.0.1"):
            return False
        if write:
            origin = self.headers.get("Origin", "")
            if origin and origin not in (f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"):
                return False
            if self.headers.get("X-BGP-UI") != "desktop":
                return False
        return True

    def do_GET(self):
        if not self._check_local():
            return self.err("仅允许本机访问", 403)
        path = urlsplit(self.path)
        route = path.path
        try:
            if route in ("/", "/index.html"):
                return self.raw((HERE / "static/index.html").read_bytes(), "text/html; charset=utf-8")
            if route in ("/static/style.css", "/static/app.js"):
                name = Path(route).name
                typ = "text/css" if name.endswith(".css") else "application/javascript"
                return self.raw((HERE / "static" / name).read_bytes(), typ+"; charset=utf-8")
            if route == "/api/system":
                return self.out(environment())
            if route == "/api/samples":
                return self.out(samples())
            if route == "/api/materials":
                return self.out(list_materials())
            if route == "/api/jobs":
                parsed = parse_qs(path.query)
                try:
                    limit = max(1, min(500, int(parsed.get("limit", ["250"])[0])))
                except ValueError:
                    limit = 250
                rows = many("SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,))
                return self.out([job_public(r) for r in rows])
            if route == "/api/stats":
                row = one("""SELECT COUNT(*) AS total,
                    SUM(CASE WHEN state='success' THEN 1 ELSE 0 END) AS succeeded,
                    SUM(CASE WHEN state='failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN state IN ('queued','running','stopping') THEN 1 ELSE 0 END) AS active
                    FROM jobs""")
                return self.out({k: (v or 0) for k,v in row.items()})
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})", route)
            if m:
                job = one("SELECT * FROM jobs WHERE id=?", (m[1],))
                if not job:
                    return self.err("任务不存在", 404)
                return self.out(job_public(job))
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/gpu", route)
            if m:
                record = get_gpu_job_status(m[1])
                return self.out(record) if record is not None else self.err("任务不存在", 404)
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/log", route)
            if m:
                j = one("SELECT log_path FROM jobs WHERE id=?", (m[1],))
                if not j:
                    return self.err("任务不存在", 404)
                target = Path(j["log_path"])
                if not target.is_file():
                    return self.out({"text": "", "size": 0})
                size = target.stat().st_size
                with target.open("rb") as f:
                    f.seek(max(0, size - 90000))
                    text = f.read().decode("utf-8", "replace")
                return self.out({"text": text, "size": size})
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/files", route)
            if m:
                return self.out(files_in_job(m[1]))
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/download", route)
            if m:
                jid = m[1]
                if not one("SELECT id FROM jobs WHERE id=?", (jid,)):
                    return self.err("任务不存在", 404)
                values = parse_qs(path.query)
                rel = values.get("path", [""])[0]
                root = (DATA / "jobs" / jid).resolve()
                target = (root / rel).resolve()
                if not rel or not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
                    return self.err("无效文件路径", 400)
                # Stream downloads in bounded chunks (may include larger ABACUS logs).
                size = target.stat().st_size
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                self._headers(ctype, size, filename=re.sub(r"[^A-Za-z0-9_.-]", "_", target.name))
                with target.open("rb") as f:
                    while chunk := f.read(1024 * 512):
                        self.wfile.write(chunk)
                return
            return self.err("接口不存在", 404)
        except Exception as exc:
            return self.err(f"服务端处理失败：{exc}", 500)

    def do_POST(self):
        if not self._check_local(write=True):
            return self.err("操作仅允许本机网页提交", 403)
        route = urlsplit(self.path).path
        try:
            if route == "/api/materials/upload":
                filename = self.headers.get("X-Filename", "")
                material = store_material(self.body(), filename, "uploaded")
                return self.out(material, 201)
            if route == "/api/materials/from-sample":
                req = json.loads(self.body(4096))
                filename = req.get("filename", "")
                if filename not in [x["filename"] for x in samples()]:
                    raise ValueError("不是允许的内置材料样例")
                p = SOURCE / "examples" / filename
                return self.out(store_material(p.read_bytes(), filename, "example"), 201)
            m = re.fullmatch(r"/api/materials/([a-f0-9]{16})/(rename|remove)", route)
            if m:
                if m[2] == "rename":
                    req = json.loads(self.body(3000))
                    return self.out(rename_material(m[1], req.get("label", "")))
                self.body(3000)
                return self.out(remove_material(m[1]))
            if route == "/api/jobs":
                req = json.loads(self.body(12000))
                job = start_job(req.get("material_id"), req.get("backend"),
                                req.get("abacus", ""), req.get("device", "cpu"),
                                req.get("note", ""))
                return self.out(job, 201)
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/cancel", route)
            if m:
                cancel_job(m[1])
                return self.out({"ok": True})
            m = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/duplicate", route)
            if m:
                old = one("SELECT * FROM jobs WHERE id=?", (m[1],))
                if not old:
                    return self.err("任务不存在", 404)
                return self.out(start_job(old["material_id"], old["backend"], old["abacus_path"],
                                          old["device"], old["note"], allow_archived=True), 201)
            if route == "/api/diagnostic":
                req = json.loads(self.body(5000))
                action = req.get("action")
                if action == "assets":
                    return self.out(run_diagnostic(["assets", "verify"], timeout=120))
                if action == "doctor":
                    return self.out(doctor(req.get("abacus", "")))
                if action == "backends":
                    return self.out(run_diagnostic(["backends"], timeout=35))
                raise ValueError("不支持的环境诊断命令")
            return self.err("接口不存在", 404)
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            return self.err(str(exc), 400)
        except Exception as exc:
            return self.err(f"服务端处理失败：{exc}", 500)


def main():
    init_store()
    threading.Thread(target=worker, daemon=True, name="bgp-job-worker").start()
    print("GNNBGP Industrial Workbench v0.6")
    print(f"ABACUS GPU: {DEFAULT_ABACUS} ({'executable' if DEFAULT_ABACUS.is_file() and os.access(DEFAULT_ABACUS, os.X_OK) else 'not configured'})")
    print(f"Source: {SOURCE}")
    print(f"BGP CLI: {BGP_BIN}")
    print(f"Workspace: {DATA}")
    print(f"Open: http://{LISTEN}:{PORT}", flush=True)
    server = ThreadingHTTPServer((LISTEN, PORT), App)
    try:
        server.serve_forever(poll_interval=0.35)
    except KeyboardInterrupt:
        print("\nStopping GUI. Interrupting current CLI process ...", flush=True)
    finally:
        with LOCK:
            processes = list(RUNNING.values())
        for proc in processes:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        server.server_close()
        db().close()


if __name__ == "__main__":
    main()
