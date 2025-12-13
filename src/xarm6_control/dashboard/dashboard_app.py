#!/usr/bin/env python3
"""
dashboard_app.py

Web dashboard for:
- Starting the XArm policy server via HPC orchestrator OR locally
- Showing orchestrator state live
- Displaying two camera feeds (base + wrist) via MJPEG
- Running the local xArm client with a chosen prompt
- Capturing and exposing local policy server stdout/stderr to the UI
- Stopping policy server:
    * LOCAL: kill local process group OR cancel startup
    * HPC: request cancel; qdel once job_id is known; stop tunnel too

Run with:
    uv run src/xarm6_control/dashboard/dashboard_app.py

Then open:
    http://localhost:9000
"""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Optional, Dict, Any, List
from enum import Enum
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import websockets.sync.client

from xarm6_control.dashboard.xarm_server_orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorState,
)

from xarm6_control.dashboard.zmq_camera_backend import (
    ZmqCameraBackend,
    mjpeg_stream_generator,
)

# ============================================================
# DEFINES
# ============================================================

DASHBOARD_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = DASHBOARD_DIR / "xarm_pipeline.sh"

DEFAULT_XARM_PROMPT = "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
XARM_PORT: int = 8000


def _find_openpi_root() -> Path:
    """
    Find the repo root robustly.
    Priority:
      1) env OPENPI_ROOT
      2) walk upwards looking for scripts/serve_policy.py
      3) fallback to parents[3] (old behaviour)
    """
    env_root = os.environ.get("OPENPI_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        return p

    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "scripts" / "serve_policy.py").exists():
            return p

    # fallback (your previous behaviour)
    return here.parents[3]


OPENPI_ROOT = _find_openpi_root()

# ============================================================
# Modes
# ============================================================

class ServerMode(str, Enum):
    HPC = "hpc"
    LOCAL = "local"


# ============================================================
# Global orchestrator + state
# ============================================================

app = FastAPI(title="XArm6 Demo Dashboard")

_status_lock = threading.Lock()
_last_status: Dict[str, Any] = {
    "state": OrchestratorState.IDLE.value,
    "message": "Idle",
    "server_mode": ServerMode.LOCAL.value,  # display mode (active if running, else preferred)
}

# Preferred mode = what user selected while idle
_preferred_mode_lock = threading.Lock()
_preferred_mode: ServerMode = ServerMode.LOCAL

# Active mode = what is actually running/orchestrating
_active_mode_lock = threading.Lock()
_active_mode: Optional[ServerMode] = None

# Cancel coordination for HPC (if job_id not known yet)
_hpc_cancel_lock = threading.Lock()
_hpc_cancel_requested: bool = False

# Camera server process
_camera_server_lock = threading.Lock()
_camera_server_process: Optional[subprocess.Popen] = None

# Orchestrator + thread
_orchestrator: Optional[Orchestrator] = None
_orch_thread: Optional[threading.Thread] = None
_orch_thread_lock = threading.Lock()

# Local policy server process + logs
_local_server_lock = threading.Lock()
_local_server_process: Optional[subprocess.Popen] = None
_local_server_log_lock = threading.Lock()
_local_server_log_lines: List[str] = []
_local_server_returncode: Optional[int] = None
_MAX_LOCAL_SERVER_LOG_LINES = 500

_local_start_thread: Optional[threading.Thread] = None
_local_start_thread_lock = threading.Lock()

# Local start cancel event (critical: allows Shutdown to cancel even if no proc exists yet)
_local_cancel_event = threading.Event()

# xArm client process + logs
_xarm_lock = threading.Lock()
_xarm_process: Optional[subprocess.Popen] = None
_xarm_log_lines: List[str] = []
_MAX_XARM_LOG_LINES = 200


# ============================================================
# Helpers: mode + status
# ============================================================

def _set_active_mode(mode: Optional[ServerMode]) -> None:
    global _active_mode
    with _active_mode_lock:
        _active_mode = mode


def _get_active_mode() -> Optional[ServerMode]:
    with _active_mode_lock:
        return _active_mode


def _set_preferred_mode(mode: ServerMode) -> None:
    global _preferred_mode
    with _preferred_mode_lock:
        _preferred_mode = mode


def _get_preferred_mode() -> ServerMode:
    with _preferred_mode_lock:
        return _preferred_mode


def _get_display_mode() -> ServerMode:
    am = _get_active_mode()
    return am if am is not None else _get_preferred_mode()


def _set_status(state: str, message: str, server_mode: Optional[ServerMode] = None) -> None:
    with _status_lock:
        _last_status["state"] = state
        _last_status["message"] = message
        _last_status["server_mode"] = (server_mode.value if server_mode is not None else _get_display_mode().value)


def _append_local_server_log(line: str) -> None:
    line = line.rstrip("\n")
    with _local_server_log_lock:
        _local_server_log_lines.append(line)
        if len(_local_server_log_lines) > _MAX_LOCAL_SERVER_LOG_LINES:
            del _local_server_log_lines[:-_MAX_LOCAL_SERVER_LOG_LINES]
    print(f"[local-policy-log] {line}")


def _set_local_server_returncode(rc: Optional[int]) -> None:
    global _local_server_returncode
    with _local_server_log_lock:
        _local_server_returncode = rc


def _append_xarm_log(line: str) -> None:
    line = line.rstrip("\n")
    with _xarm_lock:
        _xarm_log_lines.append(line)
        if len(_xarm_log_lines) > _MAX_XARM_LOG_LINES:
            del _xarm_log_lines[:-_MAX_XARM_LOG_LINES]
    print(f"[xarm-client-log] {line}")


# ============================================================
# Process state helpers
# ============================================================

def _local_server_running() -> bool:
    with _local_server_lock:
        return _local_server_process is not None and _local_server_process.poll() is None


def _local_starting() -> bool:
    with _local_start_thread_lock:
        return _local_start_thread is not None and _local_start_thread.is_alive()


def _any_server_running_or_busy() -> bool:
    with _orch_thread_lock:
        hpc_thread_alive = _orch_thread is not None and _orch_thread.is_alive()

    orch = _orchestrator
    hpc_ready = orch is not None and orch.state == OrchestratorState.READY
    local_ready = _local_server_running()

    return hpc_thread_alive or _local_starting() or hpc_ready or local_ready


def _self_heal_active_mode_if_stale() -> None:
    """
    If active_mode says LOCAL but nothing is running/starting, clear it.
    Same idea prevents the UI from getting stuck with Run disabled forever.
    """
    am = _get_active_mode()
    if am == ServerMode.LOCAL:
        if (not _local_server_running()) and (not _local_starting()):
            _set_active_mode(None)

    if am == ServerMode.HPC:
        with _orch_thread_lock:
            hpc_thread_alive = _orch_thread is not None and _orch_thread.is_alive()
        orch = _orchestrator
        hpc_ready = orch is not None and orch.state == OrchestratorState.READY
        if (not hpc_thread_alive) and (not hpc_ready):
            _set_active_mode(None)


# ============================================================
# Health checking
# ============================================================

def _ws_health_once(host: str, port: int, timeout_s: float = 3.0) -> tuple[bool, str]:
    uri = f"ws://{host}:{int(port)}"
    try:
        ws = websockets.sync.client.connect(
            uri,
            open_timeout=timeout_s,
            close_timeout=timeout_s,
        )
        ws.close()
        return True, f"WebSocket handshake succeeded at {uri}."
    except Exception as e:
        return False, f"{uri} -> {type(e).__name__}: {e}"


def _wait_for_local_server_healthy(
    proc: subprocess.Popen,
    cancel_event: threading.Event,
    timeout_s: float = 900.0,
    poll_s: float = 2.0,
) -> None:
    deadline = time.time() + timeout_s

    last_log_t = 0.0
    last_msg = ""

    while time.time() < deadline:
        if cancel_event.is_set():
            raise RuntimeError("Local startup cancelled.")

        ret = proc.poll()
        if ret is not None:
            _set_local_server_returncode(ret)
            raise RuntimeError(f"Local policy server exited during startup (rc={ret}). See Policy Server Log.")

        ok, msg = _ws_health_once("127.0.0.1", XARM_PORT, timeout_s=2.0)
        if ok:
            return

        # Throttle identical spam
        now = time.time()
        if msg != last_msg or (now - last_log_t) > 10.0:
            _append_local_server_log(f"[dashboard] Waiting for local server health... ({msg})")
            last_msg = msg
            last_log_t = now

        time.sleep(poll_s)

    raise RuntimeError(f"Timed out waiting for local server to become healthy (>{timeout_s}s).")


# ============================================================
# HPC cancel coordination
# ============================================================

def _set_hpc_cancel_requested(v: bool) -> None:
    global _hpc_cancel_requested
    with _hpc_cancel_lock:
        _hpc_cancel_requested = v


def _get_hpc_cancel_requested() -> bool:
    with _hpc_cancel_lock:
        return _hpc_cancel_requested


def _attempt_cancel_hpc_if_requested() -> None:
    """
    If cancel was requested and we now have a job_id, delete it (via orchestrator SSH)
    and stop the tunnel too. Safe to call from orchestrator callback.
    """
    if not _get_hpc_cancel_requested():
        return

    orch = _orchestrator
    if orch is None:
        return

    job_id = getattr(orch, "job_id", None)
    if not job_id:
        return

    # Make sure the orchestrator stops its loops ASAP
    try:
        orch.request_cancel()
    except Exception:
        pass

    # Stop tunnel (local)
    try:
        orch.stop_tunnel()
    except Exception:
        pass

    # Delete job (remote, via ssh aqua)
    ok, msg = orch.delete_job()

    _set_hpc_cancel_requested(False)
    _set_active_mode(None)
    _set_status(
        OrchestratorState.IDLE.value,
        f"HPC job cancelled: {job_id}\n\n{msg}",
        server_mode=ServerMode.HPC,
    )


# ============================================================
# Orchestrator callback + thread start
# ============================================================

def _on_state_change(state: OrchestratorState, message: str) -> None:
    _set_status(state.value, message, server_mode=ServerMode.HPC)
    print(f"[DASHBOARD][HPC] {state.value}: {message}")

    # If the user hit Shutdown while we're starting, cancel as soon as job_id appears
    _attempt_cancel_hpc_if_requested()

    # If we reach a terminal error/cancel state, clear active mode
    if state in {
        OrchestratorState.CANCELLED,
        OrchestratorState.ERROR_JOB_SUBMISSION,
        OrchestratorState.ERROR_JOB_STARTUP,
        OrchestratorState.ERROR_TUNNEL_START,
        OrchestratorState.ERROR_SERVER_UNHEALTHY,
        OrchestratorState.ERROR_TUNNEL_BROKEN,
        OrchestratorState.ERROR_STOP,
    }:
        _set_active_mode(None)


def _start_orchestrator_thread() -> bool:
    global _orchestrator, _orch_thread

    with _orch_thread_lock:
        if _orch_thread is not None and _orch_thread.is_alive():
            return False

        if _orchestrator is not None and _orchestrator.state == OrchestratorState.READY:
            return False

        config = OrchestratorConfig()
        orch = Orchestrator(config=config)
        orch.on_state_change = _on_state_change
        _orchestrator = orch

        def _run():
            try:
                orch.run_full_sequence()
            finally:
                pass

        t = threading.Thread(target=_run, daemon=True)
        _orch_thread = t
        t.start()
        return True


# ============================================================
# Local policy server start/stop (logs + kill group + cancel startup)
# ============================================================

def _local_server_log_reader(proc: subprocess.Popen) -> None:
    global _local_server_process
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            _append_local_server_log(line)
    except Exception as e:
        _append_local_server_log(f"[dashboard] Error reading local policy output: {e}")
    finally:
        rc = proc.wait()
        _set_local_server_returncode(rc)
        _append_local_server_log(f"[dashboard] Local policy server exited with return code {rc}")

        with _local_server_lock:
            _local_server_process = None

        if _get_active_mode() == ServerMode.LOCAL:
            _set_active_mode(None)

        if rc != 0:
            _set_status(
                "ERROR_LOCAL_EXIT",
                f"Local policy server exited (rc={rc}). See Policy Server Log.",
                server_mode=ServerMode.LOCAL,
            )
        else:
            _set_status(
                OrchestratorState.IDLE.value,
                "Local policy server stopped.",
                server_mode=ServerMode.LOCAL,
            )


def _start_local_policy_server_blocking() -> None:
    global _local_server_process

    _local_cancel_event.clear()

    with _local_server_lock:
        if _local_server_process is not None and _local_server_process.poll() is None:
            raise RuntimeError("Local policy server is already running.")

    with _local_server_log_lock:
        _local_server_log_lines.clear()
        _set_local_server_returncode(None)

    _set_status("STARTING_LOCAL", "Starting local policy server...", server_mode=ServerMode.LOCAL)

    # Always log something BEFORE we attempt to spawn
    _append_local_server_log("[dashboard] Local start requested.")
    _append_local_server_log(f"[dashboard] OPENPI_ROOT: {OPENPI_ROOT}")
    _append_local_server_log(f"[dashboard] serve_policy exists? {(OPENPI_ROOT / 'scripts' / 'serve_policy.py').exists()}")
    _append_local_server_log(f"[dashboard] XARM_PORT: {XARM_PORT}")

    if _local_cancel_event.is_set():
        raise RuntimeError("Local startup cancelled before launch.")

    if not (OPENPI_ROOT / "scripts" / "serve_policy.py").exists():
        raise RuntimeError(f"scripts/serve_policy.py not found under OPENPI_ROOT={OPENPI_ROOT}")

    # Keep using uv (matches how you run it manually), but run in its own process group.
    cmd = [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--env",
        "DEMO",
        "--port",
        str(XARM_PORT),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(OPENPI_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start local policy server: {e}") from e

    with _local_server_lock:
        _local_server_process = proc

    _append_local_server_log("[dashboard] Launched local policy server process.")
    _append_local_server_log(f"[dashboard] CWD: {OPENPI_ROOT}")
    _append_local_server_log(f"[dashboard] CMD: {' '.join(cmd)}")

    threading.Thread(target=_local_server_log_reader, args=(proc,), daemon=True).start()

    _set_status("CHECKING_LOCAL_HEALTH", "Waiting for local server to become healthy...", server_mode=ServerMode.LOCAL)
    _wait_for_local_server_healthy(proc, _local_cancel_event, timeout_s=240.0, poll_s=2.0)

    if _local_cancel_event.is_set():
        raise RuntimeError("Local startup cancelled after launch.")

    _set_status(
        OrchestratorState.READY.value,
        f"Local policy server healthy on ws://localhost:{XARM_PORT}",
        server_mode=ServerMode.LOCAL,
    )


def _start_local_policy_server_async() -> bool:
    global _local_start_thread

    with _local_start_thread_lock:
        if _local_start_thread is not None and _local_start_thread.is_alive():
            return False
        if _local_server_running():
            return False

        def _run():
            try:
                _start_local_policy_server_blocking()
            except Exception as e:
                # If this is just a cancel, keep it clean
                if "cancel" in str(e).lower():
                    _append_local_server_log(f"[dashboard] {e}")
                    _set_status(OrchestratorState.IDLE.value, "Local startup cancelled.", server_mode=ServerMode.LOCAL)
                else:
                    _append_local_server_log(f"[dashboard] Local start failed: {e}")
                    _set_status("ERROR_LOCAL_START", f"Local server failed to start.\n\n{e}", server_mode=ServerMode.LOCAL)
                _set_active_mode(None)

        _local_start_thread = threading.Thread(target=_run, daemon=True)
        _local_start_thread.start()
        return True


def _stop_local_policy_server() -> tuple[bool, str]:
    """
    Stop local server OR cancel local startup if we are in the "starting" window.
    """
    global _local_server_process

    # Cancel startup if thread is alive but proc isn't running yet (or already died)
    if _local_starting() and (not _local_server_running()):
        _local_cancel_event.set()
        _set_active_mode(None)
        _set_status(OrchestratorState.IDLE.value, "Local startup cancelled.", server_mode=ServerMode.LOCAL)
        _append_local_server_log("[dashboard] Cancelled local startup (no running proc).")
        return True, "Local startup cancelled."

    with _local_server_lock:
        proc = _local_server_process

    if proc is None or proc.poll() is not None:
        # Also clear stale active_mode if it got wedged
        if _get_active_mode() == ServerMode.LOCAL:
            _set_active_mode(None)
        return False, "Local policy server is not running."

    _set_status("STOPPING_LOCAL", "Stopping local policy server...", server_mode=ServerMode.LOCAL)
    _local_cancel_event.set()

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception as e:
        return False, f"Failed to SIGTERM local server group: {e}"

    # Give it a moment then SIGKILL if needed
    for _ in range(20):  # ~2s
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception as e:
            return False, f"Failed to SIGKILL local server group: {e}"

    with _local_server_lock:
        _local_server_process = None

    _set_active_mode(None)
    _set_status(OrchestratorState.IDLE.value, "Local policy server stopped.", server_mode=ServerMode.LOCAL)
    return True, "Local policy server stopped."


# ============================================================
# Camera backends
# ============================================================

base_camera = ZmqCameraBackend(host="172.23.224.1", port=5000, img_size=None, name="base", target_fps=15.0)
wrist_camera = ZmqCameraBackend(host="172.23.224.1", port=5001, img_size=None, name="wrist", target_fps=15.0)

# ============================================================
# HTML Frontend
# ============================================================

_INDEX_PATH = Path(__file__).with_name("index.html")
if not _INDEX_PATH.exists():
    raise RuntimeError(f"index.html not found next to dashboard_app.py at: {_INDEX_PATH}")
INDEX_HTML = _INDEX_PATH.read_text(encoding="utf-8")


# ============================================================
# Request models
# ============================================================

class RunXarmRequest(BaseModel):
    prompt: Optional[str] = None


class RunServerRequest(BaseModel):
    mode: ServerMode = ServerMode.LOCAL


class SetModeRequest(BaseModel):
    mode: ServerMode


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/set-mode", response_class=JSONResponse)
def api_set_mode(req: SetModeRequest) -> Dict[str, Any]:
    _self_heal_active_mode_if_stale()
    if _any_server_running_or_busy():
        raise HTTPException(status_code=409, detail="Cannot change mode while a server is running or starting.")
    _set_preferred_mode(req.mode)
    _set_status(OrchestratorState.IDLE.value, f"Mode set to {req.mode.value.upper()}.", server_mode=req.mode)
    return {"status": "ok", "preferred_mode": req.mode.value}


@app.get("/api/status", response_class=JSONResponse)
def api_status() -> Dict[str, Any]:
    _self_heal_active_mode_if_stale()

    with _status_lock:
        state = _last_status["state"]
        message = _last_status["message"]
        server_mode = _last_status.get("server_mode", _get_display_mode().value)

    with _orch_thread_lock:
        hpc_thread_alive = _orch_thread is not None and _orch_thread.is_alive()

    local_thread_alive = _local_starting()

    orch = _orchestrator
    hpc_ready = orch is not None and orch.state == OrchestratorState.READY
    local_ready = _local_server_running()

    server_running = bool(hpc_ready or local_ready)
    busy = bool(hpc_thread_alive or local_thread_alive)

    # Job/tunnel info only meaningful in HPC if HPC is active/running/busy
    job_id = getattr(orch, "job_id", None) if orch is not None else None
    remote_ip = getattr(orch, "remote_ip", None) if orch is not None else None
    remote_port = getattr(orch, "remote_port", None) if orch is not None else None

    active = _get_active_mode()
    if (active != ServerMode.HPC) and (not hpc_thread_alive) and (not hpc_ready):
        job_id = None
        remote_ip = None
        remote_port = None

    return {
        "state": state,
        "message": message,
        "job_id": job_id,
        "remote_ip": remote_ip,
        "remote_port": remote_port,
        "server_running": server_running,
        "busy": busy,
        "server_mode": server_mode,
        "preferred_mode": _get_preferred_mode().value,
        "active_mode": (active.value if active is not None else None),
        "hpc_running": bool(hpc_ready),
        "local_running": bool(local_ready),
        "hpc_cancel_requested": _get_hpc_cancel_requested(),
    }


@app.get("/api/policy-server-log", response_class=JSONResponse)
def api_policy_server_log() -> Dict[str, Any]:
    with _local_server_log_lock:
        log = "\n".join(_local_server_log_lines)
        rc = _local_server_returncode
    return {
        "mode": _get_display_mode().value,
        "local_running": _local_server_running(),
        "local_returncode": rc,
        "log": log,
    }


@app.get("/api/hpc-diagnostics", response_class=JSONResponse)
def api_hpc_diagnostics() -> Dict[str, Any]:
    orch = _orchestrator
    if orch is None:
        return {"ok": False, "diagnostics": "Orchestrator not initialized yet."}
    try:
        return {"ok": True, "diagnostics": orch.collect_diagnostics()}
    except Exception as e:
        return {"ok": False, "diagnostics": f"Failed to collect diagnostics: {e}"}


@app.post("/api/run-server", response_class=JSONResponse)
def api_run_server(req: RunServerRequest) -> Dict[str, Any]:
    _self_heal_active_mode_if_stale()
    if _any_server_running_or_busy():
        raise HTTPException(status_code=409, detail="Policy server already running (HPC or local).")

    _set_preferred_mode(req.mode)

    if req.mode == ServerMode.HPC:
        _set_hpc_cancel_requested(False)
        _set_status(OrchestratorState.SUBMITTING_JOB.value, "Submitting policy server job...", server_mode=ServerMode.HPC)

        started = _start_orchestrator_thread()
        if not started:
            _set_active_mode(None)
            raise HTTPException(status_code=409, detail="Server orchestration already running or server already READY.")

        _set_active_mode(ServerMode.HPC)
        return {"status": "started", "mode": "hpc"}

    # LOCAL (async so the dashboard doesn't die/block)
    started = _start_local_policy_server_async()
    if not started:
        _set_active_mode(None)
        raise HTTPException(status_code=409, detail="Local server is already starting or running.")

    _set_active_mode(ServerMode.LOCAL)
    return {"status": "starting", "mode": "local"}


@app.post("/api/stop-server", response_class=JSONResponse)
def api_stop_server() -> Dict[str, Any]:
    """
    Stop whichever server is active (or cancel startup):
    - LOCAL: kill local process group OR cancel startup
    - HPC: request cancel; delete job once job_id is known; stop tunnel too
    """
    _self_heal_active_mode_if_stale()
    mode = _get_active_mode() or _get_display_mode()

    if mode == ServerMode.LOCAL:
        ok, detail = _stop_local_policy_server()
        if not ok:
            raise HTTPException(status_code=400, detail=detail)
        return {"status": "stopped", "mode": "local", "detail": detail}

    # HPC stop/cancel
    orch = _orchestrator
    _set_active_mode(ServerMode.HPC)
    _set_hpc_cancel_requested(True)
    _set_status("CANCEL_REQUESTED", "Cancelling HPC server/job...", server_mode=ServerMode.HPC)

    if orch is not None:
        try:
            orch.request_cancel()
        except Exception:
            pass

    # If job_id already known, cancel immediately
    _attempt_cancel_hpc_if_requested()

    return {
        "status": "cancelling",
        "mode": "hpc",
        "detail": "Cancel requested. If job_id is already known it will be qdel'd immediately; otherwise it will be deleted as soon as it appears.",
    }


@app.get("/api/health", response_class=JSONResponse)
def api_health() -> Dict[str, Any]:
    mode = _get_display_mode()

    if mode == ServerMode.LOCAL:
        ok, msg = _ws_health_once("127.0.0.1", XARM_PORT, timeout_s=3.0)
        return {"status": "healthy" if ok else "unhealthy", "detail": msg}

    orch = _orchestrator
    if orch is None:
        return {"status": "unknown", "detail": "Orchestrator has not run yet."}

    ok = orch.check_policy_health()
    return {"status": "healthy" if ok else "unhealthy", "detail": orch.last_message}


@app.post("/api/start-cameras", response_class=JSONResponse)
def api_start_cameras() -> Dict[str, Any]:
    global _camera_server_process

    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Pipeline script not found at {PIPELINE_SCRIPT}")

    with _camera_server_lock:
        if _camera_server_process is not None and _camera_server_process.poll() is None:
            raise HTTPException(status_code=409, detail="Camera server already running.")

        cmd = ["bash", str(PIPELINE_SCRIPT), "cameras"]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(DASHBOARD_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start camera server: {e}")

        time.sleep(3.0)
        ret = proc.poll()
        if ret is not None and ret != 0:
            try:
                output, _ = proc.communicate(timeout=1.0)
            except Exception:
                output = "<no output captured>"
            raise HTTPException(status_code=500, detail=f"Camera server exited early with code {ret}:\n{output}")

        _camera_server_process = proc

    return {"status": "started"}


@app.get("/video/base")
def video_base() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(base_camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/wrist")
def video_wrist() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(wrist_camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/camera-status", response_class=JSONResponse)
def api_camera_status() -> Dict[str, Any]:
    with _camera_server_lock:
        server_running = (_camera_server_process is not None and _camera_server_process.poll() is None)

    return {
        "server_running": server_running,
        "base": base_camera.get_status(),
        "wrist": wrist_camera.get_status(),
    }


# ============================================================
# xArm client API
# ============================================================

@app.post("/api/run-xarm", response_class=JSONResponse)
def api_run_xarm(req: RunXarmRequest) -> Dict[str, Any]:
    global _xarm_process

    with _status_lock:
        state = _last_status["state"]

    if state != OrchestratorState.READY.value:
        raise HTTPException(status_code=400, detail=f"Policy server not READY (current state: {state}). Start it first.")

    with _xarm_lock:
        if _xarm_process is not None and _xarm_process.poll() is None:
            raise HTTPException(status_code=409, detail="xArm client is already running.")

        prompt = (req.prompt or DEFAULT_XARM_PROMPT).strip() or DEFAULT_XARM_PROMPT
        cmd = [
            "uv",
            "run",
            "src/xarm6_control/main2.py",
            "--remote_host",
            "127.0.0.1",
            "--remote_port",
            str(XARM_PORT),
            "--prompt",
            prompt,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(OPENPI_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start xArm client: {e}")

        _xarm_process = proc
        _xarm_log_lines.clear()
        _append_xarm_log("[dashboard] Starting xArm client...")
        _append_xarm_log(f"[dashboard] Prompt: {prompt}")

        def _xarm_log_reader(p: subprocess.Popen) -> None:
            global _xarm_process
            try:
                assert p.stdout is not None
                for line in p.stdout:
                    _append_xarm_log(line)
            except Exception as e:
                _append_xarm_log(f"[dashboard] Error reading xArm client output: {e}")
            finally:
                rc = p.wait()
                _append_xarm_log(f"[dashboard] xArm client exited with return code {rc}")
                with _xarm_lock:
                    if _xarm_process is p:
                        _xarm_process = None

        threading.Thread(target=_xarm_log_reader, args=(proc,), daemon=True).start()

    return {"status": "started"}


@app.post("/api/stop-xarm", response_class=JSONResponse)
def api_stop_xarm() -> Dict[str, Any]:
    global _xarm_process
    with _xarm_lock:
        proc = _xarm_process

    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=400, detail="xArm client is not running.")

    try:
        proc.terminate()
        _append_xarm_log("[dashboard] Sent terminate() to xArm client.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop xArm client: {e}")

    return {"status": "terminating"}


@app.get("/api/xarm-status", response_class=JSONResponse)
def api_xarm_status() -> Dict[str, Any]:
    with _xarm_lock:
        proc = _xarm_process
        if proc is None:
            running = False
            returncode = None
        else:
            running = proc.poll() is None
            returncode = proc.returncode
        log = "\n".join(_xarm_log_lines)

    return {"running": running, "returncode": returncode, "log": log}


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "xarm6_control.dashboard.dashboard_app:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )
