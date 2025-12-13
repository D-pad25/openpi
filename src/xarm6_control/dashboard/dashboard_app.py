#!/usr/bin/env python3
"""
dashboard_app.py

Web dashboard for:
- Starting the XArm policy server via HPC orchestrator OR locally
- Showing orchestrator state live
- Displaying two camera feeds (base + wrist) via MJPEG
- Running the local xArm client with a chosen prompt
- Capturing and exposing local policy server stdout/stderr to the UI

Run with:
    uv run src/xarm6_control/dashboard/dashboard_app.py

Then open:
    http://localhost:9000
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any, Generator, List
from enum import Enum
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import websockets.sync.client

# Import orchestrator (with CLEARING_OLD_ENDPOINT, health polling, etc.)
from xarm6_control.dashboard.xarm_server_orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorState,
)

# Import ZMQ camera backend (separate file)
from xarm6_control.dashboard.zmq_camera_backend import (
    ZmqCameraBackend,
    mjpeg_stream_generator,
)

# ============================================================
# DEFINES
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = REPO_ROOT / "xarm_pipeline.sh"

# Top-level openpi repo root (where src/ lives)
OPENPI_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_XARM_PROMPT = "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
XARM_PORT = "8000"


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
    # This will represent the mode to DISPLAY in the UI (active if running, otherwise preferred)
    "server_mode": ServerMode.LOCAL.value,
}

# Preferred mode = what user selected in UI (while idle)
_preferred_mode_lock = threading.Lock()
_preferred_mode: ServerMode = ServerMode.LOCAL

# Active mode = what is actually running (or orchestrating). None means idle.
_active_mode_lock = threading.Lock()
_active_mode: Optional[ServerMode] = None

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
_MAX_LOCAL_SERVER_LOG_LINES = 400

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
    """
    Display mode: if something is running/orchestrating -> active mode
    else -> preferred mode.
    """
    am = _get_active_mode()
    return am if am is not None else _get_preferred_mode()


def _set_status(state: str, message: str, server_mode: Optional[ServerMode] = None) -> None:
    with _status_lock:
        _last_status["state"] = state
        _last_status["message"] = message
        if server_mode is not None:
            _last_status["server_mode"] = server_mode.value
        else:
            _last_status["server_mode"] = _get_display_mode().value


# ============================================================
# xArm client logging
# ============================================================

def _append_xarm_log(line: str) -> None:
    line = line.rstrip("\n")
    with _xarm_lock:
        _xarm_log_lines.append(line)
        if len(_xarm_log_lines) > _MAX_XARM_LOG_LINES:
            del _xarm_log_lines[:-_MAX_XARM_LOG_LINES]
    print(f"[xarm-client-log] {line}")


def _xarm_log_reader(proc: subprocess.Popen) -> None:
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            _append_xarm_log(line)
    except Exception as e:
        _append_xarm_log(f"[dashboard] Error reading xArm client output: {e}")
    finally:
        rc = proc.wait()
        _append_xarm_log(f"[dashboard] xArm client exited with return code {rc}")


# ============================================================
# Local policy server logging
# ============================================================

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

        # If local was active, clear active mode so UI returns to preferred mode
        if _get_active_mode() == ServerMode.LOCAL:
            _set_active_mode(None)

        # Update status with a visible error
        if rc != 0:
            _set_status(
                "ERROR_LOCAL_EXIT",
                f"Local policy server exited (rc={rc}). See Policy Server Log.",
                server_mode=ServerMode.LOCAL,
            )


def _local_server_running() -> bool:
    with _local_server_lock:
        return _local_server_process is not None and _local_server_process.poll() is None


def _any_server_running_or_busy() -> bool:
    with _orch_thread_lock:
        hpc_thread_alive = _orch_thread is not None and _orch_thread.is_alive()

    orch = _orchestrator
    hpc_ready = orch is not None and orch.state == OrchestratorState.READY
    local_ready = _local_server_running()

    return hpc_thread_alive or hpc_ready or local_ready


# ============================================================
# Health checking
# ============================================================

def _ws_health_once(port: int, timeout_s: float = 3.0) -> tuple[bool, str]:
    uri = f"ws://localhost:{port}"
    try:
        ws = websockets.sync.client.connect(
            uri,
            open_timeout=timeout_s,
            close_timeout=timeout_s,
        )
        ws.close()
        return True, "WebSocket handshake succeeded."
    except Exception as e:
        return False, str(e)


def _wait_for_local_server_healthy(proc: subprocess.Popen, timeout_s: float = 180.0, poll_s: float = 2.0) -> None:
    """
    Wait until ws://localhost:XARM_PORT accepts a websocket handshake.
    If the process dies during startup, capture output and raise RuntimeError.
    """
    deadline = time.time() + timeout_s
    port = int(XARM_PORT)

    while time.time() < deadline:
        # If process died, capture whatever output remains
        ret = proc.poll()
        if ret is not None:
            try:
                out, _ = proc.communicate(timeout=1.0)
            except Exception:
                out = ""
            if out:
                for line in out.splitlines():
                    _append_local_server_log(line)
            _set_local_server_returncode(ret)
            raise RuntimeError(
                f"Local policy server exited during startup (rc={ret}).\n\n{out or '<no output>'}"
            )

        ok, msg = _ws_health_once(port, timeout_s=2.0)
        if ok:
            return

        _append_local_server_log(f"[dashboard] Waiting for local server health... ({msg})")
        time.sleep(poll_s)

    raise RuntimeError(f"Timed out waiting for local server to become healthy (>{timeout_s}s).")


# ============================================================
# Orchestrator callback
# ============================================================

def _on_state_change(state: OrchestratorState, message: str) -> None:
    # Orchestrator = HPC, so keep display mode consistent
    _set_status(state.value, message, server_mode=ServerMode.HPC)
    print(f"[DASHBOARD][HPC] {state.value}: {message}")


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
# Start local policy server (with logs + fail-fast)
# ============================================================

def _start_local_policy_server() -> None:
    global _local_server_process

    with _local_server_lock:
        if _local_server_process is not None and _local_server_process.poll() is None:
            raise RuntimeError("Local policy server is already running.")

    # Clear old logs
    with _local_server_log_lock:
        _local_server_log_lines.clear()
        _local_server_returncode = None

    _set_status("STARTING_LOCAL", "Starting local policy server...", server_mode=ServerMode.LOCAL)

    cmd = [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--env",
        "DEMO",
        "--port",
        XARM_PORT,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(OPENPI_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start local policy server: {e}") from e

    with _local_server_lock:
        _local_server_process = proc

    _append_local_server_log("[dashboard] Launched local policy server process.")
    _append_local_server_log(f"[dashboard] CWD: {OPENPI_ROOT}")
    _append_local_server_log(f"[dashboard] CMD: {' '.join(cmd)}")

    # Start log reader immediately
    threading.Thread(target=_local_server_log_reader, args=(proc,), daemon=True).start()

    # Wait for it to become healthy OR die (and capture its traceback)
    try:
        _set_status("CHECKING_LOCAL_HEALTH", "Waiting for local server to become healthy...", server_mode=ServerMode.LOCAL)
        _wait_for_local_server_healthy(proc, timeout_s=240.0, poll_s=2.0)
    except RuntimeError as e:
        # Make error visible in UI
        _set_status("ERROR_LOCAL_START", f"Local server failed to start.\n\n{e}", server_mode=ServerMode.LOCAL)
        # Active mode should clear (server isn't running)
        _set_active_mode(None)
        raise

    _set_status(OrchestratorState.READY.value, f"Local policy server healthy on ws://localhost:{XARM_PORT}", server_mode=ServerMode.LOCAL)


# ============================================================
# Camera backends
# ============================================================

base_camera = ZmqCameraBackend(host="127.0.0.1", port=5000, img_size=None, name="base", target_fps=15.0)
wrist_camera = ZmqCameraBackend(host="127.0.0.1", port=5001, img_size=None, name="wrist", target_fps=15.0)

# ============================================================
# HTML Frontend
# ============================================================

_INDEX_PATH = Path(__file__).with_name("index.html")
INDEX_HTML = _INDEX_PATH.read_text(encoding="utf-8")

# ============================================================
# Request models
# ============================================================

class RunXarmRequest(BaseModel):
    prompt: Optional[str] = None


class RunServerRequest(BaseModel):
    mode: ServerMode = ServerMode.LOCAL  # default local now


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
    """
    Persist the user's selected mode in the backend (while idle).
    If something is running/busy, we reject to avoid mislabeling.
    """
    if _any_server_running_or_busy():
        raise HTTPException(status_code=409, detail="Cannot change mode while a server is running or starting.")

    _set_preferred_mode(req.mode)
    _set_status(OrchestratorState.IDLE.value, f"Mode set to {req.mode.value.upper()}.", server_mode=req.mode)
    return {"status": "ok", "preferred_mode": req.mode.value}


@app.get("/api/status", response_class=JSONResponse)
def api_status() -> Dict[str, Any]:
    with _status_lock:
        state = _last_status["state"]
        message = _last_status["message"]

    # Determine running/busy
    with _orch_thread_lock:
        hpc_thread_alive = _orch_thread is not None and _orch_thread.is_alive()

    orch = _orchestrator
    hpc_ready = orch is not None and orch.state == OrchestratorState.READY
    local_ready = _local_server_running()

    # If local was active but died, clear active
    if _get_active_mode() == ServerMode.LOCAL and not local_ready:
        _set_active_mode(None)

    # For display, report active if running/busy else preferred
    display_mode = _get_display_mode().value
    server_running = bool(hpc_ready or local_ready)
    busy = bool(hpc_thread_alive or (state not in ["IDLE", "READY"] and not state.startswith("ERROR") and not server_running))

    # Job/tunnel info only meaningful in HPC mode
    job_id = orch.job_id if orch is not None else None
    remote_ip = orch.remote_ip if orch is not None else None
    remote_port = orch.remote_port if orch is not None else None
    if display_mode == ServerMode.LOCAL.value:
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
        "server_mode": display_mode,  # what UI should show/reflect
        "preferred_mode": _get_preferred_mode().value,
        "active_mode": (_get_active_mode().value if _get_active_mode() is not None else None),
        "hpc_running": bool(hpc_ready),
        "local_running": bool(local_ready),
    }


@app.get("/api/policy-server-log", response_class=JSONResponse)
def api_policy_server_log() -> Dict[str, Any]:
    """
    Return local policy server stdout/stderr log (rolling).
    For HPC mode, we don't have a live stream here; UI will still show orchestrator message.
    """
    with _local_server_log_lock:
        log = "\n".join(_local_server_log_lines)
        rc = _local_server_returncode

    running = _local_server_running()
    return {
        "mode": _get_display_mode().value,
        "local_running": running,
        "local_returncode": rc,
        "log": log,
    }


@app.post("/api/run-server", response_class=JSONResponse)
def api_run_server(req: RunServerRequest) -> Dict[str, Any]:
    if _any_server_running_or_busy():
        raise HTTPException(status_code=409, detail="Policy server already running (HPC or local).")

    # Persist preferred mode to whatever user is launching
    _set_preferred_mode(req.mode)

    if req.mode == ServerMode.HPC:
        _set_active_mode(ServerMode.HPC)
        _set_status(OrchestratorState.SUBMITTING_JOB.value, "Submitting policy server job...", server_mode=ServerMode.HPC)

        started = _start_orchestrator_thread()
        if not started:
            _set_active_mode(None)
            raise HTTPException(status_code=409, detail="Server orchestration already running or server already READY.")

        return {"status": "started", "mode": "hpc"}

    # LOCAL
    _set_active_mode(ServerMode.LOCAL)
    try:
        _start_local_policy_server()
    except RuntimeError as e:
        # /api/start-local already set ERROR_LOCAL_START with details
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "started", "mode": "local"}


@app.get("/api/health", response_class=JSONResponse)
def api_health() -> Dict[str, Any]:
    """
    Health check:
    - HPC: orchestrator websocket check
    - LOCAL: websocket handshake check on localhost
    """
    mode = _get_display_mode()

    if mode == ServerMode.LOCAL:
        ok, msg = _ws_health_once(int(XARM_PORT), timeout_s=3.0)
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
                cwd=str(REPO_ROOT),
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
            "localhost",
            "--remote_port",
            XARM_PORT,
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
