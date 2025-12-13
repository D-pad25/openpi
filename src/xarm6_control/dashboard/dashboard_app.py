#!/usr/bin/env python3
"""
dashboard_app.py

Web dashboard for:
- Starting the XArm policy server via HPC orchestrator
- Showing orchestrator state live
- Displaying two camera feeds (base + wrist) via MJPEG
- Running the local xArm client with a chosen prompt

Run with:
    uv run src/xarm6_control/dashboard/dashboard_app.py

Then open:
    http://localhost:9000
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Dict, Any, List
from enum import Enum

import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
)
from pydantic import BaseModel

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

OPENPI_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_XARM_PROMPT = (
    "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
)
XARM_PORT = "8000"


class ServerMode(str, Enum):
    """Where the policy server is running."""
    HPC = "hpc"
    LOCAL = "local"


# ============================================================
# Global orchestrator + thread + status storage
# ============================================================

app = FastAPI(title="XArm6 Demo Dashboard")

_status_lock = threading.Lock()

# Default to LOCAL (your desired main mode)
_last_status: Dict[str, Any] = {
    "state": OrchestratorState.IDLE.value,
    "message": "Idle",
    "server_mode": ServerMode.LOCAL.value,
}

# Persisted selected mode (changes via /api/set-server-mode)
_server_mode: ServerMode = ServerMode.LOCAL

_camera_server_lock = threading.Lock()
_camera_server_process: Optional[subprocess.Popen] = None

_orchestrator: Optional[Orchestrator] = None
_orch_thread: Optional[threading.Thread] = None
_orch_thread_lock = threading.Lock()

_local_server_lock = threading.Lock()
_local_server_process: Optional[subprocess.Popen] = None

# ============================================================
# xArm client process + log
# ============================================================

_xarm_lock = threading.Lock()
_xarm_process: Optional[subprocess.Popen] = None
_xarm_log_lines: List[str] = []
_MAX_XARM_LOG_LINES = 200


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


class RunXarmRequest(BaseModel):
    prompt: Optional[str] = None


class RunServerRequest(BaseModel):
    mode: ServerMode = ServerMode.LOCAL


class SetModeRequest(BaseModel):
    mode: ServerMode


# ============================================================
# Orchestrator + server helpers
# ============================================================

def _on_state_change(state: OrchestratorState, message: str) -> None:
    global _server_mode
    with _status_lock:
        _last_status["state"] = state.value
        _last_status["message"] = message
        _last_status["server_mode"] = _server_mode.value
    print(f"[DASHBOARD][{_server_mode.value.upper()}] {state.value}: {message}")


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


def _local_server_running() -> bool:
    with _local_server_lock:
        return _local_server_process is not None and _local_server_process.poll() is None


def _any_server_running() -> bool:
    with _orch_thread_lock:
        hpc_thread_alive = _orch_thread is not None and _orch_thread.is_alive()

    orch = _orchestrator
    hpc_ready = orch is not None and orch.state == OrchestratorState.READY
    local_ready = _local_server_running()

    return hpc_thread_alive or hpc_ready or local_ready


def _start_local_policy_server() -> None:
    global _local_server_process

    with _local_server_lock:
        if _local_server_process is not None and _local_server_process.poll() is None:
            raise RuntimeError("Local policy server is already running.")

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
            proc = subprocess.Popen(cmd, cwd=str(OPENPI_ROOT))
        except Exception as e:
            raise RuntimeError(f"Failed to start local policy server: {e}") from e

        _local_server_process = proc

    with _status_lock:
        _last_status["state"] = OrchestratorState.READY.value
        _last_status["message"] = f"Local policy server running on localhost:{XARM_PORT}"
        _last_status["server_mode"] = ServerMode.LOCAL.value


# ============================================================
# Cameras
# ============================================================

base_camera = ZmqCameraBackend(
    host="127.0.0.1",
    port=5000,
    img_size=None,
    name="base",
    target_fps=15.0,
)

wrist_camera = ZmqCameraBackend(
    host="127.0.0.1",
    port=5001,
    img_size=None,
    name="wrist",
    target_fps=15.0,
)

# ============================================================
# HTML Frontend
# ============================================================

_INDEX_PATH = Path(__file__).with_name("index.html")
INDEX_HTML = _INDEX_PATH.read_text(encoding="utf-8")

# ============================================================
# FastAPI routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/status", response_class=JSONResponse)
def api_status() -> Dict[str, Any]:
    with _status_lock:
        state = _last_status["state"]
        message = _last_status["message"]
        server_mode = _last_status.get("server_mode", _server_mode.value)

    orch = _orchestrator
    job_id = orch.job_id if orch is not None else None
    remote_ip = orch.remote_ip if orch is not None else None
    remote_port = orch.remote_port if orch is not None else None

    if server_mode == ServerMode.LOCAL.value:
        job_id = None
        remote_ip = None
        remote_port = None

    hpc_ready = orch is not None and orch.state == OrchestratorState.READY
    local_ready = _local_server_running()

    if (
        server_mode == ServerMode.LOCAL.value
        and state == OrchestratorState.READY.value
        and not local_ready
    ):
        with _status_lock:
            _last_status["state"] = OrchestratorState.IDLE.value
            _last_status["message"] = "Local policy server is not running."
            # keep server_mode as LOCAL
            state = _last_status["state"]
            message = _last_status["message"]

    server_running = hpc_ready or local_ready

    return {
        "state": state,
        "message": message,
        "job_id": job_id,
        "remote_ip": remote_ip,
        "remote_port": remote_port,
        "server_running": server_running,
        "server_mode": server_mode,
        "hpc_running": hpc_ready,
        "local_running": local_ready,
    }


@app.post("/api/set-server-mode", response_class=JSONResponse)
def api_set_server_mode(req: SetModeRequest) -> Dict[str, Any]:
    """
    Persist the user's selected mode on the backend.
    Only allowed when nothing is running/busy.
    """
    global _server_mode

    if _any_server_running():
        raise HTTPException(
            status_code=409,
            detail="Cannot change mode while a policy server is running/busy.",
        )

    _server_mode = req.mode
    with _status_lock:
        _last_status["server_mode"] = req.mode.value
        if _last_status["state"] == OrchestratorState.IDLE.value:
            _last_status["message"] = f"Mode set to {req.mode.value.upper()} (idle)."

    return {"status": "ok", "mode": req.mode.value}


@app.post("/api/run-server", response_class=JSONResponse)
def api_run_server(req: RunServerRequest) -> Dict[str, Any]:
    global _server_mode

    if _any_server_running():
        raise HTTPException(
            status_code=409,
            detail="Policy server already running (HPC or local).",
        )

    # Ensure backend mode matches what caller requested
    _server_mode = req.mode
    with _status_lock:
        _last_status["server_mode"] = req.mode.value

    if req.mode == ServerMode.HPC:
        started = _start_orchestrator_thread()
        if not started:
            raise HTTPException(
                status_code=409,
                detail="Server orchestration already running or server already READY.",
            )

        with _status_lock:
            _last_status["state"] = OrchestratorState.SUBMITTING_JOB.value
            _last_status["message"] = "Submitting policy server job..."
            _last_status["server_mode"] = ServerMode.HPC.value

        return {"status": "started", "mode": "hpc"}

    # LOCAL
    try:
        _start_local_policy_server()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "started", "mode": "local"}


@app.get("/api/health", response_class=JSONResponse)
def api_health() -> Dict[str, Any]:
    orch = _orchestrator
    if orch is None:
        return {"status": "unknown", "detail": "Orchestrator has not run yet."}

    ok = orch.check_policy_health()
    if ok:
        return {"status": "healthy", "detail": orch.last_message}
    else:
        return {"status": "unhealthy", "detail": orch.last_message}


@app.post("/api/start-cameras", response_class=JSONResponse)
def api_start_cameras() -> Dict[str, Any]:
    global _camera_server_process

    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline script not found at {PIPELINE_SCRIPT}",
        )

    with _camera_server_lock:
        if _camera_server_process is not None and _camera_server_process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="Camera server already running.",
            )

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
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start camera server: {e}",
            )

        time.sleep(3.0)
        ret = proc.poll()
        if ret is not None and ret != 0:
            try:
                output, _ = proc.communicate(timeout=1.0)
            except Exception:
                output = "<no output captured>"

            raise HTTPException(
                status_code=500,
                detail=f"Camera server exited early with code {ret}:\n{output}",
            )

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
        server_running = (
            _camera_server_process is not None
            and _camera_server_process.poll() is None
        )

    return {
        "server_running": server_running,
        "base": base_camera.get_status(),
        "wrist": wrist_camera.get_status(),
    }


@app.post("/api/run-xarm", response_class=JSONResponse)
def api_run_xarm(req: RunXarmRequest) -> Dict[str, Any]:
    global _xarm_process

    with _status_lock:
        state = _last_status["state"]

    if state != OrchestratorState.READY.value:
        raise HTTPException(
            status_code=400,
            detail=f"Policy server not READY (current state: {state}). Start it first.",
        )

    with _xarm_lock:
        if _xarm_process is not None and _xarm_process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="xArm client is already running.",
            )

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
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start xArm client: {e}",
            )

        _xarm_process = proc
        _xarm_log_lines.clear()
        _append_xarm_log("[dashboard] Starting xArm client...")
        _append_xarm_log(f"[dashboard] Prompt: {prompt}")

        t = threading.Thread(target=_xarm_log_reader, args=(proc,), daemon=True)
        t.start()

    return {"status": "started"}


@app.post("/api/stop-xarm", response_class=JSONResponse)
def api_stop_xarm() -> Dict[str, Any]:
    global _xarm_process
    with _xarm_lock:
        proc = _xarm_process

    if proc is None or proc.poll() is not None:
        raise HTTPException(
            status_code=400,
            detail="xArm client is not running.",
        )

    try:
        proc.terminate()
        _append_xarm_log("[dashboard] Sent terminate() to xArm client.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop xArm client: {e}",
        )

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

    return {
        "running": running,
        "returncode": returncode,
        "log": log,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "xarm6_control.dashboard.dashboard_app:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )
