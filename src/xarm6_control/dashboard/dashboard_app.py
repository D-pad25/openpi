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
from typing import Optional, Dict, Any, Generator, List

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

# Folder where this dashboard + xarm_pipeline.sh live
REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = REPO_ROOT / "xarm_pipeline.sh"

# Top-level openpi repo root (where src/ lives)
# e.g. .../Thesis/openpi
OPENPI_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_XARM_PROMPT = (
    "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
)
XARM_PORT = "8000"

# ============================================================
# Global orchestrator + thread + status storage
# ============================================================

app = FastAPI(title="XArm6 Demo Dashboard")

# Global status storage
_status_lock = threading.Lock()
_last_status: Dict[str, Any] = {
    "state": OrchestratorState.IDLE.value,
    "message": "Idle",
}

# Global camera server process (if any)
_camera_server_lock = threading.Lock()
_camera_server_process: Optional[subprocess.Popen] = None

# Orchestrator + thread
_orchestrator: Optional[Orchestrator] = None
_orch_thread: Optional[threading.Thread] = None
_orch_thread_lock = threading.Lock()

# ============================================================
# xArm client process + log
# ============================================================

_xarm_lock = threading.Lock()
_xarm_process: Optional[subprocess.Popen] = None
_xarm_log_lines: List[str] = []
_MAX_XARM_LOG_LINES = 200


def _append_xarm_log(line: str) -> None:
    """Append a line to the in-memory xArm client log."""
    line = line.rstrip("\n")
    with _xarm_lock:
        _xarm_log_lines.append(line)
        if len(_xarm_log_lines) > _MAX_XARM_LOG_LINES:
            # keep only the last N lines
            del _xarm_log_lines[:-_MAX_XARM_LOG_LINES]
    print(f"[xarm-client-log] {line}")


def _xarm_log_reader(proc: subprocess.Popen) -> None:
    """Background thread: read stdout from xArm client and store in log."""
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


# ============================================================
# Orchestrator helpers
# ============================================================

def _on_state_change(state: OrchestratorState, message: str) -> None:
    """Callback from Orchestrator to update status for UI."""
    with _status_lock:
        _last_status["state"] = state.value
        _last_status["message"] = message
    print(f"[DASHBOARD] {state.value}: {message}")


def _start_orchestrator_thread() -> bool:
    """
    Start orchestrator.run_full_sequence() in a background thread.

    Returns:
        True  if a new thread was started.
        False if an orchestration is already in progress or server is READY.
    """
    global _orchestrator, _orch_thread

    with _orch_thread_lock:
        # If thread exists and is still alive, we consider orchestration "busy".
        if _orch_thread is not None and _orch_thread.is_alive():
            return False

        # If an orchestrator exists and is already READY, don't start again.
        if _orchestrator is not None and _orchestrator.state == OrchestratorState.READY:
            return False

        # Create fresh orchestrator and thread
        config = OrchestratorConfig()
        orch = Orchestrator(config=config)
        orch.on_state_change = _on_state_change
        _orchestrator = orch

        def _run():
            try:
                orch.run_full_sequence()
            finally:
                # Keep orchestrator object so UI can still display final state.
                pass

        t = threading.Thread(target=_run, daemon=True)
        _orch_thread = t
        t.start()
        return True


# ============================================================
# Create shared camera backends (one per camera)
# ============================================================

base_camera = ZmqCameraBackend(
    host="127.0.0.1",
    port=5000,
    img_size=None,   # or (640, 480) if you want to downsample
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

# Path to the HTML file (in the same folder as this script)
_INDEX_PATH = Path(__file__).with_name("index.html")

# Read once at import time (simple and fast)
INDEX_HTML = _INDEX_PATH.read_text(encoding="utf-8")

# ============================================================
# FastAPI routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/status", response_class=JSONResponse)
def api_status() -> Dict[str, Any]:
    """
    Return latest orchestrator state + message + job/remote info.
    Also expose a simple server_running flag for the UI.
    """
    with _status_lock:
        state = _last_status["state"]
        message = _last_status["message"]

    orch = _orchestrator
    job_id = orch.job_id if orch is not None else None
    remote_ip = orch.remote_ip if orch is not None else None
    remote_port = orch.remote_port if orch is not None else None

    server_running = state == OrchestratorState.READY.value

    return {
        "state": state,
        "message": message,
        "job_id": job_id,
        "remote_ip": remote_ip,
        "remote_port": remote_port,
        "server_running": server_running,
    }


@app.post("/api/run-server", response_class=JSONResponse)
def api_run_server() -> Dict[str, Any]:
    """
    Start the orchestrator sequence in a background thread.

    Prevents starting a second sequence if the server/job is already running/busy.
    """
    started = _start_orchestrator_thread()
    if not started:
        raise HTTPException(
            status_code=409,
            detail="Server orchestration already running or server already READY.",
        )

    with _status_lock:
        _last_status["state"] = OrchestratorState.SUBMITTING_JOB.value
        _last_status["message"] = "Submitting policy server job..."

    return {"status": "started"}


@app.get("/api/health", response_class=JSONResponse)
def api_health() -> Dict[str, Any]:
    """
    Trigger an explicit health check using the orchestrator's method.
    """
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
    """
    Start the local camera nodes via xarm_pipeline.sh cameras.

    If the script exits immediately with an error (e.g. no RealSense devices),
    capture its output and return it as an HTTP 500 so the UI can show it.
    """
    global _camera_server_process

    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline script not found at {PIPELINE_SCRIPT}",
        )

    with _camera_server_lock:
        # If process exists and is still running, don't start another
        if _camera_server_process is not None and _camera_server_process.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="Camera server already running.",
            )

        cmd = ["bash", str(PIPELINE_SCRIPT), "cameras"]

        try:
            # Capture stdout+stderr so we can surface errors if it dies quickly
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

        # Give the script a short window to either stay alive or fail fast
        time.sleep(3.0)
        ret = proc.poll()
        if ret is not None and ret != 0:
            # Process exited with error – grab its output for debugging
            try:
                output, _ = proc.communicate(timeout=1.0)
            except Exception:
                output = "<no output captured>"

            raise HTTPException(
                status_code=500,
                detail=f"Camera server exited early with code {ret}:\n{output}",
            )

        # Otherwise assume it's running in the background
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
    """
    Return status for each camera so the UI can show ONLINE / DISCONNECTED,
    plus whether the camera *server* process is running.
    """
    with _camera_server_lock:
        server_running = (
            _camera_server_process is not None
            and _camera_server_process.poll() is None  # None => still running
        )

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
    """
    Start the local xArm client (pi0 / VLA policy) with an optional prompt.

    Uses:
        uv run src/xarm6_control/main2.py --remote_host localhost --remote_port 8000 --prompt "<prompt>"
    """
    global _xarm_process

    # Ensure policy server is READY before running client
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

        prompt = (req.prompt or DEFAULT_XARM_PROMPT).strip()
        if not prompt:
            prompt = DEFAULT_XARM_PROMPT

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
    """
    Soft-stop the local xArm client process (terminate).
    """
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
    """
    Return current xArm client state and recent log output.
    """
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
