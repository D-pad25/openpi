#!/usr/bin/env python3
"""
dashboard_app.py

Web dashboard for:
- Starting the XArm policy server via HPC orchestrator
- Showing orchestrator state live
- Displaying two camera feeds (base + wrist) via MJPEG

Run with:
    uv run src/xarm6_control/dashboard_app.py

Then open:
    http://localhost:9000
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Generator

import cv2  # OpenCV for camera capture (pip install opencv-python)

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
)

# Import your existing orchestrator
from xarm6_control.xarm_server_orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorState,
)

# ============================================================
# Global orchestrator + thread management
# ============================================================

app = FastAPI(title="XArm6 Demo Dashboard")

_orch_lock = threading.Lock()
_orchestrator: Optional[Orchestrator] = None
_orchestrator_thread: Optional[threading.Thread] = None


def _get_orchestrator() -> Optional[Orchestrator]:
    global _orchestrator
    return _orchestrator


def _start_orchestrator_thread() -> None:
    """
    Create a new Orchestrator and run its full sequence in a background thread.
    If one is already running, this function does nothing.
    """
    global _orchestrator, _orchestrator_thread

    with _orch_lock:
        if _orchestrator_thread is not None and _orchestrator_thread.is_alive():
            # Already running
            return

        # Fresh orchestrator instance
        config = OrchestratorConfig()
        orch = Orchestrator(config=config)
        _orchestrator = orch

        def _run():
            try:
                orch.run_full_sequence()
            finally:
                # When done (success or failure), we keep the orchestrator
                # object around so the UI can still read the last state/message.
                pass

        t = threading.Thread(target=_run, daemon=True)
        _orchestrator_thread = t
        t.start()


# ============================================================
# Simple camera streaming via MJPEG (demo-style)
# ============================================================

class CameraStream:
    """
    Very simple wrapper around OpenCV VideoCapture to serve frames as MJPEG.

    For real deployment with RealSense, you can:
    - Replace VideoCapture with pyrealsense2 pipeline, OR
    - Subscribe to your ZMQ camera publisher and decode frames here.
    """

    def __init__(self, device_index: int, name: str = "camera") -> None:
        self.device_index = device_index
        self.name = name
        self._cap = None
        self._lock = threading.Lock()

    def _ensure_open(self) -> None:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.device_index)
                # Optionally set resolution / fps here:
                # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def frames(self) -> Generator[bytes, None, None]:
        self._ensure_open()
        if self._cap is None or not self._cap.isOpened():
            # Yield a single black frame or nothing
            blank = self._black_frame()
            yield blank
            return

        while True:
            with self._lock:
                ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Encode frame as JPEG
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            data = jpeg.tobytes()
            yield data

    @staticmethod
    def _black_frame(width: int = 640, height: int = 480) -> bytes:
        """Return a black JPEG frame if camera is not available."""
        import numpy as np

        img = np.zeros((height, width, 3), dtype=np.uint8)
        ret, jpeg = cv2.imencode(".jpg", img)
        if not ret:
            return b""
        return jpeg.tobytes()


# Two logical cameras – adjust indices to match your system
base_camera = CameraStream(device_index=0, name="base")
wrist_camera = CameraStream(device_index=1, name="wrist")


def mjpeg_stream(camera: CameraStream) -> Generator[bytes, None, None]:
    """
    Generator for FastAPI StreamingResponse in multipart/x-mixed-replace format.
    """
    boundary = b"--frame"
    for frame_bytes in camera.frames():
        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# HTML Frontend
# ============================================================

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>XArm6 Demo Dashboard</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      padding: 16px 24px;
      background: #020617;
      border-bottom: 1px solid #1f2937;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      margin: 0;
      font-size: 1.3rem;
    }
    main {
      flex: 1;
      padding: 16px 24px 24px;
      display: grid;
      grid-template-columns: 2fr 1.2fr;
      grid-gap: 16px;
    }
    .card {
      background: #020617;
      border-radius: 12px;
      padding: 16px;
      border: 1px solid #1f2937;
      box-shadow: 0 10px 25px rgba(0,0,0,0.45);
    }
    .card h2 {
      margin-top: 0;
      font-size: 1.1rem;
      margin-bottom: 8px;
    }
    .cams-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-gap: 8px;
    }
    .cam-box {
      border-radius: 10px;
      overflow: hidden;
      background: #020617;
      border: 1px solid #1f2937;
      display: flex;
      flex-direction: column;
    }
    .cam-box-header {
      padding: 6px 10px;
      font-size: 0.85rem;
      border-bottom: 1px solid #1f2937;
      background: #050816;
    }
    .cam-box img {
      width: 100%;
      height: auto;
      display: block;
      background: #111827;
    }
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      border: 1px solid #1f2937;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
    }
    .status-idle   { background: #4b5563; }
    .status-busy   { background: #fbbf24; }
    .status-ready  { background: #22c55e; }
    .status-error  { background: #ef4444; }

    button {
      border-radius: 999px;
      padding: 8px 16px;
      border: none;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    button.primary {
      background: linear-gradient(to right, #22c55e, #14b8a6);
      color: #020617;
    }
    button.secondary {
      background: #020617;
      border: 1px solid #374151;
      color: #e5e7eb;
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .log {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      background: #020617;
      border-radius: 8px;
      border: 1px solid #1f2937;
      padding: 8px;
      font-size: 0.78rem;
      max-height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
    .meta-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0;
      font-size: 0.8rem;
      color: #9ca3af;
    }
    .meta-row span {
      border-radius: 999px;
      padding: 3px 10px;
      background: #020617;
      border: 1px solid #111827;
    }
    footer {
      padding: 8px 16px;
      font-size: 0.75rem;
      color: #6b7280;
      background: #020617;
      border-top: 1px solid #111827;
      text-align: right;
    }
  </style>
</head>
<body>
  <header>
    <h1>XArm6 Policy Server Dashboard</h1>
    <div>
      <button id="run-btn" class="primary">▶ Run Policy Server</button>
      <button id="health-btn" class="secondary" style="margin-left: 8px;">❤️ Check Health</button>
    </div>
  </header>

  <main>
    <section class="card">
      <h2>Camera Feeds</h2>
      <div class="cams-grid">
        <div class="cam-box">
          <div class="cam-box-header">Base Camera</div>
          <img src="/video/base" alt="Base camera stream" />
        </div>
        <div class="cam-box">
          <div class="cam-box-header">Wrist Camera</div>
          <img src="/video/wrist" alt="Wrist camera stream" />
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Server & Tunnel Status</h2>
      <div id="status-chip" class="status-chip">
        <div id="status-dot" class="status-dot status-idle"></div>
        <span id="status-text">IDLE</span>
      </div>

      <div class="meta-row">
        <span id="meta-job">Job: —</span>
        <span id="meta-remote">Remote: —</span>
      </div>

      <div style="margin-top: 8px; font-size: 0.8rem; color: #9ca3af;">
        <strong>Last message:</strong>
      </div>
      <div id="last-message" class="log"></div>
    </section>
  </main>

  <footer>
    XArm6 demo • Orchestrator-backed • MJPEG cameras
  </footer>

  <script>
    const statusChip = document.getElementById("status-chip");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const metaJob = document.getElementById("meta-job");
    const metaRemote = document.getElementById("meta-remote");
    const lastMessage = document.getElementById("last-message");
    const runBtn = document.getElementById("run-btn");
    const healthBtn = document.getElementById("health-btn");

    function updateStatusView(data) {
      const state = data.state || "IDLE";
      const msg = data.message || "";
      const jobId = data.job_id || "—";
      const remote = (data.remote_ip && data.remote_port)
        ? `${data.remote_ip}:${data.remote_port}`
        : "—";

      statusText.textContent = state;
      lastMessage.textContent = msg;
      metaJob.textContent = `Job: ${jobId}`;
      metaRemote.textContent = `Remote: ${remote}`;

      statusDot.className = "status-dot";

      const s = state.toUpperCase();
      if (s === "READY") {
        statusDot.classList.add("status-ready");
      } else if (s.startsWith("ERROR")) {
        statusDot.classList.add("status-error");
      } else if (s === "IDLE") {
        statusDot.classList.add("status-idle");
      } else {
        statusDot.classList.add("status-busy");
      }
    }

    async function fetchStatus() {
      try {
        const resp = await fetch("/api/status");
        if (!resp.ok) return;
        const data = await resp.json();
        updateStatusView(data);
      } catch (e) {
        console.error("Status poll failed:", e);
      }
    }

    async function runServer() {
      runBtn.disabled = true;
      runBtn.textContent = "Running...";
      try {
        const resp = await fetch("/api/start_server", { method: "POST" });
        if (!resp.ok) {
          const text = await resp.text();
          alert("Failed to start server: " + text);
        }
      } catch (e) {
        alert("Error starting server: " + e);
      } finally {
        setTimeout(() => {
          runBtn.disabled = false;
          runBtn.textContent = "▶ Run Policy Server";
        }, 2000);
      }
    }

    async function runHealthCheck() {
      try {
        const resp = await fetch("/api/health");
        const data = await resp.json();
        alert("Health check: " + data.status + (data.detail ? ("\\n" + data.detail) : ""));
      } catch (e) {
        alert("Health check failed: " + e);
      }
    }

    runBtn.addEventListener("click", runServer);
    healthBtn.addEventListener("click", runHealthCheck);

    // Poll status every 2 seconds
    fetchStatus();
    setInterval(fetchStatus, 2000);
  </script>
</body>
</html>
"""


# ============================================================
# FastAPI routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/status", response_class=JSONResponse)
def api_status() -> Dict[str, Any]:
    orch = _get_orchestrator()
    if orch is None:
        return {
            "state": "IDLE",
            "message": "No orchestrator has run yet.",
            "job_id": None,
            "remote_ip": None,
            "remote_port": None,
        }

    return {
        "state": orch.state.value,
        "message": orch.last_message,
        "job_id": orch.job_id,
        "remote_ip": orch.remote_ip,
        "remote_port": orch.remote_port,
    }


@app.post("/api/start_server", response_class=JSONResponse)
def api_start_server() -> Dict[str, Any]:
    with _orch_lock:
        global _orchestrator_thread

        if _orchestrator_thread is not None and _orchestrator_thread.is_alive():
            raise HTTPException(
                status_code=409,
                detail="Orchestrator is already running.",
            )

        _start_orchestrator_thread()

    return {"status": "started"}


@app.get("/api/health", response_class=JSONResponse)
def api_health() -> Dict[str, Any]:
    orch = _get_orchestrator()
    if orch is None:
        return {"status": "unknown", "detail": "Orchestrator has not run yet."}

    ok = orch.check_policy_health()
    if ok:
        return {"status": "healthy", "detail": orch.last_message}
    else:
        return {"status": "unhealthy", "detail": orch.last_message}


@app.get("/video/base")
def video_base() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(base_camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/wrist")
def video_wrist() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream(wrist_camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn

    # Listen only on localhost; adjust if you want LAN access
    uvicorn.run(
        "xarm6_control.dashboard_app:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )
