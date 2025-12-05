#!/usr/bin/env python3
"""
dashboard_app.py

Simple web dashboard for the xArm + HPC policy server orchestration.

- Serves a minimal HTML page with a "Run server" button and status display.
- When the button is pressed, it calls /api/run-server which starts the
  Orchestrator in a background task.
- The frontend polls /api/status every few seconds to show live state
  updates from the orchestrator.
"""

from __future__ import annotations

import threading
from typing import Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from xarm6_control.xarm_server_orchestrator import (
    Orchestrator,
    OrchestratorState,
)

app = FastAPI()

# Allow browser access from same machine (and others if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator + status store
orch: Optional[Orchestrator] = None
status_lock = threading.Lock()
last_status: Dict[str, Any] = {
    "state": OrchestratorState.IDLE.value,
    "message": "Idle",
}


def _on_state_change(state: OrchestratorState, message: str) -> None:
    """Callback from Orchestrator to update global status."""
    with status_lock:
        last_status["state"] = state.value
        last_status["message"] = message
    # Also print to server logs for debugging
    print(f"[DASHBOARD] {state.value}: {message}")


def _start_orchestrator_sequence():
    """Run the full sequence in a background thread."""
    global orch
    orch = Orchestrator()
    orch.on_state_change = _on_state_change
    orch.run_full_sequence()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve a simple HTML dashboard."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XArm Policy Server Dashboard</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 20px;
      background: #111827;
      color: #e5e7eb;
    }
    .card {
      max-width: 700px;
      margin: 0 auto;
      padding: 24px;
      border-radius: 16px;
      background: #1f2937;
      box-shadow: 0 15px 25px rgba(0,0,0,0.4);
    }
    h1 {
      margin-top: 0;
      font-size: 1.6rem;
    }
    button {
      padding: 10px 20px;
      font-size: 1rem;
      border: none;
      border-radius: 999px;
      cursor: pointer;
      background: #4f46e5;
      color: white;
      font-weight: 600;
    }
    button:disabled {
      background: #4b5563;
      cursor: not-allowed;
    }
    .status {
      margin-top: 16px;
      padding: 12px 16px;
      border-radius: 12px;
      background: #111827;
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    .status-label {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #9ca3af;
    }
    .status-state {
      font-size: 1rem;
      font-weight: 600;
      margin-top: 4px;
    }
    .status-message {
      font-size: 0.9rem;
      margin-top: 4px;
      color: #d1d5db;
      white-space: pre-wrap;
    }
    .status-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>XArm + HPC Policy Server</h1>
    <p>Use this panel to start the policy server on the HPC and establish the SSH tunnel.</p>

    <button id="runBtn" onclick="runServer()">Run server</button>

    <div class="status" id="statusBox">
      <div class="status-label">Current status</div>
      <div class="status-state" id="statusState">
        <span class="status-dot" id="statusDot" style="background:#6b7280;"></span>
        IDLE
      </div>
      <div class="status-message" id="statusMessage">Idle</div>
    </div>
  </div>

<script>
const statusState   = document.getElementById("statusState");
const statusMessage = document.getElementById("statusMessage");
const statusDot     = document.getElementById("statusDot");
const runBtn        = document.getElementById("runBtn");

function colourForState(state) {
  if (state === "READY") return "#22c55e";               // green
  if (state.startsWith("ERROR")) return "#ef4444";       // red
  if (state === "SUBMITTING_JOB" ||
      state === "JOB_QUEUED" ||
      state === "JOB_RUNNING_STARTING_SERVER" ||
      state === "SERVER_READY_NO_TUNNEL" ||
      state === "STARTING_TUNNEL" ||
      state === "CHECKING_POLICY_HEALTH" ||
      state === "CLEARING_OLD_ENDPOINT") return "#facc15"; // yellow
  return "#6b7280";                                      // grey
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    const state = data.state || "UNKNOWN";
    const message = data.message || "";

    statusState.innerHTML = '<span class="status-dot" id="statusDot"></span>' + state;
    const dot = document.getElementById("statusDot");
    dot.style.background = colourForState(state);
    statusMessage.textContent = message;

    // Disable button when in non-idle, non-error "busy" states
    if (state === "IDLE" || state === "READY" || state.startsWith("ERROR")) {
      runBtn.disabled = false;
    } else {
      runBtn.disabled = true;
    }
  } catch (e) {
    console.error("Error fetching status:", e);
  }
}

async function runServer() {
  runBtn.disabled = true;
  try {
    const res = await fetch("/api/run-server", { method: "POST" });
    const data = await res.json();
    // Optionally show a toast: data.message
    console.log("Run server:", data);
  } catch (e) {
    console.error("Failed to start server:", e);
    runBtn.disabled = false;
  }
}

// Poll status every 3 seconds
setInterval(fetchStatus, 3000);
// Also fetch immediately on load
fetchStatus();
</script>
</body>
</html>
"""


@app.post("/api/run-server")
async def api_run_server(background_tasks: BackgroundTasks) -> JSONResponse:
    """
    Start the orchestrator in the background.

    Returns immediately so the browser doesn't hang while the job queues,
    checkpoint loads, etc.
    """
    # Kick off orchestrator in a background task
    background_tasks.add_task(_start_orchestrator_sequence)

    with status_lock:
        # Mark as "starting"
        last_status["state"] = OrchestratorState.SUBMITTING_JOB.value
        last_status["message"] = "Submitting policy server job..."

    return JSONResponse({"ok": True, "message": "Orchestration started"})


@app.get("/api/status")
async def api_status() -> JSONResponse:
    """Return the latest orchestrator state/message."""
    with status_lock:
        return JSONResponse(
            {"state": last_status["state"], "message": last_status["message"]}
        )


# Allow running via: uv run src/xarm6_control/dashboard_app.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
