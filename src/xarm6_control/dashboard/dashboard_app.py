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
    * LOCAL: terminate local process group/session safely
    * HPC: request cancel; qdel once job_id is known; stop tunnel too

Run with:
    uv run src/xarm6_control/dashboard/dashboard_app.py

Then open:
    http://localhost:9000
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import websockets.sync.client
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import paramiko

from xarm6_control.dashboard.xarm_server_orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    OrchestratorState,
)
from xarm6_control.dashboard.zmq_camera_backend import (
    ZmqCameraBackend,
    mjpeg_stream_generator,
)

###############################################################################
# Paths / defaults
###############################################################################

HERE = Path(__file__).resolve().parent
PIPELINE_SCRIPT = HERE / "xarm_pipeline.sh"

DEFAULT_XARM_PROMPT = "Pick a ripe, red tomato and drop it in the blue bucket. [crop=tomato]"
XARM_PORT = 8000

MAX_LOCAL_SERVER_LOG_LINES = 600
MAX_XARM_LOG_LINES = 400


def _find_openpi_root() -> Path:
    """
    Find the repo root (OPENPI_ROOT) robustly, so subprocess cwd/env is correct even if
    the dashboard is launched from a different working directory.
    """
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        if (p / "scripts" / "serve_policy.py").exists() and (p / "src").exists():
            return p
    return Path(__file__).resolve().parents[3]


OPENPI_ROOT = _find_openpi_root()
INDEX_HTML = (HERE / "index.html").read_text(encoding="utf-8")

###############################################################################
# API models
###############################################################################


class ServerMode(str, Enum):
    HPC = "hpc"
    LOCAL = "local"


class RunXarmRequest(BaseModel):
    prompt: Optional[str] = None


class RunServerRequest(BaseModel):
    mode: ServerMode = ServerMode.LOCAL


class SetModeRequest(BaseModel):
    mode: ServerMode


class HpcConnectRequest(BaseModel):
    host: Optional[str] = None
    username: str
    password: str


@dataclass(frozen=True)
class StatusSnapshot:
    state: str
    message: str
    server_mode: str


###############################################################################
# Subprocess/log helpers
###############################################################################


def _make_child_env(openpi_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    src_path = str(openpi_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _popen_new_process_kwargs() -> Dict[str, Any]:
    """
    Start child processes in their own session/group when possible, so termination
    doesn't affect the dashboard.
    """
    if os.name != "nt":
        return {"start_new_session": True}

    # Best-effort Windows equivalent.
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"creationflags": creationflags} if creationflags else {}


def _terminate_process_best_effort(proc: subprocess.Popen, kill_after_s: float = 3.0) -> None:
    """
    Best-effort termination (POSIX: killpg; Windows: Ctrl-Break/terminate/kill).
    """
    if proc.poll() is not None:
        return

    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                try:
                    proc.send_signal(ctrl_break)  # type: ignore[arg-type]
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.time() + kill_after_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


class RingLog:
    """
    Thread-safe ring buffer for process logs (keeps last N lines).
    """

    def __init__(self, max_lines: int, print_prefix: str) -> None:
        self._lock = threading.Lock()
        self._lines: Deque[str] = deque(maxlen=max_lines)
        self._returncode: Optional[int] = None
        self._print_prefix = print_prefix

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._returncode = None

    def append(self, line: str) -> None:
        line = line.rstrip("\n")
        with self._lock:
            self._lines.append(line)
        print(f"[{self._print_prefix}] {line}")

    def set_returncode(self, rc: Optional[int]) -> None:
        with self._lock:
            self._returncode = rc

    def snapshot(self) -> Tuple[str, Optional[int]]:
        with self._lock:
            return ("\n".join(self._lines), self._returncode)


class ManagedProcess:
    """
    Owns a subprocess and its stdout capture thread.
    """

    def __init__(
        self,
        *,
        name: str,
        log: RingLog,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        self._name = name
        self._log = log
        self._on_exit = on_exit
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, cmd: list[str], *, cwd: str, env: Dict[str, str], clear_log: bool = True) -> subprocess.Popen:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(f"{self._name} is already running.")

            if clear_log:
                self._log.clear()

            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **_popen_new_process_kwargs(),
            )
            self._proc = proc

            threading.Thread(target=self._reader_loop, args=(proc,), daemon=True).start()
            return proc

    def stop(self) -> Tuple[bool, str]:
        with self._lock:
            proc = self._proc

        if proc is None or proc.poll() is not None:
            return False, f"{self._name} is not running."

        _terminate_process_best_effort(proc)

        with self._lock:
            if self._proc is proc:
                self._proc = None

        return True, f"{self._name} stopped."

    def _reader_loop(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._log.append(line)
        except Exception as e:
            self._log.append(f"[dashboard] Error reading {self._name} output: {e}")
        finally:
            rc = proc.wait()
            self._log.set_returncode(rc)
            self._log.append(f"[dashboard] {self._name} exited with return code {rc}")

            with self._lock:
                if self._proc is proc:
                    self._proc = None

            if self._on_exit is not None:
                try:
                    self._on_exit(rc)
                except Exception:
                    pass


###############################################################################
# Dashboard controller
###############################################################################


class DashboardController:
    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._preferred_mode: ServerMode = ServerMode.LOCAL
        self._active_mode: Optional[ServerMode] = None

        self._status = StatusSnapshot(
            state=OrchestratorState.IDLE.value,
            message="Idle",
            server_mode=ServerMode.LOCAL.value,
        )

        # HPC orchestration
        self._hpc_cancel_requested: bool = False
        self._hpc_host: str = "aqua.qut.edu.au"
        self._hpc_username: Optional[str] = None
        self._hpc_password: Optional[str] = None  # in-memory only; never persisted/logged
        self._orchestrator: Optional[Orchestrator] = None
        self._orch_thread: Optional[threading.Thread] = None

        # Local policy server
        self._local_policy_log = RingLog(MAX_LOCAL_SERVER_LOG_LINES, "local-policy-log")
        self._local_start_thread: Optional[threading.Thread] = None
        self._local_policy = ManagedProcess(
            name="Local policy server",
            log=self._local_policy_log,
            on_exit=self._on_local_policy_exit,
        )

        # xArm client
        self._xarm_log = RingLog(MAX_XARM_LOG_LINES, "xarm-client-log")
        self._xarm_client = ManagedProcess(
            name="xArm client",
            log=self._xarm_log,
        )

        # Camera server (track running; logs not currently surfaced)
        self._camera_lock = threading.Lock()
        self._camera_server_proc: Optional[subprocess.Popen] = None

        # Cameras (ZMQ backends)
        self.base_camera = ZmqCameraBackend(host="172.23.224.1", port=5000, img_size=None, name="base", target_fps=15.0)
        self.wrist_camera = ZmqCameraBackend(host="172.23.224.1", port=5001, img_size=None, name="wrist", target_fps=15.0)

    # -------------------- shared state helpers --------------------

    def _get_display_mode(self) -> ServerMode:
        return self._active_mode if self._active_mode is not None else self._preferred_mode

    def _set_status(self, *, state: str, message: str, server_mode: Optional[ServerMode] = None) -> None:
        display_mode = self._get_display_mode()
        self._status = StatusSnapshot(
            state=state,
            message=message,
            server_mode=(server_mode.value if server_mode is not None else display_mode.value),
        )

    def _local_starting(self) -> bool:
        t = self._local_start_thread
        return bool(t is not None and t.is_alive())

    def _any_server_running_or_busy(self) -> bool:
        orch = self._orchestrator
        hpc_ready = orch is not None and orch.state == OrchestratorState.READY
        hpc_thread_alive = self._orch_thread is not None and self._orch_thread.is_alive()
        local_ready = self._local_policy.running()
        return bool(hpc_ready or hpc_thread_alive or local_ready or self._local_starting())

    # -------------------- health checking --------------------

    @staticmethod
    def _ws_health_once(host: str, port: int, timeout_s: float = 3.0) -> Tuple[bool, str]:
        uri = f"ws://{host}:{port}"
        try:
            ws = websockets.sync.client.connect(uri, open_timeout=timeout_s, close_timeout=timeout_s)
            ws.close()
            return True, f"WebSocket handshake succeeded at {uri}."
        except Exception as e:
            return False, f"{uri} -> {type(e).__name__}: {e}"

    def _wait_for_local_policy_healthy(self, proc: subprocess.Popen, timeout_s: float = 900.0, poll_s: float = 2.0) -> None:
        deadline = time.time() + timeout_s
        last_log_t = 0.0
        last_msg = ""

        while time.time() < deadline:
            ret = proc.poll()
            if ret is not None:
                self._local_policy_log.set_returncode(ret)
                raise RuntimeError(f"Local policy server exited during startup (rc={ret}). See Policy Server Log.")

            ok, msg = self._ws_health_once("127.0.0.1", XARM_PORT, timeout_s=2.0)
            if ok:
                return

            now = time.time()
            if msg != last_msg or (now - last_log_t) > 10.0:
                self._local_policy_log.append(f"[dashboard] Waiting for local server health... ({msg})")
                last_msg = msg
                last_log_t = now

            time.sleep(poll_s)

        raise RuntimeError(f"Timed out waiting for local server to become healthy (>{timeout_s}s).")

    # -------------------- HPC orchestrator integration --------------------

    def _attempt_cancel_hpc_if_requested(self) -> None:
        if not self._hpc_cancel_requested:
            return
        orch = self._orchestrator
        if orch is None:
            return
        job_id = getattr(orch, "job_id", None)
        if not job_id:
            return

        try:
            orch.request_cancel()
        except Exception:
            pass
        try:
            orch.stop_tunnel()
        except Exception:
            pass

        _, msg = orch.delete_job()

        self._hpc_cancel_requested = False
        self._active_mode = None
        self._set_status(
            state=OrchestratorState.IDLE.value,
            message=f"HPC job cancelled: {job_id}\n\n{msg}",
            server_mode=ServerMode.HPC,
        )

    def _on_orchestrator_state_change(self, state: OrchestratorState, message: str) -> None:
        self._set_status(state=state.value, message=message, server_mode=ServerMode.HPC)
        print(f"[DASHBOARD][HPC] {state.value}: {message}")

        self._attempt_cancel_hpc_if_requested()

        if state in {
            OrchestratorState.CANCELLED,
            OrchestratorState.ERROR_JOB_SUBMISSION,
            OrchestratorState.ERROR_JOB_STARTUP,
            OrchestratorState.ERROR_TUNNEL_START,
            OrchestratorState.ERROR_SERVER_UNHEALTHY,
            OrchestratorState.ERROR_TUNNEL_BROKEN,
            OrchestratorState.ERROR_STOP,
        }:
            self._active_mode = None

    def _start_orchestrator_thread(self) -> bool:
        if self._orch_thread is not None and self._orch_thread.is_alive():
            return False
        if self._orchestrator is not None and self._orchestrator.state == OrchestratorState.READY:
            return False

        if not self._hpc_username or not self._hpc_password:
            # Caller should ensure credentials are set; keep error explicit for UI.
            self._emit_state_for_ui_error(
                "HPC credentials not set. Click HPC and connect with username/password.",
                OrchestratorState.ERROR_JOB_SUBMISSION,
            )
            return False

        config = OrchestratorConfig(
            host=self._hpc_host,
            username=self._hpc_username,
            password=self._hpc_password,
            repo_dir=f"/home/{self._hpc_username}/openpi",
        )
        orch = Orchestrator(config=config)
        orch.on_state_change = self._on_orchestrator_state_change
        self._orchestrator = orch

        def _run() -> None:
            orch.run_full_sequence()

        t = threading.Thread(target=_run, daemon=True)
        self._orch_thread = t
        t.start()
        return True

    def _emit_state_for_ui_error(self, msg: str, state: OrchestratorState) -> None:
        self._set_status(state=state.value, message=msg, server_mode=ServerMode.HPC)

    # -------------------- local policy server --------------------

    def _on_local_policy_exit(self, rc: int) -> None:
        if self._active_mode == ServerMode.LOCAL:
            self._active_mode = None

        if rc != 0:
            self._set_status(
                state="ERROR_LOCAL_EXIT",
                message=f"Local policy server exited (rc={rc}). See Policy Server Log.",
                server_mode=ServerMode.LOCAL,
            )
        else:
            self._set_status(
                state=OrchestratorState.IDLE.value,
                message="Local policy server stopped.",
                server_mode=ServerMode.LOCAL,
            )

    def _start_local_policy_server_blocking(self) -> None:
        self._set_status(state="STARTING_LOCAL", message="Starting local policy server...", server_mode=ServerMode.LOCAL)

        cmd = [
            "uv",
            "run",
            "scripts/serve_policy.py",
            "--env",
            "DEMO",
            "--port",
            str(XARM_PORT),
        ]

        env = _make_child_env(OPENPI_ROOT)
        proc = self._local_policy.start(cmd, cwd=str(OPENPI_ROOT), env=env, clear_log=True)

        self._local_policy_log.append("[dashboard] Launched local policy server process.")
        self._local_policy_log.append(f"[dashboard] OPENPI_ROOT: {OPENPI_ROOT}")
        self._local_policy_log.append(f"[dashboard] CMD: {' '.join(cmd)}")

        self._set_status(
            state="CHECKING_LOCAL_HEALTH",
            message="Waiting for local server to become healthy...",
            server_mode=ServerMode.LOCAL,
        )
        self._wait_for_local_policy_healthy(proc, timeout_s=900.0, poll_s=2.0)

        self._set_status(
            state=OrchestratorState.READY.value,
            message=f"Local policy server healthy on ws://localhost:{XARM_PORT}",
            server_mode=ServerMode.LOCAL,
        )

    def _start_local_policy_server_async(self) -> bool:
        if self._local_start_thread is not None and self._local_start_thread.is_alive():
            return False
        if self._local_policy.running():
            return False

        def _run() -> None:
            try:
                self._start_local_policy_server_blocking()
            except Exception as e:
                self._active_mode = None
                self._set_status(
                    state="ERROR_LOCAL_START",
                    message=f"Local server failed to start.\n\n{e}",
                    server_mode=ServerMode.LOCAL,
                )

        t = threading.Thread(target=_run, daemon=True)
        self._local_start_thread = t
        t.start()
        return True

    def _stop_local_policy_server(self) -> Tuple[bool, str]:
        self._set_status(state="STOPPING_LOCAL", message="Stopping local policy server...", server_mode=ServerMode.LOCAL)
        ok, detail = self._local_policy.stop()
        if ok:
            self._active_mode = None
            self._set_status(state=OrchestratorState.IDLE.value, message="Local policy server stopped.", server_mode=ServerMode.LOCAL)
        return ok, detail

    # -------------------- public API surface --------------------

    def set_mode(self, mode: ServerMode) -> Dict[str, Any]:
        if self._any_server_running_or_busy():
            raise HTTPException(status_code=409, detail="Cannot change mode while a server is running or starting.")
        self._preferred_mode = mode
        self._set_status(state=OrchestratorState.IDLE.value, message=f"Mode set to {mode.value.upper()}.", server_mode=mode)
        return {"status": "ok", "preferred_mode": mode.value}

    # -------------------- HPC credential session --------------------

    def hpc_session(self) -> Dict[str, Any]:
        return {
            "connected": bool(self._hpc_username and self._hpc_password),
            "host": self._hpc_host,
            "username": self._hpc_username,
        }

    def clear_hpc_session(self) -> Dict[str, Any]:
        # Stability: don't allow clearing creds while orchestration may still need them (cancel, qdel, tunnel).
        if self._any_server_running_or_busy():
            raise HTTPException(status_code=409, detail="Cannot clear HPC credentials while server is running/starting. Stop the server first.")
        self._hpc_username = None
        self._hpc_password = None
        self._orchestrator = None
        self._orch_thread = None
        return {"status": "cleared"}

    def connect_hpc(self, host: Optional[str], username: str, password: str) -> Dict[str, Any]:
        if self._any_server_running_or_busy():
            raise HTTPException(status_code=409, detail="Cannot change HPC credentials while server is running/starting.")

        host_final = (host or self._hpc_host).strip() or self._hpc_host
        username_final = username.strip()
        if not username_final:
            raise HTTPException(status_code=400, detail="username is required")
        if not password:
            raise HTTPException(status_code=400, detail="password is required")

        # Validate credentials immediately, but never log the password.
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(
                hostname=host_final,
                port=22,
                username=username_final,
                password=password,
                allow_agent=False,
                look_for_keys=False,
                timeout=10.0,
                banner_timeout=10.0,
                auth_timeout=10.0,
            )
            stdin, stdout, stderr = c.exec_command("echo connected", timeout=10.0)
            _ = stdout.read()
            _ = stderr.read()
            c.close()
        except paramiko.AuthenticationException:
            raise HTTPException(status_code=401, detail="Authentication failed. Check username/password.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to connect to HPC: {type(e).__name__}: {e}")

        self._hpc_host = host_final
        self._hpc_username = username_final
        self._hpc_password = password
        return {"status": "connected", "host": self._hpc_host, "username": self._hpc_username}

    def status(self) -> Dict[str, Any]:
        s = self._status

        orch = self._orchestrator
        hpc_thread_alive = self._orch_thread is not None and self._orch_thread.is_alive()
        local_thread_alive = self._local_starting()

        hpc_ready = orch is not None and orch.state == OrchestratorState.READY
        local_ready = self._local_policy.running()

        server_running = bool(hpc_ready or local_ready)
        busy = bool(hpc_thread_alive or local_thread_alive)

        job_id = getattr(orch, "job_id", None) if orch is not None else None
        remote_ip = getattr(orch, "remote_ip", None) if orch is not None else None
        remote_port = getattr(orch, "remote_port", None) if orch is not None else None

        if s.server_mode == ServerMode.LOCAL.value:
            job_id = None
            remote_ip = None
            remote_port = None

        return {
            "state": s.state,
            "message": s.message,
            "job_id": job_id,
            "remote_ip": remote_ip,
            "remote_port": remote_port,
            "server_running": server_running,
            "busy": busy,
            "server_mode": s.server_mode,
            "preferred_mode": self._preferred_mode.value,
            "active_mode": (self._active_mode.value if self._active_mode is not None else None),
            "hpc_running": bool(hpc_ready),
            "local_running": bool(local_ready),
            "hpc_cancel_requested": bool(self._hpc_cancel_requested),
            "openpi_root": str(OPENPI_ROOT),
        }

    def policy_server_log(self) -> Dict[str, Any]:
        log, rc = self._local_policy_log.snapshot()
        return {
            "mode": self._get_display_mode().value,
            "local_running": self._local_policy.running(),
            "local_returncode": rc,
            "log": log,
        }

    def hpc_diagnostics(self) -> Dict[str, Any]:
        orch = self._orchestrator
        if orch is None:
            return {"ok": False, "diagnostics": "Orchestrator not initialized yet."}
        try:
            return {"ok": True, "diagnostics": orch.collect_diagnostics()}
        except Exception as e:
            return {"ok": False, "diagnostics": f"Failed to collect diagnostics: {e}"}

    def run_server(self, mode: ServerMode) -> Dict[str, Any]:
        if self._any_server_running_or_busy():
            raise HTTPException(status_code=409, detail="Policy server already running (HPC or local).")

        self._preferred_mode = mode

        if mode == ServerMode.HPC:
            if not (self._hpc_username and self._hpc_password):
                raise HTTPException(status_code=400, detail="HPC credentials not set. Please connect first.")
            self._active_mode = ServerMode.HPC
            self._hpc_cancel_requested = False
            self._set_status(
                state=OrchestratorState.SUBMITTING_JOB.value,
                message="Submitting policy server job...",
                server_mode=ServerMode.HPC,
            )

            # Preserve existing (misleading) log line for compatibility with current UI expectations.
            self._local_policy_log.append("[dashboard] Run Local pressed.")

            started = self._start_orchestrator_thread()
            if not started:
                self._active_mode = None
                raise HTTPException(status_code=409, detail="Server orchestration already running or server already READY.")
            return {"status": "started", "mode": "hpc"}

        # LOCAL (async so the dashboard thread never blocks)
        self._active_mode = ServerMode.LOCAL
        started = self._start_local_policy_server_async()
        if not started:
            self._active_mode = None
            raise HTTPException(status_code=409, detail="Local server is already starting or running.")
        return {"status": "starting", "mode": "local"}

    def stop_server(self) -> Dict[str, Any]:
        mode = self._active_mode or self._get_display_mode()

        if mode == ServerMode.LOCAL:
            ok, detail = self._stop_local_policy_server()
            if not ok:
                raise HTTPException(status_code=400, detail=detail)
            return {"status": "stopped", "mode": "local", "detail": detail}

        orch = self._orchestrator
        self._active_mode = ServerMode.HPC
        self._hpc_cancel_requested = True
        self._set_status(state="CANCEL_REQUESTED", message="Cancelling HPC server/job...", server_mode=ServerMode.HPC)

        if orch is not None:
            try:
                orch.request_cancel()
            except Exception:
                pass

        self._attempt_cancel_hpc_if_requested()

        return {
            "status": "cancelling",
            "mode": "hpc",
            "detail": "Cancel requested. If job_id is already known it will be qdel'd immediately; otherwise it will be deleted as soon as it appears.",
        }

    def health(self) -> Dict[str, Any]:
        mode = self._get_display_mode()
        if mode == ServerMode.LOCAL:
            ok, msg = self._ws_health_once("127.0.0.1", XARM_PORT, timeout_s=3.0)
            return {"status": "healthy" if ok else "unhealthy", "detail": msg}

        orch = self._orchestrator
        if orch is None:
            return {"status": "unknown", "detail": "Orchestrator has not run yet."}

        ok = orch.check_policy_health()
        return {"status": "healthy" if ok else "unhealthy", "detail": orch.last_message}

    # -------------------- cameras --------------------

    def start_cameras(self) -> Dict[str, Any]:
        if not PIPELINE_SCRIPT.exists():
            raise HTTPException(status_code=500, detail=f"Pipeline script not found at {PIPELINE_SCRIPT}")

        with self._camera_lock:
            if self._camera_server_proc is not None and self._camera_server_proc.poll() is None:
                raise HTTPException(status_code=409, detail="Camera server already running.")

            cmd = ["bash", str(PIPELINE_SCRIPT), "cameras"]
            proc = subprocess.Popen(
                cmd,
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            time.sleep(3.0)
            ret = proc.poll()
            if ret is not None and ret != 0:
                try:
                    output, _ = proc.communicate(timeout=1.0)
                except Exception:
                    output = "<no output captured>"
                raise HTTPException(status_code=500, detail=f"Camera server exited early with code {ret}:\n{output}")

            self._camera_server_proc = proc

        return {"status": "started"}

    def camera_status(self) -> Dict[str, Any]:
        with self._camera_lock:
            server_running = bool(self._camera_server_proc is not None and self._camera_server_proc.poll() is None)
        return {
            "server_running": server_running,
            "base": self.base_camera.get_status(),
            "wrist": self.wrist_camera.get_status(),
        }

    # -------------------- xArm client --------------------

    def run_xarm(self, prompt: Optional[str]) -> Dict[str, Any]:
        state = self._status.state
        if state != OrchestratorState.READY.value:
            raise HTTPException(status_code=400, detail=f"Policy server not READY (current state: {state}). Start it first.")

        if self._xarm_client.running():
            raise HTTPException(status_code=409, detail="xArm client is already running.")

        prompt_final = (prompt or DEFAULT_XARM_PROMPT).strip() or DEFAULT_XARM_PROMPT

        cmd = [
            "uv",
            "run",
            "src/xarm6_control/main2.py",
            "--remote_host",
            "localhost",
            "--remote_port",
            str(XARM_PORT),
            "--prompt",
            prompt_final,
        ]
        env = _make_child_env(OPENPI_ROOT)
        self._xarm_client.start(cmd, cwd=str(OPENPI_ROOT), env=env, clear_log=True)

        self._xarm_log.append("[dashboard] Starting xArm client...")
        self._xarm_log.append(f"[dashboard] OPENPI_ROOT: {OPENPI_ROOT}")
        self._xarm_log.append(f"[dashboard] CMD: {' '.join(cmd)}")
        self._xarm_log.append(f"[dashboard] Prompt: {prompt_final}")

        time.sleep(0.4)
        if not self._xarm_client.running():
            raise HTTPException(status_code=500, detail="xArm client exited immediately. Check the xArm Client log box.")

        return {"status": "started"}

    def stop_xarm(self) -> Dict[str, Any]:
        if not self._xarm_client.running():
            raise HTTPException(status_code=400, detail="xArm client is not running.")
        ok, detail = self._xarm_client.stop()
        if not ok:
            raise HTTPException(status_code=500, detail=f"Failed to stop xArm client: {detail}")
        self._xarm_log.append("[dashboard] Sent stop signal to xArm client.")
        return {"status": "terminating"}

    def xarm_status(self) -> Dict[str, Any]:
        log, rc = self._xarm_log.snapshot()
        running = self._xarm_client.running()
        return {"running": running, "returncode": (None if running else rc), "log": log}


###############################################################################
# FastAPI app + routes
###############################################################################

app = FastAPI(title="XArm6 Demo Dashboard")
controller = DashboardController()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/set-mode", response_class=JSONResponse)
def api_set_mode(req: SetModeRequest) -> Dict[str, Any]:
    return controller.set_mode(req.mode)


@app.get("/api/status", response_class=JSONResponse)
def api_status() -> Dict[str, Any]:
    return controller.status()


@app.get("/api/policy-server-log", response_class=JSONResponse)
def api_policy_server_log() -> Dict[str, Any]:
    return controller.policy_server_log()


@app.get("/api/hpc-diagnostics", response_class=JSONResponse)
def api_hpc_diagnostics() -> Dict[str, Any]:
    return controller.hpc_diagnostics()


@app.post("/api/run-server", response_class=JSONResponse)
def api_run_server(req: RunServerRequest) -> Dict[str, Any]:
    return controller.run_server(req.mode)


@app.post("/api/stop-server", response_class=JSONResponse)
def api_stop_server() -> Dict[str, Any]:
    return controller.stop_server()


@app.get("/api/health", response_class=JSONResponse)
def api_health() -> Dict[str, Any]:
    return controller.health()


@app.post("/api/start-cameras", response_class=JSONResponse)
def api_start_cameras() -> Dict[str, Any]:
    return controller.start_cameras()


@app.get("/api/hpc/session", response_class=JSONResponse)
def api_hpc_session() -> Dict[str, Any]:
    return controller.hpc_session()


@app.post("/api/hpc/connect", response_class=JSONResponse)
def api_hpc_connect(req: HpcConnectRequest) -> Dict[str, Any]:
    return controller.connect_hpc(req.host, req.username, req.password)


@app.post("/api/hpc/clear", response_class=JSONResponse)
def api_hpc_clear() -> Dict[str, Any]:
    return controller.clear_hpc_session()


@app.get("/video/base")
def video_base() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(controller.base_camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/video/wrist")
def video_wrist() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_stream_generator(controller.wrist_camera),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/camera-status", response_class=JSONResponse)
def api_camera_status() -> Dict[str, Any]:
    return controller.camera_status()


@app.post("/api/run-xarm", response_class=JSONResponse)
def api_run_xarm(req: RunXarmRequest) -> Dict[str, Any]:
    return controller.run_xarm(req.prompt)


@app.post("/api/stop-xarm", response_class=JSONResponse)
def api_stop_xarm() -> Dict[str, Any]:
    return controller.stop_xarm()


@app.get("/api/xarm-status", response_class=JSONResponse)
def api_xarm_status() -> Dict[str, Any]:
    return controller.xarm_status()


###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "xarm6_control.dashboard.dashboard_app:app",
        host="0.0.0.0",
        port=9000,
        reload=False,
    )


