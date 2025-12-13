#!/usr/bin/env python3
"""
xarm_server_orchestrator.py

Backend orchestrator for:
- Submitting the policy server job on the HPC
- Waiting for it to come up and publish its endpoint
- Starting the SSH tunnel
- Checking WebSocket health from the local machine
- Cancelling / stopping (qdel + tunnel shutdown)

CLI:
    python xarm_server_orchestrator.py run
    python xarm_server_orchestrator.py health
    python xarm_server_orchestrator.py stop
"""

from __future__ import annotations

import enum
import re
import shlex
import subprocess
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import websockets.sync.client


# ============================
# Configuration
# ============================

@dataclass
class OrchestratorConfig:
    # SSH alias or user@host
    hpc_prefix: str = "aqua"

    # Directory where this file lives: src/xarm6_control/dashboard (or similar)
    here: Path = Path(__file__).resolve().parent

    # Pipeline script lives alongside this orchestrator
    pipeline_script: Path = field(init=False)

    # Local port to expose the policy server through the tunnel
    policy_port: int = 8000

    # Endpoint file on HPC (written by your remote job)
    # IMPORTANT: use $HOME so it expands even when quoted.
    endpoint_file: str = "$HOME/.openpi_policy_endpoint"

    # How long to wait for the endpoint file to appear
    wait_endpoint_timeout_s: int = 500
    wait_endpoint_poll_s: int = 5

    # Single WebSocket connect timeout (per attempt)
    ws_health_timeout_s: int = 3

    # Overall time to wait for the server to become healthy (with retries)
    server_health_timeout_s: int = 500
    server_health_poll_s: float = 5.0

    # After starting ssh, wait briefly and detect immediate failures
    tunnel_start_grace_s: float = 1.0

    def __post_init__(self) -> None:
        self.pipeline_script = self.here / "xarm_pipeline.sh"
        print(f"[CONFIG] Using pipeline script at: {self.pipeline_script}")


# ============================
# State machine
# ============================

class OrchestratorState(enum.Enum):
    IDLE = "IDLE"

    CLEARING_OLD_ENDPOINT = "CLEARING_OLD_ENDPOINT"
    SUBMITTING_JOB = "SUBMITTING_JOB"
    JOB_QUEUED = "JOB_QUEUED"
    JOB_RUNNING_STARTING_SERVER = "JOB_RUNNING_STARTING_SERVER"
    SERVER_READY_NO_TUNNEL = "SERVER_READY_NO_TUNNEL"
    STARTING_TUNNEL = "STARTING_TUNNEL"
    CHECKING_POLICY_HEALTH = "CHECKING_POLICY_HEALTH"
    READY = "READY"

    STOPPING = "STOPPING"
    CANCELLED = "CANCELLED"

    ERROR_JOB_SUBMISSION = "ERROR_JOB_SUBMISSION"
    ERROR_JOB_STARTUP = "ERROR_JOB_STARTUP"
    ERROR_TUNNEL_START = "ERROR_TUNNEL_START"
    ERROR_SERVER_UNHEALTHY = "ERROR_SERVER_UNHEALTHY"
    ERROR_TUNNEL_BROKEN = "ERROR_TUNNEL_BROKEN"
    ERROR_STOP = "ERROR_STOP"


@dataclass
class Orchestrator:
    config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    on_state_change: Optional[Callable[[OrchestratorState, str], None]] = None

    # Runtime attributes
    state: OrchestratorState = OrchestratorState.IDLE
    last_message: str = ""
    job_id: Optional[str] = None
    remote_ip: Optional[str] = None
    remote_port: Optional[int] = None
    tunnel_process: Optional[subprocess.Popen] = None

    # cancellation flag
    _cancel: threading.Event = field(default_factory=threading.Event, init=False)

    # --------------- helpers ---------------

    def request_cancel(self) -> None:
        """Signal the orchestrator to stop as soon as possible."""
        self._cancel.set()

    def _cancelled(self) -> bool:
        return self._cancel.is_set()

    def _emit_state(self, state: OrchestratorState, message: str = "") -> None:
        self.state = state
        self.last_message = message
        if self.on_state_change is not None:
            self.on_state_change(state, message)
        else:
            print(f"[STATE] {state.value}: {message}")

    def _run_local(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a local command and capture stdout/stderr."""
        return subprocess.run(cmd, check=check, text=True, capture_output=True)

    def _run_local_stream(self, cmd: list[str]) -> subprocess.Popen:
        """Run a local long-lived command (e.g. ssh tunnel)."""
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
        )

    def _ssh_hpc(self, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command on the HPC via SSH."""
        cmd = ["ssh", self.config.hpc_prefix, remote_cmd]
        return self._run_local(cmd, check=check)

    # --- remote path quoting helpers (CRITICAL for $HOME/~/ expansion) ---

    def _remote_path(self, p: str) -> str:
        """
        Normalize "~/" to "$HOME/" so it expands even if we quote it.
        """
        if p.startswith("~/"):
            return "$HOME/" + p[2:]
        if p == "~":
            return "$HOME"
        return p

    def _dq(self, s: str) -> str:
        """
        Double-quote for remote shell. Allows $HOME expansion.
        """
        return '"' + s.replace('"', '\\"') + '"'

    # --------------- diagnostics ---------------

    def _best_effort(self, label: str, fn) -> str:
        """Run fn() and return stdout/stderr as a debug block without raising."""
        try:
            out = fn()
            if isinstance(out, subprocess.CompletedProcess):
                s = ""
                if out.stdout:
                    s += out.stdout.strip()
                if out.stderr:
                    if s:
                        s += "\n"
                    s += out.stderr.strip()
                return f"--- {label} ---\n{s.strip() or '<no output>'}"
            return f"--- {label} ---\n{str(out).strip()}"
        except Exception as e:
            return f"--- {label} ---\n<failed: {e}>"

    def _collect_hpc_diagnostics(self) -> str:
        """Grab useful HPC-side info for the dashboard error message."""
        blocks: list[str] = []

        if self.job_id:
            jid = shlex.quote(self.job_id)
            blocks.append(self._best_effort("qstat", lambda: self._ssh_hpc(f"qstat {jid} 2>&1 || true", check=False)))
            blocks.append(self._best_effort("qstat -f", lambda: self._ssh_hpc(f"qstat -f {jid} 2>&1 || true", check=False)))
            blocks.append(self._best_effort("qpeek", lambda: self._ssh_hpc(f"qpeek {jid} 2>&1 || true", check=False)))

        ef = self._dq(self._remote_path(self.config.endpoint_file))
        blocks.append(self._best_effort("endpoint ls", lambda: self._ssh_hpc(f"ls -l {ef} 2>&1 || true", check=False)))
        blocks.append(self._best_effort("endpoint cat", lambda: self._ssh_hpc(f"cat {ef} 2>&1 || true", check=False)))

        return "\n\n".join(blocks).strip()

    def collect_diagnostics(self) -> str:
        """Public wrapper for dashboard."""
        return self._collect_hpc_diagnostics()

    # --------------- tunnel / job control ---------------

    def _tunnel_running(self) -> bool:
        return self.tunnel_process is not None and self.tunnel_process.poll() is None

    def stop_tunnel(self) -> None:
        """Terminate the SSH tunnel process if running."""
        proc = self.tunnel_process
        if proc is None:
            return
        if proc.poll() is not None:
            self.tunnel_process = None
            return

        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            self.tunnel_process = None

    def delete_job(self) -> Tuple[bool, str]:
        """Best-effort qdel of the current job_id (if known)."""
        if not self.job_id:
            return False, "No job_id known; nothing to delete."

        jid = shlex.quote(self.job_id)
        res = self._ssh_hpc(f"qdel {jid} 2>&1 || true", check=False)
        out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        out = out.strip() or "<no output>"
        return True, f"qdel {self.job_id}:\n{out}"

    def stop(self) -> None:
        """
        Stop everything:
        - set cancel flag (so run_full_sequence exits)
        - stop tunnel
        - qdel job (if known)
        """
        self.request_cancel()
        self._emit_state(
            OrchestratorState.STOPPING,
            "Stopping: cancelling orchestration, shutting down tunnel, deleting HPC job...",
        )

        debug: list[str] = []

        # Stop tunnel first (local)
        try:
            if self._tunnel_running():
                self.stop_tunnel()
                debug.append("Tunnel: terminated.")
            else:
                debug.append("Tunnel: not running.")
        except Exception as e:
            debug.append(f"Tunnel stop failed: {e}")

        # Delete HPC job (remote)
        try:
            ok, msg = self.delete_job()
            debug.append(msg)
        except Exception as e:
            debug.append(f"qdel failed: {e}")

        self._emit_state(OrchestratorState.CANCELLED, "Stopped.\n\n" + "\n".join(debug))

    # --------------- small utility steps ---------------

    def clear_remote_endpoint_file(self) -> None:
        """Remove any stale endpoint file on the HPC."""
        if self._cancelled():
            return

        self._emit_state(
            OrchestratorState.CLEARING_OLD_ENDPOINT,
            "Removing old endpoint file on HPC (if any)...",
        )
        try:
            ef = self._dq(self._remote_path(self.config.endpoint_file))
            self._ssh_hpc(f"rm -f {ef} 2>&1 || true", check=False)
        except Exception as e:
            # Non-fatal
            self.last_message = f"Failed to clear old endpoint file (non-fatal): {e}"

    # --------------- main steps ---------------

    def submit_policy_job(self) -> bool:
        """Call ./xarm_pipeline.sh serve-policy and parse the job ID."""
        self.clear_remote_endpoint_file()
        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before job submission.")
            return False

        self._emit_state(
            OrchestratorState.SUBMITTING_JOB,
            "Submitting policy server job via pipeline script...",
        )

        try:
            result = self._run_local([str(self.config.pipeline_script), "serve-policy"], check=True)
        except subprocess.CalledProcessError as e:
            stdout = (e.stdout or "").strip()
            stderr = (e.stderr or "").strip()
            msg = "Failed to submit job.\n"
            if stdout:
                msg += f"\n--- pipeline stdout ---\n{stdout}"
            if stderr:
                msg += f"\n--- pipeline stderr ---\n{stderr}"
            self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, msg.strip())
            return False
        except Exception as e:
            self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, f"Failed to submit job: {e}")
            return False

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        # PBS typically prints something like "15063933.aqua"
        job_id = None
        candidates = re.findall(r"\b\d+(?:\.\w+)?\b", stdout)
        for c in reversed(candidates):
            if "." in c:
                job_id = c
                break
        if job_id is None and candidates:
            job_id = candidates[-1]

        self.job_id = job_id

        msg = f"Job submitted. Job ID: {self.job_id or 'UNKNOWN'}; waiting for endpoint file..."
        if stdout:
            msg += f"\n\n--- pipeline stdout ---\n{stdout}"
        if stderr:
            msg += f"\n\n--- pipeline stderr ---\n{stderr}"

        self._emit_state(OrchestratorState.JOB_QUEUED, msg.strip())
        return True

    def wait_for_endpoint(self) -> bool:
        """Poll the HPC for endpoint file until it appears or timeout."""
        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before waiting for endpoint.")
            return False

        self._emit_state(
            OrchestratorState.JOB_RUNNING_STARTING_SERVER,
            "Waiting for server to start and publish endpoint on HPC...",
        )

        deadline = time.time() + self.config.wait_endpoint_timeout_s
        ef = self._dq(self._remote_path(self.config.endpoint_file))

        while time.time() < deadline:
            if self._cancelled():
                self._emit_state(OrchestratorState.CANCELLED, "Cancelled while waiting for endpoint.")
                return False

            try:
                result = self._ssh_hpc(f"cat {ef} 2>/dev/null || true", check=False)
                content = (result.stdout or "").strip()
            except Exception:
                content = ""

            if content:
                host = None
                ip = None
                port = None
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("HOST="):
                        host = line.split("=", 1)[1].strip()
                    elif line.startswith("IP="):
                        ip = line.split("=", 1)[1].strip()
                    elif line.startswith("PORT="):
                        try:
                            port = int(line.split("=", 1)[1].strip())
                        except Exception:
                            port = None

                if ip and port:
                    self.remote_ip = ip
                    self.remote_port = port
                    self._emit_state(
                        OrchestratorState.SERVER_READY_NO_TUNNEL,
                        f"Server reported on node {host or 'UNKNOWN'} with IP {ip}:{port}",
                    )
                    return True

            time.sleep(self.config.wait_endpoint_poll_s)

        msg = f"Timed out waiting for endpoint file ({self.config.wait_endpoint_timeout_s}s)."
        diag = self._collect_hpc_diagnostics()
        if diag:
            msg += "\n\n" + diag
        self._emit_state(OrchestratorState.ERROR_JOB_STARTUP, msg.strip())
        return False

    def start_tunnel(self) -> bool:
        """Start an SSH tunnel from local -> remote_ip:remote_port."""
        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before starting tunnel.")
            return False

        if not self.remote_ip or not self.remote_port:
            self._emit_state(
                OrchestratorState.ERROR_TUNNEL_START,
                "Cannot start tunnel: remote IP/port unknown. Did the endpoint file get created?",
            )
            return False

        # If an old tunnel exists, kill it first.
        try:
            self.stop_tunnel()
        except Exception:
            pass

        local_port = self.config.policy_port
        remote_port = self.remote_port

        self._emit_state(
            OrchestratorState.STARTING_TUNNEL,
            f"Starting SSH tunnel localhost:{local_port} -> {self.remote_ip}:{remote_port} via {self.config.hpc_prefix}...",
        )

        cmd = [
            "ssh",
            "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-L", f"{local_port}:{self.remote_ip}:{remote_port}",
            self.config.hpc_prefix,
        ]

        try:
            proc = self._run_local_stream(cmd)
        except OSError as e:
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, f"Failed to start tunnel: {e}")
            return False

        # Detect immediate failure
        time.sleep(self.config.tunnel_start_grace_s)
        if proc.poll() is not None:
            try:
                out, err = proc.communicate(timeout=1.0)
            except Exception:
                out, err = "", ""
            msg = "SSH tunnel exited immediately."
            if out.strip():
                msg += f"\n\n--- ssh stdout ---\n{out.strip()}"
            if err.strip():
                msg += f"\n\n--- ssh stderr ---\n{err.strip()}"
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, msg.strip())
            return False

        self.tunnel_process = proc
        return True

    # -------- health checking --------

    def _check_policy_health_once(self) -> Tuple[bool, str, str]:
        """
        Try a single WebSocket connect.

        Returns:
            (ok, kind, message)
            kind in {"ok", "tunnel", "server"}
        """
        uri = f"ws://localhost:{self.config.policy_port}"
        try:
            ws = websockets.sync.client.connect(
                uri,
                open_timeout=self.config.ws_health_timeout_s,
                close_timeout=self.config.ws_health_timeout_s,
            )
            ws.close()
            return True, "ok", "WebSocket handshake succeeded."
        except OSError as e:
            return False, "tunnel", str(e)
        except Exception as e:
            return False, "server", str(e)

    def check_policy_health(self) -> bool:
        """Single-shot health check."""
        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before health check.")
            return False

        self._emit_state(
            OrchestratorState.CHECKING_POLICY_HEALTH,
            f"Checking WebSocket health on ws://localhost:{self.config.policy_port}...",
        )

        ok, kind, msg = self._check_policy_health_once()

        if ok:
            self._emit_state(OrchestratorState.READY, "Policy server and tunnel are healthy (WebSocket handshake succeeded).")
            return True

        # If tunnel died, attach its stderr
        if kind == "tunnel" and self.tunnel_process is not None and self.tunnel_process.poll() is not None:
            try:
                out, err = self.tunnel_process.communicate(timeout=1.0)
            except Exception:
                out, err = "", ""
            extra = ""
            if out.strip():
                extra += f"\n\n--- ssh stdout ---\n{out.strip()}"
            if err.strip():
                extra += f"\n\n--- ssh stderr ---\n{err.strip()}"
            msg = msg + extra

        if kind == "tunnel":
            self._emit_state(OrchestratorState.ERROR_TUNNEL_BROKEN, f"Tunnel or listener appears down: {msg}")
        else:
            self._emit_state(OrchestratorState.ERROR_SERVER_UNHEALTHY, f"Connected but WebSocket handshake failed: {msg}")

        return False

    def wait_for_server_healthy(self) -> bool:
        """Retry health checks until healthy or timeout."""
        timeout = self.config.server_health_timeout_s
        poll = self.config.server_health_poll_s

        self._emit_state(
            OrchestratorState.CHECKING_POLICY_HEALTH,
            f"Waiting up to {timeout}s for ws://localhost:{self.config.policy_port} to become healthy...",
        )

        start = time.time()
        last_kind: Optional[str] = None
        last_msg: str = ""

        while time.time() - start < timeout:
            if self._cancelled():
                self._emit_state(OrchestratorState.CANCELLED, "Cancelled while waiting for server health.")
                return False

            ok, kind, msg = self._check_policy_health_once()
            if ok:
                self._emit_state(OrchestratorState.READY, "Policy server and tunnel are healthy (WebSocket handshake succeeded).")
                return True

            last_kind = kind
            last_msg = msg
            time.sleep(poll)

        msg = f"Server never became healthy within {timeout}s (last error [{last_kind}]: {last_msg})"
        diag = self._collect_hpc_diagnostics()
        if diag:
            msg += "\n\n" + diag

        if last_kind == "tunnel":
            self._emit_state(OrchestratorState.ERROR_TUNNEL_BROKEN, msg)
        else:
            self._emit_state(OrchestratorState.ERROR_SERVER_UNHEALTHY, msg)

        return False

    # --------------- high-level orchestration ---------------

    def run_full_sequence(self) -> None:
        """
        Full sequence for “Run server”:

        1) Clear old endpoint file on HPC
        2) Submit job via pipeline script
        3) Wait for endpoint file
        4) Start SSH tunnel
        5) Wait for WebSocket health
        """
        self._emit_state(OrchestratorState.IDLE, "Starting orchestration sequence...")

        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before start.")
            return

        if not self.submit_policy_job():
            return

        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled after job submission.")
            return

        if not self.wait_for_endpoint():
            return

        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled after endpoint received.")
            return

        if not self.start_tunnel():
            return

        time.sleep(1.0)
        self.wait_for_server_healthy()


# ============================
# CLI
# ============================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="XArm policy server orchestrator backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Full sequence: clear endpoint, submit job, wait, tunnel, wait for health")
    subparsers.add_parser("health", help="Single WebSocket health check against localhost")
    subparsers.add_parser("stop", help="Stop tunnel + qdel job (best effort)")

    args = parser.parse_args()
    orch = Orchestrator()

    if args.command == "run":
        orch.run_full_sequence()
    elif args.command == "health":
        orch.check_policy_health()
    elif args.command == "stop":
        orch.stop()


if __name__ == "__main__":
    main()
