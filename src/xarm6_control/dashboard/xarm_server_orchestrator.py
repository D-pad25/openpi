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
import os
import re
import shlex
import shutil
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

    # Runtime credentials (optional). If provided, these are used instead of hpc_prefix.
    # Password is kept in memory by the dashboard process; never persisted here.
    host: str = "aqua.qut.edu.au"
    username: str = ""
    password: str = ""

    # Assumption for dashboard: repo lives at /home/<username>/openpi
    repo_dir: str = ""
    venv_dir: str = ".venv"

    # Directory where this file lives: src/xarm6_control/dashboard (or similar)
    here: Path = Path(__file__).resolve().parent

    # Pipeline script lives alongside this orchestrator
    pipeline_script: Path = field(init=False)

    # Askpass helper shipped with the dashboard (no password in file; reads env var)
    askpass_script: Path = field(init=False)

    # Local port to expose the policy server through the tunnel
    policy_port: int = 8000

    # Endpoint file on HPC (written by your remote job)
    # IMPORTANT: use $HOME so it expands even when quoted.
    endpoint_file: str = "$HOME/.openpi_policy_endpoint"

    # How long to wait for the endpoint file to appear
    wait_endpoint_timeout_s: int = 10000
    wait_endpoint_poll_s: int = 5

    # Single WebSocket connect timeout (per attempt)
    ws_health_timeout_s: int = 3

    # Overall time to wait for the server to become healthy (with retries)
    server_health_timeout_s: int = 1000
    server_health_poll_s: float = 5.0

    # After starting ssh, wait briefly and detect immediate failures
    tunnel_start_grace_s: float = 1.0

    def __post_init__(self) -> None:
        self.pipeline_script = self.here / "xarm_pipeline.sh"
        self.askpass_script = self.here / "ssh_askpass.sh"
        print(f"[CONFIG] Using pipeline script at: {self.pipeline_script}")

        # Fill repo_dir from username if not explicitly set.
        if self.username and not self.repo_dir:
            self.repo_dir = f"/home/{self.username}/openpi"

    def ssh_target(self) -> str:
        """
        Prefer explicit host/username credentials (dashboard mode). Fall back to SSH alias (legacy mode).
        """
        if self.username:
            return f"{self.username}@{self.host}"
        return self.hpc_prefix


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
            text=True,
        )

    def _ssh_env(self) -> dict[str, str]:
        """
        Build env vars for ssh to allow non-interactive password auth via SSH_ASKPASS.

        No password is ever written to disk. It is passed only via environment.
        """
        env = os.environ.copy()
        if self.config.password:
            # Ensure askpass script is executable (best-effort).
            try:
                if self.config.askpass_script.exists():
                    os.chmod(self.config.askpass_script, 0o700)
            except Exception:
                pass

            env["OPENPI_HPC_PASSWORD"] = self.config.password
            env["SSH_ASKPASS"] = str(self.config.askpass_script)
            env["SSH_ASKPASS_REQUIRE"] = "force"
            # DISPLAY must be set for ssh to use SSH_ASKPASS.
            env.setdefault("DISPLAY", ":0")
        return env

    def _ssh_cmd(self, remote_cmd: str) -> list[str]:
        """
        Construct an ssh command that runs `remote_cmd` on the HPC.
        """
        base = ["ssh"]

        # Stability defaults: don't hang forever on auth prompts; accept new host keys (first connect).
        base += ["-o", "BatchMode=no"]
        base += ["-o", "NumberOfPasswordPrompts=1"]
        base += ["-o", "ServerAliveInterval=30"]
        base += ["-o", "ServerAliveCountMax=3"]
        base += ["-o", "StrictHostKeyChecking=accept-new"]

        target = self.config.ssh_target()

        # Use bash -lc for predictable behavior (pipes/heredocs work reliably)
        wrapped = "bash -lc " + shlex.quote(remote_cmd)

        cmd = base + [target, wrapped]

        # Force no-tty so SSH_ASKPASS is used (ssh only invokes askpass without a tty).
        if self.config.password:
            setsid = shutil.which("setsid")
            if setsid:
                cmd = [setsid, "-w"] + cmd
        return cmd

    def _ssh_hpc(self, remote_cmd: str, check: bool = True, timeout_s: float = 120.0) -> subprocess.CompletedProcess:
        """Run a command on the HPC via SSH (supports password auth via SSH_ASKPASS)."""
        cmd = self._ssh_cmd(remote_cmd)
        return subprocess.run(
            cmd,
            check=check,
            text=True,
            capture_output=True,
            env=self._ssh_env(),
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
        )

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

        try:
            if self._tunnel_running():
                self.stop_tunnel()
                debug.append("Tunnel: terminated.")
            else:
                debug.append("Tunnel: not running.")
        except Exception as e:
            debug.append(f"Tunnel stop failed: {e}")

        try:
            _, msg = self.delete_job()
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
            self.last_message = f"Failed to clear old endpoint file (non-fatal): {e}"

    # --------------- main steps ---------------

    def submit_policy_job(self) -> bool:
        """
        Submit the policy server job.

        Legacy path:
          - If no username was provided, uses the existing local pipeline script (SSH alias workflow).

        Dashboard path:
          - If username/password are provided, submits qsub directly on the HPC via ssh.
            Assumes repo on HPC is /home/<username>/openpi.
        """
        self.clear_remote_endpoint_file()
        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before job submission.")
            return False

        self._emit_state(
            OrchestratorState.SUBMITTING_JOB,
            "Submitting policy server job via pipeline script...",
        )

        if not self.config.username:
            # Legacy: rely on your existing pipeline script (may use SSH alias config).
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
        else:
            # Dashboard: submit on HPC via ssh (credentials).
            repo_dir = self.config.repo_dir or f"/home/{self.config.username}/openpi"
            venv_dir = self.config.venv_dir

            job_cmd = "export OPENPI_DATA_HOME=$HOME/.cache/openpi; uv run scripts/serve_policy.py --env DEMO_SERVER --port 8000"
            pbs_script = f"""#!/bin/bash
#PBS -N openpi_cmd
#PBS -q gpu_inter
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

cd "{repo_dir}"
source "{venv_dir}/bin/activate"

NODE_HOST=$(hostname)
NODE_IP=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')

echo "HOST=$NODE_HOST" >  "$HOME/.openpi_policy_endpoint"
echo "IP=$NODE_IP"     >> "$HOME/.openpi_policy_endpoint"
echo "PORT=8000"       >> "$HOME/.openpi_policy_endpoint"

{job_cmd}
"""
            remote = (
                "set -euo pipefail; "
                'JOBFILE=$(mktemp /tmp/openpi_job_XXXXXX.pbs); '
                'cat > "$JOBFILE" <<\'EOF\'\n'
                + pbs_script
                + "\nEOF\n"
                + 'qsub -V "$JOBFILE"; '
                + 'rm -f "$JOBFILE";'
            )

            try:
                result = self._ssh_hpc(remote, check=True, timeout_s=120.0)
            except subprocess.CalledProcessError as e:
                stdout = (e.stdout or "").strip()
                stderr = (e.stderr or "").strip()
                msg = "Failed to submit job.\n"
                if stdout:
                    msg += f"\n--- qsub stdout ---\n{stdout}"
                if stderr:
                    msg += f"\n--- qsub stderr ---\n{stderr}"
                self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, msg.strip())
                return False
            except subprocess.TimeoutExpired:
                self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, "Timed out submitting job over SSH.")
                return False
            except Exception as e:
                # Never include password in errors (it is only in env).
                self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, f"Failed to submit job: {type(e).__name__}: {e}")
                return False

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

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
            "-o", "BatchMode=no",
            "-o", "NumberOfPasswordPrompts=1",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=accept-new",
            "-L", f"{local_port}:{self.remote_ip}:{remote_port}",
            self.config.ssh_target(),
        ]
        if self.config.password:
            setsid = shutil.which("setsid")
            if setsid:
                cmd = [setsid, "-w"] + cmd

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._ssh_env(),
                stdin=subprocess.DEVNULL,
            )
        except OSError as e:
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, f"Failed to start tunnel: {e}")
            return False

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
