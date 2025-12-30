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
import socket
import socketserver
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import paramiko
import websockets.sync.client


# ============================
# Configuration
# ============================

@dataclass
class OrchestratorConfig:
    # SSH target
    host: str = "aqua.qut.edu.au"
    port: int = 22
    username: str = ""
    password: str = ""

    # Repo path assumption (required by dashboard spec)
    repo_dir: str = ""

    # Virtualenv inside repo
    venv_dir: str = ".venv"

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
        if not self.username:
            raise ValueError("OrchestratorConfig.username is required")
        if not self.password:
            raise ValueError("OrchestratorConfig.password is required")
        if not self.repo_dir:
            self.repo_dir = f"/home/{self.username}/openpi"


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

    # Paramiko tunnel server (local listener that forwards to remote_ip:remote_port via SSH transport)
    _tunnel_server: Optional[socketserver.ThreadingTCPServer] = None
    _tunnel_thread: Optional[threading.Thread] = None
    _tunnel_transport: Optional[paramiko.Transport] = None

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

    def _ssh_client(self) -> paramiko.SSHClient:
        """
        Create a short-lived SSH client.

        Important:
        - Never log the password.
        - Keep password in memory only (caller owns config).
        """
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(
            hostname=self.config.host,
            port=self.config.port,
            username=self.config.username,
            password=self.config.password,
            allow_agent=False,
            look_for_keys=False,
            timeout=10.0,
            banner_timeout=10.0,
            auth_timeout=10.0,
        )
        return c

    def _run_remote(self, remote_cmd: str, timeout_s: float = 60.0) -> Tuple[int, str, str]:
        """
        Run a remote command over SSH and return (exit_status, stdout, stderr).
        """
        client = self._ssh_client()
        try:
            stdin, stdout, stderr = client.exec_command(remote_cmd, get_pty=False, timeout=timeout_s)
            # Ensure output is drained before exit status retrieval.
            out_s = stdout.read().decode("utf-8", errors="replace")
            err_s = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            return rc, out_s, err_s
        finally:
            try:
                client.close()
            except Exception:
                pass

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
            if isinstance(out, tuple) and len(out) == 3:
                rc, stdout_s, stderr_s = out
                s = ""
                if stdout_s:
                    s += stdout_s.strip()
                if stderr_s:
                    if s:
                        s += "\n"
                    s += stderr_s.strip()
                if not s.strip():
                    s = "<no output>"
                return f"--- {label} ---\n{s.strip()}"
            return f"--- {label} ---\n{str(out).strip()}"
        except Exception as e:
            return f"--- {label} ---\n<failed: {e}>"

    def _collect_hpc_diagnostics(self) -> str:
        """Grab useful HPC-side info for the dashboard error message."""
        blocks: list[str] = []

        if self.job_id:
            jid = shlex.quote(self.job_id)
            blocks.append(self._best_effort("qstat", lambda: self._run_remote(f"qstat {jid} 2>&1 || true")))
            blocks.append(self._best_effort("qstat -f", lambda: self._run_remote(f"qstat -f {jid} 2>&1 || true")))
            blocks.append(self._best_effort("qpeek", lambda: self._run_remote(f"qpeek {jid} 2>&1 || true")))

        ef = self._dq(self._remote_path(self.config.endpoint_file))
        blocks.append(self._best_effort("endpoint ls", lambda: self._run_remote(f"ls -l {ef} 2>&1 || true")))
        blocks.append(self._best_effort("endpoint cat", lambda: self._run_remote(f"cat {ef} 2>&1 || true")))

        return "\n\n".join(blocks).strip()

    def collect_diagnostics(self) -> str:
        """Public wrapper for dashboard."""
        return self._collect_hpc_diagnostics()

    # --------------- tunnel / job control ---------------

    def _tunnel_running(self) -> bool:
        return bool(self._tunnel_thread is not None and self._tunnel_thread.is_alive() and self._tunnel_server is not None)

    def stop_tunnel(self) -> None:
        """Stop local forwarding server + underlying SSH transport."""
        srv = self._tunnel_server
        thr = self._tunnel_thread
        transport = self._tunnel_transport

        self._tunnel_server = None
        self._tunnel_thread = None
        self._tunnel_transport = None

        if srv is not None:
            try:
                srv.shutdown()
            except Exception:
                pass
            try:
                srv.server_close()
            except Exception:
                pass

        if thr is not None:
            try:
                thr.join(timeout=1.0)
            except Exception:
                pass

        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass

    def delete_job(self) -> Tuple[bool, str]:
        """Best-effort qdel of the current job_id (if known)."""
        if not self.job_id:
            return False, "No job_id known; nothing to delete."

        jid = shlex.quote(self.job_id)
        rc, out, err = self._run_remote(f"qdel {jid} 2>&1 || true")
        msg = (out or "") + ("\n" + err if err else "")
        msg = msg.strip() or "<no output>"
        return True, f"qdel {self.job_id}:\n{msg}"

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
            self._run_remote(f"rm -f {ef} 2>&1 || true")
        except Exception as e:
            self.last_message = f"Failed to clear old endpoint file (non-fatal): {e}"

    # --------------- main steps ---------------

    def submit_policy_job(self) -> bool:
        """
        Submit the policy server as a GPU job via qsub (credential-based SSH).

        The dashboard assumes the OpenPI repo exists at: /home/<username>/openpi
        """
        self.clear_remote_endpoint_file()
        if self._cancelled():
            self._emit_state(OrchestratorState.CANCELLED, "Cancelled before job submission.")
            return False

        self._emit_state(
            OrchestratorState.SUBMITTING_JOB,
            "Submitting policy server job via qsub...",
        )

        # Command that runs on the GPU node (inside qsub script)
        job_cmd = (
            "export OPENPI_DATA_HOME=$HOME/.cache/openpi; "
            "uv run scripts/serve_policy.py --env XARM --port 8000"
        )

        # Inline PBS script submitted from login node.
        # Note: no secrets are embedded in the script; auth is handled by the SSH session.
        pbs_script = f"""#!/bin/bash
#PBS -N openpi_cmd
#PBS -q gpu_inter
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

cd "{self.config.repo_dir}"
source "{self.config.venv_dir}/bin/activate"

NODE_HOST=$(hostname)
NODE_IP=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')

echo "[openpi_cmd] Running on node: $NODE_HOST with IP: $NODE_IP"
echo "[openpi_cmd] Using Python: $(which python)"
echo "[openpi_cmd] Starting command: {job_cmd}"

echo "HOST=$NODE_HOST" >  "$HOME/.openpi_policy_endpoint"
echo "IP=$NODE_IP"     >> "$HOME/.openpi_policy_endpoint"
echo "PORT=8000"       >> "$HOME/.openpi_policy_endpoint"

{job_cmd}
"""

        # Feed the script via stdin to qsub.
        remote_cmd = "qsub -V <<'EOF'\n" + pbs_script + "\nEOF\n"

        try:
            rc, stdout, stderr = self._run_remote(remote_cmd, timeout_s=60.0)
        except paramiko.AuthenticationException:
            self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, "HPC authentication failed. Check username/password/host.")
            return False
        except Exception as e:
            self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, f"Failed to submit job: {type(e).__name__}: {e}")
            return False

        stdout = (stdout or "").strip()
        stderr = (stderr or "").strip()
        if rc != 0 and not stdout and stderr:
            self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, f"Failed to submit job.\n\n{stderr}".strip())
            return False

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
            msg += f"\n\n--- qsub stdout ---\n{stdout}"
        if stderr:
            msg += f"\n\n--- qsub stderr ---\n{stderr}"

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
                rc, out, err = self._run_remote(f"cat {ef} 2>/dev/null || true")
                content = (out or "").strip()
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
            f"Starting SSH tunnel localhost:{local_port} -> {self.remote_ip}:{remote_port} via {self.config.host}...",
        )

        try:
            client = self._ssh_client()
        except paramiko.AuthenticationException:
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, "HPC authentication failed while starting tunnel.")
            return False
        except Exception as e:
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, f"Failed to connect for tunnel: {type(e).__name__}: {e}")
            return False

        transport = client.get_transport()
        if transport is None:
            try:
                client.close()
            except Exception:
                pass
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, "Failed to create SSH transport for tunnel.")
            return False

        remote_host = self.remote_ip
        remote_port_int = int(remote_port)

        class _ForwardHandler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                try:
                    chan = transport.open_channel(
                        kind="direct-tcpip",
                        dest_addr=(remote_host, remote_port_int),
                        src_addr=self.request.getsockname(),
                    )
                except Exception:
                    return

                if chan is None:
                    return

                try:
                    while True:
                        r, _, _ = select.select([self.request, chan], [], [], 1.0)  # type: ignore[name-defined]
                        if self.request in r:
                            data = self.request.recv(16384)
                            if not data:
                                break
                            chan.sendall(data)
                        if chan in r:
                            data = chan.recv(16384)
                            if not data:
                                break
                            self.request.sendall(data)
                finally:
                    try:
                        chan.close()
                    except Exception:
                        pass
                    try:
                        self.request.close()
                    except Exception:
                        pass

        # select is only used inside handler; import lazily to keep module load minimal.
        import select  # noqa: PLC0415

        class _ForwardServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        try:
            server = _ForwardServer(("127.0.0.1", int(local_port)), _ForwardHandler)
        except OSError as e:
            try:
                client.close()
            except Exception:
                pass
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, f"Failed to bind localhost:{local_port} for tunnel: {e}")
            return False

        def _serve() -> None:
            try:
                server.serve_forever(poll_interval=0.2)
            finally:
                try:
                    server.server_close()
                except Exception:
                    pass

        t = threading.Thread(target=_serve, daemon=True)
        t.start()

        # Keep references so we can stop later.
        self._tunnel_server = server
        self._tunnel_thread = t
        self._tunnel_transport = transport

        # Don't keep the SSHClient object; transport remains open.
        try:
            client.close()
        except Exception:
            pass

        # Grace period: confirm thread is alive.
        time.sleep(self.config.tunnel_start_grace_s)
        if not t.is_alive():
            self.stop_tunnel()
            self._emit_state(OrchestratorState.ERROR_TUNNEL_START, "Tunnel thread exited immediately.")
            return False

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

        # With Paramiko forwarding there is no subprocess output to attach; we just surface the error.
        if kind == "tunnel" and not self._tunnel_running():
            msg = "Tunnel appears down (forwarder not running): " + msg

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
