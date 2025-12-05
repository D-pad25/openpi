#!/usr/bin/env python3
"""
xarm_server_orchestrator.py

Backend orchestrator for:
- Submitting the policy server job on the HPC
- Waiting for it to come up and publish its endpoint
- Starting the SSH tunnel
- Checking WebSocket health from the local machine

Designed to be driven by a GUI later, but usable as a CLI now:
    python xarm_server_orchestrator.py run
"""

from __future__ import annotations

import enum
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import websockets.sync.client


# ============================
# Configuration
# ============================

@dataclass
class OrchestratorConfig:
    # SSH alias or user@host
    hpc_prefix: str = "aqua"

    # Directory where this file lives: src/xarm6_control
    here: Path = Path(__file__).resolve().parent

    # Pipeline script lives alongside this orchestrator
    pipeline_script: Path = field(init=False)

    policy_port: int = 8000
    endpoint_file: str = "~/.openpi_policy_endpoint"  # on HPC

    wait_endpoint_timeout_s: int = 300
    wait_endpoint_poll_s: int = 5
    ws_health_timeout_s: int = 3

    def __post_init__(self):
        # Adjust the name if your script is called something else
        self.pipeline_script = self.here / "xarm_pipeline.sh"


# ============================
# State machine
# ============================

class OrchestratorState(enum.Enum):
    IDLE = "IDLE"
    SUBMITTING_JOB = "SUBMITTING_JOB"
    JOB_QUEUED = "JOB_QUEUED"
    JOB_RUNNING_STARTING_SERVER = "JOB_RUNNING_STARTING_SERVER"
    SERVER_READY_NO_TUNNEL = "SERVER_READY_NO_TUNNEL"
    STARTING_TUNNEL = "STARTING_TUNNEL"
    CHECKING_POLICY_HEALTH = "CHECKING_POLICY_HEALTH"
    READY = "READY"
    ERROR_JOB_SUBMISSION = "ERROR_JOB_SUBMISSION"
    ERROR_JOB_STARTUP = "ERROR_JOB_STARTUP"
    ERROR_TUNNEL_START = "ERROR_TUNNEL_START"
    ERROR_SERVER_UNHEALTHY = "ERROR_SERVER_UNHEALTHY"
    ERROR_TUNNEL_BROKEN = "ERROR_TUNNEL_BROKEN"


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

    # --------------- helpers ---------------

    def _emit_state(self, state: OrchestratorState, message: str = "") -> None:
        self.state = state
        self.last_message = message
        # For CLI usage, just print. GUI can override via on_state_change.
        if self.on_state_change is not None:
            self.on_state_change(state, message)
        else:
            print(f"[STATE] {state.value}: {message}")

    def _run_local(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a local command and capture stdout/stderr."""
        return subprocess.run(cmd, check=check, text=True, capture_output=True)

    def _run_local_stream(self, cmd: list[str]) -> subprocess.Popen:
        """Run a local long-lived command (e.g. ssh tunnel)."""
        return subprocess.Popen(cmd)

    def _ssh_hpc(self, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command on the HPC via SSH."""
        cmd = ["ssh", self.config.hpc_prefix, remote_cmd]
        return self._run_local(cmd, check=check)

    # --------------- steps ---------------

    def submit_policy_job(self) -> bool:
        """Call ./xarm_pipeline.sh serve-policy and parse the job ID."""
        self._emit_state(OrchestratorState.SUBMITTING_JOB, "Submitting policy server job via pipeline script...")
        try:
            result = self._run_local(
                [str(self.config.pipeline_script), "serve-policy"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            msg = f"Failed to submit job: {e.stderr or e.stdout}"
            self._emit_state(OrchestratorState.ERROR_JOB_SUBMISSION, msg)
            return False

        # qsub prints something like: 15063933.aqua
        # We look for the last "word" that matches that pattern.
        stdout = result.stdout.strip()
        job_id = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.endswith(".aqua"):
                job_id = line
        self.job_id = job_id

        self._emit_state(
            OrchestratorState.JOB_QUEUED,
            f"Job submitted. Job ID: {self.job_id or 'UNKNOWN'}; waiting for endpoint file...",
        )
        return True

    def wait_for_endpoint(self) -> bool:
        """Poll the HPC for ~/.openpi_policy_endpoint until it appears or timeout."""
        self._emit_state(
            OrchestratorState.JOB_RUNNING_STARTING_SERVER,
            "Waiting for server to start and publish endpoint on HPC...",
        )

        deadline = time.time() + self.config.wait_endpoint_timeout_s

        while time.time() < deadline:
            try:
                # Read the endpoint file on HPC (or empty string if missing)
                result = self._ssh_hpc(
                    f"cat {self.config.endpoint_file} 2>/dev/null || echo ''",
                    check=True,
                )
                content = result.stdout.strip()
            except subprocess.CalledProcessError:
                content = ""

            if content:
                # Parse HOST/IP/PORT from file
                host = None
                ip = None
                port = None
                for line in content.splitlines():
                    if line.startswith("HOST="):
                        host = line.split("=", 1)[1].strip()
                    elif line.startswith("IP="):
                        ip = line.split("=", 1)[1].strip()
                    elif line.startswith("PORT="):
                        port = int(line.split("=", 1)[1].strip())

                if ip and port:
                    self.remote_ip = ip
                    self.remote_port = port
                    self._emit_state(
                        OrchestratorState.SERVER_READY_NO_TUNNEL,
                        f"Server reported on node {host} with IP {ip}:{port}",
                    )
                    return True

            time.sleep(self.config.wait_endpoint_poll_s)

        # Timeout: try to show some recent logs
        msg = f"Timed out waiting for endpoint file ({self.config.wait_endpoint_timeout_s}s)."
        if self.job_id:
            try:
                logs = self._ssh_hpc(
                    f"qpeek {self.job_id} 2>/dev/null || tail -n 50 ~/openpi_cmd.o{self.job_id.split('.')[0]}",
                    check=False,
                ).stdout
                msg += f"\nLast job logs:\n{logs}"
            except Exception as e:
                msg += f"\n(Also failed to fetch logs: {e})"

        self._emit_state(OrchestratorState.ERROR_JOB_STARTUP, msg)
        return False

    def start_tunnel(self) -> bool:
        """Start an SSH tunnel from local -> remote_ip:remote_port."""
        if not self.remote_ip or not self.remote_port:
            self._emit_state(
                OrchestratorState.ERROR_TUNNEL_START,
                "Cannot start tunnel: remote IP/port unknown. Did the endpoint file get created?",
            )
            return False

        self._emit_state(
            OrchestratorState.STARTING_TUNNEL,
            f"Starting SSH tunnel localhost:{self.remote_port} -> {self.remote_ip}:{self.remote_port} via {self.config.hpc_prefix}...",
        )

        cmd = [
            "ssh",
            "-N",  # no remote command; just forward
            "-L",
            f"{self.remote_port}:{self.remote_ip}:{self.remote_port}",
            self.config.hpc_prefix,
        ]

        try:
            proc = self._run_local_stream(cmd)
        except OSError as e:
            self._emit_state(
                OrchestratorState.ERROR_TUNNEL_START,
                f"Failed to start tunnel: {e}",
            )
            return False

        self.tunnel_process = proc
        # We don't block here; process runs in background as long as this Python process lives.
        return True

    def check_policy_health(self) -> bool:
        """Check WebSocket health by connecting to ws://localhost:policy_port."""
        self._emit_state(
            OrchestratorState.CHECKING_POLICY_HEALTH,
            f"Checking WebSocket health on ws://localhost:{self.config.policy_port}...",
        )

        uri = f"ws://localhost:{self.config.policy_port}"
        try:
            ws = websockets.sync.client.connect(
                uri,
                open_timeout=self.config.ws_health_timeout_s,
                close_timeout=self.config.ws_health_timeout_s,
            )
            ws.close()
            self._emit_state(
                OrchestratorState.READY,
                "Policy server and tunnel are healthy (WebSocket handshake succeeded).",
            )
            return True
        except OSError as e:
            # Connection refused / no listener
            self._emit_state(
                OrchestratorState.ERROR_TUNNEL_BROKEN,
                f"Tunnel or listener appears down: {e}",
            )
            return False
        except Exception as e:
            # Tunnel OK, but not speaking WebSocket properly or server unhealthy
            self._emit_state(
                OrchestratorState.ERROR_SERVER_UNHEALTHY,
                f"Connected but WebSocket handshake failed: {e}",
            )
            return False

    # --------------- high-level orchestration ---------------

    def run_full_sequence(self) -> None:
        """
        Full sequence for a “Run server” button:

        1. Submit job (serve-policy) via pipeline script
        2. Wait for endpoint file on HPC
        3. Start SSH tunnel
        4. Check WebSocket health
        """
        self._emit_state(OrchestratorState.IDLE, "Starting orchestration sequence...")

        if not self.submit_policy_job():
            return

        if not self.wait_for_endpoint():
            return

        if not self.start_tunnel():
            return

        # Give the tunnel a moment to come up
        time.sleep(2.0)

        self.check_policy_health()


# ============================
# Simple CLI interface
# ============================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="XArm policy server orchestrator backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Run the full sequence: submit job, wait, tunnel, health-check")
    subparsers.add_parser("health", help="Just run the WebSocket health check against localhost")

    args = parser.parse_args()

    orch = Orchestrator()

    if args.command == "run":
        orch.run_full_sequence()
    elif args.command == "health":
        orch.check_policy_health()


if __name__ == "__main__":
    main()
