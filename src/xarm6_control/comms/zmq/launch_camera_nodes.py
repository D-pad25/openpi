from dataclasses import dataclass
from multiprocessing import Process
import os
import shutil
import signal
import socket
import subprocess
import time

import tyro

from xarm6_control.sensors.cameras.realsense_camera import RealSenseCamera, get_device_ids
from camera_node import ZMQServerCamera


@dataclass
class Args:
    hostname: str = "0.0.0.0"
    # hostname: str = "128.32.175.167"
    kill_existing: bool = True
    kill_timeout_s: float = 2.0
    kill_force: bool = True


def _port_in_use(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def _find_listening_pids(port: int) -> list[int]:
    pids: list[int] = []

    lsof = shutil.which("lsof")
    if lsof:
        res = subprocess.run(
            [lsof, "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            capture_output=True,
            check=False,
        )
        for line in (res.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        if pids:
            return pids

    # Try ss as a fallback (common on Linux)
    ss = shutil.which("ss")
    if ss:
        res = subprocess.run(
            [ss, "-lptn", f"sport = :{port}"],
            text=True,
            capture_output=True,
            check=False,
        )
        for line in (res.stdout or "").splitlines():
            # Example: users:(("python",pid=1234,fd=3))
            if "pid=" in line:
                for part in line.split("pid=")[1:]:
                    pid_str = ""
                    for ch in part:
                        if ch.isdigit():
                            pid_str += ch
                        else:
                            break
                    if pid_str:
                        pids.append(int(pid_str))
        if pids:
            return sorted(set(pids))

    # Try fuser as a last resort
    fuser = shutil.which("fuser")
    if fuser:
        res = subprocess.run(
            [fuser, "-n", "tcp", str(port)],
            text=True,
            capture_output=True,
            check=False,
        )
        # fuser output can be like: "5000/tcp: 1234 5678"
        tokens = (res.stdout or "").replace(":", " ").split()
        for tok in tokens:
            if tok.isdigit():
                pids.append(int(tok))
        if pids:
            return sorted(set(pids))

    return []


def _kill_pids(pids: list[int], timeout_s: float, force: bool) -> None:
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception as e:
            print(f"[WARN] Failed to SIGTERM pid {pid}: {e}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        alive = [pid for pid in pids if os.path.exists(f"/proc/{pid}")]
        if not alive:
            return
        time.sleep(0.1)

    if force:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except Exception as e:
                print(f"[WARN] Failed to SIGKILL pid {pid}: {e}")


def _ensure_port_free(host: str, port: int, args: Args) -> None:
    if not _port_in_use(host, port):
        return

    print(f"[WARN] Port {port} is already in use.")
    if not args.kill_existing:
        raise RuntimeError(f"Port {port} is in use. Stop the existing camera server and retry.")

    pids = _find_listening_pids(port)
    if not pids:
        raise RuntimeError(
            f"Port {port} is in use, but no listening PID could be found. "
            "Install `lsof`, `ss`, or `fuser`, or stop the process manually."
        )

    print(f"[INFO] Attempting to stop existing listener(s) on port {port}: {pids}")
    _kill_pids(pids, timeout_s=args.kill_timeout_s, force=args.kill_force)

    if _port_in_use(host, port):
        raise RuntimeError(f"Port {port} is still in use after attempting to stop {pids}.")


def launch_server(port: int, camera_id: int, args: Args):
    camera = RealSenseCamera(camera_id)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


def main(args):
    ids = get_device_ids()
    print(f"Found RealSense devices: {ids}", flush=True)
    if not ids:
        print(
            "ERROR: No RealSense devices found. "
            "Please check USB connections and power.",
            flush=True,
        )
        raise SystemExit(1)   # <<< important: non-zero exit code
    camera_port = 5000
    camera_servers = []
    for camera_id in ids:
        # start a python process for each camera
        _ensure_port_free(args.hostname, camera_port, args)
        print(f"Launching camera {camera_id} on port {camera_port}")
        camera_servers.append(Process(target=launch_server, args=(camera_port, camera_id, args)))
        camera_port += 1

    for server in camera_servers:
        server.start()

    def _terminate_children():
        for server in camera_servers:
            if server.is_alive():
                server.terminate()
        for server in camera_servers:
            server.join(timeout=2.0)

    def _signal_handler(signum, frame):
        print(f"[INFO] Received signal {signum}. Stopping camera servers...")
        _terminate_children()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Fail fast if any child exits immediately
    time.sleep(1.0)
    exited = [s for s in camera_servers if s.exitcode not in (None, 0)]
    if exited:
        codes = {s.pid: s.exitcode for s in exited}
        _terminate_children()
        raise SystemExit(f"Camera server failed to start: {codes}")

    # Keep the parent alive so the dashboard can manage the process group
    try:
        while True:
            time.sleep(1.0)
            dead = [s for s in camera_servers if s.exitcode not in (None, 0)]
            if dead:
                codes = {s.pid: s.exitcode for s in dead}
                _terminate_children()
                raise SystemExit(f"Camera server exited: {codes}")
    finally:
        _terminate_children()

if __name__ == "__main__":
    main(tyro.cli(Args))
