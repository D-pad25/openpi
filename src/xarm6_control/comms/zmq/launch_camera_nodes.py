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
    lsof = shutil.which("lsof")
    if not lsof:
        return []
    res = subprocess.run(
        [lsof, "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        text=True,
        capture_output=True,
        check=False,
    )
    if not res.stdout:
        return []
    pids = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


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
            "Install `lsof` or stop the process manually."
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
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
