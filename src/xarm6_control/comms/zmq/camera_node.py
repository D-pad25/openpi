import os
import pickle
import threading
import time
from typing import Optional, Tuple

import numpy as np
import zmq

from xarm6_control.sensors.cameras.camera import CameraDriver

DEFAULT_CAMERA_PORT = 5000


class ZMQClientCamera(CameraDriver):
    """
    ZMQ-based camera client.

    Talks to ZMQServerCamera using REQ/REP and returns (image, depth).
    """

    def __init__(
        self,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
        timeout_ms: int = 1000,
    ):
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms

        # Use shared context
        self._context = zmq.Context.instance()
        self._socket = self._create_socket()

    def _create_socket(self) -> zmq.Socket:
        sock = self._context.socket(zmq.REQ)
        addr = f"tcp://{self._host}:{self._port}"
        sock.connect(addr)

        # Non-blocking-ish behaviour: time out instead of hanging forever
        sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        # Don't keep unsent messages around if we close/recreate
        sock.setsockopt(zmq.LINGER, 0)

        return sock

    def close(self) -> None:
        """Close the underlying ZMQ socket."""
        try:
            self._socket.close(0)
        except Exception:
            # Best-effort close; ignore errors
            pass

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Request a frame from the camera server.

        Returns:
            (image, depth) as numpy arrays.

        Raises:
            TimeoutError: if the server does not respond within timeout_ms.
            RuntimeError: for other ZMQ/serialization issues.
        """
        try:
            # Pack the desired image size (or None) and send it
            send_message = pickle.dumps(img_size)
            self._socket.send(send_message)

            # This will raise zmq.Again on timeout because of RCVTIMEO
            reply = self._socket.recv()
        except zmq.Again as e:
            raise TimeoutError(
                f"Timed out waiting for camera server at {self._host}:{self._port}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"ZMQClientCamera read error from {self._host}:{self._port}: {e}"
            ) from e

        camera_read = pickle.loads(reply)
        # We expect camera_read to be (image, depth) from RealSenseCamera.read()
        return camera_read


class ZMQServerCamera:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)

        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Sever Binding to {addr}, Camera: {camera}"
        print(debug_message)
        self._timeout_message = f"Timeout in Camera Server, Camera: {camera}"
        self._debug_timeouts = os.getenv("OPENPI_CAMERA_DEBUG") == "1"
        self._timeout_log_every_s = float(os.getenv("OPENPI_CAMERA_TIMEOUT_LOG_EVERY", "5.0"))
        self._last_timeout_log = 0.0

        # Time out waiting for client requests so we can check stop flag
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1000 ms
        self._socket.setsockopt(zmq.LINGER, 0)

        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve camera frames over ZMQ (REQ/REP)."""
        while not self._stop_event.is_set():
            try:
                # Wait for a request from client (may timeout)
                message = self._socket.recv()
            except zmq.Again:
                # No request within timeout; just loop and re-check stop flag
                if self._debug_timeouts:
                    now = time.time()
                    if (now - self._last_timeout_log) >= self._timeout_log_every_s:
                        print(self._timeout_message)
                        self._last_timeout_log = now
                continue
            except zmq.ZMQError as e:
                print(f"[ZMQServerCamera] ZMQ error receiving: {e}")
                break

            try:
                img_size = pickle.loads(message)
                camera_read = self._camera.read(img_size)
                self._socket.send(pickle.dumps(camera_read))
            except Exception as e:
                # If camera.read fails, log and send a dummy (None, None)
                print(f"[ZMQServerCamera] Error reading from camera: {e}")
                try:
                    self._socket.send(pickle.dumps((None, None)))
                except zmq.ZMQError as e2:
                    print(f"[ZMQServerCamera] Error sending error reply: {e2}")

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
