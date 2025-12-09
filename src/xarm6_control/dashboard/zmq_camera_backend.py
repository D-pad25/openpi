# xarm6_control/dashboard/zmq_camera_backend.py

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Generator

import cv2
import numpy as np

from xarm6_control.zmq_core.camera_node import ZMQClientCamera


class ZmqCameraBackend:
    """
    Non-blocking backend that:
      - connects to a ZMQServerCamera via ZMQClientCamera,
      - runs a background thread to fetch frames at target_fps,
      - stores the latest JPEG in memory.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        img_size: Optional[Tuple[int, int]] = None,
        name: str = "camera",
        target_fps: float = 15.0,
    ) -> None:
        self.host = host
        self.port = port
        self.img_size = img_size
        self.name = name
        self.target_fps = target_fps

        self._client = ZMQClientCamera(port=port, host=host)

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._last_update: float = 0.0  # 👈 track freshness

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._consecutive_errors = 0              # 👈 error counter
        self._max_errors_before_reset = 5         # 👈 tune as needed

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"ZmqCameraBackend-{self.name}",
            daemon=True,
        )
        self._thread.start()
        print(f"[ZmqCameraBackend:{self.name}] started (host={self.host}, port={self.port})")

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _reset_client(self) -> None:
        """Close and recreate the ZMQ client socket."""
        print(f"[ZmqCameraBackend:{self.name}] resetting ZMQ client...")
        try:
            # If ZMQClientCamera has a close() method, call it.
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            print(f"[ZmqCameraBackend:{self.name}] error closing client: {e}")

        # Recreate client
        self._client = ZMQClientCamera(port=self.port, host=self.host)
        self._consecutive_errors = 0

    def _run_loop(self) -> None:
        period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                image, depth = self._client.read(self.img_size)

                if image is None:
                    raise RuntimeError("Camera returned None image")

                image = np.asarray(image)
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                if image.ndim == 3 and image.shape[2] == 3:
                    bgr = image[:, :, ::-1]  # RGB -> BGR
                else:
                    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                ok, jpeg = cv2.imencode(".jpg", bgr)
                if ok:
                    with self._lock:
                        self._latest_jpeg = jpeg.tobytes()
                        self._last_update = time.time()   # 👈 mark fresh
                self._consecutive_errors = 0           # 👈 reset errors on success

            except Exception as e:
                self._consecutive_errors += 1
                print(f"[ZmqCameraBackend:{self.name}] error #{self._consecutive_errors}: {e}")

                # After a few consecutive errors, nuke and recreate the client
                if self._consecutive_errors >= self._max_errors_before_reset:
                    self._reset_client()

            # FPS pacing
            if period > 0:
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)

    # ------------------------------------------------------------------
    # Public frame access
    # ------------------------------------------------------------------

    @staticmethod
    def _black_frame(width: int = 640, height: int = 480) -> bytes:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", img)
        if not ok:
            return b""
        return jpeg.tobytes()

    def get_latest_jpeg(self, fallback_size: Tuple[int, int] = (640, 480)) -> bytes:
        """
        Return the latest JPEG frame if available and recent, otherwise a black frame.
        """
        now = time.time()
        with self._lock:
            jpeg = self._latest_jpeg
            last = self._last_update

        # Consider frame "stale" if too old (e.g. > 2 seconds)
        if jpeg is not None and last and (now - last) < 2.0:
            return jpeg

        w, h = fallback_size
        return self._black_frame(width=w, height=h)


def mjpeg_stream_generator(
    backend: ZmqCameraBackend,
    boundary: bytes = b"--frame",
    fallback_size: Tuple[int, int] = (640, 480),
) -> Generator[bytes, None, None]:
    backend.start()  # idempotent

    while True:
        frame_bytes = backend.get_latest_jpeg(fallback_size=fallback_size)
        yield (
            boundary
            + b"\r\n"
            + b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )
