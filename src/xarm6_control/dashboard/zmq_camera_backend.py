#!/usr/bin/env python3
"""
Non-blocking ZMQ camera backend for the FastAPI dashboard.

- Connects to ZMQServerCamera (REQ/REP) via ZMQClientCamera
- Runs a background thread that polls frames at a target FPS
- Keeps the latest frame as a JPEG in memory
- Provides an MJPEG generator for FastAPI StreamingResponse

Usage example (in dashboard_app.py):

    from zmq_camera_backend import (
        ZmqCameraBackend,
        mjpeg_stream_generator,
    )

    base_backend = ZmqCameraBackend(host="127.0.0.1", port=5000, name="base")
    wrist_backend = ZmqCameraBackend(host="127.0.0.1", port=5001, name="wrist")

    @app.get("/video/base")
    def video_base():
        return StreamingResponse(
            mjpeg_stream_generator(base_backend),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/video/wrist")
    def video_wrist():
        return StreamingResponse(
            mjpeg_stream_generator(wrist_backend),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
"""

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
        """
        Args:
            host: ZMQ server host (where ZMQServerCamera is running).
            port: ZMQ server port.
            img_size: Optional (width, height) passed to camera.read(img_size).
            name: For logging / debugging.
            target_fps: How often to poll frames in the background thread.
        """
        self.host = host
        self.port = port
        self.img_size = img_size
        self.name = name
        self.target_fps = target_fps

        self._client = ZMQClientCamera(port=port, host=host)

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread (idempotent)."""
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
        """Signal the background thread to stop (non-blocking)."""
        self._stop_event.set()
        # Do not join here to keep it non-blocking; it’s a daemon thread anyway.

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Background loop that pulls frames at approx target_fps."""
        period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                # Your client: image, depth = camera.read()
                image, depth = self._client.read(self.img_size)

                if image is None:
                    raise RuntimeError("Camera returned None image")

                image = np.asarray(image)
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                # Your other client does image[:, :, ::-1] => image is RGB here
                if image.ndim == 3 and image.shape[2] == 3:
                    bgr = image[:, :, ::-1]  # RGB -> BGR for OpenCV
                else:
                    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                ok, jpeg = cv2.imencode(".jpg", bgr)
                if ok:
                    with self._lock:
                        self._latest_jpeg = jpeg.tobytes()
            except Exception as e:
                print(f"[ZmqCameraBackend:{self.name}] error: {e}")

            # Simple FPS pacing
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
        Return the latest JPEG frame if available, otherwise a black frame.
        Non-blocking.
        """
        with self._lock:
            jpeg = self._latest_jpeg

        if jpeg is not None:
            return jpeg

        w, h = fallback_size
        return self._black_frame(width=w, height=h)


def mjpeg_stream_generator(
    backend: ZmqCameraBackend,
    boundary: bytes = b"--frame",
    fallback_size: Tuple[int, int] = (640, 480),
) -> Generator[bytes, None, None]:
    """
    Generator suitable for FastAPI StreamingResponse with
    media_type="multipart/x-mixed-replace; boundary=frame".

    It:
      - ensures the backend thread is started,
      - repeatedly yields the most recent JPEG frame.
    """
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
