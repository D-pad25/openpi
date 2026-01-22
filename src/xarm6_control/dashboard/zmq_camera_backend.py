# xarm6_control/dashboard/zmq_camera_backend.py

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Generator, Dict, Any

import cv2
import numpy as np

from xarm6_control.comms.zmq.camera_node import ZMQClientCamera


class ZmqCameraBackend:
    """
    Non-blocking backend that:
      - connects to a ZMQServerCamera via ZMQClientCamera,
      - runs a background thread to fetch frames at target_fps,
      - stores the latest JPEG in memory,
      - tracks connection status and auto-reconnects.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        img_size: Optional[Tuple[int, int]] = None,
        name: str = "camera",
        target_fps: float = 15.0,
        stale_secs: float = 2.0,         # how old a frame can be before we call it "stale"
    ) -> None:
        self.host = host
        self.port = port
        self.img_size = img_size
        self.name = name
        self.target_fps = target_fps
        self.stale_secs = stale_secs

        self._client = ZMQClientCamera(port=port, host=host)

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._last_update: float = 0.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._request_close: bool = False

        self._consecutive_errors = 0
        self._max_errors_before_reset = 5

        # Status fields
        self._connected: bool = False
        self._last_error: Optional[str] = None
        self._last_error_time: float = 0.0

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

    def stop(self, *, wait: bool = True, timeout_s: float = 2.0) -> None:
        """Stop the background thread and optionally wait for it to exit."""
        self._request_close = True
        self._stop_event.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def reset(self) -> None:
        """Reset the ZMQ client connection and restart the polling thread."""
        self.stop(wait=True)
        self._reset_client()
        self.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_client(self) -> None:
        """Close and recreate the ZMQ client socket."""
        print(f"[ZmqCameraBackend:{self.name}] resetting ZMQ client...")
        try:
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as e:
            print(f"[ZmqCameraBackend:{self.name}] error closing client: {e}")

        self._client = ZMQClientCamera(port=self.port, host=self.host)
        self._consecutive_errors = 0

    def _mark_error(self, e: Exception) -> None:
        with self._lock:
            self._last_error = str(e)
            self._last_error_time = time.time()
            # When we hit an error, we pessimistically mark as disconnected.
            self._connected = False

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------

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
                    now = time.time()
                    with self._lock:
                        self._latest_jpeg = jpeg.tobytes()
                        self._last_update = now
                        self._connected = True
                        self._last_error = None

                self._consecutive_errors = 0

            except Exception as e:
                self._consecutive_errors += 1
                # print(f"[ZmqCameraBackend:{self.name}] error #{self._consecutive_errors}: {e}")
                self._mark_error(e)

                if self._consecutive_errors >= self._max_errors_before_reset:
                    self._reset_client()

            # FPS pacing
            if period > 0:
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        # Close client on thread exit to avoid cross-thread close issues
        if self._request_close:
            try:
                close_fn = getattr(self._client, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception as e:
                print(f"[ZmqCameraBackend:{self.name}] error closing client on exit: {e}")
        self._request_close = False

    # ------------------------------------------------------------------
    # Public frame + status access
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
            connected = self._connected

        # "Connected" only really means something if we also have a fresh frame.
        if connected and jpeg is not None and last and (now - last) < self.stale_secs:
            return jpeg

        w, h = fallback_size
        return self._black_frame(width=w, height=h)

    def get_status(self) -> Dict[str, Any]:
        """
        Return a snapshot of camera status for the API / UI.
        """
        now = time.time()
        with self._lock:
            last = self._last_update
            last_error = self._last_error
            last_error_time = self._last_error_time
            consecutive_errors = self._consecutive_errors
            connected = self._connected

        stale = True
        if last:
            stale = (now - last) > self.stale_secs

        # "online" == we think we're connected AND frames are not stale
        online = bool(connected and not stale)

        return {
            "online": online,
            "stale": stale,
            "last_update": last,
            "last_error": last_error,
            "last_error_time": last_error_time,
            "consecutive_errors": consecutive_errors,
        }


def mjpeg_stream_generator(
    backend: ZmqCameraBackend,
    boundary: bytes = b"--frame",
    fallback_size: Tuple[int, int] = (640, 480),
    target_fps: Optional[float] = None,
) -> Generator[bytes, None, None]:
    backend.start()  # idempotent

    # Avoid busy-looping: pace output roughly to the backend's intended FPS.
    fps = backend.target_fps if target_fps is None else target_fps
    period = (1.0 / fps) if fps and fps > 0 else 0.0

    try:
        while True:
            t0 = time.time()
            frame_bytes = backend.get_latest_jpeg(fallback_size=fallback_size)
            yield (
                boundary
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            if period > 0:
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
    except GeneratorExit:
        return
