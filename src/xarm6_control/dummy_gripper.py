# dummy_gripper_client.py
# A drop-in fake gripper client with the same sync interface you call in XArmRealEnv.
# It returns "0".."255" as a string (like your TCP server) and accepts [0.0..1.0] commands.

import time
import threading
from typing import Optional

class GripperClientAsync:
    """
    Dummy gripper client to replace the real async TCP client.
    Methods provided (sync, to match your current usage):
      - send_gripper_command(value: float) -> None
      - receive_gripper_position() -> str  # "0".."255"
      - close() -> None

    Tunables:
      - slew_rate_255:  max change per second in 0..255 space (simulates motion speed)
      - latency_s:      artificial one-way latency (simulates network/driver delay)
      - noise_255:      adds ±noise to the reported position (0 = none)
      - start_norm:     initial position in [0,1]
      - verbose:        print set/get logs
    """
    def __init__(
        self,
        *,
        slew_rate_255: int = 9999,
        latency_s: float = 0.0,
        noise_255: int = 0,
        start_norm: float = 0.0,
        verbose: bool = False,
    ):
        start_norm = float(max(0.0, min(1.0, start_norm)))
        self._target = int(round(start_norm * 255))
        self._pos = self._target
        self._slew = max(1, int(slew_rate_255))
        self._lat = float(latency_s)
        self._noise = int(max(0, noise_255))
        self._verbose = verbose

        self._lock = threading.Lock()
        self._last_update = time.monotonic()

    # --- internal helpers ---
    def _step(self) -> None:
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        max_step = int(self._slew * dt)
        if max_step <= 0:
            return
        with self._lock:
            if self._pos < self._target:
                self._pos = min(self._target, self._pos + max_step)
            elif self._pos > self._target:
                self._pos = max(self._target, self._pos - max_step)

    # --- public API (sync) ---
    def send_gripper_command(self, value: float) -> None:
        """Accepts normalized [0,1], updates target (simulated motion toward it)."""
        if self._lat:
            time.sleep(self._lat)
        v = max(0.0, min(1.0, float(value)))
        t = int(round(v * 255))
        with self._lock:
            self._target = t
        if self._verbose:
            print(f"[DUMMY GRIPPER] SET {v:.3f} -> {t}")

    def receive_gripper_position(self) -> str:
        """Returns current simulated position as a stringified 0..255 (to match your parser)."""
        self._step()
        if self._lat:
            time.sleep(self._lat)
        val = self._pos
        if self._noise:
            import random
            val = max(0, min(255, val + random.randint(-self._noise, self._noise)))
        if self._verbose:
            print(f"[DUMMY GRIPPER] GET -> {val}")
        return str(val)

    def close(self) -> None:
        pass


# Optional: also provide a simple synchronous GripperClient alias for convenience.
class GripperClient(GripperClientAsync):
    pass


if __name__ == "__main__":
    # Tiny smoke test: 20% -> 0% -> 100% -> 200/255 -> 200/255 again
    g = GripperClientAsync(slew_rate_255=200, latency_s=0.0, start_norm=0.0, verbose=True)
    for cmd in [0.20, 0.0, 1.0, 200/255.0, 200/255.0]:
        g.send_gripper_command(cmd)
        for _ in range(5):
            print("Reported:", g.receive_gripper_position())
            time.sleep(0.2)
