#!/usr/bin/env python3
"""
dynamixel_usb.py

Robust USB control for an AX-series (Protocol 1.0) Dynamixel gripper using dynamixel_sdk.

Why this version is different (and why it often fixes "USB barely moves near grasp"):

1) It DOES NOT treat 0..255 as physical degrees.
   Instead, it maps your policy command (normalized 0..1 or cmd255 0..255)
   directly into a *tick range* [open_ticks .. close_ticks].

2) It can automatically derive that tick range from the servo itself:
   - Reads CW/CCW angle limits (ADDR 6, 8).
   - If those limits are configured (not 0..1023), we use them as endpoints.
     This often matches what the Teensy pipeline effectively enforces.

3) It forces "stiff" settings that the Teensy often sets:
   - Moving speed (ADDR 32) -> default max (0)
   - Torque limit (ADDR 34) -> max (1023)
   - Compliance margin/slope -> stiffer response (optional but enabled by default)
   These are common reasons why a gripper feels weak or unresponsive near closing.

4) It adds a small minimum tick step to overcome deadband/stiction under load.

If this still doesn't match ROS perfectly, the next step is to *dump the Teensy-configured
registers* (speed/torque/compliance/limits) and clone them. This file already covers the
most common ones for AX.

Requirements:
  pip install dynamixel-sdk pyserial
"""

import time
from typing import Optional, Tuple

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError as e:
    raise ImportError(
        "dynamixel-sdk not installed. Install with:\n"
        "  pip install dynamixel-sdk pyserial\n"
        "USB gripper mode requires dynamixel-sdk."
    ) from e


class DynamixelUSBGripper:
    """
    Direct USB control of an AX/Protocol 1.0 Dynamixel servo.
    Provides the same shape as your GripperClient: set/get/ping/disconnect.

    Public command interfaces:
      - set(normalized: 0..1)
      - set_degrees(cmd255: 0..255)  # Teensy-style
    """

    # ---------------- AX / Protocol 1.0 Control Table ----------------
    ADDR_CW_ANGLE_LIMIT = 6          # 2 bytes
    ADDR_CCW_ANGLE_LIMIT = 8         # 2 bytes

    ADDR_TORQUE_ENABLE = 24          # 1 byte

    ADDR_CW_COMPLIANCE_MARGIN = 26   # 1 byte
    ADDR_CCW_COMPLIANCE_MARGIN = 27  # 1 byte
    ADDR_CW_COMPLIANCE_SLOPE = 28    # 1 byte
    ADDR_CCW_COMPLIANCE_SLOPE = 29   # 1 byte

    ADDR_GOAL_POSITION = 30          # 2 bytes
    ADDR_MOVING_SPEED = 32           # 2 bytes
    ADDR_TORQUE_LIMIT = 34           # 2 bytes
    ADDR_PRESENT_POSITION = 36       # 2 bytes

    ADDR_MOVING = 46                 # 1 byte (0=not moving, 1=moving)

    DXL_PROTOCOL_VERSION = 1.0

    # Tick range for AX position mode
    TICKS_MIN = 0
    TICKS_MAX = 1023

    # Teensy-style command range
    CMD_MIN = 0.0
    CMD_MAX = 255.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        dxl_id: int = 1,
        *,
        # Preferred: explicitly set endpoints in ticks if you know them
        open_ticks: Optional[int] = None,
        close_ticks: Optional[int] = None,

        # If endpoints aren't provided, try to use servo angle limits as endpoints
        use_angle_limits_as_range: bool = True,

        # If angle limits are default (0..1023) and ticks aren't provided, fallback endpoints:
        fallback_open_ticks: int = 0,
        fallback_close_ticks: int = 650,  # conservative close; avoids hitting hard stop

        # Optional inversion (swap meaning of "open" and "close")
        invert: bool = False,

        # Make USB behave like a "stiff" gripper (common Teensy setup)
        set_max_torque_limit: bool = True,
        torque_limit: int = 1023,          # 0..1023 (AX)
        set_max_speed: bool = True,
        moving_speed: int = 0,             # AX: 0 often means "max"
        set_stiff_compliance: bool = True,
        compliance_margin: int = 0,        # smaller = tighter
        compliance_slope: int = 32,         # valid typical: 2,4,8,16,32,64,128

        # Helps when near-grasp commands are tiny and get swallowed by deadband
        min_step_ticks: int = 3,

        # Optional verification (slower but can help debugging)
        verify_move: bool = False,
        verify_timeout_s: float = 0.25,
        verify_tol_ticks: int = 6,
    ):
        self.port_name = str(port)
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)

        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)

        self._is_connected = False

        self.open_ticks_cfg = open_ticks
        self.close_ticks_cfg = close_ticks
        self.use_angle_limits_as_range = bool(use_angle_limits_as_range)
        self.fallback_open_ticks = int(fallback_open_ticks)
        self.fallback_close_ticks = int(fallback_close_ticks)
        self.invert = bool(invert)

        self.set_max_torque_limit = bool(set_max_torque_limit)
        self.torque_limit = int(torque_limit)
        self.set_max_speed = bool(set_max_speed)
        self.moving_speed = int(moving_speed)

        self.set_stiff_compliance = bool(set_stiff_compliance)
        self.compliance_margin = int(compliance_margin)
        self.compliance_slope = int(compliance_slope)

        self.min_step_ticks = int(min_step_ticks)

        self.verify_move = bool(verify_move)
        self.verify_timeout_s = float(verify_timeout_s)
        self.verify_tol_ticks = int(verify_tol_ticks)

        self._last_norm = 0.0
        self._last_target_ticks: Optional[int] = None

        # Resolved endpoints (ticks)
        self._open_ticks: Optional[int] = None
        self._close_ticks: Optional[int] = None

    # ---------------- Low-level helpers ----------------
    def _w1(self, addr: int, val: int) -> None:
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.dxl_id, addr, int(val)
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(dxl_comm_result))
        if dxl_error != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(dxl_error))

    def _w2(self, addr: int, val: int) -> None:
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, self.dxl_id, addr, int(val)
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(dxl_comm_result))
        if dxl_error != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(dxl_error))

    def _r1(self, addr: int) -> int:
        val, dxl_comm_result, dxl_error = self.packet_handler.read1ByteTxRx(
            self.port_handler, self.dxl_id, addr
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(dxl_comm_result))
        if dxl_error != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(dxl_error))
        return int(val)

    def _r2(self, addr: int) -> int:
        val, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
            self.port_handler, self.dxl_id, addr
        )
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(dxl_comm_result))
        if dxl_error != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(dxl_error))
        return int(val)

    # ---------------- Connection ----------------
    def connect(self) -> None:
        if self._is_connected:
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    return
            except Exception:
                pass
            self._is_connected = False

        # Close if needed
        try:
            if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                self.port_handler.closePort()
        except Exception:
            pass

        time.sleep(0.05)

        # Open (retry once)
        if not self.port_handler.openPort():
            time.sleep(0.15)
            self.port_handler = PortHandler(self.port_name)
            if not self.port_handler.openPort():
                raise RuntimeError(f"Failed to open port {self.port_name}")

        if not self.port_handler.setBaudRate(self.baudrate):
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on {self.port_name}")

        # Torque enable
        self._w1(self.ADDR_TORQUE_ENABLE, 1)

        # Apply "stiff" config (common difference vs Teensy)
        self._apply_profile()

        # Resolve tick range endpoints
        self._resolve_tick_range()

        self._is_connected = True
        time.sleep(0.05)
        print(
            f"[DynamixelUSB] Connected id={self.dxl_id} port={self.port_name} baud={self.baudrate} "
            f"range=open:{self._open_ticks} close:{self._close_ticks} invert={self.invert}"
        )

    def _apply_profile(self) -> None:
        # Note: all best-effort; some servos / configs may reject writes under load.
        # We still raise if a write fails, because silent misconfig = confusing behaviour.
        if self.set_max_speed:
            # AX: moving_speed 0 often means max speed
            self._w2(self.ADDR_MOVING_SPEED, max(0, min(1023, self.moving_speed)))

        if self.set_max_torque_limit:
            self._w2(self.ADDR_TORQUE_LIMIT, max(0, min(1023, self.torque_limit)))

        if self.set_stiff_compliance:
            # Keep response tight
            m = max(0, min(255, self.compliance_margin))
            s = max(0, min(255, self.compliance_slope))
            self._w1(self.ADDR_CW_COMPLIANCE_MARGIN, m)
            self._w1(self.ADDR_CCW_COMPLIANCE_MARGIN, m)
            self._w1(self.ADDR_CW_COMPLIANCE_SLOPE, s)
            self._w1(self.ADDR_CCW_COMPLIANCE_SLOPE, s)

    def disconnect(self) -> None:
        if self._is_connected:
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    try:
                        self._w1(self.ADDR_TORQUE_ENABLE, 0)
                    except Exception:
                        pass
            except Exception:
                pass

        for _ in range(3):
            try:
                if hasattr(self.port_handler, "is_open"):
                    if self.port_handler.is_open:
                        self.port_handler.closePort()
                    if not self.port_handler.is_open:
                        break
                else:
                    break
            except Exception:
                time.sleep(0.05)

        self._is_connected = False
        print(f"[DynamixelUSB] Disconnected port={self.port_name}")

    # ---------------- Range resolution ----------------
    def _resolve_tick_range(self) -> None:
        # 1) If user provided endpoints, use them
        if self.open_ticks_cfg is not None and self.close_ticks_cfg is not None:
            o = int(self.open_ticks_cfg)
            c = int(self.close_ticks_cfg)
            self._open_ticks, self._close_ticks = self._sanitize_range(o, c)
            return

        # 2) Try to use servo CW/CCW limits as endpoints (often matches Teensy behaviour)
        if self.use_angle_limits_as_range:
            cw = self._r2(self.ADDR_CW_ANGLE_LIMIT)
            ccw = self._r2(self.ADDR_CCW_ANGLE_LIMIT)

            # Wheel mode check (both 0 in many AX configs)
            if cw == 0 and ccw == 0:
                raise RuntimeError(
                    "Servo appears to be in WHEEL mode (CW=0, CCW=0). "
                    "Position control won't behave as expected. "
                    "If ROS works, this suggests your USB path is not talking to the same config/servo."
                )

            # If limits aren't the full default range, treat them as calibrated endpoints
            if not (cw == 0 and ccw == 1023):
                self._open_ticks, self._close_ticks = self._sanitize_range(cw, ccw)
                return

        # 3) Fallback to conservative endpoints
        self._open_ticks, self._close_ticks = self._sanitize_range(
            self.fallback_open_ticks, self.fallback_close_ticks
        )

    def _sanitize_range(self, open_ticks: int, close_ticks: int) -> Tuple[int, int]:
        o = max(self.TICKS_MIN, min(self.TICKS_MAX, int(open_ticks)))
        c = max(self.TICKS_MIN, min(self.TICKS_MAX, int(close_ticks)))
        if o == c:
            # Expand slightly to avoid divide-by-zero and "no motion"
            c = max(self.TICKS_MIN, min(self.TICKS_MAX, o + 10))
        return o, c

    # ---------------- Mapping ----------------
    def _norm_to_ticks(self, normalized: float) -> int:
        n = max(0.0, min(1.0, float(normalized)))
        o, c = self._open_ticks, self._close_ticks
        assert o is not None and c is not None

        if self.invert:
            n = 1.0 - n

        ticks = int(round(o + n * (c - o)))
        return max(self.TICKS_MIN, min(self.TICKS_MAX, ticks))

    def _ticks_to_norm(self, ticks: int) -> float:
        t = max(self.TICKS_MIN, min(self.TICKS_MAX, int(ticks)))
        o, c = self._open_ticks, self._close_ticks
        assert o is not None and c is not None

        # Avoid division by zero
        if c == o:
            return 0.0

        n = (t - o) / float(c - o)
        n = max(0.0, min(1.0, n))
        if self.invert:
            n = 1.0 - n
        return n

    def _cmd255_to_norm(self, cmd255: float) -> float:
        c = max(self.CMD_MIN, min(self.CMD_MAX, float(cmd255)))
        return (c - self.CMD_MIN) / (self.CMD_MAX - self.CMD_MIN)

    # ---------------- Motion helpers ----------------
    def _apply_min_step(self, target_ticks: int) -> int:
        if self._last_target_ticks is None:
            return target_ticks

        dt = target_ticks - self._last_target_ticks
        if abs(dt) < self.min_step_ticks:
            if dt == 0:
                return target_ticks
            bump = self.min_step_ticks if dt > 0 else -self.min_step_ticks
            target_ticks = self._last_target_ticks + bump
            target_ticks = max(self.TICKS_MIN, min(self.TICKS_MAX, target_ticks))
        return target_ticks

    def _wait_reached(self, target_ticks: int, timeout_s: float, tol_ticks: int) -> None:
        start = time.time()
        while (time.time() - start) < timeout_s:
            try:
                cur = self._r2(self.ADDR_PRESENT_POSITION)
                if abs(cur - target_ticks) <= tol_ticks:
                    return
            except Exception:
                pass
            time.sleep(0.01)

    # ---------------- Public API ----------------
    def ping(self, max_retries: int = 2) -> dict:
        if not self._is_connected:
            self.connect()

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                model, comm_result, err = self.packet_handler.ping(self.port_handler, self.dxl_id)
                if comm_result == COMM_SUCCESS and err == 0:
                    return {"ok": True, "model": int(model)}
                if comm_result != COMM_SUCCESS:
                    last_error = self.packet_handler.getTxRxResult(comm_result)
                else:
                    last_error = self.packet_handler.getRxPacketError(err)
            except Exception as e:
                last_error = str(e)

            if attempt < int(max_retries) - 1:
                time.sleep(0.05)

        return {"ok": False, "error": last_error or "Unknown error"}

    def set(self, normalized_value: float, max_retries: int = 2) -> dict:
        if not self._is_connected:
            self.connect()

        n = max(0.0, min(1.0, float(normalized_value)))
        target_ticks = self._norm_to_ticks(n)
        target_ticks = self._apply_min_step(target_ticks)

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                self._w2(self.ADDR_GOAL_POSITION, target_ticks)
                self._last_target_ticks = target_ticks
                self._last_norm = n

                if self.verify_move:
                    self._wait_reached(target_ticks, self.verify_timeout_s, self.verify_tol_ticks)

                return {"ok": True, "position": n, "dxl_position": target_ticks}
            except Exception as e:
                last_error = str(e)
                if attempt < int(max_retries) - 1:
                    time.sleep(0.02)

        raise RuntimeError(f"Failed to set() after {max_retries} attempts: {last_error}")

    def set_degrees(self, cmd255: float, max_retries: int = 2) -> dict:
        # cmd255 is Teensy-style 0..255, but we treat it as a normalized command.
        n = self._cmd255_to_norm(cmd255)
        out = self.set(n, max_retries=max_retries)
        out["cmd255"] = float(max(self.CMD_MIN, min(self.CMD_MAX, float(cmd255))))
        return out

    def get(self, max_retries: int = 3) -> dict:
        if not self._is_connected:
            self.connect()

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                ticks = self._r2(self.ADDR_PRESENT_POSITION)
                n = self._ticks_to_norm(ticks)
                self._last_norm = n
                return {"ok": True, "position": n, "dxl_position": ticks}
            except Exception as e:
                last_error = str(e)
                if attempt < int(max_retries) - 1:
                    time.sleep(0.03 * (attempt + 1))

        return {"ok": False, "position": self._last_norm, "cached": True, "error": last_error}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class GripperUSBClient:
    """
    Wrapper matching your existing GripperClient interface.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", **kwargs):
        self._gripper = DynamixelUSBGripper(port=port, **kwargs)

    def set(self, value: float) -> dict:
        return self._gripper.set(value)

    def set_degrees(self, cmd255: float) -> dict:
        return self._gripper.set_degrees(cmd255)

    def get(self) -> dict:
        return self._gripper.get()

    def ping(self) -> dict:
        return self._gripper.ping()

    def disconnect(self):
        self._gripper.disconnect()
