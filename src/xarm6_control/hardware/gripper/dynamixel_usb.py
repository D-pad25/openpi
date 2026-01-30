#!/usr/bin/env python3
"""
Direct USB control of a Dynamixel gripper servo using dynamixel_sdk.

Key fix (why your USB barely moved near grasp):
- Your policy/environment produces a normalized gripper command (0..1).
- Your ROS/Teensy pipeline effectively uses a CALIBRATED physical range (e.g. open=5°, close=179°).
- If USB treats the Teensy "0..255" command as literal degrees across the whole servo range,
  the gripper saturates early and then "barely moves" for fine adjustments near closing.

This implementation keeps the same external "0..255" command domain, but maps it to your
real physical open/close angles before converting to Dynamixel ticks.

Mapping:
  cmd255 in [0..255]  -->  physical_deg in [open_deg..close_deg]
  physical_deg in [0..300] --> ticks in [0..1023] (AX/Protocol 1.0 approx)

Defaults assume your Teensy values:
  open_deg=5.0, close_deg=179.0
Change these if your gripper needs different angles.
"""

import time
from typing import Optional

try:
    from dynamixel_sdk import (
        PortHandler,
        PacketHandler,
        COMM_SUCCESS,
    )
except ImportError as e:
    raise ImportError(
        "dynamixel-sdk not installed. Install with:\n"
        "  pip install dynamixel-sdk pyserial\n"
        "USB gripper mode requires dynamixel-sdk.\n"
    ) from e


class DynamixelUSBGripper:
    """
    Direct USB control of a Dynamixel servo (AX / Protocol 1.0 style control table).
    Drop-in compatible shape for your existing gripper client expectations.
    """

    # AX / Protocol 1.0 Control Table addresses
    ADDR_TORQUE_ENABLE = 24
    ADDR_GOAL_POSITION = 30
    ADDR_MOVING_SPEED = 32
    ADDR_PRESENT_POSITION = 36

    # Protocol version
    DXL_PROTOCOL_VERSION = 1.0

    # AX position mode mapping: 0..300deg => 0..1023 ticks
    DXL_FULL_SCALE_DEG = 300.0
    DXL_FULL_SCALE_TICKS = 1023

    # Teensy-style command domain
    CMD_MIN = 0.0
    CMD_MAX = 255.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        dxl_id: int = 1,
        *,
        open_deg: float = 5.0,
        close_deg: float = 179.0,
        moving_speed: Optional[int] = None,
    ):
        """
        Args:
            port: Serial port (Linux: /dev/ttyUSB0, Windows: COM3, etc.)
            baudrate: 57600 for your setup
            dxl_id: servo ID (default 1)
            open_deg: physical degrees that correspond to cmd255=0 (open)
            close_deg: physical degrees that correspond to cmd255=255 (close)
            moving_speed: optional AX Moving Speed (0..1023). If None, do not change.
                          Note: On AX, 0 often means "max speed".
        """
        self.port_name = str(port)
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)

        self.open_deg = float(open_deg)
        self.close_deg = float(close_deg)
        if self.close_deg <= self.open_deg:
            raise ValueError("close_deg must be > open_deg")

        self.moving_speed = None if moving_speed is None else int(moving_speed)
        if self.moving_speed is not None and not (0 <= self.moving_speed <= 1023):
            raise ValueError("moving_speed must be in [0, 1023]")

        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)

        self._is_connected = False
        self._last_position = 0.0  # normalized [0..1]

    # ---------------- Connection ----------------
    def connect(self):
        if self._is_connected:
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    return
            except Exception:
                pass
            self._is_connected = False

        # best-effort close
        try:
            if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                self.port_handler.closePort()
        except Exception:
            pass

        time.sleep(0.1)

        # open (retry once)
        if not self.port_handler.openPort():
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            time.sleep(0.2)
            self.port_handler = PortHandler(self.port_name)
            if not self.port_handler.openPort():
                raise RuntimeError(f"Failed to open port {self.port_name}")

        if not self.port_handler.setBaudRate(self.baudrate):
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"Failed to set baudrate to {self.baudrate}")

        # enable torque
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 1
        )
        if dxl_comm_result != COMM_SUCCESS:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"Torque enable failed: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        if dxl_error != 0:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}")

        # optional moving speed (DON'T touch unless user asked / you set it)
        if self.moving_speed is not None:
            dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                self.port_handler, self.dxl_id, self.ADDR_MOVING_SPEED, self.moving_speed
            )
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError(f"Set moving speed failed: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
            if dxl_error != 0:
                raise RuntimeError(f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}")

        self._is_connected = True
        time.sleep(0.1)
        print(f"[DynamixelUSB] Connected: id={self.dxl_id} port={self.port_name} baud={self.baudrate}")

    def disconnect(self):
        # disable torque (best effort)
        if self._is_connected:
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 0
                    )
            except Exception:
                pass

        # close port with retries
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
                time.sleep(0.1)

        self._is_connected = False
        print(f"[DynamixelUSB] Disconnected: port={self.port_name}")

    # ---------------- Mapping (the important part) ----------------
    def _clamp_cmd255(self, cmd255: float) -> float:
        return max(self.CMD_MIN, min(self.CMD_MAX, float(cmd255)))

    def _cmd255_to_physical_deg(self, cmd255: float) -> float:
        """
        Map Teensy command domain [0..255] to your physical range [open_deg..close_deg].
        """
        c = self._clamp_cmd255(cmd255)
        u = (c - self.CMD_MIN) / (self.CMD_MAX - self.CMD_MIN)  # 0..1
        return self.open_deg + u * (self.close_deg - self.open_deg)

    def _physical_deg_to_cmd255(self, deg: float) -> float:
        """
        Map physical degrees [open_deg..close_deg] back into [0..255].
        """
        d = max(self.open_deg, min(self.close_deg, float(deg)))
        u = (d - self.open_deg) / (self.close_deg - self.open_deg)  # 0..1
        return self.CMD_MIN + u * (self.CMD_MAX - self.CMD_MIN)

    def _physical_deg_to_ticks(self, deg: float) -> int:
        """
        AX position mode: 0..300deg -> 0..1023 ticks.
        """
        d = max(0.0, min(self.DXL_FULL_SCALE_DEG, float(deg)))
        ticks = int(round((d / self.DXL_FULL_SCALE_DEG) * self.DXL_FULL_SCALE_TICKS))
        return max(0, min(self.DXL_FULL_SCALE_TICKS, ticks))

    def _ticks_to_physical_deg(self, ticks: int) -> float:
        t = max(0, min(self.DXL_FULL_SCALE_TICKS, int(ticks)))
        return (t / self.DXL_FULL_SCALE_TICKS) * self.DXL_FULL_SCALE_DEG

    def _normalized_to_cmd255(self, normalized: float) -> float:
        n = max(0.0, min(1.0, float(normalized)))
        return self.CMD_MIN + n * (self.CMD_MAX - self.CMD_MIN)

    def _cmd255_to_normalized(self, cmd255: float) -> float:
        c = self._clamp_cmd255(cmd255)
        return (c - self.CMD_MIN) / (self.CMD_MAX - self.CMD_MIN)

    # ---------------- Public API ----------------
    def set_degrees(self, cmd255: float, max_retries: int = 2) -> dict:
        """
        Set using Teensy-style 0..255 command.
        """
        return self.set_angle_deg(cmd255, max_retries=max_retries)

    def set_angle_deg(self, cmd255: float, max_retries: int = 2) -> dict:
        """
        Set gripper using Teensy-style command domain (0..255).
        Internally maps into physical degrees [open_deg..close_deg] then to ticks.
        """
        if not self._is_connected:
            self.connect()

        cmd255 = self._clamp_cmd255(cmd255)
        physical_deg = self._cmd255_to_physical_deg(cmd255)
        dxl_ticks = self._physical_deg_to_ticks(physical_deg)

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_GOAL_POSITION, dxl_ticks
                )
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    normalized = self._cmd255_to_normalized(cmd255)
                    self._last_position = normalized
                    return {
                        "ok": True,
                        "position": normalized,
                        "cmd255": cmd255,
                        "physical_deg": physical_deg,
                        "dxl_position": dxl_ticks,
                    }
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                else:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"

            if attempt < int(max_retries) - 1:
                time.sleep(0.05)

        raise RuntimeError(f"Failed to set after {max_retries} attempts: {last_error}")

    def set(self, normalized_value: float, max_retries: int = 2) -> dict:
        """
        Set using normalized [0..1], consistent with your policy output.
        """
        cmd255 = self._normalized_to_cmd255(normalized_value)
        return self.set_angle_deg(cmd255, max_retries=max_retries)

    def get(self, max_retries: int = 3) -> dict:
        """
        Read present position and return normalized [0..1] consistent with cmd255 mapping.
        """
        if not self._is_connected:
            self.connect()

        time.sleep(0.01)

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                dxl_present_ticks, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_PRESENT_POSITION
                )
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    physical_deg = self._ticks_to_physical_deg(dxl_present_ticks)
                    cmd255 = self._physical_deg_to_cmd255(physical_deg)
                    normalized = self._cmd255_to_normalized(cmd255)
                    self._last_position = normalized
                    return {
                        "ok": True,
                        "position": normalized,
                        "cmd255": cmd255,
                        "physical_deg": physical_deg,
                        "dxl_position": dxl_present_ticks,
                    }
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                else:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"

            if attempt < int(max_retries) - 1:
                time.sleep(0.05 * (attempt + 1))

        # fallback to cached
        return {
            "ok": False,
            "position": self._last_position,
            "cached": True,
            "error": last_error,
        }

    def ping(self, max_retries: int = 2) -> dict:
        if not self._is_connected:
            self.connect()

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                model, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.port_handler, self.dxl_id
                )
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    time.sleep(0.05)
                    return {"ok": True, "model": model}
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = self.packet_handler.getTxRxResult(dxl_comm_result)
                else:
                    last_error = self.packet_handler.getRxPacketError(dxl_error)
            except Exception as e:
                last_error = str(e)

            if attempt < int(max_retries) - 1:
                time.sleep(0.1)

        return {"ok": False, "error": last_error or "Unknown error"}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class GripperUSBClient:
    """
    Wrapper matching your GripperClient interface.
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
