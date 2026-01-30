#!/usr/bin/env python3
"""
dynamixel_usb.py

MX-106 (Protocol 1.0) USB gripper control that matches Dynamixel Wizard.

What you asked for:
- Send commands in "degrees" from 0..255
  0   = fully open
  255 = fully closed (as shown in your Wizard screenshot)

Important:
- MX-106 uses 0..4095 ticks for ~0..360 degrees (not 0..1023 for 0..300 like AX)
- This file converts degrees -> ticks correctly for MX using Resolution Divider.

Requirements:
  pip install dynamixel-sdk pyserial
"""

from __future__ import annotations

import argparse
import platform
import time
from dataclasses import dataclass
from typing import Optional

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError as e:
    raise ImportError(
        "dynamixel-sdk not installed. Install with:\n"
        "  pip install dynamixel-sdk pyserial\n"
    ) from e


# ---------------- Protocol 1.0 MX Control Table (common) ----------------
ADDR_MODEL_NUMBER = 0            # 2 bytes
ADDR_FIRMWARE_VERSION = 2        # 1 byte
ADDR_ID = 3                      # 1 byte
ADDR_BAUD_RATE = 4               # 1 byte
ADDR_RETURN_DELAY = 5            # 1 byte

ADDR_CW_ANGLE_LIMIT = 6          # 2 bytes
ADDR_CCW_ANGLE_LIMIT = 8         # 2 bytes

ADDR_RESOLUTION_DIVIDER = 22     # 1 byte (MX only)

ADDR_TORQUE_ENABLE = 24          # 1 byte
ADDR_GOAL_POSITION = 30          # 2 bytes
ADDR_PRESENT_POSITION = 36       # 2 bytes

# Optional (not required, but handy for debug)
ADDR_MOVING_SPEED = 32           # 2 bytes
ADDR_TORQUE_LIMIT = 34           # 2 bytes


@dataclass
class GripperState:
    ok: bool
    position: float                    # normalized [0..1] over 0..255 command space
    cmd_deg: Optional[float] = None    # 0..255 command degrees (your desired interface)
    angle_deg: Optional[float] = None  # actual servo angle in degrees (0..~360)
    dxl_position: Optional[int] = None
    cached: bool = False
    error: Optional[str] = None


class DynamixelUSBGripper:
    """
    USB controller for MX-series Dynamixel in Protocol 1.0.

    Public API:
      - set_degrees(cmd_deg_0_to_255)  -> sends degrees (0..255) as you requested
      - set(norm_0_to_1)              -> maps to 0..255 degrees then sends
      - get()                         -> returns cmd_deg + actual angle_deg + ticks
      - ping()
      - disconnect()

    Notes:
      - Uses Resolution Divider to compute ticks-per-rev correctly.
      - Clamps goal ticks to CW/CCW angle limits.
    """

    DXL_PROTOCOL_VERSION = 1.0

    # MX nominal full scale:
    # - with resolution divider = 1: 4096 units per rev (0..4095)
    # We'll compute this dynamically from Resolution Divider.
    BASE_UNITS_PER_REV = 4096
    BASE_FULL_SCALE_DEG = 360.0

    # Your desired command domain
    CMD_MIN_DEG = 0.0
    CMD_MAX_DEG = 255.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        dxl_id: int = 1,
        *,
        # If direction is flipped, set this True
        invert: bool = False,
        # Small delay after writes (some USB adapters/servos behave better)
        post_write_sleep_s: float = 0.00,
        # Print register snapshot on connect
        verbose: bool = True,
    ):
        self.port_name = str(port)
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)
        self.invert = bool(invert)
        self.post_write_sleep_s = float(post_write_sleep_s)
        self.verbose = bool(verbose)

        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)

        self._is_connected = False

        # Resolved at connect()
        self.model_number: Optional[int] = None
        self.resolution_divider: int = 1
        self.units_per_rev: int = self.BASE_UNITS_PER_REV
        self.ticks_max: int = self.units_per_rev - 1
        self.cw_limit: int = 0
        self.ccw_limit: int = self.ticks_max

        self._last_norm = 0.0

    # ---------------- low-level IO ----------------
    def _w1(self, addr: int, val: int) -> None:
        comm, err = self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, addr, int(val))
        if comm != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(err))

    def _w2(self, addr: int, val: int) -> None:
        comm, err = self.packet_handler.write2ByteTxRx(self.port_handler, self.dxl_id, addr, int(val))
        if comm != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(err))

    def _r1(self, addr: int) -> int:
        val, comm, err = self.packet_handler.read1ByteTxRx(self.port_handler, self.dxl_id, addr)
        if comm != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(err))
        return int(val)

    def _r2(self, addr: int) -> int:
        val, comm, err = self.packet_handler.read2ByteTxRx(self.port_handler, self.dxl_id, addr)
        if comm != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(err))
        return int(val)

    # ---------------- conversion ----------------
    def _clamp_cmd_deg(self, deg: float) -> float:
        return max(self.CMD_MIN_DEG, min(self.CMD_MAX_DEG, float(deg)))

    def _cmd_deg_to_norm(self, deg: float) -> float:
        d = self._clamp_cmd_deg(deg)
        return (d - self.CMD_MIN_DEG) / (self.CMD_MAX_DEG - self.CMD_MIN_DEG)

    def _norm_to_cmd_deg(self, n: float) -> float:
        n = max(0.0, min(1.0, float(n)))
        return self.CMD_MIN_DEG + n * (self.CMD_MAX_DEG - self.CMD_MIN_DEG)

    def _deg_to_ticks(self, deg: float) -> int:
        """
        Convert *actual servo degrees* (0..360) to ticks (0..ticks_max),
        then clamp to CW/CCW limits.
        """
        d = float(deg)

        # Optional invert (swap direction)
        if self.invert:
            d = self.BASE_FULL_SCALE_DEG - d

        # Clamp to [0..360]
        d = max(0.0, min(self.BASE_FULL_SCALE_DEG, d))

        # Convert degrees -> ticks (using computed ticks_max)
        ticks = int(round((d / self.BASE_FULL_SCALE_DEG) * self.ticks_max))

        # Clamp within configured limits (joint mode)
        lo = min(self.cw_limit, self.ccw_limit)
        hi = max(self.cw_limit, self.ccw_limit)
        ticks = max(lo, min(hi, ticks))
        return ticks

    def _ticks_to_deg(self, ticks: int) -> float:
        t = max(0, min(self.ticks_max, int(ticks)))
        deg = (t / float(self.ticks_max)) * self.BASE_FULL_SCALE_DEG
        if self.invert:
            deg = self.BASE_FULL_SCALE_DEG - deg
        return deg

    # ---------------- connection ----------------
    def connect(self) -> None:
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

        time.sleep(0.05)

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

        # Verify comms via ping (and read model)
        ping = self.ping()
        if not ping.get("ok"):
            raise RuntimeError(f"Ping failed: {ping}")

        # Read model + resolution divider (MX)
        self.model_number = self._r2(ADDR_MODEL_NUMBER)
        fw = self._r1(ADDR_FIRMWARE_VERSION)

        # Resolution Divider exists on MX; if read fails, we’ll default to 1
        try:
            div = self._r1(ADDR_RESOLUTION_DIVIDER)
            if div <= 0:
                div = 1
            self.resolution_divider = int(div)
        except Exception:
            self.resolution_divider = 1

        # Compute units/ticks
        # units_per_rev = 4096 / divider (divider=1 => 4096, divider=2 => 2048, etc.)
        upr = int(self.BASE_UNITS_PER_REV // self.resolution_divider)
        if upr <= 0:
            upr = self.BASE_UNITS_PER_REV
        self.units_per_rev = upr
        self.ticks_max = self.units_per_rev - 1

        # Angle limits (joint mode)
        self.cw_limit = self._r2(ADDR_CW_ANGLE_LIMIT)
        self.ccw_limit = self._r2(ADDR_CCW_ANGLE_LIMIT)

        # Wheel mode check
        if self.cw_limit == 0 and self.ccw_limit == 0:
            raise RuntimeError(
                "Servo is in WHEEL mode (CW=0, CCW=0). It will NOT respond to GOAL_POSITION.\n"
                "In Dynamixel Wizard, set it back to Joint mode by setting CCW Angle Limit > 0."
            )

        # Ensure torque on
        self._w1(ADDR_TORQUE_ENABLE, 1)

        self._is_connected = True
        time.sleep(0.05)

        if self.verbose:
            try:
                spd = self._r2(ADDR_MOVING_SPEED)
                tl = self._r2(ADDR_TORQUE_LIMIT)
            except Exception:
                spd, tl = None, None

            print(
                f"[DynamixelUSB] Connected port={self.port_name} baud={self.baudrate} id={self.dxl_id}\n"
                f"  model={self.model_number} fw={fw} protocol=1.0\n"
                f"  resolution_divider={self.resolution_divider} units_per_rev={self.units_per_rev} ticks_max={self.ticks_max}\n"
                f"  CW_limit={self.cw_limit} CCW_limit={self.ccw_limit} invert={self.invert}\n"
                f"  moving_speed={spd} torque_limit={tl}"
            )

    def disconnect(self) -> None:
        if self._is_connected:
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    try:
                        self._w1(ADDR_TORQUE_ENABLE, 0)
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                self.port_handler.closePort()
        except Exception:
            pass

        self._is_connected = False
        if self.verbose:
            print(f"[DynamixelUSB] Disconnected port={self.port_name}")

    # ---------------- public API ----------------
    def ping(self, max_retries: int = 2) -> dict:
        last_error = None
        for attempt in range(int(max_retries)):
            try:
                model, comm, err = self.packet_handler.ping(self.port_handler, self.dxl_id)
                if comm == COMM_SUCCESS and err == 0:
                    return {"ok": True, "model": int(model)}
                last_error = (
                    self.packet_handler.getTxRxResult(comm)
                    if comm != COMM_SUCCESS
                    else self.packet_handler.getRxPacketError(err)
                )
            except Exception as e:
                last_error = str(e)

            if attempt < int(max_retries) - 1:
                time.sleep(0.05)

        return {"ok": False, "error": last_error or "Unknown error"}

    def set_degrees(self, cmd_deg_0_to_255: float) -> dict:
        """
        The function you want:
          send degrees in [0..255] where
            0   = open
            255 = closed
        """
        if not self._is_connected:
            self.connect()

        cmd_deg = self._clamp_cmd_deg(cmd_deg_0_to_255)

        # IMPORTANT: cmd_deg is interpreted as *actual servo degrees* (0..255 out of 0..360)
        ticks = self._deg_to_ticks(cmd_deg)
        self._w2(ADDR_GOAL_POSITION, ticks)

        if self.post_write_sleep_s > 0:
            time.sleep(self.post_write_sleep_s)

        norm = self._cmd_deg_to_norm(cmd_deg)
        self._last_norm = norm
        return {"ok": True, "position": norm, "cmd_deg": cmd_deg, "dxl_position": ticks}

    def set(self, normalized_value: float) -> dict:
        """
        Compatibility with your existing tests:
        normalized 0..1 -> command degrees 0..255 -> send.
        """
        n = max(0.0, min(1.0, float(normalized_value)))
        cmd_deg = self._norm_to_cmd_deg(n)
        out = self.set_degrees(cmd_deg)
        out["position"] = n
        return out

    def get(self, max_retries: int = 2) -> dict:
        if not self._is_connected:
            self.connect()

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                ticks = self._r2(ADDR_PRESENT_POSITION)
                angle_deg = self._ticks_to_deg(ticks)

                # Convert the *measured* servo angle (0..360) into your command space (0..255)
                cmd_deg = max(self.CMD_MIN_DEG, min(self.CMD_MAX_DEG, angle_deg))
                norm = self._cmd_deg_to_norm(cmd_deg)

                self._last_norm = norm
                return {
                    "ok": True,
                    "position": norm,
                    "cmd_deg": cmd_deg,
                    "angle_deg": angle_deg,
                    "dxl_position": ticks,
                }
            except Exception as e:
                last_error = str(e)
                if attempt < int(max_retries) - 1:
                    time.sleep(0.05 * (attempt + 1))

        return {
            "ok": False,
            "position": self._last_norm,
            "cached": True,
            "error": last_error,
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class GripperUSBClient:
    """
    Wrapper matching your existing 'client' shape.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 57600, dxl_id: int = 1, **kwargs):
        self._gripper = DynamixelUSBGripper(port=port, baudrate=baudrate, dxl_id=dxl_id, **kwargs)

    def set(self, value: float) -> dict:
        return self._gripper.set(value)

    def set_degrees(self, deg_command: float) -> dict:
        return self._gripper.set_degrees(deg_command)

    def get(self) -> dict:
        return self._gripper.get()

    def ping(self) -> dict:
        if not self._gripper._is_connected:
            self._gripper.connect()
        return self._gripper.ping()

    def disconnect(self):
        self._gripper.disconnect()


# ---------------- CLI (optional quick test) ----------------
def _default_port() -> str:
    if platform.system() == "Windows":
        return "COM3"
    return "/dev/ttyUSB0"


def main():
    ap = argparse.ArgumentParser(description="MX-106 USB gripper control: 0..255 degrees (0 open, 255 closed)")
    ap.add_argument("--port", default=_default_port())
    ap.add_argument("--baudrate", type=int, default=57600)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("get")

    p_sd = sub.add_parser("set-deg")
    p_sd.add_argument("--deg", type=float, required=True, help="0..255 degrees (0 open, 255 closed)")

    p_s = sub.add_parser("set")
    p_s.add_argument("--norm", type=float, required=True, help="0..1 (mapped to 0..255 degrees)")

    p_sw = sub.add_parser("sweep")
    p_sw.add_argument("--delay", type=float, default=0.7)
    p_sw.add_argument("--deg-list", type=float, nargs="+", default=[0, 255, 0])

    args = ap.parse_args()

    g = DynamixelUSBGripper(
        port=args.port,
        baudrate=args.baudrate,
        dxl_id=args.id,
        invert=args.invert,
        verbose=not args.quiet,
    )

    try:
        if args.cmd == "ping":
            g.connect()
            print(g.ping())
        elif args.cmd == "get":
            print(g.get())
        elif args.cmd == "set-deg":
            print(g.set_degrees(args.deg))
        elif args.cmd == "set":
            print(g.set(args.norm))
        elif args.cmd == "sweep":
            print(g.get())
            for d in args.deg_list:
                out = g.set_degrees(d)
                time.sleep(args.delay)
                meas = g.get()
                print({"cmd": out, "meas": meas})
        else:
            raise RuntimeError("Unknown command")
    finally:
        try:
            g.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
