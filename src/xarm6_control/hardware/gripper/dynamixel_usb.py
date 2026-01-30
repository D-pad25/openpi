#!/usr/bin/env python3
"""
dynamixel_usb_teensy_clone.py

A "Teensy-clone" USB controller for your Dynamixel gripper.

This intentionally matches what your Teensy code does for the gripper:
  - Protocol 1.0
  - baud 57600
  - torqueOn(ID)
  - setGoalPosition(ID, <degrees>, UNIT_DEGREE)

Key point:
- Your Teensy sends the Int16 value directly as degrees (UNIT_DEGREE).
- It does NOT set speed/torque limit/compliance/angle limits for the gripper.
- So this USB path avoids "smart" configs and just writes GOAL_POSITION ticks.

It provides:
  - DynamixelUSBGripperTeensyClone: connect/ping/set/get/disconnect
  - GripperUSBClient: wrapper matching your set/get/ping/disconnect shape
  - CLI for quick testing

Usage examples:
  # Linux / WSL
  python dynamixel_usb_teensy_clone.py --port /dev/ttyUSB0 ping
  python dynamixel_usb_teensy_clone.py --port /dev/ttyUSB0 get
  python dynamixel_usb_teensy_clone.py --port /dev/ttyUSB0 set --norm 0.5
  python dynamixel_usb_teensy_clone.py --port /dev/ttyUSB0 set-deg --deg 179

  # Windows
  python dynamixel_usb_teensy_clone.py --port COM3 ping
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


@dataclass
class GripperState:
    ok: bool
    position: float                 # normalized [0..1]
    angle_deg: Optional[float] = None
    dxl_position: Optional[int] = None
    cached: bool = False
    error: Optional[str] = None


class DynamixelUSBGripperTeensyClone:
    """
    Minimal, Teensy-like USB control for AX-series (Protocol 1.0) position mode.

    Public:
      - connect()
      - disconnect()
      - ping()
      - set(norm 0..1)
      - set_degrees(deg_command)
      - get()
    """

    # AX / Protocol 1.0 control table
    ADDR_TORQUE_ENABLE = 24          # 1 byte
    ADDR_GOAL_POSITION = 30          # 2 bytes
    ADDR_PRESENT_POSITION = 36       # 2 bytes

    DXL_PROTOCOL_VERSION = 1.0

    # AX position mapping (approx): 0..300 deg -> 0..1023 ticks
    DXL_FULL_SCALE_DEG = 300.0
    DXL_FULL_SCALE_TICKS = 1023

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        dxl_id: int = 1,
        *,
        # This is the "command domain" you *use* (what your pipeline publishes).
        # Your Teensy is receiving Int16 and feeding it directly to UNIT_DEGREE.
        # Many setups use 0..255 for gripper range.
        min_cmd_deg: float = 0.0,
        max_cmd_deg: float = 255.0,
        # If you prefer to use your known physical range (5..179), set these.
        # But default keeps the 0..255 command domain.
        clamp_to_0_300: bool = True,
    ):
        self.port_name = str(port)
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)

        self.min_cmd_deg = float(min_cmd_deg)
        self.max_cmd_deg = float(max_cmd_deg)
        if self.max_cmd_deg <= self.min_cmd_deg:
            raise ValueError("max_cmd_deg must be > min_cmd_deg")

        self.clamp_to_0_300 = bool(clamp_to_0_300)

        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)

        self._is_connected = False
        self._last_norm = 0.0

    # ---------------- conversion helpers ----------------
    def _clamp_cmd_deg(self, deg: float) -> float:
        return max(self.min_cmd_deg, min(self.max_cmd_deg, float(deg)))

    def _norm_to_cmd_deg(self, n: float) -> float:
        n = max(0.0, min(1.0, float(n)))
        return self.min_cmd_deg + n * (self.max_cmd_deg - self.min_cmd_deg)

    def _cmd_deg_to_norm(self, deg: float) -> float:
        deg = self._clamp_cmd_deg(deg)
        return (deg - self.min_cmd_deg) / (self.max_cmd_deg - self.min_cmd_deg)

    def _deg_to_ticks(self, deg: float) -> int:
        """
        Teensy uses DynamixelShield UNIT_DEGREE. For AX protocol 1.0 this is effectively:
            ticks = (deg / 300) * 1023
        """
        d = float(deg)
        if self.clamp_to_0_300:
            d = max(0.0, min(300.0, d))
        ticks = int(round((d / self.DXL_FULL_SCALE_DEG) * self.DXL_FULL_SCALE_TICKS))
        return max(0, min(self.DXL_FULL_SCALE_TICKS, ticks))

    def _ticks_to_deg(self, ticks: int) -> float:
        t = max(0, min(self.DXL_FULL_SCALE_TICKS, int(ticks)))
        return (t / self.DXL_FULL_SCALE_TICKS) * self.DXL_FULL_SCALE_DEG

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

    def _r2(self, addr: int) -> int:
        val, comm, err = self.packet_handler.read2ByteTxRx(self.port_handler, self.dxl_id, addr)
        if comm != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm))
        if err != 0:
            raise RuntimeError(self.packet_handler.getRxPacketError(err))
        return int(val)

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

        # Teensy does torqueOn() for gripper (no other profile writes)
        self._w1(self.ADDR_TORQUE_ENABLE, 1)

        self._is_connected = True
        time.sleep(0.05)
        print(f"[USB TeensyClone] Connected id={self.dxl_id} port={self.port_name} baud={self.baudrate}")

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

        try:
            if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                self.port_handler.closePort()
        except Exception:
            pass

        self._is_connected = False
        print(f"[USB TeensyClone] Disconnected port={self.port_name}")

    # ---------------- public API ----------------
    def ping(self, max_retries: int = 2) -> dict:
        if not self._is_connected:
            self.connect()

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

    def set_degrees(self, deg_command: float) -> dict:
        """
        Teensy behavior: pass the incoming Int16 directly as UNIT_DEGREE.
        Here: we convert that degrees value to ticks and write GOAL_POSITION.
        """
        if not self._is_connected:
            self.connect()

        deg_command = float(deg_command)

        # NOTE: we do NOT remap degrees; we only clamp for your own command-domain tracking
        # and optional clamp to 0..300 for tick conversion.
        norm = self._cmd_deg_to_norm(deg_command)  # for reporting
        ticks = self._deg_to_ticks(deg_command)

        self._w2(self.ADDR_GOAL_POSITION, ticks)

        self._last_norm = max(0.0, min(1.0, norm))
        return {"ok": True, "position": self._last_norm, "angle_deg": deg_command, "dxl_position": ticks}

    def set(self, normalized_value: float) -> dict:
        """
        Policy output: normalized [0..1] -> command degrees in [min_cmd_deg..max_cmd_deg]
        then written as Teensy-style degrees.
        """
        n = max(0.0, min(1.0, float(normalized_value)))
        deg = self._norm_to_cmd_deg(n)
        out = self.set_degrees(deg)
        out["position"] = n  # report requested normalized
        self._last_norm = n
        return out

    def get(self, max_retries: int = 2) -> dict:
        if not self._is_connected:
            self.connect()

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                ticks = self._r2(self.ADDR_PRESENT_POSITION)
                deg = self._ticks_to_deg(ticks)
                # Convert that measured deg into your command-domain normalized
                norm = self._cmd_deg_to_norm(deg)
                norm = max(0.0, min(1.0, norm))
                self._last_norm = norm
                return {"ok": True, "position": norm, "angle_deg": deg, "dxl_position": ticks}
            except Exception as e:
                last_error = str(e)
                if attempt < int(max_retries) - 1:
                    time.sleep(0.05 * (attempt + 1))

        return {"ok": False, "position": self._last_norm, "cached": True, "error": last_error}

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class GripperUSBClient:
    """
    Wrapper matching your existing GripperClient shape.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", **kwargs):
        self._gripper = DynamixelUSBGripperTeensyClone(port=port, **kwargs)

    def set(self, value: float) -> dict:
        return self._gripper.set(value)

    def set_degrees(self, deg_command: float) -> dict:
        return self._gripper.set_degrees(deg_command)

    def get(self) -> dict:
        return self._gripper.get()

    def ping(self) -> dict:
        return self._gripper.ping()

    def disconnect(self):
        self._gripper.disconnect()


# ---------------- CLI ----------------
def _default_port() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        return "COM3"
    # Linux / WSL / others
    return "/dev/ttyUSB0"


def main():
    parser = argparse.ArgumentParser(description="Teensy-clone USB Dynamixel gripper control")
    parser.add_argument("--port", type=str, default=_default_port())
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--id", type=int, default=1)
    parser.add_argument("--min-cmd-deg", type=float, default=0.0, help="Command domain min deg (default 0)")
    parser.add_argument("--max-cmd-deg", type=float, default=255.0, help="Command domain max deg (default 255)")
    parser.add_argument("--no-clamp-0-300", action="store_true", help="Disable clamp to 0..300 before tick conversion")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("get")

    p_set = sub.add_parser("set")
    p_set.add_argument("--norm", type=float, required=True, help="normalized 0..1")

    p_setd = sub.add_parser("set-deg")
    p_setd.add_argument("--deg", type=float, required=True, help="degree command (Teensy-style)")

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--delay", type=float, default=0.5, help="delay between moves (s)")
    p_sweep.add_argument(
        "--deg-list",
        type=float,
        nargs="+",
        default=[5, 179, 5, 255, 5],
        help="sequence of degree commands",
    )

    args = parser.parse_args()

    gripper = DynamixelUSBGripperTeensyClone(
        port=args.port,
        baudrate=args.baudrate,
        dxl_id=args.id,
        min_cmd_deg=args.min_cmd_deg,
        max_cmd_deg=args.max_cmd_deg,
        clamp_to_0_300=not args.no_clamp_0_300,
    )

    try:
        if args.cmd == "ping":
            print(gripper.ping())
        elif args.cmd == "get":
            print(gripper.get())
        elif args.cmd == "set":
            print(gripper.set(args.norm))
        elif args.cmd == "set-deg":
            print(gripper.set_degrees(args.deg))
        elif args.cmd == "sweep":
            print(gripper.ping())
            for d in args.deg_list:
                out = gripper.set_degrees(d)
                time.sleep(args.delay)
                meas = gripper.get()
                print({"cmd": out, "meas": meas})
        else:
            raise RuntimeError("Unknown command")
    finally:
        try:
            gripper.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
