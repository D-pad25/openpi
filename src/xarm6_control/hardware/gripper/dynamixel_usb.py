#!/usr/bin/env python3
"""
dynamixel_usb.py

MX-106 (Protocol 1.0) USB gripper control that matches Dynamixel Wizard.

You want:
- send "degrees" in 0..255
  0   = fully open
  255 = fully closed

Notes:
- Your MX-106 reports 0..360° (Joint mode) over 0..(units_per_rev-1) ticks.
- Wizard-style mapping uses units_per_rev (e.g., 4096) in the scale.
- We keep TORQUE ON by default even when exiting, so the servo continues moving
  after the program ends (otherwise it "barely moves" if you torque-off immediately).

Requirements:
  pip install dynamixel-sdk pyserial
"""

from __future__ import annotations

import argparse
import platform
import time
from typing import Optional, Dict, Any

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS


# ---------------- Protocol 1.0 MX Control Table ----------------
ADDR_MODEL_NUMBER = 0            # 2 bytes
ADDR_FIRMWARE_VERSION = 2        # 1 byte
ADDR_CW_ANGLE_LIMIT = 6          # 2 bytes
ADDR_CCW_ANGLE_LIMIT = 8         # 2 bytes
ADDR_RESOLUTION_DIVIDER = 22     # 1 byte (MX only)

ADDR_TORQUE_ENABLE = 24          # 1 byte
ADDR_GOAL_POSITION = 30          # 2 bytes
ADDR_MOVING_SPEED = 32           # 2 bytes
ADDR_TORQUE_LIMIT = 34           # 2 bytes
ADDR_PRESENT_POSITION = 36       # 2 bytes
ADDR_PRESENT_SPEED = 38          # 2 bytes
ADDR_PRESENT_LOAD = 40           # 2 bytes
ADDR_PRESENT_VOLTAGE = 42        # 1 byte (0.1V)
ADDR_PRESENT_TEMPERATURE = 43    # 1 byte (°C)
ADDR_MOVING = 46                 # 1 byte (0/1)

# ---------------- MX nominal full scale ----------------
BASE_UNITS_PER_REV = 4096
BASE_FULL_SCALE_DEG = 360.0

# Your desired command domain
CMD_MIN_DEG = 0.0
CMD_MAX_DEG = 255.0


def _default_port() -> str:
    return "COM3" if platform.system() == "Windows" else "/dev/ttyUSB0"


def _decode_signed_10bit_dir(raw: int) -> int:
    """Present Speed/Load encoding (Protocol 1.0): mag 0..1023, dir bit10."""
    raw = int(raw) & 0x07FF
    mag = raw & 0x03FF
    cw = (raw & 0x0400) != 0
    return -mag if cw else mag


class DynamixelUSBGripper:
    DXL_PROTOCOL_VERSION = 1.0

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        dxl_id: int = 1,
        *,
        invert: bool = False,
        verbose: bool = True,
    ):
        self.port_name = str(port)
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)
        self.invert = bool(invert)
        self.verbose = bool(verbose)

        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)

        self._is_connected = False

        self.model_number: Optional[int] = None
        self.fw: Optional[int] = None

        self.resolution_divider: int = 1
        self.units_per_rev: int = BASE_UNITS_PER_REV  # e.g., 4096/div
        self.ticks_max: int = self.units_per_rev - 1  # e.g., 4095

        self.cw_limit: int = 0
        self.ccw_limit: int = self.ticks_max

        self._last_norm: float = 0.0

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

    # ---------------- conversions ----------------
    def _clamp_cmd_deg(self, deg: float) -> float:
        return max(CMD_MIN_DEG, min(CMD_MAX_DEG, float(deg)))

    def _cmd_deg_to_norm(self, deg: float) -> float:
        d = self._clamp_cmd_deg(deg)
        return (d - CMD_MIN_DEG) / (CMD_MAX_DEG - CMD_MIN_DEG)

    def _norm_to_cmd_deg(self, n: float) -> float:
        n = max(0.0, min(1.0, float(n)))
        return CMD_MIN_DEG + n * (CMD_MAX_DEG - CMD_MIN_DEG)

    def _deg_to_ticks(self, deg_servo: float) -> int:
        """
        Wizard-style mapping:
          ticks = round((deg/360) * units_per_rev)
          then clamp to [0..ticks_max]
        """
        d = float(deg_servo)
        if self.invert:
            d = BASE_FULL_SCALE_DEG - d
        d = max(0.0, min(BASE_FULL_SCALE_DEG, d))

        ticks = int(round((d / BASE_FULL_SCALE_DEG) * self.units_per_rev))
        if ticks > self.ticks_max:
            ticks = self.ticks_max

        lo = min(self.cw_limit, self.ccw_limit)
        hi = max(self.cw_limit, self.ccw_limit)
        return max(lo, min(hi, ticks))

    def _ticks_to_deg(self, ticks: int) -> float:
        t = max(0, min(self.ticks_max, int(ticks)))
        deg = (t / float(self.units_per_rev)) * BASE_FULL_SCALE_DEG
        if self.invert:
            deg = BASE_FULL_SCALE_DEG - deg
        return deg

    # ---------------- connection ----------------
    def connect(self) -> None:
        if self._is_connected:
            try:
                if getattr(self.port_handler, "is_open", False):
                    return
            except Exception:
                pass
            self._is_connected = False

        # best-effort close
        try:
            if getattr(self.port_handler, "is_open", False):
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

        # Read key config
        self.model_number = self._r2(ADDR_MODEL_NUMBER)
        self.fw = self._r1(ADDR_FIRMWARE_VERSION)

        # MX resolution divider (1 => 4096/rev)
        try:
            div = self._r1(ADDR_RESOLUTION_DIVIDER)
            self.resolution_divider = int(div) if int(div) > 0 else 1
        except Exception:
            self.resolution_divider = 1

        upr = int(BASE_UNITS_PER_REV // self.resolution_divider)
        if upr <= 0:
            upr = BASE_UNITS_PER_REV
        self.units_per_rev = upr
        self.ticks_max = self.units_per_rev - 1

        self.cw_limit = self._r2(ADDR_CW_ANGLE_LIMIT)
        self.ccw_limit = self._r2(ADDR_CCW_ANGLE_LIMIT)

        if self.cw_limit == 0 and self.ccw_limit == 0:
            raise RuntimeError(
                "Servo is in WHEEL mode (CW=0, CCW=0). It will NOT respond to GOAL_POSITION.\n"
                "In Dynamixel Wizard set CCW Angle Limit > 0 to restore Joint mode."
            )

        # torque on (like Teensy)
        self._w1(ADDR_TORQUE_ENABLE, 1)

        self._is_connected = True
        time.sleep(0.05)

        if self.verbose:
            spd = self._r2(ADDR_MOVING_SPEED)
            tl = self._r2(ADDR_TORQUE_LIMIT)
            print(
                f"[DynamixelUSB] Connected port={self.port_name} baud={self.baudrate} id={self.dxl_id}\n"
                f"  model={self.model_number} fw={self.fw} protocol=1.0\n"
                f"  resolution_divider={self.resolution_divider} units_per_rev={self.units_per_rev} ticks_max={self.ticks_max}\n"
                f"  CW_limit={self.cw_limit} CCW_limit={self.ccw_limit} invert={self.invert}\n"
                f"  moving_speed={spd} torque_limit={tl}"
            )

    def disconnect(self, disable_torque: bool = False) -> None:
        # Only torque off if explicitly requested
        if self._is_connected and disable_torque:
            try:
                self._w1(ADDR_TORQUE_ENABLE, 0)
            except Exception:
                pass

        try:
            if getattr(self.port_handler, "is_open", False):
                self.port_handler.closePort()
        except Exception:
            pass

        self._is_connected = False
        if self.verbose:
            print(f"[DynamixelUSB] Disconnected port={self.port_name} (disable_torque={disable_torque})")

    # ---------------- diagnostics ----------------
    def read_status(self) -> Dict[str, Any]:
        if not self._is_connected:
            self.connect()

        torque = self._r1(ADDR_TORQUE_ENABLE)
        moving = self._r1(ADDR_MOVING)
        goal = self._r2(ADDR_GOAL_POSITION)
        pres = self._r2(ADDR_PRESENT_POSITION)

        v_raw = self._r1(ADDR_PRESENT_VOLTAGE)
        temp = self._r1(ADDR_PRESENT_TEMPERATURE)

        spd_raw = self._r2(ADDR_PRESENT_SPEED)
        load_raw = self._r2(ADDR_PRESENT_LOAD)

        spd = _decode_signed_10bit_dir(spd_raw)
        load = _decode_signed_10bit_dir(load_raw)

        return {
            "torque_enable": torque,
            "moving": moving,
            "goal_ticks": goal,
            "present_ticks": pres,
            "present_deg": self._ticks_to_deg(pres),
            "goal_deg": self._ticks_to_deg(goal),
            "present_speed_raw": spd_raw,
            "present_speed_signed": spd,
            "present_load_raw": load_raw,
            "present_load_signed": load,
            "voltage_v": v_raw / 10.0,
            "temperature_c": temp,
            "cw_limit": self.cw_limit,
            "ccw_limit": self.ccw_limit,
            "units_per_rev": self.units_per_rev,
            "ticks_max": self.ticks_max,
        }

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

    def set_degrees(self, cmd_deg_0_to_255: float, *, wait_s: float = 0.0, poll_hz: float = 50.0) -> dict:
        """
        Send your command domain degrees directly: 0..255.
        We interpret that as actual servo degrees (0..255°) and convert to ticks.
        """
        if not self._is_connected:
            self.connect()

        cmd_deg = self._clamp_cmd_deg(cmd_deg_0_to_255)
        ticks = self._deg_to_ticks(cmd_deg)

        self._w2(ADDR_GOAL_POSITION, ticks)
        goal_back = self._r2(ADDR_GOAL_POSITION)

        norm = self._cmd_deg_to_norm(cmd_deg)
        self._last_norm = norm

        out = {"ok": True, "position": norm, "cmd_deg": cmd_deg, "goal_ticks": ticks, "goal_readback": goal_back}

        if wait_s and wait_s > 0:
            dt = 1.0 / max(1.0, float(poll_hz))
            t0 = time.time()
            samples = []
            while (time.time() - t0) < float(wait_s):
                st = self.read_status()
                samples.append(
                    {
                        "moving": st["moving"],
                        "present_ticks": st["present_ticks"],
                        "present_deg": st["present_deg"],
                        "goal_ticks": st["goal_ticks"],
                    }
                )
                time.sleep(dt)
            out["samples"] = samples

        return out

    def set(self, normalized_value: float, *, wait_s: float = 0.0) -> dict:
        n = max(0.0, min(1.0, float(normalized_value)))
        cmd_deg = self._norm_to_cmd_deg(n)
        out = self.set_degrees(cmd_deg, wait_s=wait_s)
        out["position"] = n
        return out

    def get(self) -> dict:
        if not self._is_connected:
            self.connect()

        st = self.read_status()
        # represent measured position in your 0..255 space
        cmd_deg_meas = max(CMD_MIN_DEG, min(CMD_MAX_DEG, st["present_deg"]))
        norm = self._cmd_deg_to_norm(cmd_deg_meas)
        self._last_norm = norm
        return {
            "ok": True,
            "position": norm,
            "cmd_deg": cmd_deg_meas,
            "angle_deg": st["present_deg"],
            "dxl_position": st["present_ticks"],
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect(disable_torque=False)


class GripperUSBClient:
    """Wrapper matching your existing GripperClient shape."""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 57600, dxl_id: int = 1, **kwargs):
        self._gripper = DynamixelUSBGripper(port=port, baudrate=baudrate, dxl_id=dxl_id, **kwargs)

    def set(self, value: float, **kwargs) -> dict:
        return self._gripper.set(value, **kwargs)

    def set_degrees(self, deg_command: float, **kwargs) -> dict:
        return self._gripper.set_degrees(deg_command, **kwargs)

    def get(self) -> dict:
        return self._gripper.get()

    def ping(self) -> dict:
        return self._gripper.ping()

    def disconnect(self, disable_torque: bool = False):
        self._gripper.disconnect(disable_torque=disable_torque)


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="MX-106 USB gripper: send 0..255 degrees (0 open, 255 closed)")
    ap.add_argument("--port", default=_default_port())
    ap.add_argument("--baudrate", type=int, default=57600)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--torque-off-on-exit", action="store_true", help="Disable torque when exiting")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    sub.add_parser("get")
    sub.add_parser("diag")

    p_sd = sub.add_parser("set-deg")
    p_sd.add_argument("--deg", type=float, required=True, help="0..255 (0 open, 255 closed)")
    p_sd.add_argument("--wait", type=float, default=0.0, help="poll for this many seconds after commanding")

    p_s = sub.add_parser("set")
    p_s.add_argument("--norm", type=float, required=True, help="0..1 mapped to 0..255 degrees")
    p_s.add_argument("--wait", type=float, default=0.0)

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
            print(g.ping())
        elif args.cmd == "get":
            print(g.get())
        elif args.cmd == "diag":
            print(g.read_status())
        elif args.cmd == "set-deg":
            print(g.set_degrees(args.deg, wait_s=args.wait))
        elif args.cmd == "set":
            print(g.set(args.norm, wait_s=args.wait))
        else:
            raise RuntimeError("Unknown command")
    finally:
        try:
            g.disconnect(disable_torque=args.torque_off_on_exit)
        except Exception:
            pass


if __name__ == "__main__":
    main()
