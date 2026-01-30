#!/usr/bin/env python3
"""
Direct USB control of Dynamixel servo gripper using pyserial and dynamixel SDK.
Alternative to ROS/Teensy-based control.

UPDATED BEHAVIOUR:
- Accepts "degrees" commands in the SAME domain as your Teensy code: 0..255.
- Converts those degrees to AX/Protocol1 ticks using the AX scale (0..300deg -> 0..1023 ticks):
    ticks = (deg / 300) * 1023
  (This matches what DynamixelShield does when you call setGoalPosition(..., UNIT_DEGREE))

Requirements for USB mode:
    pip install pyserial dynamixel-sdk

Note: On Linux, you may need to add your user to the dialout group:
    sudo usermod -a -G dialout $USER
    (then logout/login for changes to take effect)
"""

import time

try:
    from dynamixel_sdk import (
        PortHandler,
        PacketHandler,
        COMM_SUCCESS,
    )
except ImportError:
    raise ImportError(
        "dynamixel-sdk not installed. Install with: pip install dynamixel-sdk\n"
        "This is required for USB gripper mode (gripper_mode='usb')."
    )


class DynamixelUSBGripper:
    """
    Direct USB control of Dynamixel gripper servo.
    Compatible with the same interface as GripperClient for drop-in replacement.
    """

    # Dynamixel control table addresses (AX series / Protocol 1.0)
    ADDR_TORQUE_ENABLE = 24
    ADDR_GOAL_POSITION = 30
    ADDR_PRESENT_POSITION = 36
    ADDR_MOVING_SPEED = 32

    # Protocol version from .ino file
    DXL_PROTOCOL_VERSION = 1.0

    # AX position mode mapping (approx): 0..300 deg => 0..1023 ticks
    DXL_FULL_SCALE_DEG = 300.0
    DXL_FULL_SCALE_TICKS = 1023

    # Default values from .ino "degrees" command domain (what you publish as Int16)
    MIN_ANGLE = 0.0    # Open position (degrees)
    MAX_ANGLE = 255.0  # Close position (degrees)

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",  # Linux default, Windows: "COM3"
        baudrate: int = 57600,       # From .ino: dxl.begin(57600)
        dxl_id: int = 1,             # DXL_MOTOR_GRIPPER_ID from .ino
        min_angle: float = MIN_ANGLE,
        max_angle: float = MAX_ANGLE,
    ):
        """
        Initialize USB Dynamixel gripper.

        Args:
            port: Serial port (e.g., "/dev/ttyUSB0" on Linux, "COM3" on Windows)
            baudrate: Communication baudrate (default 57600 from .ino)
            dxl_id: Dynamixel servo ID (default 1)
            min_angle: Open position in degrees (command domain, default 0)
            max_angle: Close position in degrees (command domain, default 255)
        """
        self.port_name = port
        self.baudrate = int(baudrate)
        self.dxl_id = int(dxl_id)
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)

        if self.max_angle <= self.min_angle:
            raise ValueError("max_angle must be > min_angle")

        # Initialize port handler and packet handler
        self.port_handler = PortHandler(self.port_name)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)

        self._is_connected = False
        self._last_position = 0.0  # Normalized [0.0, 1.0]

    def connect(self):
        """Open serial port and enable torque."""
        if self._is_connected:
            # Verify port is still open
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    return
            except Exception:
                pass
            self._is_connected = False

        # Ensure port is closed before opening
        try:
            if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                try:
                    self.port_handler.closePort()
                except Exception:
                    pass
        except Exception:
            pass

        time.sleep(0.1)

        # Open port (retry once with recreated handler)
        if not self.port_handler.openPort():
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            time.sleep(0.2)

            self.port_handler = PortHandler(self.port_name)
            if not self.port_handler.openPort():
                raise RuntimeError(f"Failed to open port {self.port_name} after retry")

        # Set baudrate
        if not self.port_handler.setBaudRate(self.baudrate):
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"Failed to set baudrate to {self.baudrate}")

        # Enable torque
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 1
        )

        if dxl_comm_result != COMM_SUCCESS:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to enable torque: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
        if dxl_error != 0:
            try:
                self.port_handler.closePort()
            except Exception:
                pass
            raise RuntimeError(f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}")

        self._is_connected = True
        time.sleep(0.1)
        print(f"[DynamixelUSB] Connected to servo ID {self.dxl_id} on {self.port_name}")

    def disconnect(self):
        """Disable torque and close port."""
        if self._is_connected:
            try:
                if hasattr(self.port_handler, "is_open") and self.port_handler.is_open:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 0
                    )
            except Exception:
                pass

        # Close port with a few retries (helps on Windows)
        max_close_attempts = 3
        for attempt in range(max_close_attempts):
            try:
                if hasattr(self.port_handler, "is_open"):
                    if self.port_handler.is_open:
                        self.port_handler.closePort()
                    if not self.port_handler.is_open:
                        break
                    if attempt < max_close_attempts - 1:
                        time.sleep(0.1)
                else:
                    break
            except Exception:
                if attempt < max_close_attempts - 1:
                    time.sleep(0.1)

        self._is_connected = False
        print(f"[DynamixelUSB] Disconnected from {self.port_name}")

    # ---------------- Mapping helpers (UPDATED) ----------------
    def _clamp_angle(self, angle_deg: float) -> float:
        return max(self.min_angle, min(self.max_angle, float(angle_deg)))

    def _angle_to_dxl_position(self, angle_deg: float) -> int:
        """
        Convert *command-domain* degrees (default 0..255) to Dynamixel position ticks (0..1023).

        Matches DynamixelShield UNIT_DEGREE behaviour for AX/Protocol1:
          0..300 deg -> 0..1023 ticks

        So:
          ticks = (deg / 300) * 1023
        """
        angle_deg = self._clamp_angle(angle_deg)
        ticks = int(round((angle_deg / self.DXL_FULL_SCALE_DEG) * self.DXL_FULL_SCALE_TICKS))
        return max(0, min(self.DXL_FULL_SCALE_TICKS, ticks))

    def _dxl_position_to_angle(self, position: int) -> float:
        """
        Convert Dynamixel ticks (0..1023) back to degrees (0..300),
        then clamp into your command-domain range (default 0..255).
        """
        pos = max(0, min(self.DXL_FULL_SCALE_TICKS, int(position)))
        angle = (pos / self.DXL_FULL_SCALE_TICKS) * self.DXL_FULL_SCALE_DEG
        return self._clamp_angle(angle)

    def _normalized_to_angle(self, normalized: float) -> float:
        """Convert normalized [0.0, 1.0] to command-domain degrees [min_angle, max_angle]."""
        n = max(0.0, min(1.0, float(normalized)))
        return self.min_angle + n * (self.max_angle - self.min_angle)

    def _angle_to_normalized(self, angle_deg: float) -> float:
        """Convert command-domain degrees to normalized [0.0, 1.0]."""
        a = self._clamp_angle(angle_deg)
        return (a - self.min_angle) / (self.max_angle - self.min_angle)

    # ---------------- Public API ----------------
    def set_angle_deg(self, angle_deg: float, max_retries: int = 2) -> dict:
        """
        Set gripper position using a direct "degree" command (default 0..255).
        """
        if not self._is_connected:
            self.connect()

        angle_deg = self._clamp_angle(angle_deg)
        dxl_position = self._angle_to_dxl_position(angle_deg)

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_GOAL_POSITION, dxl_position
                )
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    normalized_value = self._angle_to_normalized(angle_deg)
                    self._last_position = normalized_value
                    return {
                        "ok": True,
                        "position": normalized_value,
                        "angle_deg": angle_deg,
                        "dxl_position": dxl_position,
                    }
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                elif dxl_error != 0:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"

            if attempt < int(max_retries) - 1:
                time.sleep(0.05)

        raise RuntimeError(f"Failed to set angle after {max_retries} attempts: {last_error}")

    def set(self, normalized_value: float, max_retries: int = 2) -> dict:
        """
        Set gripper position using normalized value [0.0, 1.0].
        """
        if not self._is_connected:
            self.connect()

        normalized_value = max(0.0, min(1.0, float(normalized_value)))
        angle_deg = self._normalized_to_angle(normalized_value)
        dxl_position = self._angle_to_dxl_position(angle_deg)

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_GOAL_POSITION, dxl_position
                )

                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    self._last_position = normalized_value
                    return {
                        "ok": True,
                        "position": normalized_value,
                        "angle_deg": angle_deg,
                        "dxl_position": dxl_position,
                    }
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                elif dxl_error != 0:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"

            if attempt < int(max_retries) - 1:
                time.sleep(0.05)

        raise RuntimeError(f"Failed to set position after {max_retries} attempts: {last_error}")

    def get(self, max_retries: int = 3) -> dict:
        """
        Get current gripper position as normalized value [0.0, 1.0].
        """
        if not self._is_connected:
            self.connect()

        time.sleep(0.01)

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                dxl_present_position, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_PRESENT_POSITION
                )

                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    angle_deg = self._dxl_position_to_angle(dxl_present_position)
                    normalized = self._angle_to_normalized(angle_deg)
                    self._last_position = normalized
                    return {
                        "ok": True,
                        "position": normalized,
                        "angle_deg": angle_deg,
                        "dxl_position": dxl_present_position,
                    }
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                elif dxl_error != 0:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"

            if attempt < int(max_retries) - 1:
                time.sleep(0.05 * (attempt + 1))

        if self._last_position is not None:
            return {
                "ok": False,
                "position": self._last_position,
                "error": last_error,
                "cached": True,
            }
        raise RuntimeError(f"Failed to read position after {max_retries} attempts: {last_error}")

    def ping(self, max_retries: int = 2) -> dict:
        """Ping the servo to check connection."""
        if not self._is_connected:
            self.connect()

        last_error = None
        for attempt in range(int(max_retries)):
            try:
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.port_handler, self.dxl_id
                )

                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    time.sleep(0.05)
                    return {"ok": True, "model": dxl_model_number}
                if dxl_comm_result != COMM_SUCCESS:
                    last_error = self.packet_handler.getTxRxResult(dxl_comm_result)
                elif dxl_error != 0:
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
    Wrapper matching GripperClient interface for drop-in replacement.
    Uses DynamixelUSBGripper internally.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", **kwargs):
        self._gripper = DynamixelUSBGripper(port=port, **kwargs)

    def set(self, value: float) -> dict:
        """Set gripper position [0.0, 1.0]."""
        return self._gripper.set(value)

    def set_degrees(self, angle_deg: float) -> dict:
        """Set gripper using Teensy-style degree command (default 0..255)."""
        return self._gripper.set_angle_deg(angle_deg)

    def get(self) -> dict:
        """Get gripper position [0.0, 1.0]."""
        return self._gripper.get()

    def ping(self) -> dict:
        """Ping gripper."""
        return self._gripper.ping()

    def disconnect(self):
        """Disconnect gripper."""
        self._gripper.disconnect()
