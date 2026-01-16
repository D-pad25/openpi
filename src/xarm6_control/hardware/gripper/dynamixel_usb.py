#!/usr/bin/env python3
"""
Direct USB control of Dynamixel servo gripper using pyserial and dynamixel SDK.
Alternative to ROS/Teensy-based control.

Requirements for USB mode:
    pip install pyserial dynamixel-sdk

Note: On Linux, you may need to add your user to the dialout group:
    sudo usermod -a -G dialout $USER
    (then logout/login for changes to take effect)
"""

import time
from typing import Optional

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
    
    # Dynamixel control table addresses (AX series)
    ADDR_TORQUE_ENABLE = 24
    ADDR_GOAL_POSITION = 30
    ADDR_PRESENT_POSITION = 36
    ADDR_MOVING_SPEED = 32
    
    # Protocol version from .ino file
    DXL_PROTOCOL_VERSION = 1.0
    
    # Default values from .ino file
    MIN_ANGLE = 5.0   # Open position (degrees)
    MAX_ANGLE = 179.0  # Close position (degrees)
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",  # Linux default, Windows: "COM3"
        baudrate: int = 57600,  # From .ino: dxl.begin(57600)
        dxl_id: int = 1,  # DXL_MOTOR_GRIPPER_ID from .ino
        min_angle: float = 5.0,
        max_angle: float = 179.0,
    ):
        """
        Initialize USB Dynamixel gripper.
        
        Args:
            port: Serial port (e.g., "/dev/ttyUSB0" on Linux, "COM3" on Windows)
            baudrate: Communication baudrate (default 57600 from .ino)
            dxl_id: Dynamixel servo ID (default 1)
            min_angle: Open position in degrees
            max_angle: Close position in degrees
        """
        self.port_name = port
        self.baudrate = baudrate
        self.dxl_id = dxl_id
        self.min_angle = min_angle
        self.max_angle = max_angle
        
        # Initialize port handler and packet handler
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(self.DXL_PROTOCOL_VERSION)
        
        self._is_connected = False
        self._last_position = 0.0  # Normalized [0.0, 1.0]
        
    def connect(self):
        """Open serial port and enable torque."""
        if self._is_connected:
            # Verify port is still open
            try:
                if self.port_handler.is_open:
                    return  # Already connected and port is open
            except:
                pass
            # Port seems closed, mark as disconnected and reconnect
            self._is_connected = False
        
        # Ensure port is closed before opening (Windows needs explicit cleanup)
        try:
            if hasattr(self.port_handler, 'is_open') and self.port_handler.is_open:
                try:
                    self.port_handler.closePort()
                except:
                    pass
        except:
            pass
        
        # On Windows, sometimes need to recreate the port handler if it's in a bad state
        import time
        time.sleep(0.1)  # Brief delay for port to be fully released
            
        # Open port
        if not self.port_handler.openPort():
            # If open fails, recreate port handler and try once more
            try:
                self.port_handler.closePort()
            except:
                pass
            time.sleep(0.2)
            # Recreate port handler
            from dynamixel_sdk import PortHandler
            self.port_handler = PortHandler(self.port_name)
            if not self.port_handler.openPort():
                raise RuntimeError(f"Failed to open port {self.port_name} after retry")
        
        # Set baudrate
        if not self.port_handler.setBaudRate(self.baudrate):
            try:
                self.port_handler.closePort()
            except:
                pass
            raise RuntimeError(f"Failed to set baudrate to {self.baudrate}")
        
        # Enable torque
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 1
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to enable torque: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        if dxl_error != 0:
            self.port_handler.closePort()
            raise RuntimeError(f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}")
        
        self._is_connected = True
        
        # Small delay after connection to allow servo to stabilize
        import time
        time.sleep(0.1)
        
        print(f"[DynamixelUSB] Connected to servo ID {self.dxl_id} on {self.port_name}")
        
    def disconnect(self):
        """Disable torque and close port."""
        # Disable torque (only if connected and port is open)
        if self._is_connected:
            try:
                if hasattr(self.port_handler, 'is_open') and self.port_handler.is_open:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 0
                    )
            except:
                pass
        
        # Close port if open (with retries on Windows)
        import time
        max_close_attempts = 3
        for attempt in range(max_close_attempts):
            try:
                if hasattr(self.port_handler, 'is_open'):
                    if self.port_handler.is_open:
                        self.port_handler.closePort()
                    # Verify it's closed
                    if not self.port_handler.is_open:
                        break
                    elif attempt < max_close_attempts - 1:
                        time.sleep(0.1)  # Wait and retry on Windows
                else:
                    break  # Can't check, assume closed
            except Exception as e:
                if attempt < max_close_attempts - 1:
                    time.sleep(0.1)
                else:
                    # Final attempt failed, but mark as disconnected anyway
                    pass
        
        self._is_connected = False
        print(f"[DynamixelUSB] Disconnected from {self.port_name}")
        
    def _angle_to_dxl_position(self, angle_deg: float) -> int:
        """Convert angle in degrees to Dynamixel position value (0-1023)."""
        # Clamp to valid range
        angle_deg = max(self.min_angle, min(self.max_angle, angle_deg))
        # Map to 0-1023 (AX series has 300 degree range, but we use subset)
        # Position 0 = 0°, Position 1023 = ~300°
        # For our range [min_angle, max_angle], map linearly
        position = int(((angle_deg - self.min_angle) / (self.max_angle - self.min_angle)) * 1023)
        return max(0, min(1023, position))
    
    def _dxl_position_to_angle(self, position: int) -> float:
        """Convert Dynamixel position value (0-1023) to angle in degrees."""
        angle = self.min_angle + (position / 1023.0) * (self.max_angle - self.min_angle)
        return angle
    
    def _normalized_to_angle(self, normalized: float) -> float:
        """Convert normalized [0.0, 1.0] to angle in degrees."""
        # 0.0 = open (min_angle), 1.0 = closed (max_angle)
        return self.min_angle + normalized * (self.max_angle - self.min_angle)
    
    def _angle_to_normalized(self, angle_deg: float) -> float:
        """Convert angle in degrees to normalized [0.0, 1.0]."""
        # Clamp angle first
        angle_deg = max(self.min_angle, min(self.max_angle, angle_deg))
        return (angle_deg - self.min_angle) / (self.max_angle - self.min_angle)
    
    def set(self, normalized_value: float, max_retries: int = 2) -> dict:
        """
        Set gripper position using normalized value [0.0, 1.0].
        Compatible with GripperClient interface.
        
        Args:
            normalized_value: 0.0 = open, 1.0 = closed
            max_retries: Maximum number of retry attempts if write fails
            
        Returns:
            dict with "ok" and "position" keys (compatible with GripperClient)
        """
        if not self._is_connected:
            self.connect()
        
        # Clamp to [0.0, 1.0]
        normalized_value = max(0.0, min(1.0, float(normalized_value)))
        
        # Convert to angle
        angle_deg = self._normalized_to_angle(normalized_value)
        
        # Convert to Dynamixel position
        dxl_position = self._angle_to_dxl_position(angle_deg)
        
        # Retry logic for writing position
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                # Write goal position
                dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_GOAL_POSITION, dxl_position
                )
                
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    # Success
                    self._last_position = normalized_value
                    
                    return {
                        "ok": True,
                        "position": normalized_value,
                        "angle_deg": angle_deg,
                        "dxl_position": dxl_position,
                    }
                elif dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                elif dxl_error != 0:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"
            
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(0.05)
        
        # All retries failed
        raise RuntimeError(f"Failed to set position after {max_retries} attempts: {last_error}")
    
    def get(self, max_retries: int = 3) -> dict:
        """
        Get current gripper position as normalized value [0.0, 1.0].
        Compatible with GripperClient interface.
        
        Args:
            max_retries: Maximum number of retry attempts if read fails
            
        Returns:
            dict with "ok" and "position" keys (compatible with GripperClient)
        """
        if not self._is_connected:
            self.connect()
        
        # Add small delay after connect to allow servo to stabilize
        import time
        time.sleep(0.01)
        
        # Retry logic for reading position (sometimes servo needs time to respond)
        last_error = None
        for attempt in range(max_retries):
            try:
                # Read present position
                dxl_present_position, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(
                    self.port_handler, self.dxl_id, self.ADDR_PRESENT_POSITION
                )
                
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    # Success - convert to angle and normalized value
                    angle_deg = self._dxl_position_to_angle(dxl_present_position)
                    normalized = self._angle_to_normalized(angle_deg)
                    
                    self._last_position = normalized
                    
                    return {
                        "ok": True,
                        "position": normalized,
                        "angle_deg": angle_deg,
                        "dxl_position": dxl_present_position,
                    }
                elif dxl_comm_result != COMM_SUCCESS:
                    last_error = f"Comm error: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
                elif dxl_error != 0:
                    last_error = f"Servo error: {self.packet_handler.getRxPacketError(dxl_error)}"
            except Exception as e:
                last_error = f"Exception: {e}"
            
            # Wait before retry (servo may need time to respond)
            if attempt < max_retries - 1:
                time.sleep(0.05 * (attempt + 1))  # Increasing delay: 50ms, 100ms, 150ms
        
        # All retries failed - return last known position or raise error
        if self._last_position is not None:
            # Return cached position as fallback
            return {
                "ok": False,
                "position": self._last_position,
                "error": last_error,
                "cached": True,
            }
        else:
            raise RuntimeError(f"Failed to read position after {max_retries} attempts: {last_error}")
    
    def ping(self, max_retries: int = 2) -> dict:
        """Ping the servo to check connection."""
        if not self._is_connected:
            self.connect()
        
        import time
        
        # Retry logic for ping
        last_error = None
        for attempt in range(max_retries):
            try:
                dxl_model_number, dxl_comm_result, dxl_error = self.packet_handler.ping(
                    self.port_handler, self.dxl_id
                )
                
                if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                    # Small delay after successful ping to allow servo to be ready
                    time.sleep(0.05)
                    return {"ok": True, "model": dxl_model_number}
                elif dxl_comm_result != COMM_SUCCESS:
                    last_error = self.packet_handler.getTxRxResult(dxl_comm_result)
                elif dxl_error != 0:
                    last_error = self.packet_handler.getRxPacketError(dxl_error)
            except Exception as e:
                last_error = str(e)
            
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(0.1)
        
        return {"ok": False, "error": last_error or "Unknown error"}
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience wrapper matching GripperClient interface
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
    
    def get(self) -> dict:
        """Get gripper position [0.0, 1.0]."""
        return self._gripper.get()
    
    def ping(self) -> dict:
        """Ping gripper."""
        return self._gripper.ping()
    
    def disconnect(self):
        """Disconnect gripper."""
        self._gripper.disconnect()
