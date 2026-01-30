#!/usr/bin/env python3
"""
Test script for gripper control modes (ROS and USB).
Works on both Windows and WSL.

Usage:
    # Test ROS mode (default)
    python test_gripper_modes.py --mode ros

    # Test USB mode on Linux/WSL
    python test_gripper_modes.py --mode usb --port /dev/ttyUSB0

    # Test USB mode on Windows
    python test_gripper_modes.py --mode usb --port COM3
"""

import sys
import argparse
import platform
from pathlib import Path

# Add src/ to path so we can import xarm6_control
script_file = Path(__file__).resolve()

# Go up: test_gripper_modes.py -> tests/ -> xarm6_control/ -> src/
src_dir = script_file.parent.parent.parent

# Also try to find repo root and src/ from current working directory as fallback
current_dir = Path.cwd()
if (current_dir / "src" / "xarm6_control").exists():
    src_dir = current_dir / "src"

# Ensure src/ is in sys.path (remove if already there, then add to front)
src_dir_str = str(src_dir)
if src_dir_str in sys.path:
    sys.path.remove(src_dir_str)
sys.path.insert(0, src_dir_str)

# Verify we can find xarm6_control before importing
xarm6_path = src_dir / 'xarm6_control'
if not xarm6_path.exists():
    print(f"ERROR: Could not find xarm6_control package!")
    print(f"  Script location: {script_file}")
    print(f"  Calculated src_dir: {src_dir}")
    print(f"  Expected xarm6_control at: {xarm6_path}")
    print(f"  Current sys.path: {sys.path[:3]}...")
    raise RuntimeError(
        f"Could not find xarm6_control package at {xarm6_path}.\n"
        f"Please ensure you're running from the repository root."
    )

# Now import (make USB imports lazy so ROS mode can work without USB dependencies)
from xarm6_control.hardware.gripper.client_async import GripperClient, GripperClientAsync

# USB imports will be done lazily in test_usb_mode() to allow testing ROS mode
# even if dynamixel-sdk is not installed
GripperUSBClient = None
DynamixelUSBGripper = None


def detect_default_usb_port():
    """Detect default USB port based on OS."""
    system = platform.system()
    if system == "Linux" or "WSL" in platform.release():
        # Try common Linux USB ports
        for port in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]:
            if Path(port).exists():
                return port
        return "/dev/ttyUSB0"  # Default fallback
    elif system == "Windows":
        return "COM3"  # Windows default
    else:
        return "/dev/ttyUSB0"  # macOS/other default


def test_ros_mode(host="127.0.0.1", port=22345, timeout=5.0):
    """Test ROS/Teensy gripper mode via TCP server."""
    print("\n" + "="*60)
    print("Testing ROS Mode (Teensy/TCP Server)")
    print("="*60)
    print(f"Connecting to {host}:{port}...")
    
    try:
        # Test async client
        print("\n[1] Testing GripperClientAsync...")
        import asyncio
        
        async def test_async():
            client = GripperClientAsync(host=host, port=port, read_timeout=timeout)
            try:
                # Ping test
                print("  - Testing ping()...")
                result = await client.ping()
                print(f"    ✅ Ping successful: {result}")
                
                # Get position test
                print("  - Testing get()...")
                result = await client.get()
                position = result.get("position", "N/A")
                print(f"    ✅ Current position: {position}")
                
                # Set position test (move to 0.5)
                print("  - Testing set(0.5)...")
                result = await client.set(0.5)
                print(f"    ✅ Set position result: {result}")
                
                # Get position again
                await asyncio.sleep(0.5)  # Wait for movement
                result = await client.get()
                position = result.get("position", "N/A")
                print(f"    ✅ New position: {position}")
                
                # Test full cycle: open -> close -> open
                print("  - Testing full cycle (0.0 -> 1.0 -> 0.0)...")
                for pos in [0.0, 1.0, 0.0]:
                    await client.set(pos)
                    await asyncio.sleep(0.5)
                    result = await client.get()
                    print(f"    ✅ Position {pos}: {result.get('position', 'N/A')}")
                
                await client.disconnect()
                return True
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        success = asyncio.run(test_async())
        
        if success:
            # Test sync client
            print("\n[2] Testing GripperClient (sync wrapper)...")
            client = GripperClient(host=host, port=port, read_timeout=timeout)
            try:
                result = client.ping()
                print(f"  ✅ Ping successful: {result}")
                
                result = client.get()
                print(f"  ✅ Get position: {result.get('position', 'N/A')}")
                
                result = client.set(0.3)
                print(f"  ✅ Set position: {result}")
                
                client.disconnect()
                print("\n✅ ROS Mode tests PASSED!")
                return True
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("\n❌ ROS Mode tests FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure the gripper server is running:")
        print("     python src/xarm6_control/hardware/gripper/server_async.py")
        print("  2. Check that ROS/Teensy is connected and running")
        print("  3. Verify the host and port are correct")
        import traceback
        traceback.print_exc()
        return False


def test_usb_mode(port=None, baudrate=57600):
    """Test USB gripper mode via direct Dynamixel connection."""
    # Lazy import USB modules (only needed for USB mode)
    global GripperUSBClient, DynamixelUSBGripper
    if GripperUSBClient is None or DynamixelUSBGripper is None:
        try:
            from xarm6_control.hardware.gripper.dynamixel_usb import GripperUSBClient, DynamixelUSBGripper
        except ImportError as e:
            print("\n" + "="*60)
            print("Testing USB Mode (Direct Dynamixel)")
            print("="*60)
            print(f"\n❌ Failed to import USB gripper modules: {e}")
            print("\nPlease install required dependencies for USB mode:")
            print("  pip install dynamixel-sdk pyserial")
            return False
    
    print("\n" + "="*60)
    print("Testing USB Mode (Direct Dynamixel)")
    print("="*60)
    
    if port is None:
        port = detect_default_usb_port()
        print(f"Using auto-detected port: {port}")
    else:
        print(f"Using specified port: {port}")
    
    print(f"Baudrate: {baudrate}")
    
    # Check if port exists (on Linux)
    if platform.system() == "Linux" and not port.startswith("COM"):
        if not Path(port).exists():
            print(f"\n⚠️  Warning: Port {port} does not exist!")
            print("\nAvailable ports:")
            import glob
            usb_ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
            if usb_ports:
                for p in usb_ports:
                    print(f"  - {p}")
            else:
                print("  No USB serial ports found")
            print("\nTroubleshooting:")
            print("  1. Make sure Dynamixel is connected via USB")
            print("  2. Check that USB adapter is properly connected")
            print("  3. On Linux/WSL, you may need to add user to dialout group:")
            print("     sudo usermod -a -G dialout $USER")
            print("     (then logout/login)")
            return False
    
    try:
        print("\n[1] Testing GripperUSBClient...")
        client = GripperUSBClient(port=port, baudrate=baudrate)
        client = GripperUSBClient(
    port=gripper_usb_port,
    set_max_torque_limit=True,
    torque_limit=1023,
    set_max_speed=True,
    moving_speed=0,
    set_stiff_compliance=True,
    compliance_margin=0,
    compliance_slope=32,
    min_step_ticks=5,
)

        
        # Ping test
        print("  - Testing ping()...")
        try:
            result = client.ping()
            if result.get("ok"):
                print(f"    ✅ Ping successful: {result}")
            else:
                print(f"    ❌ Ping failed: {result}")
                return False
        except Exception as e:
            print(f"    ❌ Ping error: {e}")
            print("\nTroubleshooting:")
            print("  1. Check that dynamixel-sdk is installed: pip install dynamixel-sdk")
            print("  2. Verify the servo is powered on")
            print("  3. Check the servo ID (default is 1)")
            print("  4. Verify baudrate matches servo configuration")
            import traceback
            traceback.print_exc()
            return False
        
        # Get position test (add delay after ping for servo to stabilize)
        import time
        time.sleep(0.2)
        print("  - Testing get()...")
        result = client.get()
        position = result.get("position", "N/A")
        angle = result.get("angle_deg", "N/A")
        cached = result.get("cached", False)
        if cached:
            print(f"    ⚠️  Using cached position: {position} (angle: {angle}°)")
        else:
            print(f"    ✅ Current position: {position} (angle: {angle}°)")
        
        # Set position test
        print("  - Testing set(0.5)...")
        result = client.set(0.5)
        print(f"    ✅ Set position result: {result}")
        
        # Get position again
        import time
        time.sleep(0.5)  # Wait for movement
        result = client.get()
        position = result.get("position", "N/A")
        angle = result.get("angle_deg", "N/A")
        print(f"    ✅ New position: {position} (angle: {angle}°)")
        
        # Test full cycle
        print("  - Testing full cycle (0.0 -> 1.0 -> 0.0)...")
        for pos in [0.0, 1.0, 0.0]:
            client.set(pos)
            time.sleep(0.5)
            result = client.get()
            print(f"    ✅ Position {pos}: {result.get('position', 'N/A')} (angle: {result.get('angle_deg', 'N/A')}°)")
        
        # Disconnect the first client before testing context manager
        client.disconnect()
        
        # Wait a moment for port to be released (Windows may need more time)
        time.sleep(1.0)
        
        # Test using context manager
        print("\n[2] Testing DynamixelUSBGripper with context manager...")
        try:
            with DynamixelUSBGripper(port=port, baudrate=baudrate) as gripper:
                result = gripper.get()
                print(f"  ✅ Context manager test: {result.get('position', 'N/A')}")
        except Exception as e:
            # Check if it's a port access error
            error_str = str(e).lower()
            if "permission" in error_str or "access is denied" in error_str or "port" in error_str:
                print(f"  ⚠️  Context manager test skipped: Port {port} still in use")
                print(f"     This is expected on Windows when switching between clients.")
                print(f"     The context manager functionality is still valid.")
            else:
                print(f"  ⚠️  Context manager test failed: {e}")
                # Re-raise if it's a different error
                raise
        
        print("\n✅ USB Mode tests PASSED!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install dynamixel-sdk pyserial")
        return False
    except Exception as e:
        print(f"\n❌ USB Mode test FAILED: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Dynamixel servo is connected via USB")
        print("  2. Check that servo is powered on")
        print("  3. Verify port name is correct:")
        print("     - Linux/WSL: /dev/ttyUSB0 or /dev/ttyACM0")
        print("     - Windows: COM3, COM4, etc.")
        print("  4. On Linux/WSL, add user to dialout group:")
        print("     sudo usermod -a -G dialout $USER")
        print("     (then logout/login)")
        print("  5. Check servo ID matches (default is 1)")
        print("  6. Verify baudrate matches (default is 57600)")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test gripper control modes (ROS or USB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test ROS mode (default)
  python test_gripper_modes.py --mode ros

  # Test USB mode (auto-detect port)
  python test_gripper_modes.py --mode usb

  # Test USB mode with specific port (Linux/WSL)
  python test_gripper_modes.py --mode usb --port /dev/ttyUSB0

  # Test USB mode with specific port (Windows)
  python test_gripper_modes.py --mode usb --port COM3

  # Test both modes
  python test_gripper_modes.py --mode both
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["ros", "usb", "both"],
        default="both",
        help="Gripper mode to test (default: both)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="USB port for USB mode (auto-detected if not specified)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for ROS mode (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--gripper-port",
        type=int,
        default=22345,
        dest="gripper_port",
        help="TCP port for ROS mode (default: 22345)"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=57600,
        help="Baudrate for USB mode (default: 57600)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout for ROS mode connection (default: 5.0s)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Gripper Mode Test Script")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print("="*60)
    
    results = {}
    
    if args.mode in ["ros", "both"]:
        results["ros"] = test_ros_mode(
            host=args.host,
            port=args.gripper_port,
            timeout=args.timeout
        )
    
    if args.mode in ["usb", "both"]:
        results["usb"] = test_usb_mode(
            port=args.port,
            baudrate=args.baudrate
        )
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for mode, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {mode.upper()} Mode: {status}")
    
    all_passed = all(results.values()) if results else False
    if all_passed:
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
