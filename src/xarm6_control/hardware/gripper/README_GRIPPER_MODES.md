# Gripper Control Modes

The gripper can be controlled in two ways:

## 1. ROS Mode (Default)
Uses ROS/Teensy microcontroller to control the Dynamixel servo.

**Setup:**
- Teensy must be running the `stepper_only_stack.ino` sketch
- ROS must be running with rosserial connection to Teensy
- Gripper server must be running: `python src/xarm6_control/hardware/gripper/server_async.py`

**Usage:**
```bash
uv run python -m xarm6_control.cli.main --gripper_mode ros
```

## 2. USB Mode (Direct Control)
Direct USB connection to Dynamixel servo, bypassing ROS/Teensy.

**Setup:**
- Install dynamixel-sdk: `pip install dynamixel-sdk pyserial`
- Connect Dynamixel servo directly to USB port (via USB2Dynamixel or similar)
- On Linux, add user to dialout group: `sudo usermod -a -G dialout $USER` (then logout/login)
- Find the port: `ls /dev/ttyUSB*` (Linux) or check Device Manager (Windows)

**Usage:**
```bash
# Linux
uv run python -m xarm6_control.cli.main --gripper_mode usb --gripper_usb_port /dev/ttyUSB0

# Windows
uv run python -m xarm6_control.cli.main --gripper_mode usb --gripper_usb_port COM3
```

## Configuration

The gripper mode can be set via command-line arguments:

- `--gripper_mode`: `"ros"` or `"usb"` (default: `"ros"`)
- `--gripper_usb_port`: Serial port for USB mode (default: `"/dev/ttyUSB0"` on Linux, `"COM3"` on Windows)
- `--gripper_host`: Host for ROS mode TCP server (default: `"127.0.0.1"`)
- `--gripper_port`: Port for ROS mode TCP server (default: `22345`)

## Testing

A comprehensive test script is available to test both modes:

```bash
# Test both modes
python src/xarm6_control/tests/test_gripper_modes.py --mode both

# Test only ROS mode
python src/xarm6_control/tests/test_gripper_modes.py --mode ros

# Test only USB mode (auto-detect port)
python src/xarm6_control/tests/test_gripper_modes.py --mode usb

# Test USB mode with specific port (Linux/WSL)
python src/xarm6_control/tests/test_gripper_modes.py --mode usb --port /dev/ttyUSB0

# Test USB mode with specific port (Windows)
python src/xarm6_control/tests/test_gripper_modes.py --mode usb --port COM3
```

The test script will:
- ✅ Ping the gripper to verify connection
- ✅ Get current position
- ✅ Set position and verify
- ✅ Test full cycle (open → close → open)
- ✅ Provide troubleshooting hints if tests fail

## Notes

- Both modes use the same normalized interface [0.0, 1.0] where 0.0 = open, 1.0 = closed
- USB mode uses Dynamixel protocol version 1.0 (compatible with AX series servos)
- Default servo ID is 1 (can be changed in `dynamixel_usb.py`)
- Default baudrate is 57600 (matching the .ino file)
- On WSL, you may need to access Windows COM ports via `/dev/ttyS*` or use Windows Python directly