#!/bin/bash
# Quick test script for gripper modes (Linux/WSL)

echo "=== Quick Gripper Mode Test ==="
echo ""

# Check if we're in WSL
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo "⚠️  Running in WSL - USB ports may be limited"
    echo "   Consider running USB tests from Windows directly"
    echo ""
fi

# Test ROS mode
echo "Testing ROS Mode..."
python src/xarm6_control/tests/test_gripper_modes.py --mode ros || {
    echo ""
    echo "ROS mode test failed. Make sure:"
    echo "  1. Gripper server is running: python src/xarm6_control/hardware/gripper/server_async.py"
    echo "  2. ROS/Teensy is connected and running"
    exit 1
}

echo ""
echo "Testing USB Mode..."
# Try common ports
for port in /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0; do
    if [ -e "$port" ]; then
        echo "Found port: $port"
        python src/xarm6_control/tests/test_gripper_modes.py --mode usb --port "$port" && exit 0
    fi
done

echo ""
echo "⚠️  No USB ports found. USB mode test skipped."
echo ""
echo "To test USB mode manually:"
echo "  1. Connect Dynamixel servo via USB"
echo "  2. Find port: ls /dev/ttyUSB* /dev/ttyACM*"
echo "  3. Run: python src/xarm6_control/tests/test_gripper_modes.py --mode usb --port <PORT>"
echo ""
