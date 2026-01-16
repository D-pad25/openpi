@echo off
REM Quick test script for gripper modes (Windows)

echo === Quick Gripper Mode Test ===
echo.

echo Testing ROS Mode...
python src\xarm6_control\tests\test_gripper_modes.py --mode ros
if errorlevel 1 (
    echo.
    echo ROS mode test failed. Make sure:
    echo   1. Gripper server is running: python src\xarm6_control\hardware\gripper\server_async.py
    echo   2. ROS/Teensy is connected and running
    pause
    exit /b 1
)

echo.
echo Testing USB Mode...
REM Try common COM ports
for %%p in (COM3 COM4 COM5 COM6) do (
    python src\xarm6_control\tests\test_gripper_modes.py --mode usb --port %%p 2>nul && (
        echo.
        echo All tests passed!
        pause
        exit /b 0
    )
)

echo.
echo No USB ports found or tests failed. 
echo.
echo To test USB mode manually:
echo   1. Connect Dynamixel servo via USB
echo   2. Check Device Manager for COM port
echo   3. Run: python src\xarm6_control\tests\test_gripper_modes.py --mode usb --port COMx
echo.
pause
