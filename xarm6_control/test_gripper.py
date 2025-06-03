# main.py

import time
import numpy as np
from xarm_env import XArmRealEnv
import rospy

def test_gripper_only():
    """Test sending commands to the gripper and receiving feedback."""
    print("🔧 Initializing gripper communication...")
    env = XArmRealEnv()

    time.sleep(1.0)  # Allow ROS time to initialize

    test_values = [0.0, 0.2, 0.5, 0.8, 0.0]  # Normalized gripper commands

    print("🧪 Starting gripper command test...")

    try:
        for value in test_values:
            print(f"\n➡️  Sending gripper command: {value:.2f}")
            # env._publish_gripper_command(value)

            time.sleep(2.0)  # Allow time for physical or simulated movement

            feedback = env._get_normalized_gripper_position()
            print(f"📥 Gripper feedback (normalized): {feedback:.3f}")

    except rospy.ROSInterruptException:
        print("🛑 ROS interrupted.")
    except KeyboardInterrupt:
        print("🛑 Test manually stopped.")
    
    print("✅ Gripper test completed.")

if __name__ == "__main__":
    test_gripper_only()
