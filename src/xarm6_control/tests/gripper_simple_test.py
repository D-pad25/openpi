#!/usr/bin/env python3
"""
Simple sequence test for gripper async client.
Moves to 20, 0, 100, 200, 200 (in 0–255 scale),
with 2 second delays between each command.
"""

import asyncio
from xarm6_control.hardware.gripper.client_async import GripperClientAsync

def to_norm(val: int) -> float:
    """Convert 0–255 int to normalized float in [0,1]."""
    return max(0.0, min(1.0, val / 255.0))

async def main():
    cli = GripperClientAsync(host="127.0.0.1", port=22345)

    await cli.ping()  # ensure server alive

    sequence = [20, 0, 100, 200, 200]
    for val in sequence:
        norm_val = to_norm(val)
        print(f"➡️  Setting gripper to {val} (norm={norm_val:.3f})")
        resp = await cli.set(norm_val)
        print("  Response:", resp)

        # get feedback
        feedback = await cli.get()
        print("  Feedback:", feedback)

        await asyncio.sleep(2.0)

    await cli.disconnect()
    print("✅ Sequence complete.")

if __name__ == "__main__":
    asyncio.run(main())
