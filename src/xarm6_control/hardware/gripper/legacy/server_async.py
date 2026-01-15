#!/usr/bin/env python3
import asyncio
import json
import rospy
from std_msgs.msg import Int16, Float32


class GripperSocketServerAsync:
    def __init__(self, host='127.0.0.1', port=22345):
        self.host = host
        self.port = port
        self.latest_gripper_pos = None
        self.pub = rospy.Publisher('/gripper_command', Int16, queue_size=10)
        rospy.Subscriber('/gripper_position', Float32, self._gripper_cb)

    def _gripper_cb(self, msg):
        self.latest_gripper_pos = msg.data

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        print(f"[🟢] Connection from {addr}")

        try:
            while not rospy.is_shutdown():
                data = await reader.readline()
                if not data:
                    break

                try:
                    message = json.loads(data.decode())
                    cmd = message.get("cmd")
                    value = message.get("value")

                    if cmd == "SET":
                        val = max(0.0, min(1.0, float(value)))
                        int_val = int(round(val * 255))
                        self.pub.publish(int_val)
                        response = {"status": "OK", "value": int_val}
                    elif cmd == "GET":
                        response = {"position": self.latest_gripper_pos if self.latest_gripper_pos is not None else "unavailable"}
                    else:
                        response = {"error": "Unknown command"}

                except Exception as e:
                    response = {"error": str(e)}

                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

        except Exception as e:
            print(f"[⚠️] Client error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"[🔌] Disconnected from {addr}")

    async def start_server(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f"[🔌] Gripper Async Server running on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()


if __name__ == '__main__':
    rospy.init_node('gripper_socket_server_async')
    server = GripperSocketServerAsync()
    asyncio.run(server.start_server())
