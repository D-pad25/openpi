#!/usr/bin/env python3
import socket
import threading
import rospy
from std_msgs.msg import Int16, Float32

class GripperSocketBridge:
    def __init__(self, host='127.0.0.1', port=12345):
        self.pub = rospy.Publisher('/gripper_command', Int16, queue_size=10)
        self.latest_gripper_pos = None
        self._lock = threading.Lock()

        # Subscribe to gripper state (Float32)
        rospy.Subscriber('/gripper_position', Float32, self._gripper_callback)

        self.server_address = (host, port)
        self.server_thread = threading.Thread(target=self.socket_server_loop, daemon=True)
        self.server_thread.start()
        print(f"[🔌] Socket server started at {host}:{port}")

    def _gripper_callback(self, msg):
        with self._lock:
            self.latest_gripper_pos = msg.data

    def socket_server_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(self.server_address)
        server_socket.listen(1)
        print("[🔌] Server listening for incoming connections...")

        while not rospy.is_shutdown():
            client_socket, client_address = server_socket.accept()
            print(f"[🟢] Connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()

    def handle_client(self, client_socket):
        try:
            while not rospy.is_shutdown():
                data = client_socket.recv(1024)
                if not data:
                    break
                message = data.decode().strip()
                print(f"[📥] Received: '{message}'")

                if message.startswith("SET:"):
                    try:
                        gripper_val = float(message.split(":", 1)[1])
                        gripper_val = max(0.0, min(1.0, gripper_val))
                        gripper_int = int(round(gripper_val * 255))
                        self.pub.publish(gripper_int)
                        rospy.loginfo(f"Published gripper value (int): {gripper_int}")
                        client_socket.sendall(f"OK: Published {gripper_int}\n".encode())
                    except ValueError:
                        client_socket.sendall(b"ERROR: Invalid SET value\n")

                elif message == "GET":
                    with self._lock:
                        pos = self.latest_gripper_pos
                    if pos is not None:
                        client_socket.sendall(f"STATE: {pos}\n".encode())
                    else:
                        client_socket.sendall(b"STATE: unavailable\n")

                else:
                    client_socket.sendall(b"ERROR: Unknown command\n")

        except Exception as e:
            print(f"[⚠️] Client error: {e}")
        finally:
            client_socket.close()
            print("[🔌] Client disconnected")


if __name__ == '__main__':
    rospy.init_node('gripper_socket_bridge')
    bridge = GripperSocketBridge()
    rospy.spin()
