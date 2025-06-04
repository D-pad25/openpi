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
