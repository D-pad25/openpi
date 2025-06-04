import socket

class GripperClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print("[Client] Connected to gripper server")

    def send_gripper_command(self, value: float):
        try:
            self.connect()
            command = f"SET:{value:.3f}\n"
            self.sock.sendall(command.encode())
            print(f"[Client] Sent gripper command: {value:.3f}")
        except Exception as e:
            print(f"[Client Error] send: {e}")

    def receive_gripper_position(self):
        try:
            self.connect()
            self.sock.sendall(b"GET\n")
            response = self.sock.recv(1024).decode().strip()
            print(f"[Client] Received gripper state: {response}")
            return response
        except Exception as e:
            print(f"[Client Error] receive: {e}")
            return None

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[Client] Disconnected")

# Example usage
if __name__ == "__main__":
    client = GripperClient()
    client.send_gripper_command(0.5)
    client.receive_gripper_position()
    client.close()
