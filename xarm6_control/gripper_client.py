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

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("[Client] Disconnected")

    def send_command(self, message: str) -> str:
        """Send raw command and get response."""
        try:
            self.connect()
            self.sock.sendall(f"{message.strip()}\n".encode())
            response = self.sock.recv(1024).decode().strip()
            return response
        except Exception as e:
            print(f"[Client Error] {e}")
            return "ERROR"

    def send_gripper_command(self, value: float):
        response = self.send_command(f"SET:{value:.3f}")
        print(f"[Client] Sent gripper command: {value:.3f} | Server: {response}")

    def receive_gripper_position(self):
        response = self.send_command("GET")
        print(f"[Client] Gripper state: {response}")
        return response
