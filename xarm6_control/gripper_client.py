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
            print("[Client] Connected to server")

    def send_gripper_command(self, value: float):
        try:
            self.connect()
            message = f"{value:.3f}\n"
            self.sock.sendall(message.encode())
            print(f"[Client] Sent: {message.strip()}")
        except Exception as e:
            print(f"[Client Error] send: {e}")

    def receive_response(self):
        try:
            self.connect()
            response = self.sock.recv(1024).decode().strip()
            print(f"[Client] Received: {response}")
            return response
        except Exception as e:
            print(f"[Client Error] receive: {e}")
            return None

    def close(self):
        if self.sock:
            self.sock.close()
            print("[Client] Disconnected")
            self.sock = None


import time

if __name__ == "__main__":
    client = GripperClient()

    # Step 1: Receive current gripper state
    client.receive_response()

    # Step 2: Send a few commands
    for cmd in [0.25, 0.5, 0.75]:
        client.send_gripper_command(cmd)
        time.sleep(1)
        client.receive_response()

    # Step 3: Close connection
    client.close()
