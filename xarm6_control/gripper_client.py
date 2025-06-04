import socket
import time

def send_gripper_command(value: float):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(('127.0.0.1', 12345))
            message = f"{value:.3f}\n"
            client.sendall(message.encode())

            response = client.recv(1024).decode()
            print(f"[Client] Sent: {message.strip()} | Received: {response.strip()}")

    except Exception as e:
        print(f"[Client Error] {e}")

if __name__ == "__main__":
    commands = [0.0, 0.25, 0.5, 0.75, 1.0, 0.0]

    for cmd in commands:
        send_gripper_command(cmd)
        time.sleep(1)  # ⏱️ Wait 1 second between commands
