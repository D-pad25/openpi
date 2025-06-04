import socket

def send_gripper_command(client, value: float):
    message = f"{value:.3f}\n"
    client.sendall(message.encode())
    print(f"[Client] Sent gripper command: {value:.3f}")

def receive_gripper_response(client):
    response = client.recv(1024).decode()
    print(f"[Client] Received: {response.strip()}")

if __name__ == "__main__":
    commands = [0.0, 0.25, 0.5, 0.75, 1.0]

    for val in commands:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.connect(('127.0.0.1', 12345))
                send_gripper_command(client, val)
                receive_gripper_response(client)
        except Exception as e:
            print(f"[Client Error] {e}")
