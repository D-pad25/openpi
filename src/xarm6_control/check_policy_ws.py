# check_policy_ws.py
import sys
import websockets.sync.client

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
URI = f"ws://localhost:{PORT}"

try:
    ws = websockets.sync.client.connect(URI)
    ws.close()
    print("UP")
    sys.exit(0)
except Exception as e:
    print("DOWN:", e)
    sys.exit(1)
