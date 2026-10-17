"""A WebSocket server that accepts and then never speaks.

Stands in for the "socket opens, connection stays open forever, nothing arrives"
failure so the client-side watchdog can be exercised end to end against a real
socket, rather than inferred from reading the code.

    python scripts/silent_ws_server.py 8001
"""

import asyncio
import sys

from fastapi import FastAPI, WebSocket
import uvicorn

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001

app = FastAPI()


@app.websocket("/api/ws/generate")
async def silent(websocket: WebSocket) -> None:
    await websocket.accept()
    print("accepted a connection; deliberately sending nothing", flush=True)
    # Hold the socket open forever, exactly like a server that never received the
    # request. No timeout, no frames.
    await asyncio.sleep(3600)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
