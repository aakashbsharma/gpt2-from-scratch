"""Drive the generation WebSocket exactly as the frontend does.

Sends the same WSGenerateRequest shape `frontend/src/lib/api.ts` builds, then
prints every frame with a timestamp. Reports whether steps actually arrive, so a
hang can be told apart from "the server had nothing to say".
"""

import asyncio
import json
import sys
import time

import websockets

URL = "ws://127.0.0.1:8000/api/ws/generate"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai-124m"
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0


def request() -> dict:
    return {
        "type": "generate",
        "request_id": "probe-1",
        "model_id": MODEL,
        "prompt": "Every effort moves you",
        "max_new_tokens": 3,
        "temperature": 0.8,
        "top_k": 50,
        "seed": 193,
        "trace": True,
    }


async def main() -> int:
    start = time.monotonic()
    frames = 0
    async with websockets.connect(URL, open_timeout=10) as ws:
        print(f"connected in {time.monotonic() - start:.2f}s")
        await ws.send(json.dumps(request()))
        print("request sent")

        while True:
            remaining = TIMEOUT - (time.monotonic() - start)
            if remaining <= 0:
                print(f"TIMEOUT after {TIMEOUT}s with {frames} frames")
                return 1
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                print(f"TIMEOUT after {TIMEOUT}s with {frames} frames")
                return 1
            except websockets.ConnectionClosed as exc:
                print(f"closed: code={exc.rcvd.code if exc.rcvd else '?'} reason={exc.reason!r}")
                break

            frames += 1
            msg = json.loads(raw)
            elapsed = time.monotonic() - start
            kind = msg.get("type")
            if kind == "step":
                has_trace = msg.get("trace") is not None
                print(
                    f"[{elapsed:6.2f}s] step {msg['step']} piece={msg['piece']!r} "
                    f"finished={msg['finished']} trace={has_trace} "
                    f"bytes={len(raw)}"
                )
            else:
                print(f"[{elapsed:6.2f}s] {kind}: {json.dumps(msg)[:200]}")

    print(f"\ndone: {frames} frames in {time.monotonic() - start:.2f}s")
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
