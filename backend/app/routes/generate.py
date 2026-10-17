"""POST /api/trace and WS /api/ws/generate.

Two ways to get traces:

* ``POST /api/trace`` runs a single forward pass over a given token sequence and
  returns the trace. No sampling, so ``sampling`` is ``null``.
* ``WS /api/ws/generate`` generates, streaming one message per new token. Each
  message carries that step's real forward-pass trace.

Every forward pass runs under ``torch.no_grad()`` and inside the registry's
inference lock, so concurrent clients cannot interleave on a model.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from gpt2 import generate_stream

from ..registry import ModelRegistry, get_registry
from ..schemas import (
    TraceModel,
    TraceRequest,
    WSDoneMessage,
    WSErrorMessage,
    WSGenerateRequest,
    WSStepMessage,
)

router = APIRouter(tags=["generation"])
log = logging.getLogger(__name__)

# Cap the prompt for a single traced pass. Attention is quadratic in length, so
# an unbounded prompt is a memory problem, not just a slow one.
MAX_TRACE_TOKENS = 1024


@router.post("/trace", response_model=TraceModel)
def trace_once(
    req: TraceRequest, registry: ModelRegistry = Depends(get_registry)
) -> dict[str, Any]:
    """One traced forward pass over an explicit token sequence."""
    try:
        registry.get(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown model {req.model_id!r}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    with registry.inference(req.model_id) as entry:
        max_tokens = min(entry.context_length, MAX_TRACE_TOKENS)
        if len(req.token_ids) > max_tokens:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"prompt has {len(req.token_ids)} tokens but the model context "
                    f"is {max_tokens}"
                ),
            )

        ids = torch.tensor([req.token_ids], dtype=torch.long, device=entry.device)
        with torch.no_grad():
            result = entry.model.forward(
                ids,
                return_trace=True,
                tokenizer=entry.tokenizer,
                model_id=entry.id,
                mlp_top_k=req.mlp_top_k,
                lens_top_k=req.lens_top_k,
            )
        return result.to_dict()


# --------------------------------------------------------------------------- #
# WebSocket streaming
# --------------------------------------------------------------------------- #


@router.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket) -> None:
    """Stream one message per generated token, then a ``done`` message.

    Generation is synchronous CPU-bound work, so each step is run in a thread
    pool. That keeps the event loop free to accept other clients and to notice a
    disconnect while a long generation is in flight.
    """
    await websocket.accept()
    request_id = "unknown"

    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 - malformed frame
        await websocket.send_json(
            WSErrorMessage(request_id="unknown", message=f"invalid message: {exc}").model_dump()
        )
        await websocket.close(code=1003)
        return

    try:
        req = WSGenerateRequest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json(
            WSErrorMessage(request_id=str(payload.get("request_id", "unknown")),
                           message=f"invalid request: {exc}").model_dump()
        )
        await websocket.close(code=1008)
        return

    request_id = req.request_id
    registry = get_registry()

    try:
        registry.get(req.model_id)
    except KeyError:
        await websocket.send_json(
            WSErrorMessage(request_id=request_id,
                           message=f"unknown model {req.model_id!r}").model_dump()
        )
        await websocket.close(code=1008)
        return
    except RuntimeError as exc:
        await websocket.send_json(
            WSErrorMessage(request_id=request_id, message=str(exc)).model_dump()
        )
        await websocket.close(code=1011)
        return

    try:
        with registry.inference(req.model_id) as entry:
            generator = generate_stream(
                entry.model,
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                trace=req.trace,
                tokenizer=entry.tokenizer,
                device=entry.device,
                seed=req.seed,
                model_id=entry.id,
            )

            validated_contract_once = False
            last_ids: list[int] = []
            last_text = ""

            while True:
                step = await run_in_threadpool(_next_or_none, generator)
                if step is None:
                    break

                payload = _step_payload(req, step)
                if req.trace and not validated_contract_once:
                    # Validate the first trace against the Pydantic contract so
                    # schema drift fails loudly at the first step instead of
                    # quietly shipping a payload the frontend cannot read.
                    TraceModel.model_validate(payload["trace"])
                    validated_contract_once = True

                await websocket.send_json(payload)
                last_ids = step.token_ids
                last_text = step.text

                if step.finished:
                    break
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("generation failed for request %s", request_id)
        try:
            await websocket.send_json(
                WSErrorMessage(request_id=request_id,
                               message=f"{type(exc).__name__}: {exc}").model_dump()
            )
        except Exception:  # noqa: BLE001 - socket already gone
            pass
        await websocket.close(code=1011)
        return

    await websocket.send_json(
        WSDoneMessage(
            request_id=request_id, token_ids=last_ids, text=last_text
        ).model_dump()
    )
    await websocket.close(code=1000)


def _next_or_none(generator):
    """Pull one step, returning ``None`` when the generator is exhausted."""
    try:
        return next(generator)
    except StopIteration:
        return None


def _step_payload(req: WSGenerateRequest, step) -> dict[str, Any]:
    """Build one ``WSStepMessage``.

    The trace dict is passed straight through rather than re-validated on every
    step: at ~250 KB per step, Pydantic validation would dominate the streaming
    latency. The first step of each generation is validated instead.
    """
    message = WSStepMessage(
        request_id=req.request_id,
        step=step.step,
        token_id=step.token_id,
        piece=step.piece,
        token_ids=step.token_ids,
        text=step.text,
        finished=step.finished,
        trace=step.trace.to_dict() if step.trace is not None else None,
    )
    return message.model_dump()