"""POST /api/tokenize"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..registry import ModelRegistry, get_registry
from ..schemas import TokenizeRequest, TokenizeResponse

router = APIRouter(tags=["tokenizer"])


@router.post("/tokenize", response_model=TokenizeResponse)
def tokenize(
    req: TokenizeRequest, registry: ModelRegistry = Depends(get_registry)
) -> TokenizeResponse:
    """Encode text with the GPT-2 BPE tokenizer.

    Tokenization is shared by every model, so ``model_id`` is accepted for
    symmetry with the other routes but does not change the result.
    """
    try:
        registry.get(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown model {req.model_id!r}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    tokenizer = registry.tokenizer
    ids = tokenizer.encode(req.text)
    return TokenizeResponse(
        model_id=req.model_id,
        token_ids=ids,
        pieces=tokenizer.token_pieces(ids),
        text=tokenizer.decode(ids),
        num_tokens=len(ids),
    )