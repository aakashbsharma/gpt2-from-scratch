"""GET /api/models"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..registry import ModelRegistry, get_registry
from ..schemas import ModelListResponse

router = APIRouter(tags=["meta"])


@router.get("/models", response_model=ModelListResponse)
def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelListResponse:
    """Every registered model, including ones that failed to load.

    A model whose checkpoint is missing is still listed with ``ready: false`` so
    the UI can explain why it is unavailable instead of showing an empty list.
    """
    return ModelListResponse(models=registry.list_models())