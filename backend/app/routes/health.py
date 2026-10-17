"""GET /api/health"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..registry import ModelRegistry, get_registry
from ..schemas import HealthResponse
from ..settings import get_settings

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    """Liveness probe, plus which device and models are actually available."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        device=str(registry.device),
        models_loaded=registry.ready_count,
        version=settings.version,
    )