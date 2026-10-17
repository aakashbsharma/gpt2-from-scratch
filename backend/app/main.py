"""FastAPI application.

Models are loaded once in the lifespan handler and reused for the life of the
process; every request borrows them under the registry's inference lock.

Run with::

    uvicorn app.main:app --reload --port 8000

The CORS list defaults to the Vite dev server (http://localhost:5173).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .registry import ModelRegistry, get_registry, set_registry
from .routes import generate, health, models, tokenize
from .settings import get_settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
log = logging.getLogger("gpt2-visualizer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once, before the first request."""
    settings = get_settings()
    registry: ModelRegistry = app.state.registry or get_registry()
    app.state.registry = registry

    log.info("loading models from %s on device %s", settings.checkpoint_dir, settings.device_obj)
    registry.load()

    for model_id in registry.load_errors:
        log.error("model %s failed to load: %s", model_id, registry.load_errors[model_id])
    for info in registry.list_models():
        state = "ready" if info.ready else "NOT READY"
        log.info("  %-14s %s  (%d layers, %d heads)", info.id, state, info.n_layers, info.n_heads)

    if registry.ready_count == 0:
        log.error("no models are available; the API will return errors until this is fixed")

    log.info("startup complete: %d model(s) ready", registry.ready_count)
    try:
        yield
    finally:
        log.info("shutting down")


def create_app(
    *, registry: ModelRegistry | None = None, load_on_startup: bool = True
) -> FastAPI:
    """Build the application.

    ``registry`` lets tests inject a small, pre-loaded model instead of paying
    for the 500 MB OpenAI download. ``load_on_startup=False`` skips the
    lifespan load entirely.
    """
    settings = get_settings()

    app = FastAPI(
        title="gpt2-visualizer",
        version=settings.version,
        summary="Real forward-pass traces from a from-scratch GPT-2",
        lifespan=lifespan if load_on_startup else None,
    )
    app.state.registry = registry

    if registry is not None:
        set_registry(registry)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(tokenize.router, prefix="/api")
    app.include_router(generate.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "gpt2-visualizer",
            "version": settings.version,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()