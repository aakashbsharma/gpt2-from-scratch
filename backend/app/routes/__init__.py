"""API routes, one module per resource."""

from . import generate, health, models, tokenize

__all__ = ["generate", "health", "models", "tokenize"]