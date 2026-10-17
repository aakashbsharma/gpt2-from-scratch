"""FastAPI application layer.

- ``schemas``  - the wire contract, mirroring docs/trace-schema.md
- ``settings`` - paths, device, CORS, registry config
- ``registry`` - loads each model once and serializes access to it
- ``routes``   - health, models, tokenize, trace/generate
- ``main``     - app factory, lifespan, CORS
"""

__all__ = ["main", "registry", "routes", "schemas", "settings"]