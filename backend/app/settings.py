"""Paths, device and registry configuration for the API.

Everything is overridable through environment variables so the same app can run
locally, in Docker and in tests without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from gpt2 import GPT_CONFIG_124M, GPT_CONFIG_SMALL, get_device

# backend/app/settings.py -> backend/app -> backend -> gpt2-visualizer
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

APP_VERSION = "0.4.0"

# Vite dev server. Overridable for other frontends.
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

MODEL_ID_OPENAI = "openai-124m"
MODEL_ID_SCRATCH = "scratch-10ep"

MODEL_LABELS = {
    MODEL_ID_OPENAI: "GPT-2 124M (OpenAI weights)",
    MODEL_ID_SCRATCH: "GPT-2 small, 10 epochs from scratch on The Verdict",
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _find_text_file() -> Path:
    """Locate the training corpus, preferring an explicit override.

    The corpus lives at the repository root, but it has also been seen beside
    the backend, so both are probed before giving up.
    """
    override = os.environ.get("GPT2_TEXT_FILE")
    if override:
        return Path(override)
    for candidate in (
        BACKEND_DIR / "the-verdict.txt",
        PROJECT_DIR / "the-verdict.txt",
        REPO_ROOT / "the-verdict.txt",
    ):
        if candidate.is_file():
            return candidate
    return REPO_ROOT / "the-verdict.txt"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    checkpoint_dir: Path
    scratch_checkpoint: Path
    text_file: Path
    device: str
    use_small_model: bool
    cors_origins: list[str] = field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    version: str = APP_VERSION
    # Loading the 124M weights costs ~500MB of downloads; tests opt out.
    load_openai_model: bool = True

    @property
    def model_config_dict(self) -> dict:
        """Config used when the app constructs a model itself.

        The OpenAI entry always needs the real 124M shape to load the
        checkpoint. ``GPT2_USE_SMALL_MODEL`` only affects the scratch model, and
        lets tests run without the 500MB download.
        """
        return GPT_CONFIG_SMALL if self.use_small_model else GPT_CONFIG_124M

    @property
    def device_obj(self):
        return get_device(self.device)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        checkpoint_dir=Path(
            os.environ.get("GPT2_CHECKPOINT_DIR", BACKEND_DIR / "checkpoints" / "gpt2")
        ),
        scratch_checkpoint=Path(
            os.environ.get(
                "GPT2_SCRATCH_CHECKPOINT", BACKEND_DIR / "checkpoints" / "scratch.pt"
            )
        ),
        text_file=_find_text_file(),
        device=os.environ.get("GPT2_DEVICE", "auto"),
        use_small_model=_env_bool("GPT2_USE_SMALL_MODEL", False),
        cors_origins=[
            o.strip()
            for o in os.environ.get(
                "GPT2_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)
            ).split(",")
            if o.strip()
        ],
        load_openai_model=_env_bool("GPT2_LOAD_OPENAI_MODEL", True),
    )


def reset_settings_cache() -> None:
    """Clear the cached settings. Used by tests that patch the environment."""
    get_settings.cache_clear()