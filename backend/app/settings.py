"""Paths, device and registry configuration for the API.

Everything is overridable through environment variables so the same app can run
locally, in Docker and in tests without code changes. Locally, a ``.env`` file
next to ``docker-compose.yml`` (or ``backend/.env``) is read at import time;
values already present in the environment always win. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from gpt2 import (
    ALLOWED_MODEL_SIZES,
    GPT_CONFIG_124M,
    GPT_CONFIG_SMALL,
    get_device,
    get_model_config,
)

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
    "http://localhost:5174",
]

# Registry id for the OpenAI-weights entry at the *default* size. The registry
# derives the real id from ``Settings.openai_model_id`` (``openai-<size>``), so
# with the default ``GPT2_MODEL_SIZE=124M`` this constant is still the id in use;
# it stays a constant for tests and for request defaults in ``schemas``.
MODEL_ID_OPENAI = "openai-124m"
MODEL_ID_SCRATCH = "scratch-10ep"

MODEL_LABELS = {
    MODEL_ID_OPENAI: "GPT-2 124M (OpenAI weights)",
    MODEL_ID_SCRATCH: "GPT-2 small, 10 epochs from scratch on The Verdict",
}

DEFAULT_MODEL_SIZE = "124M"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _load_dotenv(path: Path) -> None:
    """Read a ``.env`` file into ``os.environ``, without overriding real env vars.

    Deliberately dependency-free: the project pins no ``python-dotenv``, and the
    only feature needed here is ``KEY=value`` lines. ``#`` comments, an optional
    ``export`` prefix and surrounding quotes are tolerated, which covers
    everything a hand-written ``.env`` tends to contain. Anything already set in
    the environment wins, so docker-compose / CI / an inline ``VAR=x cmd`` keep
    their usual precedence over the file.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'\"")


# Local overrides: the project directory first (the same `.env` docker compose
# reads for `${...}` interpolation, and where `.env.example` lives), then
# ``backend/.env`` since the backend is often run with ``cd backend``. Both are
# gitignored; see ``.env.example``. Note ``REPO_ROOT`` is deliberately *not*
# probed: it is the checkout's parent, outside the project.
for _dotenv_file in (PROJECT_DIR / ".env", BACKEND_DIR / ".env"):
    _load_dotenv(_dotenv_file)
del _dotenv_file


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
    # Published OpenAI checkpoint size to download and load; picks both the
    # download URL and the architecture config, so the two can never disagree.
    model_size: str = DEFAULT_MODEL_SIZE
    cors_origins: list[str] = field(default_factory=lambda: list(DEFAULT_CORS_ORIGINS))
    version: str = APP_VERSION
    # Loading the 124M weights costs ~500MB of downloads; tests opt out.
    load_openai_model: bool = True

    @property
    def model_config_dict(self) -> dict:
        """Config shown for a model the app cannot load a checkpoint for.

        Only the scratch entry uses it, as a placeholder so the UI can still
        describe the model while its checkpoint is missing. ``GPT2_USE_SMALL_MODEL``
        switches it to the tiny test config, which lets tests run without ever
        touching a real checkpoint.
        """
        return GPT_CONFIG_SMALL if self.use_small_model else GPT_CONFIG_124M

    @property
    def openai_model_id(self) -> str:
        """Registry id for the OpenAI-weights entry; tracks the configured size."""
        return f"openai-{self.model_size.lower()}"

    @property
    def openai_model_label(self) -> str:
        return f"GPT-2 {self.model_size} (OpenAI weights)"

    @property
    def openai_model_config(self) -> dict:
        """Architecture matching the configured checkpoint size.

        The model is built from this config and the checkpoint is loaded into it
        afterwards, so they must agree; ``load_weights_into_gpt`` asserts the
        shapes on every tensor.
        """
        return get_model_config(self.model_size)

    @property
    def device_obj(self):
        return get_device(self.device)


def _model_size_from_env() -> str:
    """Read ``GPT2_MODEL_SIZE``, failing fast on sizes OpenAI never published.

    An empty value means "unset", so a commented-out or blank line in ``.env``
    is harmless.
    """
    size = os.environ.get("GPT2_MODEL_SIZE") or DEFAULT_MODEL_SIZE
    size = size.strip()
    if size not in ALLOWED_MODEL_SIZES:
        raise ValueError(
            f"GPT2_MODEL_SIZE={size!r} is not a published GPT-2 size; "
            f"expected one of {', '.join(ALLOWED_MODEL_SIZES)}"
        )
    return size


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        checkpoint_dir=Path(
            os.environ.get("GPT2_CHECKPOINT_DIR") or BACKEND_DIR / "checkpoints" / "gpt2"
        ),
        scratch_checkpoint=Path(
            os.environ.get("GPT2_SCRATCH_CHECKPOINT")
            or BACKEND_DIR / "checkpoints" / "scratch.pt"
        ),
        text_file=_find_text_file(),
        device=os.environ.get("GPT2_DEVICE") or "auto",
        use_small_model=_env_bool("GPT2_USE_SMALL_MODEL", False),
        model_size=_model_size_from_env(),
        cors_origins=[
            o.strip()
            for o in (
                os.environ.get("GPT2_CORS_ORIGINS") or ",".join(DEFAULT_CORS_ORIGINS)
            ).split(",")
            if o.strip()
        ],
        load_openai_model=_env_bool("GPT2_LOAD_OPENAI_MODEL", True),
    )


def reset_settings_cache() -> None:
    """Clear the cached settings. Used by tests that patch the environment."""
    get_settings.cache_clear()