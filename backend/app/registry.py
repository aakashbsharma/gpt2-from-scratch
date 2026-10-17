"""Model registry: loads each model once and hands out safe access.

Two entries:

* ``openai-124m``  - the real GPT-2 124M, weights from OpenAI's checkpoint.
* ``scratch-10ep`` - a small model trained from scratch by
  ``scripts/train_small.py``. Optional: if the checkpoint file is absent the
  entry is still listed but marked not ready, so the UI can say so instead of
  failing.

Concurrency
-----------
All inference goes through :meth:`ModelRegistry.inference`, which holds a lock
for the duration of a forward pass. This is not paranoia about torch itself:
``GPTModel.forward(return_trace=True)`` writes the traced logits to the shared
``model._trace_logits`` attribute, so two overlapping requests would clobber each
other's logits and a response could serialize the wrong tensor.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from gpt2 import (
    GPTModel,
    GPT2Tokenizer,
    download_and_load_gpt2,
    get_device,
    get_tokenizer,
    load_weights_into_gpt,
)

from .schemas import ModelInfo
from .settings import (
    MODEL_ID_OPENAI,
    MODEL_ID_SCRATCH,
    MODEL_LABELS,
    Settings,
    get_settings,
)


@dataclass
class LoadedModel:
    """One model plus everything the API needs to describe and run it.

    ``model`` is ``None`` when the entry failed to load; ``config`` is kept
    separately so ``info()`` can still describe the model to the UI.
    """

    id: str
    label: str
    source: str  # "openai" | "scratch"
    model: GPTModel | None
    tokenizer: GPT2Tokenizer
    config: dict | None = None
    ready: bool = True
    note: str | None = None

    def __post_init__(self) -> None:
        if self.config is None and self.model is not None:
            self.config = self.model.cfg

    def info(self) -> ModelInfo:
        cfg = self.config or {}
        return ModelInfo(
            id=self.id,
            label=self.label,
            n_layers=int(cfg.get("n_layers", 0)),
            n_heads=int(cfg.get("n_heads", 0)),
            emb_dim=int(cfg.get("emb_dim", 0)),
            context_length=int(cfg.get("context_length", 0)),
            vocab_size=int(cfg.get("vocab_size", 0)),
            source=self.source,  # type: ignore[arg-type]
            ready=self.ready,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def context_length(self) -> int:
        return self.model.context_length


class ModelRegistry:
    """Owns the loaded models and serializes access to them."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._loaded = False
        self.load_errors: dict[str, str] = {}

    # -- loading ------------------------------------------------------------ #

    def load(self, *, force: bool = False) -> "ModelRegistry":
        """Load every model once. Idempotent unless ``force`` is set."""
        with self._load_lock:
            if self._loaded and not force:
                return self
            self._models.clear()
            self.load_errors.clear()

            tokenizer = get_tokenizer()
            device = self.settings.device_obj

            if self.settings.load_openai_model:
                self._load_openai(tokenizer, device)
            self._load_scratch(tokenizer, device)

            self._loaded = True
            return self

    def _load_openai(self, tokenizer: GPT2Tokenizer, device: torch.device) -> None:
        try:
            from gpt2 import GPT_CONFIG_124M

            model = GPTModel(GPT_CONFIG_124M).eval()
            _, params = download_and_load_gpt2(
                "124M", str(self.settings.checkpoint_dir), reuse_existing=True
            )
            load_weights_into_gpt(model, params)
            model.to(device).eval()
            self._models[MODEL_ID_OPENAI] = LoadedModel(
                id=MODEL_ID_OPENAI,
                label=MODEL_LABELS[MODEL_ID_OPENAI],
                source="openai",
                model=model,
                tokenizer=tokenizer,
                ready=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface, don't crash startup
            self.load_errors[MODEL_ID_OPENAI] = f"{type(exc).__name__}: {exc}"

    def _load_scratch(self, tokenizer: GPT2Tokenizer, device: torch.device) -> None:
        path = self.settings.scratch_checkpoint
        try:
            if not Path(path).is_file():
                self._models[MODEL_ID_SCRATCH] = LoadedModel(
                    id=MODEL_ID_SCRATCH,
                    label=MODEL_LABELS[MODEL_ID_SCRATCH],
                    source="scratch",
                    model=None,
                    tokenizer=tokenizer,
                    config=dict(self.settings.model_config_dict),
                    ready=False,
                    note=(
                        f"checkpoint not found at {path}; "
                        "run scripts/train_small.py to create it"
                    ),
                )
                return

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model = GPTModel(ckpt["config"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            self._models[MODEL_ID_SCRATCH] = LoadedModel(
                id=MODEL_ID_SCRATCH,
                label=MODEL_LABELS[MODEL_ID_SCRATCH],
                source="scratch",
                model=model,
                tokenizer=tokenizer,
                ready=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.load_errors[MODEL_ID_SCRATCH] = f"{type(exc).__name__}: {exc}"
            self._models[MODEL_ID_SCRATCH] = LoadedModel(
                id=MODEL_ID_SCRATCH,
                label=MODEL_LABELS[MODEL_ID_SCRATCH],
                source="scratch",
                model=None,
                tokenizer=tokenizer,
                config=dict(self.settings.model_config_dict),
                ready=False,
                note=f"{type(exc).__name__}: {exc}",
            )

    # -- access ------------------------------------------------------------- #

    def get(self, model_id: str) -> LoadedModel:
        """Return a ready model. Raises ``KeyError`` if unknown, ``RuntimeError``
        if known but not loadable, so routes can map them to 404 and 503."""
        if model_id not in self._models:
            raise KeyError(model_id)
        entry = self._models[model_id]
        if not entry.ready or entry.model is None:
            raise RuntimeError(entry.note or f"model {model_id!r} is not ready")
        return entry

    def has(self, model_id: str) -> bool:
        entry = self._models.get(model_id)
        return entry is not None and entry.ready and entry.model is not None

    def list_models(self) -> list[ModelInfo]:
        return [m.info() for m in self._models.values()]

    @property
    def device(self) -> torch.device:
        return self.settings.device_obj

    @property
    def tokenizer(self) -> GPT2Tokenizer:
        return get_tokenizer()

    @contextmanager
    def inference(self, model_id: str) -> Iterator[LoadedModel]:
        """Acquire a model and hold the inference lock for the block's duration.

        Callers must keep this context manager open for the whole forward pass,
        including trace serialization, since the traced logits live on the shared
        model instance.
        """
        entry = self.get(model_id)
        with self._lock:
            yield entry

    @property
    def ready_count(self) -> int:
        return sum(1 for m in self._models.values() if m.ready and m.model is not None)


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Process-wide registry, created lazily."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def set_registry(registry: ModelRegistry | None) -> None:
    """Override the process registry (tests, scripts)."""
    global _registry
    _registry = registry