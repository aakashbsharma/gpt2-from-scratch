"""From-scratch GPT-2 (124M) with forward-pass tracing.

Pure model package: no web framework, no FastAPI, no app-level config. The API
layer imports from here and nothing here imports from the API layer.

Typical use::

    from gpt2 import GPTModel, GPT_CONFIG_124M, get_tokenizer
    from gpt2 import download_and_load_gpt2, load_weights_into_gpt

    model = GPTModel(GPT_CONFIG_124M)
    settings, params = download_and_load_gpt2("124M", "checkpoints/gpt2")
    load_weights_into_gpt(model, params)

    logits, trace = model.forward_with_trace(token_ids, get_tokenizer())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    ALLOWED_MODEL_SIZES,
    GPT_CONFIG_124M,
    GPT_CONFIG_1558M,
    GPT_CONFIG_355M,
    GPT_CONFIG_774M,
    GPT_CONFIG_SMALL,
    MODEL_CONFIGS,
    SMALL_RUN_CONFIG,
    describe_device,
    get_device,
    get_model_config,
)
from .generate import (
    StreamStep,
    generate,
    generate_stream,
    generate_text_simple,
)
from .layers import FeedForward, GELU, LayerNorm, MultiHeadAttention
from .model import GPTModel, TransformerBlock
from .tokenizer import GPT2Tokenizer, get_tokenizer
from .trace import (
    BlockTrace,
    HeadAttention,
    LogitLensEntry,
    MLPHiddenSummary,
    PCAProjection,
    PositionPrediction,
    SamplingCandidate,
    SamplingStep,
    TensorStats,
    TokenInfo,
    TopNeuron,
    Trace,
    VectorSeries,
    fit_pca,
    logit_lens,
)
from .weights import assign, load_weights_into_gpt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .download import download_and_load_gpt2

__all__ = [
    "ALLOWED_MODEL_SIZES",
    "BlockTrace",
    "FeedForward",
    "GELU",
    "GPT2Tokenizer",
    "GPT_CONFIG_124M",
    "GPT_CONFIG_1558M",
    "GPT_CONFIG_355M",
    "GPT_CONFIG_774M",
    "GPT_CONFIG_SMALL",
    "GPTModel",
    "HeadAttention",
    "LayerNorm",
    "LogitLensEntry",
    "MODEL_CONFIGS",
    "MLPHiddenSummary",
    "MultiHeadAttention",
    "PCAProjection",
    "PositionPrediction",
    "SMALL_RUN_CONFIG",
    "SamplingCandidate",
    "SamplingStep",
    "StreamStep",
    "TensorStats",
    "TokenInfo",
    "TopNeuron",
    "Trace",
    "TransformerBlock",
    "VectorSeries",
    "assign",
    "describe_device",
    "download_and_load_gpt2",
    "fit_pca",
    "generate",
    "generate_stream",
    "generate_text_simple",
    "get_device",
    "get_model_config",
    "get_tokenizer",
    "load_weights_into_gpt",
    "logit_lens",
]


def __getattr__(name: str):
    """Expose the download helpers lazily.

    ``gpt2.download`` pulls in ``requests``/``tqdm`` and its checkpoint reader
    pulls in TensorFlow; resolving it on first attribute access keeps ``import
    gpt2`` cheap for callers that only need the model.
    """
    if name in ("download_and_load_gpt2", "download_file", "load_gpt2_params_from_tf_ckpt"):
        from . import download as _download

        return getattr(_download, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")