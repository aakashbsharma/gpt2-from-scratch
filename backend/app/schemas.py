"""Pydantic models mirroring docs/trace-schema.md.

This module is the server-side contract. Every field here has a counterpart in
``frontend/src/types/trace.ts`` and an entry in the schema doc. If you change a
field, change all three.

The models are also used to *validate real traces*: ``test_api.py`` will build a
trace from the actual model and round-trip it through :class:`TraceModel`, which
is the cheapest way to catch a schema drift between the tracer and the contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Leaf types
# --------------------------------------------------------------------------- #


class TokenInfo(BaseModel):
    """One input token."""

    model_config = ConfigDict(extra="forbid")

    id: int
    piece: str


class VectorSeries(BaseModel):
    """An ``emb_dim``-wide activation tensor, reduced for transport.

    Never a raw vector: per-position L2 norms plus 2D PCA coordinates in the
    basis shared by the whole trace.
    """

    model_config = ConfigDict(extra="forbid")

    norms: list[float] = Field(..., description="Per-position L2 norm, length T.")
    pca: list[list[float]] = Field(
        ..., description="Per-position [x, y] in the shared PCA basis, length T."
    )
    shape: list[int] = Field(..., description="Source shape, [1, T, emb_dim].")
    explained_variance: list[float] = Field(
        ..., description="Length 2, this tensor's variance captured by each PCA axis."
    )
    delta_norms: list[float] | None = Field(
        default=None,
        description="Per-position L2 norm change from the operation's actual input.",
    )


class TopNeuron(BaseModel):
    """One of the strongest activations in the MLP hidden layer."""

    model_config = ConfigDict(extra="forbid")

    index: int
    value: float


class MLPHiddenSummary(BaseModel):
    """Summary of the ``(T, 4 * emb_dim)`` post-GELU activations."""

    model_config = ConfigDict(extra="forbid")

    norms: list[float] = Field(..., description="Per-position L2 norm, length T.")
    top: list[list[TopNeuron]] = Field(
        ..., description="Top-k neurons per position, each sorted descending."
    )
    shape: list[int] = Field(..., description="[1, T, 4 * emb_dim].")
    top_k: int


class HeadAttention(BaseModel):
    """Attention weights and output norm for a single head."""

    model_config = ConfigDict(extra="forbid")

    head: int
    weights: list[list[float]] = Field(
        ...,
        description=(
            "T x T attention matrix, float16-quantized. weights[i][j] is attention "
            "from query i to key j; entries where j > i are exactly 0.0 (causal)."
        ),
    )
    out_norm: list[float] = Field(
        ...,
        description="L2 norm of this head's context vector per position, length T.",
    )


class TensorStats(BaseModel):
    """Summary statistics for a tensor. The raw tensor is never shipped."""

    model_config = ConfigDict(extra="forbid")

    shape: list[int]
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    norm: float


class BlockTrace(BaseModel):
    """Everything observable inside one transformer block."""

    model_config = ConfigDict(extra="forbid")

    index: int
    ln1_out: VectorSeries
    heads: list[HeadAttention]
    resid_after_attn: VectorSeries
    ln2_out: VectorSeries
    mlp_hidden: MLPHiddenSummary
    resid_after_mlp: VectorSeries


class LogitLensPosition(BaseModel):
    """Top tokens from one lens distribution at one prompt position."""

    model_config = ConfigDict(extra="forbid")

    position: int
    top_tokens: list[int]
    top_pieces: list[str]
    top_probs: list[float] = Field(..., description="Top-three probabilities.")


class LogitLensEntry(BaseModel):
    """Readout through the final LayerNorm and output head at one depth."""

    model_config = ConfigDict(extra="forbid")

    layer: int | Literal["emb"]
    positions: list[LogitLensPosition]
    final_token_prob: float
    final_token_rank: int
    entropy: float


class PositionPrediction(BaseModel):
    """The model's real top-1 next-token prediction at one prompt position."""

    model_config = ConfigDict(extra="forbid")

    position: int
    token_id: int
    piece: str
    prob: float


class SamplingCandidate(BaseModel):
    """One token in the traceable portion of the sampling distribution."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    id: int
    token: str
    logit: float
    p_raw: float
    p_temp: float
    kept: bool
    p_final: float


class SamplingTail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    p_raw: float
    p_temp: float


class ChosenToken(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    id: int
    token: str
    p_raw: float
    p_final: float


class SamplingStep(BaseModel):
    """Complete per-step record of the sampling distribution and draw."""

    model_config = ConfigDict(extra="forbid")

    step: int
    temperature: float
    top_k: int | None
    seed: int | None
    candidates: list[SamplingCandidate]
    tail: SamplingTail
    topk_mass_kept: float
    entropy_raw: float
    entropy_final: float
    u: float | None
    cdf_lo: float | None
    cdf_hi: float | None
    chosen: ChosenToken
    greedy_id: int
    was_argmax: bool
    eos_id: int | None
    eos_hit: bool


class TraceModel(BaseModel):
    """A full forward-pass record. Mirrors ``gpt2.trace.Trace.to_dict()``."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    input_token_ids: list[int]
    input_shape: list[int] = Field(..., description="Always [1, T].")
    config: dict[str, Any]

    tokens: list[TokenInfo]
    token_emb: VectorSeries
    pos_emb: VectorSeries
    emb_sum: VectorSeries
    blocks: list[BlockTrace]
    final_norm: VectorSeries
    logits: TensorStats

    position_predictions: list[PositionPrediction]
    top_tokens: list[int]
    top_pieces: list[str]
    top_probs: list[float]

    logit_lens: list[LogitLensEntry]
    sampling: SamplingStep | None = None


# --------------------------------------------------------------------------- #
# REST request/response models
# --------------------------------------------------------------------------- #


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    n_layers: int
    n_heads: int
    emb_dim: int
    context_length: int
    vocab_size: int
    source: Literal["openai", "scratch"]
    ready: bool


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: int
    version: str


class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., max_length=10_000)
    model_id: str = "openai-124m"


class TokenizeResponse(BaseModel):
    model_id: str
    token_ids: list[int]
    pieces: list[str]
    text: str
    num_tokens: int


class TraceRequest(BaseModel):
    """One traced forward pass over an explicit token sequence.

    No sampling is performed, so ``TraceModel.sampling`` is always ``null``.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = "openai-124m"
    token_ids: list[int] = Field(..., min_length=1)
    mlp_top_k: int = Field(default=20, ge=1, le=100)
    lens_top_k: int = Field(default=3, ge=1, le=3)


class WSGenerateRequest(BaseModel):
    """Client message for ``/api/ws/generate``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["generate"] = "generate"
    request_id: str
    model_id: str = "openai-124m"
    prompt: str = Field(..., max_length=10_000)
    max_new_tokens: int = Field(default=20, ge=1, le=200)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None
    trace: bool = True


class WSStepMessage(BaseModel):
    """One streamed generation step."""

    type: Literal["step"] = "step"
    request_id: str
    step: int
    token_id: int
    piece: str
    token_ids: list[int] = Field(
        ..., description="Sequence including the newly sampled token."
    )
    text: str
    finished: bool
    trace: TraceModel | None = None


class WSDoneMessage(BaseModel):
    type: Literal["done"] = "done"
    request_id: str
    token_ids: list[int]
    text: str


class WSErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    request_id: str
    message: str