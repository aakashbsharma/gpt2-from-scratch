"""Trace containers, PCA projection and logit lens.

A :class:`Trace` is the complete record of one real forward pass. Everything in
it is derived from tensors that the forward pass actually computed; nothing is
synthesized, interpolated or hand-animated.

Two rules shape the design:

**Never ship a raw activation vector.** The residual stream is 768-dimensional
per position. Sending those would be megabytes per step. Instead each
``emb_dim``-wide tensor is reduced to a per-position L2 norm plus a 2D PCA
projection, with the PCA basis fitted once per trace so that points from
different layers are directly comparable in the same coordinate frame.

**Never ship un-rounded floats.** Every float is rounded to 4 decimals, and
attention weights are quantized to float16 first, which is plenty of precision
for values in ``[0, 1]`` and roughly halves the attention payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

# --- rounding / serialization helpers ------------------------------------- #

FLOAT_PRECISION = 4
# Guard against NaN/Infinity reaching JSON, which is not valid JSON.
MAX_MAGNITUDE = 1e4


def r4(x: Any) -> float:
    """Round a scalar to 4 decimals, mapping NaN/Inf to finite stand-ins."""
    v = float(x)
    if math.isnan(v):
        return 0.0
    if math.isinf(v):
        return MAX_MAGNITUDE if v > 0 else -MAX_MAGNITUDE
    v = round(v, FLOAT_PRECISION)
    if abs(v) > MAX_MAGNITUDE:
        v = MAX_MAGNITUDE if v > 0 else -MAX_MAGNITUDE
    # Normalize negative zero so the JSON has no "-0.0".
    return v + 0.0


def r4_list(xs: Any) -> list[float]:
    return [r4(v) for v in xs]


def f16_list(xs: Any) -> list[float]:
    """Quantize through float16, then round, for compact attention matrices."""
    return r4_list(torch.as_tensor(xs, dtype=torch.float32).to(torch.float16).tolist())


def f16_nested(rows: Any) -> list[list[float]]:
    return [f16_list(row) for row in rows]


# --- basic summaries ------------------------------------------------------- #


@dataclass
class TensorStats:
    """Summary statistics for a tensor, cheap enough to compute for every layer."""

    shape: list[int]
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    norm: float

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "TensorStats":
        flat = t.detach().float().flatten()
        return cls(
            shape=list(t.shape),
            dtype=str(t.dtype).replace("torch.", ""),
            min=r4(flat.min()),
            max=r4(flat.max()),
            mean=r4(flat.mean()),
            std=r4(flat.std()) if flat.numel() > 1 else 0.0,
            norm=r4(flat.norm()),
        )


@dataclass
class VectorSeries:
    """An ``(T, emb_dim)`` activation tensor, reduced for transport.

    This is the replacement for shipping raw embeddings/residuals: per-position
    L2 norms plus 2D PCA coordinates in a basis shared by the whole trace.
    """

    # Per-position L2 norm. Length T.
    norms: list[float]
    # Per-position [x, y] in the trace's PCA basis. Length T.
    pca: list[list[float]]
    # Shape of the source tensor, e.g. [1, 4, 768].
    shape: list[int]
    # Fraction of variance captured by the 2D projection.
    explained_variance: list[float]
    # Per-position change in L2 norm from the operation's actual input.
    delta_norms: list[float] | None = None

    @property
    def num_positions(self) -> int:
        return len(self.norms)


@dataclass
class TokenInfo:
    """One input token."""

    id: int
    piece: str


@dataclass
class HeadAttention:
    """Attention weights and output norm for a single head.

    ``weights[i][j]`` is attention from query position ``i`` to key position
    ``j``. Post-softmax, float16-quantized. Because of the causal mask every
    entry where ``j > i`` is exactly ``0.0`` and each row sums to ``1.0``.
    """

    head: int
    weights: list[list[float]]
    # L2 norm of this head's context vector per position. Length T.
    out_norm: list[float]


@dataclass
class TopNeuron:
    """One of the strongest activations in the MLP hidden layer."""

    index: int
    value: float


@dataclass
class MLPHiddenSummary:
    """Summary of the ``(T, 4 * emb_dim)`` post-GELU activations.

    Raw size is 12,288 floats per position, so only the top ``top_k`` neurons
    per position and the L2 norm are kept.
    """

    # L2 norm per position. Length T.
    norms: list[float]
    # Top activating neurons per position, each sorted by descending value.
    top: list[list[TopNeuron]]
    shape: list[int]
    top_k: int


@dataclass
class BlockTrace:
    """Everything observable inside one transformer block."""

    index: int
    ln1_out: VectorSeries
    heads: list[HeadAttention]
    resid_after_attn: VectorSeries
    ln2_out: VectorSeries
    mlp_hidden: MLPHiddenSummary
    resid_after_mlp: VectorSeries


@dataclass
class LogitLensPosition:
    """Top tokens from one lens distribution at one prompt position."""

    position: int
    top_tokens: list[int]
    top_pieces: list[str]
    top_probs: list[float]


@dataclass
class LogitLensEntry:
    """Readout through the final LayerNorm and output head at one depth."""

    layer: int | str
    positions: list[LogitLensPosition]
    final_token_prob: float
    final_token_rank: int
    entropy: float


@dataclass
class PositionPrediction:
    """The model's real top-1 next-token prediction at one prompt position."""

    position: int
    token_id: int
    piece: str
    prob: float


@dataclass
class SamplingCandidate:
    """One token in the traceable portion of the sampling distribution."""

    rank: int
    id: int
    token: str
    logit: float
    p_raw: float
    p_temp: float
    kept: bool
    p_final: float


@dataclass
class SamplingTail:
    """Probability mass beyond the explicitly listed candidate tokens."""

    p_raw: float
    p_temp: float


@dataclass
class ChosenToken:
    """The token selected by the inverse-CDF draw."""

    rank: int
    id: int
    token: str
    p_raw: float
    p_final: float


@dataclass
class SamplingStep:
    """Complete per-step record of the sampling distribution and draw."""

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

    @property
    def sampled_token_id(self) -> int:
        return self.chosen.id

    @property
    def sampled_piece(self) -> str:
        return self.chosen.token

    @property
    def sampled_prob(self) -> float:
        return self.chosen.p_final

    @property
    def greedy(self) -> bool:
        return self.was_argmax


@dataclass
class Trace:
    """A full forward-pass record."""

    model_id: str
    input_token_ids: list[int]
    input_shape: list[int]
    config: dict[str, Any]

    tokens: list[TokenInfo] = field(default_factory=list)
    token_emb: VectorSeries | None = None
    pos_emb: VectorSeries | None = None
    emb_sum: VectorSeries | None = None
    blocks: list[BlockTrace] = field(default_factory=list)
    final_norm: VectorSeries | None = None
    logits: TensorStats | None = None

    # Real next-token prediction at every prompt position (final layer).
    position_predictions: list[PositionPrediction] = field(default_factory=list)
    # Convenience: top-5 at the last position, same as the final lens entry.
    top_tokens: list[int] = field(default_factory=list)
    top_pieces: list[str] = field(default_factory=list)
    top_probs: list[float] = field(default_factory=list)

    logit_lens: list[LogitLensEntry] = field(default_factory=list)

    # Only set when this trace came from generate_stream().
    sampling: SamplingStep | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible types.

        Written by hand rather than via ``dataclasses.asdict`` so the
        ``Trace.logits_tensor`` back-reference (a live torch tensor used by the
        parity tests) is excluded.
        """
        return _to_json(self)


def _to_json(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        out = {}
        for f in obj.__dataclass_fields__.values():
            if f.name == "logits_tensor":
                continue
            out[f.name] = _to_json(getattr(obj, f.name))
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #


@dataclass
class PCAProjection:
    """A 2D basis fitted per trace, shared by every projected tensor."""

    components: torch.Tensor  # (2, emb_dim)
    mean: torch.Tensor  # (emb_dim,)
    explained_variance: list[float]

    def project(self, t: torch.Tensor) -> list[list[float]]:
        """Project a ``(..., emb_dim)`` tensor to 2D coordinates."""
        flat = t.detach().float().reshape(-1, t.shape[-1])
        centered = flat - self.mean
        proj = centered @ self.components.T
        return [[r4(row[0]), r4(row[1])] for row in proj.tolist()]

    def series(
        self, t: torch.Tensor, baseline: torch.Tensor | None = None
    ) -> VectorSeries:
        """Reduce an ``(..., emb_dim)`` tensor to a transportable VectorSeries."""
        flat = t.detach().float().reshape(-1, t.shape[-1])
        norms = [r4(v) for v in flat.norm(dim=-1).tolist()]
        delta_norms = None
        if baseline is not None:
            baseline_flat = baseline.detach().float().reshape(-1, baseline.shape[-1])
            if flat.shape != baseline_flat.shape:
                raise ValueError("Activation and baseline must have matching shapes")
            delta_norms = r4_list(
                (flat.norm(dim=-1) - baseline_flat.norm(dim=-1)).tolist()
            )
        return VectorSeries(
            norms=norms,
            pca=self.project(t),
            shape=list(t.shape),
            explained_variance=self.tensor_explained_variance(t),
            delta_norms=delta_norms,
        )

    def tensor_explained_variance(self, t: torch.Tensor) -> list[float]:
        """Measure this tensor's variance along the trace-wide PCA axes."""
        flat = t.detach().float().reshape(-1, t.shape[-1])
        centered = flat - flat.mean(dim=0, keepdim=True)
        total = float(centered.square().sum())
        if total == 0.0:
            return [0.0] * self.components.shape[0]
        projected = centered @ self.components.T
        captured = projected.square().sum(dim=0) / total
        return r4_list(captured.tolist())


def fit_pca(tensors: Sequence[torch.Tensor], n_components: int = 2) -> PCAProjection:
    """Fit one shared 2D basis across every activation tensor in a trace.

    Stacking all tensors before fitting means a point from block 0 and a point
    from block 11 land in the same coordinate frame, so the frontend can plot
    the whole depth-wise trajectory in one view.

    Residuals grow in norm with depth (roughly 5 to 430 for GPT-2 124M), which
    means the leading components are dominated by the deep residual directions
    and early-layer points cluster near the origin. That is a real property of
    the residual stream, not an artifact, and the UI should scale per-layer if
    it wants to show early detail.
    """
    rows = torch.cat(
        [t.detach().float().reshape(-1, t.shape[-1]) for t in tensors if t is not None],
        dim=0,
    )
    dim = rows.shape[-1]
    mean = rows.mean(dim=0)

    if rows.shape[0] < 2:
        components = torch.zeros(n_components, dim)
        components[:, 0] = 1.0
        return PCAProjection(components, mean, [0.0] * n_components)

    centered = rows - mean
    try:
        _, svals, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        # Singular covariance (e.g. degenerate or zeroed weights).
        components = torch.zeros(n_components, dim)
        components[:, 0] = 1.0
        return PCAProjection(components, mean, [0.0] * n_components)

    k = min(n_components, vh.shape[0])
    components = torch.zeros(n_components, dim)
    components[:k] = vh[:k]

    energy = svals**2
    total = float(energy.sum()) or 1.0
    explained = [r4(float(e) / total) for e in energy[:k].tolist()]
    while len(explained) < n_components:
        explained.append(0.0)

    return PCAProjection(components, mean, explained)


# --------------------------------------------------------------------------- #
# Logit lens
# --------------------------------------------------------------------------- #


@torch.no_grad()
def logit_lens(
    model: Any,
    resid_after_mlp: Sequence[torch.Tensor],
    emb_sum: torch.Tensor,
    final_token_id: int,
    tokenizer: Any = None,
    top_k: int = 3,
) -> list[LogitLensEntry]:
    """Read out top predictions at every position and residual-stream depth.

    The summed embedding is layer ``"emb"``; subsequent entries correspond to
    blocks 1..N. Only top-k token records are retained for transport.
    """
    entries: list[LogitLensEntry] = []
    hidden_states = [("emb", emb_sum), *enumerate(resid_after_mlp, start=1)]
    for layer, resid in hidden_states:
        logits = model.out_head(model.final_norm(resid.detach()))[0]
        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, indices = torch.topk(
            probs, k=min(top_k, 3, probs.shape[-1]), dim=-1
        )
        positions: list[LogitLensPosition] = []
        for position in range(probs.shape[0]):
            idx_list = [int(t) for t in indices[position].tolist()]
            positions.append(
                LogitLensPosition(
                    position=position,
                    top_tokens=idx_list,
                    top_pieces=tokenizer.token_pieces(idx_list) if tokenizer else [],
                    top_probs=r4_list(top_probs[position].tolist()),
                )
            )
        last_logits = logits[-1]
        final_logit = last_logits[final_token_id]
        final_token_rank = int((last_logits > final_logit).sum()) + 1
        last_probs = probs[-1]
        nonzero = last_probs[last_probs > 0]
        entropy = float(-(nonzero * nonzero.log()).sum())
        entries.append(
            LogitLensEntry(
                layer=layer,
                positions=positions,
                final_token_prob=r4(float(last_probs[final_token_id])),
                final_token_rank=final_token_rank,
                entropy=r4(entropy),
            )
        )
    return entries