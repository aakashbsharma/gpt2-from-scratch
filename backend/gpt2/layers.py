"""The building blocks of a GPT-2 transformer block.

Everything here is hand written on purpose. In particular
:class:`MultiHeadAttention` computes the attention matrix explicitly with
``Q @ K^T`` and a masked softmax instead of calling
``torch.nn.functional.scaled_dot_product_attention`, because the visualizer needs
the per-head attention weights as a real tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """GELU via the tanh approximation used by the original GPT-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, contract back."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # Expansion
            GELU(),  # Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # Contraction
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention returning the attention weights.

    The attention matrix is only materialized when ``return_attn`` is set, so the
    non-tracing path stays lean while the tracing path gets exactly the values the
    forward pass actually used (pre-dropout and pre-softmax-masked).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal multi-head self-attention.

        Returns just the context vector by default. With ``return_attn=True``
        returns ``(context_vec, attn_weights, head_ctx)`` where ``attn_weights``
        is ``(b, num_heads, T, T)`` and ``head_ctx`` is the per-head value
        aggregation ``(b, num_heads, T, head_dim)`` before the heads are
        concatenated, which the tracer needs for per-head output norms.
        """
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split into heads: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Explicit scaled dot-product attention with a causal mask.
        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (b, num_heads, num_tokens, head_dim) before the heads are merged.
        head_ctx = attn_weights @ values
        context_vec = head_ctx.transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_attn:
            return context_vec, attn_weights, head_ctx
        return context_vec