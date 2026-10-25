"""The GPT-2 model itself: ``TransformerBlock`` and ``GPTModel``.

``forward`` and ``TransformerBlock.forward`` both take ``return_trace: bool``.
When ``False`` (the default) the code path is byte-for-byte the same as the
notebook implementation and no extra tensors are allocated beyond the attention
matrix the computation already needs. When ``True``, the values that were
actually computed are recorded into a :class:`~gpt2.trace.Trace`.

Tracing forces ``eval()`` mode for the duration of the pass, because dropout
would otherwise randomize the very numbers the UI claims to be showing.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import FeedForward, LayerNorm, MultiHeadAttention
from .trace import (
    BlockTrace,
    HeadAttention,
    LogitLensEntry,
    MLPHiddenSummary,
    PositionPrediction,
    TensorStats,
    TokenInfo,
    TopNeuron,
    Trace,
    VectorSeries,
    f16_nested,
    fit_pca,
    logit_lens,
    r4,
    r4_list,
)

# Number of strongest MLP activations reported per position.
MLP_TOP_K = 20
# Number of logit-lens candidates reported per position and depth.
LENS_TOP_K = 3


class TransformerBlock(nn.Module):
    """Pre-norm block: LN -> attention -> add, then LN -> MLP -> add."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(
        self, x: torch.Tensor, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        record: dict[str, Any] = {}

        shortcut = x
        x = self.norm1(x)
        if return_trace:
            record["ln1_out"] = x
        x = self.att(x, return_attn=return_trace)
        if return_trace:
            x, record["attn"], record["head_ctx"] = x
        x = self.drop_shortcut(x)
        x = x + shortcut
        if return_trace:
            record["resid_after_attn"] = x

        shortcut = x
        x = self.norm2(x)
        if return_trace:
            record["ln2_out"] = x

        # The FeedForward module is Linear -> GELU -> Linear. Running the layers
        # individually is the only way to capture the post-GELU activations,
        # which is why the tracing path does not just call self.ff(x).
        ff_hidden = self.ff.layers[1](self.ff.layers[0](x))
        if return_trace:
            record["mlp_hidden"] = ff_hidden
        x = self.ff.layers[2](ff_hidden)
        x = self.drop_shortcut(x)
        x = x + shortcut
        if return_trace:
            record["resid_after_mlp"] = x

        if return_trace:
            return x, record
        return x


class GPTModel(nn.Module):
    """Decoder-only transformer with learned positional embeddings."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Not a dataclass field; excluded from Trace.to_dict(). Kept so the
        # parity tests can compare traced logits against a plain forward pass.
        self._trace_logits: torch.Tensor | None = None

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        in_idx: torch.Tensor,
        return_trace: bool = False,
        tokenizer: Any = None,
        model_id: str = "unknown",
        mlp_top_k: int = MLP_TOP_K,
        lens_top_k: int = LENS_TOP_K,
    ) -> torch.Tensor | Trace:
        """Forward pass.

        With ``return_trace=False`` returns the logits tensor, exactly as before.
        With ``return_trace=True`` returns a :class:`Trace` and stashes the logits
        on ``self._trace_logits`` so they can be compared against a plain pass.
        """
        if not return_trace:
            _, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            return self.out_head(x)

        return self._forward_traced(
            in_idx,
            tokenizer=tokenizer,
            model_id=model_id,
            mlp_top_k=mlp_top_k,
            lens_top_k=lens_top_k,
        )

    # -- traced forward ------------------------------------------------------ #

    @torch.no_grad()
    def _forward_traced(
        self,
        in_idx: torch.Tensor,
        tokenizer: Any = None,
        model_id: str = "unknown",
        mlp_top_k: int = MLP_TOP_K,
        lens_top_k: int = LENS_TOP_K,
    ) -> Trace:
        if in_idx.shape[0] != 1:
            raise ValueError(
                f"Tracing requires batch size 1, got {in_idx.shape[0]}. "
                "The trace schema is defined per single sequence."
            )

        was_training = self.training
        self.eval()  # dropout off, so recorded values are the real ones

        try:
            seq_len = in_idx.shape[1]
            tok_embeds = self.tok_emb(in_idx)
            # pos_emb is a (seq_len, emb_dim) lookup table, so it broadcasts
            # against tok_emb. The traced path unsqueezes it to (1, T, emb) so
            # every VectorSeries in the trace reports a uniform shape; the plain
            # forward path is left byte-for-byte identical to the notebook.
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            emb_sum = tok_embeds + pos_embeds.unsqueeze(0)
            x = self.drop_emb(emb_sum)
            pos_embeds = pos_embeds.unsqueeze(0)

            records: list[dict[str, Any]] = []
            for block in self.trf_blocks:
                x, record = block(x, return_trace=True)
                records.append(record)

            x = self.final_norm(x)
            logits = self.out_head(x)
        finally:
            if was_training:
                self.train()

        # detach: the traced pass runs under no_grad, but parameters still carry
        # requires_grad=True, and keeping the graph alive would both waste memory
        # and emit warnings when the caller reads the logits.
        logits = logits.detach()

        self._trace_logits = logits
        return self._build_trace(
            in_idx, tok_embeds, pos_embeds, emb_sum, records, x, logits,
            tokenizer=tokenizer, model_id=model_id,
            mlp_top_k=mlp_top_k, lens_top_k=lens_top_k,
        )

    def _build_trace(
        self,
        in_idx: torch.Tensor,
        tok_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        emb_sum: torch.Tensor,
        records: list[dict[str, Any]],
        final_normed: torch.Tensor,
        logits: torch.Tensor,
        tokenizer: Any = None,
        model_id: str = "unknown",
        mlp_top_k: int = MLP_TOP_K,
        lens_top_k: int = LENS_TOP_K,
    ) -> Trace:
        # One shared PCA basis for the whole trace, so points from every layer
        # are directly comparable in the frontend.
        pooled = [tok_embeds, pos_embeds, emb_sum, final_normed]
        for rec in records:
            pooled.extend(
                [rec["ln1_out"], rec["resid_after_attn"], rec["ln2_out"], rec["resid_after_mlp"]]
            )
        pca = fit_pca(pooled)

        token_ids = [int(t) for t in in_idx[0].tolist()]
        tokens = [
            TokenInfo(id=t, piece=tokenizer.token_piece(t) if tokenizer else "")
            for t in token_ids
        ]

        blocks: list[BlockTrace] = []
        block_input = emb_sum
        for i, rec in enumerate(records):
            attn = rec["attn"]  # (1, n_heads, T, T)
            heads = [
                HeadAttention(
                    head=h,
                    weights=f16_nested(attn[0, h]),
                    # Norm of this head's context vector, before out_proj.
                    out_norm=r4_list(_head_out_norm(rec, h)),
                )
                for h in range(attn.shape[1])
            ]
            blocks.append(
                BlockTrace(
                    index=i,
                    ln1_out=pca.series(rec["ln1_out"], block_input),
                    heads=heads,
                    resid_after_attn=pca.series(rec["resid_after_attn"], block_input),
                    ln2_out=pca.series(rec["ln2_out"], rec["resid_after_attn"]),
                    mlp_hidden=_mlp_summary(rec["mlp_hidden"], mlp_top_k),
                    resid_after_mlp=pca.series(
                        rec["resid_after_mlp"], rec["resid_after_attn"]
                    ),
                )
            )
            block_input = rec["resid_after_mlp"]

        # Per-position top-1 prediction from the final layer.
        probs = torch.softmax(logits[0].float(), dim=-1)
        top1 = torch.topk(probs, k=1, dim=-1)
        position_predictions = []
        for pos in range(logits.shape[1]):
            tid = int(top1.indices[pos, 0])
            position_predictions.append(
                PositionPrediction(
                    position=pos,
                    token_id=tid,
                    piece=tokenizer.token_piece(tid) if tokenizer else "",
                    prob=r4(top1.values[pos, 0]),
                )
            )

        top_probs, top_idx = torch.topk(probs[-1], k=min(5, probs.shape[-1]))
        top_idx_list = [int(t) for t in top_idx.tolist()]
        final_token_id = int(top1.indices[-1, 0])

        return Trace(
            model_id=model_id,
            input_token_ids=token_ids,
            input_shape=list(in_idx.shape),
            config=dict(self.cfg),
            tokens=tokens,
            token_emb=pca.series(tok_embeds),
            pos_emb=pca.series(pos_embeds),
            emb_sum=pca.series(emb_sum),
            blocks=blocks,
            final_norm=pca.series(final_normed),
            logits=TensorStats.from_tensor(logits),
            position_predictions=position_predictions,
            top_tokens=top_idx_list,
            top_pieces=tokenizer.token_pieces(top_idx_list) if tokenizer else [],
            top_probs=r4_list(top_probs.tolist()),
            logit_lens=logit_lens(
                self,
                [rec["resid_after_mlp"] for rec in records],
                emb_sum,
                final_token_id,
                tokenizer=tokenizer,
                top_k=lens_top_k,
            ),
        )

    # -- introspection ------------------------------------------------------- #

    @property
    def logits_tensor(self) -> torch.Tensor | None:
        """Logits from the most recent traced forward pass, or ``None``.

        Deliberately excluded from ``Trace.to_dict()``; the parity tests use it
        to compare a traced pass against a plain one.
        """
        return self._trace_logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def context_length(self) -> int:
        return self.cfg["context_length"]

    @property
    def n_layers(self) -> int:
        return self.cfg["n_layers"]

    @property
    def n_heads(self) -> int:
        return self.cfg["n_heads"]


def _head_out_norm(rec: dict[str, Any], head: int) -> list[float]:
    """L2 norm of one head's context vector, per position.

    ``attn`` is ``(1, n_heads, T, T)`` and ``head_ctx`` is
    ``(1, n_heads, T, head_dim)``, the per-head value aggregation before the
    heads are concatenated and passed through ``out_proj``.
    """
    head_ctx = rec["head_ctx"]  # (1, n_heads, T, head_dim)
    return head_ctx[0, head].norm(dim=-1).tolist()


def _mlp_summary(hidden: torch.Tensor, top_k: int) -> MLPHiddenSummary:
    """Top-k activations per position plus per-position L2 norm."""
    flat = hidden.detach().float()[0]  # (T, 4 * emb_dim)
    k = min(top_k, flat.shape[-1])
    values, indices = torch.topk(flat, k=k, dim=-1)
    norms = flat.norm(dim=-1).tolist()
    return MLPHiddenSummary(
        norms=r4_list(norms),
        top=[
            [TopNeuron(index=int(i), value=r4(v)) for i, v in zip(idx_row, val_row)]
            for idx_row, val_row in zip(indices.tolist(), values.tolist())
        ],
        shape=list(hidden.shape),
        top_k=k,
    )