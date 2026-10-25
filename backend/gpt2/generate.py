"""Text generation: greedy, and top-k / temperature sampling.

``StepGenerator`` is the piece the streaming API uses. It keeps the growing token
sequence and yields one decoded step at a time, which is what lets the frontend
render a genuinely incremental trace instead of a canned animation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .trace import (
    ChosenToken,
    SamplingCandidate,
    SamplingStep,
    SamplingTail,
    r4,
)


def generate_text_simple(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy decoding: always extend with the argmax token."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Sample with optional top-k filtering and temperature scaling.

    ``temperature <= 0`` falls back to greedy (argmax) decoding. The notebook
    spelled this parameter ``temperatue``; the name is corrected here and the
    keyword is accepted as an alias so old call sites still work.
    """
    if temperature is None:
        temperature = 0.0

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and bool((idx_next == eos_id).all()):
                break

            idx = torch.cat((idx, idx_next), dim=1)
    finally:
        if was_training:
            model.train()

    return idx


# Backwards-compatible alias for the notebook's typo'd keyword.
def generate_compat(model, idx, max_new_tokens, context_size, **kwargs):
    if "temperatue" in kwargs:
        kwargs["temperature"] = kwargs.pop("temperatue")
    return generate(model, idx, max_new_tokens, context_size, **kwargs)


# --------------------------------------------------------------------------- #
# Step-wise traced generation
# --------------------------------------------------------------------------- #


@dataclass
class StreamStep:
    """What ``generate_stream`` yields: a trace plus the sampling decision.

    ``step`` lives here rather than being read off the trace, because a step is
    produced with ``trace=False`` too, and the index must not depend on whether
    tracing was requested.
    """

    step: int
    trace: Any
    token_id: int
    piece: str
    token_ids: list[int]
    text: str
    finished: bool

    @property
    def sampling(self):
        """The sampling decision, or ``None`` when this step was not traced."""
        return None if self.trace is None else self.trace.sampling


def _filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Mask everything below the k-th highest logit with ``-inf``."""
    if top_k is None:
        return logits
    k = min(top_k, logits.shape[-1])
    top_logits, _ = torch.topk(logits, k)
    min_val = top_logits[:, -1]
    return torch.where(
        logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
    )


def _inverse_cdf_draw(
    probabilities: torch.Tensor, generator: torch.Generator | None
) -> tuple[int, float, float, float]:
    """Draw one token and return its id, u, and CDF interval."""
    u = float(torch.rand(1, device=probabilities.device, generator=generator).item())
    order = torch.argsort(probabilities, descending=True)
    ordered = probabilities[order]
    cdf = ordered.cumsum(dim=-1)
    rank = int(torch.searchsorted(cdf, torch.tensor(u, device=cdf.device), right=True))
    rank = min(rank, len(order) - 1)
    cdf_lo = float(cdf[rank - 1]) if rank else 0.0
    cdf_hi = float(cdf[rank])
    return int(order[rank]), u, cdf_lo, cdf_hi


def _top_k_mask(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Return a boolean mask selecting exactly the top-k logit indices."""
    mask = torch.ones_like(logits, dtype=torch.bool)
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        indices = torch.topk(logits, k=k, dim=-1).indices
        mask = torch.zeros_like(mask)
        mask.scatter_(-1, indices, True)
    return mask


def _entropy(probabilities: torch.Tensor) -> float:
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * nonzero.log()).sum())


def generate_stream(
    model,
    prompt: str | None = None,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    top_k: int | None = None,
    trace: bool = True,
    *,
    input_ids: list[int] | None = None,
    tokenizer=None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    eos_id: int | None = None,
    model_id: str = "unknown",
    lens_top_k: int = 5,
    mlp_top_k: int = 20,
) -> Iterator[StreamStep]:
    """Yield one :class:`StreamStep` per generated token, each with a real trace.

    Every step runs a genuine forward pass over the tokens generated so far,
    reads the logits from that pass, applies top-k filtering and temperature
    scaling, and samples. The yielded trace describes that exact forward pass,
    and its ``sampling`` field records the filtered candidate set with
    probabilities *after* temperature scaling, the sampled token, and whether
    EOS was produced.

    With ``trace=False`` the trace still runs through the model but the
    expensive per-tensor summaries are skipped, yielding lightweight steps.

    The generator stops after ``max_new_tokens`` or as soon as EOS is sampled.
    """
    from .tokenizer import get_tokenizer

    if (prompt is None) == (input_ids is None):
        raise ValueError("Pass exactly one of prompt or input_ids")

    tokenizer = tokenizer or get_tokenizer()
    ids = list(input_ids) if input_ids is not None else tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Cannot generate from an empty prompt")

    device = torch.device(device)
    eos_id = tokenizer.end_of_text_id() if eos_id is None else eos_id
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    model.eval()
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    for step_index in range(max_new_tokens):
        # Context is truncated to the model's window, like the notebook.
        context = tokens[:, -model.context_length :]

        if trace:
            step_trace = model.forward(
                context,
                return_trace=True,
                tokenizer=tokenizer,
                model_id=model_id,
                mlp_top_k=mlp_top_k,
                lens_top_k=lens_top_k,
            )
            logits = model.logits_tensor
        else:
            step_trace = None
            with torch.no_grad():
                logits = model(context)

        last = logits[:, -1, :]
        raw_logits = last[0].float()
        p_raw = torch.softmax(raw_logits, dim=-1)
        p_temp = (
            torch.softmax(raw_logits / temperature, dim=-1)
            if temperature > 0.0
            else p_raw
        )
        kept_mask = _top_k_mask(raw_logits, top_k)
        greedy_id = int(torch.argmax(raw_logits).item())
        if temperature > 0.0:
            masked_logits = raw_logits.masked_fill(~kept_mask, float("-inf"))
            p_final = torch.softmax(masked_logits / temperature, dim=-1)
            token_id, u, cdf_lo, cdf_hi = _inverse_cdf_draw(p_final, generator)
        else:
            p_final = torch.zeros_like(p_raw)
            p_final[greedy_id] = 1.0
            token_id = greedy_id
            u = cdf_lo = cdf_hi = None

        eos_hit = eos_id is not None and token_id == eos_id

        # Listing the full vocabulary when top-k is disabled is necessary:
        # every token then has nonzero p_final and must be represented in the
        # recorded distribution.
        n_cand = (
            raw_logits.numel()
            if top_k is None
            else min(max(top_k, 50), raw_logits.numel())
        )
        cand_idx = torch.argsort(raw_logits, descending=True)[:n_cand]
        rank_by_id = {int(token): rank + 1 for rank, token in enumerate(cand_idx.tolist())}
        candidates = [
            SamplingCandidate(
                rank=rank + 1,
                id=int(token_id),
                token=tokenizer.token_piece(int(token_id)),
                logit=r4(float(raw_logits[token_id])),
                p_raw=r4(float(p_raw[token_id])),
                p_temp=r4(float(p_temp[token_id])),
                kept=bool(kept_mask[token_id]),
                p_final=r4(float(p_final[token_id])),
            )
            for rank, token_id in enumerate(cand_idx.tolist())
        ]
        listed_mask = torch.zeros_like(kept_mask)
        listed_mask[cand_idx] = True
        chosen_rank = rank_by_id[token_id]
        chosen = ChosenToken(
            rank=chosen_rank,
            id=token_id,
            token=tokenizer.token_piece(token_id),
            p_raw=r4(float(p_raw[token_id])),
            p_final=r4(float(p_final[token_id])),
        )

        if step_trace is not None:
            step_trace.sampling = SamplingStep(
                step=step_index,
                temperature=r4(temperature or 0.0),
                top_k=top_k,
                seed=seed,
                candidates=candidates,
                tail=SamplingTail(
                    p_raw=r4(float(p_raw[~listed_mask].sum())),
                    p_temp=r4(float(p_temp[~listed_mask].sum())),
                ),
                topk_mass_kept=r4(float(p_temp[kept_mask].sum())),
                entropy_raw=r4(_entropy(p_raw)),
                entropy_final=r4(_entropy(p_final)),
                u=u,
                cdf_lo=cdf_lo,
                cdf_hi=cdf_hi,
                chosen=chosen,
                greedy_id=greedy_id,
                was_argmax=token_id == greedy_id,
                eos_id=eos_id,
                eos_hit=eos_hit,
            )

        if not eos_hit:
            tokens = torch.cat(
                (tokens, torch.tensor([[token_id]], dtype=torch.long, device=device)), dim=1
            )

        yield StreamStep(
            step=step_index,
            trace=step_trace,
            token_id=token_id,
            piece=tokenizer.token_piece(token_id),
            token_ids=[int(t) for t in tokens[0].tolist()],
            text=tokenizer.decode([int(t) for t in tokens[0].tolist()]),
            finished=eos_hit or step_index == max_new_tokens - 1,
        )

        if eos_hit:
            return
