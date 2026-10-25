"""Parity tests: the refactored package must match the original implementation.

Two independent references are used:

1. The **notebook** semantics, checked by reproducing the exact generation the
   notebook recorded for the OpenAI 124M weights.
2. HuggingFace ``GPT2LMHeadModel``, when transformers is installed, for a strict
   numeric comparison of every logit.

Skipped rather than failed when a reference is unavailable, so the suite runs in
a bare environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gpt2 import (
    GPT_CONFIG_124M,
    GPT_CONFIG_SMALL,
    GPTModel,
    download_and_load_gpt2,
    generate,
    generate_stream,
    get_tokenizer,
    load_weights_into_gpt,
)

# Populated once by _load_openai_model(); see its docstring for why.
_MODEL_CACHE: GPTModel | None = None

CHECKPOINT_DIR = Path(
    os.environ.get("GPT2_CHECKPOINT_DIR", Path(__file__).resolve().parents[1] / "checkpoints" / "gpt2")
)
PROMPT = "Life is full of"

# Must be an explicit repo id: the bare name "gpt2" resolves to the local
# `backend/gpt2/` package directory, which has no model files.
HF_MODEL_ID = os.environ.get("HF_GPT2_ID", "openai-community/gpt2")

# Tolerances, set from measured diffs rather than guessed (see the docstrings on
# the tests that use them):
#
#   logits:    7.63e-05  -> 1e-4 is the requested tolerance and passes.
#              Residual is float32 reduction-order difference between our matmul
#              and HF's, so it cannot be reduced to zero.
#   attention: 2.84e-04  -> must stay looser than the logits tolerance, because
#              the trace deliberately quantizes attention to float16 for
#              transport. float16 near 1.0 resolves to ~2.4e-04, so 1e-4 is
#              unreachable by construction and asserting it would be a false
#              failure.
HF_LOGIT_ATOL = 1e-4
HF_ATTENTION_ATOL = 5e-4

# What notebooks/gpt2.ipynb printed for the OpenAI 124M weights with
# torch.manual_seed(193), top_k=50, temperature=0.9, 25 new tokens.
NOTEBOOK_PREFIX = "Life is full of new problems at work"


def _load_openai_model() -> GPTModel:
    """Load the OpenAI 124M weights once per session and share the instance.

    Built with ``reuse_existing=True`` so no HTTP request is made when the
    checkpoint is already cached. Without this, every slow test re-issued seven
    requests (one per file) purely to compare file sizes, which made the suite
    both slow and prone to failing on a flaky network.
    """
    if not CHECKPOINT_DIR.exists():
        pytest.skip(
            f"no OpenAI checkpoint at {CHECKPOINT_DIR}; run download_and_load_gpt2 first"
        )

    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model = GPTModel(GPT_CONFIG_124M).eval()
        _, params = download_and_load_gpt2(
            "124M", str(CHECKPOINT_DIR), reuse_existing=True
        )
        load_weights_into_gpt(model, params)
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE


# --------------------------------------------------------------------------- #
# Structural / local invariants
# --------------------------------------------------------------------------- #


def test_config_matches_notebook():
    assert GPT_CONFIG_124M == {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }


def test_124m_parameter_count():
    """GPT-2 124M is 163,037,184 params, counting the tied embedding twice.

    This matches HuggingFace's reported count for ``gpt2``. The 124,439,808
    unique parameters come from counting ``out_head`` (38,597,376) once, since
    it shares storage with ``tok_emb``.
    """
    model = GPTModel(GPT_CONFIG_124M)
    assert model.num_parameters() == 163_037_184
    unique = model.num_parameters() - model.out_head.weight.numel()
    assert unique == 124_439_808


def test_attention_is_not_fused():
    """The whole point of hand-written attention: real weights exist."""
    import inspect

    from gpt2 import layers

    src = inspect.getsource(layers.MultiHeadAttention.forward)
    assert "scaled_dot_product_attention" not in src
    assert "attn_weights" in src


def test_output_head_is_not_tied_on_a_fresh_model():
    """A newly constructed model must NOT have tied weights.

    Tying happens only in ``load_weights_into_gpt``, which copies the same
    ``wte`` tensor into both parameters. An untrained model therefore has two
    independently initialized matrices, and asserting they are equal here would
    be wrong.
    """
    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_SMALL)
    assert model.out_head.weight.data_ptr() != model.tok_emb.weight.data_ptr()
    assert not torch.equal(model.out_head.weight, model.tok_emb.weight)


def test_load_weights_ties_output_head_to_token_embedding():
    """After loading OpenAI weights, out_head and tok_emb must share wte."""
    from gpt2 import assign

    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_SMALL)

    wte = np.random.randn(GPT_CONFIG_SMALL["vocab_size"], GPT_CONFIG_SMALL["emb_dim"]).astype(
        np.float32
    )
    params = {"wte": wte, "wpe": np.zeros((GPT_CONFIG_SMALL["context_length"], GPT_CONFIG_SMALL["emb_dim"]), dtype=np.float32)}
    for i in range(GPT_CONFIG_SMALL["n_layers"]):
        e = GPT_CONFIG_SMALL["emb_dim"]
        params["blocks"] = params.get("blocks", []) + [
            {
                "attn": {
                    "c_attn": {
                        "w": np.zeros((e, 3 * e), dtype=np.float32),
                        "b": np.zeros(3 * e, dtype=np.float32),
                    },
                    "c_proj": {"w": np.zeros((e, e), dtype=np.float32), "b": np.zeros(e, dtype=np.float32)},
                },
                "mlp": {
                    "c_fc": {"w": np.zeros((e, 4 * e), dtype=np.float32), "b": np.zeros(4 * e, dtype=np.float32)},
                    "c_proj": {"w": np.zeros((4 * e, e), dtype=np.float32), "b": np.zeros(e, dtype=np.float32)},
                },
                "ln_1": {"g": np.ones(e, dtype=np.float32), "b": np.zeros(e, dtype=np.float32)},
                "ln_2": {"g": np.ones(e, dtype=np.float32), "b": np.zeros(e, dtype=np.float32)},
            }
        ]
    params["g"] = np.ones(GPT_CONFIG_SMALL["emb_dim"], dtype=np.float32)
    params["b"] = np.zeros(GPT_CONFIG_SMALL["emb_dim"], dtype=np.float32)

    from gpt2 import load_weights_into_gpt

    load_weights_into_gpt(model, params)

    assert torch.equal(model.out_head.weight, model.tok_emb.weight)
    assert torch.allclose(model.tok_emb.weight, torch.tensor(wte))


def test_traced_and_plain_logits_identical():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    inp = torch.tensor([tok.encode(PROMPT)], dtype=torch.long)

    with torch.no_grad():
        plain = model(inp)
    model.forward(inp, return_trace=True, tokenizer=tok)

    assert torch.allclose(plain, model.logits_tensor, atol=1e-6)
    assert (plain - model.logits_tensor).abs().max().item() == 0.0


def test_untraced_forward_returns_logits_only():
    """return_trace=False must return the bare tensor, not a Trace."""
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    inp = torch.tensor([tok.encode(PROMPT)], dtype=torch.long)
    out = model(inp)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, len(tok.encode(PROMPT)), GPT_CONFIG_SMALL["vocab_size"])


def test_tracing_restores_training_mode():
    model = GPTModel(GPT_CONFIG_SMALL)
    model.train()
    inp = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model.forward(inp, return_trace=True)
    assert model.training, "trace must not leave the model in eval mode"

    model.eval()
    model.forward(inp, return_trace=True)
    assert not model.training


def test_dropout_is_disabled_during_trace():
    """Dropout must be off while tracing or the trace is not reproducible."""
    torch.manual_seed(42)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL)
    model.train()
    inp = torch.tensor([tok.encode(PROMPT)], dtype=torch.long)

    a = model.forward(inp, return_trace=True, tokenizer=tok)
    b = model.forward(inp, return_trace=True, tokenizer=tok)

    assert a.logits.mean == b.logits.mean
    assert a.emb_sum.norms == b.emb_sum.norms


# --------------------------------------------------------------------------- #
# vs OpenAI weights (the notebook reference)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_openai_weights_reproduce_notebook_generation():
    """Reproduce the exact string the notebook printed from the real weights."""
    model = _load_openai_model()
    tok = get_tokenizer()
    # Pinned to CPU: the reference string was produced there, and the cached
    # model lives on CPU. Sampling under top_k/temperature is not bit-stable
    # across devices, so running this on the configured GPU could flip tokens.
    device = next(model.parameters()).device

    torch.manual_seed(193)
    out = generate(
        model,
        torch.tensor([tok.encode(PROMPT)], dtype=torch.long, device=device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=0.9,
    )
    text = tok.decode(out[0].tolist())
    assert text.startswith(NOTEBOOK_PREFIX), text


@pytest.mark.slow
def test_openai_weights_greedy_is_coherent():
    """Sanity check that the mapping is right: greedy decoding reads like English."""
    model = _load_openai_model()
    tok = get_tokenizer()
    out = generate(
        model,
        torch.tensor([tok.encode(PROMPT)], dtype=torch.long),
        max_new_tokens=20,
        context_size=GPT_CONFIG_124M["context_length"],
        temperature=0.0,
    )
    text = tok.decode(out[0].tolist())
    assert text.startswith(PROMPT)
    # A mis-mapped weight mapping produces punctuation soup almost immediately.
    assert sum(ch.isalpha() or ch.isspace() for ch in text) / len(text) > 0.8, text


@pytest.mark.slow
def test_parity_against_huggingface():
    """Every logit must match HF's GPT2LMHeadModel to within float tolerance.

    Measured max abs diff is ~7.6e-05 across several prompts. That residual is
    float32 reduction-order difference between two different matmul
    implementations, not a modelling error, so it cannot be driven to zero. The
    ``atol=1e-4`` below is the requested tolerance and passes with headroom.
    """
    transformers = pytest.importorskip("transformers")

    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)

    ref = transformers.GPT2LMHeadModel.from_pretrained(HF_MODEL_ID).eval()

    with torch.no_grad():
        ours = model(inp)
        theirs = ref(inp).logits

    assert ours.shape == theirs.shape == (1, len(ids), 50257)
    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < HF_LOGIT_ATOL, f"max logit diff {max_diff}"
    assert torch.allclose(ours, theirs, atol=HF_LOGIT_ATOL), f"max logit diff {max_diff}"

    # And the argmax next token must agree exactly.
    assert int(ours[0, -1].argmax()) == int(theirs[0, -1].argmax())


@pytest.mark.slow
@pytest.mark.parametrize(
    "prompt", ["Life is full of", "Every effort moves you", "The", "In the beginning"]
)
def test_parity_against_huggingface_multiple_prompts(prompt):
    """Tolerance must hold across prompt lengths, not just one lucky input."""
    transformers = pytest.importorskip("transformers")

    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(prompt)
    inp = torch.tensor([ids], dtype=torch.long)

    ref = transformers.GPT2LMHeadModel.from_pretrained(HF_MODEL_ID).eval()
    with torch.no_grad():
        ours = model(inp)
        theirs = ref(inp).logits

    assert torch.allclose(ours, theirs, atol=HF_LOGIT_ATOL), (
        f"max diff for {prompt!r}: {(ours - theirs).abs().max().item()}"
    )
    assert torch.equal(ours[0, -1].argmax(), theirs[0, -1].argmax())


@pytest.mark.slow
def test_traced_logits_match_huggingface():
    """The traced path must not perturb the numbers it records."""
    transformers = pytest.importorskip("transformers")

    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)

    ref = transformers.GPT2LMHeadModel.from_pretrained(HF_MODEL_ID).eval()
    with torch.no_grad():
        theirs = ref(inp).logits

    model.forward(inp, return_trace=True, tokenizer=tok, model_id="openai-124m")
    ours = model.logits_tensor

    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < HF_LOGIT_ATOL, f"max traced logit diff {max_diff}"


@pytest.mark.slow
def test_trace_position_predictions_match_huggingface():
    """The per-position top-1 predictions must be the real model's, not noise."""
    transformers = pytest.importorskip("transformers")

    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)

    trace = model.forward(inp, return_trace=True, tokenizer=tok, model_id="openai-124m")

    ref = transformers.GPT2LMHeadModel.from_pretrained(HF_MODEL_ID).eval()
    with torch.no_grad():
        ref_logits = ref(inp).logits[0]

    for p in trace.position_predictions:
        expected = int(ref_logits[p.position].argmax())
        assert p.token_id == expected, (p.position, p.token_id, expected)


@pytest.mark.slow
def test_attention_weights_match_huggingface():
    """The traced attention matrix must equal HF's, not merely look plausible.

    Compared across **all 12 layers x 12 heads**, not just block 0, so a
    mis-mapped weight in any single block is caught.

    The tolerance is looser than the logits one because our attention values are
    deliberately quantized to float16 for transport; measured diff is ~2.8e-04.
    """
    transformers = pytest.importorskip("transformers")

    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)

    trace = model.forward(inp, return_trace=True, tokenizer=tok, model_id="openai-124m")
    # Each HeadAttention.weights is (seq, seq); stacking over every (block, head)
    # gives (n_layers * n_heads, seq, seq), a leading batch axis matches HF.
    ours = torch.tensor([h.weights for b in trace.blocks for h in b.heads]).unsqueeze(0)

    ref = transformers.GPT2LMHeadModel.from_pretrained(
        HF_MODEL_ID, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        ref_out = ref(inp, output_attentions=True)
    # HF returns a tuple of per-layer [1, heads, seq, seq] tensors.
    theirs = torch.cat(list(ref_out.attentions), dim=1)

    assert ours.shape == theirs.shape == (1, 12 * 12, len(ids), len(ids))
    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < HF_ATTENTION_ATOL, f"max attention diff {max_diff}"


@pytest.mark.slow
def test_attention_matches_huggingface_before_quantization():
    """Separates model error from transport error.

    Reconstructs the un-quantized attention from the model directly and compares
    that against HF at the tight logits tolerance. If this passes but the
    serialized comparison is looser, the difference is our float16 transport and
    nothing else.
    """
    transformers = pytest.importorskip("transformers")

    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)

    # Raw attention straight out of the module, no float16 involved.
    with torch.no_grad():
        x = model.tok_emb(inp) + model.pos_emb(torch.arange(inp.shape[1]))
        x = model.drop_emb(x)
        _, attn, _ = model.trf_blocks[0].att(model.trf_blocks[0].norm1(x), return_attn=True)
    ours = attn[0]  # [heads, seq, seq]

    ref = transformers.GPT2LMHeadModel.from_pretrained(
        HF_MODEL_ID, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        theirs = ref(inp, output_attentions=True).attentions[0][0]

    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < HF_LOGIT_ATOL, f"raw attention diff {max_diff}"


@pytest.mark.slow
def test_logit_lens_last_depth_matches_real_prediction():
    """The deepest lens entry is the model's true prediction."""
    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)

    trace = model.forward(inp, return_trace=True, tokenizer=tok, model_id="openai-124m")
    with torch.no_grad():
        logits = model(inp)

    real_top = torch.topk(logits[0, -1], k=5).indices.tolist()
    assert trace.logit_lens[-1].positions[-1].top_tokens == [int(t) for t in real_top[:3]]
    assert trace.logit_lens[-1].positions[-1].top_tokens == trace.top_tokens[:3]


@pytest.mark.slow
def test_logit_lens_varies_with_depth():
    """Intermediate read-outs must differ from the final one."""
    model = _load_openai_model()
    tok = get_tokenizer()
    inp = torch.tensor([tok.encode(PROMPT)], dtype=torch.long)
    trace = model.forward(inp, return_trace=True, tokenizer=tok, model_id="openai-124m")

    all_top = [tuple(e.positions[-1].top_tokens) for e in trace.logit_lens]
    assert len(set(all_top)) > 1, "logit lens is constant across depth"
    assert len(trace.logit_lens) == 13


@pytest.mark.slow
def test_generate_stream_matches_greedy_reference():
    """Step-wise greedy generation must equal the batch generate() path."""
    model = _load_openai_model()
    tok = get_tokenizer()
    ids = tok.encode(PROMPT)

    streamed = [
        s.token_id
        for s in generate_stream(
            model, prompt=PROMPT, max_new_tokens=8, temperature=0.0, tokenizer=tok
        )
    ]
    batch = generate(
        model, torch.tensor([ids], dtype=torch.long),
        max_new_tokens=8, context_size=1024, temperature=0.0,
    )
    assert streamed == batch[0, len(ids):].tolist()