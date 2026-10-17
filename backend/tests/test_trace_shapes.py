"""Shape and structural tests for the Phase 2 trace payload.

Cheap: they run on the small config with a random model, so the whole file
executes in well under a second. Tests needing the 124M OpenAI checkpoint or the
transformers package are marked ``slow`` in ``test_parity.py``.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from gpt2 import (
    GPT_CONFIG_124M,
    GPT_CONFIG_SMALL,
    GPTModel,
    generate_stream,
    get_tokenizer,
)
from gpt2.trace import TensorStats, f16_nested, fit_pca, r4

PROMPT = "Every effort moves you"
N_LAYERS = GPT_CONFIG_SMALL["n_layers"]
N_HEADS = GPT_CONFIG_SMALL["n_heads"]
EMB = GPT_CONFIG_SMALL["emb_dim"]


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    ids = tok.encode(PROMPT)
    inp = torch.tensor([ids], dtype=torch.long)
    trace = model.forward(inp, return_trace=True, tokenizer=tok, model_id="test")
    return {"ids": ids, "n": len(ids), "model": model, "tok": tok, "trace": trace}


@pytest.fixture(scope="module")
def d(fixture):
    return fixture["trace"].to_dict()


# --------------------------------------------------------------------------- #
# return_trace contract
# --------------------------------------------------------------------------- #


def test_untraced_forward_returns_logits_tensor(fixture):
    out = fixture["model"](torch.tensor([fixture["ids"]], dtype=torch.long))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, fixture["n"], GPT_CONFIG_SMALL["vocab_size"])


def test_traced_forward_returns_trace(fixture):
    assert type(fixture["trace"]).__name__ == "Trace"


def test_traced_logits_match_untraced_exactly(fixture):
    model, ids, trace = fixture["model"], fixture["ids"], fixture["trace"]
    plain = model(torch.tensor([ids], dtype=torch.long))
    assert torch.allclose(plain, model.logits_tensor, atol=1e-6)
    assert (plain - model.logits_tensor).abs().max().item() == 0.0


def test_block_untraced_returns_tensor_and_traced_returns_tuple(fixture):
    model, ids = fixture["model"], fixture["ids"]
    inp = torch.tensor([ids], dtype=torch.long)
    seq = inp.shape[1]
    emb = model.tok_emb(inp) + model.pos_emb(torch.arange(seq))

    block = model.trf_blocks[0]
    with torch.no_grad():
        plain = block(emb)
        traced, record = block(emb, return_trace=True)

    assert isinstance(plain, torch.Tensor)
    assert torch.allclose(plain, traced, atol=1e-6)
    assert set(record) == {
        "ln1_out",
        "attn",
        "head_ctx",
        "resid_after_attn",
        "ln2_out",
        "mlp_hidden",
        "resid_after_mlp",
    }


def test_traced_logits_are_detached(fixture):
    assert fixture["model"].logits_tensor.requires_grad is False


def test_tracing_requires_batch_size_one(fixture):
    inp = torch.tensor([fixture["ids"]] * 2, dtype=torch.long)
    with pytest.raises(ValueError, match="batch size 1"):
        fixture["model"].forward(inp, return_trace=True)


def test_tracing_restores_training_mode(fixture):
    model = GPTModel(GPT_CONFIG_SMALL)
    model.train()
    model.forward(torch.tensor([[1, 2, 3]]), return_trace=True)
    assert model.training
    model.eval()
    model.forward(torch.tensor([[1, 2, 3]]), return_trace=True)
    assert not model.training


def test_traced_logits_excluded_from_serialization(d):
    assert "logits_tensor" not in d
    assert "_trace_logits" not in json.dumps(d)


# --------------------------------------------------------------------------- #
# top-level shape
# --------------------------------------------------------------------------- #


def test_trace_matches_schema_top_level(d):
    for key in (
        "model_id",
        "input_token_ids",
        "input_shape",
        "config",
        "tokens",
        "token_emb",
        "pos_emb",
        "emb_sum",
        "blocks",
        "final_norm",
        "logits",
        "position_predictions",
        "top_tokens",
        "top_pieces",
        "top_probs",
        "logit_lens",
        "sampling",
    ):
        assert key in d, key


def test_input_metadata(d, fixture):
    assert d["input_token_ids"] == fixture["ids"]
    assert d["input_shape"] == [1, fixture["n"]]
    assert d["model_id"] == "test"
    assert d["config"]["n_layers"] == N_LAYERS


def test_tokens_ids_and_pieces(d, fixture):
    assert len(d["tokens"]) == fixture["n"]
    assert [t["id"] for t in d["tokens"]] == fixture["ids"]
    assert all(isinstance(t["piece"], str) for t in d["tokens"])


# --------------------------------------------------------------------------- #
# embeddings and residuals as VectorSeries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", ["token_emb", "pos_emb", "emb_sum", "final_norm"])
def test_embedding_series_shape(d, fixture, key):
    s = d[key]
    assert set(s) == {
        "norms", "pca", "shape", "explained_variance", "delta_norms"
    }
    assert len(s["norms"]) == fixture["n"]
    assert len(s["pca"]) == fixture["n"]
    assert all(len(p) == 2 for p in s["pca"])
    assert s["shape"] == [1, fixture["n"], EMB]
    assert len(s["explained_variance"]) == 2


def test_no_raw_embedding_vectors_shipped(d, fixture):
    """Every emb_dim tensor must be reduced, never sent as a raw vector."""
    for key in ("token_emb", "pos_emb", "emb_sum", "final_norm"):
        s = d[key]
        assert len(s["norms"]) == fixture["n"], key
        assert all(len(p) == 2 for p in s["pca"]), key


def test_pca_explained_variance_is_per_tensor(d):
    """The shared axes capture different fractions of variance by activation."""
    series = [
        d["blocks"][0][key]
        for key in ("ln1_out", "resid_after_attn", "ln2_out", "resid_after_mlp")
    ]
    values = [tuple(s["explained_variance"]) for s in series]
    assert len(set(values)) > 1, values


def test_pca_explained_variance_is_a_fraction(d):
    series = [d["emb_sum"]] + [
        block[key]
        for block in d["blocks"]
        for key in ("ln1_out", "resid_after_attn", "ln2_out", "resid_after_mlp")
    ]
    for item in series:
        ev = item["explained_variance"]
        assert all(0.0 <= v <= 1.0 for v in ev)
        assert sum(ev) <= 1.0001


def test_block_deltas_use_stage_input_baselines(d):
    block_input = d["emb_sum"]["norms"]
    for block in d["blocks"]:
        expected = {
            "ln1_out": block_input,
            "resid_after_attn": block_input,
            "ln2_out": block["resid_after_attn"]["norms"],
            "resid_after_mlp": block["resid_after_attn"]["norms"],
        }
        for key, baseline in expected.items():
            current = block[key]
            wanted = [
                round(value - previous, 4)
                for value, previous in zip(current["norms"], baseline)
            ]
            assert all(
                abs(actual - expected) <= 0.00011
                for actual, expected in zip(current["delta_norms"], wanted)
            ), (block["index"], key, current["delta_norms"], wanted)
        block_input = block["resid_after_mlp"]["norms"]


def test_embedding_norms_are_positive(d):
    for key in ("token_emb", "pos_emb", "emb_sum"):
        assert all(v > 0 for v in d[key]["norms"])


def test_residual_norm_grows_with_depth(d, fixture):
    """Residuals grow strongly with depth.

    Growth is not strictly monotonic: on real 124M weights the per-block deltas
    are [50.98, -1.06, 4.04, ...] -- one small dip at block 1, then
    accelerating. We assert the overall trend, not per-block monotonicity.
    """
    norms = [b["resid_after_mlp"]["norms"][-1] for b in d["blocks"]]
    assert len(norms) == N_LAYERS
    assert norms[-1] > norms[0], norms


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #


def test_one_block_per_layer(d):
    assert len(d["blocks"]) == N_LAYERS
    assert [b["index"] for b in d["blocks"]] == list(range(N_LAYERS))


def test_block_fields(d):
    for b in d["blocks"]:
        assert set(b) == {
            "index",
            "ln1_out",
            "heads",
            "resid_after_attn",
            "ln2_out",
            "mlp_hidden",
            "resid_after_mlp",
        }


def test_head_count(d):
    for b in d["blocks"]:
        assert len(b["heads"]) == N_HEADS
        assert [h["head"] for h in b["heads"]] == list(range(N_HEADS))


def test_attention_is_causal(d, fixture):
    n = fixture["n"]
    for b in d["blocks"]:
        for h in b["heads"]:
            w = h["weights"]
            assert len(w) == n
            assert all(len(row) == n for row in w)
            for i, row in enumerate(w):
                for j in range(i + 1, n):
                    assert row[j] == 0.0, f"non-causal [{i}][{j}]={row[j]}"


def test_attention_rows_are_nonnegative_and_bounded(d, fixture):
    for b in d["blocks"]:
        for h in b["heads"]:
            for row in h["weights"]:
                assert all(0.0 <= v <= 1.0 for v in row)


def test_attention_rows_sum_to_one(d, fixture):
    """Each query's attention must form a distribution over its allowed keys.

    Tolerance is set from the float16 transport, not chosen to make the test
    pass: quantizing a value near 1.0 to float16 and then rounding to 4 decimals
    shifts the row sum by up to ~4e-04 (measured on GPT-2 124M: max deviation
    4.0e-04, min 0.99960, max 1.00030). 2e-03 leaves headroom without being so
    loose that a genuinely broken distribution would slip through.

    Note this cannot catch a missing causal mask on its own -- an unmasked
    softmax still sums to 1. The structural check for that is
    ``test_attention_is_causal``.
    """
    atol = 2e-3
    worst = 0.0
    for b in d["blocks"]:
        for h in b["heads"]:
            for row in h["weights"]:
                worst = max(worst, abs(sum(row) - 1.0))
    assert worst < atol, f"worst row-sum deviation {worst}"


def test_attention_stacked_shape_is_heads_by_t_by_t(d, fixture):
    """Stacking one block's heads yields [n_heads, T, T]."""
    n = fixture["n"]
    for b in d["blocks"]:
        stacked = torch.tensor([h["weights"] for h in b["heads"]])
        assert stacked.shape == (N_HEADS, n, n)
        assert stacked.shape[1] == stacked.shape[2], "attention must be square"


def test_per_head_out_norm(d, fixture):
    for b in d["blocks"]:
        for h in b["heads"]:
            assert len(h["out_norm"]) == fixture["n"]
            assert all(v >= 0 for v in h["out_norm"])


def test_mlp_hidden_summary(d, fixture):
    for b in d["blocks"]:
        mh = b["mlp_hidden"]
        assert set(mh) == {"norms", "top", "shape", "top_k"}
        assert mh["shape"] == [1, fixture["n"], 4 * EMB]
        assert mh["top_k"] == 20
        assert len(mh["norms"]) == fixture["n"]
        assert len(mh["top"]) == fixture["n"]
        for row in mh["top"]:
            assert len(row) == 20
            assert [n["index"] for n in row] == [n["index"] for n in row]


def test_mlp_top_neurons_sorted_descending(d):
    for b in d["blocks"]:
        for row in b["mlp_hidden"]["top"]:
            vals = [n["value"] for n in row]
            assert vals == sorted(vals, reverse=True), vals


def test_mlp_top_indices_within_hidden_dim(d):
    width = 4 * EMB
    for b in d["blocks"]:
        for row in b["mlp_hidden"]["top"]:
            assert all(0 <= n["index"] < width for n in row)


def test_mlp_norms_positive(d):
    for b in d["blocks"]:
        assert all(v > 0 for v in b["mlp_hidden"]["norms"])


# --------------------------------------------------------------------------- #
# logits and predictions
# --------------------------------------------------------------------------- #


def test_logits_stats(d, fixture):
    lg = d["logits"]
    assert lg["shape"] == [1, fixture["n"], GPT_CONFIG_SMALL["vocab_size"]]
    assert lg["dtype"] == "float32"
    assert lg["min"] <= lg["mean"] <= lg["max"]
    assert lg["std"] >= 0.0


def test_position_predictions_cover_prompt(d, fixture):
    pp = d["position_predictions"]
    assert len(pp) == fixture["n"]
    assert [p["position"] for p in pp] == list(range(fixture["n"]))
    for p in pp:
        assert set(p) == {"position", "token_id", "piece", "prob"}
        assert 0.0 <= p["prob"] <= 1.0


def test_top_k_at_last_position(d):
    assert len(d["top_tokens"]) == 5
    assert len(d["top_pieces"]) == 5
    assert len(d["top_probs"]) == 5
    assert d["top_probs"] == sorted(d["top_probs"], reverse=True)


# --------------------------------------------------------------------------- #
# logit lens
# --------------------------------------------------------------------------- #


def test_logit_lens_has_embedding_and_one_entry_per_block(d):
    assert len(d["logit_lens"]) == N_LAYERS + 1
    assert [e["layer"] for e in d["logit_lens"]] == [
        "emb", *range(1, N_LAYERS + 1)
    ]


def test_logit_lens_entry_fields(d):
    for e in d["logit_lens"]:
        assert set(e) == {
            "layer", "positions", "final_token_prob", "final_token_rank", "entropy"
        }
        assert len(e["positions"]) == len(d["input_token_ids"])
        assert all(len(p["top_tokens"]) == 3 for p in e["positions"])
        assert all(len(p["top_pieces"]) == 3 for p in e["positions"])
        assert all(len(p["top_probs"]) == 3 for p in e["positions"])
        assert all(
            p["top_probs"] == sorted(p["top_probs"], reverse=True)
            for p in e["positions"]
        )


def test_logit_lens_last_entry_equals_real_prediction(d):
    """The deepest lens read-out is the model's actual next-token prediction."""
    last_position = d["logit_lens"][-1]["positions"][-1]
    assert last_position["top_tokens"] == d["top_tokens"][:3]
    assert last_position["top_pieces"] == d["top_pieces"][:3]
    assert d["logit_lens"][-1]["final_token_rank"] == 1


def test_final_lens_top_three_matches_actual_probabilities(fixture, d):
    model = fixture["model"]
    inputs = torch.tensor([fixture["ids"]], dtype=torch.long)
    with torch.no_grad():
        actual = torch.softmax(model(inputs)[0].float(), dim=-1)
    final_positions = d["logit_lens"][-1]["positions"]
    for position, entry in enumerate(final_positions):
        expected = actual[position, entry["top_tokens"]].tolist()
        assert torch.allclose(
            torch.tensor(entry["top_probs"]),
            torch.tensor(expected),
            atol=0.000051,
            rtol=0,
        )


def test_logit_lens_differs_across_depth(d):
    """If every depth gave the same answer the lens would be meaningless."""
    all_top = [tuple(e["positions"][-1]["top_tokens"]) for e in d["logit_lens"]]
    assert len(set(all_top)) > 1, all_top


# --------------------------------------------------------------------------- #
# The real 124M structure: 12 blocks, 12 heads, [12, T, T], 12 lens entries
# --------------------------------------------------------------------------- #
#
# These use a *randomly initialised* GPT_CONFIG_124M. The assertions are all
# config-driven (layer count, head count, tensor shapes), so they do not depend
# on weight values and need no 500 MB checkpoint download. Constructing the
# model costs ~0.8 s and one traced pass ~0.2 s, which is cheap enough to keep
# these in the fast suite. Weight *values* are covered by the `slow` parity
# tests in test_parity.py.


@pytest.fixture(scope="module")
def trace_124m():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_124M).eval()
    ids = tok.encode(PROMPT)
    trace = model.forward(
        torch.tensor([ids], dtype=torch.long),
        return_trace=True,
        tokenizer=tok,
        model_id="rand-124m",
    )
    return {"trace": trace, "n": len(ids), "tok": tok}


@pytest.mark.parametrize(
    "key",
    ["token_emb", "pos_emb", "emb_sum", "final_norm"],
)
def test_124m_embedding_series_are_768_wide(trace_124m, key):
    s = trace_124m["trace"].to_dict()[key]
    assert s["shape"] == [1, trace_124m["n"], 768]
    assert len(s["norms"]) == trace_124m["n"]


def test_124m_records_exactly_12_blocks(trace_124m):
    blocks = trace_124m["trace"].blocks
    assert len(blocks) == 12
    assert [b.index for b in blocks] == list(range(12))


def test_124m_attention_is_12_by_t_by_t_per_block(trace_124m):
    """The headline shape assertion: each block carries 12 [T, T] matrices."""
    n = trace_124m["n"]
    for block in trace_124m["trace"].blocks:
        assert len(block.heads) == 12
        stacked = torch.tensor([h.weights for h in block.heads])
        assert stacked.shape == (12, n, n), f"block {block.index}: {stacked.shape}"


def test_124m_attention_rows_sum_to_one(trace_124m):
    stacked = torch.tensor(
        [h.weights for b in trace_124m["trace"].blocks for h in b.heads]
    )
    assert stacked.shape == (12 * 12, trace_124m["n"], trace_124m["n"])
    sums = stacked.sum(-1)
    assert float((sums - 1.0).abs().max()) < 2e-3


def test_124m_attention_upper_triangle_is_exactly_zero(trace_124m):
    """Every entry above the diagonal must be exactly 0.0, not merely small."""
    for block in trace_124m["trace"].blocks:
        stacked = torch.tensor([h.weights for h in block.heads])
        upper = torch.triu(stacked, diagonal=1)
        assert torch.count_nonzero(upper) == 0, (
            f"block {block.index} has upper-triangle mass"
        )


def test_124m_logit_lens_has_embedding_and_12_blocks(trace_124m):
    entries = trace_124m["trace"].logit_lens
    assert len(entries) == 13
    assert entries[0].layer == "emb"
    assert [e.layer for e in entries] == ["emb", *range(1, 13)]


def test_124m_logit_lens_last_entry_is_the_real_prediction(trace_124m):
    trace = trace_124m["trace"]
    assert trace.logit_lens[-1].layer == 12
    assert trace.logit_lens[-1].positions[-1].top_tokens == trace.top_tokens[:3]
    assert trace.logit_lens[-1].positions[-1].top_pieces == trace.top_pieces[:3]
    assert trace.logit_lens[-1].final_token_rank == 1


def test_124m_position_predictions_cover_all_positions(trace_124m):
    pp = trace_124m["trace"].position_predictions
    assert len(pp) == trace_124m["n"]
    assert [p.position for p in pp] == list(range(trace_124m["n"]))


def test_124m_mlp_hidden_is_4x_emb_dim(trace_124m):
    for block in trace_124m["trace"].blocks:
        assert block.mlp_hidden.shape == [1, trace_124m["n"], 4 * 768]


def test_124m_generate_stream_yields_twelve_block_traces():
    """Each streamed step's trace must carry all 12 blocks and 12 lens entries."""
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_124M).eval()
    steps = list(
        generate_stream(
            model, prompt=PROMPT, max_new_tokens=1, temperature=0.0,
            tokenizer=tok, seed=1,
        )
    )
    trace = steps[0].trace
    assert len(trace.blocks) == 12
    assert len(trace.blocks[0].heads) == 12
    assert len(trace.logit_lens) == 13
    assert trace.sampling is not None


# --------------------------------------------------------------------------- #
# numeric contract
# --------------------------------------------------------------------------- #


def _walk_floats(obj, path="$"):
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_floats(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_floats(v, f"{path}[{i}]")


def test_every_float_rounded_to_4_decimals(d):
    offenders = [
        (p, v) for p, v in _walk_floats(d) if round(v, 4) != v
    ]
    assert not offenders, offenders[:5]


def test_no_nan_or_infinity(d):
    offenders = [
        (p, v) for p, v in _walk_floats(d)
        if math.isnan(v) or math.isinf(v)
    ]
    assert not offenders, offenders[:5]


def test_attention_is_float16_quantized(d):
    """Attention values carry float16 precision, then 4-decimal serialization.

    The value must equal round(float16(x), 4): float16 sets the precision
    ceiling, and the 4-decimal rounding is applied on top for compact JSON.
    """
    offenders = []
    for b in d["blocks"]:
        for h in b["heads"]:
            for row in h["weights"]:
                for v in row:
                    fp16 = float(torch.tensor(v).half().item())
                    if round(fp16, 4) != v:
                        offenders.append((v, fp16))
    assert not offenders, offenders[:5]


def test_json_serializable_without_nan(d):
    blob = json.dumps(d, allow_nan=False)
    assert "NaN" not in blob and "Infinity" not in blob


def test_trace_dict_size_is_bounded(d):
    """Payload must stay small enough to stream per generation step."""
    size = len(json.dumps(d))
    assert size < 200_000, size


def test_r4_handles_nonfinite():
    assert r4(float("nan")) == 0.0
    assert r4(float("inf")) == 1e4
    assert r4(float("-inf")) == -1e4
    assert r4(-0.0) == 0.0
    assert r4(1.23456789) == 1.2346


def test_f16_nested_quantizes():
    # 0.123456 is not representable in float16, and the 4-decimal rounding of
    # the quantized value lands on a different number than the float32 original.
    out = f16_nested([[0.123456, 0.5], [0.25, 0.75]])
    assert out[0][0] == round(float(torch.tensor(0.123456).half().item()), 4)
    assert out[0][0] != 0.123456


def test_tensor_stats_single_element_std():
    assert TensorStats.from_tensor(torch.tensor([7.0])).std == 0.0


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #


def test_fit_pca_shapes():
    torch.manual_seed(0)
    tensors = [torch.randn(10, 32) for _ in range(3)]
    pca = fit_pca(tensors)
    assert pca.components.shape == (2, 32)
    assert pca.mean.shape == (32,)
    assert len(pca.explained_variance) == 2
    pts = pca.project(tensors[0])
    assert len(pts) == 10 and all(len(p) == 2 for p in pts)


def test_fit_pca_single_row():
    pca = fit_pca([torch.randn(1, 8)])
    assert pca.components.shape == (2, 8)


def test_fit_pca_degenerate_zero_input():
    pca = fit_pca([torch.zeros(5, 8)])
    assert pca.components.shape == (2, 8)
    assert all(math.isfinite(c) for row in pca.project(torch.zeros(5, 8)) for c in row)


def test_pca_projection_is_deterministic():
    torch.manual_seed(0)
    t = torch.randn(6, 16)
    assert fit_pca([t]).project(t) == fit_pca([t]).project(t)


# --------------------------------------------------------------------------- #
# generate_stream
# --------------------------------------------------------------------------- #


def test_generate_stream_yields_one_trace_per_token(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=3,
            temperature=0.0, top_k=10, tokenizer=fixture["tok"], seed=1,
        )
    )
    assert len(steps) == 3
    for s in steps:
        assert s.trace is not None
        # The trace describes the forward pass *input* (the context that
        # produced the logits). The sampled token is appended afterwards, so
        # step.token_ids == trace.input_token_ids + [sampled].
        assert s.token_ids == s.trace.input_token_ids + [s.token_id]


def test_stream_step_ids_and_text_are_consistent(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=3,
            temperature=0.0, tokenizer=fixture["tok"], seed=1,
        )
    )
    for s in steps:
        assert fixture["tok"].decode(s.token_ids) == s.text
        assert fixture["tok"].token_piece(s.token_id) == s.piece


def test_generate_stream_context_grows_each_step(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=4,
            temperature=0.0, tokenizer=fixture["tok"], seed=1,
        )
    )
    lens = [len(s.trace.input_token_ids) for s in steps]
    assert lens == [len(fixture["ids"]) + i for i in range(4)]


def test_generate_stream_step_index_increments(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=3,
            temperature=0.0, tokenizer=fixture["tok"], seed=1,
        )
    )
    assert [s.sampling.step for s in steps] == [0, 1, 2]


def test_greedy_picks_top_candidate(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=3,
            temperature=0.0, top_k=10, tokenizer=fixture["tok"], seed=1,
        )
    )
    for s in steps:
        sp = s.sampling
        assert sp.greedy is True
        best = max(sp.candidates, key=lambda c: c.p_final)
        assert sp.chosen.id == best.id
        assert abs(sp.chosen.p_final - best.p_final) < 1e-3
        assert sp.was_argmax is True
        assert sp.u is None


def test_top_k_limits_candidate_count(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=1.0, top_k=5, tokenizer=fixture["tok"], seed=2,
        )
    )
    assert len(steps[0].sampling.candidates) == 50


def test_candidates_sorted_and_probabilities_sum_to_one(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=1.0, top_k=10, tokenizer=fixture["tok"], seed=3,
        )
    )
    sp = steps[0].sampling
    kept_probs = [c.p_final for c in sp.candidates if c.kept]
    assert abs(sum(kept_probs) - 1.0) < 1e-3
    assert all(c.p_final == 0.0 for c in sp.candidates if not c.kept)
    assert sp.u is not None
    assert sp.cdf_lo <= sp.u < sp.cdf_hi


def test_sampling_fields_present(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=0.7, top_k=8, tokenizer=fixture["tok"], seed=4,
        )
    )
    sp = steps[0].sampling
    assert sp.temperature == 0.7
    assert sp.top_k == 8
    assert sp.seed == 4
    assert sp.was_argmax is False
    assert sp.eos_hit is False
    assert sp.sampled_piece
    for c in sp.candidates:
        assert set(c.__dataclass_fields__) == {
            "rank", "id", "token", "logit", "p_raw", "p_temp", "kept", "p_final"
        }


def test_sampling_appears_in_serialized_trace(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=0.0, top_k=4, tokenizer=fixture["tok"], seed=5,
        )
    )
    d = steps[0].trace.to_dict()
    assert d["sampling"]["step"] == 0
    assert len(d["sampling"]["candidates"]) == 50
    assert d["sampling"]["chosen"]["id"] == steps[0].token_id
    assert d["sampling"]["u"] is None
    assert d["sampling"]["was_argmax"] is True


def test_stream_sampling_seed_reproduces_tokens(fixture):
    def run():
        return [
            step.token_id
            for step in generate_stream(
                fixture["model"], prompt=PROMPT, max_new_tokens=4,
                temperature=0.8, top_k=20, tokenizer=fixture["tok"], seed=73,
            )
        ]

    assert run() == run()


def test_sampling_draw_lands_inside_chosen_cdf_interval(fixture):
    step = next(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=0.8, top_k=12, tokenizer=fixture["tok"], seed=74,
        )
    )
    sampling = step.sampling
    assert sampling.u is not None
    assert sampling.cdf_lo <= sampling.u < sampling.cdf_hi
    assert sampling.chosen.p_final > 0


def test_sampling_probability_mass_and_non_kept_candidates(fixture):
    sampling = next(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=0.8, top_k=12, tokenizer=fixture["tok"], seed=75,
        )
    ).sampling
    kept = [candidate for candidate in sampling.candidates if candidate.kept]
    assert abs(sum(candidate.p_final for candidate in kept) - 1.0) < 1e-3
    assert all(candidate.p_final == 0.0 for candidate in sampling.candidates if not candidate.kept)


def test_temperature_zero_is_argmax_without_draw(fixture):
    sampling = next(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=1,
            temperature=0.0, top_k=12, tokenizer=fixture["tok"], seed=76,
        )
    ).sampling
    assert sampling.was_argmax is True
    assert sampling.chosen.id == sampling.greedy_id
    assert sampling.u is None
    assert sampling.cdf_lo is None
    assert sampling.cdf_hi is None
    assert sampling.entropy_final == 0.0


def test_temperature_scales_probabilities(fixture):
    """Higher temperature must flatten the distribution (lower peak prob)."""
    model = fixture["model"]
    tok = fixture["tok"]
    inp = torch.tensor([fixture["ids"]], dtype=torch.long)
    with torch.no_grad():
        logits = model(inp)[0, -1]
    top = torch.topk(logits, 5).indices
    filtered = torch.where(
        logits < logits[top[-1]], torch.tensor(float("-inf")), logits
    )
    peaks = [
        torch.softmax(filtered / t, dim=-1).max().item() for t in (0.5, 1.0, 2.0)
    ]
    assert peaks == sorted(peaks, reverse=True), peaks


class _EosBias(torch.nn.Module):
    """Wrap out_head so the EOS logit always wins.

    Simply adding a constant to the ``w_eos`` row does *not* work: the
    final LayerNorm output is centered, so ``sum(h) ~= 0`` and a uniform boost
    moves the logit by almost nothing. Adding a constant directly to the logit
    is the reliable way to force the argmax.
    """

    def __init__(self, base, eos_id: int, vocab_size: int, boost: float = 50.0):
        super().__init__()
        self.base = base
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.boost = boost

    def forward(self, h):
        out = self.base(h).clone()
        out[..., self.eos_id] = out[..., self.eos_id] + self.boost
        return out


def test_generate_stream_stops_on_eos():
    """EOS must terminate the generator and be reported on the step."""
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    eos = tok.end_of_text_id()
    model.out_head = _EosBias(model.out_head, eos, GPT_CONFIG_SMALL["vocab_size"])

    steps = list(
        generate_stream(
            model, prompt=PROMPT, max_new_tokens=5, temperature=0.0,
            tokenizer=tok, seed=1,
        )
    )
    assert len(steps) == 1
    assert steps[0].token_id == eos
    assert steps[0].sampling.eos_hit is True
    assert steps[0].finished is True
    # EOS is not appended to the sequence, since it terminates generation.
    assert steps[0].token_ids == tok.encode(PROMPT)


def test_eos_is_not_appended_but_other_tokens_are():
    """Without EOS, each sampled token extends the sequence by exactly one."""
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    steps = list(
        generate_stream(
            model, prompt=PROMPT, max_new_tokens=3, temperature=0.0, tokenizer=tok
        )
    )
    n0 = len(tok.encode(PROMPT))
    assert [len(s.token_ids) for s in steps] == [n0 + 1, n0 + 2, n0 + 3]


def test_generate_stream_trace_false_gives_lightweight_steps(fixture):
    steps = list(
        generate_stream(
            fixture["model"], prompt=PROMPT, max_new_tokens=2,
            temperature=0.0, trace=False, tokenizer=fixture["tok"], seed=1,
        )
    )
    assert len(steps) == 2
    for s in steps:
        assert s.trace is None
        assert s.token_id >= 0
        assert s.piece


def test_generate_stream_requires_prompt_or_ids(fixture):
    with pytest.raises(ValueError):
        list(generate_stream(fixture["model"], max_new_tokens=1))
    with pytest.raises(ValueError):
        list(
            generate_stream(
                fixture["model"], prompt="hi", input_ids=[1, 2], max_new_tokens=1
            )
        )


def test_generate_stream_rejects_empty_prompt(fixture):
    with pytest.raises(ValueError):
        list(
            generate_stream(
                fixture["model"], prompt="", max_new_tokens=1, tokenizer=fixture["tok"]
            )
        )


def test_generate_stream_is_reproducible_with_seed(fixture):
    def run():
        return [
            s.token_id
            for s in generate_stream(
                fixture["model"], prompt=PROMPT, max_new_tokens=4,
                temperature=1.0, top_k=20, tokenizer=fixture["tok"], seed=42,
            )
        ]

    assert run() == run()