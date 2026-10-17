"""Route tests for the FastAPI app, using Starlette's TestClient.

The whole file runs against a **small, randomly initialised** model injected
through ``create_app(registry=...)``. That keeps the suite fast and removes the
500 MB OpenAI download from the critical path; the real weights are exercised by
``test_parity.py``.

Weight *values* are irrelevant here. What these tests verify is HTTP behaviour:
status codes, response shapes, error handling, concurrency and the WebSocket
protocol.
"""

from __future__ import annotations

import json
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import ModelRegistry, set_registry
from app.schemas import (
    MLPHiddenSummary,
    SamplingStep,
    TokenizeRequest,
    TraceModel,
    TraceRequest,
    VectorSeries,
    WSDoneMessage,
    WSErrorMessage,
    WSGenerateRequest,
    WSStepMessage,
)
from app.settings import (
    MODEL_ID_OPENAI,
    MODEL_ID_SCRATCH,
    PROJECT_DIR,
    Settings,
)
from gpt2 import GPT_CONFIG_SMALL, GPTModel, generate_stream, get_tokenizer

PROMPT = "Every effort moves you"


def _registry(*, with_scratch: bool = True, scratch_path=None) -> ModelRegistry:
    """A registry holding one ready small model (plus an optional scratch entry)."""
    settings = Settings(
        checkpoint_dir=PROJECT_DIR / "backend" / "checkpoints" / "gpt2",
        scratch_checkpoint=scratch_path or (PROJECT_DIR / "backend" / "checkpoints" / "nope.pt"),
        text_file=PROJECT_DIR.parent / "the-verdict.txt",
        device="cpu",
        use_small_model=True,
        load_openai_model=False,
    )
    registry = ModelRegistry(settings)

    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()

    from app.registry import LoadedModel

    registry._models[MODEL_ID_OPENAI] = LoadedModel(
        id=MODEL_ID_OPENAI,
        label="test small model",
        source="openai",
        model=model,
        tokenizer=tok,
        ready=True,
    )
    if with_scratch:
        registry._models[MODEL_ID_SCRATCH] = LoadedModel(
            id=MODEL_ID_SCRATCH,
            label="test scratch",
            source="scratch",
            model=model,
            tokenizer=tok,
            ready=True,
        )
    registry._loaded = True
    return registry


@pytest.fixture
def registry():
    return _registry()


@pytest.fixture
def client(registry):
    app = create_app(registry=registry, load_on_startup=False)
    with TestClient(app) as c:
        yield c
    set_registry(None)


# --------------------------------------------------------------------------- #
# GET /api/health
# --------------------------------------------------------------------------- #


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["models_loaded"] >= 1
    assert body["version"]


def test_root_points_at_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["health"] == "/api/health"


# --------------------------------------------------------------------------- #
# GET /api/models
# --------------------------------------------------------------------------- #


def test_models_lists_registered_models(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert {m["id"] for m in models} >= {MODEL_ID_OPENAI, MODEL_ID_SCRATCH}
    for m in models:
        assert m["ready"] is True
        assert m["source"] in ("openai", "scratch")
        assert m["n_layers"] == GPT_CONFIG_SMALL["n_layers"]
        assert m["n_heads"] == GPT_CONFIG_SMALL["n_heads"]
        assert m["emb_dim"] == GPT_CONFIG_SMALL["emb_dim"]


def test_models_reports_unavailable_scratch_without_crashing(registry):
    """A missing scratch checkpoint must be listed as not-ready, not fatal."""
    reg = _registry(with_scratch=False)
    from app.registry import LoadedModel

    reg._models[MODEL_ID_SCRATCH] = LoadedModel(
        id=MODEL_ID_SCRATCH,
        label="scratch",
        source="scratch",
        model=None,
        tokenizer=get_tokenizer(),
        ready=False,
        note="checkpoint not found",
    )
    app = create_app(registry=reg, load_on_startup=False)
    with TestClient(app) as c:
        r = c.get("/api/models")
        assert r.status_code == 200
        scratch = next(m for m in r.json()["models"] if m["id"] == MODEL_ID_SCRATCH)
        assert scratch["ready"] is False
    set_registry(None)


def test_models_empty_registry_is_not_an_error():
    reg = _registry(with_scratch=False)
    reg._models.clear()
    app = create_app(registry=reg, load_on_startup=False)
    with TestClient(app) as c:
        assert c.get("/api/models").json() == {"models": []}
        assert c.get("/api/health").json()["models_loaded"] == 0
    set_registry(None)


# --------------------------------------------------------------------------- #
# POST /api/tokenize
# --------------------------------------------------------------------------- #


def test_tokenize_encodes_text(client):
    r = client.post("/api/tokenize", json={"text": PROMPT, "model_id": MODEL_ID_OPENAI})
    assert r.status_code == 200
    body = r.json()
    assert body["num_tokens"] == len(body["token_ids"]) == len(body["pieces"])
    assert body["model_id"] == MODEL_ID_OPENAI
    # Pieces must join back to the original text.
    assert "".join(body["pieces"]) == PROMPT


def test_tokenize_matches_reference_tokenizer(client):
    tok = get_tokenizer()
    expected = tok.encode(PROMPT)
    body = client.post("/api/tokenize", json={"text": PROMPT, "model_id": MODEL_ID_OPENAI}).json()
    assert body["token_ids"] == expected
    assert body["pieces"] == tok.token_pieces(expected)


def test_tokenize_empty_text(client):
    body = client.post("/api/tokenize", json={"text": "", "model_id": MODEL_ID_OPENAI}).json()
    assert body["token_ids"] == []
    assert body["num_tokens"] == 0


def test_tokenize_rejects_missing_field(client):
    assert client.post("/api/tokenize", json={"model_id": MODEL_ID_OPENAI}).status_code == 422


def test_tokenize_rejects_unknown_model(client):
    r = client.post("/api/tokenize", json={"text": "hi", "model_id": "nope"})
    assert r.status_code == 404


def test_tokenize_rejects_oversized_text(client):
    assert client.post("/api/tokenize", json={"text": "x" * 10_001}).status_code == 422


# --------------------------------------------------------------------------- #
# POST /api/trace
# --------------------------------------------------------------------------- #


def test_trace_returns_valid_trace(client):
    ids = get_tokenizer().encode(PROMPT)
    r = client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": ids})
    assert r.status_code == 200
    body = r.json()
    TraceModel.model_validate(body)  # contract holds over the wire
    assert body["input_token_ids"] == ids
    assert len(body["blocks"]) == GPT_CONFIG_SMALL["n_layers"]
    assert len(body["position_predictions"]) == len(ids)


def test_trace_has_null_sampling(client):
    ids = get_tokenizer().encode(PROMPT)
    body = client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": ids}).json()
    assert body["sampling"] is None


def test_trace_logit_lens_matches_block_count(client):
    ids = get_tokenizer().encode(PROMPT)
    body = client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": ids}).json()
    assert len(body["logit_lens"]) == body["config"]["n_layers"] + 1
    assert len(body["blocks"][0]["heads"]) == body["config"]["n_heads"]


def test_trace_attention_is_causal_over_the_wire(client):
    ids = get_tokenizer().encode(PROMPT)
    body = client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": ids}).json()
    n = len(ids)
    for block in body["blocks"]:
        for head in block["heads"]:
            for i, row in enumerate(head["weights"]):
                assert all(v == 0.0 for v in row[i + 1 :])
                assert abs(sum(row) - 1.0) < 2e-3


def test_trace_respects_mlp_top_k(client):
    ids = get_tokenizer().encode(PROMPT)
    body = client.post(
        "/api/trace",
        json={"model_id": MODEL_ID_OPENAI, "token_ids": ids, "mlp_top_k": 5, "lens_top_k": 3},
    ).json()
    assert body["blocks"][0]["mlp_hidden"]["top_k"] == 5
    assert all(len(row) == 5 for row in body["blocks"][0]["mlp_hidden"]["top"])
    assert len(body["logit_lens"][0]["positions"][0]["top_tokens"]) == 3


def test_trace_rejects_empty_token_ids(client):
    r = client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": []})
    assert r.status_code == 422


def test_trace_rejects_unknown_model(client):
    ids = get_tokenizer().encode(PROMPT)
    assert client.post("/api/trace", json={"model_id": "nope", "token_ids": ids}).status_code == 404


def test_trace_rejects_prompt_longer_than_context(client):
    ids = get_tokenizer().encode(PROMPT)
    too_long = list(ids * (GPT_CONFIG_SMALL["context_length"] // max(len(ids), 1) + 2))
    r = client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": too_long})
    assert r.status_code == 422
    assert "context" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# WS /api/ws/generate
# --------------------------------------------------------------------------- #


def _collect(ws, expected_steps, timeout=60.0):
    """Read messages until ``done`` or an error arrives."""
    messages = []
    while True:
        raw = ws.receive_json()
        if raw.get("type") == "done":
            messages.append(WSDoneMessage.model_validate(raw))
            break
        if raw.get("type") == "error":
            messages.append(WSErrorMessage.model_validate(raw))
            break
        messages.append(WSStepMessage.model_validate(raw))
        if len(messages) > expected_steps + 5:
            raise AssertionError("too many messages")
    return messages


def test_ws_streams_one_step_per_token(client):
    payload = {
        "type": "generate",
        "request_id": "r1",
        "model_id": MODEL_ID_OPENAI,
        "prompt": PROMPT,
        "max_new_tokens": 3,
        "temperature": 0.0,
        "seed": 1,
        "trace": True,
    }
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json(payload)
        messages = _collect(ws, 3)

    steps = [m for m in messages if m.type == "step"]
    done = [m for m in messages if m.type == "done"][0]
    assert len(steps) == 3
    assert [s.step for s in steps] == [0, 1, 2]
    assert len(done.token_ids) == len(get_tokenizer().encode(PROMPT)) + 3
    assert done.text == get_tokenizer().decode(done.token_ids)


def test_ws_step_traces_are_valid_and_carry_sampling(client):
    payload = {
        "type": "generate", "request_id": "r2", "model_id": MODEL_ID_OPENAI,
        "prompt": PROMPT, "max_new_tokens": 2, "temperature": 0.8,
        "top_k": 10, "seed": 3, "trace": True,
    }
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json(payload)
        messages = _collect(ws, 2)

    steps = [m for m in messages if m.type == "step"]
    for s in steps:
        assert s.trace is not None
        TraceModel.model_validate(s.trace.model_dump())
        assert s.trace.sampling is not None
        assert s.trace.sampling.step == s.step
        assert len(s.trace.blocks) == GPT_CONFIG_SMALL["n_layers"]


def test_ws_trace_input_ids_are_the_context(client):
    """trace.input_token_ids is the context; step.token_ids includes the sample."""
    payload = {
        "type": "generate", "request_id": "r3", "model_id": MODEL_ID_OPENAI,
        "prompt": PROMPT, "max_new_tokens": 2, "temperature": 0.0, "trace": True,
    }
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json(payload)
        steps = [m for m in _collect(ws, 2) if m.type == "step"]

    for s in steps:
        assert s.token_ids == list(s.trace.input_token_ids) + [s.token_id]


def test_ws_without_trace_sends_null(client):
    payload = {
        "type": "generate", "request_id": "r4", "model_id": MODEL_ID_OPENAI,
        "prompt": PROMPT, "max_new_tokens": 2, "temperature": 0.0, "trace": False,
    }
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json(payload)
        steps = [m for m in _collect(ws, 2) if m.type == "step"]
    assert len(steps) == 2
    assert all(s.trace is None for s in steps)


def test_ws_greedy_is_reproducible(client):
    def run():
        payload = {
            "type": "generate", "request_id": "r5", "model_id": MODEL_ID_OPENAI,
            "prompt": PROMPT, "max_new_tokens": 3, "temperature": 0.0, "trace": False,
        }
        with client.websocket_connect("/api/ws/generate") as ws:
            ws.send_json(payload)
            steps = [m for m in _collect(ws, 3) if m.type == "step"]
        return [s.token_id for s in steps]

    assert run() == run()


def test_ws_sampling_seed_is_reproducible(client):
    def run():
        payload = {
            "type": "generate", "request_id": "r6", "model_id": MODEL_ID_OPENAI,
            "prompt": PROMPT, "max_new_tokens": 4, "temperature": 1.0,
            "top_k": 20, "seed": 42, "trace": False,
        }
        with client.websocket_connect("/api/ws/generate") as ws:
            ws.send_json(payload)
            steps = [m for m in _collect(ws, 4) if m.type == "step"]
        return [s.token_id for s in steps]

    assert run() == run()


def test_ws_rejects_unknown_model(client):
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json({
            "type": "generate", "request_id": "r7", "model_id": "nope",
            "prompt": PROMPT, "max_new_tokens": 1,
        })
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "unknown model" in msg["message"]


def test_ws_rejects_invalid_request(client):
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json({"type": "generate", "prompt": PROMPT})  # no request_id
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "invalid request" in msg["message"]


def test_ws_rejects_out_of_range_temperature(client):
    with client.websocket_connect("/api/ws/generate") as ws:
        ws.send_json({
            "type": "generate", "request_id": "r8", "model_id": MODEL_ID_OPENAI,
            "prompt": PROMPT, "max_new_tokens": 1, "temperature": -5,
        })
        msg = ws.receive_json()
    assert msg["type"] == "error"


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #


def test_cors_allows_vite_dev_server(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_preflight(client):
    r = client.options(
        "/api/trace",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_inference_lock_serializes_forward_passes(registry):
    """Two threads tracing at once must not clobber each other's logits.

    ``forward(return_trace=True)`` stores logits on the shared model instance.
    Without the registry lock a slow request can overwrite them mid-serialization
    and ship the wrong tensor. This test drives concurrent traced passes and
    checks every response matches its own input.
    """
    tok = get_tokenizer()
    reg = registry
    prompt_a, prompt_b = "Every effort moves you", "Life is full of"
    ids_a, ids_b = tok.encode(prompt_a), tok.encode(prompt_b)

    errors: list[str] = []
    results: dict[str, list[int]] = {}

    def worker(label: str, ids: list[int], expected: list[int]) -> None:
        try:
            for _ in range(6):
                with reg.inference(MODEL_ID_OPENAI) as entry:
                    x = torch.tensor([ids], dtype=torch.long, device=entry.device)
                    with torch.no_grad():
                        entry.model.forward(
                            x, return_trace=True, tokenizer=entry.tokenizer,
                            model_id=entry.id,
                        )
                        got = entry.model.logits_tensor[0, -1].tolist()
                results.setdefault(label, got)
                assert results[label] == got, f"{label}: logits changed between runs"
                assert got == expected or True  # value check below
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    with torch.no_grad():
        expected_a = reg.get(MODEL_ID_OPENAI).model(
            torch.tensor([ids_a])
        )[0, -1].tolist()
        expected_b = reg.get(MODEL_ID_OPENAI).model(
            torch.tensor([ids_b])
        )[0, -1].tolist()

    threads = [
        threading.Thread(target=worker, args=("a", ids_a, expected_a)),
        threading.Thread(target=worker, args=("b", ids_b, expected_b)),
        threading.Thread(target=worker, args=("a", ids_a, expected_a)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)

    assert not errors, errors
    assert results["a"] == expected_a
    assert results["b"] == expected_b
    assert results["a"] != results["b"], "the two prompts must differ for this to mean anything"


def test_registry_lock_is_reentrant_across_sequential_requests(client):
    ids = get_tokenizer().encode(PROMPT)
    bodies = [
        client.post("/api/trace", json={"model_id": MODEL_ID_OPENAI, "token_ids": ids}).json()
        for _ in range(4)
    ]
    # Identical inputs must give identical outputs; proves no state leaks.
    assert all(b == bodies[0] for b in bodies)


# --------------------------------------------------------------------------- #
# Export artefacts (written by scripts/export_traces.py)
# --------------------------------------------------------------------------- #


EXPORT_DIR = PROJECT_DIR / "frontend" / "public" / "traces"


@pytest.mark.parametrize("name", ["effort", "verdict", "fox", "life"])
def test_exported_traces_are_valid(name):
    path = EXPORT_DIR / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{path} not generated; run scripts/export_traces.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["name"] == name
    assert doc["steps"]
    for step in doc["steps"]:
        WSStepMessage.model_validate(step)
        if step["trace"]:
            TraceModel.model_validate(step["trace"])
    WSDoneMessage.model_validate(doc["done"])


def test_exported_traces_match_ws_step_shape():
    """Replay mode must consume the same objects the live WS path sends."""
    path = EXPORT_DIR / "effort.json"
    if not path.is_file():
        pytest.skip("run scripts/export_traces.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    step = doc["steps"][0]
    assert set(step) == set(WSStepMessage.model_fields)

# --------------------------------------------------------------------------- #
# Schema contract
#
# These validate what the tracer and the routes actually produce against the
# Pydantic models, rather than trusting hand-written examples. A field added to
# the tracer without a matching schema field fails here, not in the browser.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def raw_trace():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    return model.forward(
        torch.tensor([tok.encode(PROMPT)]), return_trace=True, tokenizer=tok,
        model_id="scratch-small",
    ).to_dict()


@pytest.fixture(scope="module")
def raw_stream_trace():
    torch.manual_seed(0)
    tok = get_tokenizer()
    model = GPTModel(GPT_CONFIG_SMALL).eval()
    step = next(
        iter(
            generate_stream(
                model, prompt=PROMPT, max_new_tokens=1, temperature=0.8, top_k=12,
                tokenizer=tok, seed=7, model_id="scratch-small",
            )
        )
    )
    return step.trace.to_dict()


def test_trace_model_accepts_real_trace(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    assert parsed.model_id == "scratch-small"
    assert len(parsed.blocks) == GPT_CONFIG_SMALL["n_layers"]


def test_schema_roundtrips_without_loss(raw_trace):
    """Validate then re-serialize must produce the same payload."""
    parsed = TraceModel.model_validate(raw_trace)
    assert json.loads(parsed.model_dump_json()) == raw_trace


def test_stream_trace_with_sampling_validates(raw_stream_trace):
    parsed = TraceModel.model_validate(raw_stream_trace)
    assert parsed.sampling is not None
    assert parsed.sampling.step == 0
    assert parsed.sampling.was_argmax is False
    assert len(parsed.sampling.candidates) == 50


def test_plain_trace_has_null_sampling(raw_trace):
    assert TraceModel.model_validate(raw_trace).sampling is None


def test_logit_lens_matches_config(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    assert len(parsed.logit_lens) == parsed.config["n_layers"] + 1
    assert [e.layer for e in parsed.logit_lens] == [
        "emb", *range(1, parsed.config["n_layers"] + 1)
    ]


def test_last_lens_entry_equals_top_tokens(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    final_position = parsed.logit_lens[-1].positions[-1]
    assert final_position.top_tokens == parsed.top_tokens[:3]
    assert final_position.top_pieces == parsed.top_pieces[:3]
    assert parsed.logit_lens[-1].final_token_rank == 1


def test_position_predictions_cover_prompt(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    assert len(parsed.position_predictions) == len(parsed.input_token_ids)
    assert [p.position for p in parsed.position_predictions] == list(
        range(len(parsed.input_token_ids))
    )


def test_blocks_and_heads_counts(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    for block in parsed.blocks:
        assert len(block.heads) == parsed.config["n_heads"]


def test_embedding_series_have_uniform_shape(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    t = len(parsed.input_token_ids)
    for series in (parsed.token_emb, parsed.pos_emb, parsed.emb_sum, parsed.final_norm):
        assert len(series.norms) == t
        assert len(series.pca) == t
        assert all(len(p) == 2 for p in series.pca)
        assert series.shape == [1, t, parsed.config["emb_dim"]]


def test_pca_variance_is_per_tensor(raw_trace):
    """Shared PCA axes should produce tensor-specific explained variance."""
    parsed = TraceModel.model_validate(raw_trace)
    values = [
        tuple(getattr(parsed.blocks[0], name).explained_variance)
        for name in (
            "ln1_out",
            "resid_after_attn",
            "ln2_out",
            "resid_after_mlp",
        )
    ]
    assert len(set(values)) > 1


def test_mlp_hidden_top_k(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    width = 4 * parsed.config["emb_dim"]
    for block in parsed.blocks:
        assert block.mlp_hidden.top_k == 20
        assert block.mlp_hidden.shape == [1, len(parsed.input_token_ids), width]
        for row in block.mlp_hidden.top:
            assert len(row) == 20
            assert all(0 <= n.index < width for n in row)


def test_attention_is_square_and_causal(raw_trace):
    parsed = TraceModel.model_validate(raw_trace)
    t = len(parsed.input_token_ids)
    for block in parsed.blocks:
        for head in block.heads:
            assert len(head.weights) == t
            assert all(len(row) == t for row in head.weights)
            assert len(head.out_norm) == t
            for i, row in enumerate(head.weights):
                assert all(v == 0.0 for v in row[i + 1 :])


def test_rejects_unknown_field(raw_trace):
    bad = dict(raw_trace)
    bad["mystery_field"] = 1
    with pytest.raises(Exception):
        TraceModel.model_validate(bad)


def test_rejects_missing_required_field(raw_trace):
    bad = dict(raw_trace)
    del bad["logit_lens"]
    with pytest.raises(Exception):
        TraceModel.model_validate(bad)


def test_rejects_wrong_type(raw_trace):
    bad = dict(raw_trace)
    bad["input_token_ids"] = "not a list"
    with pytest.raises(Exception):
        TraceModel.model_validate(bad)


def test_rejects_raw_vector_series():
    """A VectorSeries carrying raw emb_dim vectors must not validate."""
    bad = {
        "norms": [1.0, 2.0],
        "pca": [[0.0, 0.0], [1.0, 1.0]],
        "shape": [1, 2, 768],
        "explained_variance": [0.5, 0.5],
        "vectors": [[0.0] * 768, [0.0] * 768],
    }
    with pytest.raises(Exception):
        VectorSeries.model_validate(bad)


def test_sampling_step_requires_candidates():
    with pytest.raises(Exception):
        SamplingStep.model_validate(
            {
                "step": 0, "temperature": 1.0, "top_k": 5, "greedy": False,
                "sampled_token_id": 1, "sampled_piece": "a", "sampled_prob": 0.5,
                "eos_id": 50256, "eos_hit": False,
            }
        )


def test_mlp_hidden_rejects_wrong_top_k():
    with pytest.raises(Exception):
        MLPHiddenSummary.model_validate(
            {"norms": [1.0], "top": [], "shape": [1, 1, 3072]}
        )


def test_trace_request_defaults():
    req = TraceRequest(token_ids=[1, 2, 3])
    assert req.model_id == "openai-124m"
    assert req.mlp_top_k == 20
    assert req.lens_top_k == 3


def test_trace_request_rejects_empty_tokens():
    with pytest.raises(Exception):
        TraceRequest(token_ids=[])


def test_trace_request_bounds():
    with pytest.raises(Exception):
        TraceRequest(token_ids=[1], mlp_top_k=0)
    with pytest.raises(Exception):
        TraceRequest(token_ids=[1], lens_top_k=999)


def test_ws_request_allows_null_top_k():
    req = WSGenerateRequest(request_id="abc", prompt="hi", top_k=None)
    assert req.top_k is None
    assert req.trace is True


def test_ws_request_rejects_negative_temperature():
    with pytest.raises(Exception):
        WSGenerateRequest(request_id="abc", prompt="hi", temperature=-1.0)


def test_ws_request_defaults():
    req = WSGenerateRequest(request_id="abc", prompt="hi")
    assert req.max_new_tokens == 20
    assert req.temperature == 0.0
    assert req.seed is None


def test_tokenize_request_defaults():
    req = TokenizeRequest(text="hi")
    assert req.model_id == "openai-124m"


def test_tokenize_request_rejects_oversized_text():
    with pytest.raises(Exception):
        TokenizeRequest(text="x" * 10_001)


def test_documented_sampling_example_validates():
    example = {
        "step": 0, "temperature": 0.9, "top_k": 50, "seed": 193,
        "candidates": [
            {
                "rank": 1, "id": 2011, "token": " forward", "logit": 6.2031,
                "p_raw": 0.196, "p_temp": 0.181, "kept": True, "p_final": 0.196,
            }
        ],
        "tail": {"p_raw": 0.12, "p_temp": 0.18},
        "topk_mass_kept": 0.82, "entropy_raw": 4.82, "entropy_final": 2.39,
        "u": 0.137, "cdf_lo": 0.0, "cdf_hi": 0.196,
        "chosen": {
            "rank": 1, "id": 2011, "token": " forward",
            "p_raw": 0.196, "p_final": 0.196,
        },
        "greedy_id": 2011, "was_argmax": True,
        "eos_id": 50256, "eos_hit": False,
    }
    assert SamplingStep.model_validate(example).chosen.id == 2011


def test_documented_vector_series_example_validates():
    example = {
        "norms": [15.1815, 15.8803, 16.3209, 17.0851],
        "pca": [[8.5169, -5.9978], [-5.094, -9.8302],
                [7.9445, 9.2975], [-11.4876, 6.2953]],
        "shape": [1, 4, 768],
        "explained_variance": [0.9854, 0.0066],
    }
    assert len(VectorSeries.model_validate(example).pca) == 4
