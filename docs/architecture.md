# Architecture

`backend/gpt2/` is a pure model package. It has no FastAPI import, no app config,
and nothing in it imports from `backend/app/`. The API layer depends on this
package; this package depends only on torch/numpy/tiktoken.

The code was extracted from `notebooks/gpt2.ipynb` with no architectural change:
the same hand-written attention, the same weight mapping, the same weight tying.
See [What changed](#what-changed-during-the-refactor) for the small list of
deliberate fixes.

## Module map

| Module | Contents |
| --- | --- |
| `config.py` | `GPT_CONFIG_124M`, the small-run config, `get_device()` (CUDA-if-available), `ALLOWED_MODEL_SIZES` |
| `layers.py` | `LayerNorm`, `GELU` (tanh approx), `FeedForward`, `MultiHeadAttention` |
| `model.py` | `TransformerBlock`, `GPTModel` with both the plain `forward` and the traced `forward_with_trace` |
| `tokenizer.py` | `GPT2Tokenizer` over `tiktoken`, plus a cached `get_tokenizer()` |
| `download.py` | OpenAI checkpoint download + TF-checkpoint → NumPy dict conversion |
| `weights.py` | `assign()` shape check and `load_weights_into_gpt()` |
| `generate.py` | `generate_text_simple`, `generate`, and the streaming `StepGenerator` |
| `trace.py` | `Trace` dataclasses, `fit_pca` / `PCAProjection`, `logit_lens`, `TensorStats`, rounding helpers |
| `app/settings.py` | Paths, device, CORS, registry config; all env-overridable |
| `app/registry.py` | Loads each model once, serializes access, reports load failures |
| `app/main.py` | App factory, lifespan, CORS, router wiring |
| `app/routes/*.py` | `health`, `models`, `tokenize`, `generate` (POST `/api/trace` + WS) |
| `scripts/train_small.py` | Trains `scratch-10ep` from *The Verdict* |
| `scripts/export_traces.py` | Writes canned traces for static hosting |
| `tests/test_trace_shapes.py` | Trace shape + numeric-contract tests |
| `tests/test_parity.py` | Traced-vs-plain and ours-vs-HuggingFace parity (slow subset needs the 124M checkpoint) |
| `tests/test_api.py` | Route tests via `TestClient`, plus schema-contract tests |

## The API layer

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness, device, how many models are actually ready |
| `GET` | `/api/models` | Every registered model with `ready` flag; a model whose checkpoint is missing is still listed, so the UI can explain why |
| `POST` | `/api/tokenize` | Encode text with the GPT-2 BPE tokenizer |
| `POST` | `/api/trace` | One traced forward pass over a given token sequence (`sampling` is `null`) |
| `WS` | `/api/ws/generate` | Stream one message per generated token, then `done` |

CORS defaults to the Vite dev server (`http://localhost:5173`), overridable via
`GPT2_CORS_ORIGINS`.

### Models

`openai-124m` is the real GPT-2 124M from OpenAI's checkpoint; `GPT2_MODEL_SIZE`
selects a different published size (`355M`, `774M`, `1558M`), in which case the
id becomes `openai-355m` and the architecture config is chosen to match the
downloaded weights.
`scratch-10ep` comes from `scripts/train_small.py` and is **optional**: if
`checkpoints/scratch.pt` is absent the entry is still listed with
`ready: false` and a note, rather than failing startup.

### Why inference needs a lock

All forward passes go through `ModelRegistry.inference()`, which holds a
`threading.Lock` for the whole block. This is not about torch being unsafe: it is
because `forward(return_trace=True)` writes the traced logits to the shared
`model._trace_logits` attribute. Two overlapping requests would clobber each
other's logits and a response could serialize the wrong tensor. The lock covers
serialization too, not just the forward pass.

Generation over the WebSocket runs each step in a thread pool
(`run_in_threadpool`), so the event loop stays free to serve other clients and to
notice a disconnect mid-generation.

The first trace of each WebSocket generation is validated against the Pydantic
contract before being sent; later steps pass their dict straight through, since
re-validating ~250 KB per step would dominate streaming latency.

### Running it

```bash
cd backend
pip install -r requirements.txt

# optional: create the scratch model (~35 s on CPU)
python -m scripts.train_small

# optional: export static traces for the frontend
python -m scripts.export_traces

uvicorn app.main:app --reload --port 8000
```

Startup takes ~30 s because the 124M checkpoint is read from disk. Set
`GPT2_LOAD_OPENAI_MODEL=0` to skip it.

Environment variables: `GPT2_MODEL_SIZE`, `GPT2_CHECKPOINT_DIR`,
`GPT2_SCRATCH_CHECKPOINT`, `GPT2_DEVICE`, `GPT2_CORS_ORIGINS`, `GPT2_TEXT_FILE`,
`GPT2_LOAD_OPENAI_MODEL`, `GPT2_USE_SMALL_MODEL`. Locally these can live in a
`.env` next to `docker-compose.yml` (or `backend/.env`), read at import time by
a dependency-free loader in `settings.py`; anything already set in the real
environment wins over the file. `.env.example` documents them all.

## Training the scratch model

`scripts/train_small.py` reproduces the notebook's from-scratch run: AdamW with
`lr=4e-4` / `weight_decay=0.1`, a 75/25 split on *The Verdict*, and a greedy
sample printed each epoch. Defaults to the small config (13.3 M params, ~35 s for
10 epochs on CPU); `--config 124m` runs the notebook's real size.

**The nan fix.** The validation loader uses `drop_last=False`. At the notebook's
context length of 1024 the validation split is ~1281 tokens, which yields a
*single* sequence; with `batch_size=2` and `drop_last=True` the loader therefore
has **zero** batches, and the loss helper returns `nan` — exactly the
`Validation Loss : nan` the notebook printed. Keeping the final partial batch
gives one real batch and a finite loss.

Observed 10-epoch run (small config): train 10.97 → 8.64, validation 10.89 → 8.76,
finite at every epoch.

## Exported traces

`scripts/export_traces.py` writes four canned prompts to
`frontend/public/traces/<name>.json`. Each file holds the same
`WSStepMessage` payloads the live WebSocket sends, so the frontend's replay hook
needs no special case. All four validate against the Pydantic contract.

They are large — attention is quadratic in prompt length — around 1.4 MB for a
4-token prompt and 2.6 MB for a 9-token one. Use `--max-new-tokens 4` or
`--no-traces` to shrink.

## Tests

```bash
cd backend
pytest                # everything
pytest -m "not slow"   # no checkpoint or transformers needed (~7s)
pytest -m slow         # real 124M weights + HF comparison (~23s)
```

`slow` marks tests that need the 500 MB OpenAI checkpoint and/or the
`transformers` package. Point `GPT2_CHECKPOINT_DIR` at a directory containing
`124M/` (default: `backend/checkpoints/gpt2`).

The test suite loads the 124M checkpoint **once per session** with
`reuse_existing=True`, which issues no HTTP request when the files are already
cached. Without it, every slow test re-download-checked seven files over the
network, which made the suite slow (136s) and prone to failing on a flaky
connection — observed as three spurious `RequestException` failures. With the
flag the slow suite runs in ~23s and passes with the network fully blocked.

### Tolerances are measured, not guessed

| Comparison | Measured max diff | Asserted | Why |
| --- | --- | --- | --- |
| Traced vs plain logits | `0.0` | exact | Same computation; must be bit-identical |
| Ours vs HF logits | `7.63e-05` | `atol=1e-4` | float32 reduction-order difference between two matmul implementations; cannot reach zero |
| Ours vs HF attention (traced) | `2.84e-04` | `atol=5e-4` | float16 transport quantization dominates; `1e-4` is unreachable by construction |
| Ours vs HF attention (unquantized) | `<1e-4` | `atol=1e-4` | Proves the looser traced tolerance is transport, not model error |
| Attention row sums | `4.0e-04` | `<2e-3` | float16 + 4-decimal rounding on values near 1.0 |

`test_attention_matches_huggingface_before_quantization` exists specifically to
separate the two: if raw attention matches HF at `1e-4` but the traced payload
does not, the difference is our quantization and nothing else.

Note that the row-sum check cannot catch a missing causal mask by itself — an
unmasked softmax still sums to 1. The structural check for that is the
upper-triangle assertion, which requires exact `0.0`.

### The 12/12/12 structural tests need no checkpoint

`test_124m_*` asserts 12 blocks, 12 heads, `[12, T, T]` attention per block and
12 logit-lens entries on a **randomly initialised** `GPT_CONFIG_124M`. Those
assertions are config-driven, not weight-driven, so they need no download:
constructing the model costs ~0.8 s and one traced pass ~0.2 s. They therefore
live in the fast suite. Weight *values* are covered by the `slow` tests.

### Mutation-checked

The Phase 3 tests were verified to actually fail when the behaviour they guard is
broken: removing the causal mask, dropping a logit-lens entry, removing the
4-decimal rounding, removing float16 quantization, perturbing traced logits, and
recording only 11 of 12 blocks are each caught.

## Model

```
input token ids (batch, seq_len)
  -> tok_emb  (vocab_size, emb_dim)
  -> pos_emb  (context_length, emb_dim)   learned, added to token embedding
  -> dropout
  -> n_layers x TransformerBlock
  -> final LayerNorm
  -> out_head (no bias, weights tied to tok_emb)
  -> logits (batch, seq_len, vocab_size)
```

Each `TransformerBlock` is pre-norm, with two residual connections:

```
x = x + Dropout(Attention(LayerNorm(x)))
x = x + Dropout(FeedForward(LayerNorm(x)))
```

### Attention is deliberately not fused

`MultiHeadAttention.forward` computes `queries @ keys.transpose(2, 3)`, applies
the causal mask with `masked_fill(mask, -inf)`, scales by `head_dim ** -0.5`, and
softmaxes. It does **not** call `torch.nn.functional.scaled_dot_product_attention`
(F.sdpa).

That is the one hard constraint in this package. Flash/memory-efficient attention
never materializes the `(num_heads, seq, seq)` matrix, so there would be nothing
to show in `AttentionHeatmap`. The explicit path costs more memory and time and
buys us the real attention tensor.

`forward(x, return_attn=True)` returns `(context_vec, attn_weights)`; the default
call returns the context vector only and does not allocate the attention matrix
beyond what the computation itself needs.

## Tracing

`GPTModel.forward(in_idx, return_trace=False, ...)` has two modes.

`return_trace=False` (default) returns the logits tensor and is the same code
path as the notebook implementation — no extra tensors are allocated beyond the
attention matrix the computation already requires. Measured on GPT-2 124M: 41 ms
per call at T=4.

`return_trace=True` returns a `Trace` and stashes the logits on
`model.logits_tensor`, so the parity tests can assert the traced and untraced
paths agree exactly (verified: `max diff == 0.0`).

Tracing forces `eval()` for the duration of the pass and restores the previous
mode afterwards. Without this, dropout would randomize the very numbers the UI
claims to be showing, making traces irreproducible.

### What is recorded

Per forward pass, at batch size 1:

- `token_emb`, `pos_emb`, `emb_sum` (token + position)
- per block: `ln1_out`, every head's attention weights and output norm,
  `resid_after_attn`, `ln2_out`, the `mlp_hidden` summary, `resid_after_mlp`
- `final_norm` and `logits`
- the real top-1 prediction at every prompt position
- a logit-lens entry for the summed embedding and each block depth, at every
  prompt position

Tracing requires batch size 1 and raises `ValueError` otherwise; the schema is
defined per single sequence.

### Never ship a raw activation vector

Residuals are 768-dimensional per position. Every `emb_dim` tensor is reduced by
`PCAProjection.series` to a `VectorSeries`: per-position L2 norms plus 2D PCA
coordinates.

The PCA basis is fitted **once per trace** over the pooled activations of every
recorded tensor. That matters: a shared basis means a point from block 0 and a
point from block 11 land in the same coordinate frame, so the frontend can plot
the entire depth-wise trajectory in one view rather than normalizing per layer.

One honest caveat, documented in the schema: residual norms grow with depth
(roughly 5 → 430 for GPT-2 124M), so the leading component is dominated by deep
residual directions and early-layer points cluster near the origin. Explained
variance is measured independently for each tensor along the shared PCA axes;
those fractions are expected to differ by activation.

### Precision

Every float is rounded to 4 decimals by `r4`, which also maps NaN to `0.0` and
±Inf to ±1e4, since neither is valid JSON. Attention is additionally quantized
to float16 (`f16_nested`), which is ample for values in `[0, 1]` and roughly
halves the attention payload. So an attention value equals
`round(float16(x), 4)`: float16 sets the precision ceiling, 4-decimal rounding is
applied on top for compactness.

### Logit lens

`logit_lens` applies the model's own `final_norm` and `out_head` to the summed
embedding and each block's `resid_after_mlp`. It records the top-three tokens
and probabilities at every prompt position; each entry also summarizes the
final predicted token's probability, rank, and entropy at the last position.
This costs `(n_layers + 1) * T` vocabulary readouts.

The final entry (block `n_layers`) matches the model's actual top-three
probabilities at every position, and its `final_token_rank` is 1. On real
124M weights the lens shows genuine depth-wise progression:

```
L0..L7   ["'re", "'ve", "'ll"]     contractions
L8..L10  [" toward", " towards"]  prepositions
L11      [" forward", " to"]       the correct continuation
```

### Cost

Tracing is ~4.5x the cost of an untraced forward (185 ms vs 41 ms for 124M at
T=4), dominated by PCA and serialization. Payload size grows quadratically in
`T` because attention is `n_layers * n_heads * T²` values: 75 KB at T=4, 250 KB
at T=11, 2 MB at T=41. See "Payload size" in [trace-schema.md](trace-schema.md)
for the streaming implications and the mitigation options.

## Generation

`generate()` implements greedy decoding (`temperature <= 0`) and top-k +
temperature sampling, matching the notebook.

`generate_stream(model, prompt, max_new_tokens, temperature, top_k, trace=True)`
is the step-wise generator. Each iteration runs one real forward pass over the
tokens generated so far, reads the logits from *that* pass, applies top-k
filtering and temperature scaling, and samples. The yielded `StreamStep` carries
that pass's `Trace` with `sampling` populated: the filtered candidate set with
probabilities recomputed after temperature (i.e. the distribution the token was
actually drawn from), the sampled token, and whether EOS was hit.

Note the relationship between the two id lists: `step.token_ids` is the sequence
*including* the newly sampled token, while `step.trace.input_token_ids` is the
*context* that produced the logits. So
`step.token_ids == step.trace.input_token_ids + [step.token_id]`.

EOS terminates the stream and is **not** appended to the sequence.

`trace=False` skips the per-tensor summaries for lightweight streaming.

## Weight loading

`load_weights_into_gpt(gpt, params)` maps the OpenAI checkpoint onto our modules:

| OpenAI name | Destination |
| --- | --- |
| `wte` | `tok_emb.weight` **and** `out_head.weight` (tied) |
| `wpe` | `pos_emb.weight` |
| `blocks[i].attn.c_attn.w` / `.b` | split three ways → `W_query/W_key/W_value`, weights transposed |
| `blocks[i].attn.c_proj.w` / `.b` | `att.out_proj` |
| `blocks[i].mlp.c_fc.w` / `.b` | `ff.layers[0]` |
| `blocks[i].mlp.c_proj.w` / `.b` | `ff.layers[2]` |
| `blocks[i].ln_1.g` / `.b` | `norm1.scale` / `norm1.shift` |
| `blocks[i].ln_2.g` / `.b` | `norm2.scale` / `norm2.shift` |
| `g` / `b` | `final_norm.scale` / `final_norm.shift` |

Transposes are required because TensorFlow stores linear layers as
`(in, out)` while PyTorch uses `(out, in)`.

## Generation

See the "Generation" section above for `generate` and `generate_stream`.

## What changed during the refactor

Behavioral fixes, all intentional:

1. `download.py` uses `verify=True`. The original disabled certificate
   verification, printing an `InsecureRequestWarning` per file.
2. `download.py` imports TensorFlow lazily inside `_latest_checkpoint` and
   `load_gpt2_params_from_tf_ckpt`, so `import gpt2` does not pay TF's ~5s
   startup or require TF at all when a NumPy param file already exists.
   `gpt2.__getattr__` also resolves `download_and_load_gpt2` on first access.
3. `get_device()` prefers CUDA (then MPS) instead of hardcoding CPU.
4. `GELU` uses `math.sqrt(2/pi)` instead of building a `torch.tensor` per call,
   which keeps it correct on non-CPU devices.
5. `generate()` compares against `eos_id` with an explicit `is not None` check
   and a tensor `.all()`, so a greedy run does not blow up on a shape comparison.
6. The notebook's `temperatue` typo is fixed to `temperature`, with
   `generate_compat` accepting the old keyword.
7. `MultiHeadAttention.forward` returns a third tensor, `head_ctx`, the per-head
   value aggregation before `out_proj`. The tracer needs it for per-head output
   norms. The default call path is unchanged.
8. `forward_with_trace` was replaced by `forward(..., return_trace=True)`, and
   the logits are exposed via the `logits_tensor` property. `StepGenerator` was
   removed in favour of `generate_stream`.

Kept deliberately unchanged:

- attention is hand written, no `F.scaled_dot_product_attention`
- pre-norm block ordering and residual structure
- the fused `c_attn` QKV split and every transpose in the weight mapping
- output-head weight tying to `tok_emb`
- the tanh approximation of GELU