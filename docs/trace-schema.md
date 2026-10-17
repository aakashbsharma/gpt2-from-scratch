# Trace JSON schema (source of truth)

This file defines the wire format for **Phase 2 tracing**. Mirrored by:

- `backend/app/schemas.py` (Pydantic v2)
- `frontend/src/types/trace.ts` (TypeScript)

If you change a field here, change it in both.

## Design rules

1. **Everything is real.** Every number comes from a tensor produced by an actual
   forward pass. No placeholder, interpolated or hand-animated values.
2. **Never ship a raw activation vector.** Residuals are 768-dimensional per
   position. Each `emb_dim` tensor is reduced to a per-position L2 norm plus a 2D
   PCA coordinate in a basis shared by the whole trace.
3. **All floats rounded to 4 decimals.** Attention is quantized to float16 first.
4. **No NaN, no Infinity.** Not valid JSON. Guarded in `gpt2.trace.r4`.
5. **Batch size 1 only.** The schema is defined per single sequence.

---

## Envelope: `Trace`

```jsonc
{
  "model_id": "openai-124m",
  "input_token_ids": [6109, 3626, 6100, 345],
  "input_shape": [1, 4],
  "config": { "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
              "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": true },

  "tokens":                [ /* TokenInfo, length T */ ],
  "token_emb":             { /* VectorSeries */ },
  "pos_emb":               { /* VectorSeries */ },
  "emb_sum":               { /* VectorSeries */ },
  "blocks":                [ /* BlockTrace, length n_layers */ ],
  "final_norm":            { /* VectorSeries */ },
  "logits":                { /* TensorStats */ },

  "position_predictions": [ /* PositionPrediction, length T */ ],
  "top_tokens":   [2011, 262, 664, 13, 407],
  "top_pieces":   [" forward", " to", " along", ".", ","],
  "top_probs":    [0.196, 0.1186, 0.0642, 0.0311, 0.0204],

  "logit_lens": [ /* LogitLensEntry, length n_layers + 1 */ ],
  "sampling":   null   // or SamplingStep, when produced by generate_stream
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `model_id` | `string` | Which registered model produced the trace |
| `input_token_ids` | `number[]` | Exact ids fed to the model, length `T` |
| `input_shape` | `number[]` | `[1, T]` |
| `config` | `object` | The model's config dict |
| `tokens` | `TokenInfo[]` | One per prompt position |
| `token_emb` | `VectorSeries` | Token embedding lookup |
| `pos_emb` | `VectorSeries` | Learned position embedding |
| `emb_sum` | `VectorSeries` | `token_emb + pos_emb`, the residual stream entering block 0 |
| `blocks` | `BlockTrace[]` | Exactly `config.n_layers`, in order |
| `final_norm` | `VectorSeries` | Residual stream after the last block, post final LayerNorm |
| `logits` | `TensorStats` | The logits tensor, `[1, T, vocab_size]` |
| `position_predictions` | `PositionPrediction[]` | Real top-1 next-token prediction at **every** position, length `T` |
| `top_tokens` | `number[]` | Top-5 at the last position, descending by probability |
| `top_pieces` | `string[]` | Decoded pieces, parallel to `top_tokens` |
| `top_probs` | `number[]` | Softmax probabilities over the full vocabulary |
| `logit_lens` | `LogitLensEntry[]` | Exactly `n_layers + 1`, summed embedding plus each block |
| `sampling` | `SamplingStep \| null` | `null` unless produced by `generate_stream` |

`logits` holds the full `[1, T, vocab_size]` tensor only as **statistics**; the
raw tensor is not serialized. Use `position_predictions` and `top_*` for the
actual per-position distribution.

---

## `TokenInfo`

```jsonc
{ "id": 6109, "piece": "Every" }
```

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | `number` | Token id |
| `piece` | `string` | Decoded byte-level piece |

---

## `VectorSeries`

The replacement for raw embeddings/residuals. Every `emb_dim`-wide tensor in the
trace appears as one of these.

```jsonc
{
  "norms": [15.1815, 15.8803, 16.3209, 17.0851],
  "pca":   [[8.5169, -5.9978], [-5.094, -9.8302], [7.9445, 9.2975], [-11.4876, 6.2953]],
  "shape": [1, 4, 768],
  "explained_variance": [0.9854, 0.0066],
  "delta_norms": null
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `norms` | `number[]` | Per-position L2 norm, length `T` |
| `pca` | `number[][]` | Per-position `[x, y]`, length `T`, in the trace's shared basis |
| `shape` | `number[]` | Source tensor shape, always `[1, T, emb_dim]` |
| `explained_variance` | `number[]` | Length 2. This tensor's variance captured by each shared PCA axis, in `[0, 1]` |
| `delta_norms` | `number[] \| null` | Per-position norm change from the block stage's actual input; `null` for non-block series |

**The PCA basis is fitted once per trace** over the pooled activations of
`token_emb`, `pos_emb`, `emb_sum`, every `ln1_out` / `resid_after_attn` /
`ln2_out` / `resid_after_mlp`, and `final_norm`. Every `VectorSeries` therefore
shares the same coordinate frame, so a point from block 0 and a point from
block 11 are directly comparable in one view. Explained variance is computed
independently for each tensor against those shared axes.

Note on scale: residual norms grow with depth (roughly 5 → 430 for GPT-2 124M),
so the leading component is dominated by deep residual directions and
early-layer points cluster near the origin. The explained variance varies by tensor, even though all tensors use the shared
PCA coordinate frame.

For block stage deltas, `ln1_out` and `resid_after_attn` compare to the block
input residual (`emb_sum` for block 0, otherwise the previous block's
`resid_after_mlp`); `ln2_out` and `resid_after_mlp` compare to
`resid_after_attn`.

---

## `BlockTrace`

```jsonc
{
  "index": 0,
  "ln1_out":           { /* VectorSeries */ },
  "heads":             [ /* HeadAttention, length n_heads */ ],
  "resid_after_attn":  { /* VectorSeries */ },
  "ln2_out":           { /* VectorSeries */ },
  "mlp_hidden":        { /* MLPHiddenSummary */ },
  "resid_after_mlp":   { /* VectorSeries */ }
}
```

| Field | Meaning |
| --- | --- |
| `ln1_out` | Output of the first LayerNorm (attention input) |
| `heads` | One entry per head |
| `resid_after_attn` | `x + Dropout(Attention(LN1(x)))` |
| `ln2_out` | Output of the second LayerNorm (MLP input) |
| `mlp_hidden` | Post-GELU activations, `4 * emb_dim` wide, summarized |
| `resid_after_mlp` | Block output: `x + Dropout(MLP(LN2(x)))` |

`ln1_out` / `ln2_out` / `final_norm` all share the same `norms` value
(`emb_dim ** 0.5`), which is the expected signature of a LayerNorm. That is a
useful self-check in the UI.

---

## `HeadAttention`

```jsonc
{
  "head": 0,
  "weights": [[1.0, 0.0, 0.0, 0.0], [0.4958, 0.5039, 0.0, 0.0]],
  "out_norm": [0.8459, 0.7558, 0.7489, 0.6837]
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `head` | `number` | Head index within the block |
| `weights` | `number[][]` | `T x T` attention matrix, float16-quantized |
| `out_norm` | `number[]` | L2 norm of this head's context vector per position, length `T` |

- `weights[i][j]` is attention from **query** position `i` to **key** position `j`.
- Because of the causal mask, every entry where `j > i` is exactly `0.0`.
- Rows sum to `1.0` (within float16 quantization).
- Values are post-softmax, and **pre-dropout**: tracing forces `eval()` mode so
  dropout is identity, which makes the trace reproducible.
- `out_norm` is measured on the head's context vector *before* `out_proj`, so
  heads can be compared before the output projection mixes them.

Derive per-key aggregates client-side rather than shipping them, e.g. the arc
view's per-key mean is the column mean of `weights`.

---

## `MLPHiddenSummary`

```jsonc
{
  "norms": [7.9996, 7.6803, 8.2114, 8.0432],
  "top": [[{"index": 402, "value": 1.6713}, {"index": 101, "value": 1.4586}]],
  "shape": [1, 4, 3072],
  "top_k": 20
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `norms` | `number[]` | Per-position L2 norm, length `T` |
| `top` | `TopNeuron[][]` | Top 20 neurons per position, length `T`, each sorted descending |
| `shape` | `number[]` | `[1, T, 4 * emb_dim]` |
| `top_k` | `number` | `20`, or `4 * emb_dim` if that is smaller |

`TopNeuron` is `{ "index": number, "value": number }` where `index` is a neuron
position in `[0, 4 * emb_dim)`.

---

## `TensorStats`

```jsonc
{
  "shape": [1, 4, 50257],
  "dtype": "float32",
  "min": -142.8834,
  "max": 118.2219,
  "mean": 0.0132,
  "std": 1.8402,
  "norm": 1873.44
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `shape` | `number[]` | Full shape of the source tensor |
| `dtype` | `string` | e.g. `float32`, without the `torch.` prefix |
| `min` `max` `mean` `std` `norm` | `number` | Over the whole flattened tensor |

There is no `sample` field: all raw values are excluded from the payload.

---

## `LogitLensEntry`

Each layer applies the model's `final_norm` and `out_head` to the summed
embedding (`"emb"`) or that block's `resid_after_mlp`. Top-three predictions
are stored at every prompt position. The final-token metrics always refer to
the actual layer-`n_layers` argmax at the final prompt position.

```jsonc
{
  "layer": 8,
  "positions": [
    { "position": 0, "top_tokens": [664, 1404, 13],
      "top_pieces": [" toward", " into", "."], "top_probs": [0.0721, 0.0611, 0.0338] }
  ],
  "final_token_prob": 0.0831,
  "final_token_rank": 4,
  "entropy": 5.7912
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `layer` | `"emb" \| number` | Summed embedding or block number `1..n_layers` |
| `positions` | `LogitLensPosition[]` | Top-three tokens/probabilities for positions `0..T-1` |
| `final_token_prob` | `number` | Probability of the final block's last-position argmax at this layer |
| `final_token_rank` | `number` | Rank of that same token at this layer (`1` is top) |
| `entropy` | `number` | Entropy at this layer's last prompt position |

`LogitLensPosition` is `{ position, top_tokens, top_pieces, top_probs }`;
the three arrays are descending by probability. There are exactly
`n_layers + 1` entries: `"emb"` followed by blocks 1 through `n_layers`. The
final entry's last-position top-three values equal the actual final model
distribution's top three; its `final_token_rank` is 1.

The lens is an approximation at early layers: the output head was trained to
read the final layer, so early guesses can be rough or overconfident.

---

## `PositionPrediction`

```jsonc
{ "position": 3, "token_id": 2011, "piece": " forward", "prob": 0.1702 }
```

| Field | Type | Meaning |
| --- | --- | --- |
| `position` | `number` | Prompt position, `0..T-1` |
| `token_id` | `number` | Argmax token at that position |
| `piece` | `string` | Decoded piece |
| `prob` | `number` | Probability of that token |

Length `T`. This is the model's genuine full-prompt prediction at each position.

---

## `SamplingStep`

Only present when the trace came from `generate_stream`.

```jsonc
{
  "step": 0,
  "temperature": 0.9,
  "top_k": 50,
  "seed": 193,
  "candidates": [
    {
      "rank": 1, "id": 2011, "token": " forward", "logit": 6.2031,
      "p_raw": 0.196, "p_temp": 0.181, "kept": true, "p_final": 0.196
    }
  ],
  "tail": { "p_raw": 0.12, "p_temp": 0.18 },
  "topk_mass_kept": 0.82,
  "entropy_raw": 4.82,
  "entropy_final": 2.39,
  "u": 0.137,
  "cdf_lo": 0.0,
  "cdf_hi": 0.196,
  "chosen": {
    "rank": 1, "id": 2011, "token": " forward",
    "p_raw": 0.196, "p_final": 0.196
  },
  "greedy_id": 2011,
  "was_argmax": true,
  "eos_id": 50256,
  "eos_hit": false
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `step` | `number` | 0-based generation step |
| `temperature` | `number` | Applied temperature; `0` means greedy |
| `top_k` | `number \| null` | `null` disables top-k filtering |
| `seed` | `number \| null` | Seed used for this generation's local random generator |
| `candidates` | `SamplingCandidate[]` | Candidates ordered by descending raw logit |
| `tail` | `{ p_raw, p_temp }` | Raw and temperature-scaled mass beyond listed candidates |
| `topk_mass_kept` | `number` | `p_temp` mass of tokens retained by top-k, before renormalization |
| `entropy_raw` / `entropy_final` | `number` | Entropy of the full raw and final sampling distributions |
| `u` | `number \| null` | Uniform random draw; `null` for temperature 0 |
| `cdf_lo` / `cdf_hi` | `number \| null` | Chosen token's half-open CDF interval; `null` when no draw |
| `chosen` | `ChosenToken` | Selected token, rank and its raw/final probabilities |
| `greedy_id` | `number` | Highest-logit token before sampling |
| `was_argmax` | `boolean` | Whether the selected token equals `greedy_id` |
| `eos_id` | `number \| null` | EOS token id, or `null` if disabled |
| `eos_hit` | `boolean` | `true` when this step sampled EOS |

`SamplingCandidate` contains `{ rank, id, token, logit, p_raw, p_temp, kept,
p_final }`. `p_raw` is softmax over unmodified logits; `p_temp` is softmax over
the full vocabulary after temperature; `p_final` is softmax after applying the
top-k mask and temperature. Non-kept candidates have `p_final: 0`.

- `candidates` lists the top `max(top_k, 50)` logits (capped at vocabulary
  size). When top-k is disabled, all vocabulary tokens are listed so the
  complete final distribution and chosen CDF slice remain observable.
- For positive temperature, `u` is drawn from the local seeded generator and
  the chosen token is the first token in descending `p_final` order whose CDF
  exceeds `u`. Temperature zero is deterministic: `u` and CDF bounds are null,
  and `p_final` is a one-hot argmax distribution.
- Candidates remain logit-ranked for display; `cdf_lo` and `cdf_hi` describe
  the chosen token's interval in descending `p_final` order.
- EOS is **not** appended to the token sequence, since it terminates generation.

---

## Payload size

Measured on real GPT-2 124M weights, prompt `"Every effort moves you"`:

| T (tokens) | Trace size | Notes |
| --- | --- | --- |
| 4 | 75 KB | |
| 11 | 250 KB | |
| 41 | 2.0 MB | |
| 41, untraced forward | 91 ms | vs 41 ms untraced at T=4 |

Attention dominates and grows **quadratically** in `T`
(`n_layers * n_heads * T²` values). Per-step size for a greedy 8-token
continuation: 77, 96, 117, 141, 165, 192, 220, 250 KB — 1.3 MB total.

This is a real constraint for the streaming API. If long generations are needed,
the options are (a) cap the prompt window, (b) downsample the attention matrix
past a threshold `T`, or (c) ship attention for the most recent `K` positions
only. Any of these changes the contract above and must be reflected in the
schema, the Pydantic models and the TS types.

---

## REST endpoints

### `GET /api/health`

```jsonc
{ "status": "ok", "device": "cpu", "models_loaded": 2, "version": "0.2.0" }
```

### `GET /api/models`

```jsonc
{ "models": [ { "id": "openai-124m", "label": "GPT-2 124M (OpenAI weights)",
                "n_layers": 12, "n_heads": 12, "emb_dim": 768,
                "context_length": 1024, "vocab_size": 50257,
                "source": "openai", "ready": true } ] }
```

### `POST /api/tokenize`

Request: `{ "text": "Every effort moves you", "model_id": "openai-124m" }`

Response:

```jsonc
{ "model_id": "openai-124m", "token_ids": [6109, 3626, 6100, 345],
  "pieces": ["Every", " effort", " moves", " you"],
  "text": "Every effort moves you", "num_tokens": 4 }
```

### `POST /api/trace`

One traced forward pass over a given sequence. No sampling, so
`sampling` is always `null`.

Request:

```jsonc
{ "model_id": "openai-124m", "token_ids": [6109, 3626, 6100, 345],
  "mlp_top_k": 20, "lens_top_k": 3 }
```

Response: a `Trace`.

---

## WebSocket `/api/ws/generate`

Client message:

```jsonc
{ "type": "generate", "request_id": "abc-123", "model_id": "openai-124m",
  "prompt": "Every effort moves you", "max_new_tokens": 20,
  "temperature": 0.9, "top_k": 50, "seed": 193, "trace": true }
```

With `"trace": true` the server emits one message per generated token, each
carrying that step's real forward-pass trace:

```jsonc
{ "type": "step", "request_id": "abc-123", "step": 0,
  "token_id": 2011, "piece": " forward",
  "token_ids": [6109, 3626, 6100, 345, 2011],
  "text": "Every effort moves you forward",
  "finished": false,
  "trace": { /* Trace, with sampling populated */ } }
```

With `"trace": false` the `trace` field is `null`:

```jsonc
{ "type": "step", "request_id": "abc-123", "step": 0,
  "token_id": 2011, "piece": " forward",
  "token_ids": [ /* ... */ ], "text": " /* ... */ ",
  "finished": false, "trace": null }
```

Then:

```jsonc
{ "type": "done", "request_id": "abc-123",
  "token_ids": [ /* full sequence */ ], "text": " /* full text */ " }
```

Or on error:

```jsonc
{ "type": "error", "request_id": "abc-123", "message": " /* human readable */ " }
```

Rules:

- `temperature <= 0` means greedy. `top_k: null` disables top-k filtering.
- `step.token_ids` is the sequence **including** the newly sampled token, while
  `step.trace.input_token_ids` is the **context** that produced the logits. So
  `token_ids == trace.input_token_ids + [token_id]`.
- `step.trace` is a genuine forward pass over `trace.input_token_ids`. Nothing is
  interpolated, replayed or cached.
- `step` is 0-based and increments by one per generated token.
- The server closes with code `1000` after `done`.