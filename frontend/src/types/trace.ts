/**
 * Trace types mirroring docs/trace-schema.md.
 *
 * Server-side counterpart: backend/app/schemas.py
 * Contract:             docs/trace-schema.md
 *
 * If you change a field here, change it in both.
 *
 * Invariants worth knowing when writing UI code:
 *
 * - These types mirror the **wire format exactly**, so property names are
 *   snake_case as they appear in JSON. No name mapping is needed anywhere; if a
 *   component wants camelCase for its own state, normalize at the store
 *   boundary rather than rewriting these.
 * - No field here ever contains a raw `emb_dim` activation vector. Embeddings
 *   and residuals arrive as `VectorSeries`: per-position L2 norms plus 2D PCA
 *   coordinates. See `docs/trace-schema.md` for why.
 * - Every `VectorSeries` in one trace shares a single PCA basis, so their
 *   `pca` points live in the same coordinate frame. `explained_variance`
 *   reports each tensor's variance captured by those shared axes.
 * - Attention matrices are causal: `weights[i][j]` is 0 whenever `j > i`, and
 *   rows sum to 1. Upper triangle is structurally empty, not merely small.
 * - All numbers are already rounded to 4 decimals (attention to float16
 *   precision), so equality comparisons on them are safe but cumulative error
 *   over 12 layers is possible. Use a tolerance for anything numerical.
 */

/** Shape of the model's config dict. Values are model-specific. */
export interface ModelConfig {
  vocab_size: number;
  context_length: number;
  emb_dim: number;
  n_heads: number;
  n_layers: number;
  drop_rate: number;
  qkv_bias: boolean;
  [key: string]: number | boolean | string;
}

/** One input token. */
export interface TokenInfo {
  id: number;
  piece: string;
}

/**
 * An `emb_dim`-wide activation tensor, reduced for transport.
 *
 * `norms` is the per-position L2 norm; `pca` is the same positions projected to
 * 2D in the trace's shared basis.
 */
export interface VectorSeries {
  /** Per-position L2 norm. Length T. */
  norms: number[];
  /** Per-position [x, y] coordinates. Length T. */
  pca: number[][];
  /** Source tensor shape, always `[1, T, emb_dim]`. */
  shape: number[];
  /** Length 2. Variance captured by each PCA axis, in [0, 1]. */
  explained_variance: number[];
  /** Norm change from the actual stage input; null for non-block activations. */
  delta_norms: number[] | null;
}

/** One of the strongest activations in the MLP hidden layer. */
export interface TopNeuron {
  /** Neuron position within `[0, 4 * emb_dim)`. */
  index: number;
  value: number;
}

/** Summary of the `(T, 4 * emb_dim)` post-GELU activations. */
export interface MLPHiddenSummary {
  /** Per-position L2 norm. Length T. */
  norms: number[];
  /** Top-k neurons per position, each sorted by descending value. */
  top: TopNeuron[][];
  /** `[1, T, 4 * emb_dim]`. */
  shape: number[];
  /** Usually 20, or `4 * emb_dim` if that is smaller. */
  top_k: number;
}

/**
 * Attention weights and output norm for a single head.
 *
 * `weights[i][j]` is attention from query position `i` to key position `j`.
 * Post-softmax, pre-dropout (tracing forces eval mode), float16-quantized.
 */
export interface HeadAttention {
  head: number;
  /** `T x T` matrix. `weights[i][j]` is exactly 0 for `j > i` (causal mask). */
  weights: number[][];
  /**
   * L2 norm of this head's context vector per position, measured before
   * `out_proj`, so heads are comparable before the output projection mixes
   * them. Length T.
   */
  out_norm: number[];
}

/** Statistics for a tensor. The raw tensor is never shipped. */
export interface TensorStats {
  shape: number[];
  /** e.g. `float32`, without a `torch.` prefix. */
  dtype: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  norm: number;
}

/** Everything observable inside one transformer block. */
export interface BlockTrace {
  index: number;
  /** Output of the first LayerNorm (attention input). */
  ln1_out: VectorSeries;
  heads: HeadAttention[];
  /** `x + Dropout(Attention(LN1(x)))`. */
  resid_after_attn: VectorSeries;
  /** Output of the second LayerNorm (MLP input). */
  ln2_out: VectorSeries;
  /** Post-GELU activations, `4 * emb_dim` wide, summarized. */
  mlp_hidden: MLPHiddenSummary;
  /** Block output: `x + Dropout(MLP(LN2(x)))`. */
  resid_after_mlp: VectorSeries;
}

/** Top lens tokens at one prompt position. */
export interface LogitLensPosition {
  position: number;
  top_tokens: number[];
  top_pieces: string[];
  top_probs: number[];
}

/** Predictions read out of the residual stream at one depth. */
export interface LogitLensEntry {
  /** `"emb"` or block number, `1 .. n_layers`. */
  layer: number | 'emb';
  /** One top-3 prediction record per position in the trace context. */
  positions: LogitLensPosition[];
  /** Probability/rank of the final layer's last-position argmax at this depth. */
  final_token_prob: number;
  final_token_rank: number;
  /** Entropy of this layer's last-position lens distribution. */
  entropy: number;
}

/** The model's real top-1 next-token prediction at one prompt position. */
export interface PositionPrediction {
  position: number;
  token_id: number;
  piece: string;
  prob: number;
}

/** One candidate in the visible portion of the sampling distribution. */
export interface SamplingCandidate {
  rank: number;
  id: number;
  token: string;
  logit: number;
  p_raw: number;
  p_temp: number;
  kept: boolean;
  p_final: number;
}

export interface SamplingTail {
  p_raw: number;
  p_temp: number;
}

export interface ChosenToken {
  rank: number;
  id: number;
  token: string;
  p_raw: number;
  p_final: number;
}

/**
 * The sampling decision made after one forward pass.
 *
 * Each candidate records its raw, temperature-scaled, and final probability.
 */
export interface SamplingStep {
  /** 0-based generation step. */
  step: number;
  /** Applied temperature. `<= 0` means greedy. */
  temperature: number;
  /** `null` disables top-k filtering. */
  top_k: number | null;
  /** Seed passed to the per-generation random generator. */
  seed: number | null;
  candidates: SamplingCandidate[];
  tail: SamplingTail;
  topk_mass_kept: number;
  entropy_raw: number;
  entropy_final: number;
  /** Random draw, null for temperature 0. */
  u: number | null;
  cdf_lo: number | null;
  cdf_hi: number | null;
  chosen: ChosenToken;
  greedy_id: number;
  was_argmax: boolean;
  /** `null` if EOS handling is disabled. */
  eos_id: number | null;
  /** `true` when this step sampled EOS and generation stopped. */
  eos_hit: boolean;
}

/** A full forward-pass record. */
export interface Trace {
  model_id: string;
  /** Exact ids fed to the model. Length T. */
  input_token_ids: number[];
  /** Always `[1, T]`. */
  input_shape: number[];
  config: ModelConfig;

  /** One per prompt position. */
  tokens: TokenInfo[];
  /** Token embedding lookup. */
  token_emb: VectorSeries;
  /** Learned position embedding. */
  pos_emb: VectorSeries;
  /** `token_emb + pos_emb`, the residual stream entering block 0. */
  emb_sum: VectorSeries;
  /** Exactly `config.n_layers`, in order. */
  blocks: BlockTrace[];
  /** After the last block, post final LayerNorm. */
  final_norm: VectorSeries;
  /** Statistics for the `[1, T, vocab_size]` logits. */
  logits: TensorStats;

  /** Real top-1 prediction at every position. Length T. */
  position_predictions: PositionPrediction[];
  /** Top-5 at the last position, descending by probability. */
  top_tokens: number[];
  top_pieces: string[];
  top_probs: number[];

  /** `config.n_layers + 1`: summed embedding plus every block output. */
  logit_lens: LogitLensEntry[];
  /** `null` unless the trace came from streaming generation. */
  sampling: SamplingStep | null;
}

// --------------------------------------------------------------------------- //
// REST payloads
// --------------------------------------------------------------------------- //

export interface ModelInfo {
  id: string;
  label: string;
  n_layers: number;
  n_heads: number;
  emb_dim: number;
  context_length: number;
  vocab_size: number;
  source: 'openai' | 'scratch';
  ready: boolean;
}

export interface ModelListResponse {
  models: ModelInfo[];
}

export interface HealthResponse {
  status: string;
  device: string;
  models_loaded: number;
  version: string;
}

export interface TokenizeRequest {
  text: string;
  model_id: string;
}

export interface TokenizeResponse {
  model_id: string;
  token_ids: number[];
  pieces: string[];
  text: string;
  num_tokens: number;
}

/** No sampling is performed, so `Trace.sampling` is always `null`. */
export interface TraceRequest {
  model_id: string;
  token_ids: number[];
  mlp_top_k?: number;
  lens_top_k?: number;
}

// --------------------------------------------------------------------------- //
// WebSocket messages
// --------------------------------------------------------------------------- //

export interface WSGenerateRequest {
  type: 'generate';
  request_id: string;
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  /** `<= 0` means greedy. */
  temperature: number;
  /** `null` disables top-k filtering. */
  top_k: number | null;
  seed: number | null;
  /** When false, `WSStepMessage.trace` is `null`. */
  trace: boolean;
}

/**
 * One streamed generation step.
 *
 * `token_ids` is the sequence *including* the newly sampled token, while
 * `trace.input_token_ids` is the *context* that produced the logits. So
 * `token_ids === [...trace.input_token_ids, token_id]`.
 */
export interface WSStepMessage {
  type: 'step';
  request_id: string;
  step: number;
  token_id: number;
  piece: string;
  token_ids: number[];
  text: string;
  finished: boolean;
  trace: Trace | null;
}

export interface WSDoneMessage {
  type: 'done';
  request_id: string;
  token_ids: number[];
  text: string;
}

export interface WSErrorMessage {
  type: 'error';
  request_id: string;
  message: string;
}

export type WSResponse = WSStepMessage | WSDoneMessage | WSErrorMessage;

// --------------------------------------------------------------------------- //
// Exported replay bundles
// --------------------------------------------------------------------------- //

/** Sampling parameters that produced a replay bundle. */
export interface TraceParams {
  max_new_tokens: number;
  temperature: number;
  top_k: number | null;
  seed: number | null;
}

/**
 * One `frontend/public/traces/<name>.json` file.
 *
 * Produced by `backend/scripts/export_traces.py`. It is a whole generation
 * captured to disk, so it holds the same `WSStepMessage` payloads the live
 * WebSocket sends plus the envelope the exporter adds around them. That is what
 * lets the replay hook and the live hook share one interface.
 */
export interface TraceBundle {
  name: string;
  model_id: string;
  prompt: string;
  params: TraceParams;
  done: WSDoneMessage;
  /** One entry per generated token, in order. */
  steps: WSStepMessage[];
  /** Trace of the last step that carried one, or `null` if all had `trace: false`. */
  final_trace: Trace | null;
}

/** Names of the bundles shipped in `public/traces`. */
export const REPLAY_TRACE_NAMES = ['effort', 'fox', 'life', 'verdict'] as const;

export type ReplayTraceName = (typeof REPLAY_TRACE_NAMES)[number];

// --------------------------------------------------------------------------- //
// Data-source interface
// --------------------------------------------------------------------------- //

/** Lifecycle shared by every data source, live or replayed. */
export type SourceStatus = 'idle' | 'loading' | 'running' | 'done' | 'error';

/**
 * The single interface both `useLiveGeneration` and `useReplay` implement.
 *
 * Components consume *this*, never a hook directly, so switching
 * `ModeToggle` between modes swaps the data source without any component
 * knowing which one it is talking to.
 */
export interface TraceSource {
  status: SourceStatus;
  /** `WSStepMessage`s received so far, oldest first. */
  steps: WSStepMessage[];
  /** Running generated text, i.e. the newest step's `text`. */
  text: string;
  /** Model id the steps came from. */
  modelId: string;
  /** Prompt the steps came from. */
  prompt: string;
  /** Human-readable message when `status === 'error'`, else `null`. */
  error: string | null;
  /** True when the newest step's `finished` flag is set. */
  finished: boolean;
  /** Steps per second over the last few steps, or `null` before any step. */
  stepsPerSecond: number | null;
  /** Abort an in-flight run. No-op when idle or done. */
  stop: () => void;
  /** Drop all accumulated state back to `idle`. */
  reset: () => void;
}

/** Request accepted by both sources. Ignored fields are harmless. */
export interface GenerateOptions {
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number | null;
  seed: number | null;
  /** Replay ignores this: exported bundles always carry traces. */
  trace?: boolean;
}

// --------------------------------------------------------------------------- //
// Derived helpers
// --------------------------------------------------------------------------- //

/** Number of prompt positions `T` in a trace. */
export function traceLength(trace: Trace): number {
  return trace.input_token_ids.length;
}

/**
 * Per-key mean attention for one head, i.e. the column mean of `weights`.
 *
 * Derived client-side rather than shipped. Useful for the arc view, where each
 * key position's total attention received is drawn as an arc from the query.
 */
export function columnMeans(head: HeadAttention): number[] {
  const n = head.weights.length;
  if (n === 0) return [];
  const out = new Array<number>(n).fill(0);
  for (const row of head.weights) {
    for (let j = 0; j < n; j += 1) out[j] += row[j];
  }
  return out.map((v) => v / n);
}

/**
 * Per-query total attention already attended to earlier positions.
 *
 * Returns, for each query `i`, the sum of `weights[i][j]` over `j <= i`, which
 * is the "how much of this position's attention went backwards" quantity.
 */
export function backwardAttention(head: HeadAttention): number[] {
  return head.weights.map((row, i) => {
    let sum = 0;
    for (let j = 0; j <= i; j += 1) sum += row[j];
    return sum;
  });
}

/** Indices of the strongest MLP neurons for one position, already sorted. */
export function topNeuronIndices(mlp: MLPHiddenSummary, position: number): number[] {
  return (mlp.top[position] ?? []).map((n) => n.index);
}

/** True when the trace carries sampling metadata from streaming generation. */
export function isStreamingTrace(trace: Trace): boolean {
  return trace.sampling !== null;
}

/**
 * The trace of the last step that carries one.
 *
 * Steps streamed with `trace: false` hold `null`, so callers cannot assume
 * `steps[steps.length - 1].trace` exists. Returns `null` when no step has a
 * trace, which is the "nothing to draw yet" case.
 */
export function latestTrace(steps: WSStepMessage[]): Trace | null {
  for (let i = steps.length - 1; i >= 0; i -= 1) {
    const t = steps[i].trace;
    if (t) return t;
  }
  return null;
}

/** One pipeline stage, in execution order. `kind` drives the icon and layout. */
export type StageKind =
  | 'tokens'
  | 'embeddings'
  | 'block'
  | 'final_norm'
  | 'logits'
  | 'sample';

/** A node on the horizontal pipeline. Derived, never hand-written per model. */
export interface PipelineStage {
  /** Stable id: `tokens`, `embeddings`, `block-3`, `final_norm`, ... */
  key: string;
  kind: StageKind;
  label: string;
  /** Block index, for `kind === 'block'`. */
  blockIndex?: number;
  /** True once this stage's data is present in the trace. */
  ready: boolean;
}

/**
 * Build the pipeline stage list for a trace.
 *
 * Block count comes from `trace.blocks.length` (which the contract guarantees
 * equals `config.n_layers`), never from a hardcoded 12, so this is correct for
 * the small config too.
 */
export function pipelineStages(trace: Trace | null): PipelineStage[] {
  const stages: PipelineStage[] = [
    { key: 'tokens', kind: 'tokens', label: 'Tokens', ready: trace !== null },
    { key: 'embeddings', kind: 'embeddings', label: 'Embeddings', ready: trace !== null },
  ];

  const nLayers = trace ? trace.blocks.length : 0;
  for (let i = 0; i < nLayers; i += 1) {
    const block = trace!.blocks[i];
    stages.push({
      key: `block-${i}`,
      kind: 'block',
      label: `Block ${i + 1}`,
      blockIndex: block ? block.index : i,
      ready: block !== undefined,
    });
  }

  stages.push({ key: 'final_norm', kind: 'final_norm', label: 'Final Norm', ready: trace !== null });
  stages.push({ key: 'logits', kind: 'logits', label: 'Logits', ready: trace !== null });
  stages.push({
    key: 'sample',
    kind: 'sample',
    label: 'Sample',
    ready: trace !== null && trace.sampling !== null,
  });

  return stages;
}