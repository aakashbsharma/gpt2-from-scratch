/**
 * The internals of one transformer block: LN → Attention → Add → LN → MLP → Add.
 *
 * Every value here is read from the trace, including stage deltas computed by
 * the backend against each operation's actual input.
 */

import { motion } from 'framer-motion';

import { formatCompact, formatFixed } from '@/lib/format';
import { useTraceStore, selectCurrentTrace, selectSelectedBlock } from '@/store/useTraceStore';
import type { BlockTrace, Trace, VectorSeries } from '@/types/trace';

/** Store-connected shell: resolves the current block and open state. */
export function BlockDetail() {
  const trace = useTraceStore(selectCurrentTrace);
  const block = useTraceStore(selectSelectedBlock);
  const open = useTraceStore((s) => s.detailOpen);
  const setOpen = useTraceStore((s) => s.setDetailOpen);
  const selectedBlock = useTraceStore((s) => s.selectedBlock);
  const selectedHead = useTraceStore((s) => s.selectedHead);

  if (!open || !trace || !block || selectedBlock === null) return null;

  return (
    <BlockDetailView
      trace={trace}
      block={block}
      blockIndex={selectedBlock}
      headIndex={selectedHead}
      onClose={() => setOpen(false)}
    />
  );
}

/** The panel body, driven purely by props so it renders from any caller. */
export function BlockDetailView({
  trace,
  block,
  blockIndex,
  headIndex,
  onClose,
}: {
  trace: Trace;
  block: BlockTrace;
  blockIndex: number;
  headIndex: number;
  onClose: () => void;
}) {
  const hiddenWidth = block.mlp_hidden.shape[2] ?? 0;
  const last = trace.input_token_ids.length - 1;

  return (
    <motion.aside
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      className="panel"
      aria-label={`Block ${blockIndex + 1} detail`}
    >
      <header className="panel-header">
        <h2 className="panel-title">
          Block {blockIndex + 1} <span className="text-slate-500">/ {trace.config.n_layers}</span>
        </h2>
        <div className="flex items-center gap-3">
          <span className="value text-[11px] text-slate-500">
            T = {trace.input_token_ids.length} · {block.heads.length} heads · mlp {hiddenWidth} wide
          </span>
          <button type="button" className="btn" onClick={onClose}>
            close
          </button>
        </div>
      </header>

      {/* The vertical order below is the real execution order of the block. */}
      <ol className="divide-y divide-ink-800">
        <StageRow
          label="LayerNorm 1"
          formula="y = (x − μ) / σ · g + b"
          note="normalizes the residual stream before attention"
          series={block.ln1_out}
          position={last}
        />

        <AttentionRow block={block} headIndex={headIndex} />

        <StageRow
          label="Add 1"
          formula="x = x + Attention(LN1(x))"
          note="residual merge; the stream keeps its pre-attention information"
          series={block.resid_after_attn}
          position={last}
        />

        <StageRow
          label="LayerNorm 2"
          formula="y = (x − μ) / σ · g + b"
          note="normalizes again before the MLP"
          series={block.ln2_out}
          position={last}
        />

        <MlpRow block={block} position={last} />

        <StageRow
          label="Add 2"
          formula="x = x + MLP(LN2(x))"
          note="block output, and the block's contribution to the final logit"
          series={block.resid_after_mlp}
          position={last}
        />
      </ol>
    </motion.aside>
  );
}

/**
 * One step of the block: label, formula, the real per-position norms, and the
 * change across it.
 */
function StageRow({
  label,
  formula,
  note,
  series,
  position,
}: {
  label: string;
  formula: string;
  note: string;
  series: VectorSeries;
  position: number;
}) {
  const norms = series.norms;
  const here = norms[position] ?? 0;
  const delta = series.delta_norms?.[position];
  const ev = series.explained_variance;

  return (
    <li className="px-3 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-xs font-semibold text-slate-200">{label}</h3>
        <code className="tnum text-[11px] text-accent-500/90">{formula}</code>
      </div>
      <p className="mt-0.5 text-[11px] text-slate-500">{note}</p>

      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px]">
        <Metric label={`‖x @ ${position}`} value={formatCompact(here)} strong />
        {delta !== undefined && (
          <Metric label="Δ vs input" value={`${delta >= 0 ? '+' : ''}${formatCompact(delta)}`} />
        )}
        <Metric label="mean ‖·‖" value={formatCompact(mean(norms))} />
        <Metric label="max ‖·‖" value={formatCompact(Math.max(...norms, 0))} />
        <Metric label="PC1 var" value={formatFixed(ev[0] ?? 0, 3)} />
      </div>

      <NormSparkline norms={norms} highlight={position} />
    </li>
  );
}

function AttentionRow({ block, headIndex }: { block: BlockTrace; headIndex: number }) {
  const head = block.heads[headIndex];
  if (!head) return null;
  const T = head.weights.length;
  const outNormLast = head.out_norm[T - 1] ?? 0;
  let hi = 0;
  for (const row of head.weights) for (const w of row) if (w > hi) hi = w;

  return (
    <li className="px-3 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-xs font-semibold text-slate-200">
          Multi-Head Attention
          <span className="ml-2 font-normal text-slate-500">head {head.head}</span>
        </h3>
        <code className="tnum text-[11px] text-accent-500/90">
          softmax(QKᵀ / √d<sub>head</sub> + mask)V
        </code>
      </div>
      <p className="mt-0.5 text-[11px] text-slate-500">
        {block.heads.length} heads are computed in parallel, then concatenated and mixed by
        out_proj. The matrix below is post-softmax, pre-dropout — tracing forces eval mode, so
        dropout contributes nothing.
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px]">
        <Metric label="matrix" value={`${T}×${T} causal`} strong />
        <Metric label="max weight" value={formatCompact(hi, 3)} />
        <Metric label={`‖head out @ ${T - 1}`} value={formatCompact(outNormLast)} />
        <Metric label="attn matrix" value={formatCompact(matrixNorm(head.weights))} />
      </div>

      <HeadBars block={block} activeHead={headIndex} />
    </li>
  );
}

function MlpRow({ block, position }: { block: BlockTrace; position: number }) {
  const mlp = block.mlp_hidden;
  const hiddenWidth = mlp.shape[2] ?? 0;
  const top = mlp.top[position] ?? [];

  return (
    <li className="px-3 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="text-xs font-semibold text-slate-200">
          MLP
          <span className="ml-2 font-normal text-slate-500">
            {hiddenWidth} → {hiddenWidth / 4} → {hiddenWidth}
          </span>
        </h3>
        <code className="tnum text-[11px] text-accent-500/90">GELU(W₂ · GELU(W₁ · x))</code>
      </div>
      <p className="mt-0.5 text-[11px] text-slate-500">
        the tanh approximation of GELU. Only the strongest{' '}
        <span className="value">{mlp.top_k}</span> of{' '}
        <span className="value">{hiddenWidth}</span> post-GELU activations per position are
        transmitted; the rest of the vector is not sent.
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-[11px]">
        <Metric label={`‖hidden @ ${position}`} value={formatCompact(mlp.norms[position] ?? 0)} strong />
        <Metric label="mean ‖hidden‖" value={formatCompact(mean(mlp.norms))} />
        <Metric label="max ‖hidden‖" value={formatCompact(Math.max(...mlp.norms, 0))} />
      </div>

      <div className="mt-2 space-y-1">
        {top.slice(0, 6).map((neuron, rank) => (
          <div key={neuron.index} className="flex items-center gap-2 text-[11px]">
            <span className="value w-8 shrink-0 text-slate-500">#{rank}</span>
            <span className="value w-16 shrink-0 text-slate-400">n{neuron.index}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-ink-800">
              <div
                className="h-full rounded bg-accent-500/70"
                style={{ width: `${barWidth(neuron.value, top[0]?.value ?? 0)}%` }}
              />
            </div>
            <span className="value w-16 shrink-0 text-right text-slate-300">
              {formatCompact(neuron.value, 3)}
            </span>
          </div>
        ))}
        {top.length === 0 && <p className="text-[11px] text-slate-500">no recorded neurons</p>}
      </div>
    </li>
  );
}

/** Per-head output norm at the last position, as a small bar chart. */
function HeadBars({ block, activeHead }: { block: BlockTrace; activeHead: number }) {
  const last = block.heads[0] ? block.heads[0].out_norm.length - 1 : 0;
  const values = block.heads.map((h) => h.out_norm[last] ?? 0);
  const hi = Math.max(...values, 0);
  return (
    <div className="mt-2 flex items-end gap-1">
      {values.map((v, i) => (
        <div key={i} className="flex flex-1 flex-col items-center gap-0.5">
          <div className="flex h-8 w-full items-end">
            <div
              className={[
                'w-full rounded-t',
                i === activeHead ? 'bg-accent-400' : 'bg-ink-600',
              ].join(' ')}
              style={{ height: `${barWidth(v, hi)}%` }}
            />
          </div>
          <span className="value text-[9px] text-slate-500">{i}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * Per-position L2 norms as a filled sparkline.
 *
 * Drawn rather than tabled because the shape is the point: it shows where in the
 * prompt the residual stream is large, which is where attention tends to land.
 */
function NormSparkline({ norms, highlight }: { norms: number[]; highlight: number }) {
  if (norms.length === 0) return null;
  const width = 240;
  const height = 28;
  const hi = Math.max(...norms, 0) || 1;
  const step = width / Math.max(1, norms.length - 1);
  const points = norms.map((v, i) => `${i * step},${height - (v / hi) * (height - 2)}`);
  const area = `0,${height} ${points.join(' ')} ${(norms.length - 1) * step},${height}`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="mt-2 h-7 w-full max-w-[240px]"
      preserveAspectRatio="none"
      role="img"
      aria-label="per-position norm profile"
    >
      <polygon points={area} className="fill-accent-500/15" />
      <polyline points={points.join(' ')} className="fill-none stroke-accent-500/70" strokeWidth={1} />
      {highlight >= 0 && highlight < norms.length && (
        <circle
          cx={highlight * step}
          cy={height - (norms[highlight] / hi) * (height - 2)}
          r={2.5}
          className="fill-accent-400"
        />
      )}
    </svg>
  );
}

function Metric({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="label">{label}</span>
      <span className={`value ${strong ? 'text-accent-400' : 'text-slate-300'}`}>{value}</span>
    </span>
  );
}

/** Frobenius-style summary of the attention matrix, `sqrt(Σ w²)`. */
function matrixNorm(weights: number[][]): number {
  let sum = 0;
  for (const row of weights) for (const w of row) sum += w * w;
  return Math.sqrt(sum);
}

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let total = 0;
  for (const v of values) total += v;
  return total / values.length;
}

/** Percent width, clamped: a zero maximum yields an empty bar, not `NaN%`. */
function barWidth(value: number, max: number): number {
  if (max <= 0) return 0;
  return Math.min(100, Math.max(0, (value / max) * 100));
}
