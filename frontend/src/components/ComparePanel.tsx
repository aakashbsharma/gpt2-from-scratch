/**
 * Side-by-side comparison of two registered models on one prompt.
 *
 * This runs its own generations rather than reusing the main store's, because it
 * needs two traces at once — the store holds exactly one current step by design.
 * Each side gets an independent socket and its own trace state, and the panel
 * only renders when both have delivered a trace for the chosen layer.
 *
 * What is worth comparing is not the two outputs' prose but the divergence point:
 * the layer at which their logit lenses first disagree is usually far earlier than
 * where the outputs visibly differ, which is the lesson the panel exists to show.
 */

import { motion } from 'framer-motion';

import { formatCompact, formatPercent } from '@/lib/format';
import { useModels } from '@/hooks/useModels';
import type { LogitLensEntry, Trace } from '@/types/trace';

import { useCompareRun } from '@/hooks/useCompareRun';
import { LogitLensStrip } from '@/components/LogitLens';
import { useTraceStore } from '@/store/useTraceStore';

export function ComparePanel() {
  const { models, ready, loading, offline } = useModels();
  const run = useCompareRun(ready);
  const selectedPosition = useTraceStore((s) => s.selectedPosition);

  return (
    <section className="panel" aria-label="Model comparison">
      <header className="panel-header">
        <h2 className="panel-title">Compare models</h2>
        <span className="text-[11px] text-slate-500">
          same prompt, two registered checkpoints, one layer
        </span>
      </header>

      <div className="space-y-3 px-3 py-3">
        {loading && <p className="text-xs text-slate-500">loading registry…</p>}
        {offline && !loading && (
          <p className="text-xs text-amber-400/90">
            the backend registry is unreachable, so comparison is disabled. Replay mode still
            works from the exported traces.
          </p>
        )}

        {!loading && ready.length < 2 && !offline && (
          <p className="text-xs text-slate-500">
            comparison needs two ready models; the registry reports{' '}
            <span className="value">{ready.length}</span>.{' '}
            {models
              .filter((m) => !m.ready)
              .map((m) => `${m.id} is not ready (checkpoint missing)`)
              .join('; ')}
          </p>
        )}

        <form
          className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto_auto]"
          onSubmit={(e) => {
            e.preventDefault();
            void run.start();
          }}
        >
          <label className="flex flex-col gap-1">
            <span className="label">prompt (identical for both models)</span>
            <input
              value={run.prompt}
              onChange={(e) => run.setPrompt(e.target.value)}
              disabled={ready.length < 2}
              className="tnum rounded border border-ink-600 bg-ink-850 px-2 py-1.5 text-xs text-slate-200
                         placeholder:text-ink-500 focus:border-accent-500 focus:outline-none"
              placeholder="Every effort moves you"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="label">model A</span>
            <select
              value={run.modelA}
              onChange={(e) => run.setModelA(e.target.value)}
              disabled={ready.length < 2}
              className="rounded border border-ink-600 bg-ink-850 px-2 py-1.5 text-xs text-slate-200 focus:border-accent-500 focus:outline-none"
            >
              {ready.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="label">model B</span>
            <select
              value={run.modelB}
              onChange={(e) => run.setModelB(e.target.value)}
              disabled={ready.length < 2}
              className="rounded border border-ink-600 bg-ink-850 px-2 py-1.5 text-xs text-slate-200 focus:border-accent-500 focus:outline-none"
            >
              {ready.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>

          <div className="flex items-end gap-2 sm:col-span-3">
            <button
              type="submit"
              className="btn btn-active"
              disabled={run.busy || ready.length < 2 || run.prompt.trim().length === 0}
            >
              {run.busy ? 'running…' : 'run both'}
            </button>
            {run.busy && (
              <button type="button" className="btn" onClick={run.cancel}>
                cancel
              </button>
            )}
            {run.error && <span className="text-[11px] text-rose-400">{run.error}</span>}
          </div>
        </form>

        {(run.resultA || run.resultB) && (
          <>
            <div className="flex flex-wrap items-center gap-2 border-t border-ink-800 pt-3">
              <span className="label">attention layer</span>
              {run.layers.map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => run.setLayer(l)}
                  className={['btn px-2 py-0.5 text-[11px]', run.layer === l ? 'btn-active' : ''].join(' ')}
                >
                  {l + 1}
                </button>
              ))}
              <span className="label ml-2">head</span>
              {run.headCount > 0 &&
                Array.from({ length: run.headCount }, (_, h) => (
                  <button
                    key={h}
                    type="button"
                    onClick={() => run.setHead(h)}
                    className={['btn px-2 py-0.5 text-[11px]', run.head === h ? 'btn-active' : ''].join(' ')}
                  >
                    {h}
                  </button>
                ))}
              <span className="ml-auto text-[11px] text-slate-500">
                {divergenceNote(run.resultA?.trace ?? null, run.resultB?.trace ?? null)}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <CompareSide
                title={run.modelA}
                result={run.resultA}
                layer={run.layer}
                head={run.head}
                other={run.resultB?.trace ?? null}
                selectedPosition={selectedPosition}
              />
              <CompareSide
                title={run.modelB}
                result={run.resultB}
                layer={run.layer}
                head={run.head}
                other={run.resultA?.trace ?? null}
                selectedPosition={selectedPosition}
              />
            </div>
          </>
        )}
      </div>
    </section>
  );
}

interface SideResult {
  modelId: string;
  label: string;
  text: string;
  trace: Trace | null;
  error: string | null;
  pending: boolean;
}

function CompareSide({
  title,
  result,
  layer,
  head,
  other,
  selectedPosition,
}: {
  title: string;
  result: SideResult | null;
  layer: number;
  head: number;
  other: Trace | null;
  selectedPosition: number | null;
}) {
  if (!result) {
    return (
      <div className="rounded border border-dashed border-ink-600 p-3 text-xs text-slate-500">
        waiting for {title}…
      </div>
    );
  }
  if (result.error) {
    return (
      <div className="rounded border border-rose-500/40 p-3 text-xs text-rose-400">
        {title}: {result.error}
      </div>
    );
  }
  if (result.pending) {
    return (
      <div className="rounded border border-dashed border-ink-600 p-3 text-xs text-slate-500">
        {title}: generating…
      </div>
    );
  }

  const trace = result.trace;
  const block = trace?.blocks[layer] ?? null;
  const headData = block?.heads[head] ?? null;
  const lens = trace?.logit_lens ?? [];
  const deepest = lens[lens.length - 1] ?? null;
  const position = Math.min(
    selectedPosition ?? (trace?.input_token_ids.length ?? 1) - 1,
    (trace?.input_token_ids.length ?? 1) - 1,
  );
  const agreement = topPiecesAgree(trace, other, layer, position);
  const deepestPosition = deepest?.positions[deepest.positions.length - 1];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-2 rounded border border-ink-700 bg-ink-850/60 p-3"
    >
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-xs font-semibold text-slate-200">{result.label || result.modelId}</h3>
        {deepest && (
          <span className="value text-[11px] text-accent-400">
            top {JSON.stringify(deepestPosition?.top_pieces[0] ?? '')} ·{' '}
            {formatPercent(deepestPosition?.top_probs[0] ?? 0, 1)}
          </span>
        )}
      </div>

      <p className="rounded bg-ink-900 px-2 py-1.5 text-[11px] leading-relaxed text-slate-300">
        {result.text || <span className="text-slate-500">(empty)</span>}
      </p>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
        <Metric label="layers" value={String(trace?.config.n_layers ?? '—')} />
        <Metric label="emb_dim" value={String(trace?.config.emb_dim ?? '—')} />
        <Metric
          label={`block ${layer + 1} resid ‖`}
          value={formatCompact(block?.resid_after_mlp.norms.slice(-1)[0] ?? 0, 2)}
        />
        <Metric
          label="head ctx ‖"
          value={formatCompact(headData?.out_norm.slice(-1)[0] ?? 0, 3)}
        />
        <Metric
          label="logit max"
          value={formatCompact(trace?.logits.max ?? 0, 1)}
        />
        <Metric
          label="lens top-1"
          value={lensEntryOf(lens, layer)?.positions[position]?.top_pieces[0] ?? '—'}
        />
      </div>

      {/* Where this model's prediction forms, layer by layer. */}
      <div className="border-t border-ink-800 pt-2">
        <LogitLensStrip
          trace={trace}
          position={position}
          label={`logit lens · ${lens.length} depths`}
        />
      </div>

      {/* The head's strongest key target, next to the other model's. */}
      <div className="border-t border-ink-800 pt-2">
        <p className="label">
          layer {layer + 1} · head {head} · strongest key at final position
        </p>
        <ol className="mt-1 space-y-0.5">
          {(headData?.weights.slice(-1)[0] ?? [])
            .map((w, j) => ({ j, w }))
            .sort((a, b) => b.w - a.w)
            .slice(0, 5)
            .map(({ j, w }) => (
              <li key={j} className="flex items-baseline gap-2 text-[11px]">
                <span className="value w-5 shrink-0 text-slate-500">{j}</span>
                <span className="w-24 shrink-0 truncate text-slate-300">
                  {JSON.stringify(trace?.tokens[j]?.piece ?? '')}
                </span>
                <span className="value text-slate-400">{formatPercent(w, 1)}</span>
              </li>
            ))}
          {!headData && <li className="text-[11px] text-slate-500">—</li>}
        </ol>
      </div>

      {agreement !== null && (
        <p
          className={[
            'text-[11px]',
            agreement ? 'text-emerald-400/90' : 'text-amber-400/90',
          ].join(' ')}
        >
          {agreement
            ? `both models predict the same top token at layer ${layer + 1}`
            : `the two models already disagree at layer ${layer + 1}`}
        </p>
      )}
    </motion.div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline justify-between gap-2">
      <span className="label">{label}</span>
      <span className="value truncate text-slate-300">{value}</span>
    </span>
  );
}

function lensEntryOf(lens: LogitLensEntry[], layer: number): LogitLensEntry | undefined {
  return lens.find((entry) => entry.layer === layer + 1);
}

/** Do both models' lenses agree on the top token at this depth? */
function topPiecesAgree(
  a: Trace | null,
  b: Trace | null,
  layer: number,
  position: number,
): boolean | null {
  const ea = lensEntryOf(a?.logit_lens ?? [], layer);
  const eb = lensEntryOf(b?.logit_lens ?? [], layer);
  if (!ea || !eb) return null;
  return (
    (ea.positions[position]?.top_pieces[0] ?? null) ===
    (eb.positions[position]?.top_pieces[0] ?? null)
  );
}

/**
 * The deepest layer at which the two lenses agree on the top token.
 *
 * Everything below that depth is guaranteed identical, which bounds how early the
 * divergence happened without shipping both full traces' attention.
 */
function divergenceNote(a: Trace | null, b: Trace | null): string {
  if (!a || !b) return '';
  const len = Math.min(a.logit_lens.length, b.logit_lens.length);
  for (let i = 0; i < len; i += 1) {
    const aPositions = a.logit_lens[i].positions;
    const bPositions = b.logit_lens[i].positions;
    if (
      (aPositions[aPositions.length - 1]?.top_pieces[0] ?? '') !==
      (bPositions[bPositions.length - 1]?.top_pieces[0] ?? '')
    ) {
      return `first lens disagreement at layer ${i + 1}`;
    }
  }
  return 'lenses agree at every depth';
}
