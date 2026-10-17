/**
 * Read the residual stream through the model's final LayerNorm and output head.
 */

import { curveMonotoneX, line } from 'd3-shape';
import { scaleLinear } from 'd3-scale';
import { motion } from 'framer-motion';

import { formatPercent } from '@/lib/format';
import { useTraceStore } from '@/store/useTraceStore';
import type { LogitLensEntry, Trace } from '@/types/trace';

export function LogitLens({ trace }: { trace: Trace | null }) {
  const activeBlock = useTraceStore((s) => s.selectedBlock);
  const selectedPosition = useTraceStore((s) => s.selectedPosition);
  const selectBlock = useTraceStore((s) => s.selectBlock);
  const lens = trace?.logit_lens ?? [];

  if (!trace || lens.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-3 py-8 text-xs text-slate-500">
        no logit lens yet
      </div>
    );
  }

  const position = Math.min(
    selectedPosition ?? trace.input_token_ids.length - 1,
    trace.input_token_ids.length - 1,
  );
  const points = lens.map((entry, index) => ({ index, probability: entry.final_token_prob }));
  const x = scaleLinear().domain([0, Math.max(1, lens.length - 1)]).range([12, 388]);
  const maxProbability = Math.max(...points.map((point) => point.probability), 0.01);
  const y = scaleLinear().domain([0, maxProbability]).range([76, 8]);
  const path = line<{ index: number; probability: number }>()
    .x((point) => x(point.index))
    .y((point) => y(point.probability))
    .curve(curveMonotoneX)(points);
  const answerFormsAt = lens.findIndex((entry) => entry.final_token_rank === 1);
  const formingEntry = answerFormsAt >= 0 ? lens[answerFormsAt] : null;

  return (
    <div className="p-3">
      <p className="mb-2 text-[11px] leading-relaxed text-slate-500">
        Early layers are rough: the output head was trained to read the final layer, so early
        guesses are approximate.
      </p>

      <div className="mb-2 rounded border border-ink-800 bg-ink-900/50 px-2 py-1">
        <div className="mb-1 flex items-baseline justify-between text-[10px]">
          <span className="label">final token probability · last context position</span>
          {formingEntry && (
            <span className="value text-accent-400">
              answer forms at layer {formingEntry.layer}
            </span>
          )}
        </div>
        <svg
          viewBox="0 0 400 88"
          className="h-20 w-full overflow-visible"
          role="img"
          aria-label="Final predicted token probability across logit-lens layers"
        >
          <line x1="12" x2="388" y1="77" y2="77" className="stroke-ink-700" />
          {path && <path d={path} fill="none" className="stroke-accent-500" strokeWidth="2" />}
          {points.map((point) => (
            <circle
              key={point.index}
              cx={x(point.index)}
              cy={y(point.probability)}
              r={point.index === answerFormsAt ? 4 : 2.5}
              className={
                point.index === answerFormsAt
                  ? 'fill-accent-300 stroke-ink-950'
                  : 'fill-accent-500'
              }
            />
          ))}
          {points.map((point, index) => (
            <text
              key={`layer-${point.index}`}
              x={x(point.index)}
              y="87"
              textAnchor="middle"
              className="fill-slate-500 text-[7px]"
            >
              {lens[index].layer === 'emb' ? 'emb' : lens[index].layer}
            </text>
          ))}
        </svg>
      </div>

      <p className="mb-1 text-[10px] text-slate-500">
        Position {position}: top three lens predictions. Rank and bar track the final token at the
        last context position.
      </p>
      <ol className="@container space-y-0.5">
        {lens.map((entry, index) => (
          <LensRow
            key={entry.layer}
            entry={entry}
            previous={index > 0 ? lens[index - 1] : null}
            position={position}
            isLast={index === lens.length - 1}
            isActive={entry.layer !== 'emb' && activeBlock === entry.layer - 1}
            barScale={(probability) => (maxProbability ? (probability / maxProbability) * 100 : 0)}
            onSelect={() => {
              if (entry.layer !== 'emb') selectBlock(entry.layer - 1);
            }}
          />
        ))}
      </ol>
    </div>
  );
}

function LensRow({
  entry,
  previous,
  position,
  isLast,
  isActive,
  barScale,
  onSelect,
}: {
  entry: LogitLensEntry;
  previous: LogitLensEntry | null;
  position: number;
  isLast: boolean;
  isActive: boolean;
  barScale: (probability: number) => number;
  onSelect: () => void;
}) {
  const selected = entry.positions[position];
  const previousTop = previous?.positions[position]?.top_pieces[0] ?? null;
  const top = selected?.top_pieces[0] ?? '';
  const rank = entry.final_token_rank;

  return (
    <motion.li
      layout="position"
      transition={{ duration: 0.25 }}
      className={[
        'group flex items-center gap-2 rounded px-1.5 py-1',
        isActive ? 'bg-accent-500/10' : 'hover:bg-ink-800',
      ].join(' ')}
    >
      <button
        type="button"
        onClick={onSelect}
        disabled={entry.layer === 'emb'}
        className="flex w-full min-w-0 items-center gap-2 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60 disabled:cursor-default"
        title={entry.layer === 'emb' ? 'summed embedding lens' : `open block ${entry.layer}`}
      >
        <span className="value w-7 shrink-0 text-slate-500">
          {entry.layer === 'emb' ? 'emb' : `L${entry.layer}`}
        </span>
        <span className="value w-7 shrink-0 text-right text-[10px] text-slate-500">
          #{rank}
        </span>
        <span className="min-w-0 flex-1 truncate text-[10px] text-slate-300">
          {selected?.top_pieces
            .map(
              (piece, index) =>
                `${JSON.stringify(piece)} ${formatPercent(selected.top_probs[index] ?? 0, 1)}`,
            )
            .join(' · ')}
        </span>
        <span
          className="relative h-3 min-w-0 flex-1 overflow-hidden rounded bg-ink-800"
          title={`final token probability ${entry.final_token_prob}`}
        >
          <span
            className={[
              'absolute inset-y-0 left-0 rounded',
              isLast ? 'bg-accent-500/70' : 'bg-ink-500',
            ].join(' ')}
            style={{ width: `${barScale(entry.final_token_prob)}%` }}
            aria-hidden
          />
        </span>
        <span className="value w-12 shrink-0 text-right text-[11px] text-slate-300">
          {formatPercent(entry.final_token_prob, 1)}
        </span>
        <span className="value w-3 shrink-0 text-center text-[10px] text-accent-500/70">
          {previousTop !== null && previousTop !== top ? '↻' : ''}
        </span>
      </button>
    </motion.li>
  );
}

/** Standalone comparison lens: read-only, with both traces using one position. */
export function LogitLensStrip({
  trace,
  label,
  position,
}: {
  trace: Trace | null;
  label: string;
  position?: number;
}) {
  const lens = trace?.logit_lens ?? [];
  if (!trace || lens.length === 0) {
    return <p className="text-[11px] text-slate-500">{label}: no lens</p>;
  }
  const selectedPosition = Math.min(position ?? trace.input_token_ids.length - 1, trace.input_token_ids.length - 1);

  return (
    <div className="min-w-0">
      <p className="label mb-1">{label} · position {selectedPosition}</p>
      <ol className="space-y-0.5">
        {lens.map((entry) => {
          const selected = entry.positions[selectedPosition];
          return (
            <li key={entry.layer} className="flex min-w-0 items-baseline gap-2 text-[11px]">
              <span className="value w-8 shrink-0 text-slate-500">
                {entry.layer === 'emb' ? 'emb' : `L${entry.layer}`}
              </span>
              <span className="min-w-0 flex-1 truncate text-slate-200">
                {selected?.top_pieces.map((piece) => JSON.stringify(piece)).join(' · ')}
              </span>
              <span className="value shrink-0 text-slate-400">
                {formatPercent(entry.final_token_prob, 1)}
              </span>
              <span className="value w-8 shrink-0 text-right text-slate-500">
                #{entry.final_token_rank}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
