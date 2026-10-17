/**
 * The T×T attention matrix of the selected head, drawn with D3.
 *
 * Layout: query rows top to bottom, key columns left to right, matching the
 * `weights[i][j]` = "query i attends to key j" convention the type docs use.
 *
 * The causal mask is drawn, not just implied. The upper triangle is exactly
 * `0.0` by construction — structurally empty, not merely small — so it is
 * rendered as a hatched dead zone. That makes the mask *visible*, which is the
 * single most common thing people misread in an attention plot.
 *
 * Colour: `interpolateViridis` over `[0, max]` of this head. A shared domain
 * across heads would be more comparable but would wash out sharp heads, and the
 * question this view answers ("where does *this* head look?") is within-head.
 */

import { scaleSequential } from 'd3-scale';
import { interpolateViridis } from 'd3-scale-chromatic';

import type { HeadAttention, TokenInfo } from '@/types/trace';

import { formatCompact, formatPercent } from '@/lib/format';

export function AttentionHeatmap({
  head,
  tokens,
  queryPosition,
  onQueryChange,
}: {
  head: HeadAttention | null;
  tokens: TokenInfo[];
  /** Row highlighted, i.e. the query whose distribution is expanded below. */
  queryPosition: number;
  onQueryChange: (position: number) => void;
}) {
  if (!head || head.weights.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center px-3 text-xs text-slate-500">
        select a block to see its attention matrix
      </div>
    );
  }

  const T = head.weights.length;
  const weights = head.weights;

  // Domain is [0, max of this head]. `scaleSequential` expects a domain; passing
  // an explicit range keeps the darkest cell genuinely black at 0.
  let hi = 0;
  for (const row of weights) for (const w of row) if (w > hi) hi = w;
  const color = scaleSequential(interpolateViridis).domain([0, hi || 1]);

  const queryRow = weights[queryPosition] ?? weights[queryPosition % T] ?? [];
  // Sorted strongest-first, so the readout below the matrix answers "what is
  // this position looking at" without the reader scanning the grid.
  const attended = queryRow
    .map((w, j) => ({ j, w }))
    .filter((c) => c.j <= queryPosition)
    .sort((a, b) => b.w - a.w);

  return (
    <div className="p-3">
      <div className="flex flex-wrap gap-4">
        <div className="min-w-0 flex-1">
          <HeatGrid
            weights={weights}
            tokens={tokens}
            color={color}
            queryPosition={queryPosition}
            onQueryChange={onQueryChange}
            outNorm={head.out_norm}
          />
        </div>

        <div className="w-full shrink-0 space-y-3 sm:w-56">
          <Colorbar hi={hi} color={color} />

          <div>
            <p className="label">
              query {queryPosition} attends to
            </p>
            <ul className="mt-1 space-y-1">
              {attended.slice(0, 8).map(({ j, w }) => (
                <li key={j} className="flex items-center gap-2 text-[11px]">
                  <span className="value w-16 shrink-0 truncate text-slate-400">
                    {JSON.stringify(tokens[j]?.piece ?? '')}
                  </span>
                  <span className="value w-6 shrink-0 text-ink-500">{j}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded bg-ink-800">
                    <div
                      className="h-full rounded"
                      style={{
                        width: `${pct(w, queryRow[queryPosition] || 1)}%`,
                        background: color(w),
                      }}
                    />
                  </div>
                  <span className="value w-11 shrink-0 text-right text-slate-300">
                    {formatPercent(w, 1)}
                  </span>
                </li>
              ))}
              {attended.length === 0 && <li className="text-[11px] text-slate-500">—</li>}
            </ul>
          </div>

          <p className="text-[11px] leading-relaxed text-slate-500">
            Rows sum to 1. The hatched upper triangle is the causal mask:{' '}
            <span className="value">weights[i][j] === 0</span> for{' '}
            <span className="value">j &gt; i</span>, structurally. Values are
            float16-quantized and rounded to 4 decimals, so a row sum is 1 to
            within ~4e-4 rather than exactly 1.
          </p>
        </div>
      </div>
    </div>
  );
}

function HeatGrid({
  weights,
  tokens,
  color,
  queryPosition,
  onQueryChange,
  outNorm,
}: {
  weights: number[][];
  tokens: TokenInfo[];
  color: (v: number) => string;
  queryPosition: number;
  onQueryChange: (position: number) => void;
  outNorm: number[];
}) {
  const T = weights.length;
  const maxNorm = Math.max(...outNorm, 0) || 1;

  return (
    <div className="flex gap-1">
      {/* Column headers: the key positions, rotated so they fit. */}
      <div className="flex shrink-0 flex-col-reverse">
        <div className="h-4" />
        {tokens.slice(0, T).map((t, j) => (
          <div
            key={t.id}
            className="flex items-center justify-center"
            style={{ height: `${cellSize}px`, width: '1.4rem' }}
            title={`key ${j}: ${JSON.stringify(t.piece)}`}
          >
            <span className="value -rotate-90 whitespace-nowrap text-[9px] text-slate-500">
              {shortPiece(t.piece, j)}
            </span>
          </div>
        ))}
      </div>

      <div className="min-w-0">
        <svg
          width={T * cellSize + 1}
          height={T * cellSize + 1}
          role="img"
          aria-label={`${T} by ${T} attention matrix, causal mask visible`}
          className="overflow-visible rounded-sm border border-ink-700"
        >
          {weights.map((row, i) =>
            row.map((w, j) => {
              const masked = j > i;
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={j * cellSize}
                    y={i * cellSize}
                    width={cellSize}
                    height={cellSize}
                    fill={masked ? 'transparent' : color(w)}
                    className={masked ? 'mask-hatch' : ''}
                  />
                  {masked && (
                    <rect
                      x={j * cellSize}
                      y={i * cellSize}
                      width={cellSize}
                      height={cellSize}
                      fill="none"
                      className="stroke-ink-800"
                      strokeWidth={0.5}
                    />
                  )}
                </g>
              );
            }),
          )}

          {/* Diagonal: a position always attends to itself. */}
          {Array.from({ length: T }, (_, i) => (
            <line
              key={`diag-${i}`}
              x1={i * cellSize + cellSize}
              y1={i * cellSize}
              x2={i * cellSize}
              y2={i * cellSize + cellSize}
              className="stroke-accent-400/25"
              strokeWidth={1}
            />
          ))}

          {/* Row highlight for the selected query. */}
          <rect
            x={0}
            y={queryPosition * cellSize}
            width={T * cellSize}
            height={cellSize}
            fill="none"
            className="stroke-accent-400"
            strokeWidth={1.5}
            pointerEvents="none"
          />
        </svg>

        {/* Row labels. A button per row, so clicking the axis selects a query. */}
        <div className="mt-1 flex flex-col">
          {tokens.slice(0, T).map((t, i) => (
            <button
              key={`${t.id}-row`}
              type="button"
              onClick={() => onQueryChange(i)}
              title={`query ${i}: ${JSON.stringify(t.piece)}`}
              className={[
                'flex items-center gap-1 text-left text-[9px] leading-none',
                i === queryPosition ? 'text-accent-400' : 'text-slate-500 hover:text-slate-300',
              ].join(' ')}
              style={{ height: `${cellSize}px`, width: '100%' }}
            >
              <span className="value w-3 shrink-0">{i}</span>
              <span className="truncate">{JSON.stringify(t.piece)}</span>
              <span
                className="ml-auto shrink-0 rounded-sm bg-ink-700"
                style={{
                  width: `${Math.max(2, ((outNorm[i] ?? 0) / maxNorm) * 24)}px`,
                  height: '3px',
                }}
                title={`context norm ${formatCompact(outNorm[i] ?? 0)}`}
              />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function Colorbar({ hi, color }: { hi: number; color: (v: number) => string }) {
  const stops = Array.from({ length: 24 }, (_, i) => color((i / 23) * hi));
  return (
    <div>
      <p className="label">attention weight</p>
      <div className="mt-1 h-2 w-full rounded" style={{ background: `linear-gradient(to right, ${stops.join(',')})` }} />
      <div className="mt-0.5 flex justify-between text-[10px] text-slate-500">
        <span className="value">0</span>
        <span className="value">{formatCompact(hi, 3)}</span>
      </div>
    </div>
  );
}

/** Cell edge in px. Small, because a 12-token prompt must still fit a column. */
const cellSize = 18;

/** Two characters of the piece, so adjacent labels stay distinguishable. */
function shortPiece(piece: string, index: number): string {
  const visible = piece.replace(/\n/g, ' ').replace(/^\s+/, '');
  if (visible.length === 0) return String(index);
  return visible.length <= 3 ? visible : visible.slice(0, 3);
}

function pct(value: number, rowMax: number): number {
  if (rowMax <= 0) return 0;
  return Math.min(100, (value / rowMax) * 100);
}
