/**
 * The same attention pattern as the heatmap, as arcs.
 *
 * The heatmap answers "which cells are hot"; this answers "what is the shape of
 * the pattern" — an induction head's near-diagonal stripe, a copying head's
 * offset-by-one stripe, or a diffuse head's even fan are all immediately legible
 * as geometry and are much less legible as a grid.
 *
 * One arc per key position, from the query on the top rail to the key on the
 * bottom rail. Arc height scales with attention weight, so weight becomes
 * curvature. Strong arcs are drawn last so they sit above the weak ones rather
 * than being buried by them.
 */

import { scaleLinear } from 'd3-scale';
import { motion } from 'framer-motion';

import { formatPercent } from '@/lib/format';
import type { HeadAttention, TokenInfo } from '@/types/trace';

export function AttentionArcs({
  head,
  tokens,
  queryPosition,
  onQueryChange,
}: {
  head: HeadAttention | null;
  tokens: TokenInfo[];
  queryPosition: number;
  onQueryChange: (position: number) => void;
}) {
  if (!head || head.weights.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center px-3 text-xs text-slate-500">
        select a block to see its attention arcs
      </div>
    );
  }

  const T = head.weights.length;
  const row = head.weights[queryPosition] ?? [];
  // Only the causal half of the row is drawable; the rest is the mask.
  const arcs = row.map((w, j) => ({ j, w })).filter((a) => a.j <= queryPosition);
  const hi = arcs.reduce((m, a) => Math.max(m, a.w), 0) || 1;

  const width = 560;
  const margin = { top: 40, right: 20, bottom: 52, left: 20 };
  const innerW = width - margin.left - margin.right;
  const innerH = 190;
  const height = margin.top + innerH + margin.bottom;

  const x = scaleLinear().domain([0, Math.max(1, T - 1)]).range([margin.left, margin.left + innerW]);
  const railY = margin.top + innerH;

  // Weak arcs are flattened toward the rail rather than dropped: a fan of
  // near-zero arcs is part of the picture, and hiding them would overstate how
  // focused the head is.
  const heightScale = scaleLinear().domain([0, hi]).range([6, innerH]).clamp(true);
  const widthScale = scaleLinear().domain([0, hi]).range([0.6, 3.5]).clamp(true);

  const ordered = [...arcs].sort((a, b) => a.w - b.w);
  const strongest = arcs.reduce(
    (best, cur) => (cur.w > best.w ? cur : best),
    arcs[0] ?? { j: -1, w: 0 },
  );

  return (
    <div className="p-3">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        aria-label={`attention arcs from key positions to query ${queryPosition}`}
      >
        {/* Key rail. */}
        <line x1={margin.left} y1={railY} x2={margin.left + innerW} y2={railY} className="stroke-ink-700" />

        {ordered.map(({ j, w }) => {
          const x1 = x(queryPosition);
          const x2 = x(j);
          const lift = heightScale(w);
          // Quadratic control point halfway up: a symmetric arc whose apex height
          // is `lift`, which keeps the shape interpretable across token counts.
          const midX = (x1 + x2) / 2;
          const d = `M ${x1} ${railY} Q ${midX} ${railY - 2 * lift} ${x2} ${railY}`;
          return (
            <motion.path
              key={j}
              d={d}
              fill="none"
              strokeWidth={widthScale(w)}
              strokeLinecap="round"
              className="stroke-accent-400"
              strokeOpacity={0.18 + 0.72 * (w / hi)}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: 0.45, ease: 'easeOut', delay: (ordered.length - j) * 0.012 }}
            >
              <title>
                {`key ${j}: ${formatPercent(w, 1)}`}
              </title>
            </motion.path>
          );
        })}

        {/* Key nodes. */}
        {tokens.slice(0, T).map((t, j) => {
          const masked = j > queryPosition;
          return (
            <g key={`k-${t.id}`}>
              <circle
                cx={x(j)}
                cy={railY}
                r={masked ? 2 : 4}
                className={masked ? 'fill-ink-600' : 'fill-accent-500'}
              />
              <text
                x={x(j)}
                y={railY + 14}
                textAnchor="middle"
                fontSize={9}
                fontFamily="ui-monospace, monospace"
                className={masked ? 'fill-ink-600' : 'fill-slate-500'}
              >
                {j}
              </text>
              <text
                x={x(j)}
                y={railY + 25}
                textAnchor="middle"
                fontSize={9}
                fontFamily="ui-monospace, monospace"
                className={masked ? 'fill-ink-600' : 'fill-slate-400'}
              >
                {clip(JSON.stringify(t.piece))}
              </text>
            </g>
          );
        })}

        {/* Query marker on the top rail. */}
        <circle cx={x(queryPosition)} cy={railY} r={5} className="fill-accent-400" />
        <text
          x={x(queryPosition)}
          y={margin.top - 12}
          textAnchor="middle"
          fontSize={11}
          fontFamily="ui-monospace, monospace"
          className="fill-accent-400"
        >
          query {queryPosition} {clip(JSON.stringify(tokens[queryPosition]?.piece ?? ''))}
        </text>
      </svg>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span className="label mr-1">query</span>
        {tokens.slice(0, T).map((t, i) => (
          <button
            key={`q-${t.id}`}
            type="button"
            onClick={() => onQueryChange(i)}
            className={['btn px-1.5 py-0.5 text-[10px]', i === queryPosition ? 'btn-active' : ''].join(' ')}
          >
            <span className="value">{i}</span> {clip(JSON.stringify(t.piece))}
          </button>
        ))}
      </div>

      <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
        Arc height and thickness both scale with{' '}
        <span className="value">weights[{queryPosition}][key]</span>, so a focused head shows a few
        tall arcs and a diffuse one a flat fan. Only positions{' '}
        <span className="value">j ≤ {queryPosition}</span> are drawable. Strongest target: key{' '}
        <span className="value text-accent-400">{strongest.j >= 0 ? strongest.j : '—'}</span> at{' '}
        <span className="value">
          {strongest.j >= 0 ? formatPercent(strongest.w, 1) : '—'}
        </span>
        .
      </p>
    </div>
  );
}

function clip(s: string, max = 6): string {
  return s.length <= max ? s : `${s.slice(0, max - 1)}…`;
}
