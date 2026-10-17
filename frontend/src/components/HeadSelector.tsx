/**
 * Head picker for the selected block.
 *
 * The bar behind each head number is that head's context-vector L2 norm at the
 * selected query position, measured before `out_proj`. That is the right
 * quantity to rank heads by: before the output projection mixes them, it is
 * each head's own contribution, so a head with a tall bar is one the block is
 * actually using rather than one that merely has a large raw magnitude.
 */

import { scaleLinear } from 'd3-scale';

import { formatCompact } from '@/lib/format';
import type { BlockTrace } from '@/types/trace';

export function HeadSelector({
  block,
  selectedHead,
  queryPosition,
  onSelect,
}: {
  block: BlockTrace | undefined;
  selectedHead: number;
  queryPosition: number;
  onSelect: (head: number) => void;
}) {
  if (!block || block.heads.length === 0) {
    return (
      <div className="px-3 py-3 text-xs text-slate-500">no block selected</div>
    );
  }

  const norms = block.heads.map((h) => h.out_norm[queryPosition] ?? 0);
  const x = scaleLinear()
    .domain([0, Math.max(...norms, 0) || 1])
    .range([0, 100]);

  return (
    <div className="px-3 py-2.5">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="label">head · ‖context @ {queryPosition}</span>
        <span className="value text-[10px] text-slate-500">
          max {formatCompact(Math.max(...norms, 0))}
        </span>
      </div>

      <div className="flex flex-wrap gap-1">
        {block.heads.map((head, i) => {
          const active = i === selectedHead;
          return (
            <button
              key={head.head}
              type="button"
              onClick={() => onSelect(head.head)}
              aria-pressed={active}
              title={`head ${head.head}: context norm ${formatCompact(norms[i])}`}
              className={[
                'relative flex h-9 w-9 items-end justify-center overflow-hidden rounded border pb-0.5',
                'text-[10px] font-semibold transition-colors',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60',
                active
                  ? 'border-accent-500 text-accent-400'
                  : 'border-ink-600 text-slate-400 hover:border-ink-500 hover:text-slate-200',
              ].join(' ')}
            >
              <span
                className={[
                  'absolute inset-x-0 bottom-0',
                  active ? 'bg-accent-500/25' : 'bg-ink-600/50',
                ].join(' ')}
                style={{ height: `${x(norms[i])}%` }}
                aria-hidden
              />
              <span className="relative">{head.head}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
