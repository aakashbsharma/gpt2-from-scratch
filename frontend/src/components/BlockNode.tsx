/**
 * One transformer block on the pipeline.
 *
 * The residual norm is the headline number, because it is the quantity that
 * makes the residual stream's depth-wise growth visible: block 0 and block 11
 * sit side by side on the pipeline and their norms differ by ~100x on real 124M
 * weights. That growth is a real property of the architecture, not an artefact,
 * so it is shown as-is.
 */

import { motion } from 'framer-motion';

import { formatCompact } from '@/lib/format';
import type { BlockTrace, PipelineStage } from '@/types/trace';

export function BlockNode({
  stage,
  block,
  active,
  onSelect,
}: {
  stage: PipelineStage;
  block: BlockTrace | undefined;
  active: boolean;
  onSelect: () => void;
}) {
  // The last position is the one every other readout in the app is anchored to
  // (logit lens, sampling, next-token prediction), so the block norm is too.
  const residNorm = lastNorm(block?.resid_after_mlp.norms);
  const maxAttn = maxAttention(block);
  const ready = block !== undefined;

  return (
    <motion.button
      type="button"
      onClick={onSelect}
      disabled={!ready}
      aria-label={`Open block ${stage.label}`}
      className={[
        'group flex w-full flex-col items-center gap-1.5 rounded-md px-0.5 py-1',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60',
        ready ? 'cursor-pointer' : 'cursor-not-allowed',
      ].join(' ')}
      animate={{ opacity: ready ? 1 : 0.35 }}
      transition={{ duration: 0.25 }}
      whileHover={ready ? { y: -2 } : undefined}
      whileTap={ready ? { scale: 0.97 } : undefined}
    >
      <div
        className={[
          'flex h-[34px] w-[34px] items-center justify-center rounded-full border text-[11px] font-bold transition-colors',
          ready
            ? 'border-ink-600 bg-ink-800 text-slate-300 group-hover:border-accent-500/70 group-hover:text-accent-400'
            : 'border-dashed border-ink-600 bg-transparent text-ink-500',
          active ? 'ring-2 ring-accent-500/60' : '',
        ].join(' ')}
      >
        {ready ? (stage.blockIndex ?? 0) + 1 : '?'}
      </div>

      <span className="text-center text-[10px] leading-tight text-slate-400">{stage.label}</span>

      {ready ? (
        <span className="flex flex-col items-center text-[10px] leading-tight">
          <span className="value text-slate-500">‖{formatCompact(residNorm, 2)}</span>
          <span className="value text-ink-500">a {formatCompact(maxAttn, 3)}</span>
        </span>
      ) : (
        <span className="text-[10px] text-ink-500">—</span>
      )}
    </motion.button>
  );
}

function lastNorm(norms: readonly number[] | undefined): number {
  if (!norms || norms.length === 0) return 0;
  return norms[norms.length - 1];
}

/**
 * Largest attention weight anywhere in the block.
 *
 * Doubles as a crude "how focused is this layer" indicator: early layers tend to
 * be diffuse, deeper layers pick a few positions out.
 */
function maxAttention(block: BlockTrace | undefined): number {
  if (!block) return 0;
  let hi = 0;
  for (const head of block.heads) {
    for (const row of head.weights) {
      for (const w of row) if (w > hi) hi = w;
    }
  }
  return hi;
}
