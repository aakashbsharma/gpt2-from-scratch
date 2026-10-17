import { formatPercent } from '@/lib/format';
import type { SamplingStep } from '@/types/trace';

export function SamplingStats({ sampling }: { sampling: SamplingStep | null }) {
  if (!sampling) return null;

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-ink-800 px-3 py-3 text-[11px] sm:grid-cols-3">
      <Stat label="chosen rank" value={`#${sampling.chosen.rank}`} />
      <Stat label="p_raw" value={formatPercent(sampling.chosen.p_raw, 2)} />
      <Stat label="p_final" value={formatPercent(sampling.chosen.p_final, 2)} accent />
      <Stat label="entropy raw" value={sampling.entropy_raw.toFixed(4)} />
      <Stat label="entropy final" value={sampling.entropy_final.toFixed(4)} />
      <span className="flex items-center">
        <span
          className={[
            'rounded px-1.5 py-0.5 text-[10px]',
            sampling.was_argmax
              ? 'bg-amber-500/10 text-amber-300'
              : 'bg-ink-700 text-slate-400',
          ].join(' ')}
        >
          {sampling.was_argmax ? 'argmax' : 'sampled'}
        </span>
      </span>
      <p className="col-span-full text-[10px] text-slate-500">
        top-k kept {formatPercent(sampling.topk_mass_kept, 2)} of the temperature-scaled mass before
        renormalizing.
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <span className="flex items-baseline justify-between gap-2">
      <span className="text-slate-500">{label}</span>
      <span className={`value ${accent ? 'text-accent-400' : 'text-slate-300'}`}>{value}</span>
    </span>
  );
}
