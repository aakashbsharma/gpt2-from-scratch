import { formatPercent, formatTemperature, formatTopK } from '@/lib/format';
import type { SamplingStep, Trace } from '@/types/trace';

const MAX_UNFILTERED_ROWS = 50;

export function SamplingFunnel({ trace }: { trace: Trace | null }) {
  const sampling = trace?.sampling ?? null;

  if (!trace) return <SamplingMessage>No trace yet.</SamplingMessage>;
  if (!sampling) {
    return (
      <div className="px-3 py-3 text-[11px] leading-relaxed text-slate-500">
        No sampling trace is attached to this forward pass. The model's top prediction is{' '}
        <span className="text-slate-300">{JSON.stringify(trace.top_pieces[0] ?? '')}</span>{' '}
        ({formatPercent(trace.top_probs[0] ?? 0, 2)}).
      </div>
    );
  }

  const candidates = visibleCandidates(sampling);
  const maxRaw = Math.max(...sampling.candidates.map((candidate) => candidate.p_raw), 1e-12);
  const maxTemp = Math.max(...sampling.candidates.map((candidate) => candidate.p_temp), 1e-12);
  const maxFinal = Math.max(...sampling.candidates.map((candidate) => candidate.p_final), 1e-12);
  const omitted = sampling.candidates.length - candidates.length;
  const firstDiscarded = candidates.findIndex((candidate) => !candidate.kept);

  return (
    <div className="px-3 py-3">
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-500">
        <span>step {sampling.step + 1}</span>
        <span>T {formatTemperature(sampling.temperature)}</span>
        <span>top-k {formatTopK(sampling.top_k)}</span>
        {sampling.seed !== null && <span>seed {sampling.seed}</span>}
        <span>{sampling.candidates.length.toLocaleString()} candidates traced</span>
      </div>

      <div className="grid grid-cols-[minmax(7rem,1.2fr)_repeat(3,minmax(4rem,1fr))] gap-2 border-b border-ink-700 pb-1 text-[10px] uppercase tracking-wide text-slate-500">
        <span>Candidate</span>
        <span>Raw</span>
        <span>Temperature</span>
        <span>Final</span>
      </div>
      <ol className="max-h-80 overflow-y-auto">
        {candidates.map((candidate, index) => {
          const chosen = candidate.id === sampling.chosen.id;
          const isCut =
            sampling.top_k !== null &&
            (firstDiscarded === index ||
              (firstDiscarded < 0 && candidate.rank === sampling.top_k));
          return (
            <li
              key={candidate.id}
              title={`logit: ${candidate.logit}`}
              className={[
                'grid grid-cols-[minmax(7rem,1.2fr)_repeat(3,minmax(4rem,1fr))] items-center gap-2 border-b border-ink-800 py-1.5 text-[10px]',
                candidate.kept ? '' : 'text-slate-600',
                chosen ? 'bg-accent-500/5 text-accent-300' : '',
                isCut
                  ? firstDiscarded === index
                    ? 'border-t-2 border-dashed border-amber-500/70'
                    : 'border-b-2 border-dashed border-amber-500/70'
                  : '',
              ].join(' ')}
            >
              <span className="min-w-0 truncate" title={candidate.token}>
                <span className="value mr-1 text-slate-500">{candidate.rank}.</span>
                {JSON.stringify(candidate.token)}
                {chosen && <span className="ml-1 text-accent-400">chosen</span>}
              </span>
              <ProbabilityBar value={candidate.p_raw} max={maxRaw} />
              <ProbabilityBar value={candidate.p_temp} max={maxTemp} />
              <ProbabilityBar
                value={candidate.p_final}
                max={maxFinal}
                muted={!candidate.kept}
              />
            </li>
          );
        })}
      </ol>

      {omitted > 0 && (
        <p className="mt-1 text-[10px] text-slate-500">
          Showing {candidates.length} of {sampling.candidates.length.toLocaleString()} traced
          candidates; {omitted.toLocaleString()} more are omitted from this view.
        </p>
      )}
      <p className="mt-1 text-[10px] text-slate-500">
        Top-k cut line separates discarded candidates. Unlisted tail mass: raw{' '}
        {formatPercent(sampling.tail.p_raw, 2)}, temperature{' '}
        {formatPercent(sampling.tail.p_temp, 2)}.
      </p>
    </div>
  );
}

function visibleCandidates(sampling: SamplingStep) {
  if (sampling.top_k !== null) {
    return sampling.candidates.slice(0, Math.max(sampling.top_k, MAX_UNFILTERED_ROWS));
  }
  const candidates = sampling.candidates.slice(0, MAX_UNFILTERED_ROWS);
  const chosen = sampling.candidates.find((candidate) => candidate.id === sampling.chosen.id);
  if (chosen && !candidates.some((candidate) => candidate.id === chosen.id)) {
    candidates.push(chosen);
    candidates.sort((left, right) => left.rank - right.rank);
  }
  return candidates;
}

function ProbabilityBar({
  value,
  max,
  muted = false,
}: {
  value: number;
  max: number;
  muted?: boolean;
}) {
  return (
    <span className="relative flex min-w-0 items-center justify-end overflow-hidden rounded bg-ink-800 px-1 py-0.5">
      <span
        className={`absolute inset-y-0 left-0 ${muted ? 'bg-slate-600/20' : 'bg-accent-500/20'}`}
        style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
        aria-hidden
      />
      <span className="value relative truncate text-right text-slate-300">
        {formatPercent(value, 2)}
      </span>
    </span>
  );
}

function SamplingMessage({ children }: { children: string }) {
  return <div className="px-3 py-4 text-xs text-slate-500">{children}</div>;
}
