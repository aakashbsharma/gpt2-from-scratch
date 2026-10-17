import type { WSStepMessage } from '@/types/trace';

export function SamplingHistory({
  steps,
  currentStep,
  onSelectStep,
}: {
  steps: WSStepMessage[];
  currentStep: number;
  onSelectStep: (step: number) => void;
}) {
  if (steps.length === 0) return null;
  const probabilities = steps.map((step) => step.trace?.sampling?.chosen.p_final ?? null);
  const available = probabilities.filter((prob): prob is number => prob !== null);
  const max = Math.max(...available, 1e-12);
  const segments: string[][] = [];
  let segment: string[] = [];
  probabilities.forEach((prob, index) => {
    if (prob === null) {
      if (segment.length) segments.push(segment);
      segment = [];
      return;
    }
    const x = steps.length <= 1 ? 50 : (index / (steps.length - 1)) * 100;
    const y = 26 - (prob / max) * 22;
    segment.push(`${x},${y}`);
  });
  if (segment.length) segments.push(segment);

  return (
    <div className="border-t border-ink-800 px-3 py-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-[11px] font-medium text-slate-300">Sampling history</h3>
        {available.length > 1 && (
          <svg viewBox="0 0 100 30" role="img" aria-label="Chosen-token probability across steps" className="h-7 w-24">
            {segments.map((points, index) => (
              <polyline
                key={index}
                points={points.join(' ')}
                fill="none"
                stroke="#38bdf8"
                strokeWidth="2"
              />
            ))}
          </svg>
        )}
      </div>
      <div className="grid grid-cols-[2.5rem_minmax(3rem,1fr)_2.5rem_3.5rem_3.5rem_auto] gap-2 px-1.5 pb-1 text-[9px] uppercase tracking-wide text-slate-600">
        <span>Step</span>
        <span>Token</span>
        <span>Rank</span>
        <span className="text-right">p_final</span>
        <span className="text-right">Entropy</span>
        <span className="text-right">Mode</span>
      </div>
      <ol className="max-h-48 space-y-1 overflow-y-auto">
        {steps.map((step, index) => {
          const sampling = step.trace?.sampling;
          const selected = index === currentStep;
          return (
            <li key={`${index}-${step.token_id}`}>
              <button
                type="button"
                onClick={() => onSelectStep(index)}
                className={[
                  'grid w-full grid-cols-[2.5rem_minmax(3rem,1fr)_2.5rem_3.5rem_3.5rem_auto] items-center gap-2 rounded px-1.5 py-1 text-left text-[10px]',
                  selected ? 'bg-accent-500/10 text-accent-300' : 'text-slate-400 hover:bg-ink-800',
                ].join(' ')}
                title={`Select generation step ${index + 1}`}
              >
                <span className="value text-slate-500">{index + 1}</span>
                <span className="truncate" title={step.piece}>
                  {JSON.stringify(step.piece)}
                </span>
                <span className="value">{sampling ? `#${sampling.chosen.rank}` : '—'}</span>
                <span className="value text-right">
                  {sampling ? `${(sampling.chosen.p_final * 100).toFixed(2)}%` : '—'}
                </span>
                <span className="value text-right">
                  {sampling ? sampling.entropy_final.toFixed(3) : '—'}
                </span>
                <span className="text-right text-amber-300">
                  {sampling?.was_argmax ? 'argmax' : ''}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {!available.length && (
        <p className="mt-1 text-[10px] text-slate-600">
          Per-step sampling details were not included in these traces.
        </p>
      )}
    </div>
  );
}
