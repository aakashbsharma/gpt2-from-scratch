import { formatPercent } from '@/lib/format';
import type { SamplingStep } from '@/types/trace';

const WIDTH = 1000;
const VISIBLE_SLICES = 64;

export function DrawStrip({ sampling }: { sampling: SamplingStep | null }) {
  if (!sampling) return null;

  const ordered = [...sampling.candidates].sort((left, right) => left.rank - right.rank);
  const visible = ordered.slice(0, VISIBLE_SLICES);
  const omittedMass = ordered
    .slice(VISIBLE_SLICES)
    .reduce((total, candidate) => total + candidate.p_final, 0);
  const chosenVisible = visible.some((candidate) => candidate.id === sampling.chosen.id);
  const chosenX = sampling.u === null ? null : sampling.u * WIDTH;
  let cumulative = 0;

  return (
    <div className="border-t border-ink-800 px-3 py-3">
      <div className="mb-1 flex items-center justify-between text-[10px] text-slate-500">
        <span>Draw from p_final</span>
        <span>{sampling.u === null ? 'argmax, no draw' : `u = ${sampling.u.toFixed(4)}`}</span>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} 32`}
        preserveAspectRatio="none"
        role="img"
        aria-label={
          sampling.u === null
            ? 'Greedy argmax selection with no random draw'
            : `Random draw ${sampling.u.toFixed(4)} through the final probability distribution`
        }
        className="h-8 w-full overflow-visible rounded"
      >
        <rect x="0" y="0" width={WIDTH} height="32" fill="#1e293b" />
        {visible.map((candidate) => {
          const x = cumulative * WIDTH;
          const width = candidate.p_final * WIDTH;
          cumulative += candidate.p_final;
          const chosen = candidate.id === sampling.chosen.id;
          return (
            <rect
              key={candidate.id}
              x={x}
              y="2"
              width={Math.max(width, 0)}
              height="28"
              fill={chosen ? '#38bdf8' : candidate.kept ? '#475569' : '#334155'}
              stroke={chosen ? '#7dd3fc' : 'none'}
              strokeWidth="2"
            >
              <title>
                {candidate.rank}. {JSON.stringify(candidate.token)}: {formatPercent(candidate.p_final, 3)}
              </title>
            </rect>
          );
        })}
        {omittedMass > 0 && (
          <rect
            x={cumulative * WIDTH}
            y="2"
            width={omittedMass * WIDTH}
            height="28"
            fill={chosenVisible ? '#334155' : '#38bdf8'}
          >
            <title>
              {chosenVisible ? 'Omitted candidate mass' : `Chosen token ${JSON.stringify(sampling.chosen.token)} is in omitted candidates`}
            </title>
          </rect>
        )}
        {chosenX !== null && (
          <line
            x1={chosenX}
            x2={chosenX}
            y1="0"
            y2="32"
            stroke="#f8fafc"
            strokeWidth="2"
            strokeDasharray="3 2"
          />
        )}
      </svg>
      <p className="mt-1 text-[11px] text-slate-400">
        {sampling.u === null ? (
          <span>argmax, no draw</span>
        ) : (
          <>
            u = <span className="value text-slate-200">{sampling.u.toFixed(4)}</span> landed in{' '}
            <span className="text-accent-300">{JSON.stringify(sampling.chosen.token)}</span>
          </>
        )}
        {sampling.cdf_lo !== null && sampling.cdf_hi !== null && (
          <span className="ml-2 value text-slate-500">
            [{sampling.cdf_lo.toFixed(4)}, {sampling.cdf_hi.toFixed(4)})
          </span>
        )}
      </p>
    </div>
  );
}
