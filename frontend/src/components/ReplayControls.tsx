/**
 * Replay transport: which bundle, play/pause, and step through it.
 *
 * Exists only in replay mode, but reads the same store the pipeline does, so
 * stepping here moves every other panel. It is the manual counterpart to the
 * automatic playback the hook runs on a timer.
 */

import { formatCompact } from '@/lib/format';
import { useTraceStore } from '@/store/useTraceStore';
import type { ReplayTraceName } from '@/types/trace';

import type { Replay } from '@/hooks/useReplay';

export function ReplayControls({ replay }: { replay: Replay }) {
  const currentName = useTraceStore((s) => s.replayName);
  const status = useTraceStore((s) => s.status);
  const error = useTraceStore((s) => s.error);
  const revealed = useTraceStore((s) => s.steps.length);
  const prompt = useTraceStore((s) => s.prompt);

  const total = replay.totalSteps;
  const atEnd = total > 0 && revealed >= total;

  return (
    <section className="panel" aria-label="Replay controls">
      <div className="flex flex-wrap items-center gap-2 px-3 py-2.5">
        <span className="label">trace</span>
        <div className="flex flex-wrap gap-1">
          {replay.available.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => void replay.load(name)}
              aria-pressed={currentName === name}
              className={['btn px-2 py-0.5 text-[11px]', currentName === name ? 'btn-active' : ''].join(' ')}
            >
              {name}
            </button>
          ))}
        </div>

        <span className="mx-1 h-4 w-px bg-ink-600" aria-hidden />

        <button
          type="button"
          className="btn"
          onClick={atEnd ? () => void replay.load(currentName as ReplayTraceName) : replay.stepForward}
          disabled={status === 'loading' || atEnd}
          title="reveal the next recorded step"
        >
          step ▸
        </button>
        <button type="button" className="btn" onClick={replay.stepBack} disabled={revealed === 0}>
          ◂ back
        </button>
        <button type="button" className="btn" onClick={replay.playAll} disabled={total === 0}>
          jump to end
        </button>

        <button
          type="button"
          className={['btn', replay.playing ? 'btn-active' : ''].join(' ')}
          onClick={() => replay.setPlaying(!replay.playing)}
          disabled={atEnd}
        >
          {replay.playing ? '❙❙ pause' : '▶ play'}
        </button>

        <label className="flex items-center gap-1.5">
          <span className="label">speed</span>
          <input
            type="range"
            min={100}
            max={2000}
            step={100}
            value={replay.intervalMs}
            onChange={(e) => replay.setIntervalMs(Number(e.target.value))}
            className="w-24 accent-accent-500"
            aria-label="playback interval in milliseconds"
          />
          <span className="value w-14 text-[10px] text-slate-500">
            {formatCompact(1000 / Math.max(1, replay.intervalMs), 1)}/s
          </span>
        </label>

        <span className="value ml-auto text-[11px] text-slate-500">
          {revealed} / {total || '?'} steps
        </span>
      </div>

      {status === 'error' && error && (
        <p className="border-t border-ink-800 px-3 py-1.5 text-[11px] text-rose-400">{error}</p>
      )}

      {prompt && (
        <p className="border-t border-ink-800 px-3 py-1.5 text-[11px] text-slate-500">
          recorded prompt · <span className="text-slate-300">{JSON.stringify(prompt)}</span>
        </p>
      )}
    </section>
  );
}

/** Live-mode status strip: model readiness and what the run is doing. */
export function LiveStatus({
  connected,
  stepsPerSecond,
  status,
}: {
  connected: boolean;
  stepsPerSecond: number | null;
  status: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
      <span
        className={[
          'inline-flex items-center gap-1.5',
          connected ? 'text-emerald-400' : 'text-slate-500',
        ].join(' ')}
      >
        <span
          className={[
            'h-1.5 w-1.5 rounded-full',
            connected ? 'bg-emerald-400' : 'bg-ink-500',
          ].join(' ')}
          aria-hidden
        />
        {connected ? 'socket open' : 'socket closed'}
      </span>
      <span className="value text-slate-500">{status}</span>
      {stepsPerSecond !== null && (
        <span className="value text-slate-500">{formatCompact(stepsPerSecond, 1)} steps/s</span>
      )}
    </div>
  );
}
