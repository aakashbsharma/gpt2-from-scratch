/**
 * Switches the app between live generation and replay.
 *
 * The point of the toggle is that it changes *nothing* downstream: both modes are
 * driven through the same store and the same `TraceSource` interface, so the
 * pipeline, heatmap, arcs, lens and sampling chart all render identically. This
 * component therefore only dispatches `setMode` and re-runs the load for the new
 * mode; it does not remount or reconfigure any visualization.
 */

import { motion } from 'framer-motion';

import { STATIC_ONLY } from '@/lib/env';
import { formatTemperature, formatTopK } from '@/lib/format';
import { useTraceStore, type Mode } from '@/store/useTraceStore';

export function ModeToggle() {
  const mode = useTraceStore((s) => s.mode);
  const setMode = useTraceStore((s) => s.setMode);
  const replayName = useTraceStore((s) => s.replayName);
  const replayMeta = useTraceStore((s) => s.replayMeta);
  const modelId = useTraceStore((s) => s.modelId);
  const status = useTraceStore((s) => s.status);
  const steps = useTraceStore((s) => s.steps);

  // In a static bundle there is no backend to talk to, so Live is not offered
  // rather than offered-and-broken. `STATIC_ONLY` is compile-time constant, so
  // the Live entry is absent from the shipped array.
  const modes: { id: Mode; label: string; hint: string }[] = STATIC_ONLY
    ? [{ id: 'replay', label: 'Replay', hint: 'play an exported trace from public/traces' }]
    : [
        { id: 'replay', label: 'Replay', hint: 'play an exported trace from public/traces' },
        { id: 'live', label: 'Live', hint: 'generate over the WebSocket against the backend' },
      ];

  return (
    <div className="flex items-center gap-3">
      <div
        role="tablist"
        aria-label="Data source"
        className="relative flex rounded-md border border-ink-600 bg-ink-850 p-0.5"
      >
        {modes.map((m) => {
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              role="tab"
              type="button"
              aria-selected={active}
              title={m.hint}
              onClick={() => setMode(m.id)}
              className={[
                'relative z-10 rounded px-3 py-1 text-xs font-medium transition-colors',
                'focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60',
                active ? 'text-accent-400' : 'text-slate-400 hover:text-slate-200',
              ].join(' ')}
            >
              {active && (
                <motion.span
                  layoutId="mode-toggle-pill"
                  className="absolute inset-0 -z-10 rounded bg-accent-500/15 ring-1 ring-accent-500/50"
                  transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                />
              )}
              {m.label}
            </button>
          );
        })}
      </div>

      <RunSummary
        mode={mode}
        modelId={modelId}
        status={status}
        stepCount={steps.length}
        replayName={replayName}
        replayMeta={replayMeta}
      />
    </div>
  );
}

function RunSummary({
  mode,
  modelId,
  status,
  stepCount,
  replayName,
  replayMeta,
}: {
  mode: Mode;
  modelId: string;
  status: string;
  stepCount: number;
  replayName: string | null;
  replayMeta: { name: string; params: { temperature: number; top_k: number | null } | null } | null;
}) {
  const statusColor =
    status === 'error'
      ? 'text-rose-400'
      : status === 'running'
        ? 'text-accent-400'
        : status === 'done'
          ? 'text-emerald-400'
          : 'text-slate-500';

  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[11px]">
      <span className={`value ${statusColor}`}>{status}</span>
      {modelId && <span className="value text-slate-400">{modelId}</span>}
      {mode === 'replay' ? (
        <>
          {replayName && <span className="value text-slate-500">traces/{replayName}.json</span>}
          {replayMeta?.params && (
            <span className="value text-slate-600">
              recorded: temp {formatTemperature(replayMeta.params.temperature)} · k{' '}
              {formatTopK(replayMeta.params.top_k)}
            </span>
          )}
        </>
      ) : (
        stepCount > 0 && <span className="value text-slate-500">{stepCount} steps</span>
      )}
    </div>
  );
}
