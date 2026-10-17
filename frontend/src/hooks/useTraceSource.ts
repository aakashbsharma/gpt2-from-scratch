/**
 * Shared plumbing for the two trace sources.
 *
 * Both `useLiveGeneration` and `useReplay` return a `TraceSource` and both
 * write into the store through the identical action pair, so this hook owns
 * exactly the parts they must not diverge on: which store fields constitute a
 * source, status transitions, rate measurement and reset-on-new-run.
 */

import { useCallback, useState } from 'react';

import { useTraceStore } from '@/store/useTraceStore';
import type { TraceSource } from '@/types/trace';

/** Rolling window for the steps-per-second readout. */
const RATE_WINDOW_MS = 2000;

/** A source plus the handle a source uses to record an arriving step. */
export interface TraceSourceHandle extends TraceSource {
  /** Record a step arrival. Called once per received step by each source. */
  recordStep: () => void;
  /** Forget arrival timestamps without touching the accumulated steps. */
  clearRate: () => void;
  /** Drop accumulated state back to `idle`. */
  reset: () => void;
}

/**
 * Subscribe to the slice of the store that makes up a `TraceSource`.
 *
 * Fields are selected individually rather than subscribing to the whole store,
 * so a `status` change does not re-render consumers that only read selection
 * state.
 */
export function useTraceSource(): TraceSourceHandle {
  const status = useTraceStore((s) => s.status);
  const error = useTraceStore((s) => s.error);
  const steps = useTraceStore((s) => s.steps);
  const text = useTraceStore((s) => s.text);
  const modelId = useTraceStore((s) => s.modelId);
  const prompt = useTraceStore((s) => s.prompt);
  const setStatus = useTraceStore((s) => s.setStatus);
  const resetStore = useTraceStore((s) => s.reset);

  // Timestamps of recent arrivals. Kept in a ref and mirrored into state, since
  // the rate is a readout rather than a value anything branches on.
  const [times, setTimes] = useState<number[]>([]);

  const recordStep = useCallback(() => {
    const now = performance.now();
    setTimes((prev) => {
      const cutoff = now - RATE_WINDOW_MS;
      const kept = prev.filter((t) => t >= cutoff);
      kept.push(now);
      return kept;
    });
  }, []);

  const clearRate = useCallback(() => setTimes([]), []);

  const reset = useCallback(() => {
    setTimes([]);
    resetStore();
  }, [resetStore]);

  return {
    status,
    steps,
    text,
    modelId,
    prompt,
    error,
    finished: steps.length > 0 && steps[steps.length - 1]?.finished === true,
    stepsPerSecond: rateOf(times),
    recordStep,
    clearRate,
    stop: () => setStatus('done'),
    reset,
  };
}

/**
 * Steps per second over the trailing window.
 *
 * `length` arrivals spanning `span` ms is `length - 1` intervals, so a run that
 * emits a single step reports `null` rather than a fabricated rate.
 */
function rateOf(times: number[]): number | null {
  if (times.length < 2) return null;
  const span = times[times.length - 1] - times[0];
  if (span <= 0) return null;
  return ((times.length - 1) / span) * 1000;
}
