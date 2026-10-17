/**
 * Replay of exported traces from `public/traces/*.json`.
 *
 * Implements `TraceSource`, the same interface `useLiveGeneration` implements.
 * The exported files hold the identical `WSStepMessage` payloads the live socket
 * sends, so replay pushes them into the store one at a time and the pipeline
 * animates exactly as it does live — no component can tell the difference.
 *
 * The one honest difference: replay is a recording, so it cannot honour
 * `max_new_tokens`, `temperature`, `top_k` or `seed` from a request. Those live
 * in the bundle's `params` and are surfaced read-only.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { publicPath } from '@/lib/env';
import { useTraceStore } from '@/store/useTraceStore';
import {
  REPLAY_TRACE_NAMES,
  type ReplayTraceName,
  type TraceBundle,
  type TraceSource,
} from '@/types/trace';

import { useTraceSource } from './useTraceSource';

export interface Replay extends TraceSource {
  /** Names of the bundles available under `/traces`. */
  available: readonly ReplayTraceName[];
  /** Load a bundle and start playing it back. Resolves false if it 404s. */
  load: (name: ReplayTraceName, options?: { autoplay?: boolean }) => Promise<boolean>;
  /** Reveal the next step. No-op once every step has been revealed. */
  stepForward: () => void;
  /** Reveal every remaining step at once. */
  playAll: () => void;
  /** Hide the most recently revealed step. */
  stepBack: () => void;
  /** True while a playback timer is pending. */
  playing: boolean;
  setPlaying: (playing: boolean) => void;
  /** Milliseconds between revealed steps. */
  intervalMs: number;
  setIntervalMs: (ms: number) => void;
  /** Steps in the loaded bundle, for the progress readout. */
  totalSteps: number;
}

const DEFAULT_INTERVAL_MS = 700;

export function useReplay(): Replay {
  const base = useTraceSource();

  const [playing, setPlaying] = useState(false);
  const [intervalMs, setIntervalMsState] = useState(DEFAULT_INTERVAL_MS);
  const [totalSteps, setTotalSteps] = useState(0);

  const timer = useRef<number | null>(null);
  /** Held in a ref, not state: it is payload-sized and nothing renders from it. */
  const bundle = useRef<TraceBundle | null>(null);

  const startRun = useTraceStore((s) => s.startRun);
  const pushStep = useTraceStore((s) => s.pushStep);
  const setSteps = useTraceStore((s) => s.setSteps);
  const setStatus = useTraceStore((s) => s.setStatus);
  const revealed = useTraceStore((s) => s.steps.length);
  /** Bumped on every mode switch and new run; see the timer guard below. */
  const runId = useTraceStore((s) => s.runId);

  const { recordStep, clearRate, reset: resetBase, ...source } = base;

  const clearTimer = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    clearTimer();
    setPlaying(false);
    setStatus('done');
  }, [clearTimer, setPlaying, setStatus]);

  const reset = useCallback(() => {
    clearTimer();
    setPlaying(false);
    setTotalSteps(0);
    bundle.current = null;
    resetBase();
  }, [clearTimer, resetBase]);

  /** Reveal the next recorded step, if there is one. */
  const stepForward = useCallback(() => {
    const steps = bundle.current?.steps;
    if (!steps || revealed >= steps.length) {
      clearTimer();
      setPlaying(false);
      setStatus('done');
      return;
    }
    const next = steps[revealed];
    recordStep();
    pushStep(next);
    const isLast = revealed + 1 >= steps.length;
    if (next.finished || isLast) {
      clearTimer();
      setPlaying(false);
      setStatus('done');
    }
  }, [clearTimer, pushStep, recordStep, revealed, setPlaying, setStatus]);

  const playAll = useCallback(() => {
    const steps = bundle.current?.steps;
    if (!steps) return;
    clearTimer();
    setPlaying(false);
    setSteps(steps);
    setStatus('done');
  }, [clearTimer, setPlaying, setStatus, setSteps]);

  const stepBack = useCallback(() => {
    const steps = bundle.current?.steps;
    if (!steps) return;
    clearTimer();
    setPlaying(false);
    if (revealed <= 0) return;
    setSteps(steps.slice(0, revealed - 1));
    setStatus('loading');
  }, [clearTimer, revealed, setPlaying, setStatus, setSteps]);

  // Switching away from replay must stop playback. `setMode` bumps `runId`, which
  // makes the timer effect below ignore the stale timer, but `playing` would stay
  // true — so the moment the user switched back, playback would resume against a
  // different runId with no bundle loaded.
  useEffect(
    () =>
      useTraceStore.subscribe((state, prev) => {
        if (state.runId !== prev.runId && state.mode !== 'replay') {
          clearTimer();
          setPlaying(false);
        }
      }),
    [clearTimer, setPlaying],
  );

  const load = useCallback(
    async (name: ReplayTraceName, options?: { autoplay?: boolean }): Promise<boolean> => {
      clearTimer();
      setPlaying(false);
      clearRate();

      setStatus('loading');
      try {
        // `publicPath` resolves against the deploy base, so a GitHub Pages
        // subpath (`/my-repo/`) loads from `/my-repo/traces/...`.
        const res = await fetch(publicPath(`traces/${name}.json`));
        if (!res.ok) throw new Error(`${res.status} ${res.statusText} for traces/${name}.json`);
        const doc = (await res.json()) as TraceBundle;
        bundle.current = doc;
        setTotalSteps(doc.steps.length);

        startRun({
          modelId: doc.model_id,
          prompt: doc.prompt,
          maxNewTokens: doc.params.max_new_tokens,
          replayName: doc.name,
          replayMeta: { name: doc.name, params: doc.params, intervalMs },
        });

        if (doc.steps.length === 0) {
          setStatus('done');
          return true;
        }

        if (options?.autoplay === false) {
          // Show the first step and hold, so the UI is populated but paused.
          recordStep();
          pushStep(doc.steps[0]);
          setStatus(doc.steps[0].finished ? 'done' : 'loading');
          return true;
        }

        setStatus('running');
        setPlaying(true);
        return true;
      } catch (err) {
        bundle.current = null;
        setTotalSteps(0);
        setStatus('error', err instanceof Error ? err.message : 'failed to load trace');
        return false;
      }
    },
    [clearRate, clearTimer, intervalMs, pushStep, recordStep, setPlaying, setStatus, startRun],
  );

  // Playback timer: an interval rather than a chained timeout, so the cadence
  // does not drift and `stop` has exactly one handle to clear.
  //
  // Guarded on `runId` as well as `playing`. This hook is mounted
  // unconditionally — hooks cannot be called conditionally — so switching to Live
  // mode does not unmount it, and without this guard its interval keeps firing
  // and pushes *recorded* steps into a live generation. That produces a socket
  // that looks alive while the UI shows a mix of two unrelated runs.
  useEffect(() => {
    clearTimer();
    if (!playing) return;
    if (revealed >= totalSteps) {
      setPlaying(false);
      return;
    }
    timer.current = window.setInterval(stepForward, Math.max(50, intervalMs));
    return clearTimer;
  }, [clearTimer, intervalMs, playing, revealed, runId, setPlaying, stepForward, totalSteps]);

  // The interval is also shown in the toolbar, so mirror it into the store.
  const setIntervalMs = useCallback(
    (ms: number) => {
      setIntervalMsState(ms);
      useTraceStore.setState((s) => ({
        replayMeta: s.replayMeta ? { ...s.replayMeta, intervalMs: ms } : s.replayMeta,
      }));
    },
    [setIntervalMsState],
  );

  useEffect(() => clearTimer, [clearTimer]);

  return {
    ...source,
    available: REPLAY_TRACE_NAMES,
    load,
    stepForward,
    playAll,
    stepBack,
    playing,
    setPlaying,
    intervalMs,
    setIntervalMs,
    totalSteps,
    stop,
    reset,
  };
}

// Note: `GenerateOptions` is deliberately not a parameter of `load`. Which
// prompt, temperature or seed the *recorded* run used is fixed in the bundle,
// so accepting and ignoring them would imply control that does not exist.
