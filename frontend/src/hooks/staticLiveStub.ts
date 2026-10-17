/**
 * Stand-in for the live source in a `build:static` bundle.
 *
 * Aliased in place of `./liveSocket` by `vite.config.ts`, so the real socket code
 * — `new WebSocket`, the message protocol, the reconnect-free lifecycle — is
 * absent from the static output rather than present-but-unreachable.
 *
 * Every method is safe to call and reports why nothing happened. This is a
 * deliberate choice over throwing: `App.tsx` constructs this hook unconditionally
 * (hooks cannot be called conditionally), so a static build does call these
 * functions. They must return a coherent `TraceSource` rather than raise, or the
 * app would crash on mount in the one mode that is supposed to be the simplest.
 *
 * `connected` is always false, so any UI that shows socket status reads
 * "closed", which is accurate.
 */

import { useCallback } from 'react';

import { useTraceStore } from '@/store/useTraceStore';
import type { GenerateOptions, TraceSource } from '@/types/trace';

import { useTraceSource } from './useTraceSource';

export interface LiveGeneration extends TraceSource {
  generate: (opts: GenerateOptions) => Promise<boolean>;
  connected: boolean;
}

/** Shown if anything ever surfaces this instead of hiding the control. */
const WHY = 'live generation needs the backend; this is a static replay build';

export function useLiveGeneration(): LiveGeneration {
  const base = useTraceSource();

  const generate = useCallback(async (): Promise<boolean> => {
    useTraceStore.getState().setStatus('error', WHY);
    return false;
  }, []);

  return { ...base, generate, connected: false };
}
