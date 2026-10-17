/**
 * Drives the two concurrent generations behind `ComparePanel`.
 *
 * Kept out of `useTraceStore` deliberately: that store holds one current step by
 * design, while comparison needs two traces simultaneously. This hook owns its
 * own pair of sockets and its own result state, and returns the small control
 * surface `ComparePanel` needs.
 *
 * Both sides are greedy by default. Sampling would make the comparison
 * irreproducible — two runs of the same prompt could legitimately differ for
 * reasons unrelated to the weights, which is precisely the confound this panel
 * exists to avoid.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { buildGenerateRequest, wsGenerateUrl } from '@/lib/api';
import type { ModelInfo, Trace, WSResponse } from '@/types/trace';

export interface CompareSideResult {
  modelId: string;
  label: string;
  text: string;
  /** Trace of the first step, which is the one over the bare prompt. */
  trace: Trace | null;
  error: string | null;
  pending: boolean;
}

export interface CompareRun {
  prompt: string;
  setPrompt: (v: string) => void;
  modelA: string;
  setModelA: (v: string) => void;
  modelB: string;
  setModelB: (v: string) => void;
  start: () => Promise<void>;
  cancel: () => void;
  busy: boolean;
  error: string | null;
  resultA: CompareSideResult | null;
  resultB: CompareSideResult | null;
  /** Selectable layer indices, from whichever traces have arrived. */
  layers: number[];
  layer: number;
  setLayer: (l: number) => void;
  head: number;
  setHead: (h: number) => void;
  headCount: number;
}

/** Tokens generated per side. Small: the comparison is about the first step. */
const COMPARE_MAX_NEW_TOKENS = 6;

/**
 * How long a socket may be open with no frame at all before it is treated as
 * failed. Generous enough for a cold first traced pass on CPU, short enough that
 * a silent server surfaces as an error rather than an indefinite wait.
 */
const COMPARE_SILENCE_TIMEOUT_MS = 30_000;

/**
 * Drives the two concurrent generations behind `ComparePanel`.
 *
 * `ready` is passed in rather than fetched here: `ComparePanel` already holds the
 * registry, and calling `useModels` in both would subscribe twice to the same
 * thing. The registry request is shared and cached now, so this is about not
 * duplicating state rather than about saving a request.
 */
export function useCompareRun(ready: ModelInfo[]): CompareRun {
  const [prompt, setPrompt] = useState('Every effort moves you');
  const [modelA, setModelA] = useState('');
  const [modelB, setModelB] = useState('');
  const [resultA, setResultA] = useState<CompareSideResult | null>(null);
  const [resultB, setResultB] = useState<CompareSideResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);

  const sockets = useRef<WebSocket[]>([]);
  const seq = useRef(0);

  // Default both selects to the first two ready models as the registry arrives.
  // `ready` is memoized by `useModels`, so this effect runs when the registry
  // actually changes rather than on every render.
  useEffect(() => {
    if (ready.length === 0) return;
    setModelA((cur) => (ready.some((m) => m.id === cur) ? cur : ready[0].id));
    setModelB((cur) =>
      ready.some((m) => m.id === cur) && cur !== modelA ? cur : (ready[1]?.id ?? ready[0].id),
    );
  }, [ready, modelA]);

  const cancel = useCallback(() => {
    seq.current += 1;
    for (const ws of sockets.current) {
      // Detach `onmessage` and `onerror` so a late frame from an abandoned run
      // cannot touch state, but KEEP `onclose`. `runSide` settles from `onclose`,
      // so nulling it first meant a cancelled socket's promise never resolved —
      // a permanently pending `Promise.all` and a leaked closure holding the
      // trace. The close still fires `onclose`, so the promise settles normally.
      ws.onmessage = null;
      ws.onerror = null;
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close(1000, 'cancelled');
      }
    }
    sockets.current = [];
    setBusy(false);
  }, []);

  const runSide = useCallback(
    (modelId: string, label: string, requestId: string, ticket: number) =>
      new Promise<CompareSideResult>((resolve) => {
        // Settle exactly once, from whichever path fires first.
        //
        // `resolve` is idempotent, but the *callers* are not: `start()` awaits
        // `Promise.all([runSide(A), runSide(B)])`, so a socket that neither errors
        // nor closes leaves that await pending forever. `busy` stays true, the
        // button stays disabled, and two sockets sit open on the server with
        // nothing driving them — which looks exactly like "the WebSocket hangs".
        let settled = false;
        const settle = (result: CompareSideResult) => {
          if (settled) return;
          settled = true;
          clearWatchdog();
          resolve(result);
        };
        let watchdog: number | null = null;
        const clearWatchdog = () => {
          if (watchdog !== null) {
            window.clearTimeout(watchdog);
            watchdog = null;
          }
        };

        let trace: Trace | null = null;
        let text = '';
        let ws: WebSocket;
        try {
          ws = new WebSocket(wsGenerateUrl());
        } catch (err) {
          settle({
            modelId,
            label,
            text: '',
            trace: null,
            error: err instanceof Error ? err.message : 'could not open websocket',
            pending: false,
          });
          return;
        }
        sockets.current.push(ws);

        ws.onopen = () => {
          // Watchdog: if the socket opens and the server says nothing at all, that
          // is a failure, not a wait. The server answers in well under a second
          // for a traced step on CPU, so a long silence means it never got the
          // request — which is the one failure mode that otherwise looks like an
          // indefinite hang.
          watchdog = window.setTimeout(() => {
            settle({
              modelId,
              label,
              text,
              trace,
              error: 'no response from the server after opening the socket',
              pending: false,
            });
          }, COMPARE_SILENCE_TIMEOUT_MS);

          try {
            ws.send(
              JSON.stringify(
                buildGenerateRequest(requestId, {
                  model_id: modelId,
                  prompt,
                  max_new_tokens: COMPARE_MAX_NEW_TOKENS,
                  // Greedy: a sampled comparison would confound the weights with
                  // the seed, which defeats the purpose.
                  temperature: 0,
                  top_k: null,
                  seed: null,
                  trace: true,
                }),
              ),
            );
          } catch (err) {
            // A throw from `send` leaves the socket open with the request never
            // delivered, and the server waiting on `receive_json()` forever. Both
            // sides report nothing, so catch it and settle with a reason.
            settle({
              modelId,
              label,
              text,
              trace,
              error: `could not send the request: ${err instanceof Error ? err.message : String(err)}`,
              pending: false,
            });
          }
        };

        ws.onmessage = (event) => {
          if (ticket !== seq.current) return;
          // Any frame proves the server is talking; the watchdog is done.
          clearWatchdog();
          let msg: WSResponse;
          try {
            msg = JSON.parse(event.data as string) as WSResponse;
          } catch {
            return;
          }
          if (msg.type === 'step') {
            // Keep the *first* trace: it is the one over the bare prompt, which
            // is what both sides must be compared on.
            if (trace === null && msg.trace) trace = msg.trace;
            text = msg.text;
          } else if (msg.type === 'error') {
            settle({ modelId, label, text, trace, error: msg.message, pending: false });
          } else if (msg.type === 'done') {
            settle({ modelId, label, text, trace, error: null, pending: false });
          }
        };

        ws.onerror = () => {
          settle({ modelId, label, text, trace, error: 'websocket error', pending: false });
        };

        ws.onclose = () => {
          settle({ modelId, label, text, trace, error: null, pending: false });
        };
      }),
    [prompt],
  );

  const start = useCallback(async () => {
    if (!modelA || !modelB || modelA === modelB) {
      setError('pick two different models');
      return;
    }
    // One counter guards both cancel and restart: bumping it invalidates every
    // in-flight `runSide`, whose socket handlers compare their ticket against it.
    cancel();
    const ticket = ++seq.current;

    setBusy(true);
    setError(null);
    setResultA(null);
    setResultB(null);
    setLayer(0);
    setHead(0);

    const labelOf = (id: string) => ready.find((m) => m.id === id)?.label ?? id;

    // Both sockets open in parallel: the backend serializes inference behind a
    // lock anyway, so running them concurrently costs no extra wall time.
    const [a, b] = await Promise.all([
      runSide(modelA, labelOf(modelA), `${ticket}-a`, ticket),
      runSide(modelB, labelOf(modelB), `${ticket}-b`, ticket),
    ]);

    // A newer run started (or the user cancelled) while these were in flight.
    if (ticket !== seq.current) return;
    setResultA(a);
    setResultB(b);
    setBusy(false);
    sockets.current = [];

    const trace = a.trace ?? b.trace;
    if (trace) {
      // Compare at a depth that is actually informative: the last block is
      // where both models are most likely to agree trivially.
      setLayer(Math.max(0, trace.blocks.length - 2));
    }
    if (a.error && b.error) setError('both models failed — is the backend running?');
  }, [cancel, modelA, modelB, ready, runSide]);

  useEffect(
    () => () => {
      seq.current += 1;
      for (const ws of sockets.current) ws.close(1000, 'unmounted');
      sockets.current = [];
    },
    [],
  );

  const layers = deriveLayers(resultA?.trace ?? null, resultB?.trace ?? null);
  const headCount = resultA?.trace?.blocks[layer]?.heads.length ?? resultB?.trace?.blocks[layer]?.heads.length ?? 0;

  return {
    prompt,
    setPrompt,
    modelA,
    setModelA,
    modelB,
    setModelB,
    start,
    cancel,
    busy,
    error,
    resultA,
    resultB,
    layers,
    layer: Math.min(layer, Math.max(0, layers.length - 1)),
    setLayer,
    head: Math.min(head, Math.max(0, headCount - 1)),
    setHead,
    headCount,
  };
}

/**
 * Layer indices both sides can show.
 *
 * Truncated to the shallower model: two checkpoints with different depths have
 * no shared layers past the smaller `n_layers`, and showing a one-sided layer
 * would silently compare nothing.
 */
function deriveLayers(a: Trace | null, b: Trace | null): number[] {
  const n = Math.min(a?.blocks.length ?? 0, b?.blocks.length ?? 0);
  return Array.from({ length: n }, (_, i) => i);
}
