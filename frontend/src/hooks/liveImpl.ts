/**
 * The real live-generation source: one WebSocket per run.
 *
 * Implements `TraceSource`, the same interface `useReplay` implements, so the UI
 * is mode-agnostic: switching `ModeToggle` swaps which source is mounted and
 * nothing downstream changes.
 *
 * Protocol notes that shape the code:
 * - The server sends one `WSStepMessage` per generated token, then a
 *   `WSDoneMessage`, then closes.
 * - `WSStepMessage.trace` is `null` when the request set `trace: false`. The UI
 *   still animates in that case, it just has no per-step tensors.
 * - An error frame arrives before the socket closes with a non-1000 code, so both
 *   paths are handled.
 *
 * This module is aliased out of the static build (see `vite.config.ts`), so
 * nothing here reaches a `build:static` bundle. `@/hooks/useLiveGeneration` is the
 * stable import path callers use; it re-exports either this or the stub.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { buildGenerateRequest, wsGenerateUrl } from '@/lib/api';
import { useTraceStore } from '@/store/useTraceStore';
import type { GenerateOptions, TraceSource, WSResponse } from '@/types/trace';

import { useTraceSource } from './useTraceSource';

let requestCounter = 0;

/**
 * How long a socket may be open with no frame before the run is declared failed.
 * See the watchdog in `generate`. Deliberately long: it exists to turn an
 * indefinite silent hang into a legible error, not to second-guess a slow model.
 */
const SILENCE_TIMEOUT_MS = 30_000;

export interface LiveGeneration extends TraceSource {
  /** Open the socket and start generating. Resolves false if the socket never opened. */
  generate: (opts: GenerateOptions) => Promise<boolean>;
  /** True while a socket is open. */
  connected: boolean;
}

export function useLiveGeneration(): LiveGeneration {
  const base = useTraceSource();
  const socket = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  const { recordStep, clearRate, reset: resetBase, ...source } = base;
  const startRun = useTraceStore((s) => s.startRun);
  const pushStep = useTraceStore((s) => s.pushStep);
  const setStatus = useTraceStore((s) => s.setStatus);

  const closeSocket = useCallback(() => {
    const ws = socket.current;
    socket.current = null;
    setConnected(false);
    if (!ws) return;
    // Detach handlers first: closing a live socket can still deliver a frame,
    // and a stale handler would push a step into a run we already abandoned.
    ws.onmessage = null;
    ws.onerror = null;
    ws.onclose = null;
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close(1000, 'client closed');
    }
  }, [setConnected]);

  const generate = useCallback(
    async (opts: GenerateOptions): Promise<boolean> => {
      closeSocket();
      // A new run invalidates the previous rate, so the readout does not mix runs.
      clearRate();

      const requestId = `live-${Date.now().toString(36)}-${++requestCounter}`;
      startRun({
        modelId: opts.model_id,
        prompt: opts.prompt,
        maxNewTokens: opts.max_new_tokens,
      });

      const request = buildGenerateRequest(requestId, {
        model_id: opts.model_id,
        prompt: opts.prompt,
        max_new_tokens: opts.max_new_tokens,
        temperature: opts.temperature,
        top_k: opts.top_k,
        seed: opts.seed,
        trace: opts.trace !== false,
      });

      return new Promise<boolean>((resolve) => {
        let ws: WebSocket;
        try {
          ws = new WebSocket(wsGenerateUrl());
        } catch (err) {
          setStatus('error', err instanceof Error ? err.message : 'could not open websocket');
          resolve(false);
          return;
        }
        socket.current = ws;
        let opened = false;
        let settled = false;

        const finish = (value: boolean) => {
          if (settled) return;
          settled = true;
          if (watchdog !== null) {
            window.clearTimeout(watchdog);
            watchdog = null;
          }
          resolve(value);
        };

        /**
         * Silence watchdog.
         *
         * If the socket opens and the server then says nothing, that is a
         * failure, not a wait. From the browser it is indistinguishable from
         * "generating", and on the server it looks like a socket that is simply
         * open forever — the one symptom with no error anywhere to explain it.
         *
         * Cleared by the first frame, so it only ever guards the handshake. 30 s is
         * far longer than a cold traced forward pass on CPU needs (~0.2 s measured
         * for 124M at T=4, and up to a few seconds on a busy machine).
         */
        let watchdog: number | null = null;

        ws.onopen = () => {
          opened = true;
          setConnected(true);
          setStatus('running');
          watchdog = window.setTimeout(() => {
            setStatus(
              'error',
              `the socket opened but the server sent nothing for ${SILENCE_TIMEOUT_MS / 1000}s ` +
                `— the request may not have reached ${wsGenerateUrl()}`,
            );
            finish(false);
          }, SILENCE_TIMEOUT_MS);
          // Wrapped because a throw here is invisible: the socket stays open, the
          // server never receives the request and waits on `receive_json()`
          // forever, and neither side reports an error. That is precisely the
          // "connection open, nothing happening" signature.
          try {
            ws.send(JSON.stringify(request));
          } catch (err) {
            setStatus('error', `could not send the request: ${err instanceof Error ? err.message : String(err)}`);
            closeSocket();
            finish(false);
          }
        };

        ws.onmessage = (event) => {
          // Any frame proves the server is talking; the watchdog is done.
          if (watchdog !== null) {
            window.clearTimeout(watchdog);
            watchdog = null;
          }
          let msg: WSResponse;
          try {
            msg = JSON.parse(event.data as string) as WSResponse;
          } catch {
            setStatus('error', 'received a malformed frame from the server');
            return;
          }
          if (msg.request_id !== requestId) return;

          switch (msg.type) {
            case 'step':
              recordStep();
              pushStep(msg);
              break;
            case 'done':
              setStatus('done');
              closeSocket();
              finish(true);
              break;
            case 'error':
              setStatus('error', msg.message);
              closeSocket();
              finish(true);
              break;
          }
        };

        ws.onerror = () => {
          // `onerror` carries no detail by design; the close code does.
          setStatus('error', 'websocket error — is the backend running on :8000?');
        };

        ws.onclose = (event) => {
          setConnected(false);
          if (socket.current === ws) socket.current = null;
          // 1000 after `done` is the normal path and already handled. Anything
          // else means the stream was cut short.
          if (!opened) {
            setStatus('error', `could not connect to ${wsGenerateUrl()}`);
          } else if (event.code !== 1000) {
            setStatus('error', `stream closed with code ${event.code}`);
          } else {
            setStatus('done');
          }
          finish(false);
        };
      });
    },
    [clearRate, closeSocket, pushStep, recordStep, setConnected, setStatus, startRun],
  );

  // Abort any in-flight generation when the hook unmounts, so a mode switch
  // mid-generation does not leave a socket pushing into an abandoned store.
  useEffect(() => () => closeSocket(), [closeSocket]);

  const stop = useCallback(() => {
    closeSocket();
    setStatus('done');
  }, [closeSocket, setStatus]);

  const reset = useCallback(() => {
    closeSocket();
    resetBase();
  }, [closeSocket, resetBase]);

  return { ...source, generate, connected, stop, reset };
}
