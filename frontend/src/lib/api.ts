/**
 * Typed wrappers over the backend HTTP + WebSocket surface.
 *
 * Two build modes:
 * - dev: requests go to `/api/...` and are proxied by Vite (see vite.config.ts)
 * - build: `VITE_API_BASE` points at the deployed backend, and the WebSocket
 *   origin is derived from it so the two can never drift apart.
 */

import type {
  HealthResponse,
  ModelInfo,
  ModelListResponse,
  TokenizeRequest,
  TokenizeResponse,
  Trace,
  TraceRequest,
  WSGenerateRequest,
  WSResponse,
} from '@/types/trace';

import { STATIC_ONLY } from './env';

const RAW_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

export const API_BASE = RAW_BASE;

/**
 * `ws://` or `wss://` origin for the API, derived from `API_BASE`.
 *
 * Throws in a static bundle rather than returning a plausible-looking URL: a
 * caller that reaches this is live code that should have been compiled out, and
 * failing loudly at that point beats opening a socket that can never connect.
 */
export function wsOrigin(): string {
  if (STATIC_ONLY) {
    throw new Error('live generation is not available in a static build');
  }
  if (RAW_BASE) return RAW_BASE.replace(/^http/, 'ws');
  if (typeof window === 'undefined') return 'ws://localhost:8000';
  return `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`;
}

function url(path: string): string {
  return `${RAW_BASE}${path}`;
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    // FastAPI puts the useful part in `detail`; fall back to the status line.
    let detail = `${res.status} ${res.statusText}`;
    try {
      const parsed = (await res.json()) as { detail?: string };
      if (parsed.detail) detail = parsed.detail;
    } catch {
      /* body was not JSON; the status line is the best we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url(path), { signal });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
  return (await res.json()) as T;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson<HealthResponse>('/api/health', signal);
}

/**
 * The model registry, shared across every consumer.
 *
 * Three components need this list — `LiveControls` for its picker,
 * `ComparePanel` and `useCompareRun` via `useModels` — and they all mount on the
 * same page load. Without a cache that is three identical `GET /api/models` on
 * every visit, which is both wasteful and misleading in the network tab when
 * diagnosing whether something is polling.
 *
 * Two mechanisms, because they solve different problems:
 * - **in-flight dedupe**: concurrent callers await the same promise, so mounting
 *   three consumers at once issues exactly one request.
 * - **a short TTL**: `ready` flips when a checkpoint finishes loading, so a cache
 *   with no expiry would freeze the UI showing "not ready" after the backend came
 *   up. 15 s is short enough to recover without a reload and long enough that a
 *   session does not re-request on every navigation.
 *
 * No `AbortSignal`: the request is shared, so letting one consumer's unmount
 * cancel it would break the other two. Consumers that unmount simply drop the
 * result — see `useModels`, which uses a local `cancelled` flag.
 *
 * A failed fetch resolves to an empty list rather than throwing: the app must
 * still work in replay mode with no backend running at all, which is exactly
 * what a static deploy looks like. The empty result is cached too, so an offline
 * backend does not turn into a request storm.
 */
const MODELS_TTL_MS = 15_000;

let modelsCache: { at: number; list: ModelInfo[] } | null = null;
let modelsInFlight: Promise<ModelInfo[]> | null = null;

export function fetchModels(): Promise<ModelInfo[]> {
  // A static bundle never reaches this — callers are behind `STATIC_ONLY` guards —
  // but the early return keeps the guard local and makes the request impossible
  // to trigger by accident. Being compile-time constant, the rest of this
  // function is eliminated from a static build.
  if (STATIC_ONLY) return Promise.resolve([]);

  const now = Date.now();
  if (modelsCache && now - modelsCache.at < MODELS_TTL_MS) {
    return Promise.resolve(modelsCache.list);
  }

  if (!modelsInFlight) {
    modelsRequestCount += 1;
    modelsInFlight = getJson<ModelListResponse>('/api/models')
      .then((body) => body.models ?? [])
      .catch(() => [] as ModelInfo[])
      .then((list) => {
        modelsCache = { at: Date.now(), list };
        modelsInFlight = null;
        return list;
      });
  }

  return modelsInFlight;
}

/** Drop the cached registry, so the next `fetchModels` re-requests. */
export function invalidateModelsCache(): void {
  modelsCache = null;
}

/** Exposed for the smoke check, which asserts the dedupe actually dedupes. */
export function __modelsRequestCount(): number {
  return modelsRequestCount;
}

let modelsRequestCount = 0;

export function tokenize(req: TokenizeRequest, signal?: AbortSignal): Promise<TokenizeResponse> {
  return postJson<TokenizeResponse>('/api/tokenize', req, signal);
}

export function traceOnce(req: TraceRequest, signal?: AbortSignal): Promise<Trace> {
  return postJson<Trace>('/api/trace', req, signal);
}

export function wsGenerateUrl(): string {
  return `${wsOrigin()}/api/ws/generate`;
}

export function buildGenerateRequest(
  requestId: string,
  opts: {
    model_id: string;
    prompt: string;
    max_new_tokens: number;
    temperature: number;
    top_k: number | null;
    seed: number | null;
    trace: boolean;
  },
): WSGenerateRequest {
  return {
    type: 'generate',
    request_id: requestId,
    model_id: opts.model_id,
    prompt: opts.prompt,
    max_new_tokens: opts.max_new_tokens,
    temperature: opts.temperature,
    top_k: opts.top_k,
    seed: opts.seed,
    trace: opts.trace,
  };
}

export type { WSResponse };
