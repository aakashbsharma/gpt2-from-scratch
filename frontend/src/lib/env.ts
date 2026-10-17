/**
 * Build-time environment flags.
 *
 * `STATIC_ONLY` is the one that matters: it is set by the `build:static` script
 * and switches the app into a mode that needs no backend at all — no WebSocket,
 * no `/api` fetch, no model registry. Everything is served from the exported
 * bundles in `public/traces`.
 *
 * Why a flag rather than "just don't run the backend": a static deploy is a
 * different *program*, not the same program with a server missing. Live mode
 * would show a permanently red "could not connect to ws://localhost:8000" and
 * offer a Generate button that cannot work. Gating the entry points at build
 * time means the shipped bundle does not contain the dead paths at all, and
 * cannot mislead anyone.
 *
 * The flag is a compile-time constant, so `if (!STATIC_ONLY) { ... }` blocks are
 * dead-code-eliminated by the bundler rather than merely skipped at runtime.
 */

/**
 * Read from `MODE`, which Vite sets from the `--mode` flag. `vite build --mode
 * static` therefore sets it at build time and `vite build` leaves it `production`,
 * with no env var to thread through an npm script.
 */
const isStatic = import.meta.env.MODE === 'static';

/** True in a `build:static` bundle: replay from `public/traces` only. */
export const STATIC_ONLY = isStatic;

/** True when a backend is expected. Inverse of `STATIC_ONLY`. */
export const LIVE_AVAILABLE = !isStatic;

/**
 * Public base path, needed for GitHub Pages-style subpath deploys
 * (`/my-repo/`). Vite's `base` is injected at build time; `BASE_URL` always ends
 * with a slash, so string concatenation needs no separator of its own.
 */
export const BASE_URL = import.meta.env.BASE_URL;

/**
 * Resolves a path inside `public/` against the deploy base path.
 *
 * Written as a function rather than an inline template at each call site, since
 * getting the leading slash wrong yields a request for `/assets/traces/...`
 * that 404s only on a subpath deploy — invisible locally.
 */
export function publicPath(file: string): string {
  return `${BASE_URL}${file.replace(/^\/+/, '')}`;
}
