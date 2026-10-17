/**
 * The live-generation source, as imported by the rest of the app.
 *
 * This module exists to be aliased. `vite.config.ts` rewrites the specifier
 * `@/hooks/liveImpl` to `./staticLiveStub` for `vite build --mode static`, so the
 * real socket implementation is never pulled into a static bundle at all — not
 * merely guarded at runtime. Every
 * `import { useLiveGeneration } from '@/hooks/useLiveGeneration'` in the app keeps
 * working unchanged in both modes, which is what lets `App.tsx` stay
 * mode-agnostic.
 *
 * The import below deliberately goes through the `@/hooks/liveImpl` specifier
 * rather than a relative `./liveImpl`. Vite's `resolve.alias` matches the
 * *import specifier*, not the resolved file path, so an alias aimed at a resolved
 * path silently does nothing — the build succeeds and the bundle is unchanged,
 * which is exactly the failure this indirection exists to prevent.
 *
 * `STATIC_ONLY` is re-exported for the cases where the distinction still has to
 * be visible in the client bundle: which controls to render, whether the mode
 * toggle offers Live, whether the model registry is reachable.
 *
 * This pattern only works because both implementations are genuinely
 * interchangeable — same interface, same store actions — which is the same
 * property that lets `ModeToggle` switch sources at runtime.
 */

export { STATIC_ONLY } from '@/lib/env';

export type { LiveGeneration } from '@/hooks/liveImpl';

/**
 * Resolved at build time: the real socket in a normal build, the inert stub in a
 * static one. Both modules export an identically-shaped surface, so the alias is
 * invisible to callers.
 */
export * from '@/hooks/liveImpl';