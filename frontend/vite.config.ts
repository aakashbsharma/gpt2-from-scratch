import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';



const SRC = fileURLToPath(new URL('./src', import.meta.url));

/** The specifier `src/hooks/useLiveGeneration.ts` imports the real source by. */
const LIVE_IMPL = '@/hooks/liveImpl';
const STATIC_STUB = `${SRC}/hooks/staticLiveStub.ts`;

/**
 * A build is *static* when it is invoked as `vite build --mode static`, which is
 * what `npm run build:static` does.
 *
 * Driven off `--mode` rather than an environment variable: setting one from an
 * npm script needs `cross-env` (a new dependency) to survive Windows shells, and
 * `--mode` already selects `.env.static`, so one switch controls both.
 *
 * `src/lib/env.ts` reads the same signal through `import.meta.env.MODE`, so the
 * `if (!STATIC_ONLY)` branches in the UI are dead-code-eliminated. That alone is
 * not enough to keep live code out of the bundle, though: hooks cannot be called
 * conditionally, so `App.tsx` calls `useLiveGeneration()` unconditionally. The
 * `resolve.alias` below handles that half — see the comment there.
 */
export default defineConfig(({ mode }) => {
  const isStatic = mode === 'static';

  // `loadEnv` is needed because `base` is resolved in Node, not in the client
  // bundle: VITE_BASE lives in `.env.static` and must be read here too.
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  const BACKEND = env.VITE_API_URL ?? process.env.VITE_BACKEND_ORIGIN ?? 'http://localhost:8000';
  return {
    plugins: [react()],
    resolve: {
      // Swap the live source for an inert stub in a static build.
      //
      // Module aliasing rather than a runtime `if (STATIC_ONLY)` because hooks
      // cannot be called conditionally, so `App.tsx` calls
      // `useLiveGeneration()` unconditionally. A guard inside the hook would
      // still leave every WebSocket call, the message protocol and the socket
      // lifecycle in the static bundle. Aliasing removes the module from the
      // graph entirely, so the static output genuinely has no live code.
      alias: [
        // Must come BEFORE the `@` alias: aliases are applied in order, and the
        // `@` entry would otherwise rewrite `@/hooks/liveImpl` to a path first,
        // leaving this entry nothing to match.
        ...(isStatic ? [{ find: LIVE_IMPL, replacement: STATIC_STUB }] : []),
        { find: '@', replacement: SRC },
      ],
    },
    server: {
      port: 5173,
      // No proxy for a static build: there is no backend to reach, and silently
      // forwarding would mask a genuinely missing server.
      proxy: isStatic
        ? undefined
        : { '/api': { target: BACKEND, changeOrigin: true, ws: true } },
    },

    // `vite preview` falls back to `server.proxy` when `preview.proxy` is
    // *undefined*, so a static preview would proxy /api to a backend that happens
    // to be running — making a bundle that is not standalone look like it works.
    // An empty object is "no proxy rules", which is what verifying standalone
    // operation actually requires. (`proxy: false` is not an option in Vite 5's
    // types, and an empty object is the supported spelling.)
    preview: isStatic ? { proxy: {} } : {},

    // Subpath deploys (GitHub Pages) set VITE_BASE=/repo-name/ in `.env.static`.
    // Defaults to '/', which is correct for a root deploy and for `vite preview`.
    base: env.VITE_BASE ?? '/',

    build: {
      // Separate output so a static deploy can never clobber the Docker build.
      outDir: isStatic ? 'dist-static' : 'dist',
      sourcemap: true,
      // Traces are 1.4–2.7 MB of JSON fetched at runtime and never imported, so
      // they stay out of the bundle and are served as plain static files.
      assetsInlineLimit: 4096,
    },
  };
});
