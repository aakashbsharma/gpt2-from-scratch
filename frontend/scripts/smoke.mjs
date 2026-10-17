/**
 * Bundles and runs `src/__smoke__/entry.tsx`.
 *
 * Uses esbuild directly rather than adding a test-runner dependency: the point is
 * to execute the real components and the real store against real trace JSON, and
 * that needs nothing more than a bundler that understands TSX and the `@` alias.
 *
 * Also the runner for the `inspect:*` scripts (`--inspect=lens`), which render a
 * single component and print its exact output. They share this configuration so a
 * one-off question can be settled against rendered text instead of by squinting
 * at a screenshot — leading spaces and off-by-one row labels are both easy to
 * think you saw and hard to confirm by eye.
 *
 *   npm run smoke               run the assertions (live module graph)
 *   npm run smoke:static        run them against the static module graph
 *   npm run inspect:lens        print LogitLens's rendered top-1 per layer
 */

import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import { rmSync } from 'node:fs';
import { join } from 'node:path';

/** `--static` checks the static-build module graph, so `staticLiveStub` runs too. */
const STATIC_SMOKE = process.argv.includes('--static');

/**
 * `--inspect=<name>` runs `scripts/inspect-<name>.tsx` instead of the smoke entry.
 *
 * The bundle is imported into this same process, so `process.argv` is unchanged
 * when it runs — which is why the inspect scripts read their arguments from
 * `process.argv` directly rather than having them plumbed through here.
 */
const INSPECT = process.argv
  .find((a) => a.startsWith('--inspect='))
  ?.slice('--inspect='.length);

const SRC = join(process.cwd(), 'src');
const ENTRY = INSPECT
  ? join(process.cwd(), 'scripts', `inspect-${INSPECT}.tsx`)
  : join(SRC, '__smoke__', 'entry.tsx');
const OUT = join(process.cwd(), 'node_modules', '.smoke', 'entry.mjs');

await build({
  entryPoints: [ENTRY],
  outfile: OUT,
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  jsx: 'automatic',

  define: {
    // React's dev build warns about `useSyncExternalStore` on the server; the
    // warnings are noise here and the assertions are the signal.
    'process.env.NODE_ENV': '"production"',
    // Vite injects these; esbuild does not, so supply the same defaults.
    'import.meta.env.VITE_API_URL': '""',
    'import.meta.env.BASE_URL': '""',
    // `production` matches a normal `vite build`, which is the mode the smoke
    // check asserts against. `src/lib/env.ts` reads this to derive STATIC_ONLY.
    'import.meta.env.MODE': STATIC_SMOKE ? '"static"' : '"production"',
    'import.meta.env.DEV': 'false',
  },

  // react-dom/server is CommonJS and `require`s node builtins. Bundling it as
  // ESM rewrites those calls into an unsupported dynamic require, so give it a
  // real `require` via createRequire.
  banner: {
    js: "import { createRequire as __createRequire } from 'node:module';\nconst require = __createRequire(import.meta.url);",
  },

  logLevel: 'warning',

  alias: { '@': SRC },

  // esbuild's `alias` only accepts an object, which cannot express ordering. Vite's
  // alias is an ordered array precisely because ordering matters there: `@` must not
  // swallow `@/hooks/liveImpl` first. A resolve plugin gets the ordering.
  plugins: STATIC_SMOKE
    ? [
        {
          name: 'static-live-alias',
          setup(pluginBuild) {
            pluginBuild.onResolve({ filter: /^@\/hooks\/liveImpl$/ }, () => ({
              path: join(SRC, 'hooks', 'staticLiveStub.ts'),
            }));
          },
        },
      ]
    : [],
});

try {
  await import(pathToFileURL(OUT).href);
} finally {
  rmSync(join(process.cwd(), 'node_modules', '.smoke'), { recursive: true, force: true });
}
