# frontend

Vite + React + TypeScript + Tailwind + Zustand + D3 + Framer Motion.

```bash
npm install
npm run dev            # http://localhost:5173, proxies /api to :8000
npm run build          # tsc -b && vite build      -> dist/
npm run build:static   # replay-only, no backend    -> dist-static/
npm run smoke          # renders the real components against real trace JSON
npm run smoke:static   # same, against the static-build module graph
```

The app works with **no backend running**: replay mode loads the exported bundles
from `public/traces/*.json`. Live mode needs `uvicorn app.main:app --port 8000`.

> `public/traces/*.json` is gitignored. Run
> `cd ../backend && python -m scripts.export_traces` once after cloning, or the
> replay buttons 404. `build:static` checks for them and says so if missing.

## Layout

```
src/
├── types/trace.ts        wire types, plus TraceSource and pipelineStages
├── lib/env.ts            STATIC_ONLY, VITE_BASE, publicPath()
├── lib/api.ts            typed HTTP + WebSocket wrappers
├── lib/format.ts         number formatting, decided in one place
├── store/useTraceStore.ts  trace list, current step, selected block/head
├── hooks/
│   ├── useTraceSource.ts    the shared TraceSource subscription + rate
│   ├── useLiveGeneration.ts the stable import path callers use
│   ├── liveImpl.ts          the real WebSocket source (aliased out when static)
│   ├── staticLiveStub.ts    the inert replacement (see "Static builds")
│   ├── useReplay.ts         /traces/*.json source
│   ├── useModels.ts         model registry
│   └── useCompareRun.ts     two concurrent generations for ComparePanel
└── components/           Pipeline, BlockDetail, Attention*, LogitLens, ...
```

`useLiveGeneration.ts` is the path the app imports; it re-exports either
`liveImpl.ts` or `staticLiveStub.ts` depending on the build.

## One interface, two sources

`useLiveGeneration` and `useReplay` both return `TraceSource`:

```ts
interface TraceSource {
  status: SourceStatus;
  steps: WSStepMessage[];
  text: string;
  modelId: string;
  prompt: string;
  error: string | null;
  finished: boolean;
  stepsPerSecond: number | null;
  stop(): void;
  reset(): void;
}
```

Both write into `useTraceStore` through the same actions (`startRun`,
`pushStep`, `setStatus`), and every component reads the store. So `ModeToggle`
swapping between them changes the transport and nothing else — no component knows
which mode is active.

This works because the exported bundles hold the *same* `WSStepMessage` payloads
the live socket sends. `TraceBundle` is the only thing replay adds: an envelope
with the prompt, the recording's sampling parameters, and the step list.

## What is derived vs read

Components read the trace; they never recompute model outputs. The few derived
values are in `types/trace.ts` or the store's selectors, and each one says what
it is for:

- `pipelineStages(trace)` builds the node list from `trace.blocks.length`, so it
  is correct for the small config too. Nothing assumes 12.
- `selectDisplayTokens` appends the just-sampled token to the context. That
  difference is exactly one token, because `step.token_ids` includes it and
  `step.trace.input_token_ids` does not.
- `selectCurrentTrace` falls back to the newest available trace when a step was
  streamed with `trace: false`, rather than rendering an empty panel.
- `BlockDetailView` shows `Δ = resid_after − ln_in`, computed from two shipped
  norms. Where a number genuinely cannot be recovered client-side (the ~50k
  logits top-k discarded), the UI says so rather than inventing one.

## Rendering choices worth knowing

- **The causal mask is drawn, not implied.** The upper triangle is exactly `0.0`
  by construction, so `AttentionHeatmap` hatches it. Attention matrices are the
  most misread plot in this area; showing the dead zone removes the ambiguity.
- **Attention colour is per-head** (`[0, max]` of that head) rather than shared.
  A shared domain would be more comparable across heads but would wash out sharp
  ones, and the question the view answers is within-head.
- **Arc height and thickness both scale with weight**, and weak arcs are flattened
  toward the rail rather than dropped — a fan of near-zero arcs is part of the
  picture.
- **Residual norms grow with depth** (~5 → 430 for 124M). That is a real property
  of the residual stream, so it is shown as-is rather than normalized per layer.
- **`prefers-reduced-motion`** disables the pipeline sweep and the entry
  animations; the continuous sweep is decorative.

## Smoke check

`npm run smoke` bundles `src/__smoke__/entry.tsx` with esbuild and runs it. It
does three things:

1. Asserts numeric invariants on all four real traces — the upper triangle is
   exactly zero, the deepest logit-lens entry equals `top_tokens[0]`, sampling
   candidates sum to 1 within float16 tolerance, the lens has one entry per block.
2. Renders the real components with real trace props and checks the output HTML.
3. `npm run smoke:static` re-runs it against the static module graph, which is
   the only coverage of `staticLiveStub` — it never executes in the default run.

Not a browser test, and not a substitute for one. What it does catch is the class
of bug `tsc` cannot see: a selector returning the wrong shape, a crash on the
empty state, a panel indexing past the end of a real attention matrix.

Two constraints worth knowing if you extend it:

- `renderToString` reads zustand's *initial* state (it is the server snapshot), so
  the data-driven assertions mount the presentational components directly rather
  than going through `App`.
- A hook cannot be called outside a render. To exercise one, render a probe
  component and capture the value from inside it — then call the captured
  callbacks from module scope, where no React dispatcher is active. That is the
  shape of the real failure the static check guards: a build that must not crash
  when something calls the stub.

## Static builds

`npm run build:static` produces `dist-static/` — a bundle with **no backend
dependency at all**. It is not the same app with a server missing; it is a
different program:

| | `build` | `build:static` |
| --- | --- | --- |
| output | `dist/` | `dist-static/` |
| Live mode | yes | not offered in the UI |
| Compare panel | yes | hidden (needs the registry) |
| `new WebSocket` in the bundle | yes | **no** |
| `/api/...` strings in the bundle | yes | **no** |
| served by | nginx, proxying `/api` | any static host |

That table is checkable, and worth checking:

```bash
npm run build:static
grep -c 'WebSocket'       dist-static/assets/*.js   # 0
grep -c '/api/ws/generate' dist-static/assets/*.js  # 0
grep -c '/api/models'     dist-static/assets/*.js   # 0
```

Measured on the current tree: `dist/` is 348.65 kB and contains all three;
`dist-static/` is 330.09 kB and contains none of them, while still containing the
`traces/` path it needs.

`npm run preview:static` serves `dist-static/` with proxying explicitly disabled,
so if a bundle is *not* standalone it fails visibly instead of quietly reaching a
backend that happens to be running.

### How the live code gets out

Two mechanisms, because one is not enough:

1. **`src/lib/env.ts`** reads `import.meta.env.MODE === 'static'` into a
   `STATIC_ONLY` constant. `if (!STATIC_ONLY)` branches in the UI are then
   dead-code-eliminated by the bundler.

2. **A `resolve.alias`** in `vite.config.ts` rewrites the specifier
   `@/hooks/liveImpl` to `src/hooks/staticLiveStub.ts`. Mechanism 1 alone is not
   enough, because hooks cannot be called conditionally — `App.tsx` calls
   `useLiveGeneration()` unconditionally, so a runtime guard would leave every
   WebSocket call, the message protocol and the socket lifecycle in the bundle.
   Aliasing removes the module from the graph entirely.

Three details that are easy to get wrong, and silently so — the build succeeds and
the output is unchanged in every case:

- The alias must be listed **before** the `@` alias. Aliases apply in order, and
  `@` would otherwise rewrite `@/hooks/liveImpl` to a path first, leaving nothing
  to match.
- Vite matches the alias against the **import specifier**, not the resolved path.
  An alias aimed at an absolute file path is a no-op.
- esbuild (used by `smoke:static`) takes `alias` as an object, which cannot express
  ordering at all, so the equivalent substitution there is a resolve plugin.

All three were observed failing during development, which is why they are commented
at the call site rather than left implicit.

### The pre-flight check

`build:static` runs `npm run check:traces` first. Traces are gitignored and
generated by a script that needs the backend, so the most likely way to reach this
build is to skip that step — and the result is a deploy where every trace button
404s with no explanation. The check verifies each bundle parses, carries the step
list the replay hook reads, and actually contains traces:

```
checking 4 trace bundle(s)
  ok   effort.json      1.39 MB  4 steps, 4 with traces, T=4
  ...
  total 7.78 MB of trace data will be served as static files
traces OK
```

It also catches a truncated file (an interrupted export), which is worse than a
missing one: it loads, then fails mid-render.

### Deploying

For GitHub Pages or any subpath deploy, copy `.env.static.example` to
`.env.static` and set `VITE_BASE=/your-repo-name/`. All `public/` URLs go through
one `publicPath()` helper in `src/lib/env.ts`, so the base path is applied in a
single place. Getting it wrong is silent locally — asset requests only 404 on the
deployed subpath.

## The registry request is shared

Three components need the model list — `LiveControls` (its picker),
`ComparePanel`, and `useCompareRun` — and all of them mount on the same page
load. Fetched independently that is three identical `GET /api/models` per visit,
which is wasteful and actively misleading in the network tab when you are trying
to work out whether something is polling.

`fetchModels` in `lib/api.ts` dedupes and caches:

- **in-flight dedupe** — concurrent callers await the same promise, so mounting
  three consumers at once issues exactly one request
- **a 15 s TTL** — `ready` flips when a checkpoint finishes loading, so a cache
  with no expiry would freeze the UI on "not ready" after the backend came up
- **no `AbortSignal`** — the request is shared, so one consumer unmounting must
  not cancel it for the other two; consumers use a local `cancelled` flag instead

`useModels` memoizes `ready` on `[models]`, so the effect that defaults the
compare selects runs when the registry actually changes rather than every render.
`useCompareRun` takes `ready` as an argument instead of calling `useModels`
itself, so the registry is subscribed once.

In a static build `fetchModels` returns `[]` immediately and the request is
eliminated from the bundle entirely — asserted by `smoke:static` and by grepping
`dist-static/`.

## Selector stability: a rule that is easy to get wrong

**A zustand selector must return the same reference for unchanged state.** React's
`useSyncExternalStore` calls the selector on *every* render to detect change. A
selector that allocates — returning a new array or object each call — looks like a
store update on every read, so the component re-renders forever and React throws
`The result of getSnapshot should be cached to avoid an infinite loop`.

The failure is a **blank page with no useful error**, and it shipped here once:
`selectDisplayTokens` returned a fresh `.map()`ed array and `selectNextToken` a
fresh object. Both were mounted unconditionally, so the whole app failed to render.

The rule, concretely:

| kind | example | safe as a selector |
| --- | --- | --- |
| primitive | `(s) => s.status` | yes |
| reference held in state | `(s) => s.steps` | yes |
| reference into a stored object | `selectCurrentTrace` → `steps[i].trace` | yes |
| anything that allocates | `deriveDisplayTokens`, `.map(...)`, `{...}` | **no** |

Allocating derivations are named `derive*` in `useTraceStore.ts` and callers wrap
them in `useMemo` keyed on a stable selection:

```tsx
const trace = useTraceStore(selectCurrentTrace);              // stable
const nextToken = useMemo(() => deriveNextToken(useTraceStore.getState()), [trace]);
```

`npm run smoke` asserts this directly — it calls each selector twice on unchanged
state and requires `===`. That check exists because `renderToString` *cannot* see
this bug: server rendering reads the store's initial snapshot, which never
changes, so the loop never starts.

## Deploying with Docker

`Dockerfile` builds to `dist` and serves it with nginx, which also proxies `/api`
to the backend so the browser stays same-origin. Set `VITE_API_BASE` only if the
API lives on a different origin; leave it empty to use the proxy.

`docker compose build --build-arg STATIC=1` produces the replay-only image
instead.
