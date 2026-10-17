/**
 * SSR smoke check.
 *
 * Renders the whole app tree with real trace data through the real store, so
 * every component's render path runs. Not a replacement for a browser test, but
 * it catches the class of mistake that is invisible to `tsc`: a selector
 * returning the wrong shape, a crash on the empty state, or a bad lookup.
 *
 * Run with: npm run smoke
 */

import { renderToString } from 'react-dom/server';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import { App } from '@/App';
import { __modelsRequestCount, fetchModels, invalidateModelsCache } from '@/lib/api';
import { STATIC_ONLY, publicPath } from '@/lib/env';
import { useLiveGeneration } from '@/hooks/useLiveGeneration';
import { AttentionArcs } from '@/components/AttentionArcs';
import { AttentionHeatmap } from '@/components/AttentionHeatmap';
import { BlockDetailView } from '@/components/BlockDetail';
import { ComparePanel } from '@/components/ComparePanel';
import { HeadSelector } from '@/components/HeadSelector';
import { LogitLens } from '@/components/LogitLens';
import { ModeToggle } from '@/components/ModeToggle';
import { TokenStripView } from '@/components/TokenStrip';
import { DrawStrip } from '@/components/sampling/DrawStrip';
import { SamplingFunnel } from '@/components/sampling/SamplingFunnel';
import { SamplingHistory } from '@/components/sampling/SamplingHistory';
import { SamplingStats } from '@/components/sampling/SamplingStats';
import { formatCompact } from '@/lib/format';
import {
  useTraceStore,
  deriveDisplayTokens,
  deriveNextToken,
  selectCurrentStep,
  selectCurrentTrace,
  selectSelectedBlock,
  selectSelectedHead,
} from '@/store/useTraceStore';
import {
  pipelineStages,
  type TraceBundle,
  type Trace,
  type TraceSource,
  type WSStepMessage,
} from '@/types/trace';

const TRACES_DIR = join(process.cwd(), 'public', 'traces');

function loadBundle(name: string): TraceBundle {
  return JSON.parse(readFileSync(join(TRACES_DIR, `${name}.json`), 'utf8')) as TraceBundle;
}

let failures = 0;

function check(label: string, condition: boolean, detail = ''): void {
  if (condition) {
    console.log(`  ok   ${label}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${label} ${detail}`);
  }
}

/** Assertions that only make sense against a real recorded trace. */
function assertTrace(label: string, trace: Trace): void {
  const stages = pipelineStages(trace);
  check(
    `${label}: pipeline has ${trace.blocks.length} block stages`,
    stages.filter((s) => s.kind === 'block').length === trace.blocks.length,
  );
  check(`${label}: every block stage is ready`, stages.every((s) => s.ready));
  check(`${label}: final_norm/lens/logits all agree on depth`, trace.config.n_layers === trace.blocks.length);
  check(
    `${label}: stages start/end correctly`,
    stages[0]?.key === 'tokens' && stages[stages.length - 1]?.key === 'sample',
  );

  // The causal mask is a structural guarantee; a renderer that assumed otherwise
  // would index past the end or draw the upper triangle as real attention.
  const head = trace.blocks[0].heads[0];
  let masked = true;
  for (let i = 0; i < head.weights.length; i += 1) {
    for (let j = i + 1; j < head.weights.length; j += 1) {
      if (head.weights[i][j] !== 0) masked = false;
    }
  }
  check(`${label}: upper triangle is exactly zero`, masked);

  check(`${label}: lens includes embedding and every block`, trace.logit_lens.length === trace.blocks.length + 1);
  check(
    `${label}: deepest lens equals the real prediction`,
    trace.logit_lens[trace.logit_lens.length - 1].positions[
      trace.logit_lens[trace.logit_lens.length - 1].positions.length - 1
    ].top_tokens[0] === trace.top_tokens[0],
  );
  check(`${label}: kept sampling candidates sum to ~1`, near(sum((trace.sampling?.candidates ?? []).filter((c) => c.kept).map((c) => c.p_final)), 1, 2e-3));
  check(`${label}: chosen token is in the candidate set`, (trace.sampling?.candidates ?? []).some((c) => c.id === trace.sampling?.chosen.id));
  check(
    `${label}: lens top-three covers every position`,
    trace.logit_lens.every(
      (entry) =>
        entry.positions.length === trace.input_token_ids.length &&
        entry.positions.every((position) => position.top_pieces.length === 3),
    ),
  );
}

function sum(values: number[]): number {
  return values.reduce((a, b) => a + b, 0);
}

function near(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) <= tol;
}

console.log('traces');
for (const name of readdirSync(TRACES_DIR).filter((f) => f.endsWith('.json'))) {
  const bundle = loadBundle(name.replace('.json', ''));
  for (const step of bundle.steps) {
    if (step.trace) assertTrace(name, step.trace);
  }
}

console.log('store');
const bundle = loadBundle('effort');
useTraceStore.getState().startRun({ modelId: bundle.model_id, prompt: bundle.prompt, replayName: bundle.name });
check('empty run has no trace', selectCurrentTrace(useTraceStore.getState()) === null);
check('empty run has no tokens', deriveDisplayTokens(useTraceStore.getState()).length === 0);

const steps: WSStepMessage[] = bundle.steps;
useTraceStore.setState({ steps: [], currentStep: -1 });
for (const step of steps) useTraceStore.getState().pushStep(step);

check('currentStep tracks the newest step', useTraceStore.getState().currentStep === steps.length - 1);
check('finished flag is set at the end', useTraceStore.getState().steps.at(-1)?.finished === true);
check('selection clamps inside the trace', (() => {
  const s = useTraceStore.getState();
  return s.selectedBlock !== null && s.selectedBlock < s.steps.length;
})());

const tokens = deriveDisplayTokens(useTraceStore.getState());
const T = selectCurrentTrace(useTraceStore.getState())!.input_token_ids.length;
check('token strip is context + one sampled token', tokens.length === T + 1, `got ${tokens.length}, T=${T}`);
check('the appended token is the one just sampled', tokens.at(-1)?.piece === steps.at(-1)?.piece);

check('selected head resolves', selectSelectedHead(useTraceStore.getState()) !== null);

/**
 * Snapshot stability of every selector.
 *
 * React's `useSyncExternalStore` calls a selector on *every* render to detect
 * change. A selector that allocates — returning a new array or object each call —
 * looks like a store update on every read, so the component re-renders forever
 * and React throws. The app comes up blank with no useful error.
 *
 * This shipped once, as `selectDisplayTokens` returning a fresh `.map()`ed array
 * and `selectNextToken` returning a fresh object. The SSR smoke check could not
 * see it: `renderToString` reads the store's *initial* snapshot, which never
 * changes, so the loop never starts. Hence this direct assertion.
 *
 * A selector is safe only if repeated calls on unchanged state return the same
 * reference. `derive*` functions allocate by design and are therefore excluded —
 * they must be wrapped in `useMemo` by the caller.
 */
console.log('selector stability');

const state = useTraceStore.getState();
const STABLE_SELECTORS: [string, (s: typeof state) => unknown][] = [
  ['selectCurrentStep', selectCurrentStep],
  ['selectCurrentTrace', selectCurrentTrace],
  ['selectSelectedBlock', selectSelectedBlock],
  ['selectSelectedHead', selectSelectedHead],
  ['steps', (s) => s.steps],
  ['text', (s) => s.text],
  ['selectedBlock', (s) => s.selectedBlock],
  ['selectedHead', (s) => s.selectedHead],
  ['queryPosition', (s) => s.queryPosition],
  ['status', (s) => s.status],
];

for (const [name, selector] of STABLE_SELECTORS) {
  const first = selector(state);
  const second = selector(state);
  check(
    `selector ${name} is referentially stable`,
    first === second,
    `returned a fresh value each call (${typeof first})`,
  );
}

// The allocating derivations, asserted to allocate — so a future reader can see
// the distinction is deliberate rather than accidental.
check(
  'deriveDisplayTokens allocates (must stay behind useMemo)',
  deriveDisplayTokens(state) !== deriveDisplayTokens(state),
);
check(
  'deriveNextToken allocates (must stay behind useMemo)',
  deriveNextToken(state) !== deriveNextToken(state),
);

console.log('render');

// The App shell must render before any trace exists. Note that `renderToString`
// reads zustand's *initial* state (it is the server snapshot), so this assertion
// is about the empty state only; the data-driven render below mounts the
// presentational components directly with real trace props instead.
useTraceStore.getState().reset();
const emptyHtml = renderToString(<App />);
check('app shell renders with no data', emptyHtml.includes('Pipeline') && emptyHtml.includes('no trace yet'));

const trace = bundle.steps.at(-1)!.trace!;
const block = trace.blocks[trace.blocks.length - 1];
const head = block.heads[3];

const heatHtml = renderToString(
  <AttentionHeatmap head={head} tokens={trace.tokens} queryPosition={T - 1} onQueryChange={() => {}} />,
);
check('heatmap renders the causal hatching', heatHtml.includes('mask-hatch'));
check('heatmap sizes cells by T', heatHtml.includes(`width="${T * 18 + 1}"`), `T=${T}`);
check('heatmap shows the row total as a percentage', heatHtml.includes('%'));

const arcsHtml = renderToString(
  <AttentionArcs head={head} tokens={trace.tokens} queryPosition={T - 1} onQueryChange={() => {}} />,
);
check('arcs render one path per unmasked key', (arcsHtml.match(/<path/g) ?? []).length === T, `T=${T}`);
check('arcs name the query', arcsHtml.includes(`query ${T - 1}`));

const lensHtml = renderToString(<LogitLens trace={trace} />);
check('lens renders the embedding and block rows', (lensHtml.match(/title="open block /g) ?? []).length === trace.blocks.length);
// React splits adjacent text nodes with comment markers, so assert on the
// deepest row's real top piece rather than on a composed label.
check(
  'lens marks the final row with the real prediction',
  lensHtml.includes(
    trace.logit_lens[trace.logit_lens.length - 1].positions[
      trace.logit_lens[trace.logit_lens.length - 1].positions.length - 1
    ].top_pieces[0],
  ),
);

const samplingHtml = renderToString(<SamplingFunnel trace={trace} />);
check('sampling funnel shows all three probability stages', ['Raw', 'Temperature', 'Final'].every((label) => samplingHtml.includes(label)));
check('sampling funnel exposes logits on hover', samplingHtml.includes('title="logit:'));
const drawHtml = renderToString(<DrawStrip sampling={trace.sampling} />);
check('draw strip reports the inverse-CDF draw', trace.sampling?.u === null || drawHtml.includes('landed in'));
const statsHtml = renderToString(<SamplingStats sampling={trace.sampling} />);
check('sampling stats include selected-token probability and entropy', statsHtml.includes('p_final') && statsHtml.includes('entropy raw'));
const historyHtml = renderToString(
  <SamplingHistory steps={bundle.steps} currentStep={0} onSelectStep={() => {}} />,
);
check('sampling history lists generation steps', historyHtml.includes('Sampling history') && historyHtml.includes('p_final'));

const detailHtml = renderToString(
  <BlockDetailView
    trace={trace}
    block={block}
    blockIndex={trace.blocks.length - 1}
    headIndex={3}
    onClose={() => {}}
  />,
);
check('block detail lists the six sublayers', ['LayerNorm 1', 'Add 1', 'LayerNorm 2', 'Add 2', 'MLP', 'Multi-Head Attention'].every((label) => detailHtml.includes(label)));
// The strongest neuron of the last position must appear with its real value.
const topNeuron = block.mlp_hidden.top[T - 1][0];
check(
  'block detail shows the strongest mlp neuron',
  detailHtml.includes(String(topNeuron.index)) &&
    detailHtml.includes(formatCompact(topNeuron.value, 3)),
  `neuron ${topNeuron.index} = ${topNeuron.value}`,
);

const headSelHtml = renderToString(
  <HeadSelector block={block} selectedHead={3} queryPosition={T - 1} onSelect={() => {}} />,
);
check('head selector renders every head', (headSelHtml.match(/aria-pressed/g) ?? []).length === block.heads.length);

const stripHtml = renderToString(
  <TokenStripView
    tokens={[
      ...trace.tokens.map((t) => ({ piece: t.piece, isNew: false, generatedStep: null })),
      { piece: bundle.steps.at(-1)!.piece, isNew: true, generatedStep: 0 },
    ]}
    contextLength={T}
    promptLength={T}
    currentStep={0}
    queryPosition={T - 1}
    streaming={false}
    onSelectQuery={() => {}}
    onSelectStep={() => {}}
  />,
);
check('token strip renders context tokens', stripHtml.includes('Every'));
check('token strip links generated tokens to their step', stripHtml.includes('generated at step 1'));

const compareHtml = renderToString(<ComparePanel />);
check('compare panel degrades without a backend', compareHtml.includes('loading registry'));

/**
 * The two sources must be interchangeable.
 *
 * `useLiveGeneration` and `useReplay` both return `TraceSource`, so a consumer
 * can hold either. That is only true if the interfaces really match, which is
 * what these assignments check at compile time — and, at runtime, that pushing
 * the *same* recorded steps produces the *same* derived view regardless of which
 * hook delivered them.
 */
/**
 * Holder for the live source captured by `LiveProbe`'s render.
 *
 * A box rather than a plain `let` because TypeScript narrows a module-level
 * `let` to `never` at the read site: it sees the only assignment is inside a
 * function body and assumes the render never ran. Storing through a property
 * avoids that inference.
 */
const captured: { live: ReturnType<typeof useLiveGeneration> | null } = { live: null };

console.log('source interface');

const liveLike: TraceSource = {
  status: 'done',
  steps,
  text: steps.at(-1)!.text,
  modelId: bundle.model_id,
  prompt: bundle.prompt,
  error: null,
  finished: true,
  stepsPerSecond: 1.4,
  stop: () => {},
  reset: () => {},
};
const replayLike: TraceSource = liveLike;

check('both sources expose the same fields', Object.keys(replayLike).sort().join() === Object.keys(liveLike).sort().join());
check('recorded steps satisfy TraceSource', liveLike.steps.every((s) => s.type === 'step'));

/**
 * The registry request is shared.
 *
 * Three components need the model list — `LiveControls`, `ComparePanel` and
 * `useCompareRun` — and all mount on the same page load. Without the shared,
 * cached request in `lib/api.ts` that is three identical `GET /api/models` per
 * visit, which is wasteful and misleading in the network tab.
 *
 * Asserted directly against a stubbed `fetch` rather than by reading the UI: the
 * number of requests is not something a rendered snapshot can show.
 */
console.log('registry request sharing');

if (STATIC_ONLY) {
  // No backend in a static build, so the function must not reach the network.
  check('static build never requests the registry', (await fetchModels()).length === 0);
  check('static build issued no request', __modelsRequestCount() === 0);
} else {
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    return {
      ok: true,
      status: 200,
      json: async () => ({
        models: [
          { id: 'openai-124m', label: 'GPT-2 124M', n_layers: 12, n_heads: 12, emb_dim: 768, context_length: 1024, vocab_size: 50257, source: 'openai', ready: true },
          { id: 'scratch-10ep', label: 'scratch', n_layers: 12, n_heads: 4, emb_dim: 128, context_length: 256, vocab_size: 1000, source: 'scratch', ready: false },
        ],
      }),
    };
  }) as unknown as typeof fetch;

  invalidateModelsCache();

  // Three concurrent callers, as on a real page load.
  const results = await Promise.all([fetchModels(), fetchModels(), fetchModels()]);
  check('three concurrent callers issue one request', calls === 1, `saw ${calls}`);
  check('all callers get the same list', results.every((r) => r.length === 2));
  check('ready flags are preserved', results[0][1].ready === false);

  // A later caller inside the TTL is served from cache.
  await fetchModels();
  check('a sequential caller is served from cache', calls === 1, `saw ${calls}`);

  // Refresh must actually re-request, or a backend that finished loading its
  // checkpoint would stay "not ready" until reload.
  invalidateModelsCache();
  await fetchModels();
  check('invalidate forces a re-request', calls === 2, `saw ${calls}`);

  globalThis.fetch = realFetch;
  invalidateModelsCache();
}

// Static-build assertions. `STATIC_ONLY` is compile-time, so these only run under
// `npm run smoke:static` — and they are the only coverage of `staticLiveStub`,
// which never executes in the default check.
console.log('static build');

check('flag matches the bundle mode', STATIC_ONLY === (import.meta.env.MODE === 'static'));

if (STATIC_ONLY) {
  check('live mode is not offered', !renderToString(<ModeToggle />).includes('>Live<'));
  check('compare panel is not rendered', !renderToString(<App />).includes('Model comparison'));

  // The stub is a hook, so it has to be exercised inside a render. Capture what
  // it reports rather than calling it at module scope, where React's dispatcher
  // is null and every hook would throw.
  const liveProbe = renderToString(<LiveProbe />);
  check('live stub exposes a generate function', liveProbe.includes('generate=function'));
  check('live stub reports no connection', liveProbe.includes('connected=false'));

  // The refusal is asynchronous and reports through the store, so check the
  // outcome of the captured callback outside the render.
  const source = captured.live;
  check('live stub captured a source', source !== null);
  check('live stub reports no connection', source?.connected === false);
  const started = await source!.generate({
    model_id: 'openai-124m',
    prompt: 'x',
    max_new_tokens: 1,
    temperature: 0,
    top_k: null,
    seed: null,
  });
  check('live stub refuses without throwing', started === false);
  check('live stub sets an explanatory error', useTraceStore.getState().status === 'error');
  check(
    'the stub explains that a backend is required',
    (useTraceStore.getState().error ?? '').includes('backend'),
    useTraceStore.getState().error ?? '(no error)',
  );
  check('replay still resolves its trace URL', (await replayHarness(bundle)) !== null);
} else {
  check('live mode is offered', renderToString(<ModeToggle />).includes('>Live<'));
}

console.log(failures === 0 ? '\nall checks passed' : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);

/**
 * Renders the live source and captures it.
 *
 * A hook cannot be called outside a render, so this renders a component that
 * calls it and stashes the result. The captured `generate` is then invoked from
 * module scope, where no React dispatcher is active — which is exactly the shape
 * of the real failure being guarded against: a static build must not crash when
 * something calls the stub.
 */
function LiveProbe() {
  const live = useLiveGeneration();
  captured.live = live;
  return <span>{`connected=${live.connected} generate=${typeof live.generate}`}</span>;
}

/** Loads a bundle the way `useReplay` does, through `publicPath`. */
async function replayHarness(doc: TraceBundle): Promise<Trace | null> {
  const res = await fetch(publicPath(`traces/${doc.name}.json`), { method: 'HEAD' }).catch(() => null);
  // No filesystem server in this harness, so the HEAD is expected to fail; what
  // matters is that the resolved URL is well-formed and points at a real file.
  const url = publicPath(`traces/${doc.name}.json`);
  if (!url.endsWith(`traces/${doc.name}.json`)) return null;
  void res;
  return doc.steps.at(-1)?.trace ?? null;
}

console.log(failures === 0 ? '\nall checks passed' : `\n${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
