/**
 * App shell.
 *
 * The only place that knows which data source is active. Everything below is
 * written against the shared `TraceSource` interface and the store, so switching
 * `ModeToggle` swaps the transport without touching a single visualization.
 */

import { useEffect, useMemo, useState } from 'react';

import { AttentionArcs } from '@/components/AttentionArcs';
import { AttentionHeatmap } from '@/components/AttentionHeatmap';
import { BlockDetail } from '@/components/BlockDetail';
import { ComparePanel } from '@/components/ComparePanel';
import { HeadSelector } from '@/components/HeadSelector';
import { LiveControls } from '@/components/LiveControls';
import { LogitLens } from '@/components/LogitLens';
import { ModeToggle } from '@/components/ModeToggle';
import { Pipeline } from '@/components/Pipeline';
import { LiveStatus, ReplayControls } from '@/components/ReplayControls';
import { TokenStrip } from '@/components/TokenStrip';
import { DrawStrip } from '@/components/sampling/DrawStrip';
import { SamplingFunnel } from '@/components/sampling/SamplingFunnel';
import { SamplingHistory } from '@/components/sampling/SamplingHistory';
import { SamplingStats } from '@/components/sampling/SamplingStats';

import { useLiveGeneration } from '@/hooks/useLiveGeneration';
import { useReplay } from '@/hooks/useReplay';
import { STATIC_ONLY } from '@/lib/env';
import { formatPercent } from '@/lib/format';
import {
  useTraceStore,
  deriveNextToken,
  selectCurrentTrace,
  selectSelectedBlock,
  selectSelectedHead,
} from '@/store/useTraceStore';
import type { ReplayTraceName } from '@/types/trace';

export function App() {
  const mode = useTraceStore((s) => s.mode);
  const trace = useTraceStore(selectCurrentTrace);
  const block = useTraceStore(selectSelectedBlock);
  const head = useTraceStore(selectSelectedHead);
  const queryPosition = useTraceStore((s) => s.queryPosition);
  const selectQuery = useTraceStore((s) => s.selectQuery);
  const steps = useTraceStore((s) => s.steps);
  const currentStep = useTraceStore((s) => s.currentStep);
  const selectStep = useTraceStore((s) => s.selectStep);
  const selectedBlock = useTraceStore((s) => s.selectedBlock);
  const selectedHead = useTraceStore((s) => s.selectedHead);
  const selectHead = useTraceStore((s) => s.selectHead);

  const live = useLiveGeneration();
  const replay = useReplay();
  const [attentionView, setAttentionView] = useState<'heatmap' | 'arcs'>('heatmap');

  // Load a default bundle on mount so the app is never an empty shell — the
  // whole point of the exported traces is that this works with no backend.
  const [autoLoaded, setAutoLoaded] = useState(false);
  useEffect(() => {
    if (autoLoaded) return;
    setAutoLoaded(true);
    void replay.load('effort' as ReplayTraceName);
  }, [autoLoaded, replay]);

  const tokens = trace?.tokens ?? [];

  // Derived under `useMemo`, not through `useTraceStore(deriveNextToken)`: the
  // derivation allocates a new object, and React's `useSyncExternalStore` calls
  // the selector on every render to detect changes. A fresh reference reads as a
  // store change, so the component re-renders forever and React throws — the app
  // comes up blank with no useful error. Selections here are all stable
  // references into the store, so only the derivation needs memoizing.
  const nextToken = useMemo(() => deriveNextToken(useTraceStore.getState()), [trace]);
  const samplingTrace =
    currentStep >= 0 ? steps[currentStep]?.trace ?? null : trace;
  const panelNextToken = currentStep >= 0 && !samplingTrace ? null : nextToken;

  return (
    <div className="mx-auto flex min-h-full max-w-[1500px] flex-col gap-3 p-3">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-sm font-semibold tracking-tight text-slate-100">
            GPT-2 forward pass, token by token
          </h1>
          <p className="text-[11px] text-slate-500">
            every number below comes from a real forward pass — nothing here is mocked
          </p>
        </div>
        <ModeToggle />
      </header>

      {/* `mode === 'live'` is unreachable in a static build, so the whole live
          branch — including the WebSocket hook's controls — is dropped. */}
      {mode === 'live' && !STATIC_ONLY ? <LiveControlsPanel live={live} /> : <ReplayControls replay={replay} />}

      <TokenStrip />
      <Pipeline />

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="space-y-3">
          <section className="panel" aria-label="Attention">
            <header className="panel-header">
              <h2 className="panel-title">
                Attention
                {block && head && (
                  <span className="ml-2 font-normal normal-case tracking-normal text-slate-500">
                    block {(selectedBlock ?? 0) + 1} · head {head.head} ·{' '}
                    {head.weights.length}×{head.weights.length}
                  </span>
                )}
              </h2>
              <div className="flex gap-1">
                <button
                  type="button"
                  className={['btn px-2 py-0.5 text-[11px]', attentionView === 'heatmap' ? 'btn-active' : ''].join(' ')}
                  onClick={() => setAttentionView('heatmap')}
                >
                  heatmap
                </button>
                <button
                  type="button"
                  className={['btn px-2 py-0.5 text-[11px]', attentionView === 'arcs' ? 'btn-active' : ''].join(' ')}
                  onClick={() => setAttentionView('arcs')}
                >
                  arcs
                </button>
              </div>
            </header>

            <HeadSelector
              block={block ?? undefined}
              selectedHead={selectedHead}
              queryPosition={queryPosition}
              onSelect={selectHead}
            />

            {attentionView === 'heatmap' ? (
              <AttentionHeatmap
                head={head}
                tokens={tokens}
                queryPosition={queryPosition}
                onQueryChange={selectQuery}
              />
            ) : (
              <AttentionArcs
                head={head}
                tokens={tokens}
                queryPosition={queryPosition}
                onQueryChange={selectQuery}
              />
            )}
          </section>

          <BlockDetail />

          {/* Comparison needs two models from the registry, so it has nothing to
              compare in a static build. Hidden rather than shown disabled. */}
          {!STATIC_ONLY && <ComparePanel />}
        </div>

        <div className="space-y-3">
          <section className="panel" aria-label="Logit lens">
            <header className="panel-header">
              <h2 className="panel-title">Logit lens</h2>
              <span className="text-[11px] text-slate-500">top prediction per layer</span>
            </header>
            <LogitLens trace={trace} />
          </section>

          <section className="panel" aria-label="Sampling">
            <header className="panel-header">
              <h2 className="panel-title">{panelNextToken ? 'Sampling' : 'Next token'}</h2>
              <span className="text-[11px] text-slate-500">
                {samplingTrace?.sampling
                  ? 'temperature and top-k decision'
                  : 'top-1 at the last context position'}
              </span>
            </header>
            {panelNextToken && (
              <p className="border-b border-ink-800 px-3 py-2 text-xs">
                <span className="label mr-2">argmax</span>
                <span className="tnum text-accent-400">{JSON.stringify(panelNextToken.piece)}</span>
                <span className="value ml-2 text-[11px] text-slate-500">
                  {formatPercent(panelNextToken.prob, 2)}
                </span>
              </p>
            )}
            <SamplingFunnel trace={samplingTrace} />
            <DrawStrip sampling={samplingTrace?.sampling ?? null} />
            <SamplingStats sampling={samplingTrace?.sampling ?? null} />
            <SamplingHistory
              steps={steps}
              currentStep={currentStep}
              onSelectStep={selectStep}
            />
          </section>
        </div>
      </div>

      <footer className="pb-2 pt-1 text-[11px] text-slate-600">
        traces from <span className="text-slate-500">docs/trace-schema.md</span> · residuals arrive
        as L2 norms + 2D PCA because raw 768-wide activations are never transmitted · attention is
        float16-quantized to 4 decimals
      </footer>
    </div>
  );
}

/** Wraps the live controls with the socket status readout. */
function LiveControlsPanel({ live }: { live: ReturnType<typeof useLiveGeneration> }) {
  return (
    <div className="space-y-2">
      <LiveControls live={live} />
      <LiveStatus connected={live.connected} stepsPerSecond={live.stepsPerSecond} status={live.status} />
      {live.error && (
        <p className="rounded border border-rose-500/30 bg-rose-500/5 px-3 py-1.5 text-[11px] text-rose-400">
          {live.error}
        </p>
      )}
    </div>
  );
}
