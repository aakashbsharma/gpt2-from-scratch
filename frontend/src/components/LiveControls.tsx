/**
 * Live-generation controls.
 *
 * Mirrors the parameters `WSGenerateRequest` actually accepts. `temperature <= 0`
 * and `top_k: null` are sent as-is rather than being validated client-side into
 * some other value, so what the socket receives is what the UI displays.
 */

import { useEffect, useState } from 'react';

import { useModels } from '@/hooks/useModels';
import type { LiveGeneration } from '@/hooks/useLiveGeneration';

export function LiveControls({ live }: { live: LiveGeneration }) {
  // `useModels` rather than a private `fetchModels` call: the registry request is
  // shared and cached, so this adds no request of its own, and an empty result
  // doubles as the "backend unreachable" signal.
  const { models } = useModels();

  const [modelId, setModelId] = useState('');
  const [prompt, setPrompt] = useState('Every effort moves you');
  const [maxNew, setMaxNew] = useState(8);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [useTopK, setUseTopK] = useState(true);
  const [withTrace, setWithTrace] = useState(true);

  // Default to the first ready model once the registry lands, but never
  // overwrite a choice the user has already made.
  useEffect(() => {
    if (modelId) return;
    const firstReady = models.find((m) => m.ready);
    if (firstReady) setModelId(firstReady.id);
  }, [models, modelId]);

  const running = live.status === 'running' || live.status === 'loading';
  const greedy = temperature <= 0;

  return (
    <section className="panel" aria-label="Live generation controls">
      <div className="grid grid-cols-1 gap-3 px-3 py-3 md:grid-cols-2 lg:grid-cols-4">
        <label className="flex flex-col gap-1 md:col-span-2 lg:col-span-4">
          <span className="label">prompt</span>
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="tnum rounded border border-ink-600 bg-ink-850 px-2 py-1.5 text-xs text-slate-200
                       placeholder:text-ink-500 focus:border-accent-500 focus:outline-none"
            placeholder="Every effort moves you"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">model</span>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="rounded border border-ink-600 bg-ink-850 px-2 py-1.5 text-xs text-slate-200 focus:border-accent-500 focus:outline-none"
          >
            {models.length === 0 && <option value="">no models reported</option>}
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.ready}>
                {m.label}
                {m.ready ? '' : ' (not ready)'}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">max new tokens</span>
          <input
            type="number"
            min={1}
            max={64}
            value={maxNew}
            onChange={(e) => setMaxNew(Math.max(1, Number(e.target.value) || 1))}
            className="tnum rounded border border-ink-600 bg-ink-850 px-2 py-1.5 text-xs text-slate-200 focus:border-accent-500 focus:outline-none"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">
            temperature{' '}
            <span className="value text-slate-400">{greedy ? 'greedy' : temperature.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="accent-accent-500"
          />
          <span className="text-[10px] text-slate-600">0 selects argmax</span>
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">
            top-k{' '}
            <span className="value text-slate-400">{useTopK ? topK : 'off'}</span>
          </span>
          <input
            type="range"
            min={1}
            max={200}
            step={1}
            value={topK}
            disabled={!useTopK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="accent-accent-500 disabled:opacity-40"
          />
          <span className="flex items-center gap-1.5 text-[10px] text-slate-600">
            <input
              type="checkbox"
              checked={useTopK}
              onChange={(e) => setUseTopK(e.target.checked)}
              className="accent-accent-500"
            />
            filter the tail
          </span>
        </label>

        <label className="flex items-center gap-2 self-end pb-1 text-[11px] text-slate-400">
          <input
            type="checkbox"
            checked={withTrace}
            onChange={(e) => setWithTrace(e.target.checked)}
            className="accent-accent-500"
          />
          send traces
          <span className="text-[10px] text-slate-600">
            off ≈ 4.5× faster, but nothing to visualize
          </span>
        </label>

        <div className="flex items-end gap-2">
          <button
            type="button"
            className="btn btn-active"
            disabled={running || !modelId || prompt.trim().length === 0}
            onClick={() => {
              void live.generate({
                model_id: modelId,
                prompt,
                max_new_tokens: maxNew,
                temperature,
                top_k: useTopK ? topK : null,
                seed: null,
                trace: withTrace,
              });
            }}
          >
            {running ? 'generating…' : 'generate'}
          </button>
          {running && (
            <button type="button" className="btn" onClick={live.stop}>
              stop
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
