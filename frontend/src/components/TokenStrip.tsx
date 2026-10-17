/**
 * The token sequence, with each sampled token linked to its generation step.
 */

import { useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

import {
  useTraceStore,
  deriveDisplayTokens,
  selectCurrentTrace,
} from '@/store/useTraceStore';

export function TokenStrip() {
  const steps = useTraceStore((s) => s.steps);
  const currentStep = useTraceStore((s) => s.currentStep);
  const trace = useTraceStore(selectCurrentTrace);
  const queryPosition = useTraceStore((s) => s.queryPosition);
  const selectQuery = useTraceStore((s) => s.selectQuery);
  const selectStep = useTraceStore((s) => s.selectStep);
  const status = useTraceStore((s) => s.status);

  const tokens = useMemo(
    () => deriveDisplayTokens(useTraceStore.getState()),
    [steps, currentStep, trace],
  );
  const promptLength = tokens.filter((token) => token.generatedStep === null).length;
  const contextLength =
    trace?.input_token_ids.length ?? promptLength + Math.max(0, currentStep);

  return (
    <TokenStripView
      tokens={tokens}
      contextLength={contextLength}
      promptLength={promptLength}
      currentStep={currentStep}
      queryPosition={queryPosition}
      streaming={status === 'running'}
      onSelectQuery={selectQuery}
      onSelectStep={selectStep}
    />
  );
}

export function TokenStripView({
  tokens,
  contextLength,
  promptLength,
  currentStep,
  queryPosition,
  streaming,
  onSelectQuery,
  onSelectStep,
}: {
  tokens: { piece: string; isNew: boolean; generatedStep: number | null }[];
  contextLength: number;
  promptLength: number;
  currentStep: number;
  queryPosition: number;
  streaming: boolean;
  onSelectQuery: (position: number) => void;
  onSelectStep: (step: number) => void;
}) {
  const contextStart = Math.max(0, promptLength + Math.max(0, currentStep) - contextLength);

  return (
    <section className="panel" aria-label="Token sequence">
      <header className="panel-header">
        <h2 className="panel-title">Tokens</h2>
        <span className="value text-[11px] text-slate-500">
          {contextLength} in context
          {tokens.length > promptLength ? ` · ${tokens.length - promptLength} generated` : ''}
          {streaming && <span className="ml-2 text-accent-500">streaming…</span>}
        </span>
      </header>

      <div className="flex flex-wrap items-center gap-1 px-3 py-3">
        {tokens.length === 0 && (
          <p className="text-xs text-slate-500">
            run a generation, or load a trace in replay mode
          </p>
        )}

        <AnimatePresence initial={false}>
          {tokens.map((token, i) => {
            const queryIndex = i - contextStart;
            const isContext =
              currentStep >= 0 && queryIndex >= 0 && queryIndex < contextLength;
            const selected =
              token.generatedStep === null &&
              isContext &&
              queryIndex === queryPosition;
            return (
              <motion.span
                key={`${i}-${token.piece}`}
                initial={{ opacity: 0, scale: 0.8, y: 4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.22, ease: 'easeOut' }}
              >
                <button
                  type="button"
                  disabled={!isContext && token.generatedStep === null}
                  onClick={() => {
                    if (token.generatedStep !== null) onSelectStep(token.generatedStep);
                    else if (isContext) onSelectQuery(queryIndex);
                  }}
                  title={
                    token.generatedStep !== null
                      ? `generated at step ${token.generatedStep + 1} — click to inspect that sampling step`
                      : isContext
                        ? `query position ${queryIndex} — click to focus attention and the logit lens here`
                        : 'outside the current forward-pass context'
                  }
                  className={[
                    'tnum rounded border px-1.5 py-1 text-xs transition-colors',
                    token.isNew
                      ? 'border-accent-500 bg-accent-500/20 font-semibold text-accent-300'
                      : selected
                        ? 'border-accent-500/70 bg-ink-700 text-accent-400'
                        : 'border-ink-600 bg-ink-800 text-slate-300',
                    isContext || token.generatedStep !== null
                      ? 'cursor-pointer hover:border-accent-500/60'
                      : 'cursor-default',
                  ].join(' ')}
                >
                  {visible(token.piece)}
                </button>
              </motion.span>
            );
          })}
        </AnimatePresence>
      </div>

      {tokens.some((token) => token.generatedStep !== null) && (
        <p className="border-t border-ink-700 px-3 py-1.5 text-[11px] text-slate-500">
          Click a generated token to inspect its step; click a context token to select the
          attention query and logit-lens position.
        </p>
      )}
    </section>
  );
}

/** Make GPT-2's whitespace markers visible without hiding the shape of a word. */
function visible(piece: string): string {
  return piece
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t')
    .replace(/ /g, '·');
}
