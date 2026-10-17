/**
 * The horizontal forward-pass pipeline.
 *
 * Tokens → Embeddings → Block 1..n → Final Norm → Logits → Sample.
 *
 * Nodes are laid out from `pipelineStages(trace)`, which derives the block count
 * from the trace rather than assuming 12. Activity animates left to right while
 * a run is streaming and the token sampled at the end of the step has not yet
 * been folded back into the context, which is exactly when the forward pass is
 * actually happening. When idle the sweep stops, because nothing is computing.
 */

import { AnimatePresence, motion } from 'framer-motion';

import { formatFixed, formatPercent } from '@/lib/format';
import { useTraceStore, selectCurrentStep, selectCurrentTrace } from '@/store/useTraceStore';
import { pipelineStages, type PipelineStage, type Trace } from '@/types/trace';

import { BlockNode } from './BlockNode';

export function Pipeline() {
  const trace = useTraceStore(selectCurrentTrace);
  const step = useTraceStore(selectCurrentStep);
  const status = useTraceStore((s) => s.status);
  const steps = useTraceStore((s) => s.steps);
  const maxNewTokens = useTraceStore((s) => s.maxNewTokens);
  const selectBlock = useTraceStore((s) => s.selectBlock);

  const stages = pipelineStages(trace);
  const nBlocks = trace ? trace.blocks.length : 0;

  // The sweep is the "a forward pass is running right now" signal. It runs while
  // a step is still being produced, and holds once the generation is complete.
  const running = status === 'running' || status === 'loading';
  const sweeping = running && step !== null && step.finished !== true;

  // Highlight the stage the current step's data has just reached: the sample
  // node at the end, or the last block when there is no sampling step (a plain
  // `/api/trace` run has `sampling === null`).
  const activeKey = trace?.sampling ? 'sample' : `block-${Math.max(0, nBlocks - 1)}`;

  return (
    <section className="panel" aria-label="Forward pass pipeline">
      <header className="panel-header">
        <h2 className="panel-title">Pipeline</h2>
        <PipelineSummary trace={trace} />
      </header>

      <div className="overflow-x-auto px-3 py-4">
        <div className="relative min-w-[900px]">
          {/* Rail behind the nodes. */}
          <div className="absolute left-0 right-0 top-[26px] h-px bg-ink-700" aria-hidden />

          {sweeping && <SweepTrack stageCount={stages.length} />}

          <ol className="relative flex items-start gap-1">
            {stages.map((stage) => (
              <li key={stage.key} className="flex flex-1 flex-col items-center">
                <StageNode
                  stage={stage}
                  trace={trace}
                  active={stage.key === activeKey}
                  onSelect={selectBlock}
                />
              </li>
            ))}
          </ol>
        </div>
      </div>

      <AnimatePresence>
        {step && (
          <motion.p
            key={step.step}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="border-t border-ink-700 px-3 py-2 text-xs text-slate-400"
          >
            step{' '}
            <span className="value text-slate-200">{step.step + 1}</span> of{' '}
            <span className="value text-slate-200">{maxNewTokens ?? steps.length}</span> · context{' '}
            <span className="value text-slate-200">{trace?.input_token_ids.length ?? 0}</span> tokens ·
            sampled{' '}
            <span className="value text-accent-400">{JSON.stringify(step.piece)}</span>
            {trace?.sampling && !trace.sampling.was_argmax && (
              <>
                {' '}
                at{' '}
                <span className="value text-slate-200">
                  {formatPercent(trace.sampling.chosen.p_final, 1)}
                </span>{' '}
                after temperature {formatFixed(trace.sampling.temperature, 2)}
              </>
            )}
            {trace?.sampling?.temperature === 0 && <> (argmax: no draw)</>}
          </motion.p>
        )}
      </AnimatePresence>
    </section>
  );
}

/**
 * The travelling highlight.
 *
 * A single element whose `left` is animated from 0% to 100% across the rail,
 * with the duration proportional to the stage count so the perceived speed
 * stays constant however many blocks the model has.
 */
function SweepTrack({ stageCount }: { stageCount: number }) {
  const duration = Math.max(1.6, stageCount * 0.16);
  return (
    <div className="pointer-events-none absolute inset-x-0 top-[26px] h-px overflow-visible" aria-hidden>
      <motion.div
        className="absolute -top-[3px] h-[7px] w-[7px] rounded-full bg-accent-400 shadow-[0_0_10px_2px_rgba(45,212,191,0.55)]"
        initial={{ left: '0%', opacity: 0 }}
        animate={{ left: '100%', opacity: [0, 1, 1, 0] }}
        transition={{ duration, ease: 'linear', repeat: Infinity, repeatDelay: 0.35 }}
      />
      <motion.div
        className="absolute top-0 h-px bg-gradient-to-r from-accent-500/70 to-transparent"
        initial={{ left: '0%', width: '0%', opacity: 0 }}
        animate={{ left: '0%', width: '18%', opacity: [0, 0.9, 0.9, 0] }}
        transition={{ duration, ease: 'linear', repeat: Infinity, repeatDelay: 0.35 }}
      />
    </div>
  );
}

function StageNode({
  stage,
  trace,
  active,
  onSelect,
}: {
  stage: PipelineStage;
  trace: Trace | null;
  active: boolean;
  onSelect: (blockIndex: number | null) => void;
}) {
  // Blocks are interactive; the other stages are informational only.
  if (stage.kind === 'block') {
    return (
      <BlockNode
        stage={stage}
        block={trace && stage.blockIndex !== undefined ? trace.blocks[stage.blockIndex] : undefined}
        active={active}
        onSelect={() => onSelect(stage.blockIndex ?? null)}
      />
    );
  }

  return (
    <motion.div
      className="flex w-full flex-col items-center gap-1.5"
      animate={{ opacity: stage.ready ? 1 : 0.35 }}
      transition={{ duration: 0.25 }}
    >
      <div
        className={[
          'flex h-[34px] w-[34px] items-center justify-center rounded-full border text-[11px] font-bold',
          stage.ready
            ? 'border-ink-600 bg-ink-800 text-slate-300'
            : 'border-dashed border-ink-600 bg-transparent text-ink-500',
          active && stage.ready ? 'ring-2 ring-accent-500/60' : '',
        ].join(' ')}
      >
        {stage.kind === 'sample' && stage.ready ? (
          <SamplingBadge trace={trace} />
        ) : (
          STAGE_GLYPH[stage.kind]
        )}
      </div>
      <span className="text-center text-[10px] leading-tight text-slate-400">{stage.label}</span>
      <StageCaption stage={stage} trace={trace} />
    </motion.div>
  );
}

const STAGE_GLYPH: Record<string, string> = {
  tokens: 'T',
  embeddings: 'E',
  final_norm: 'N',
  logits: 'L',
  sample: 'S',
};

function SamplingBadge({ trace }: { trace: Trace | null }) {
  const sampling = trace?.sampling;
  if (!sampling) return <span>S</span>;
  return (
    <span className="value text-[10px] text-accent-400">{formatPercent(sampling.chosen.p_final, 0)}</span>
  );
}

/** The one number that makes the stage worth looking at. */
function StageCaption({ stage, trace }: { stage: PipelineStage; trace: Trace | null }) {
  if (!trace) return null;
  switch (stage.kind) {
    case 'tokens':
      return (
        <span className="value text-[10px] text-slate-500">
          {trace.input_token_ids.length} ids
        </span>
      );
    case 'embeddings': {
      const norm = trace.emb_sum.norms[trace.emb_sum.norms.length - 1];
      return <span className="value text-[10px] text-slate-500">‖{formatFixed(norm, 2)}</span>;
    }
    case 'final_norm': {
      const norm = trace.final_norm.norms[trace.final_norm.norms.length - 1];
      return <span className="value text-[10px] text-slate-500">‖{formatFixed(norm, 2)}</span>;
    }
    case 'logits':
      return (
        <span className="value text-[10px] text-slate-500">
          ±{formatFixed(trace.logits.max, 1)}
        </span>
      );
    default:
      return null;
  }
}

function PipelineSummary({ trace }: { trace: Trace | null }) {
  if (!trace) {
    return <span className="text-xs text-slate-500">no trace yet</span>;
  }
  const { n_layers, n_heads, emb_dim, context_length } = trace.config;
  return (
    <span className="value text-[11px] text-slate-500">
      {trace.model_id} · {n_layers}L/{n_heads}H/{emb_dim}d · ctx {context_length}
    </span>
  );
}
