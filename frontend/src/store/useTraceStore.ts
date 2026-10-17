/**
 * Single source of truth for the visualization.
 *
 * Both data sources (`useLiveGeneration`, `useReplay`) write into this store
 * through the same two actions, so no component needs to know which mode it is
 * running in. It also holds the selection state that has to survive a step
 * change: the selected block, head, query position and open detail panel.
 */

import { create } from 'zustand';

import type {
  PipelineStage,
  SourceStatus,
  Trace,
  TraceParams,
  WSStepMessage,
} from '@/types/trace';
import { latestTrace, pipelineStages } from '@/types/trace';

export type Mode = 'live' | 'replay';

/** How far through a replayed generation playback has advanced. */
export interface ReplayMeta {
  name: string;
  params: TraceParams | null;
  /** Wall-clock ms between revealed steps. */
  intervalMs: number;
}

export interface TraceState {
  mode: Mode;
  status: SourceStatus;
  error: string | null;

  /** `WSStepMessage`s received so far, oldest first. Identical in both modes. */
  steps: WSStepMessage[];
  text: string;
  modelId: string;
  prompt: string;
  maxNewTokens: number | null;

  /** Bumped on every reset so sources can drop state from a previous run. */
  runId: number;
  /** Name of the loaded replay bundle, for the mode toggle's label. */
  replayName: string | null;
  replayMeta: ReplayMeta | null;

  /** Index into `steps` currently displayed. */
  currentStep: number;
  selectedBlock: number | null;
  selectedHead: number;
  /** Query row highlighted in the attention views. */
  queryPosition: number;
  /** Lens position selected from the token strip; null defaults the lens to the final position. */
  selectedPosition: number | null;
  /** Whether the block detail drawer is open. */
  detailOpen: boolean;

  // --- actions ------------------------------------------------------------ //
  setMode: (mode: Mode) => void;
  reset: (patch?: Partial<Pick<TraceState, 'mode' | 'status' | 'error' | 'modelId' | 'prompt' | 'replayName' | 'replayMeta'>>) => void;
  startRun: (init: { modelId: string; prompt: string; maxNewTokens?: number; replayName?: string; replayMeta?: ReplayMeta | null }) => void;
  pushStep: (step: WSStepMessage) => void;
  setStatus: (status: SourceStatus, error?: string | null) => void;
  setSteps: (steps: WSStepMessage[]) => void;

  selectBlock: (index: number | null) => void;
  selectHead: (head: number) => void;
  selectQuery: (position: number) => void;
  selectStep: (index: number) => void;
  setDetailOpen: (open: boolean) => void;
}

/**
 * Keep a selection inside the current trace.
 *
 * Stepping changes block count only when the model changes, but head count and
 * context length are per-trace, so a head chosen in a 12-head block would point
 * nowhere in another. Clamping here means callers never render an out-of-range
 * index.
 */
function traceAtStep(steps: WSStepMessage[], stepIndex: number): Trace | null {
  for (let index = Math.min(stepIndex, steps.length - 1); index >= 0; index -= 1) {
    if (steps[index]?.trace) return steps[index].trace;
  }
  return stepIndex < 0 ? latestTrace(steps) : null;
}

function clampSelections(
  state: TraceState,
  steps: WSStepMessage[],
  stepIndex: number,
): Partial<TraceState> {
  const trace = traceAtStep(steps, stepIndex);
  if (!trace) {
    return {
      selectedBlock: null,
      selectedHead: 0,
      queryPosition: 0,
      selectedPosition: null,
    };
  }
  const nHeads = trace.blocks[0]?.heads.length ?? 1;
  const T = trace.input_token_ids.length;
  return {
    selectedBlock:
      state.selectedBlock !== null && state.selectedBlock < trace.blocks.length
        ? state.selectedBlock
        : 0,
    selectedHead: clampInt(state.selectedHead, 0, nHeads - 1),
    queryPosition: clampInt(state.queryPosition, 0, T - 1),
    selectedPosition:
      state.selectedPosition === null
        ? null
        : clampInt(state.selectedPosition, 0, T - 1),
  };
}

function clampInt(value: number, lo: number, hi: number): number {
  if (hi < lo) return lo;
  return Math.min(hi, Math.max(lo, value));
}

const initialState = {
  mode: 'replay' as Mode,
  status: 'idle' as SourceStatus,
  error: null as string | null,
  steps: [] as WSStepMessage[],
  text: '',
  modelId: '',
  prompt: '',
  maxNewTokens: null,
  runId: 0,
  replayName: null as string | null,
  replayMeta: null as ReplayMeta | null,
  currentStep: -1,
  selectedBlock: 0,
  selectedHead: 0,
  queryPosition: 0,
  selectedPosition: null as number | null,
  detailOpen: false,
};

export const useTraceStore = create<TraceState>((set, get) => ({
  ...initialState,

  setMode: (mode) => {
    if (get().mode === mode) return;
    // Switching modes abandons any in-flight run; the new source calls
    // `startRun`, which resets everything downstream of that anyway.
    //
    // `runId` is bumped so a source still mid-flight can tell its run was
    // superseded. That covers the *socket*, but not a source whose timer is
    // still ticking: `useReplay` is mounted unconditionally (hooks cannot be
    // called conditionally), so its playback interval keeps firing after the
    // switch and pushes recorded steps into a live run. `useReplay` watches
    // `runId` for exactly this.
    set({
      mode,
      status: 'idle',
      error: null,
      steps: [],
      text: '',
      maxNewTokens: null,
      currentStep: -1,
      selectedPosition: null,
      runId: get().runId + 1,
    });
  },

  reset: (patch) =>
    set((state) => ({
      ...initialState,
      ...patch,
      mode: patch?.mode ?? state.mode,
      runId: state.runId + 1,
    })),

  startRun: ({ modelId, prompt, maxNewTokens, replayName, replayMeta }) =>
    set((state) => ({
      steps: [],
      text: '',
      error: null,
      status: 'loading',
      modelId,
      prompt,
      maxNewTokens: maxNewTokens ?? replayMeta?.params?.max_new_tokens ?? null,
      replayName: replayName ?? null,
      replayMeta: replayMeta ?? null,
      currentStep: -1,
      detailOpen: false,
      runId: state.runId + 1,
      // Re-select a sensible head/position for the new trace's shape.
      ...clampSelections({
        ...state,
        selectedBlock: 0,
        selectedHead: 0,
        queryPosition: 0,
        selectedPosition: null,
      }, [], -1),
    })),

  pushStep: (step) =>
    set((state) => {
      const steps = [...state.steps, step];
      const followsLatest =
        state.currentStep < 0 || state.currentStep === state.steps.length - 1;
      const currentStep = followsLatest ? steps.length - 1 : state.currentStep;
      return {
        steps,
        text: step.text || state.text,
        currentStep,
        status: step.finished ? 'done' : state.status,
        ...clampSelections(state, steps, currentStep),
      };
    }),

  /** Bulk-load a whole bundle at once (used when replay jumps to the end). */
  setSteps: (steps) =>
    set((state) => {
      const currentStep = steps.length - 1;
      return {
        steps,
        text: steps.length ? steps[currentStep].text : '',
        currentStep,
        ...clampSelections(state, steps, currentStep),
      };
    }),

  setStatus: (status, error = null) => set({ status, error }),

  selectBlock: (index) => set({ selectedBlock: index, detailOpen: index !== null }),
  selectHead: (head) => set({ selectedHead: head }),
  selectQuery: (position) =>
    set({ queryPosition: position, selectedPosition: position }),
  selectStep: (index) =>
    set((state) => {
      const currentStep =
        state.steps.length === 0 ? -1 : clampInt(index, 0, state.steps.length - 1);
      return {
        currentStep,
        ...clampSelections(state, state.steps, currentStep),
      };
    }),
  setDetailOpen: (detailOpen) => set({ detailOpen }),
}));

// --------------------------------------------------------------------------- //
// Selectors
// --------------------------------------------------------------------------- //

/** The step the UI is showing. */
export function selectCurrentStep(state: TraceState): WSStepMessage | null {
  if (state.currentStep < 0 || state.currentStep >= state.steps.length) return null;
  return state.steps[state.currentStep] ?? null;
}

/**
 * The trace backing the current view.
 *
 * Prefers the displayed step's own trace. When that step was sent without a
 * trace, use the nearest preceding traced step rather than showing future data.
 */
export function selectCurrentTrace(state: TraceState): Trace | null {
  const current = selectCurrentStep(state);
  if (current?.trace) return current.trace;
  return traceAtStep(state.steps, state.currentStep);
}

export function selectStages(state: TraceState): PipelineStage[] {
  return pipelineStages(selectCurrentTrace(state));
}

/**
 * The most probable next token at the last context position.
 *
 * The prediction the whole app is explaining, derived in one place: the trace's
 * own `top_tokens[0]`, which the backend asserts equals the deepest logit-lens
 * entry.
 *
 * Returns a fresh object, so — like `deriveDisplayTokens` — it must not be used
 * directly as a zustand selector. Callers wrap it in `useMemo`.
 */
export function deriveNextToken(state: TraceState): { piece: string; prob: number } | null {
  const trace = selectCurrentTrace(state);
  if (!trace) return null;
  return { piece: trace.top_pieces[0] ?? '', prob: trace.top_probs[0] ?? 0 };
}

/** The block the detail drawer is showing, or `null`. */
export function selectSelectedBlock(state: TraceState) {
  const trace = selectCurrentTrace(state);
  if (!trace || state.selectedBlock === null) return null;
  return trace.blocks[state.selectedBlock] ?? null;
}

/** The selected head of the selected block, or `null`. */
export function selectSelectedHead(state: TraceState) {
  const block = selectSelectedBlock(state);
  if (!block) return null;
  return block.heads[state.selectedHead] ?? block.heads[0] ?? null;
}



/**
 * On-screen token sequence: the trace's context, plus the token sampled from it.
 *
 * That trailing token is exactly what `TokenStrip` appends as the loop repeats,
 * so it is derived here rather than stored twice.
 *
 * **This is a plain derivation, not a zustand selector, and must not be passed
 * to `useTraceStore(...)`.** It builds a new array on every call, and React's
 * `useSyncExternalStore` calls the selector on every render to detect changes —
 * a fresh reference reads as "the store changed", so the component re-renders
 * forever and React throws. That bug shipped as a blank page and was invisible
 * to `npm run smoke`, because server rendering reads the store's initial
 * snapshot, which never changes.
 *
 * Callers wrap it in `useMemo` on a stable dependency. Same rule for
 * `selectNextToken` below, and for any future derivation added here.
 */
export function deriveDisplayTokens(state: TraceState): DisplayToken[] {
  const firstTrace = state.steps.find((step) => step.trace)?.trace ?? null;
  const promptLength =
    firstTrace?.input_token_ids.length ??
    Math.max(0, (state.steps[0]?.token_ids.length ?? 0) - 1);
  const promptPieces = firstTrace
    ? firstTrace.tokens.map((token) => token.piece)
    : Array.from({ length: promptLength }, () => '?');
  return [
    ...promptPieces.map((piece) => ({
      piece,
      isNew: false,
      generatedStep: null,
    })),
    ...state.steps.map((step, index) => ({
      piece: step.piece,
      isNew: index === state.currentStep,
      generatedStep: index,
    })),
  ];
}

/** One entry in the token strip. */
export interface DisplayToken {
  piece: string;
  /** True for the generated token associated with the selected step. */
  isNew: boolean;
  /** Index of the generation step that produced this token, or null for prompt tokens. */
  generatedStep: number | null;
}
