/**
 * Formatting and small numeric helpers shared by the visualization components.
 *
 * These exist so that "how many decimals" is decided in one place: trace values
 * arrive rounded to 4 decimals, and showing all four is noise, so the helpers
 * trim trailing zeros instead of inventing precision.
 */

/** Scientific notation below this magnitude, where 4 decimals read as `0.0000`. */
const SCI_BELOW = 1e-4;

/** Fixed-decimal value, e.g. `formatFixed(0.18359, 2) === "0.18"`. */
export function formatFixed(value: number, decimals: number): string {
  if (!Number.isFinite(value)) return '—';
  return value.toFixed(decimals);
}

/**
 * Compact number for dense tables: fixed notation normally, scientific notation
 * when the value is too small for fixed notation to carry information.
 */
export function formatCompact(value: number, decimals = 4): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs !== 0 && abs < SCI_BELOW) return value.toExponential(1);
  // `toFixed` then strip trailing zeros, so 0.5000 renders as 0.5.
  const fixed = value.toFixed(decimals);
  return fixed.includes('.') ? fixed.replace(/0+$/, '').replace(/\.$/, '') : fixed;
}

/** Percentage from a 0..1 probability. */
export function formatPercent(prob: number, decimals = 1): string {
  if (!Number.isFinite(prob)) return '—';
  return `${(prob * 100).toFixed(decimals)}%`;
}

/** `null` top-k means "no filter", which is worth saying explicitly. */
export function formatTopK(topK: number | null): string {
  return topK === null ? 'off' : String(topK);
}

/** `temperature <= 0` selects argmax, so display it as greedy. */
export function formatTemperature(temperature: number): string {
  return temperature <= 0 ? 'greedy' : temperature.toFixed(2);
}


