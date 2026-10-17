/**
 * Render `LogitLens` for one recorded step and compare each layer's top-1
 * against the payload, character for character.
 *
 * Written because the rendered view and the JSON can disagree in ways a
 * screenshot cannot settle. A leading space is exactly the kind of thing you
 * convince yourself you saw, and exactly the kind of thing that matters: `' you'`
 * and `'you'` are different GPT-2 tokens.
 *
 * Run: npm run inspect:lens -- <trace> <stepIndex>
 */

import { renderToStaticMarkup } from 'react-dom/server';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { LogitLens } from '@/components/LogitLens';
import type { TraceBundle } from '@/types/trace';

// The bundle runs in the same process as scripts/smoke.mjs, so `process.argv`
// still carries `--inspect=lens`. Positional args are whatever follows it.
const POSITIONAL = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const name = POSITIONAL[0] ?? 'effort';
const stepIndex = Number(POSITIONAL[1] ?? '3');

const doc = JSON.parse(
  readFileSync(join(process.cwd(), 'public', 'traces', `${name}.json`), 'utf8'),
) as TraceBundle;

const step = doc.steps[stepIndex];
if (!step?.trace) throw new Error(`${name} step ${stepIndex} has no trace`);

const trace = step.trace;
const html = renderToStaticMarkup(<LogitLens trace={trace} />);

/**
 * Strip tags and decode the entities React emits.
 *
 * `JSON.stringify` wraps each piece in real double quotes, which render as
 * `&quot;`. Decoding them is what makes a leading space visible instead of
 * silently lost — the whole point of the comparison.
 */
const text = html
  .replace(/<[^>]+>/g, '')
  .replace(/&quot;/g, '"')
  .replace(/&#x27;/g, "'")
  .replace(/&amp;/g, '&')
  .replace(/&lt;/g, '<')
  .replace(/&gt;/g, '>')
  .split('')
  .map((s) => s.trim())
  .filter(Boolean);

/** Quoted strings, in render order: each lens row emits top-1 then runner-up. */
const pieces = text.filter((t) => t.startsWith('"') && t.endsWith('"'));

console.log(`${name} step ${stepIndex}: T=${trace.input_token_ids.length}, position ${trace.input_token_ids.length - 1}`);
console.log(`quoted strings rendered: ${pieces.length} (expect ${trace.logit_lens.length * 2})`);
console.log('');
console.log('layer | payload top-1 | rendered  | match');
console.log('------|----------------|-----------|------');

let mismatches = 0;
trace.logit_lens.forEach((entry, i) => {
  const expected = `"${entry.top_pieces[0] ?? ''}"`;
  const rendered = pieces[i * 2] ?? '(missing)';
  const match = rendered === expected;
  if (!match) mismatches += 1;
  console.log(
    `L${String(entry.layer).padEnd(4)} | ${expected.padEnd(14)} | ${rendered.padEnd(9)} | ${match ? 'ok' : 'MISMATCH'}`,
  );
});

console.log('');
if (pieces.length !== trace.logit_lens.length * 2) {
  console.log(`NOTE: expected ${trace.logit_lens.length * 2} quoted strings, saw ${pieces.length}`);
}

// The header prose names the final row. It must use the same 0-based layer index
// the rows are labelled with, or the text points at a row that is not there —
// "Row 12" against a row labelled "→11" is the exact off-by-one this checks.
const lastLayer = trace.logit_lens.length - 1;
const prose = html.replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&amp;/g, '&');
const namesFinalRow = prose.includes(`→${lastLayer}`);
if (!namesFinalRow) mismatches += 1;
console.log(`header names the final row as →${lastLayer}: ${namesFinalRow ? 'ok' : 'MISMATCH'}`);
const claimsRowCount = /\bRow\s+\d+\b/.test(prose);
if (claimsRowCount) {
  console.log('NOTE: header still uses a 1-based "Row N" label against 0-based row labels');
}

console.log(`mismatches: ${mismatches}`);
