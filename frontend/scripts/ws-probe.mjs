/**
 * Probe the generation WebSocket the way the browser does, through the Vite dev
 * proxy.
 *
 * `backend/scripts/ws_probe.py` proves the server side works. This isolates the
 * other half of the path — the dev-server proxy, and the exact frame the
 * frontend sends — so a hang can be attributed to one or the other instead of
 * guessed at.
 *
 * Uses Node's global WebSocket, so there is no dependency and the handshake is
 * the same one the browser performs.
 *
 *   node scripts/ws-probe.mjs [url] [timeoutSeconds]
 */

const URL = process.argv[2] ?? 'ws://localhost:5174/api/ws/generate';
const TIMEOUT_S = Number(process.argv[3] ?? 45);

/** Byte-identical to `buildGenerateRequest` in src/lib/api.ts. */
const request = {
  type: 'generate',
  request_id: 'node-probe-1',
  model_id: 'openai-124m',
  prompt: 'Every effort moves you',
  max_new_tokens: 3,
  temperature: 0.8,
  top_k: 50,
  seed: 193,
  trace: true,
};

const started = Date.now();
const at = () => `[${((Date.now() - started) / 1000).toFixed(2).padStart(6)}s]`;

let frames = 0;
let sawOpen = false;

const ws = new WebSocket(URL);

const timer = setTimeout(() => {
  console.log(`${at()} TIMEOUT after ${TIMEOUT_S}s: open=${sawOpen} frames=${frames}`);
  process.exit(1);
}, TIMEOUT_S * 1000);

ws.onopen = () => {
  sawOpen = true;
  console.log(`${at()} open — sending request`);
  ws.send(JSON.stringify(request));
};

ws.onmessage = (event) => {
  frames += 1;
  const msg = JSON.parse(event.data);
  if (msg.type === 'step') {
    console.log(
      `${at()} step ${msg.step} piece=${JSON.stringify(msg.piece)} ` +
        `finished=${msg.finished} trace=${msg.trace !== null} bytes=${event.data.length}`,
    );
  } else {
    console.log(`${at()} ${msg.type}: ${JSON.stringify(msg).slice(0, 160)}`);
  }
};

ws.onerror = (event) => {
  console.log(`${at()} error: ${event.message ?? '(no detail — see the server log)'}`);
};

ws.onclose = (event) => {
  clearTimeout(timer);
  console.log(`${at()} close code=${event.code} reason=${JSON.stringify(event.reason)} frames=${frames}`);
  process.exit(frames > 0 ? 0 : 1);
};

console.log(`connecting to ${URL}`);
