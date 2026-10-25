# gpt2-visualizer

A FastAPI + React app that visualizes the **real** forward pass of a from-scratch
GPT-2 (124M) implementation, token by token. Nothing in the UI is mocked: every
number comes from a tensor recorded during an actual forward pass.

![Demo — token streaming through the pipeline, attention heatmap and logit lens](docs/demo.gif)

<!--
GIF PLACEHOLDER

Record a short demo and drop it at `docs/demo.gif`:

  1. `docker compose up --build`, open http://localhost:8080
  2. Replay mode -> `effort`. Press play, then click Block 6.
  3. Switch Attention to `arcs`, pick head 9, click the last token.
  4. Show the LogitLens rows changing top-1 as depth increases.

Then resize to <=800px wide and under ~4 MB, or reference an external URL:

  ![Demo](https://github.com/user/repo/raw/main/docs/demo.gif)

Tools: `ffmpeg -i capture.mov -vf "fps=12,scale=800:-1" -loop 0 docs/demo.gif`
or asciinema/svg-term for a terminal recording.
-->

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 | `backend/gpt2` pure model package + docs | done |
| 2 | Tracing: `return_trace`, logit lens, PCA reduction, sampling step, `generate_stream` | done |
| 3 | Test suite (shape, numeric contract, HF parity) | done |
| 4 | FastAPI app, training + export scripts, compose | done |
| 5 | `frontend` React app | done |
| 6 | Docs, Docker, static build | done |

## Quick start

### Docker (recommended)

```bash
cd gpt2-visualizer
docker compose up --build
```

Then open **http://localhost:8080**.

First start takes ~3 minutes: the 500 MB GPT-2 124M checkpoint is downloaded and
loaded. Subsequent starts are ~30 s. Replay mode works as soon as the frontend is
up, before the backend finishes loading — the exported traces need no model.

To skip the download and run without the real weights:

```bash
GPT2_LOAD_OPENAI_MODEL=0 docker compose up --build
```

`openai-124m` will be listed with `ready: false` and the UI will say why.

### Choosing the model size

Which OpenAI checkpoint is downloaded is configurable, and the architecture
config is selected to match it, so the model and the weights can never disagree:

```bash
# local dev: a .env next to docker-compose.yml (or backend/.env) is read by the backend
echo "GPT2_MODEL_SIZE=355M" >> .env

# docker: compose interpolates that same .env file, or use the shell
GPT2_MODEL_SIZE=355M docker compose up --build
```

Valid sizes are `124M` (default), `355M`, `774M` and `1558M`. The registry id
follows the size — `openai-355m` — and so does the download, which grows fast:
~500 MB for 124M, ~1.5 GB for 355M, ~3 GB for 774M, ~6 GB for 1558M. Bigger
models are also slow to run per token; the traces were designed around 124M.

Copy `.env.example` to `.env` for the full list of backend settings
(`GPT2_DEVICE`, `GPT2_CHECKPOINT_DIR`, `GPT2_LOAD_OPENAI_MODEL`, ...). Values
already set in the environment always override what is in the file.

### Local development

Two terminals. Backend:

```bash
cd gpt2-visualizer/backend
python -m venv .venv && source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python -m scripts.export_traces     # optional, enables replay mode
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd gpt2-visualizer/frontend
npm install
npm run dev            # http://localhost:5173, proxies /api to :8000
```

### Optional: train the scratch model

`scratch-10ep` is a 13.3 M-parameter GPT-2 trained for 10 epochs on *The
Verdict*, ~35 s on CPU:

```bash
cd gpt2-visualizer/backend
python -m scripts.train_small
```

Until it runs, that model is listed with `ready: false` rather than being hidden,
so the UI can explain the absence instead of silently showing one model.

## Two ways to run it

### Live mode

Needs the backend. Generates over a WebSocket and streams one traced forward pass
per token:

```bash
docker compose up --build      # or the local dev setup above
```

Open the app, pick **Live**, set a prompt, press **Generate**. Each new token
appends to the context and the whole pipeline animates again over the longer
sequence. Each step also carries that token's real sampling decision — the
filtered candidate distribution, the temperature/top-k settings, and the draw —
which the Sampling panel visualizes. **Stop** closes the socket; the traces
already received stay on screen.

### Replay mode

Needs **no backend at all** — this is the static-hosting path. Plays the
pre-recorded traces in `frontend/public/traces/*.json`:

```bash
cd gpt2-visualizer/frontend
npm run dev            # Replay is the default mode
```

The bundles hold the same `WSStepMessage` payloads the live socket sends, so
`ModeToggle` swaps the transport and no component changes.

To regenerate them:

```bash
cd gpt2-visualizer/backend
python -m scripts.export_traces          # writes frontend/public/traces/*.json
```

> The traces are **gitignored** — they are 7.8 MB of generated JSON and the
> exporter needs the backend's dependencies and the 124M checkpoint. Run it once
> after cloning, or the replay buttons 404. `npm run build:static` checks for
> them and tells you this if they are missing.

### Building for static hosting

```bash
cd gpt2-visualizer/frontend
npm run build:static         # -> dist-static/
npm run preview:static       # serve it locally at :4173
```

Upload `dist-static/` to any static host. There is no server component: the
build has no WebSocket code and makes no `/api` calls, verified by grepping the
output. Only the mode toggle's **Replay** option is offered, and the comparison
panel is hidden because it needs two models from the registry.

> `preview:static` runs with proxying explicitly disabled. `vite preview`
> otherwise falls back to `server.proxy`, so with a backend running on `:8000`
> it would silently proxy `/api` and make a bundle that is *not* standalone look
> like it works. If you check `/api/models` while previewing and get the SPA
> fallback page back rather than JSON, that is the expected result.

For a subpath deploy (GitHub Pages project sites), set the base path in
`frontend/.env.static` — see `frontend/.env.static.example`.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            React app (Vite)                  │
                    │                                              │
                    │   ModeToggle ──┬── Live     ──▶ WebSocket     │
                    │                │                              │
                    │                └── Replay   ──▶ fetch        │
                    │                                     │        │
                    │                        useTraceStore (Zustand) │
                    │                                     │        │
                    │   ┌─────────────┬───────────────────┤        │
                    │   ▼             ▼                   ▼        │
                    │ Pipeline   BlockDetail        LogitLens       │
                    │ Tokens→…→Sample  LN→Attn→Add→LN→MLP→Add         │
                    │   │             │                                │
                    │   │        AttentionHeatmap / Arcs / HeadSelector│
                    │   │             │                                │
                    │   │   TokenStrip ──▶ Sampling ──▶ loop           │
                    │   │   Funnel · Draw · Stats · History            │
                    │   └─────────────┴────────────────────────────────┘
                    └──────────────┬──────────────────────────┬────────┘
                     /api/ws/generate                  /traces/*.json
                                   │                          │
                    ┌──────────────▼──────────┐    ┌──────────▼─────────┐
                    │  FastAPI (:8000)        │    │ exported bundles    │
                    │  registry → gpt2 model  │    │ (gitignored, 7.8MB) │
                    └──────────────┬──────────┘    └────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────────┐
                    │  backend/gpt2 — pure model package           │
                    │  hand-written attention (no F.sdpa, so the    │
                    │  T×T matrix actually exists to visualize)     │
                    │  forward(return_trace=True) → Trace           │
                    └─────────────────────────────────────────────┘
```

Data flows one way: both sources write `WSStepMessage`s into `useTraceStore`, and
every component reads the store. Neither component knows which mode is active,
which is why `ModeToggle` is the only place that changes.

Read [docs/architecture.md](docs/architecture.md) for the model and tracing
detail, and [docs/trace-schema.md](docs/trace-schema.md) for the JSON contract.

## Layout

```
gpt2-visualizer/
├── docs/                 architecture.md, trace-schema.md, demo.gif
├── notebooks/            the original notebook, untouched
├── docker-compose.yml    backend + frontend
├── backend/
│   ├── gpt2/             pure model package (no web code)
│   ├── app/              FastAPI: settings, registry, routes
│   ├── scripts/          train_small.py, export_traces.py, plus ws_probe.py
│   │                     and silent_ws_server.py (socket debugging helpers)
│   ├── tests/            parity, trace-shape, settings, API route tests
│   └── checkpoints/      gitignored weights
└── frontend/
    ├── src/types/        the JSON contract, mirrored from docs/trace-schema.md
    ├── src/store/        useTraceStore: steps, current step, selected block/head
    ├── src/lib/          api.ts (typed HTTP + WebSocket), env.ts, format.ts
    ├── src/hooks/        useLiveGeneration / useReplay / useModels / useCompareRun
    │                     liveImpl.ts vs staticLiveStub.ts for static builds
    ├── src/components/   Pipeline, BlockDetail, Attention*, LogitLens,
    │                     ComparePanel, sampling/ (Funnel, Draw, Stats, History)
    ├── scripts/          smoke.mjs, check-traces.mjs, ws-probe.mjs, inspect-lens
    ├── public/traces/    gitignored exported bundles
    └── Dockerfile        node build -> nginx, proxies /api
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness, device, models ready |
| `GET` | `/api/models` | Registry listing with `ready` flags |
| `POST` | `/api/tokenize` | GPT-2 BPE encoding |
| `POST` | `/api/trace` | One traced forward pass |
| `WS` | `/api/ws/generate` | Stream one trace per generated token, then `done` |

`POST /api/trace` runs one forward pass over explicit token ids with no
sampling, so its trace's `sampling` field is always `null`; the WebSocket path
is the one that records the sampling decision.

Two models are registered: `openai-124m` (real OpenAI weights; the id becomes
`openai-355m` etc. when `GPT2_MODEL_SIZE` is set) and `scratch-10ep`
(from `scripts/train_small.py`, listed as not-ready when its checkpoint is absent).

Interactive reference at http://localhost:8000/docs.

## Tests

```bash
cd gpt2-visualizer/backend
pytest                # 190 tests
pytest -m "not slow"   # 176 tests, ~40 s, no checkpoint needed
pytest -m slow         # 14 tests: real 124M weights + HuggingFace comparison
```

```bash
cd gpt2-visualizer/frontend
npm run smoke           # renders the real components against the real traces
npm run smoke:static    # same, against the static-build module graph
npm run inspect:lens -- effort 0
                        # renders one LogitLens step and diffs it vs the JSON
```

`smoke` asserts numeric invariants on all four recorded traces (`effort`,
`verdict`, `fox`, `life`) — the causal upper triangle is exactly `0.0`, the
deepest logit-lens entry equals `top_tokens[0]`, sampling candidates sum to 1
within float16 tolerance — then renders each component and checks the output.
It is not a browser test; see [frontend/README.md](frontend/README.md) for what
it does and does not cover.

If generation hangs, two probes split the blame between the two halves of the
socket path: `python -m scripts.ws_probe` speaks the frontend's exact frame to
the real backend, and `node scripts/ws-probe.mjs` does the same through the
Vite dev proxy. `python -m scripts.silent_ws_server 8001` is a server that
accepts and never answers, for exercising the client-side watchdog against a
real socket.

## Using the model directly

```python
from gpt2 import GPTModel, GPT_CONFIG_124M, get_tokenizer

tok = get_tokenizer()
model = GPTModel(GPT_CONFIG_124M).eval()
trace = model.forward(
    torch.tensor([tok.encode("Life is full of")]),
    return_trace=True, tokenizer=tok, model_id="scratch",
)
print(trace.top_pieces[:3])
```

Load the real OpenAI weights:

```python
from gpt2 import download_and_load_gpt2, load_weights_into_gpt

model = GPTModel(GPT_CONFIG_124M).eval()
_, params = download_and_load_gpt2("124M", "checkpoints/gpt2", reuse_existing=True)
load_weights_into_gpt(model, params)
```

Stream a generation with a trace per token:

```python
from gpt2 import generate_stream

for step in generate_stream(model, prompt="Every effort moves you",
                            max_new_tokens=5, temperature=0.9, top_k=50):
    print(step.step, repr(step.piece), step.trace.logit_lens[-1].top_pieces[0])
```

## Source

Derived from the from-scratch implementation in `notebooks/gpt2.ipynb`, itself
following *Build a Large Language Model (From Scratch)* by Sebastian Raschka.
