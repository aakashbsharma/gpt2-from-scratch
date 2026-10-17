"""Export precomputed generation traces for static hosting.

Writes one JSON file per canned prompt to ``frontend/public/traces/<name>.json``
so the frontend can replay a full trace with no backend at all. This is the
"replay" mode in the UI, and it must produce byte-identical results to the live
WebSocket path, so both read from the same :func:`gpt2.generate_stream`.

Usage::

    python -m scripts.export_traces
    python -m scripts.export_traces --model openai-124m --max-new-tokens 8
    python -m scripts.export_traces --out ../frontend/public/traces

Each file has the shape::

    { name, model_id, prompt, params, text, token_ids, steps: [...], final_trace }

where every ``steps[i]`` is a ``WSStepMessage`` payload, identical to what
``/api/ws/generate`` would send. The frontend's replay hook therefore needs no
special case: it reads the same objects either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow `python scripts/export_traces.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.registry import ModelRegistry  # noqa: E402
from app.schemas import WSDoneMessage, WSStepMessage  # noqa: E402
from app.settings import BACKEND_DIR, PROJECT_DIR, get_settings  # noqa: E402
from gpt2 import generate_stream, get_tokenizer  # noqa: E402

# Canned prompts: two taken from the training corpus, two generic. The corpus
# ones demonstrate what 10 epochs on 5k tokens does and does not learn.
CANNED_PROMPTS: list[tuple[str, str]] = [
    ("effort", "Every effort moves you"),
    ("verdict", "The Verdict was a story about"),
    ("fox", "The quick brown fox jumps over the lazy"),
    ("life", "Life is full of"),
]

DEFAULT_OUT = PROJECT_DIR / "frontend" / "public" / "traces"


def export_one(
    registry: ModelRegistry,
    model_id: str,
    name: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    out_dir: Path,
    keep_traces: bool = True,
) -> tuple[Path, int, int]:
    """Run one real generation and write its traces to disk.

    Returns ``(path, num_steps, bytes_written)``.
    """
    with registry.inference(model_id) as entry:
        generator = generate_stream(
            entry.model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            trace=keep_traces,
            tokenizer=entry.tokenizer,
            device=entry.device,
            seed=seed,
            model_id=model_id,
        )

        steps: list[dict] = []
        final_trace = None
        last_ids: list[int] = []
        last_text = ""

        for step in generator:
            trace_payload = step.trace.to_dict() if step.trace is not None else None
            if trace_payload is not None and trace_payload["sampling"] is None:
                raise ValueError("A traced generation step is missing its sampling record")
            payload = WSStepMessage(
                request_id=name,
                step=step.sampling.step if step.sampling is not None else len(steps),
                token_id=step.token_id,
                piece=step.piece,
                token_ids=step.token_ids,
                text=step.text,
                finished=step.finished,
                trace=trace_payload,
            ).model_dump()
            steps.append(payload)
            if step.trace is not None:
                final_trace = payload["trace"]
            last_ids = step.token_ids
            last_text = step.text
            if step.finished:
                break

    document = {
        "name": name,
        "model_id": model_id,
        "prompt": prompt,
        "params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
        },
        "done": WSDoneMessage(
            request_id=name, token_ids=last_ids, text=last_text
        ).model_dump(),
        "steps": steps,
        "final_trace": final_trace,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    blob = json.dumps(document, allow_nan=False, indent=1)
    path.write_text(blob, encoding="utf-8")
    return path, len(steps), len(blob)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai-124m")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=193)
    parser.add_argument("--no-traces", action="store_true",
                        help="write sampling steps without per-step traces (small files)")
    parser.add_argument("--limit", type=int, default=0, help="export only N prompts (0 = all)")
    args = parser.parse_args(argv)

    registry = ModelRegistry().load()
    if not registry.has(args.model):
        errs = registry.load_errors.get(args.model)
        print(f"error: model {args.model!r} is not available", file=sys.stderr)
        if errs:
            print(f"       {errs}", file=sys.stderr)
        else:
            print("       run scripts/train_small.py, or check the OpenAI checkpoint",
                  file=sys.stderr)
        return 1

    prompts = CANNED_PROMPTS[: args.limit] if args.limit else CANNED_PROMPTS
    out_dir = Path(args.out)

    print(f"model   : {args.model}")
    print(f"device  : {registry.device}")
    print(f"params  : temperature={args.temperature} top_k={args.top_k} "
          f"seed={args.seed} max_new_tokens={args.max_new_tokens}")
    print(f"output  : {out_dir}\n")

    total = 0
    for name, prompt in prompts:
        path, n_steps, size = export_one(
            registry,
            args.model,
            name,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            out_dir=out_dir,
            keep_traces=not args.no_traces,
        )
        total += size
        print(f"  {name:<8} {n_steps:>2} steps  {size / 1e6:6.2f} MB  {path.name}")

    print(f"\n{total / 1e6:.2f} MB total in {out_dir}")
    if not args.no_traces and total > 40e6:
        print("note: files are large; consider --max-new-tokens 4 or --no-traces",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())