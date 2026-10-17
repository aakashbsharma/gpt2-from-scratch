"""Train the small "scratch-10ep" model and save it for the registry.

Reproduces the notebook's from-scratch run on *The Verdict*: AdamW with
``lr=4e-4`` and ``weight_decay=0.1``, a 75/25 split, and per-epoch samples.

Usage::

    python -m scripts.train_small                     # small config (fast)
    python -m scripts.train_small --config 124m       # the notebook's real size
    python -m scripts.train_small --epochs 10 --device cuda

The validation loader uses ``drop_last=False``.

Why that matters: at the notebook's context length of 1024 the validation split
is only ~1281 tokens, which yields a *single* sequence. With ``batch_size=2`` and
``drop_last=True`` the loader therefore has **zero** batches, and the loss helper
returns ``nan`` -- which is exactly the ``Validation Loss : nan`` the notebook
printed. Keeping the final partial batch gives one real batch and a finite
validation loss.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):  # allow `python scripts/train_small.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import BACKEND_DIR, get_settings  # noqa: E402
from gpt2 import (  # noqa: E402
    GPT_CONFIG_124M,
    GPT_CONFIG_SMALL,
    GPTModel,
    generate_text_simple,
    get_device,
    get_tokenizer,
)


class TextChunkDataset(Dataset):
    """Overlapping ``(input, input_shifted_by_one)`` windows over a token stream."""

    def __init__(self, ids: list[int], max_length: int, stride: int) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for i in range(0, len(ids) - max_length, stride):
            self.input_ids.append(torch.tensor(ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(ids[i + 1 : i + max_length + 1]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def make_loader(
    ids: list[int],
    *,
    batch_size: int,
    max_length: int,
    stride: int,
    drop_last: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = TextChunkDataset(ids, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(loader: DataLoader, model, device) -> float:
    """Mean loss over a loader. Returns ``nan`` only when it is genuinely empty."""
    total, n = 0.0, 0
    for input_batch, target_batch in loader:
        total += calc_loss_batch(input_batch, target_batch, model, device).item()
        n += 1
    if n == 0:
        return float("nan")
    return total / n


def safe_print(text: str) -> None:
    """Print text that may contain bytes the console codec cannot encode.

    Windows consoles default to cp1252, and a randomly initialised model happily
    emits replacement characters that raise UnicodeEncodeError mid-run.
    """
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("small", "124m"), default="small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--text-file", default=str(settings.text_file))
    parser.add_argument("--out", default=str(settings.scratch_checkpoint))
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--stride-ratio", type=float, default=0.5)
    parser.add_argument("--sample-tokens", type=int, default=40)
    args = parser.parse_args(argv)

    cfg = GPT_CONFIG_SMALL if args.config == "small" else GPT_CONFIG_124M
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    text_path = Path(args.text_file)
    if not text_path.is_file():
        print(f"error: training corpus not found at {text_path}", file=sys.stderr)
        print("       set --text-file or GPT2_TEXT_FILE", file=sys.stderr)
        return 1
    text = text_path.read_text(encoding="utf-8")

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(text)

    split = int(args.train_ratio * len(text))
    train_ids = tokenizer.encode(text[:split])
    val_ids = tokenizer.encode(text[split:])
    print(f"corpus  : {text_path} ({len(text)} chars, {len(token_ids)} tokens)")
    print(f"config  : {args.config} {cfg}")
    print(f"device  : {device}")
    print(f"train   : {len(train_ids)} tokens")
    print(f"val     : {len(val_ids)} tokens")

    max_length = cfg["context_length"]
    stride = max(1, int(max_length * args.stride_ratio))

    train_loader = make_loader(
        train_ids, batch_size=args.batch_size, max_length=max_length,
        stride=stride, drop_last=True, shuffle=True,
    )
    # drop_last=False is the fix for the notebook's nan validation loss: the
    # validation split is small enough to produce a final partial batch, and
    # discarding it leaves the loader empty.
    val_loader = make_loader(
        val_ids, batch_size=args.batch_size, max_length=max_length,
        stride=stride, drop_last=False, shuffle=False,
    )

    if len(train_loader) == 0:
        print("error: training loader is empty; reduce --batch-size or context", file=sys.stderr)
        return 1
    print(f"loaders : {len(train_loader)} train batches, {len(val_loader)} val batches")
    if len(val_loader) == 0:
        print(
            "warning: validation loader is still empty; validation loss will be nan",
            file=sys.stderr,
        )

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(f"params  : {model.num_parameters():,}")
    print(f"epochs  : {args.epochs}\n")

    seed_context = torch.tensor([tokenizer.encode("Every effort moves you")], device=device)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1

        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device)
            val_loss = calc_loss_loader(val_loader, model, device)
            sample_ids = generate_text_simple(
                model, seed_context, args.sample_tokens, model.context_length
            )
        sample = tokenizer.decode(sample_ids[0].tolist()).replace("\n", " ")

        flag = "" if val_loss == val_loss else "   <-- STILL NAN"
        print(
            f"epoch {epoch:>2}/{args.epochs}  train {total / max(n,1):.4f}  "
            f"train_eval {train_loss:.4f}  val {val_loss:.4f}{flag}"
        )
        safe_print(f"           sample: {sample}")

    elapsed = time.time() - t0

    if val_loader and val_loss != val_loss:
        print("\nerror: validation loss is still nan", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "epochs": args.epochs,
            "train_loss": total / max(n, 1),
            "val_loss": val_loss,
            "text_file": str(text_path),
            "seed": args.seed,
        },
        out,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"\ntrained in {elapsed:.1f}s")
    print(f"saved {out} ({size_mb:.1f} MB)")
    print(f"final train loss {total / max(n,1):.4f}  val loss {val_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())