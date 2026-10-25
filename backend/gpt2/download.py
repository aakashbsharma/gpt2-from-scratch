"""Download the original OpenAI GPT-2 checkpoint and convert it to NumPy.

Derived from ``gpt_download3.py``. Two changes from the notebook version:

* ``verify=True`` on the HTTPS request (the original disabled certificate
  verification, which emitted an ``InsecureRequestWarning`` on every file).
* TensorFlow is imported lazily inside the function that reads the checkpoint,
  so importing this module does not pay the ~5s TF startup cost and does not
  require TF at all if you already have a NumPy param file.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from tqdm import tqdm

from .config import ALLOWED_MODEL_SIZES

BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/models"

FILENAMES = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]


def download_file(
    url: str,
    destination: str,
    *,
    verify: bool = True,
    chunk_size: int = 1024,
    force: bool = False,
    reuse_existing: bool = False,
) -> str:
    """Stream ``url`` to ``destination``, skipping files already up to date.

    By default an existing file is still validated by comparing its size against
    the remote ``content-length``, which costs one HTTP request per file. That
    is the notebook's behaviour and it is the safe default.

    ``reuse_existing=True`` trusts any non-empty local file and issues **no**
    request at all. Use it when the checkpoint is already cached and network
    access is unavailable or slow (tests, offline runs); combine with
    ``force=True`` to re-download regardless.
    """
    if reuse_existing and not force:
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            print(f"File already present, reusing: {destination}")
            return destination

    try:
        response = requests.get(url, stream=True, verify=verify, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("content-length", 0))

        if not force and os.path.exists(destination):
            if file_size == os.path.getsize(destination):
                print(f"File already exists and is up-to-date: {destination}")
                return destination

        description = url.split("/")[-1]
        with tqdm(
            total=file_size,
            unit="iB",
            unit_scale=True,
            desc=description,
        ) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(
            f"Error downloading {url}: {e}"
        ) from e

    return destination


def download_and_load_gpt2(
    model_size: str,
    models_dir: str,
    *,
    verify: bool = True,
    force: bool = False,
    reuse_existing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download a GPT-2 checkpoint and return ``(settings, params)``.

    ``params`` is a nested dict keyed the way the original TF checkpoint is:
    ``wte``, ``wpe``, ``g``, ``b``, and ``blocks[i]`` with ``attn``/``mlp``/``ln_1``
    /``ln_2`` sub-dicts.
    """
    if model_size not in ALLOWED_MODEL_SIZES:
        raise ValueError(f"Model size not in {ALLOWED_MODEL_SIZES}")

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    for filename in FILENAMES:
        file_url = f"{BASE_URL}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(
            file_url, file_path, verify=verify, force=force, reuse_existing=reuse_existing
        )

    tf_ckpt_path = _latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def _latest_checkpoint(model_dir: str) -> str:
    """Locate the checkpoint prefix inside ``model_dir``."""
    import tensorflow as tf  # lazy: only needed to read a TF checkpoint

    return tf.train.latest_checkpoint(model_dir)


def load_gpt2_params_from_tf_ckpt(
    ckpt_path: str, settings: dict[str, Any]
) -> dict[str, Any]:
    """Flatten a TF checkpoint into a nested dict of squeezed NumPy arrays."""
    import tensorflow as tf  # lazy: see _latest_checkpoint

    params: dict[str, Any] = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = _squeeze(tf.train.load_variable(ckpt_path, name))

        variable_name_parts = name.split("/")[1:]  # skip the 'model/' prefix

        target_dict: Any = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[variable_name_parts[-1]] = variable_array

    return params


def _squeeze(a):
    import numpy as np

    return np.squeeze(a)