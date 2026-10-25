"""Settings: model-size configuration and the dependency-free ``.env`` reader.

Everything here is config-only: no model is built and nothing is downloaded, so
these tests stay fast and offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.settings import (
    MODEL_ID_OPENAI,
    Settings,
    _load_dotenv,
    _model_size_from_env,
    get_settings,
    reset_settings_cache,
)
from gpt2 import (
    ALLOWED_MODEL_SIZES,
    GPT_CONFIG_124M,
    GPT_CONFIG_355M,
    MODEL_CONFIGS,
    get_model_config,
)


def _settings(**overrides) -> Settings:
    kwargs = dict(
        checkpoint_dir=Path("checkpoints"),
        scratch_checkpoint=Path("scratch.pt"),
        text_file=Path("the-verdict.txt"),
        device="cpu",
        use_small_model=False,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


# -- model size -------------------------------------------------------------- #


def test_default_size_is_124m_and_keeps_the_legacy_id():
    settings = _settings()

    assert settings.model_size == "124M"
    # The constant every test, script and request default refers to must keep
    # matching what the registry actually uses at the default size.
    assert settings.openai_model_id == MODEL_ID_OPENAI == "openai-124m"
    assert settings.openai_model_config is GPT_CONFIG_124M


@pytest.mark.parametrize("size", ALLOWED_MODEL_SIZES)
def test_config_and_id_follow_the_size(size):
    settings = _settings(model_size=size)

    assert settings.openai_model_config == get_model_config(size)
    assert settings.openai_model_id == f"openai-{size.lower()}"
    assert size in settings.openai_model_label


def test_larger_sizes_get_the_published_architectures():
    assert _settings(model_size="355M").openai_model_config == GPT_CONFIG_355M
    assert GPT_CONFIG_355M["n_layers"] == 24
    assert MODEL_CONFIGS["774M"]["n_layers"] == 36
    assert MODEL_CONFIGS["1558M"]["n_layers"] == 48


def test_every_allowed_size_has_a_config():
    assert set(MODEL_CONFIGS) == set(ALLOWED_MODEL_SIZES)


def test_unknown_size_is_rejected_by_get_model_config():
    with pytest.raises(ValueError, match="not in"):
        get_model_config("999M")


def test_size_env_var_selects_the_config(monkeypatch):
    monkeypatch.setenv("GPT2_MODEL_SIZE", "774M")
    reset_settings_cache()
    try:
        settings = get_settings()
    finally:
        reset_settings_cache()

    assert settings.model_size == "774M"
    assert settings.openai_model_config == MODEL_CONFIGS["774M"]


def test_size_env_var_is_trimmed(monkeypatch):
    monkeypatch.setenv("GPT2_MODEL_SIZE", "  355M  ")

    assert _model_size_from_env() == "355M"


def test_empty_size_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GPT2_MODEL_SIZE", "")

    assert _model_size_from_env() == "124M"


def test_invalid_size_env_var_fails_fast(monkeypatch):
    monkeypatch.setenv("GPT2_MODEL_SIZE", "999M")
    reset_settings_cache()
    try:
        with pytest.raises(ValueError, match="GPT2_MODEL_SIZE"):
            get_settings()
    finally:
        reset_settings_cache()


# -- .env loading ------------------------------------------------------------ #


def test_dotenv_sets_vars_the_environment_does_not_have(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "export GPT2_MODEL_SIZE=355M\n"
        "GPT2_DEVICE='cuda:0'\n"
        'GPT2_CORS_ORIGINS="http://a.test"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("GPT2_MODEL_SIZE", raising=False)
    monkeypatch.delenv("GPT2_DEVICE", raising=False)
    monkeypatch.delenv("GPT2_CORS_ORIGINS", raising=False)

    _load_dotenv(env_file)

    assert os.environ["GPT2_MODEL_SIZE"] == "355M"
    assert os.environ["GPT2_DEVICE"] == "cuda:0"
    assert os.environ["GPT2_CORS_ORIGINS"] == "http://a.test"


def test_dotenv_never_overrides_the_real_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GPT2_DEVICE=cuda\n", encoding="utf-8")
    monkeypatch.setenv("GPT2_DEVICE", "cpu")

    _load_dotenv(env_file)

    assert os.environ["GPT2_DEVICE"] == "cpu"


def test_dotenv_ignores_malformed_lines(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("not-an-assignment\n=novalue\nGPT2_DEVICE=cpu\n", encoding="utf-8")
    monkeypatch.delenv("GPT2_DEVICE", raising=False)

    _load_dotenv(env_file)  # must not raise

    assert os.environ["GPT2_DEVICE"] == "cpu"


def test_dotenv_on_a_missing_file_is_a_no_op(tmp_path):
    _load_dotenv(tmp_path / "does-not-exist.env")  # must not raise
