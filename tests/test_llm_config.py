# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from src.llm.explainer import DEFAULT_BASE_URL, DEFAULT_MODEL, get_llm_config


def _write_env(path: Path, key: str = "test-secret-value") -> None:
    path.write_text(
        "\n".join(
            (
                f"LLM_API_KEY={key}",
                "LLM_BASE_URL=https://api.deepseek.com/v1/chat/completions",
                "LLM_MODEL=deepseek-v4-flash",
            )
        ),
        encoding="utf-8",
    )


def test_dynamic_config_loads_explicit_env_path_from_any_cwd(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / "config" / ".env"
    env_path.parent.mkdir(parents=True)
    _write_env(env_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", env_path)

    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()

    monkeypatch.chdir(first_cwd)
    first = get_llm_config()
    monkeypatch.chdir(second_cwd)
    second = get_llm_config()

    for config in (first, second):
        assert config.configured is True
        assert config.base_url == "https://api.deepseek.com/v1/chat/completions"
        assert config.model == "deepseek-v4-flash"
        assert "test-secret-value" not in repr(config)


def test_config_is_reloaded_without_module_reimport(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    _write_env(env_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", env_path)

    assert get_llm_config().configured is True

    env_path.write_text("LLM_API_KEY=\n", encoding="utf-8")
    refreshed = get_llm_config()
    assert refreshed.configured is False
    assert refreshed.base_url == DEFAULT_BASE_URL
    assert refreshed.model == DEFAULT_MODEL


def test_explicit_env_path_argument_overrides_default(tmp_path, monkeypatch) -> None:
    custom_env = tmp_path / "custom.env"
    _write_env(custom_env, key="another-test-value")
    missing_default = tmp_path / "missing.env"
    monkeypatch.setattr("src.llm.explainer.DEFAULT_ENV_PATH", missing_default)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)

    config = get_llm_config(custom_env)
    assert config.configured is True
    assert config.env_path == custom_env
    assert "another-test-value" not in repr(config)
