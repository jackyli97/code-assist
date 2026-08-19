import stat
from pathlib import Path
from typing import Any

import pytest

from ai_coding_assistant import config


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect config file locations at Path.home() into a tmp dir.

    The module-level CONFIG_DIR/CONFIG_ENV are computed from Path.home() at
    import time, so patching Path.home after the fact has no effect. Patch
    the resolved constants directly to mimic what a CLI invocation would see.
    """
    config_dir = tmp_path / ".config" / "code-assist"
    config_env = config_dir / ".env"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_ENV", config_env)
    return tmp_path


def test_load_config_reads_from_config_env(
    monkeypatch: pytest.MonkeyPatch, fake_home: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_load_dotenv(path: Path, override: bool) -> bool:
        calls["path"] = path
        calls["override"] = override
        return True

    monkeypatch.setattr(config, "load_dotenv", fake_load_dotenv)

    config.load_config()

    assert calls["path"] == config.CONFIG_ENV
    assert calls["override"] is False


def test_get_api_key_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")

    assert config.get_api_key() == "sk-test-123"


def test_get_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OpenRouter API key is not configured"):
        config.get_api_key()


def test_get_base_url_defaults_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    assert config.get_base_url() == "https://openrouter.ai/api/v1"


def test_get_base_url_honors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")

    assert config.get_base_url() == "https://example.test/v1"


def test_set_api_key_writes_file_with_secure_permissions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_home: Path,
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-secret-abc")

    config.set_api_key()

    assert config.CONFIG_ENV.read_text() == "OPENROUTER_API_KEY=sk-secret-abc\n"
    mode = stat.S_IMODE(config.CONFIG_ENV.stat().st_mode)
    assert mode == 0o600

    out = capsys.readouterr().out
    assert "OpenRouter API key configured." in out


def test_set_api_key_strips_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch, fake_home: Path
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "  sk-padded  ")

    config.set_api_key()

    assert config.CONFIG_ENV.read_text() == "OPENROUTER_API_KEY=sk-padded\n"


def test_set_api_key_rejects_empty_input(
    monkeypatch: pytest.MonkeyPatch, fake_home: Path
) -> None:
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "   ")

    with pytest.raises(ValueError, match="API key cannot be empty"):
        config.set_api_key()

    assert not config.CONFIG_ENV.exists()
