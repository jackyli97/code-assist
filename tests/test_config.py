import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from ai_coding_assistant import config
from ai_coding_assistant.models import ModelLookup


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

@pytest.fixture
def fake_lookup(monkeypatch: pytest.MonkeyPatch) -> Callable[[ModelLookup], None]:
    """Stub out the network-backed model lookup with a fixed result."""

    def install(result: ModelLookup) -> None:
        monkeypatch.setattr(config, "lookup_model", lambda _model: result)

    return install


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


def read_saved_config() -> dict[str, str | None]:
    """Read the config file the way load_config eventually will."""
    return dotenv_values(config.CONFIG_ENV)


def test_set_preferred_model_writes_model_and_context_limit(
    fake_lookup: Callable[[ModelLookup], None], fake_home: Path
) -> None:
    fake_lookup(ModelLookup(found=True, context_limit=200_000))

    config.set_preferred_model("anthropic/claude-haiku-4.5")

    assert read_saved_config() == {
        "OPENROUTER_MODEL": "anthropic/claude-haiku-4.5",
        "OPENROUTER_MODEL_CONTEXT_LIMIT": "200000",
    }
    assert stat.S_IMODE(config.CONFIG_ENV.stat().st_mode) == 0o600


def test_set_preferred_model_preserves_existing_api_key(
    monkeypatch: pytest.MonkeyPatch,
    fake_lookup: Callable[[ModelLookup], None],
    fake_home: Path,
) -> None:
    """Saving a model must not clobber the key -- both live in one file."""
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-secret-abc")
    config.set_api_key()

    fake_lookup(ModelLookup(found=True, context_limit=200_000))
    config.set_preferred_model("anthropic/claude-haiku-4.5")

    assert read_saved_config() == {
        "OPENROUTER_API_KEY": "sk-secret-abc",
        "OPENROUTER_MODEL": "anthropic/claude-haiku-4.5",
        "OPENROUTER_MODEL_CONTEXT_LIMIT": "200000",
    }


def test_set_api_key_preserves_existing_model_settings(
    monkeypatch: pytest.MonkeyPatch,
    fake_lookup: Callable[[ModelLookup], None],
    fake_home: Path,
) -> None:
    """Rotating the key must not drop the saved model."""
    fake_lookup(ModelLookup(found=True, context_limit=200_000))
    config.set_preferred_model("anthropic/claude-haiku-4.5")

    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-rotated")
    config.set_api_key()

    assert read_saved_config() == {
        "OPENROUTER_MODEL": "anthropic/claude-haiku-4.5",
        "OPENROUTER_MODEL_CONTEXT_LIMIT": "200000",
        "OPENROUTER_API_KEY": "sk-rotated",
    }


def test_set_preferred_model_drops_stale_context_limit(
    fake_lookup: Callable[[ModelLookup], None], fake_home: Path
) -> None:
    """A model with no known limit must not inherit the previous model's."""
    fake_lookup(ModelLookup(found=True, context_limit=200_000))
    config.set_preferred_model("anthropic/claude-haiku-4.5")

    fake_lookup(ModelLookup(found=True, context_limit=None))
    config.set_preferred_model("some/unmetered-model")

    assert read_saved_config() == {"OPENROUTER_MODEL": "some/unmetered-model"}

def test_set_preferred_model_preserves_existing_model_settings(
    monkeypatch: pytest.MonkeyPatch,
    fake_lookup: Callable[[ModelLookup], None],
    fake_home: Path,
) -> None:
    """Rotating the model must not drop the saved api key."""
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-rotated")
    config.set_api_key()

    fake_lookup(ModelLookup(found=True, context_limit=200_000))
    config.set_preferred_model("anthropic/claude-haiku-4.5")

    assert read_saved_config() == {
        "OPENROUTER_MODEL": "anthropic/claude-haiku-4.5",
        "OPENROUTER_MODEL_CONTEXT_LIMIT": "200000",
        "OPENROUTER_API_KEY": "sk-rotated",
    }

def test_set_preferred_model_rejects_unknown_model(
    fake_lookup: Callable[[ModelLookup], None], fake_home: Path
) -> None:
    fake_lookup(ModelLookup(found=False))

    with pytest.raises(ValueError, match="could not be validated"):
        config.set_preferred_model("not/a-real-model")

    assert not config.CONFIG_ENV.exists()


def test_set_preferred_model_keeps_config_on_rejected_model(
    monkeypatch: pytest.MonkeyPatch,
    fake_lookup: Callable[[ModelLookup], None],
    fake_home: Path,
) -> None:
    """A failed validation must leave an already-configured key untouched."""
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "sk-secret-abc")
    config.set_api_key()

    fake_lookup(ModelLookup(found=False))
    with pytest.raises(ValueError, match="could not be validated"):
        config.set_preferred_model("not/a-real-model")

    assert read_saved_config() == {"OPENROUTER_API_KEY": "sk-secret-abc"}


def test_get_model_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    assert config.get_model() is None


def test_get_model_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

    assert config.get_model() == "anthropic/claude-haiku-4.5"


def test_get_model_context_limit_parses_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL_CONTEXT_LIMIT", "200000")

    assert config.get_model_context_limit() == 200_000


def test_get_model_context_limit_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL_CONTEXT_LIMIT", raising=False)

    assert config.get_model_context_limit() is None
