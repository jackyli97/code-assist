# config.py

from pathlib import Path
from dotenv import load_dotenv
import getpass
import os
from ai_coding_assistant.models import lookup_model

CONFIG_DIR = Path.home() / ".config" / "code-assist"
CONFIG_ENV = CONFIG_DIR / ".env"


def load_config() -> None:
    load_dotenv(CONFIG_ENV, override=False)


def _read_config() -> dict[str, str]:
    """Parse the config file into a key/value mapping (empty if absent)."""
    if not CONFIG_ENV.exists():
        return {}

    values: dict[str, str] = {}
    for line in CONFIG_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()

    return values


def _update_config(updates: dict[str, str | None]) -> None:
    """Merge updates into the config file, preserving unrelated settings.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    values = _read_config()
    for key, value in updates.items():
        if value is None:
            values.pop(key, None)
        else:
            values[key] = value

    CONFIG_ENV.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    CONFIG_ENV.chmod(0o600)


def get_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is not configured. "
            "Run `code-assist config set-key`."
        )

    return api_key


def get_base_url() -> str:
    return os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )


def set_api_key() -> None:
    api_key = getpass.getpass("OpenRouter API key: ").strip()

    if not api_key:
        raise ValueError("API key cannot be empty.")

    _update_config({"OPENROUTER_API_KEY": api_key})

    print("OpenRouter API key configured.")


def set_preferred_model(model: str) -> None:
    validate_model_res = lookup_model(model)
    if not validate_model_res.found:
        raise ValueError(
            f"The model {model} could not be validated. "
            "Double check model id/name on https://openrouter.ai/models and try again"
        )

    context_limit = validate_model_res.context_limit
    _update_config(
        {
            "OPENROUTER_MODEL": model,
            "OPENROUTER_MODEL_CONTEXT_LIMIT": (
                str(context_limit) if context_limit else None
            ),
        }
    )

def get_model() -> str | None:
    model = os.getenv("OPENROUTER_MODEL")

    return model


def get_model_context_limit() -> int | None:
    context_limit = os.getenv("OPENROUTER_MODEL_CONTEXT_LIMIT")

    return int(context_limit) if context_limit is not None else context_limit