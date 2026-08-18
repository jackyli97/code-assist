# config.py

from pathlib import Path
from dotenv import load_dotenv
import getpass
import os


CONFIG_DIR = Path.home() / ".config" / "code-assist"
CONFIG_ENV = CONFIG_DIR / ".env"


def load_config() -> None:
    load_dotenv(CONFIG_ENV, override=False)


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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    api_key = getpass.getpass("OpenRouter API key: ").strip()

    if not api_key:
        raise ValueError("API key cannot be empty.")

    CONFIG_ENV.write_text(
        f"OPENROUTER_API_KEY={api_key}\n"
    )

    CONFIG_ENV.chmod(0o600)

    print("OpenRouter API key configured.")