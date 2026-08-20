import requests


def get_context_limit(target_model_id: str) -> int | None:
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        # Network hiccup, non-2xx status, or malformed JSON — fall back to
        # "unknown context limit" so a temporary API blip doesn't crash the CLI.
        return None

    for model in data.get("data", []):
        if model.get("id") == target_model_id:
            return model.get("context_length")

    return None
