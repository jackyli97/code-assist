import requests
from dataclasses import dataclass
@dataclass
class ModelLookup:
    found: bool
    context_limit: int | None = None


def lookup_model(target_model_id: str) -> ModelLookup:
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return ModelLookup(found=False)

    for model in data.get("data", []):
        if model.get("id") == target_model_id:
            return ModelLookup(found=True, context_limit=model.get("context_length"))

    return ModelLookup(found=False)
    
