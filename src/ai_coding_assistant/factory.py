from pathlib import Path

from ai_coding_assistant.config import (
    get_api_key,
    get_base_url,
    load_config,
    get_model,
    get_model_context_limit
)

from openai import OpenAI
from ai_coding_assistant.agents import LlmAgent
from openai.types.chat import ChatCompletionFunctionToolParam
from typing import Iterable

def create_agent(tools: Iterable[ChatCompletionFunctionToolParam]) -> LlmAgent: 
    load_config()

    api_key = get_api_key()
    base_url = get_base_url()
    model = get_model()
    context_limit = get_model_context_limit()


    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    workspace = Path.cwd().resolve()

    return LlmAgent(
        client=client,
        workspace=workspace,
        tools=tools,
        model=model,
        context_limit=context_limit
    )