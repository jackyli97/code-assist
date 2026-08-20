from pathlib import Path

from ai_coding_assistant.config import (
    get_api_key,
    get_base_url,
    load_config,
)

from openai import OpenAI
from ai_coding_assistant.agents import LlmAgent
from openai.types.chat import ChatCompletionFunctionToolParam
from typing import Iterable

def create_agent(tools: Iterable[ChatCompletionFunctionToolParam]) -> LlmAgent: 
    load_config()

    api_key = get_api_key()
    base_url = get_base_url()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    workspace = Path.cwd().resolve()

    return LlmAgent(
        client=client,
        workspace=workspace,
        tools=tools
    )