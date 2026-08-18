import click
from pathlib import Path

from ai_coding_assistant.config import (
    get_api_key,
    get_base_url,
    load_config,
    set_api_key,
)

from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool
from ai_coding_assistant.tools import ReadTool, WriteTool, BashTool
from ai_coding_assistant.agents import LlmAgent

CONFIG_ENV = Path.home() / ".config" / "code-assist" / ".env"


def load_config():
    # Read config only from the user-level config file, never from the project the
    # CLI is pointed at -- that project's .env belongs to it, not to us.
    # Real environment variables still win, for overrides and CI.
    load_dotenv(CONFIG_ENV, override=False)

@click.group(invoke_without_command=True)
@click.option("-p", "--prompt", help="Prompt for the coding agent.")
@click.pass_context
def main(ctx: click.Context, prompt: str | None) -> None:
    """Chat with an AI coding agent about your project."""

    # A subcommand such as `config` is being run.
    if ctx.invoked_subcommand is not None:
        return

    if prompt is not None:
        run_agent(prompt)


@main.group()
def config() -> None:
    """Configuration options."""
    pass


@config.command("set-key")
def config_set_key() -> None:
    """Configure your OpenRouter API key."""
    set_api_key()


def run_agent(prompt: str) -> None:
    load_config()

    api_key = get_api_key()
    base_url = get_base_url()

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    read_tool = pydantic_function_tool(
        model=ReadTool.args_model,
        name=ReadTool.name,
        description=ReadTool.description,
    )

    write_tool = pydantic_function_tool(
        model=WriteTool.args_model,
        name=WriteTool.name,
        description=WriteTool.description,
    )

    bash_tool = pydantic_function_tool(
        model=BashTool.args_model,
        name=BashTool.name,
        description=BashTool.description,
    )

    workspace = Path.cwd().resolve()

    llm_agent = LlmAgent(
        client=client,
        workspace=workspace,
    )

    llm_agent.agentic_loop_call(
        prompt=prompt,
        tools=[read_tool, write_tool, bash_tool],
    )


if __name__ == "__main__":
    main()
