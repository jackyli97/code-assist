import click
from pathlib import Path

from ai_coding_assistant.config import set_api_key

from dotenv import load_dotenv
from ai_coding_assistant.tools import get_tools
from ai_coding_assistant.factory import create_agent
from ai_coding_assistant.cli_runner import run_agent, run_interactive

CONFIG_ENV = Path.home() / ".config" / "code-assist" / ".env"


def load_config():
    # Read config only from the user-level config file, never from the project the
    # CLI is pointed at -- that project's .env belongs to it, not to us.
    # Real environment variables still win, for overrides and CI.
    load_dotenv(CONFIG_ENV, override=False)

@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="code-assist")
@click.option("-p", "--prompt", help="Prompt for the coding agent.")
@click.pass_context
def main(ctx: click.Context, prompt: str | None) -> None:
    """
    Chat with an AI coding agent about your project.
    Just type 'code-assist' to start an interactive session, or 'code-assist -p' for single prompt
    """

    # A subcommand such as `config` is being run.
    if ctx.invoked_subcommand is not None:
        return

    tools = get_tools()
    agent = create_agent(tools)

    if prompt: # single shot prompt
        run_agent(agent=agent, prompt=prompt)
    else: # just code-assist was entered
        run_interactive(agent=agent)


@main.group()
def config() -> None:
    """Configuration options."""
    pass


@config.command("set-key")
def config_set_key() -> None:
    """Configure your OpenRouter API key."""
    set_api_key()

if __name__ == "__main__":
    main()
