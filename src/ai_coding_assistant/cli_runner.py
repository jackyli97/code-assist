from ai_coding_assistant.agents import LlmAgent
from openai.types.chat import ChatCompletionFunctionToolParam
import click
from dataclasses import astuple

def run_agent(agent: LlmAgent, tools: list[ChatCompletionFunctionToolParam], prompt: str) -> None:
    response = agent.agentic_loop_call(
        prompt=prompt,
        tools=tools,
    )
    content,run_prompt_tokens,run_completion_tokens, session_prompt_tokens, session_completion_tokens = (
        response.content,
        response.run_prompt_tokens, 
        response.run_completion_tokens,
        response.session_prompt_tokens,
        response.session_completion_tokens
    )
    click.echo(content)
    if run_prompt_tokens and run_completion_tokens:
        click.echo(
            f"Run tokens: {run_prompt_tokens:,} input / "
            f"{run_completion_tokens:,} output"
        )
    if session_prompt_tokens and session_completion_tokens:
            click.echo(
                f"Session tokens: {session_prompt_tokens:,} input / "
                f"{session_completion_tokens:,} output"
            )

def run_interactive(agent: LlmAgent, tools: list[ChatCompletionFunctionToolParam]) -> None:
    click.echo(click.style("code-assist v0.1.0", fg="green", bold=True))
    click.echo(click.style(f"Workspace: ", fg="cyan") + str(agent.workspace))
    click.echo(click.style(f"Model: ", fg="white") + click.style(agent.model, fg="yellow"))
    click.echo("Type your request below, or '/help' for available commands.\n")

    while True:
        try:
            prompt = click.prompt(">", prompt_suffix=" ").strip()

            if not prompt:
                continue

            if prompt in ("/exit", "exit"):
                click.echo("\nExiting...")
                return

            if prompt == "/clear":
                agent.clear_history()
                click.echo("Conversation cleared.")
                continue

            if prompt == "/status":
                click.echo("Session status")
                click.echo(f"  Model: {agent.model}")
                click.echo(f"  Workspace: {agent.workspace}")
                click.echo(
                    f"  Tokens: {agent.session_prompt_tokens:,} input / "
                    f"{agent.session_completion_tokens:,} output"
                )
                continue

            if prompt == "/help":
                click.echo("\nAvailable Commands:")
                click.echo("  /help   - Show this help menu")
                click.echo("  /clear  - Reset the conversation history")
                click.echo("  /status  - Show current session status and token usage")
                click.echo("  /exit   - Quit the assistant (or use Ctrl+C)")
                click.echo()
                continue

            run_agent(agent=agent, tools=tools, prompt=prompt)

        except click.Abort:
            click.echo("\nExiting...")
            return
        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting...")
            return