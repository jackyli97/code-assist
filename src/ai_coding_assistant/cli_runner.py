from ai_coding_assistant.agents import LlmAgent
from openai.types.chat import ChatCompletionFunctionToolParam
import click
import math

def run_agent(agent: LlmAgent, tools: list[ChatCompletionFunctionToolParam], prompt: str) -> None:
    response = agent.agentic_loop_call(
        prompt=prompt,
        tools=tools,
    )
    content,run_prompt_tokens,run_completion_tokens = (
        response.content,
        response.run_prompt_tokens, 
        response.run_completion_tokens,
    )
    click.echo(content)
    if run_prompt_tokens and run_completion_tokens:
        click.echo(
            f"Run tokens: {run_prompt_tokens:,} input / "
            f"{run_completion_tokens:,} output"
        )
        click.echo(
            f"Session tokens: {agent.session_prompt_tokens:,} input / "
            f"{agent.session_completion_tokens:,} output"
        )

    # conditionally log warning if context utilization over 80%
    if agent.context_limit:
        context = agent.context
        utilization = math.ceil((context / agent.context_limit) * 100)
        if utilization > 80:
            click.echo(
                click.style(
                    f"Warning: context is ~{utilization:.0%} full.\n"
                    "Use /clear to start fresh or /model to switch models.\n"
                    "Use /status to view context usage and limit",
                    fg="yellow",
                )
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
                click.echo(f"  Model's context limit: {f"{agent.context_limit} tokens" if agent.context_limit else "N/A"}")
                click.echo(f"  Workspace: {agent.workspace}")
                click.echo(
                    f"  API usage: {agent.session_prompt_tokens:,} input / "
                    f"{agent.session_completion_tokens:,} output"
                )
                if agent.context_limit:
                    context = agent.context
                    click.echo(
                        f"  Context: ~{context:,} tokens / "
                        f"{agent.context_limit:,} tokens "
                        f"({math.ceil((context / agent.context_limit) * 100)}%)"
                    )
                continue

            if prompt == "/help":
                click.echo("\nAvailable Commands:")
                click.echo("  /help   - Show this help menu")
                click.echo("  /clear  - Reset the conversation history")
                click.echo("  /status  - Show model, workspace, and token and context usage")
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
        