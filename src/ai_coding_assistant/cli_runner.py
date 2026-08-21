from ai_coding_assistant.agents import LlmAgent
from ai_coding_assistant.models import lookup_model
import click
import math

def run_agent(agent: LlmAgent, prompt: str) -> None:
    response = agent.agentic_loop_call(prompt=prompt)
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
                    f"Warning: context is ~{utilization}% full.\n"
                    "Use /clear to start fresh, /compact to compact history, or /model to switch models.\n"
                    "Use /status to view context usage and limit",
                    fg="yellow",
                )
            )

def run_interactive(agent: LlmAgent) -> None:
    click.echo(click.style("code-assist v0.1.0", fg="green", bold=True))
    click.echo(click.style(f"Workspace: ", fg="cyan") + str(agent.workspace))
    click.echo(click.style(f"Model: ", fg="white") + click.style(agent.model, fg="yellow"))
    click.echo("Type your request below, or '/help' for available commands.\n")

    while True:
        try:
            prompt = click.prompt(">", prompt_suffix=" ").strip()

            command, _, argument = prompt.partition(" ")
            argument = argument.strip()

            if not command:
                continue

            if command in ("/exit", "exit"):
                click.echo("\nExiting...")
                return

            if command == "/clear":
                agent.clear_history()
                click.echo("Conversation cleared.")
                continue

            if command == "/compact":
                message_compaction_result = agent.compact_message_history()
                if message_compaction_result.success:
                    click.echo(
                        "Message compaction completed.\n"
                        "If still near limit, consider clearing history or switching to model with larger context"
                    )
                else:
                    click.echo(
                        "Message compaction failed with reason:\n"
                        f"{message_compaction_result.failure_reason}"
                    ) 
                if message_compaction_result.run_prompt_tokens and message_compaction_result.run_completion_tokens:
                        click.echo(
                            f"Run tokens: {message_compaction_result.run_prompt_tokens:,} input / "
                            f"{message_compaction_result.run_completion_tokens:,} output"
                        )
                        click.echo(
                            f"Session tokens: {agent.session_prompt_tokens:,} input / "
                            f"{agent.session_completion_tokens:,} output"
                        )
                continue

            if command == "/model":
                if argument:
                    # validate model
                    validate_model_res = lookup_model(argument)
                    if not validate_model_res.found:
                        click.echo(
                            f"The model {argument} could not be validated. "
                            "Double check model id/name on https://openrouter.ai/models and try again"
                        )
                    else:
                        agent.update_model(model=argument, context_limit=validate_model_res.context_limit)
                else:
                    click.echo(f"  Model: {agent.model}")
                    click.echo(f"  Model's context limit: {f"{agent.context_limit} tokens" if agent.context_limit else "N/A"}") 

                continue

            if command == "/status":
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

            if command == "/help":
                click.echo("\nAvailable Commands:")
                click.echo("  /clear  - Reset the conversation history")
                click.echo("  /compact  - Compacts conversation history via summarization")
                click.echo("  /exit   - Quit the assistant (or use Ctrl+C)")
                click.echo("  /help   - Show this help menu")
                click.echo("  /model  - Show the current model, or switch with /model <model-id>")
                click.echo("  /status  - Show model, workspace, and token and context usage")
                click.echo("\nSafety:")
                click.echo(
                    "Shell commands are not OS-sandboxed in v0.1.0. "
                    "Review commands carefully before approving execution."
                )
                continue

            run_agent(agent=agent, prompt=prompt)


        except click.Abort:
            click.echo("\nExiting...")
            return
        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting...")
            return
        