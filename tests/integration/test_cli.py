"""End-to-end runs of the `code-assist` entrypoint.

Everything from `main()` down is real here: Click argument parsing, the config
file, the factory, the agent loop, the tools, and the REPL — including the
permission prompt, which is answered over stdin exactly as a user would. Only
``OpenAI`` is swapped for a scripted double, so these tests also pin down what
the CLI hands the LLM client: the configured model, key and base URL, and a
workspace equal to the directory the CLI was started in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pytest
from click.testing import CliRunner, Result
from pytest_mock import MockerFixture

from ai_coding_assistant import config
from ai_coding_assistant.main import main

from tests.integration.support import (
    PYTHON,
    LlmRequest,
    ScriptedLLM,
    ScriptedTurn,
    Turn,
    bash,
    calls,
    read,
    says,
    write,
)

API_KEY = "sk-or-test-key"
CONFIG_MODEL = "test/model-from-config"


@dataclass
class CliHarness:
    workspace: Path
    config_env: Path
    llm: ScriptedLLM
    openai_ctor: Any
    runner: CliRunner = field(default_factory=CliRunner)

    def write_config(self, **values: str) -> None:
        self.config_env.parent.mkdir(parents=True, exist_ok=True)
        self.config_env.write_text(
            "".join(f"{key}={value}\n" for key, value in values.items())
        )

    def run(
        self,
        *args: str,
        script: Iterable[ScriptedTurn] = (),
        input: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Result:
        self.llm.extend(script)
        return self.runner.invoke(main, list(args), input=input, env=env or {})

    @property
    def client_kwargs(self) -> dict[str, Any]:
        return self.llm.init_kwargs


@pytest.fixture
def cli(
    workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> CliHarness:
    """Run the CLI from inside the temp workspace with an isolated config file."""
    config_dir = tmp_path / "fake-home" / ".config" / "code-assist"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_ENV", config_dir / ".env")
    monkeypatch.chdir(workspace)

    llm = ScriptedLLM()

    def fake_openai(**kwargs: Any) -> ScriptedLLM:
        llm.init_kwargs = dict(kwargs)
        return llm

    ctor = mocker.patch(
        "ai_coding_assistant.factory.OpenAI", side_effect=fake_openai
    )

    return CliHarness(
        workspace=workspace, config_env=config_dir / ".env", llm=llm, openai_ctor=ctor
    )


# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------


def test_single_shot_prompt_runs_the_agent_and_prints_the_answer(
    cli: CliHarness,
) -> None:
    result = cli.run(
        "-p",
        "what does this project do",
        script=[calls(read("README.md")), says("It is a calculator demo project.")],
        env={"OPENROUTER_API_KEY": API_KEY, "OPENROUTER_MODEL": CONFIG_MODEL},
    )

    assert result.exit_code == 0, result.output
    assert "It is a calculator demo project." in result.output
    # Tool activity is surfaced to the user as it happens.
    assert "● Read:" in result.output
    assert "✓ Done" in result.output
    # ...along with the run and session token accounting.
    assert "Run tokens: 200 input / 40 output" in result.output
    assert "Session tokens: 200 input / 40 output" in result.output


def test_single_shot_passes_the_configured_model_key_and_base_url(
    cli: CliHarness,
) -> None:
    result = cli.run(
        "-p",
        "hello",
        script=[says("hi")],
        env={
            "OPENROUTER_API_KEY": API_KEY,
            "OPENROUTER_MODEL": CONFIG_MODEL,
            "OPENROUTER_BASE_URL": "https://example.test/api/v1",
        },
    )

    assert result.exit_code == 0, result.output
    assert cli.client_kwargs == {
        "api_key": API_KEY,
        "base_url": "https://example.test/api/v1",
    }
    assert cli.llm.models_used == [CONFIG_MODEL]


def test_default_model_and_base_url_are_used_when_unconfigured(
    cli: CliHarness,
) -> None:
    result = cli.run(
        "-p", "hello", script=[says("hi")], env={"OPENROUTER_API_KEY": API_KEY}
    )

    assert result.exit_code == 0, result.output
    assert cli.client_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert cli.llm.models_used == ["anthropic/claude-haiku-4.5"]


def test_config_file_supplies_the_key_and_model(cli: CliHarness) -> None:
    cli.write_config(
        OPENROUTER_API_KEY="sk-or-from-config-file",
        OPENROUTER_MODEL=CONFIG_MODEL,
        OPENROUTER_MODEL_CONTEXT_LIMIT="123456",
    )

    result = cli.run("-p", "hello", script=[says("hi")])

    assert result.exit_code == 0, result.output
    assert cli.client_kwargs["api_key"] == "sk-or-from-config-file"
    assert cli.llm.models_used == [CONFIG_MODEL]


def test_the_projects_own_dotenv_is_never_loaded(cli: CliHarness) -> None:
    """A documented guarantee: the workspace's .env belongs to the project."""
    (cli.workspace / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-from-the-project\n"
        "OPENROUTER_MODEL=project/should-never-be-used\n"
    )
    cli.write_config(OPENROUTER_API_KEY="sk-or-from-config-file")

    result = cli.run("-p", "hello", script=[says("hi")])

    assert result.exit_code == 0, result.output
    assert cli.client_kwargs["api_key"] == "sk-or-from-config-file"
    assert "project/should-never-be-used" not in cli.llm.models_used


def test_environment_overrides_the_config_file(cli: CliHarness) -> None:
    cli.write_config(
        OPENROUTER_API_KEY="sk-or-from-config-file", OPENROUTER_MODEL=CONFIG_MODEL
    )

    result = cli.run(
        "-p",
        "hello",
        script=[says("hi")],
        env={"OPENROUTER_API_KEY": "sk-or-from-env", "OPENROUTER_MODEL": "test/env-model"},
    )

    assert result.exit_code == 0, result.output
    assert cli.client_kwargs["api_key"] == "sk-or-from-env"
    assert cli.llm.models_used == ["test/env-model"]


def test_missing_api_key_stops_before_any_llm_call(cli: CliHarness) -> None:
    result = cli.run("-p", "hello")

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "config set-key" in str(result.exception)
    assert cli.llm.requests == []


def test_workspace_is_the_directory_the_cli_was_started_in(
    cli: CliHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting inside a subdirectory makes *that* directory the boundary."""
    monkeypatch.chdir(cli.workspace / "src")

    def check(request: LlmRequest) -> Turn:
        assert str(cli.workspace / "src") in request.system_prompts[0]
        # Relative reads resolve against the subdirectory...
        assert "def add" in request.tool_results[0]["output"]
        # ...and the parent project is now out of bounds.
        assert request.tool_results[1]["success"] is False
        assert "outside workspace" in request.tool_results[1]["output"]
        return says("Only src/ is in scope.")

    result = cli.run(
        "-p",
        "read the calculator and the readme",
        script=[calls(read("calculator.py"), read("../README.md")), check],
        env={"OPENROUTER_API_KEY": API_KEY},
    )

    assert result.exit_code == 0, result.output


def test_context_warning_is_printed_when_the_window_fills_up(cli: CliHarness) -> None:
    result = cli.run(
        "-p",
        "hello",
        script=[says("hi")],
        env={
            "OPENROUTER_API_KEY": API_KEY,
            "OPENROUTER_MODEL_CONTEXT_LIMIT": "1000",  # ~800 tokens of tools + prompt
        },
    )

    assert result.exit_code == 0, result.output
    assert "Warning: context is" in result.output
    assert "/compact" in result.output


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def test_interactive_session_keeps_context_between_prompts(cli: CliHarness) -> None:
    def second(request: LlmRequest) -> Turn:
        assert request.user_prompts == [
            "what does this project do",
            "and what is the bug",
        ]
        assert "A calculator library" in request.text()
        return says("add() subtracts instead of adding.")

    result = cli.run(
        script=[
            calls(read("README.md")),
            says("It is a calculator demo project."),
            second,
        ],
        input="what does this project do\nand what is the bug\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY, "OPENROUTER_MODEL": CONFIG_MODEL},
    )

    assert result.exit_code == 0, result.output
    assert "code-assist v0.1.0" in result.output
    assert f"Workspace: {cli.workspace}" in result.output
    assert "It is a calculator demo project." in result.output
    assert "add() subtracts instead of adding." in result.output
    assert "Exiting..." in result.output


def test_interactive_write_is_approved_from_stdin(cli: CliHarness) -> None:
    result = cli.run(
        script=[calls(write("notes/added.md", "a new note\n")), says("Added the note.")],
        input="add a note\ny\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY},
    )

    assert result.exit_code == 0, result.output
    assert "code-assist wants to perform an action:" in result.output
    assert "Allow? [y] once / [s] session / [n] deny" in result.output
    assert (cli.workspace / "notes" / "added.md").read_text() == "a new note\n"


def test_interactive_write_is_denied_from_stdin(cli: CliHarness) -> None:
    def after_denial(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert "denied permission" in request.last_tool_result["output"]
        return says("Left it alone.")

    result = cli.run(
        script=[calls(write("README.md", "clobbered")), after_denial],
        input="rewrite the readme\nn\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY},
    )

    assert result.exit_code == 0, result.output
    assert "✗ Failed" in result.output
    assert "clobbered" not in (cli.workspace / "README.md").read_text()


def test_interactive_session_grant_is_only_asked_once(cli: CliHarness) -> None:
    result = cli.run(
        script=[
            calls(bash([PYTHON, "-c", "print('one')"])),
            calls(bash([PYTHON, "-c", "print('two')"])),
            says("Ran both."),
        ],
        # A single 's' answer covers the second command too — no second prompt
        # is offered, so '/exit' is the very next line of input.
        input="run both commands\ns\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY},
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("code-assist wants to perform an action:") == 1
    assert "Ran both." in result.output


def test_interactive_whitespace_only_input_is_ignored(cli: CliHarness) -> None:
    result = cli.run(
        script=[says("An answer.")],
        input="   \nreal question\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY},
    )

    assert result.exit_code == 0, result.output
    # A whitespace-only line is skipped without reaching the LLM.
    assert len(cli.llm.requests) == 1
    assert cli.llm.requests[0].user_prompts == ["real question"]


def test_interactive_clear_resets_the_conversation(cli: CliHarness) -> None:
    def after_clear(request: LlmRequest) -> Turn:
        assert request.user_prompts == ["second question"]
        assert "first question" not in request.text()
        return says("Fresh answer.")

    result = cli.run(
        script=[says("First answer."), after_clear],
        input="first question\n/clear\nsecond question\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY},
    )

    assert result.exit_code == 0, result.output
    assert "Conversation cleared." in result.output


def test_interactive_status_reports_model_workspace_and_usage(
    cli: CliHarness,
) -> None:
    result = cli.run(
        script=[says("An answer.", prompt_tokens=1200, completion_tokens=300)],
        input="a question\n/status\n/exit\n",
        env={
            "OPENROUTER_API_KEY": API_KEY,
            "OPENROUTER_MODEL": CONFIG_MODEL,
            "OPENROUTER_MODEL_CONTEXT_LIMIT": "123456",
        },
    )

    assert result.exit_code == 0, result.output
    assert f"Model: {CONFIG_MODEL}" in result.output
    assert "Model's context limit: 123456 tokens" in result.output
    assert f"Workspace: {cli.workspace}" in result.output
    assert "API usage: 1,200 input / 300 output" in result.output
    assert "/ 123,456 tokens" in result.output


def test_interactive_help_lists_commands_and_the_sandbox_warning(
    cli: CliHarness,
) -> None:
    result = cli.run(input="/help\n/exit\n", env={"OPENROUTER_API_KEY": API_KEY})

    assert result.exit_code == 0, result.output
    for command in ("/clear", "/compact", "/exit", "/help", "/model", "/status"):
        assert command in result.output
    assert "not OS-sandboxed" in result.output
    assert cli.llm.requests == []


def test_interactive_model_command_switches_the_model_for_later_prompts(
    cli: CliHarness,
) -> None:
    result = cli.run(
        script=[says("Answer one."), says("Answer two.")],
        input=(
            "first question\n"
            "/model\n"
            "/model test/switched-model\n"
            "second question\n"
            "/exit\n"
        ),
        env={"OPENROUTER_API_KEY": API_KEY, "OPENROUTER_MODEL": CONFIG_MODEL},
    )

    assert result.exit_code == 0, result.output
    assert f"Model: {CONFIG_MODEL}" in result.output
    assert cli.llm.models_used == [CONFIG_MODEL, "test/switched-model"]


def test_interactive_model_command_rejects_an_unknown_model(
    cli: CliHarness, mocker: MockerFixture
) -> None:
    from ai_coding_assistant.models import ModelLookup

    mocker.patch(
        "ai_coding_assistant.cli_runner.lookup_model",
        return_value=ModelLookup(found=False),
    )

    result = cli.run(
        script=[says("Answer.")],
        input="/model not-a-real/model\na question\n/exit\n",
        env={"OPENROUTER_API_KEY": API_KEY, "OPENROUTER_MODEL": CONFIG_MODEL},
    )

    assert result.exit_code == 0, result.output
    assert "could not be validated" in result.output
    # The session keeps the model it already had.
    assert cli.llm.models_used == [CONFIG_MODEL]


def test_interactive_compact_summarizes_a_full_context(cli: CliHarness) -> None:
    result = cli.run(
        script=[
            calls(read("README.md")),
            says("It is a calculator demo project."),
            says("SUMMARY: the user asked what the project does."),
        ],
        input="what does this project do\n/compact\n/status\n/exit\n",
        env={
            "OPENROUTER_API_KEY": API_KEY,
            "OPENROUTER_MODEL_CONTEXT_LIMIT": "1000",  # forces real compaction
        },
    )

    assert result.exit_code == 0, result.output
    assert "Message compaction completed." in result.output

    # A real summarization call was made: no tools offered, compaction prompt used.
    compaction_request = cli.llm.requests[-1]
    assert compaction_request.tools is None
    assert "Summarize the conversation history" in compaction_request.system_prompts[0]


def test_interactive_exits_cleanly_on_end_of_input(cli: CliHarness) -> None:
    result = cli.run(input="", env={"OPENROUTER_API_KEY": API_KEY})

    assert result.exit_code == 0
    assert "Exiting..." in result.output


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def test_config_set_model_writes_the_config_without_starting_an_agent(
    cli: CliHarness,
) -> None:
    result = cli.run("config", "set-model", "test/chosen-model")

    assert result.exit_code == 0, result.output
    saved = cli.config_env.read_text()
    assert "OPENROUTER_MODEL=test/chosen-model" in saved
    assert "OPENROUTER_MODEL_CONTEXT_LIMIT=200000" in saved
    assert cli.openai_ctor.call_count == 0


def test_config_set_key_stores_the_key_with_restricted_permissions(
    cli: CliHarness, mocker: MockerFixture
) -> None:
    mocker.patch("ai_coding_assistant.config.getpass.getpass", return_value=API_KEY)

    result = cli.run("config", "set-key")

    assert result.exit_code == 0, result.output
    assert f"OPENROUTER_API_KEY={API_KEY}" in cli.config_env.read_text()
    assert cli.config_env.stat().st_mode & 0o777 == 0o600
    assert cli.openai_ctor.call_count == 0


def test_version_flag_does_not_start_a_session(cli: CliHarness) -> None:
    result = cli.run("--version")

    assert result.exit_code == 0
    assert "0.1.0" in result.output
    assert cli.openai_ctor.call_count == 0
