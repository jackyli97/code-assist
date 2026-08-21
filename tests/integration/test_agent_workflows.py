"""Multi-step agent runs against a real workspace.

The unit suite stubs ``execute_tool`` and asserts the loop's bookkeeping. These
tests do the opposite: the loop, the tool registry, the path guards, the
permission prompt and the filesystem are all real, and only the model is
scripted. What they prove is that tool output actually flows back into the
next request, and that the agent's writes and commands land on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from tests.integration.support import (
    CALCULATOR_FIXED,
    CALCULATOR_WITH_BUG,
    OUTSIDE_SECRET,
    PYTHON,
    WORKSPACE_SECRET,
    AgentHarness,
    LlmRequest,
    Turn,
    bash,
    calls,
    read,
    says,
    use_tool,
    write,
)

BuildAgent = Callable[..., AgentHarness]


# ---------------------------------------------------------------------------
# The headline workflow: search -> read -> edit -> verify
# ---------------------------------------------------------------------------


def test_search_read_edit_and_verify_workflow(build_agent: BuildAgent) -> None:
    """grep -> Read -> Write -> re-run checks, each step fed by the last."""

    def after_grep(request: LlmRequest) -> Turn:
        # The grep output must have reached the model before it can pick a file.
        assert request.last_tool_result["success"] is True
        assert "src/calculator.py" in request.last_tool_result["output"]
        return calls(read("src/calculator.py"))

    def after_read(request: LlmRequest) -> Turn:
        # The buggy line must be visible in the Read output.
        assert "return a - b" in request.last_tool_result["output"]
        return calls(write("src/calculator.py", CALCULATOR_FIXED))

    def after_write(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is True
        return calls(bash([PYTHON, "tests/check_calculator.py"]))

    def after_checks(request: LlmRequest) -> Turn:
        result = request.last_tool_result
        assert result["success"] is True, result["output"]
        assert "all checks passed" in result["output"]
        return says("Fixed add() and the checks now pass.")

    harness = build_agent(
        [
            calls(bash(["grep", "-rn", "def add", "src"])),
            after_grep,
            after_read,
            after_write,
            after_checks,
        ]
    )

    response = harness.agent.agentic_loop_call(prompt="add() is broken, fix it")

    assert response.content == "Fixed add() and the checks now pass."
    assert harness.llm.tool_names_requested == ["Bash", "Read", "Write", "Bash"]
    assert len(harness.llm.requests) == 5

    # The fix is on disk, and the checks genuinely pass against it.
    assert (harness.workspace / "src" / "calculator.py").read_text() == CALCULATOR_FIXED

    # One run, holding every message from the whole loop.
    assert len(harness.agent.runs) == 1
    roles = [m["role"] for m in harness.agent.runs[0]]
    assert roles.count("tool") == 4


def test_every_tool_result_is_returned_against_its_call_id(
    build_agent: BuildAgent,
) -> None:
    """Parallel tool calls in one turn must each get their own tool message."""
    read_call = read("README.md")
    write_call = write("notes/new.md", "drafted\n")

    harness = build_agent(
        [calls(read_call, write_call), says("Read the readme and left a note.")]
    )

    harness.agent.agentic_loop_call(prompt="read the readme then leave a note")

    final_request = harness.llm.requests[-1]
    tool_messages = {
        m["tool_call_id"]: json.loads(str(m["content"]))
        for m in final_request.messages
        if m.get("role") == "tool"
    }

    assert set(tool_messages) == {read_call.id, write_call.id}
    assert "demo project" in tool_messages[read_call.id]["output"]
    assert tool_messages[write_call.id]["success"] is True
    assert (harness.workspace / "notes" / "new.md").read_text() == "drafted\n"


# ---------------------------------------------------------------------------
# Failure recovery
# ---------------------------------------------------------------------------


def test_agent_recovers_from_a_failing_shell_command(build_agent: BuildAgent) -> None:
    """A non-zero exit is surfaced as a failed tool result the agent can act on."""

    def after_failed_checks(request: LlmRequest) -> Turn:
        result = request.last_tool_result
        assert result["success"] is False
        assert "AssertionError" in result["output"]
        assert "add(2, 3) returned -1" in result["output"]
        return calls(write("src/calculator.py", CALCULATOR_FIXED))

    def after_fix(_: LlmRequest) -> Turn:
        return calls(bash([PYTHON, "tests/check_calculator.py"]))

    def after_rerun(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is True
        return says("The failing check pointed at add(); fixed and re-ran.")

    harness = build_agent(
        [
            calls(bash([PYTHON, "tests/check_calculator.py"])),
            after_failed_checks,
            after_fix,
            after_rerun,
        ]
    )

    response = harness.agent.agentic_loop_call(prompt="run the checks and fix failures")

    assert "fixed and re-ran" in response.content
    # The run's history records the failure *and* the recovery, in that order.
    successes = [r["success"] for r in harness.llm.requests[-1].tool_results]
    assert successes == [False, True, True]
    assert (harness.workspace / "src" / "calculator.py").read_text() == CALCULATOR_FIXED


def test_missing_file_read_does_not_abort_the_run(build_agent: BuildAgent) -> None:
    def after_missing(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert "File not found" in request.last_tool_result["output"]
        return calls(read("src/calculator.py"))

    harness = build_agent(
        [
            calls(read("src/calculatr.py")),  # typo
            after_missing,
            says("Found it at src/calculator.py."),
        ]
    )

    response = harness.agent.agentic_loop_call(prompt="show me the calculator")

    assert response.content == "Found it at src/calculator.py."


def test_unknown_tool_is_reported_without_crashing(build_agent: BuildAgent) -> None:
    def after_unknown(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert "Unknown tool" in request.last_tool_result["output"]
        return says("I only have Read, Write and Bash.")

    harness = build_agent(
        [calls(use_tool("Delete", file_path="README.md")), after_unknown]
    )

    response = harness.agent.agentic_loop_call(prompt="delete the readme")

    assert response.content == "I only have Read, Write and Bash."


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def test_denied_write_leaves_the_file_untouched(build_agent: BuildAgent) -> None:
    def after_denial(request: LlmRequest) -> Turn:
        result = request.last_tool_result
        assert result["success"] is False
        assert "denied permission" in result["output"]
        return says("Understood — I left calculator.py alone.")

    harness = build_agent(
        [calls(write("src/calculator.py", CALCULATOR_FIXED)), after_denial],
        permissions="n",
    )

    response = harness.agent.agentic_loop_call(prompt="fix add()")

    assert response.content == "Understood — I left calculator.py alone."
    assert (
        harness.workspace / "src" / "calculator.py"
    ).read_text() == CALCULATOR_WITH_BUG
    assert len(harness.permission_prompts) == 1


def test_denied_command_never_runs(build_agent: BuildAgent) -> None:
    marker = "should-not-exist.txt"

    harness = build_agent(
        [
            calls(bash([PYTHON, "-c", f"open({marker!r}, 'w').write('x')"])),
            says("I did not run it."),
        ],
        permissions="n",
    )

    harness.agent.agentic_loop_call(prompt="create a marker file")

    assert not (harness.workspace / marker).exists()


def test_reads_never_prompt_for_permission(build_agent: BuildAgent) -> None:
    harness = build_agent(
        [calls(read("README.md"), read("notes/todo.md")), says("Both read.")],
        permissions=None,  # any prompt at all fails the test
    )

    harness.agent.agentic_loop_call(prompt="read the readme and the todo list")

    assert harness.permission_prompts == []


def test_session_grant_prompts_once_then_applies_to_later_calls(
    build_agent: BuildAgent,
) -> None:
    """'s' covers the rest of the session for that tool — but only that tool."""
    harness = build_agent(
        [
            calls(write("notes/a.md", "a\n")),
            calls(write("notes/b.md", "b\n")),
            calls(bash([PYTHON, "-c", "print('hi')"])),
            says("Wrote both notes and ran the command."),
        ],
        permission_answers=("s", "y"),  # 's' for Write, then a fresh ask for Bash
        permissions=None,
    )

    harness.agent.agentic_loop_call(prompt="write two notes then print hi")

    assert (harness.workspace / "notes" / "a.md").exists()
    assert (harness.workspace / "notes" / "b.md").exists()
    # Write asked once; the second write reused the grant; Bash asked separately.
    assert len(harness.permission_prompts) == 2


def test_session_grant_survives_across_prompts(build_agent: BuildAgent) -> None:
    harness = build_agent(
        [
            calls(write("notes/first.md", "1\n")),
            says("done"),
            calls(write("notes/second.md", "2\n")),
            says("done again"),
        ],
        permission_answers=("s",),
        permissions=None,
    )

    harness.agent.agentic_loop_call(prompt="write the first note")
    harness.agent.agentic_loop_call(prompt="now write the second note")

    assert (harness.workspace / "notes" / "second.md").exists()
    assert len(harness.permission_prompts) == 1


# ---------------------------------------------------------------------------
# Workspace containment — reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_path, expected_error",
    [
        ("../outside/secret.txt", "outside workspace"),
        ("src/../../outside/secret.txt", "outside workspace"),
        ("../../etc/passwd", "outside workspace"),
        ("/etc/passwd", "absolute path"),
    ],
)
def test_read_cannot_escape_the_workspace(
    build_agent: BuildAgent, file_path: str, expected_error: str
) -> None:
    def after_block(request: LlmRequest) -> Turn:
        result = request.last_tool_result
        assert result["success"] is False
        assert expected_error in result["output"]
        return says("That path is outside the project.")

    harness = build_agent([calls(read(file_path)), after_block])

    harness.agent.agentic_loop_call(prompt=f"read {file_path}")

    # The blocked file's contents never reached the model.
    assert OUTSIDE_SECRET not in harness.llm.all_text()


def test_read_cannot_follow_a_symlink_out_of_the_workspace(
    build_agent: BuildAgent, workspace: Path, outside_dir: Path
) -> None:
    """Paths are resolved before the boundary check, so symlinks don't help."""
    (workspace / "escape_hatch").symlink_to(outside_dir, target_is_directory=True)

    def after_block(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert "outside workspace" in request.last_tool_result["output"]
        return says("Blocked.")

    harness = build_agent([calls(read("escape_hatch/secret.txt")), after_block])

    harness.agent.agentic_loop_call(prompt="read escape_hatch/secret.txt")

    assert OUTSIDE_SECRET not in harness.llm.all_text()


# ---------------------------------------------------------------------------
# Workspace containment — writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_kind, expected_error",
    [
        ("relative", "outside workspace"),
        ("traversal", "outside workspace"),
        ("absolute", "absolute path"),
    ],
)
def test_write_cannot_escape_the_workspace(
    build_agent: BuildAgent,
    tmp_path: Path,
    outside_dir: Path,
    path_kind: str,
    expected_error: str,
) -> None:
    # Each case names a distinct target so the "nothing was created" assertion
    # can point at exactly the path this run tried to plant a file at.
    target, file_path = {
        "relative": (outside_dir / "planted.txt", "../outside/planted.txt"),
        "traversal": (tmp_path / "planted.txt", "../../planted.txt"),
        "absolute": (
            outside_dir / "planted-abs.txt",
            str(outside_dir / "planted-abs.txt"),
        ),
    }[path_kind]

    def after_block(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert expected_error in request.last_tool_result["output"]
        return says("Blocked.")

    harness = build_agent([calls(write(file_path, "planted")), after_block])

    harness.agent.agentic_loop_call(prompt=f"write to {file_path}")

    assert not target.exists()
    # And the file that already lived outside the workspace is untouched.
    assert (outside_dir / "secret.txt").read_text() == OUTSIDE_SECRET


def test_write_through_a_symlinked_directory_is_blocked(
    build_agent: BuildAgent, workspace: Path, outside_dir: Path
) -> None:
    (workspace / "escape_hatch").symlink_to(outside_dir, target_is_directory=True)

    harness = build_agent(
        [calls(write("escape_hatch/planted.txt", "planted")), says("Blocked.")]
    )

    harness.agent.agentic_loop_call(prompt="write through the symlink")

    assert not (outside_dir / "planted.txt").exists()


def test_write_creates_and_overwrites_only_inside_the_workspace(
    build_agent: BuildAgent,
) -> None:
    harness = build_agent(
        [
            calls(write("notes/todo.md", "- done\n")),
            calls(write("src/new_module.py", "VALUE = 1\n")),
            says("Updated the todo list and added a module."),
        ]
    )

    harness.agent.agentic_loop_call(prompt="tidy up the project")

    assert (harness.workspace / "notes" / "todo.md").read_text() == "- done\n"
    assert (harness.workspace / "src" / "new_module.py").read_text() == "VALUE = 1\n"


# ---------------------------------------------------------------------------
# Workspace containment — commands
# ---------------------------------------------------------------------------


def test_command_cwd_is_confined_to_the_workspace(
    build_agent: BuildAgent, workspace: Path
) -> None:
    def after_escape_attempt(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert "outside workspace" in request.last_tool_result["output"]
        return calls(bash([PYTHON, "-c", "import os; print(os.getcwd())"], cwd="src"))

    def after_valid_cwd(request: LlmRequest) -> Turn:
        result = request.last_tool_result
        assert result["success"] is True
        assert result["output"].strip().endswith("/src")
        return says("Ran from the src directory.")

    harness = build_agent(
        [
            calls(bash([PYTHON, "-c", "print(1)"], cwd="..")),
            after_escape_attempt,
            after_valid_cwd,
        ]
    )

    harness.agent.agentic_loop_call(prompt="where am I running from?")

    assert str(workspace) in harness.llm.requests[-1].last_tool_result["output"]


def test_command_with_a_nonexistent_cwd_reports_a_usable_error(
    build_agent: BuildAgent,
) -> None:
    def after_bad_cwd(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        assert "does not exist" in request.last_tool_result["output"]
        return says("That directory isn't there.")

    harness = build_agent(
        [calls(bash([PYTHON, "-c", "print(1)"], cwd="nope/nowhere")), after_bad_cwd]
    )

    harness.agent.agentic_loop_call(prompt="run something in nope/nowhere")


def test_commands_run_without_a_shell(build_agent: BuildAgent) -> None:
    """shell=False means shell metacharacters are inert, not interpreted."""
    harness = build_agent(
        [
            calls(bash([PYTHON, "-c", "print('safe')", ";", "touch pwned.txt"])),
            says("Done."),
        ]
    )

    harness.agent.agentic_loop_call(prompt="print safe")

    assert not (harness.workspace / "pwned.txt").exists()


# ---------------------------------------------------------------------------
# Sensitive paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_call_factory",
    [
        pytest.param(lambda: read(".env"), id="read-dotenv"),
        pytest.param(lambda: read("../project/.env"), id="read-dotenv-roundabout"),
        pytest.param(lambda: write(".env", "API_SECRET=leaked"), id="write-dotenv"),
        pytest.param(
            lambda: write(".env.production", "API_SECRET=leaked"), id="write-dotenv-suffix"
        ),
        pytest.param(lambda: bash(["cat", ".env"]), id="bash-cat-dotenv"),
        pytest.param(
            lambda: bash(["cp", ".env", "notes/copy.txt"]), id="bash-copy-dotenv"
        ),
        pytest.param(lambda: read(".ssh/id_rsa"), id="read-ssh-key"),
    ],
)
def test_sensitive_paths_are_blocked_for_every_tool(
    build_agent: BuildAgent, tool_call_factory: Callable[[], object]
) -> None:
    def after_block(request: LlmRequest) -> Turn:
        assert request.last_tool_result["success"] is False
        return says("I can't touch that file.")

    harness = build_agent([calls(tool_call_factory()), after_block])  # type: ignore[arg-type]

    harness.agent.agentic_loop_call(prompt="handle the env file")

    # The secret never reached the model, and .env was not modified or copied.
    assert WORKSPACE_SECRET not in harness.llm.all_text()
    assert (harness.workspace / ".env").read_text() == f"{WORKSPACE_SECRET}\n"
    assert not (harness.workspace / "notes" / "copy.txt").exists()


def test_env_lookalikes_are_still_readable(build_agent: BuildAgent) -> None:
    """.envrc / backup.env aren't dotenv files; blocking them would be wrong."""
    harness = build_agent(
        [calls(read(".envrc"), read("backup.env")), says("Read both.")]
    )
    (harness.workspace / ".envrc").write_text("export EDITOR=vim\n")
    (harness.workspace / "backup.env").write_text("NOT_SECRET=1\n")

    harness.agent.agentic_loop_call(prompt="read the shell config files")

    results = harness.llm.requests[-1].tool_results
    assert [r["success"] for r in results] == [True, True]
