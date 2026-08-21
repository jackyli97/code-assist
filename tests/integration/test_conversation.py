"""Conversation state across multiple prompts in one session.

These exercise the thing an interactive session is for: what the agent sends on
prompt N+1 after everything that happened on prompt N — including tool
transcripts, /clear, /compact and a mid-session model switch. History assembly,
token estimation and compaction all run for real against the real tool schema,
so the numbers here are the numbers a live session would see.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from tests.integration.support import (
    PYTHON,
    AgentHarness,
    LlmRequest,
    Turn,
    bash,
    calls,
    read,
    says,
    write,
)

BuildAgent = Callable[..., AgentHarness]


# ---------------------------------------------------------------------------
# Carrying context between prompts
# ---------------------------------------------------------------------------


def test_context_carries_across_prompts(build_agent: BuildAgent) -> None:
    def second_prompt(request: LlmRequest) -> Turn:
        # Everything from the first exchange is still on the wire.
        assert request.user_prompts == [
            "what does this project do",
            "now summarize it in one line",
        ]
        assert "A calculator library" in request.text()  # the Read output
        assert "It is a calculator demo project." in request.text()
        return says("A calculator demo.")

    harness = build_agent(
        [
            calls(read("README.md")),
            says("It is a calculator demo project."),
            second_prompt,
        ]
    )

    harness.agent.agentic_loop_call(prompt="what does this project do")
    harness.agent.agentic_loop_call(prompt="now summarize it in one line")

    assert len(harness.agent.runs) == 2
    # The system prompt is sent exactly once per request, and names the workspace.
    for request in harness.llm.requests:
        assert len(request.system_prompts) == 1
        assert str(harness.workspace) in request.system_prompts[0]


def test_history_preserves_the_full_tool_transcript(build_agent: BuildAgent) -> None:
    """Tool calls and their results survive into later prompts, in order."""

    def later(request: LlmRequest) -> Turn:
        roles = [m["role"] for m in request.messages]
        # system, user, assistant(tool_calls), tool, assistant, user
        assert roles == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
        ]
        assert [r["success"] for r in request.tool_results] == [True]
        return says("Yes, the checks failed earlier because add() was wrong.")

    harness = build_agent(
        [
            calls(read("src/calculator.py")),
            says("add() subtracts."),
            later,
        ]
    )

    harness.agent.agentic_loop_call(prompt="look at the calculator")
    response = harness.agent.agentic_loop_call(prompt="what did you find?")

    assert "add() was wrong" in response.content


def test_clear_history_starts_a_fresh_conversation(build_agent: BuildAgent) -> None:
    def after_clear(request: LlmRequest) -> Turn:
        assert request.user_prompts == ["second question"]
        assert "first question" not in request.text()
        assert len(request.system_prompts) == 1
        return says("Fresh start.")

    harness = build_agent(
        [calls(read("README.md")), says("First answer."), after_clear]
    )

    harness.agent.agentic_loop_call(prompt="first question")
    harness.agent.clear_history()

    assert harness.agent.runs == []
    # Cleared history shrinks the estimate back to the system prompt + tools.
    assert harness.agent.estimate_context_tokens() < harness.agent.context

    harness.agent.agentic_loop_call(prompt="second question")


def test_model_switch_mid_session_keeps_history(build_agent: BuildAgent) -> None:
    harness = build_agent(
        [says("Answer from the first model."), says("Answer from the second model.")],
        model="test/model-one",
    )

    harness.agent.agentic_loop_call(prompt="first question")
    harness.agent.update_model(model="test/model-two", context_limit=400_000)
    harness.agent.agentic_loop_call(prompt="second question")

    assert harness.llm.models_used == ["test/model-one", "test/model-two"]
    assert harness.agent.context_limit == 400_000
    # Switching models does not drop what was said before it.
    assert "first question" in harness.llm.requests[-1].text()


# ---------------------------------------------------------------------------
# Token and context accounting
# ---------------------------------------------------------------------------


def test_session_token_totals_accumulate_across_prompts(
    build_agent: BuildAgent,
) -> None:
    harness = build_agent(
        [
            calls(read("README.md")),
            says("Answer one.", prompt_tokens=500, completion_tokens=40),
            says("Answer two.", prompt_tokens=700, completion_tokens=60),
        ]
    )

    first = harness.agent.agentic_loop_call(prompt="first")
    # Two LLM calls in run one: the default 100/20 turn plus the scripted 500/40.
    assert first.run_prompt_tokens == 600
    assert first.run_completion_tokens == 60

    second = harness.agent.agentic_loop_call(prompt="second")
    assert second.run_prompt_tokens == 700
    assert second.run_completion_tokens == 60

    assert harness.agent.session_prompt_tokens == 1300
    assert harness.agent.session_completion_tokens == 120


def test_context_estimate_tracks_the_real_conversation(
    build_agent: BuildAgent,
) -> None:
    harness = build_agent(
        [
            calls(bash([PYTHON, "-c", "print('x' * 4000)"])),
            says("That printed a long line."),
            says("Nothing more to add."),
        ]
    )

    baseline = harness.agent.estimate_context_tokens()  # system prompt + tool schema
    harness.agent.agentic_loop_call(prompt="print a long line")
    after_big_tool_output = harness.agent.context

    harness.agent.agentic_loop_call(prompt="anything else?")

    assert baseline > 0
    # A ~4KB command output is really carried in the estimate.
    assert after_big_tool_output > baseline + 500
    assert harness.agent.context > after_big_tool_output
    assert harness.agent.context == harness.agent.estimate_context_tokens()


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def test_compaction_is_skipped_while_the_context_is_small(
    build_agent: BuildAgent,
) -> None:
    harness = build_agent([says("Answer.")], context_limit=200_000)

    harness.agent.agentic_loop_call(prompt="a question")
    before = list(harness.agent.runs)

    result = harness.agent.compact_message_history()

    assert result.success is True
    assert harness.agent.compacted_message is None
    assert harness.agent.runs == before
    assert len(harness.llm.requests) == 1  # no summarization call was made


def test_compaction_replaces_history_with_a_summary_the_next_prompt_uses(
    build_agent: BuildAgent,
) -> None:
    def after_compaction(request: LlmRequest) -> Turn:
        # The summary is carried as a second system message, ahead of the prompt.
        assert len(request.system_prompts) == 2
        assert "SUMMARY: fixed add() in src/calculator.py" in request.system_prompts[1]
        # ...and the raw earlier turns are gone.
        assert "the calculator is broken" not in request.text()
        assert request.user_prompts == ["what is left to do?"]
        return says("Only the notes need updating.")

    harness = build_agent(
        [
            calls(read("src/calculator.py")),
            calls(write("src/calculator.py", "def add(a, b):\n    return a + b\n")),
            says("Fixed it."),
            says("SUMMARY: fixed add() in src/calculator.py"),  # the compaction call
            after_compaction,
        ]
    )

    harness.agent.agentic_loop_call(prompt="the calculator is broken, fix it")

    # Squeeze the context limit so the real 0.8 utilization trigger fires on the
    # history we just produced, rather than on hand-written numbers.
    harness.agent.context_limit = math.ceil(harness.agent.context / 0.9)

    result = harness.agent.compact_message_history()

    assert result.success is True
    assert result.run_prompt_tokens == 100 and result.run_completion_tokens == 20
    assert harness.agent.runs == []
    assert harness.agent.compacted_message is not None

    # The compaction call itself must not offer tools.
    compaction_request = harness.llm.requests[-1]
    assert compaction_request.tools is None
    assert "Summarize the conversation history" in compaction_request.system_prompts[0]

    harness.agent.agentic_loop_call(prompt="what is left to do?")


def test_compaction_failure_leaves_the_conversation_usable(
    build_agent: BuildAgent, mocker
) -> None:
    """If summarization blows up, history is preserved and the session continues."""
    harness = build_agent([says("First answer."), says("Second answer.")])

    harness.agent.agentic_loop_call(prompt="first")
    harness.agent.context_limit = math.ceil(harness.agent.context / 0.9)

    mocker.patch.object(
        harness.agent,
        "_llm_summarize_messages",
        side_effect=RuntimeError("upstream 500"),
    )

    result = harness.agent.compact_message_history()

    assert result.success is False
    assert result.failure_reason is not None
    assert len(harness.agent.runs) == 1
    assert harness.agent.compacted_message is None

    # The next prompt still carries the un-compacted history.
    harness.agent.agentic_loop_call(prompt="second")
    assert "first" in harness.llm.requests[-1].text()


def test_recompaction_folds_the_previous_summary_in(build_agent: BuildAgent) -> None:
    def second_summary(request: LlmRequest) -> Turn:
        # The old summary is part of what gets re-summarized.
        assert "SUMMARY ONE" in request.text()
        return says("SUMMARY TWO")

    harness = build_agent(
        [
            says("First answer."),
            says("SUMMARY ONE"),
            says("Second answer."),
            second_summary,
        ]
    )

    harness.agent.agentic_loop_call(prompt="first")
    harness.agent.context_limit = math.ceil(harness.agent.context / 0.9)
    harness.agent.compact_message_history()

    harness.agent.agentic_loop_call(prompt="second")
    harness.agent.compact_message_history()

    summary = harness.agent.compacted_message
    assert summary is not None
    assert "SUMMARY TWO" in str(summary["content"])
    assert "SUMMARY ONE" not in str(summary["content"])


def test_workspace_is_whatever_directory_the_agent_was_pointed_at(
    build_agent: BuildAgent, tmp_path: Path
) -> None:
    """Two agents on two workspaces stay in their own sandbox."""
    other = tmp_path / "other-project"
    other.mkdir()
    (other / "README.md").write_text("# other project\n")

    def check(request: LlmRequest) -> Turn:
        assert "# other project" in request.last_tool_result["output"]
        assert str(other) in request.system_prompts[0]
        return says("Read the other project's readme.")

    harness = build_agent([calls(read("README.md")), check], root=other)

    harness.agent.agentic_loop_call(prompt="read the readme")
