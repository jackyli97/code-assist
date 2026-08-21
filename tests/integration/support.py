"""Test doubles and builders shared by the integration suite.

Kept out of ``conftest.py`` so test modules can import these helpers directly
without going through pytest's plugin module.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from ai_coding_assistant.agents import LlmAgent

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_CONTEXT_LIMIT = 200_000
PYTHON = sys.executable
WORKSPACE_SECRET = "API_SECRET=super-secret-workspace-value"
OUTSIDE_SECRET = "outside-the-workspace-secret-value"


# ---------------------------------------------------------------------------
# Recorded requests
# ---------------------------------------------------------------------------


@dataclass
class LlmRequest:
    """One recorded call to ``client.chat.completions.create``."""

    model: str
    messages: list[dict[str, Any]]
    tools: Any | None

    @property
    def system_prompts(self) -> list[str]:
        return [
            str(m.get("content", "")) for m in self.messages if m.get("role") == "system"
        ]

    @property
    def user_prompts(self) -> list[str]:
        return [
            str(m.get("content", "")) for m in self.messages if m.get("role") == "user"
        ]

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        """Tool result payloads (``{"success": ..., "output": ...}``) in order."""
        return [
            json.loads(str(m["content"]))
            for m in self.messages
            if m.get("role") == "tool"
        ]

    @property
    def last_tool_result(self) -> dict[str, Any]:
        results = self.tool_results
        if not results:
            raise AssertionError("no tool results in this request's message history")
        return results[-1]

    def text(self) -> str:
        return json.dumps(self.messages, default=str)


# ---------------------------------------------------------------------------
# Scripted turns
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single scripted assistant response."""

    content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] | None = None
    prompt_tokens: int = 100
    completion_tokens: int = 20

    def to_completion(self, index: int) -> ChatCompletion:
        return ChatCompletion(
            id=f"chatcmpl-{index}",
            model="scripted",
            object="chat.completion",
            created=1710000000,
            choices=[
                Choice(
                    finish_reason="tool_calls" if self.tool_calls else "stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=self.content,
                        tool_calls=self.tool_calls,
                    ),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
        )


# A scripted turn is either a fixed Turn or a callable that inspects the request
# the agent just sent and decides how to respond — which is how these tests
# assert that tool output actually made it back to the model.
ScriptedTurn = Turn | Callable[[LlmRequest], Turn]

_tool_call_ids = count()


def use_tool(name: str, **arguments: Any) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=f"call_{name.lower()}_{next(_tool_call_ids)}",
        type="function",
        function=Function(name=name, arguments=json.dumps(arguments)),
    )


def read(file_path: str, **kwargs: Any) -> ChatCompletionMessageToolCall:
    return use_tool("Read", file_path=file_path, **kwargs)


def write(file_path: str, content: str) -> ChatCompletionMessageToolCall:
    return use_tool("Write", file_path=file_path, content=content)


def bash(command: list[str], cwd: str = ".") -> ChatCompletionMessageToolCall:
    return use_tool("Bash", command=command, cwd=cwd)


def calls(*tool_calls: ChatCompletionMessageToolCall) -> Turn:
    """A turn in which the assistant asks for one or more tool calls."""
    return Turn(content=None, tool_calls=list(tool_calls))


def says(content: str, **kwargs: Any) -> Turn:
    """A terminal turn: plain assistant text, no tool calls."""
    return Turn(content=content, **kwargs)


# ---------------------------------------------------------------------------
# The fake LLM client
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Minimal stand-in for ``OpenAI`` that replays ``script`` in order."""

    def __init__(self, script: Iterable[ScriptedTurn] = ()) -> None:
        self.script: deque[ScriptedTurn] = deque(script)
        self.requests: list[LlmRequest] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.init_kwargs: dict[str, Any] = {}

    def extend(self, script: Iterable[ScriptedTurn]) -> None:
        self.script.extend(script)

    def _create(
        self,
        *,
        model: str,
        messages: Iterable[Any],
        tools: Any | None = None,
        **_: Any,
    ) -> ChatCompletion:
        request = LlmRequest(
            model=model, messages=[dict(m) for m in messages], tools=tools
        )
        self.requests.append(request)

        if not self.script:
            raise AssertionError(
                f"ScriptedLLM ran out of turns on request #{len(self.requests)}; "
                "the agent asked for more LLM calls than the test scripted"
            )

        turn = self.script.popleft()
        if callable(turn):
            turn = turn(request)
        return turn.to_completion(len(self.requests))

    # -- convenience accessors used by assertions -------------------------

    @property
    def models_used(self) -> list[str]:
        return [request.model for request in self.requests]

    @property
    def tool_names_requested(self) -> list[str]:
        """Tool names the agent executed, in order, deduped by call id."""
        names: list[str] = []
        seen: set[str] = set()
        for request in self.requests:
            for message in request.messages:
                for tool_call in message.get("tool_calls") or []:
                    if tool_call["id"] in seen:
                        continue
                    seen.add(tool_call["id"])
                    names.append(tool_call["function"]["name"])
        return names

    def all_text(self) -> str:
        """Everything ever sent to the model, serialized."""
        return json.dumps([r.messages for r in self.requests], default=str)


# ---------------------------------------------------------------------------
# Workspace fixtures content
# ---------------------------------------------------------------------------


CALCULATOR_WITH_BUG = '''\
"""Tiny arithmetic helpers."""


def add(a, b):
    return a - b


def subtract(a, b):
    return a - b
'''

# NOTE: the fixed source must differ in *size* from the buggy one, not just in
# content. CPython validates __pycache__ entries on (mtime, size), and a
# same-second rewrite of the same length would silently re-import the buggy
# bytecode when the checks are re-run.
CALCULATOR_FIXED = '''\
"""Tiny arithmetic helpers (add() fixed)."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
'''

CHECK_SCRIPT = '''\
import sys

sys.path.insert(0, "src")

from calculator import add, subtract

assert add(2, 3) == 5, f"add(2, 3) returned {add(2, 3)}"
assert subtract(5, 2) == 3, f"subtract(5, 2) returned {subtract(5, 2)}"
print("all checks passed")
'''


def build_workspace(root: Path) -> Path:
    """Create a small but realistic project for the agent to operate on."""
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "notes").mkdir()

    (root / "README.md").write_text(
        "# demo project\n\nA calculator library used by the integration suite.\n"
    )
    (root / "src" / "calculator.py").write_text(CALCULATOR_WITH_BUG)
    (root / "tests" / "check_calculator.py").write_text(CHECK_SCRIPT)
    (root / "notes" / "todo.md").write_text("- fix the add() bug\n")
    (root / ".env").write_text(f"{WORKSPACE_SECRET}\n")

    return root


# ---------------------------------------------------------------------------
# Agent harness
# ---------------------------------------------------------------------------


@dataclass
class AgentHarness:
    agent: LlmAgent
    llm: ScriptedLLM
    workspace: Path
    permission_prompts: list[str] = field(default_factory=list)
