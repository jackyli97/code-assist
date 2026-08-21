"""Shared fixtures for the integration suite.

These tests wire the *real* pieces together — the agentic loop, the real tool
implementations, real files on a real temporary workspace, and (in test_cli.py)
the real Click entrypoint. Only two things are faked:

  * the LLM itself, via :class:`support.ScriptedLLM`, a stand-in for ``OpenAI``
    that replays a scripted list of turns and records every request it received;
  * the OpenRouter model lookup, which would otherwise make a live HTTP call.

Everything else — path safety, permission prompting, subprocess execution,
message-history assembly, token accounting — runs for real.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest
from pytest_mock import MockerFixture

from ai_coding_assistant.agents import LlmAgent
from ai_coding_assistant.models import ModelLookup
from ai_coding_assistant.tools import get_tools

from tests.integration.support import (
    DEFAULT_CONTEXT_LIMIT,
    DEFAULT_MODEL,
    OUTSIDE_SECRET,
    AgentHarness,
    ScriptedLLM,
    ScriptedTurn,
    build_workspace,
)


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_live_http(mocker: MockerFixture) -> None:
    """Fail loudly if anything in the integration suite reaches the network."""
    mocker.patch(
        "ai_coding_assistant.models.requests.get",
        side_effect=AssertionError(
            "integration tests must not make live HTTP calls to OpenRouter"
        ),
    )

@pytest.fixture(autouse=True)
def fake_model_lookup(mocker: MockerFixture) -> None:
    mocker.patch(
        "ai_coding_assistant.cli_runner.lookup_model",
        return_value=ModelLookup(
            found=True,
            context_limit=200_000,
        ),
    )


@pytest.fixture(autouse=True)
def stub_model_lookup(mocker: MockerFixture) -> None:
    """Stub the model lookup at every import site that uses it."""
    lookup = ModelLookup(found=True, context_limit=DEFAULT_CONTEXT_LIMIT)
    for module in ("agents", "cli_runner", "config"):
        mocker.patch(f"ai_coding_assistant.{module}.lookup_model", return_value=lookup)


@pytest.fixture(autouse=True)
def isolated_openrouter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real OPENROUTER_* environment out of the tests.

    ``load_dotenv`` mutates ``os.environ`` for the rest of the process, so this
    also stops values loaded by one test from leaking into the next.
    """
    for key in list(os.environ):
        if key.startswith("OPENROUTER_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@pytest.fixture
def outside_dir(tmp_path: Path) -> Path:
    """A directory that is a sibling of — not inside — the workspace."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(OUTSIDE_SECRET)
    return outside


@pytest.fixture
def workspace(tmp_path: Path, outside_dir: Path) -> Path:
    return build_workspace(tmp_path / "project")


# ---------------------------------------------------------------------------
# Permission prompting
# ---------------------------------------------------------------------------


@pytest.fixture
def answer_permissions(mocker: MockerFixture) -> Callable[..., list[str]]:
    """Drive the real permission prompt without a terminal.

    Patches ``click.prompt`` as seen by ``agents`` so
    ``_request_permission_for_tool`` — including its ``PermissionChoice``
    parsing — runs for real. Returns the list of prompt texts, which grows as
    prompts are answered.
    """

    def install(*answers: str, always: str | None = None) -> list[str]:
        queued = deque(answers)
        asked: list[str] = []

        def fake_prompt(text: str, **_: Any) -> str:
            asked.append(text)
            if queued:
                return queued.popleft()
            if always is not None:
                return always
            raise AssertionError(
                f"unexpected permission prompt (#{len(asked)}): {text!r}"
            )

        mocker.patch("ai_coding_assistant.agents.click.prompt", side_effect=fake_prompt)
        return asked

    return install


# ---------------------------------------------------------------------------
# Agent harness
# ---------------------------------------------------------------------------


@pytest.fixture
def build_agent(
    workspace: Path, answer_permissions: Callable[..., list[str]]
) -> Callable[..., AgentHarness]:
    """Build an agent with the real tool set, pointed at the temp workspace.

    ``permissions`` is the standing answer for every permission prompt
    (``None`` means "any prompt is a test failure"); ``permission_answers``
    supplies a queue of answers consumed before the standing one.
    """

    def build(
        script: Iterable[ScriptedTurn],
        *,
        permissions: str | None = "y",
        permission_answers: tuple[str, ...] = (),
        model: str = DEFAULT_MODEL,
        context_limit: int | None = DEFAULT_CONTEXT_LIMIT,
        root: Path | None = None,
    ) -> AgentHarness:
        prompts = answer_permissions(*permission_answers, always=permissions)
        llm = ScriptedLLM(script)
        agent = LlmAgent(
            client=llm,  # type: ignore[arg-type]
            tools=get_tools(),
            workspace=root or workspace,
            model=model,
            context_limit=context_limit,
        )
        return AgentHarness(
            agent=agent, llm=llm, workspace=root or workspace, permission_prompts=prompts
        )

    return build
