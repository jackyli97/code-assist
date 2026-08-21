from ai_coding_assistant.agents import LlmAgent, AgentResponse, PermissionChoice  # noqa: F401

import json
import pytest
import tiktoken
from pathlib import Path
from unittest.mock import MagicMock
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from pytest_mock import MockerFixture
import random
from ai_coding_assistant.tools import TOOL_REGISTRY, ToolCallResult
from ai_coding_assistant.models import ModelLookup


@pytest.fixture
def stub_context_limit(mocker: MockerFixture) -> None:
    """Prevent LlmAgent.__init__ from making a live HTTP call to OpenRouter."""
    mocker.patch(
        "ai_coding_assistant.agents.lookup_model",
        return_value=ModelLookup(
            found=True,
            context_limit=200_000
        ),
    )

@pytest.fixture
def stub_permission_request_returns_yes(mocker: MockerFixture) -> None:
    mocker.patch.object(
        LlmAgent, "_request_permission_for_tool",
        return_value=PermissionChoice(
            "y"
        )
    )

@pytest.fixture
def stub_permission_request_returns_no(mocker: MockerFixture) -> None:
    mocker.patch.object(
        LlmAgent, "_request_permission_for_tool",
        return_value=PermissionChoice(
            "n"
        )
    )

@pytest.fixture
def stub_permission_request_returns_session(mocker: MockerFixture) -> None:
    mocker.patch.object(
        LlmAgent, "_request_permission_for_tool",
        return_value=PermissionChoice(
            "s"
        )
    )

def create_mock_completion(id_str: str, message: ChatCompletionMessage) -> ChatCompletion:
    prompt_tokens=random.randint(150,300)
    completion_tokens=random.randint(300,500)
    return ChatCompletion(
        id=id_str,
        model="gpt-4o",
        object="chat.completion",
        created=1710000000,
        choices=[
            Choice(
                finish_reason="tool_calls" if message.tool_calls else "stop",
                index=0,
                message=message
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens+completion_tokens
        )
    )

@pytest.fixture
def mock_read_tool_call():
    return ChatCompletionMessageToolCall(
        id="call_read_0",
        type="function",
        function=Function(name="Read", arguments='{"file_path": "something"}')
    )

@pytest.fixture
def mock_write_tool_call():
    return ChatCompletionMessageToolCall(
        id="call_write_0",
        type="function",
        function=Function(name="Write", arguments='{"file_path": "something", "content": "something"}')
    )

@pytest.fixture
def mock_bash_tool_call():
    return ChatCompletionMessageToolCall(
        id="call_bash_0",
        type="function",
        function=Function(name="Bash", arguments='{"command": ["something"], "cwd": "something"}')
    )

@pytest.fixture
def mock_no_tool_call_openai():
    client = MagicMock()
    
    response_with_message = create_mock_completion(
        "chatcmpl-1", 
        ChatCompletionMessage(role="assistant", content="this is what this repository does", tool_calls=None)
    )
    
    client.chat.completions.create.return_value = response_with_message
    return client

@pytest.fixture
def mock_single_tool_call_openai(mock_read_tool_call: ChatCompletionMessageToolCall):
    client = MagicMock()
    
    response_1 = create_mock_completion(
        "chatcmpl-1", 
        ChatCompletionMessage(role="assistant", content=None, tool_calls=[mock_read_tool_call])
    )
    response_2 = create_mock_completion(
        "chatcmpl-2", 
        ChatCompletionMessage(role="assistant", content="summary of the repository", tool_calls=None)
    )
    
    client.chat.completions.create.side_effect = [response_1,response_2]
    return client

@pytest.fixture
def mock_multi_call_openai(
    mock_read_tool_call: ChatCompletionMessageToolCall,
    mock_bash_tool_call: ChatCompletionMessageToolCall
):
    client = MagicMock()
    
    response_1 = create_mock_completion(
        "chatcmpl-1", 
        ChatCompletionMessage(role="assistant", content=None, tool_calls=[mock_read_tool_call])
    )
    
    response_2 = create_mock_completion(
        "chatcmpl-2", 
        ChatCompletionMessage(role="assistant", content=None, tool_calls=[mock_bash_tool_call])
    )
    
    response_3 = create_mock_completion(
        "chatcmpl-3", 
        ChatCompletionMessage(role="assistant", content="summary of the repository", tool_calls=None)
    )

    responses = [response_1, response_2, response_3]
    client.chat.completions.create.side_effect = responses
    client.mock_responses = responses
    return client

@pytest.fixture
def mock_multi_tools_in_single_response_openai(
    mock_read_tool_call: ChatCompletionMessageToolCall,
    mock_write_tool_call: ChatCompletionMessageToolCall
):
    client = MagicMock()
    
    response_1 = create_mock_completion(
        "chatcmpl-1", 
        ChatCompletionMessage(role="assistant", content=None, tool_calls=[mock_read_tool_call, mock_write_tool_call])
    )
    
    response_2 = create_mock_completion(
        "chatcmpl-3", 
        ChatCompletionMessage(role="assistant", content="summary of the repository", tool_calls=None)
    )

    client.chat.completions.create.side_effect = [response_1, response_2]
    return client

@pytest.fixture
def mock_max_iterations_call_openai(
):
    client = MagicMock()
    responses: list[ChatCompletion] = []
    tools = [
        ("read", {"name": "Read", "arguments": '{"file_path": "something"}'}),
        ("write", {"name": "Write", "arguments": '{"file_path": "something", "content": "something"}'}),
        ("bash", {"name": "Bash", "arguments": '{"command": ["something"], "cwd": "something"}'}),
    ]

    for i in range(11):
        rand_tool = random.randint(0,2)
        tool_call = (
            ChatCompletionMessageToolCall(
                id=f"call_{tools[rand_tool][0]}_{i}",
                type="function",
                function=Function(name=tools[rand_tool][1]["name"], arguments=tools[rand_tool][1]["arguments"])
            )
        )
        responses.append(
            create_mock_completion(
            f"chatcmpl-{i}", 
            ChatCompletionMessage(role="assistant", content=None, tool_calls=[tool_call])
            )
        )

    client.chat.completions.create.side_effect = responses
    return client


def test_agentic_loop_call_no_tool_calls(mock_no_tool_call_openai: MagicMock, tmp_path: Path, mocker: MockerFixture):
    agent = LlmAgent(client=mock_no_tool_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)

    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do")
    assert mock_no_tool_call_openai.chat.completions.create.call_count == 1
    assert mock_execute_tool.call_count == 0

def test_agentic_loop_call_single_tool_call(
        mock_single_tool_call_openai: MagicMock, 
        tmp_path: Path,
        mocker: MockerFixture,
        mock_read_tool_call: ChatCompletionMessageToolCall
    ):
    agent = LlmAgent(client=mock_single_tool_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do")

    assert mock_single_tool_call_openai.chat.completions.create.call_count == 2
    assert mock_execute_tool.call_count == 1
    mock_execute_tool_call = mock_execute_tool.call_args_list[0]
    assert mock_execute_tool_call.kwargs['tool_call'].id == mock_read_tool_call.id

def test_agentic_loop_call_multi_tool_call(
        mock_multi_call_openai: MagicMock, 
        tmp_path: Path,
        mocker: MockerFixture,
        mock_read_tool_call: ChatCompletionMessageToolCall,
        mock_bash_tool_call: ChatCompletionMessageToolCall
    ):
    agent = LlmAgent(client=mock_multi_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do")

    assert mock_multi_call_openai.chat.completions.create.call_count == 3
    assert mock_execute_tool.call_count == 2

    for tool_call in mock_execute_tool.call_args_list:
        assert tool_call.kwargs['tool_call'].id in [
            mock_read_tool_call.id,
            mock_bash_tool_call.id
        ]

def test_agentic_loop_call_multi_tools_at_once(
        mock_multi_tools_in_single_response_openai: MagicMock, 
        tmp_path: Path,
        mocker: MockerFixture,
        mock_read_tool_call: ChatCompletionMessageToolCall,
        mock_write_tool_call: ChatCompletionMessageToolCall
    ):
    agent = LlmAgent(client=mock_multi_tools_in_single_response_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do")

    assert mock_multi_tools_in_single_response_openai.chat.completions.create.call_count == 2
    assert mock_execute_tool.call_count == 2
    for tool_call in mock_execute_tool.call_args_list:
        assert tool_call.kwargs['tool_call'].id in [
            mock_read_tool_call.id,
            mock_write_tool_call.id
        ]

def test_agentic_loop_call_max_iterations_reached(
    mock_max_iterations_call_openai: MagicMock,
    tmp_path: Path,
    mocker: MockerFixture,
):
    agent = LlmAgent(client=mock_max_iterations_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )
    with pytest.raises(RuntimeError, match="Agent exceeded max iterations"):
        agent.agentic_loop_call(prompt="what does this repository do")

    assert mock_max_iterations_call_openai.chat.completions.create.call_count == 11 # don't include initial call as an iteration
    assert mock_execute_tool.call_count == 10

@pytest.mark.parametrize(
    "tool_call_fixture_name",
    [
        "mock_read_tool_call",
        "mock_write_tool_call",
        "mock_bash_tool_call"
    ],
)
def test_execute_call_success(
    tool_call_fixture_name: str, 
    tmp_path: Path, 
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    stub_permission_request_returns_yes: None,
):
    tool_call: ChatCompletionMessageToolCall = request.getfixturevalue(tool_call_fixture_name)
    function_name = tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))

    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    result = agent.execute_tool(tool_call)
    assert result.success

def test_execute_call_fails_invalid_tool(tmp_path: Path,  stub_permission_request_returns_yes: None):
    invalid_tool_call: ChatCompletionMessageToolCall = ChatCompletionMessageToolCall(
        id="call_read_0",
        type="function",
        function=Function(name="Delete", arguments='{"file_path": "something"}')
    )
    client = MagicMock()

    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    result = agent.execute_tool(invalid_tool_call)

    assert not result.success
    assert "Unknown tool" in result.output

def test_execute_call_fails_model_validate_json(
    mock_read_tool_call: ChatCompletionMessageToolCall,
    tmp_path: Path,
    stub_permission_request_returns_yes: None,
):
    client = MagicMock()
    
    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    mock_read_tool_call.function.arguments = '{"file_paths": "something"}' #typo

    result = agent.execute_tool(mock_read_tool_call)

    assert not result.success
    assert "validation errors" in result.output


# ---------------------------------------------------------------------------
# Message history
# ---------------------------------------------------------------------------

def test_message_history_preserves_across_agent_calls(
    mock_multi_call_openai: MagicMock,
    tmp_path: Path,
    mocker: MockerFixture,    
):
    agent = LlmAgent(client=mock_multi_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do")

    assert len(agent.runs) == 1
    assert len(agent.runs[0])  == len(agent.build_message_history()) - 1 # subtract the system prompt

    # second run on the same agent: reset the side effect using saved mock responses
    mock_multi_call_openai.chat.completions.create.side_effect = mock_multi_call_openai.mock_responses
    agent.agentic_loop_call(prompt="what does this repository do")

    assert len(agent.runs) == 2
    assert len(agent.runs[0]) + len(agent.runs[1]) == len(agent.build_message_history()) - 1 # subtract the system prompt


# ---------------------------------------------------------------------------
# Token tracking — agent.context is refreshed at the end of every
# agentic_loop_call via _update_agent_usages -> updated token usages.
# ---------------------------------------------------------------------------

def test_token_usage(
    mock_multi_call_openai: MagicMock,
    tmp_path: Path,
    mocker: MockerFixture,
):
    agent = LlmAgent(client=mock_multi_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    expected_prompt_tokens = sum(
        r.usage.prompt_tokens for r in mock_multi_call_openai.mock_responses if r.usage
    )
    expected_completion_tokens = sum(
        r.usage.completion_tokens for r in mock_multi_call_openai.mock_responses if r.usage
    )

    result = agent.agentic_loop_call(prompt="what does this repository do")

    assert result.run_prompt_tokens == expected_prompt_tokens
    assert result.run_completion_tokens == expected_completion_tokens
    assert agent.session_prompt_tokens == expected_prompt_tokens
    assert agent.session_completion_tokens == expected_completion_tokens

    # second run on the same agent: reset the side effect using saved mock responses
    mock_multi_call_openai.chat.completions.create.side_effect = mock_multi_call_openai.mock_responses
    result_2 = agent.agentic_loop_call(prompt="what does this repository do")

    assert result_2.run_prompt_tokens == expected_prompt_tokens
    assert result_2.run_completion_tokens == expected_completion_tokens
    assert agent.session_prompt_tokens == expected_prompt_tokens * 2
    assert agent.session_completion_tokens == expected_completion_tokens * 2


# ---------------------------------------------------------------------------
# Context tracking — agent.context is refreshed at the end of every
# agentic_loop_call via _update_agent_usages -> estimate_context_tokens.
# ---------------------------------------------------------------------------


def test_agentic_loop_call_updates_context_from_zero(
    stub_context_limit: None,
    mock_no_tool_call_openai: MagicMock,
    tmp_path: Path,
) -> None:
    agent = LlmAgent(client=mock_no_tool_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)

    assert agent.context == 0

    agent.agentic_loop_call(prompt="hello")

    assert agent.context > 0


def test_agentic_loop_call_context_grows_across_runs(
    stub_context_limit: None,
    mock_multi_call_openai: MagicMock,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    # agent.messages persists across loop calls, so a second run should
    # produce a strictly larger context estimate than the first.
    agent = LlmAgent(client=mock_multi_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)
    mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="first prompt")
    first_context = agent.context

    mock_multi_call_openai.chat.completions.create.side_effect = (
        mock_multi_call_openai.mock_responses
    )
    agent.agentic_loop_call(prompt="second prompt")
    second_context = agent.context

    assert first_context > 0
    assert second_context > first_context


def test_agentic_loop_call_context_matches_direct_estimation(
    stub_context_limit: None,
    mock_no_tool_call_openai: MagicMock,
    tmp_path: Path,
) -> None:
    # The context stored on the agent after a run should equal a fresh call
    # to estimate_context_tokens on the same messages/tools — proves the
    # update path in _update_agent_usages is wired to the same estimator.
    agent = LlmAgent(client=mock_no_tool_call_openai, workspace=tmp_path, tools=[MagicMock()]*3)

    agent.agentic_loop_call(prompt="hello")

    assert agent.context == agent.estimate_context_tokens()


# ---------------------------------------------------------------------------
# estimate_context_tokens — direct unit tests.
# ---------------------------------------------------------------------------


def test_estimate_context_tokens_empty_state_is_small_but_nonzero(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    # With no messages and no tools we're just encoding "[system prompt message]" + "[]".
    # Should be equal to the system prompt serialized
    encoding = tiktoken.get_encoding("o200k_base")
    agent = LlmAgent(client=MagicMock(), workspace=tmp_path, tools=[])
    system_message = agent._get_system_message()
    system_message_json = agent._serialize_json([system_message])
    empty_tools_json = agent._serialize_json([])
    system_message_empty_tools_tokens = len(encoding.encode(system_message_json)) + len(encoding.encode(empty_tools_json))

    tokens = agent.estimate_context_tokens()

    assert tokens == system_message_empty_tools_tokens


def test_estimate_context_tokens_grows_with_more_messages(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    agent = LlmAgent(client=MagicMock(), workspace=tmp_path, tools=[MagicMock()]*3)

    agent.runs = [[{"role": "user", "content": "hi"}]]
    small = agent.estimate_context_tokens()

    agent.runs[0].append(
        {"role": "assistant", "content": "a much longer response " * 100}
    )
    big = agent.estimate_context_tokens()

    assert big > small


def test_estimate_context_tokens_grows_with_tools(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    agent_without_tools = LlmAgent(client=MagicMock(), workspace=tmp_path, tools=[])
    agent_with_tools = LlmAgent(client=MagicMock(), workspace=tmp_path, tools=[MagicMock()]*3)
    without = agent_without_tools.estimate_context_tokens()
    with_tools = agent_with_tools.estimate_context_tokens()

    assert with_tools > without


# ---------------------------------------------------------------------------
# _llm_summarize_messages — thin wrapper around client.chat.completions.create
# with a compaction prompt. Tests focus on the response-shape branching.
# ---------------------------------------------------------------------------


def _make_agent(client: MagicMock, tmp_path: Path, tools: list | None = None) -> LlmAgent:
    return LlmAgent(client=client, tools=tools or [], workspace=tmp_path)


def test_llm_summarize_messages_returns_content_on_success(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = create_mock_completion(
        "chatcmpl-summary",
        ChatCompletionMessage(role="assistant", content="short summary", tool_calls=None),
    )
    agent = _make_agent(client, tmp_path)

    result = agent._llm_summarize_messages("some serialized history")

    assert result.content == "short summary"
    # single LLM call, no tools arg (compaction shouldn't invoke tools)
    assert client.chat.completions.create.call_count == 1
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert "tools" not in call_kwargs


def test_llm_summarize_messages_raises_when_no_choices(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    client = MagicMock()
    empty_response = create_mock_completion(
        "chatcmpl-empty",
        ChatCompletionMessage(role="assistant", content="ignored", tool_calls=None),
    )
    empty_response.choices = []
    client.chat.completions.create.return_value = empty_response
    agent = _make_agent(client, tmp_path)

    with pytest.raises(RuntimeError, match="no choices"):
        agent._llm_summarize_messages("history")


def test_llm_summarize_messages_raises_when_empty_content(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = create_mock_completion(
        "chatcmpl-nocontent",
        ChatCompletionMessage(role="assistant", content=None, tool_calls=None),
    )
    agent = _make_agent(client, tmp_path)

    with pytest.raises(RuntimeError, match="no content"):
        agent._llm_summarize_messages("history")


# ---------------------------------------------------------------------------
# compact_message_history — orchestration around _llm_summarize_messages.
# Patch _llm_summarize_messages in most tests so we test state transitions,
# not the LLM branching (covered above).
# ---------------------------------------------------------------------------


def _sample_run(idx: int) -> list:
    """Build a small canned run: user prompt + assistant response."""
    return [
        {"role": "user", "content": f"prompt {idx}"},
        {"role": "assistant", "content": f"response {idx}"},
    ]


def test_compact_returns_failure_when_no_context_limit(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = None  # override the stubbed value
    agent.context = 1_000_000  # doesn't matter — the None check runs first

    result = agent.compact_message_history()

    assert result.success is False
    assert result.failure_reason is not None
    assert "context limit" in result.failure_reason.lower()


def test_compact_is_noop_when_below_utilization_threshold(
    stub_context_limit: None, tmp_path: Path, mocker: MockerFixture,
) -> None:
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 50_000  # 50% utilization, well below the 0.8 trigger
    agent.runs = [_sample_run(0)]
    original_runs = list(agent.runs)

    spy = mocker.spy(agent, "_llm_summarize_messages")
    result = agent.compact_message_history()

    assert result.success is True
    assert result.failure_reason is None
    assert spy.call_count == 0                # no LLM call fired
    assert agent.runs == original_runs         # state unchanged
    assert agent.compacted_message is None


def test_compact_returns_failure_when_over_threshold_but_no_runs(
    stub_context_limit: None, tmp_path: Path,
) -> None:
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 90_000
    agent.runs = []

    result = agent.compact_message_history()

    assert result.success is False
    assert result.failure_reason is not None
    assert "not enough messages" in result.failure_reason.lower()


def test_compact_summarizes_and_reduces_runs_on_happy_path(
    stub_context_limit: None, tmp_path: Path, mocker: MockerFixture,
) -> None:
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 90_000  # 90% utilization → triggers compaction
    agent.runs = [_sample_run(0), _sample_run(1), _sample_run(2)]

    mock_summarize = mocker.patch.object(
        agent, "_llm_summarize_messages",
        return_value=AgentResponse(
            content="condensed summary text"
        )
    )

    result = agent.compact_message_history()

    assert result.success is True
    assert mock_summarize.call_count == 1
    # compacted_message installed with the returned summary embedded
    assert agent.compacted_message is not None
    assert agent.compacted_message["role"] == "system"
    assert type(agent.compacted_message["content"]) == str and "condensed summary text" in agent.compacted_message["content"]
    # at least the first run was consumed; runs list is strictly shorter
    assert len(agent.runs) < 3


def test_compact_returns_failure_when_llm_summarize_raises(
    stub_context_limit: None, tmp_path: Path, mocker: MockerFixture,
) -> None:
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 95_000
    agent.runs = [_sample_run(0), _sample_run(1)]

    mocker.patch.object(
        agent, "_llm_summarize_messages",
        side_effect=RuntimeError("simulated LLM failure"),
    )

    result = agent.compact_message_history()

    assert result.success is False
    assert result.failure_reason is not None
    assert "error" in result.failure_reason.lower()
    # state should NOT mutate on failure
    assert agent.compacted_message is None
    assert len(agent.runs) == 2


def test_compact_consumes_multiple_runs_when_needed(
    stub_context_limit: None, tmp_path: Path, mocker: MockerFixture,
) -> None:
    # With multiple runs and high utilization, the loop should consume at
    # least one run. Exact count depends on per-run token size, so assert
    # relative behavior: runs list shrinks and compacted_message is set.
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 95_000
    agent.runs = [_sample_run(i) for i in range(5)]

    mocker.patch.object(
        agent, "_llm_summarize_messages", return_value=AgentResponse(
            content="summary"
        ),
    )

    result = agent.compact_message_history()

    assert result.success is True
    assert agent.compacted_message is not None
    assert len(agent.runs) < 5


def test_compact_includes_existing_summary_when_recompacting(
    stub_context_limit: None, tmp_path: Path, mocker: MockerFixture,
) -> None:
    # If a compacted_message already exists (from a prior compaction), it
    # should be included in the payload sent to _llm_summarize_messages so
    # the new summary can subsume the old one.
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 95_000
    agent.compacted_message = {
        "role": "system",
        "content": "Summary of earlier conversation history:\n\nold summary content",
    }
    agent.runs = [_sample_run(0)]

    mock_summarize = mocker.patch.object(
        agent, "_llm_summarize_messages", return_value=AgentResponse(
            content="new combined summary"
        ),
    )

    result = agent.compact_message_history()

    assert result.success is True
    # the payload passed to the LLM includes the old summary
    (payload,) = mock_summarize.call_args.args
    assert "old summary content" in payload
    # and the new summary replaces the old one on the agent
    assert type(agent.compacted_message["content"]) == str and "new combined summary" in agent.compacted_message["content"]


def test_compact_adds_llm_tokens_to_session_totals(
    stub_context_limit: None, tmp_path: Path, mocker: MockerFixture,
) -> None:
    agent = _make_agent(MagicMock(), tmp_path)
    agent.context_limit = 100_000
    agent.context = 95_000
    agent.runs = [_sample_run(0)]
    agent.session_prompt_tokens = 500
    agent.session_completion_tokens = 200

    mocker.patch.object(
        agent, "_llm_summarize_messages",
        return_value=AgentResponse(
            content="summary", run_prompt_tokens=100, run_completion_tokens=50,
        ),
    )

    result = agent.compact_message_history()

    assert agent.session_prompt_tokens == 600
    assert agent.session_completion_tokens == 250
    # also surfaced on the result for callers to display
    assert result.run_prompt_tokens == 100
    assert result.run_completion_tokens == 50

@pytest.mark.parametrize(
    "tool_call_fixture_name",
    [
        "mock_read_tool_call",
        "mock_write_tool_call",
        "mock_bash_tool_call"
    ],
)
def test_execute_call_with_rejected_permissions(
    tool_call_fixture_name: str, 
    tmp_path: Path, 
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    stub_permission_request_returns_no: None,
):
    tool_call: ChatCompletionMessageToolCall = request.getfixturevalue(tool_call_fixture_name)
    function_name = tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mock_tool_call = mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))

    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    result = agent.execute_tool(tool_call)

    if tool.requires_permissions:
        assert not result.success
        assert mock_tool_call.call_count == 0
    else:
        assert result.success
        assert mock_tool_call.call_count == 1

    # ensure permission not granted for session
    assert not agent.granted_tool_permissions[tool]

@pytest.mark.parametrize(
    "tool_call_fixture_name",
    [
        "mock_read_tool_call",
        "mock_write_tool_call",
        "mock_bash_tool_call"
    ],
)
def test_execute_call_with_granted_permissions(
    tool_call_fixture_name: str, 
    tmp_path: Path, 
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    stub_permission_request_returns_yes: None,
):
    tool_call: ChatCompletionMessageToolCall = request.getfixturevalue(tool_call_fixture_name)
    function_name = tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mock_tool_call = mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))

    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    result = agent.execute_tool(tool_call)

    assert result.success
    assert mock_tool_call.call_count == 1

    # ensure permission not granted for session
    assert not agent.granted_tool_permissions[tool]

@pytest.mark.parametrize(
    "tool_call_fixture_name",
    [
        "mock_read_tool_call",
        "mock_write_tool_call",
        "mock_bash_tool_call"
    ],
)
def test_execute_call_permissions_only_asked_if_required(
    tool_call_fixture_name: str, 
    tmp_path: Path, 
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
):
    tool_call: ChatCompletionMessageToolCall = request.getfixturevalue(tool_call_fixture_name)
    function_name = tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))


    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    mock_permissions_call = mocker.patch.object(
        agent, "_request_permission_for_tool",
        return_value=PermissionChoice(
            "y"
        )
    )

    agent.execute_tool(tool_call)

    if tool.requires_permissions:
        assert mock_permissions_call.called
    else:
        assert not mock_permissions_call.called


@pytest.mark.parametrize(
    "tool_call_fixture_name",
    [
        "mock_read_tool_call",
        "mock_write_tool_call",
        "mock_bash_tool_call"
    ],
)
def test_execute_call_with_granted_permissions_for_session(
    tool_call_fixture_name: str, 
    tmp_path: Path, 
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
    stub_permission_request_returns_session: None,
):
    tool_call: ChatCompletionMessageToolCall = request.getfixturevalue(tool_call_fixture_name)
    function_name = tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))

    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    result = agent.execute_tool(tool_call)

    assert result.success

    if tool.requires_permissions:
        assert agent.granted_tool_permissions[tool]

def test_execute_call_doesnt_ask_for_permission_after_granted_for_session(
    mock_bash_tool_call: ChatCompletionMessageToolCall,
    tmp_path: Path, 
    mocker: MockerFixture,
    request: pytest.FixtureRequest,
):
    function_name = mock_bash_tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))

    agent = LlmAgent(client=client, workspace=tmp_path, tools=[MagicMock()]*3)

    mock_permissions_call = mocker.patch.object(
        agent, "_request_permission_for_tool",
        return_value=PermissionChoice(
            "s"
        )
    )

    result = agent.execute_tool(mock_bash_tool_call)

    assert result.success
    assert mock_permissions_call.call_count == 1
    assert agent.granted_tool_permissions[tool]

    # call tool again and ensure we don't ask for permissions again
    
    result = agent.execute_tool(mock_bash_tool_call)

    assert result.success
    assert mock_permissions_call.call_count == 1
