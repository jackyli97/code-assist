from ai_coding_assistant.agents import LlmAgent  # noqa: F401

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from ai_coding_assistant.tools import ToolCallResult
from pytest_mock import MockerFixture
import random
from ai_coding_assistant.tools import TOOL_REGISTRY

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

    for i in range(31):
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
    agent = LlmAgent(client=mock_no_tool_call_openai, workspace=tmp_path)

    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)
    assert mock_no_tool_call_openai.chat.completions.create.call_count == 1
    assert mock_execute_tool.call_count == 0

def test_agentic_loop_call_single_tool_call(
        mock_single_tool_call_openai: MagicMock, 
        tmp_path: Path,
        mocker: MockerFixture,
        mock_read_tool_call: ChatCompletionMessageToolCall
    ):
    agent = LlmAgent(client=mock_single_tool_call_openai, workspace=tmp_path)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)

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
    agent = LlmAgent(client=mock_multi_call_openai, workspace=tmp_path)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)

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
    agent = LlmAgent(client=mock_multi_tools_in_single_response_openai, workspace=tmp_path)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )

    agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)

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
    agent = LlmAgent(client=mock_max_iterations_call_openai, workspace=tmp_path)
    mock_execute_tool = mocker.patch.object(
        agent, "execute_tool",
        return_value=ToolCallResult(success=True, output="stubbed"),
    )
    with pytest.raises(RuntimeError, match="Agent exceeded max iterations"):
        agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)

    assert mock_max_iterations_call_openai.chat.completions.create.call_count == 31 # don't include initial call as an iteration
    assert mock_execute_tool.call_count == 30

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
    request: pytest.FixtureRequest
):
    tool_call: ChatCompletionMessageToolCall = request.getfixturevalue(tool_call_fixture_name)
    function_name = tool_call.function.name
    tool = TOOL_REGISTRY[function_name]
    client = MagicMock()

    mocker.patch.object(tool, "call", return_value=ToolCallResult(
        success=True,
        output="stubbed"
    ))

    agent = LlmAgent(client=client, workspace=tmp_path)

    result = agent.execute_tool(tool_call)
    assert result.success

def test_execute_call_fails_invalid_tool(tmp_path: Path):
    invalid_tool_call: ChatCompletionMessageToolCall = ChatCompletionMessageToolCall(
        id="call_read_0",
        type="function",
        function=Function(name="Delete", arguments='{"file_path": "something"}')
    )
    client = MagicMock()

    agent = LlmAgent(client=client, workspace=tmp_path)

    result = agent.execute_tool(invalid_tool_call)

    assert not result.success
    assert "Unknown tool" in result.output

def test_execute_call_fails_model_validate_json(
    mock_read_tool_call: ChatCompletionMessageToolCall,
    tmp_path: Path
):
    client = MagicMock()
    
    agent = LlmAgent(client=client, workspace=tmp_path)

    mock_read_tool_call.function.arguments = '{"file_paths": "something"}' #typo

    result = agent.execute_tool(mock_read_tool_call)

    assert not result.success
    assert "validation errors" in result.output

def test_token_usage(
    mock_multi_call_openai: MagicMock,
    tmp_path: Path,
    mocker: MockerFixture,
):
    agent = LlmAgent(client=mock_multi_call_openai, workspace=tmp_path)
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

    result = agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)

    assert result.run_prompt_tokens == expected_prompt_tokens
    assert result.run_completion_tokens == expected_completion_tokens
    assert result.session_prompt_tokens == expected_prompt_tokens
    assert result.session_completion_tokens == expected_completion_tokens
    assert agent.session_prompt_tokens == expected_prompt_tokens
    assert agent.session_completion_tokens == expected_completion_tokens

    # second run on the same agent: reset the side effect using saved mock responses
    mock_multi_call_openai.chat.completions.create.side_effect = mock_multi_call_openai.mock_responses
    result_2 = agent.agentic_loop_call(prompt="what does this repository do", tools=[MagicMock()]*3)

    assert result_2.run_prompt_tokens == expected_prompt_tokens
    assert result_2.run_completion_tokens == expected_completion_tokens
    assert result_2.session_prompt_tokens == expected_prompt_tokens * 2
    assert result_2.session_completion_tokens == expected_completion_tokens * 2
    assert agent.session_prompt_tokens == expected_prompt_tokens * 2
    assert agent.session_completion_tokens == expected_completion_tokens * 2