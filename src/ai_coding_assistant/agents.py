import json
import click

from openai import OpenAI
from ai_coding_assistant.tools import TOOL_REGISTRY, ToolCallResult
from typing import Iterable, List, cast
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCall
from pathlib import Path
from pydantic import ValidationError
from dataclasses import dataclass

@dataclass
class AgentResponse():
    content: str
    run_prompt_tokens: int | None
    run_completion_tokens: int | None
    session_prompt_tokens: int | None
    session_completion_tokens: int | None

class LlmAgent():
    def __init__(self, client: OpenAI, workspace: Path, model="anthropic/claude-haiku-4.5"):
        self.client = client
        self.model = model
        self.workspace = workspace
        self.max_iterations: int = 30
        self.messages: List[ChatCompletionMessageParam] = []
        self.session_prompt_tokens: int = 0
        self.session_completion_tokens: int = 0

    def agentic_loop_call(self, prompt: str, tools: Iterable[ChatCompletionFunctionToolParam]) -> AgentResponse:
        system_prompt = f"""
        You are an AI coding assistant operating on a local project.

        Project workspace root:
        {self.workspace}

        Rules:
        - Treat the workspace root as the project boundary.
        - Use relative paths whenever possible.
        - Do not assume you can access files outside the workspace.
        - Do not read or modify sensitive files such as .env files.
        - When running commands, choose the appropriate cwd relative to the workspace.
        - Do not use cd; set the command cwd instead.
        - Inspect relevant files before making changes.

        Prompts should be about the project and scope should be within project directory, offtopic prompts 
        shouldn't be answered, and no tools should be called. Return a message reiterating the intended usage.
        """
        
        self.messages.extend([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
        response = self.client.chat.completions.create(
            model = self.model,
            messages=self.messages,
            tools=tools
        )

        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("no choices in response")

        run_prompt_tokens: int = response.usage.prompt_tokens if response.usage else 0
        run_completion_tokens: int = response.usage.completion_tokens if response.usage else 0

        message = response.choices[0].message
        iterations: int = 0

        while message.tool_calls:
            if iterations == self.max_iterations:
                raise RuntimeError(f"Agent exceeded max iterations ({self.max_iterations}). ")
            # add tool calls assistant wants to call to messages
            self.messages.append(cast(ChatCompletionMessageParam, message.model_dump()))
            # make tool call
            for tool_call in message.tool_calls:
                if tool_call.type == "function":

                    # log to user tool to be called
                    click.echo(f"● {tool_call.function.name}: {tool_call.function.arguments}")

                    tool_output = self.execute_tool(tool_call=tool_call)

                    # add tool call response to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({
                            "success": tool_output.success,
                            "output": tool_output.output,
                        })
                    })

                    # log to user tool call
                    if tool_output.success:
                        click.echo("  ✓ Done")
                    else:
                        click.echo(f"  ✗ Failed: {tool_output.output}")


            # call llm with new tool result and assign response to newest response
            response = self.client.chat.completions.create(
                model = self.model,
                messages=self.messages,
                tools=tools
            )

            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("no choices in response")
            message = response.choices[0].message
            run_prompt_tokens += response.usage.prompt_tokens if response.usage else 0
            run_completion_tokens += response.usage.completion_tokens if response.usage else 0

            iterations += 1

        if message.content:
            self.session_prompt_tokens += run_prompt_tokens
            self.session_completion_tokens += run_completion_tokens
            return AgentResponse(
                content=message.content,
                run_prompt_tokens=run_prompt_tokens,
                run_completion_tokens=run_completion_tokens,
                session_prompt_tokens=self.session_prompt_tokens,
                session_completion_tokens=self.session_completion_tokens
            )

        raise RuntimeError("LLM returned no content and no tool calls")

    def execute_tool(self, tool_call: ChatCompletionMessageFunctionToolCall) -> ToolCallResult:
        function_name = tool_call.function.name
        function_arguments = tool_call.function.arguments

        # validate the args provided by LLM against the args defined for tool
        if function_name not in TOOL_REGISTRY:
            return ToolCallResult(
                success=False,
                output=f"Unknown tool: {function_name!r}. Available tools: {sorted(TOOL_REGISTRY)}",
            )
        tool = TOOL_REGISTRY[function_name]
        tool_arg_model = tool.args_model
        try:
            validated_args = tool_arg_model.model_validate_json(function_arguments)
        except ValidationError as e:
            return ToolCallResult(success=False, output=str(e))

        return tool.call(validated_args, self.workspace)

    def clear_history(self):
        self.messages = []
