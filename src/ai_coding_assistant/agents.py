import json
import click
import tiktoken

from openai import OpenAI
from ai_coding_assistant.tools import TOOL_REGISTRY, ToolCallResult
from typing import Iterable, List, cast
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageParam, ChatCompletionMessageFunctionToolCall
from pathlib import Path
from pydantic import ValidationError
from dataclasses import dataclass
from ai_coding_assistant.models import lookup_model

@dataclass
class AgentResponse():
    content: str
    run_prompt_tokens: int | None = None
    run_completion_tokens: int | None = None

@dataclass
class CompactMessagesResult():
    success: bool
    failure_reason: str | None = None
    run_prompt_tokens: int | None = None
    run_completion_tokens: int | None = None

class LlmAgent():
    def __init__(
        self, 
        client: OpenAI, 
        tools: Iterable[ChatCompletionFunctionToolParam], 
        workspace: Path, 
        model: str | None = None,
        context_limit: int | None = None,
    ):
        self.client = client
        self.model = model or self.DEFAULT_MODEL
        self.tools = tools
        self.workspace = workspace
        self.max_iterations: int = 10
        self.runs: List[List[ChatCompletionMessageParam]] = [] # list of messages from each run
        self.session_prompt_tokens: int = 0
        self.session_completion_tokens: int = 0
        self.context_limit = context_limit or lookup_model(self.model).context_limit
        self.context = 0
        self.compacted_message: ChatCompletionMessageParam | None = None

    DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
    CONTEXT_UTILIZATION_WARNING_RATIO = 0.8
    CONTEXT_UTILIZATION_RESOLUTION_RATIO = 0.5

    def _get_system_message(self) -> ChatCompletionMessageParam:
        system_prompt =  f"""
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

        return {"role": "system", "content": system_prompt}

    # ---------------------------------------------------------------------------
    # Core agent functions: LLM calling and tool calling
    # ---------------------------------------------------------------------------

    def agentic_loop_call(self, prompt: str) -> AgentResponse:
        current_messages: List[ChatCompletionMessageParam] = []
        current_messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model = self.model,
            messages=self.build_message_history(current_messages=current_messages),
            tools=self.tools
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
            current_messages.append(cast(ChatCompletionMessageParam, message.model_dump()))
            # make tool call
            for tool_call in message.tool_calls:
                if tool_call.type == "function":

                    # log to user tool to be called
                    click.echo(f"● {tool_call.function.name}: {tool_call.function.arguments}")

                    tool_output = self.execute_tool(tool_call=tool_call)

                    # add tool call response to messages
                    current_messages.append({
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
                messages=self.build_message_history(current_messages=current_messages),
                tools=self.tools
            )

            if not response.choices or len(response.choices) == 0:
                raise RuntimeError("no choices in response")
            message = response.choices[0].message
            run_prompt_tokens += response.usage.prompt_tokens if response.usage else 0
            run_completion_tokens += response.usage.completion_tokens if response.usage else 0

            iterations += 1

        # update messages history: append final result, update global
        current_messages.append(cast(ChatCompletionMessageParam, message.model_dump()))
        self.runs.append(current_messages)

        # update usages
        self._update_agent_usages(run_completion_tokens=run_completion_tokens, run_prompt_tokens=run_prompt_tokens)

        if message.content:
            return AgentResponse(
                content=message.content,
                run_prompt_tokens=run_prompt_tokens,
                run_completion_tokens=run_completion_tokens,
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

    # ---------------------------------------------------------------------------
    # Agent state management
    # ---------------------------------------------------------------------------

    def build_message_history(
        self,
        current_messages: list[ChatCompletionMessageParam] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = [self._get_system_message()]

        if self.compacted_message:
            messages.append(self.compacted_message)

        for run in self.runs:
            messages.extend(run)

        if current_messages:
            messages.extend(current_messages)

        return messages
    
    def clear_history(self):
        self.compacted_message = None
        self.runs = []

    def update_model(self, model: str, context_limit: int | None) -> None:
        self.model = model
        self.context_limit = context_limit

    def compact_message_history(self) -> CompactMessagesResult:
        if not self.context_limit:
            return CompactMessagesResult(
                success=False,
                failure_reason=(
                    "Could not define a context limit for the model, "
                    "unable to compact"
                ),
            )

        utilization = self.context / self.context_limit

        if utilization < self.CONTEXT_UTILIZATION_WARNING_RATIO:
            return CompactMessagesResult(success=True)

        if not self.runs:
            return CompactMessagesResult(
                success=False,
                failure_reason="Not enough messages in history, unable to compact",
            )

        target_context = self.context_limit * self.CONTEXT_UTILIZATION_RESOLUTION_RATIO
        encoding = tiktoken.get_encoding("o200k_base")

        # we will compact messages grouped by each run, that way we don't lose important
        # context throughout, such as the prompt, tool, and response groupings
        last_run_to_compact_idx = 0 
        final_messages_to_compact = []

        if self.compacted_message:
            final_messages_to_compact.append(self.compacted_message)

        removed_tokens = 0

        while last_run_to_compact_idx < len(self.runs):
            messages_to_compact = self.runs[last_run_to_compact_idx]
            messages_to_compact_json = self._serialize_json(messages_to_compact)

            compacted_tokens = len(
                encoding.encode(messages_to_compact_json)
            )

            final_messages_to_compact.extend(messages_to_compact)

            removed_tokens += compacted_tokens
            remaining_context = self.context - removed_tokens


            if remaining_context <= target_context:
                break

            last_run_to_compact_idx += 1

        final_messages_to_compact_json = self._serialize_json(final_messages_to_compact)

        try:
            response = self._llm_summarize_messages(
                final_messages_to_compact_json
            )
            compacted_history, run_prompt_tokens, run_completion_tokens = (
                response.content, response.run_prompt_tokens, response.run_completion_tokens
            )
        except RuntimeError:
            return CompactMessagesResult(
                success=False,
                failure_reason="Ran into error while compacting history",
            )

        compacted_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": (
                    "Summary of earlier conversation history:\n\n"
                    f"{compacted_history}"
                ),
            }

        self.compacted_message = compacted_message

        # update runs to no longer include the summarized runs, but do pre-append the summary as one run
        updated_runs: List[List[ChatCompletionMessageParam]] = []

        # add the remaining runs to history
        for i in range(last_run_to_compact_idx+1, len(self.runs)):
            messages_from_run = self.runs[i]
            updated_runs.append(messages_from_run)

        self.runs = updated_runs
        self._update_agent_usages(run_prompt_tokens=run_prompt_tokens or 0, run_completion_tokens=run_completion_tokens or 0)

        return CompactMessagesResult(success=True, run_prompt_tokens=run_prompt_tokens, run_completion_tokens=run_completion_tokens)

    def _llm_summarize_messages(self, serialized_messages_to_compact: str) -> AgentResponse:
        compaction_prompt = f"""
        Summarize the conversation history below so another coding
        agent can continue the session without needing the original messages.

        Preserve:
        - the user's goals and requirements
        - architectural decisions and reasoning
        - important implementation details
        - file names and paths
        - relevant code changes
        - unresolved problems and TODOs
        - errors encountered and their resolutions
        - commands or tests that matter

        Remove:
        - repetition
        - conversational filler
        - obsolete intermediate discussion

        Be concise but do not omit information needed to continue working.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": compaction_prompt,
                },
                {
                    "role": "user",
                    "content": serialized_messages_to_compact,
                },
            ],
        )

        if not response.choices or len(response.choices) == 0:
            raise RuntimeError("no choices in response")

        if response.choices[0].message.content:
            run_prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            run_completion_tokens = response.usage.completion_tokens if response.usage else 0
            return AgentResponse(
                content=response.choices[0].message.content,
                run_prompt_tokens=run_prompt_tokens,
                run_completion_tokens=run_completion_tokens,
            ) 
        else:
            raise RuntimeError("LLM returned no content while attempting to compact history")

    # ---------------------------------------------------------------------------
    # Agent tokens and context management
    # ---------------------------------------------------------------------------

    def _update_agent_usages(self, run_prompt_tokens: int, run_completion_tokens: int) -> None:
        self.context = self.estimate_context_tokens()
        self.session_prompt_tokens += run_prompt_tokens
        self.session_completion_tokens += run_completion_tokens

    def estimate_context_tokens(self) -> int:
        encoding = tiktoken.get_encoding("o200k_base")

        messages_json = self._serialize_json(self.build_message_history())
        tools_json = self._serialize_json(self.tools)

        return (
            len(encoding.encode(messages_json))
            + len(encoding.encode(tools_json))
        )

    def _serialize_json(
        self,
        message_or_tool: list[ChatCompletionMessageParam] | Iterable[ChatCompletionFunctionToolParam],
    ) -> str:
        return json.dumps(
            message_or_tool,
            default=str,
            ensure_ascii=False,
        )
