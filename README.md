# ai-coding-assistant

A local CLI coding agent that can explore a project, modify files, and run commands using LLMs through OpenRouter.

`code-assist` operates from your project directory and supports both one-shot prompts and persistent interactive sessions.

## Features

* **Interactive coding sessions** — maintain conversation context across multiple prompts in a terminal REPL.
* **Single-shot prompts** — run one-off coding tasks directly from the command line.
* **Workspace-aware file access** — read and write tools are restricted to paths within the directory where `code-assist` is started.
* **File reading** — inspect targeted sections of files with bounded line ranges.
* **File writing** — create or overwrite files within the project workspace.
* **Command execution** — run commands with a workspace-relative working directory.
* **Permission controls** — file writes and shell commands require approval before execution, with the option to approve once or for the remainder of the session.
* **Sensitive-path protection** — direct access to `.env`, `.env.*`, and `.ssh` paths is blocked.
* **Configurable models** — use supported models available through OpenRouter and change models during an interactive session.
* **Conversation management** — clear or compact conversation history as context usage grows.
* **Token and context tracking** — inspect API token usage and estimated context-window utilization during interactive sessions.

## Requirements

* Python 3.13+
* [uv](https://docs.astral.sh/uv/)
* An [OpenRouter](https://openrouter.ai/) API key

## Installation

Clone the repository and install `code-assist` as a CLI:

```bash
git clone <repo>
cd ai-coding-assistant
uv tool install .
```

If `code-assist` isn't found afterwards, run:

```bash
uv tool update-shell
```

Then restart your shell.

Verify the installation with:

```bash
code-assist --version
```

## Configuration

### OpenRouter API key

Configure your OpenRouter API key:

```bash
code-assist config set-key
```

You'll be prompted to enter the key securely. Code Assist stores its configuration in:

```text
~/.config/code-assist/.env
```

with file permissions restricted to your user.

Code Assist deliberately does not load `.env` files from the project you're working on. Project environment files belong to the project rather than the CLI.

You can alternatively provide the API key through your environment:

```bash
export OPENROUTER_API_KEY="..."
```

Environment variables take precedence over values stored in the Code Assist configuration file.

### Model

The default model is:

```text
anthropic/claude-haiku-4.5
```

Set a preferred OpenRouter model with:

```bash
code-assist config set-model <model-id>
```

For example:

```bash
code-assist config set-model anthropic/claude-sonnet-4
```

Model IDs are validated against OpenRouter before being saved.

## Usage

Run `code-assist` from the directory you want to use as the project workspace.

### Interactive mode

Start an interactive session:

```bash
code-assist
```

The session preserves conversation history between prompts, allowing the agent to continue working with context from previous requests.

Example:

```text
code-assist v0.1.0
Workspace: /path/to/project
Model: anthropic/claude-haiku-4.5
Type your request below, or '/help' for available commands.

> explain how authentication works in this project

...

> now add tests for it
```

Interactive mode supports:

```text
/clear              Reset conversation history
/compact            Compact conversation history using LLM summarization
/exit               Exit the session
/help               Show available commands
/model              Show the current model and its context limit
/model <model-id>   Change the model for the current session
/status             Show workspace, model, token usage, and context usage
```

You can also exit with `Ctrl+C`.

### Single-shot mode

For a single request without starting an interactive session:

```bash
code-assist -p "explain what this repo does"
```

or:

```bash
code-assist --prompt "run the tests and explain any failures"
```

The process exits after the agent completes the request.

## Permissions and Safety

Code Assist distinguishes between read-only file inspection and actions that can modify your system.

File reads do not require approval. File writes and shell commands require explicit permission before they are executed.

When approval is required, Code Assist prompts you with the proposed action:

```text
code-assist wants to perform an action:

Bash: {
  "command": [
    "pytest",
    "-q"
  ],
  "cwd": "."
}

Allow? [y] once / [s] session / [n] deny [n]:
```

You can:

* `y` — approve this action once
* `s` — approve that tool for the remainder of the current session
* `n` — deny the action

Session permissions are kept in memory and reset when Code Assist exits.

### Workspace restrictions

Direct file reads and writes must use relative paths and are restricted to the project workspace. Attempts to traverse outside the workspace or use absolute paths are rejected.

Code Assist also blocks direct access to sensitive paths including:

```text
.env
.env.*
.ssh/
```

Shell commands use a working directory restricted to the project workspace and are executed without a shell (`shell=False`).

### Shell execution is not OS-sandboxed

> **Important:** Shell commands are executed directly on the host system and are **not OS-sandboxed in v0.1.0**.

Permission prompts, command validation, and workspace-relative working directories reduce accidental or unauthorized actions, but they do not provide process-level isolation.

An approved executable runs with the permissions of the current OS user and may be capable of accessing resources outside the project workspace.

Review shell commands carefully before approving them.

## Context Management

Interactive sessions accumulate conversation history so the agent can reason about previous prompts and actions.

Use:

```text
/status
```

to inspect the current model, context limit, approximate context usage, workspace, and cumulative API token usage.

When a conversation becomes large, you can use:

```text
/compact
```

to summarize older conversation history while retaining information needed to continue working.

To discard conversation history entirely:

```text
/clear
```

## Development

Install development dependencies:

```bash
uv sync
```

Run the CLI from the development environment:

```bash
uv run code-assist
```

or:

```bash
uv run code-assist -p "..."
```

Run the test suite with:

```bash
uv run pytest
```

If you have already installed the CLI with `uv tool install .`, reinstall it to pick up local changes:

```bash
uv tool install . --reinstall
```

## Current Limitations

`ai-coding-assistant` is currently at `v0.1.0`.

Notable limitations include:

* Shell execution is not isolated in a container or OS-level sandbox.
* Conversation history exists only for the lifetime of the interactive process.
* Session-level permission grants are not persisted between sessions.
* File writes replace the target file contents rather than applying structured patches.
* Tool execution is limited to file reads, file writes, and command execution.

## License

Licensed under the MIT License. See `LICENSE` for details.
