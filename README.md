# ai-coding-assistant

## Install as a CLI

```
git clone <repo>
cd ai-coding-assistant
uv tool install .
```

If `code-assist` isn't found afterwards, run `uv tool update-shell` and restart your shell.

### Configure OpenRouter

Configure your OpenRouter API key:

```bash
code-assist config set-key
```

You'll be prompted to enter your key securely. Code Assist stores it in:

```text
~/.config/code-assist/.env
```

with permissions restricted to your user.

Code Assist deliberately ignores `.env` files in the project you're working on — those belong to the project, not the CLI.

You can also set `OPENROUTER_API_KEY` directly in your environment. An environment variable takes precedence over the key stored in `~/.config/code-assist/.env`.

## Use

Run `code-assist` from the root directory of the project you want it to work on:

```
code-assist -p "explain what this repo does"
```

The assistant reads, writes, and runs commands inside the directory you invoke it from.

## Develop

```
uv sync
uv run code-assist -p "..."
```

Re-run `uv tool install . --reinstall` to push local changes into the installed CLI.
