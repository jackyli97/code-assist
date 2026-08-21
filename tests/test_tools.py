from pathlib import Path

import pytest

from ai_coding_assistant.tools import (
    ReadTool, 
    ReadToolArgs, 
    resolve_safe_path,
    WriteTool,
    WriteToolArgs,
    validate_command,
    BashTool,
    BashToolArgs
)

from subprocess import CalledProcessError
from collections.abc import Callable

LARGE_FILE_SIZE = 1250
MAX_LINES = 500

@pytest.fixture
def generate_large_file() -> Callable[[Path], Path]:

    def create_file(cwd: Path):
        file_path = (cwd / "test_1250.txt").resolve()
        with open(file_path, "w") as f:
            for i in range(1, LARGE_FILE_SIZE+1):
                f.write(f"This is line number {i}\n")
        return file_path

    return create_file

# ---------------------------------------------------------------------------
# validate_command — single source of truth for path safety rules.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command",
    [
        ["cat", ".env"],                               # bare .env
        ["cat", "~/.ssh/id_rsa"],                      # .ssh as path component
        ["cp", "public.txt", ".env"],                  # .env in destination arg
        ["grep", "TOKEN", "app/.env"],                 # .env in non-adjacent arg
        ["run", "--config", ".env.production"],        # .env.* as flag value
        ["cat", "src/config/.env"],                    # .env leaf under nested dirs
        ["cp", "id_rsa", "~/.ssh/id_rsa.bak"],         # .ssh anywhere in path
        ["cat", ".env.local"],
        ["cat", ".env.production"],
    ],
)
def test_validate_command_rejects_sensitive_commands(command: list[str]) -> None:
    with pytest.raises(PermissionError):
        validate_command(command)


@pytest.mark.parametrize(
    "command",
    [
        ["python", "-c", "print('hi')"],
        ["cat", "config.txt"],
        [".venv/bin/python", "-V"],           # .venv is not .env — allowed
        ["cat", ".envrc"],                     # .envrc is not .env / .env.* — allowed
        ["grep", "environment", "README.md"],  # word "environment" as data
        ["cat", "backup.env"],                 # .env suffix on non-dotenv name — allowed
        ["./.env-loader", "start"],            # .env-loader is a script, not a dotenv
    ],
)
def test_validate_command_allows_safe_commands(command: list[str]) -> None:
    validate_command(command)  # should not raise

# ---------------------------------------------------------------------------
# resolve_safe_path — single source of truth for path safety rules.
# ---------------------------------------------------------------------------


def test_resolve_safe_path_returns_resolved_absolute_path(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("x")

    result = resolve_safe_path(tmp_path, "hello.txt")

    assert result == (tmp_path / "hello.txt").resolve()
    assert result.is_absolute()


def test_resolve_safe_path_rejects_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_safe_path(tmp_path, "/etc/passwd")


def test_resolve_safe_path_rejects_escape_via_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="outside workspace"):
        resolve_safe_path(tmp_path, "../outside.txt")


def test_resolve_safe_path_rejects_dotenv(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="sensitive path"):
        resolve_safe_path(tmp_path, ".env")


@pytest.mark.parametrize(
    "variant",
    [".env.local", ".env.production", ".env.example"],
)
def test_resolve_safe_path_rejects_dotenv_variants(
    tmp_path: Path, variant: str
) -> None:
    with pytest.raises(PermissionError, match="sensitive path"):
        resolve_safe_path(tmp_path, variant)


@pytest.mark.parametrize(
    "sensitive_path",
    [".ssh/id_rsa", ".ssh/known_hosts", "config/.ssh/keys/personal"],
)
def test_resolve_safe_path_rejects_ssh_paths(
    tmp_path: Path, sensitive_path: str
) -> None:
    with pytest.raises(PermissionError, match="sensitive path"):
        resolve_safe_path(tmp_path, sensitive_path)


def test_resolve_safe_path_rejects_symlink_pointing_outside_workspace(
    tmp_path: Path,
) -> None:
    outside_target = tmp_path.parent / "outside_target"
    outside_target.write_text("secret")
    link = tmp_path / "link"
    link.symlink_to(outside_target)

    with pytest.raises(PermissionError, match="outside workspace"):
        resolve_safe_path(tmp_path, "link")


def test_resolve_safe_path_allows_env_like_filenames(tmp_path: Path) -> None:
    # Only `.env` and `.env.*` are blocked. Names that merely contain "env"
    # are allowed — this pins the current contract so a future overzealous
    # rule change is intentional, not accidental.
    (tmp_path / "myenv.txt").write_text("x")
    (tmp_path / ".envrc").write_text("x")

    assert resolve_safe_path(tmp_path, "myenv.txt") == (tmp_path / "myenv.txt").resolve()
    assert resolve_safe_path(tmp_path, ".envrc") == (tmp_path / ".envrc").resolve()


# ---------------------------------------------------------------------------
# ReadTool._read_file — the actual read logic. Raises on error, returns on
# success. Adapter concerns (mapping exceptions to ToolCallResult) belong to
# `call` and are tested separately below.
# ---------------------------------------------------------------------------


def test__read_file_returns_file_contents(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world\n")

    assert "hello world" in ReadTool._read_file("hello.txt", tmp_path)


def test__read_file_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ReadTool._read_file("nope.txt", tmp_path)


def test__read_file_raises_when_path_is_directory(tmp_path: Path) -> None:
    (tmp_path / "somedir").mkdir()

    with pytest.raises(IsADirectoryError):
        ReadTool._read_file("somedir", tmp_path)

def test__read_file_large_file_over_multiple_calls(generate_large_file: Callable[[Path], Path], tmp_path: Path) -> None:
    # 1250 lines will be read over 3 calls
    large_file = generate_large_file(tmp_path).name

    # call 1 returns lines 1-500
    start_line_1 = 1
    result_1 = ReadTool._read_file(file_path=large_file, workspace=tmp_path, start_line=start_line_1, max_lines=MAX_LINES)
    result_1_num_lines= len(result_1.split("\n")[1:]) #exclude header
    assert result_1_num_lines == MAX_LINES
    assert f"Lines {start_line_1}-{start_line_1 + result_1_num_lines - 1}" in result_1
    assert "end of file" not in result_1

    # # call 2 returns lines 501-1000
    start_line_2 = result_1_num_lines + 1
    result_2 = ReadTool._read_file(file_path=large_file, workspace=tmp_path, start_line=start_line_2, max_lines=MAX_LINES)
    result_2_num_lines= len(result_2.split("\n")[1:]) #exclude header
    assert result_2_num_lines == MAX_LINES
    assert f"Lines {start_line_2}-{start_line_2 + result_2_num_lines - 1}" in result_2
    assert "end of file" not in result_2

    # # call 3 returns lines 1001-1250 and end of file
    start_line_3 = result_1_num_lines + result_2_num_lines + 1
    result_3 = ReadTool._read_file(file_path=large_file, workspace=tmp_path, start_line=start_line_3, max_lines=MAX_LINES)
    result_3_num_lines= len(result_3.split("\n")[1:]) #exclude header
    assert result_3_num_lines == LARGE_FILE_SIZE - result_1_num_lines - result_2_num_lines 
    assert f"Lines {start_line_3}-{start_line_3 + result_3_num_lines - 1}" in result_3
    assert "end of file" in result_3



# ---------------------------------------------------------------------------
# ReadTool.call — adapter that maps helper exceptions to ToolCallResult.
# The parametrized test below monkeypatches `read_file` so each case exercises
# purely the try/except mapping, decoupled from filesystem setup.
# ---------------------------------------------------------------------------


def test_read_tool_call_returns_contents_on_success(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world\n")

    result = ReadTool.call(ReadToolArgs(file_path="hello.txt"), tmp_path)

    assert result.success is True
    assert "hello world" in result.output


@pytest.mark.parametrize(
    "raised, expected_substr",
    [
        (FileNotFoundError(), "File not found"),
        (IsADirectoryError(), "directory"),
        (
            PermissionError("Access to sensitive path is not allowed: .env"),
            "sensitive path",
        ),
        (ValueError("Path must be relative to the workspace root"), "relative"),
        (OSError("disk exploded"), "Failure"),
    ],
)
def test_read_tool_call_wraps_helper_exception_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_substr: str,
) -> None:
    def raise_it(file_path: str, workspace: Path, start_line: int, max_lines: int) -> str:
        raise raised

    monkeypatch.setattr(ReadTool, "_read_file", staticmethod(raise_it))

    result = ReadTool.call(ReadToolArgs(file_path="anything"), tmp_path)

    assert result.success is False
    assert expected_substr in result.output

# # ---------------------------------------------------------------------------
# # WriteTool._write_file — the actual write logic. Raises on error, returns on
# # success.
# # ---------------------------------------------------------------------------

def test__write_file_writes_to_file_that_doesnt_exist(tmp_path: Path) -> None:
    full_file_path = tmp_path / "hello.txt"
    contents = "hello world\n"

    WriteTool._write_file("hello.txt", contents, tmp_path)

    assert full_file_path.exists()
    assert full_file_path.read_text() == contents

def test__write_file_overwrites_existing_file(tmp_path: Path) -> None:
    full_file_path = tmp_path / "hello.txt"
    full_file_path.write_text("x")

    new_contents = "hello world\n"

    WriteTool._write_file("hello.txt", new_contents, tmp_path)

    assert full_file_path.read_text() == new_contents

def test__write_file_fails_writing_to_directory_that_doesnt_exist(tmp_path: Path) -> None:
    contents = "hello world\n"

    with pytest.raises(FileNotFoundError):
        WriteTool._write_file("somedir/hello.txt", contents, tmp_path)

def test__write_file_succeeds_writing_to_directory_that_does_exist(tmp_path: Path) -> None:
    (tmp_path / "somedir").mkdir()
    full_file_path = tmp_path / "somedir" / "hello.txt"
    contents = "hello world\n"

    WriteTool._write_file("somedir/hello.txt", contents, tmp_path)

    assert full_file_path.exists()
    assert full_file_path.read_text() == contents

# # ---------------------------------------------------------------------------
# # WriteTool.call — adapter that maps helper exceptions to ToolCallResult.
# # ---------------------------------------------------------------------------

def test_write_call_returns_contents_on_success(tmp_path: Path) -> None:
    result = WriteTool.call(WriteToolArgs(file_path="hello.txt", content="hello world\n"), tmp_path)

    assert result.success is True
    assert result.output == "Created the file"

@pytest.mark.parametrize(
    "raised, expected_substr",
    [
        (FileNotFoundError(), "No such directory"),
        (
            PermissionError("Access to sensitive path is not allowed: .env"),
            "sensitive path",
        ),
        (ValueError("Path must be relative to the workspace root"), "relative"),
        (OSError("disk exploded"), "Failure"),
    ],
)
def test_write_tool_call_wraps_helper_exception_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_substr: str,
) -> None:
    def raise_it(file_path: str, content: str, workspace: Path) -> str:
        raise raised

    monkeypatch.setattr(WriteTool, "_write_file", staticmethod(raise_it))

    result = WriteTool.call(WriteToolArgs(file_path="anything", content="anything"), tmp_path)

    assert result.success is False
    assert expected_substr in result.output

# # ---------------------------------------------------------------------------
# # BashTool._execute_command — the actual bash execution logic. Raises on error, returns on
# # success.
# # ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command,expected_stdout",
    [
        (["echo", "hello"], "hello\n"),
        (["printf", "hello world"], "hello world"),
        (["printf", "%s", "hello world"], "hello world"),
        (["printf", "line1\nline2\n"], "line1\nline2\n"),
        (["printf", "hello 世界"], "hello 世界"),
    ],
)
def test__execute_command_returns_expected_stdout(
    tmp_path: Path,
    command: list[str],
    expected_stdout: str,
):
    result = BashTool._execute_command(
        command=command,
        cwd=".",
        workspace=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == expected_stdout
    assert result.stderr == ""

@pytest.mark.parametrize(
    "command,expected_stdout",
    [
        (["ls"], {"README.md", "src"}),
        (["cat", "README.md"], {"hello world"}),
        (["find", ".", "-name", "*.py"], {"./src/main.py"}),
        (["grep", "hello", "README.md"], {"hello world"}),
        (["head", "-n", "1", "README.md"], {"hello world"}),
        (["tail", "-n", "1", "README.md"], {"second line"}),
    ],
)
def test__execute_command_read_commands(
    tmp_path: Path,
    command: list[str],
    expected_stdout: set[str],
):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "README.md").write_text(
        "hello world\n"
        "second line\n"
    )

    result = BashTool._execute_command(
        command=command,
        cwd=".",
        workspace=tmp_path,
    )

    assert result.returncode == 0

    for expected in expected_stdout:
        assert expected in result.stdout

def test__execute_command_fails_when_cwd_doesnt_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BashTool._execute_command(
        command=["anything"],
        cwd="src",
        workspace=tmp_path,
    )

def test__execute_command_fails_when_cwd_is_a_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BashTool._execute_command(
        command=["anything"],
        cwd="file.txt",
        workspace=tmp_path,
    )

# # ---------------------------------------------------------------------------
# # BashTool.call — adapter that maps helper exceptions to ToolCallResult.
# # ---------------------------------------------------------------------------

def test_bash_call_returns_contents_on_success(tmp_path: Path) -> None:
    (tmp_path / "README.md").touch()
    result = BashTool.call(BashToolArgs(command=["ls"], cwd="."), tmp_path)

    assert result.success is True
    assert result.output.strip() == "README.md"

@pytest.mark.parametrize(
    "raised, expected_substr",
    [
        (FileNotFoundError(), "Path does not exist"),
        (NotADirectoryError, "This is a file"),
        (
            PermissionError("Access to sensitive path is not allowed: .env"),
            "sensitive path",
        ),
        (ValueError("Path must be relative to the workspace root"), "relative"),
        (CalledProcessError(1, cmd=["anything"]), "Failure"),
    ],
)
def test_bash_tool_call_wraps_helper_exception_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: Exception,
    expected_substr: str,
) -> None:
    def raise_it(command: list[str], cwd: str, workspace: Path) -> str:
        raise raised

    monkeypatch.setattr(BashTool, "_execute_command", staticmethod(raise_it))

    result = BashTool.call(BashToolArgs(command=["anything"], cwd="anything"), tmp_path)

    assert result.success is False
    assert expected_substr in result.output