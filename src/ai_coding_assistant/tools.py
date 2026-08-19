from abc import ABC, abstractmethod
from typing import Dict, Type
from pydantic import BaseModel, Field, ConfigDict
import subprocess
from subprocess import CompletedProcess, CalledProcessError
from pathlib import Path
from dataclasses import dataclass

class ReadToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(description=(
        "The path to the file to read, relative to the project workspace root. "
        "Must be a relative path; absolute paths (e.g. '/etc/passwd') are rejected. "
        "Needs to be a file path, not a directory path."
    )
)

class WriteToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(description=(
        "The path of the file to write to, relative to the project workspace root. "
        "Must be a relative path; absolute paths (e.g. '/etc/passwd') are rejected. "
        "Needs to be a file path, not a directory path."
    )
)
    content: str = Field(description="The content to write to the file")

class BashToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: list[str] = Field(
    description=(
        "Command to execute as a list, with the executable first followed "
        "by each argument. Example: ['pytest', '-v', 'tests/']."
    ),
    min_length=1
)
    cwd: str = Field(
    default=".",
    description=(
        "Working directory in which to execute the command, relative to the "
        "project workspace root. Use '.' to execute from the workspace root. "
        "Must be a relative path; absolute paths are rejected."
    ),
)

@dataclass
class ToolCallResult():
    success: bool
    output: str

class BaseTool(ABC):
    name: str
    args_model: Type[BaseModel]
    description: str

    @staticmethod
    @abstractmethod
    def call(args: BaseModel, workspace: Path) -> ToolCallResult:
        # call the tool function
        pass

class ReadTool(BaseTool):
    name = "Read"
    args_model = ReadToolArgs
    description = "Read and return the contents of a file"

    @staticmethod
    def call(args: ReadToolArgs, workspace: Path) -> ToolCallResult:
        file_path = args.file_path
        try:
            return ToolCallResult(success=True, output=ReadTool._read_file(file_path, workspace))
        except ValueError as e:
            return ToolCallResult(success=False, output=str(e))
        except PermissionError as e:
            return ToolCallResult(success=False, output=str(e))
        except FileNotFoundError:
            return ToolCallResult(success=False, output="File not found, please use bash tool to locate the file")
        except IsADirectoryError:
            return ToolCallResult(success=False, output="Trying to read a directory, please try again with a valid file or use bash tool to read directory")
        except OSError:
            return ToolCallResult(success=False, output="Failure occured while reading to file")

    @staticmethod
    def _read_file(file_path: str, workspace: Path) -> str:
        path = resolve_safe_path(workspace, file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)
        if not path.is_file():
            raise IsADirectoryError(file_path)
        with open(path, "r", encoding="utf-8") as file:
            file_contents = file.read()
            return file_contents

class WriteTool(BaseTool):
    name="Write"
    args_model = WriteToolArgs
    description = "Write content to a file"

    @staticmethod
    def call(args: WriteToolArgs, workspace: Path) -> ToolCallResult:
        file_path = args.file_path
        content = args.content
        try:
            WriteTool._write_file(file_path, content, workspace)
            return ToolCallResult(success=True, output="Created the file")
        except ValueError as e:
            return ToolCallResult(success=False, output=str(e))
        except PermissionError as e:
            return ToolCallResult(success=False, output=str(e))
        except FileNotFoundError as e:
            return ToolCallResult(success=False, output="No such directory, please ensure directories in path exist(file does not need to exist)")
        except OSError:
            return ToolCallResult(success=False, output="Failure occured while writing to file")

    @staticmethod
    def _write_file(file_path: str, content: str, workspace: Path):
        """
        Writes to an existing or new file for a valid path

        Raises:
            FileNotFoundError: If the path is invalid due to a directory not existing
        """
        path = resolve_safe_path(workspace, file_path)        
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

class BashTool(BaseTool):
    name="Bash"
    args_model=BashToolArgs
    description="Execute a shell command"

    @staticmethod
    def call(args: BashToolArgs, workspace: Path) -> ToolCallResult:
        command = args.command
        cwd = args.cwd
        try:
            command_exec_res = BashTool._execute_command(command, cwd, workspace)
            return (
                ToolCallResult(success=True, output=command_exec_res.stdout) if command_exec_res.returncode == 0
                else ToolCallResult(success=False, output=command_exec_res.stderr)
            )
        except ValueError as e:
            return ToolCallResult(success=False, output=str(e))
        except PermissionError as e:
            return ToolCallResult(success=False, output=str(e))
        except FileNotFoundError:
            return ToolCallResult(success=False, output="Path does not exist, please try again with a valid path or create this path first")
        except NotADirectoryError:
            return ToolCallResult(success=False, output="This is a file, please try command again with a directory")
        except CalledProcessError:
            return ToolCallResult(success=False, output="Failure occured while running process")

    @staticmethod
    def _execute_command(command: list[str], cwd: str, workspace: Path) -> CompletedProcess[str]:
        working_dir = resolve_safe_path(workspace, cwd)

        if not working_dir.exists():
            raise FileNotFoundError(working_dir)
        if working_dir.is_file():
            raise NotADirectoryError(working_dir)
        
        # validate command is safe
        validate_command(command)

        result = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            shell=False,
            timeout=30,
        )

        return result

TOOL_REGISTRY: Dict[str, Type[BaseTool]] = {
    "Read": ReadTool,
    "Write": WriteTool,
    "Bash": BashTool
}

SENSITIVE_FILENAMES: frozenset[str] = frozenset({".env"})
SENSITIVE_FILENAME_PREFIXES: tuple[str, ...] = (".env.",)
SENSITIVE_DIR_NAMES: frozenset[str] = frozenset({".ssh"})


def is_sensitive_path(path_str: str) -> bool:
    """True if ``path_str`` references a sensitive filesystem location.

    Single source of truth for what counts as sensitive. Used by both
    ``resolve_safe_path`` (file access) and ``validate_command`` (bash args)
    so the two check-sites can't drift apart.

    Blocks:
      - Files named exactly ``.env``
      - Files named ``.env.<anything>`` (``.env.local``, ``.env.production``)
      - Any path containing ``.ssh`` as a directory component
    """
    p = Path(path_str)
    if p.name in SENSITIVE_FILENAMES:
        return True
    if p.name.startswith(SENSITIVE_FILENAME_PREFIXES):
        return True
    if SENSITIVE_DIR_NAMES.intersection(p.parts):
        return True
    return False


def validate_command(command: list[str]) -> None:
    for arg in command:
        if is_sensitive_path(arg):
            raise PermissionError(
                f"Command references a sensitive path: {arg}"
            )


def resolve_safe_path(
    workspace: Path,
    file_path: str,
) -> Path:
    if Path(file_path).is_absolute():
        raise ValueError(
            f"Path must be relative to the workspace root, got absolute path: {file_path}"
        )

    workspace = workspace.resolve()
    path = (workspace / file_path).resolve()

    if not path.is_relative_to(workspace):
        raise PermissionError(
            f"Path is outside workspace: {file_path}"
        )

    if is_sensitive_path(str(path)):
        raise PermissionError(
            f"Access to sensitive path is not allowed: {path.name}"
        )

    return path