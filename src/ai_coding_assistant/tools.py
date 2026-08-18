from abc import ABC, abstractmethod
from typing import Dict, Type
from pydantic import BaseModel, Field, ConfigDict
import subprocess
from subprocess import CompletedProcess
from pathlib import Path
from dataclasses import dataclass

class ReadToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(description=(
        "The path to the file to read, relative to the project workspace root"
        "Needs to be a file path, not a directory path"
    )
)

class WriteToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str = Field(description=(
        "The path of the file to write to, relative to the project workspace root"
        "Needs to be a file path, not a directory path"
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
        "project workspace root. Use '.' to execute from the workspace root."
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
            return ToolCallResult(success=True, output=ReadTool.read_file(file_path, workspace))
        except FileNotFoundError:
            return ToolCallResult(success=False, output="File not found, please use bash tool to locate the file")
        except IsADirectoryError:
            return ToolCallResult(success=False, output="Trying to read a directory, please try again with a valid file or use bash tool to read directory")
        except OSError:
            return ToolCallResult(success=False, output="Failure occured while reading to file")

    @staticmethod
    def read_file(file_path: str, workspace: Path) -> str:

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
            WriteTool.write_file(file_path, content, workspace)
            return ToolCallResult(success=True, output="Created the file")
        except IsADirectoryError:
            return ToolCallResult(success=False, output="Trying to write to a file path, please provide a file path, not a directory path")
        except IOError:
            return ToolCallResult(success=False, output="Failure occured while writing to file")

    @staticmethod
    def write_file(file_path: str, content: str, workspace: Path):
        path = resolve_safe_path(workspace, file_path)

        if not path.is_file():
            raise IsADirectoryError(file_path)
        
        # create new file or override existing
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
            command_exec_res = BashTool.execute_command(command, cwd, workspace)
            return (
                ToolCallResult(success=True, output=command_exec_res.stdout) if command_exec_res.returncode == 0 
                else ToolCallResult(success=False, output=command_exec_res.stderr)
            )
        except FileNotFoundError:
            return ToolCallResult(success=False, output="Path does not exist, please try again with a valid path or create this path first")
        except NotADirectoryError:
            return ToolCallResult(success=False, output="This is a file, please try command again with a directory")

    @staticmethod
    def execute_command(command: list[str], cwd: str, workspace: Path) -> CompletedProcess[str]:
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

SENSITIVE_PATTERNS = {
    ".env",
    ".ssh",
}

def validate_command(command: list[str]) -> None:
    for arg in command:
        if any(pattern in arg for pattern in SENSITIVE_PATTERNS):
            raise PermissionError(
                f"Command references a sensitive path: {arg}"
            )

def resolve_safe_path(
    workspace: Path,
    file_path: str,
) -> Path:
    workspace = workspace.resolve()
    path = (workspace / file_path).resolve()

    if not path.is_relative_to(workspace):
        raise PermissionError(
            f"Path is outside workspace: {file_path}"
        )

    if path.name == ".env" or path.name.startswith(".env."):
        raise PermissionError(
            f"Access to sensitive file is not allowed: {path.name}"
        )

    return path
    
    
