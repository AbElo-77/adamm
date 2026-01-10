import subprocess
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass(frozen=True)
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    start_time: str
    end_time: str

class GromacsRunner:

    def __init__(self, options):
        self.options = options
        self.engine_version = self._detect_version()

    def _detect_version(self) -> str:
        try:
            result = subprocess.run(
                [self.options.gmx_binary, "--version"],
                capture_output=True,
                text=True
            )
            return result.stdout.splitlines()[0]
        except Exception:
            return "unknown"

    def _execute(self, cmd: List[str]) -> CommandResult:
        start = datetime.now().isoformat()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        end = datetime.now().isoformat()
        return CommandResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            start_time=start,
            end_time=end
        )
