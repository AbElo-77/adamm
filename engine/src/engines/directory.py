import os
import subprocess
from pathlib import Path

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

@dataclass
class Directory:
    root_dir: Path
    output_dir: Path
    env: dict[str, str] = None

    def set_up(self):
        gmxlib = os.environ.get("GMXLIB", "")
        env = os.environ.copy()
        env["GMXLIB"] = str(self.root_dir.resolve()) + (":" + gmxlib if gmxlib else "")
        self.env = env

    def run(self, cmd: List[str], input = None) -> CommandResult:

        if input is not None: 
            proc = subprocess.run(
                cmd,
                cwd=self.output_dir,
                env=self.env,
                input=input,
                check=True,
                capture_output=True,
                text=True
            )
        else: 
            proc = subprocess.run(
                cmd,
                cwd=self.output_dir,
                env=self.env,
                check=True, 
                capture_output=True, 
                text=True
            )
        
        return CommandResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr
        )