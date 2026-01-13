import subprocess
from dataclasses import dataclass
from typing import List
from .inputs import GROMACSInputs, GROMACSExecution

@dataclass(frozen=True)
class CommandResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

class GromacsRunner:

    def __init__(self, options: GROMACSExecution):
        self.options = options

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

    def run_grompp(
        self,
        inputs: GROMACSInputs,
        output_tpr: str
    ) -> CommandResult:
        cmd = [
            self.options.gmx_binary, "grompp",
            "-f", inputs.mdp.source_file,
            "-c", inputs.structure_file,
            "-p", inputs.topology_file,
            "-o", output_tpr,
        ]

        if inputs.index_file:
            cmd.extend(["-n", inputs.index_file])

        return self._execute(cmd)

    def run_mdrun(
        self,
        tpr_file: str,
        deffnm: str
    ) -> CommandResult:
        cmd = [
            self.options.gmx_binary, "mdrun",
            "-s", tpr_file,
            "-deffnm", deffnm,
        ]

        if self.options.ntmpi:
            cmd.extend(["-ntmpi", str(self.options.ntmpi)])

        if self.options.ntomp:
            cmd.extend(["-ntomp", str(self.options.ntomp)])

        return self._execute(cmd)
    
    def _execute(self, cmd: List[str]) -> CommandResult:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        return CommandResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr
        )