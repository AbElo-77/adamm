import subprocess
from dataclasses import dataclass
from typing import List
from engine.src.engines.gromacs.inputs import GROMACSInputs, GROMACSExecution, SystemArtifacts

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
        cmd = [self.options.gromacs_binary, "--version"]
        result = self._execute(cmd)
        if result.returncode != 0:
            raise RuntimeError("Failed to detect GROMACS version")
        first_line = result.stdout.splitlines()[0]
        return first_line.strip()

    def _execute(self, cmd: List[str]) -> CommandResult:
        print("Running:", " ".join(cmd))
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

    def _check(self, result: CommandResult, step: str):
        if result.returncode != 0:
            raise RuntimeError(f"{step} failed:\n{result.stderr}")


    def run_pdb2gmx(
        self,
        input_pdb: str,
        output_gro: str,
        topology: str,
        forcefield: str,
        water_model: str,
    ) -> CommandResult:
        cmd = [
            self.options.gromacs_binary, "pdb2gmx",
            "-f", input_pdb,
            "-o", output_gro,
            "-p", topology,
            "-ff", forcefield,
            "-water", water_model,
        ]
        return self._execute(cmd)

    def run_editconf(
        self,
        input_gro: str,
        output_gro: str,
        box_type: str = "cubic",
        box_size: float = 1.0
    ) -> CommandResult:
        cmd = [
            self.options.gromacs_binary, "editconf",
            "-f", input_gro,
            "-o", output_gro,
            "-bt", box_type,
            "-d", str(box_size),
        ]
        return self._execute(cmd)

    def run_solvate(
        self,
        input_gro: str,
        output_gro: str,
        topology: str,
        solvent: str = "spc216"
    ) -> CommandResult:
        cmd = [
            self.options.gromacs_binary, "solvate",
            "-cp", input_gro,
            "-cs", solvent,
            "-o", output_gro,
            "-p", topology,
        ]
        return self._execute(cmd)

    def run_genion(
        self,
        input_tpr: str,
        output_gro: str,
        topology: str,
        pname: str = "NA",
        nname: str = "CL",
        concentration: float = 0.15,
        neutralize: bool = True,
    ) -> CommandResult:
        cmd = [
            self.options.gromacs_binary, "genion",
            "-s", input_tpr,
            "-o", output_gro,
            "-p", topology,
            "-pname", pname,
            "-nname", nname,
            "-conc", str(concentration),
        ]
        if neutralize:
            cmd.append("-neutral")
        return self._execute(cmd)

    def run_grompp(
        self,
        inputs: GROMACSInputs,
        output_tpr: str
    ) -> CommandResult:
        cmd = [
            self.options.gromacs_binary, "grompp",
            "-f", inputs.mdp,
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
            self.options.gromacs_binary, "mdrun",
            "-s", tpr_file,
            "-deffnm", deffnm,
        ]
        if self.options.ntmpi:
            cmd.extend(["-ntmpi", str(self.options.ntmpi)])
        if self.options.ntomp:
            cmd.extend(["-ntomp", str(self.options.ntomp)])
        return self._execute(cmd)

def build_solvated_system(
    runner: GromacsRunner,
    pdb: str,
    forcefield: str,
    water_model: str,
    ions_mdp: str,
) -> SystemArtifacts:

    top = "topol.top"

    r = runner.run_pdb2gmx(
        pdb, "processed.gro", top, forcefield, water_model
    )
    runner._check(r, "pdb2gmx")

    r = runner.run_editconf("processed.gro", "boxed.gro")
    runner._check(r, "editconf")

    r = runner.run_solvate("boxed.gro", "solvated.gro", top)
    runner._check(r, "solvate")

    ions_tpr = "ions.tpr"
    r = runner.run_grompp(
        GROMACSInputs(
            mdp=ions_mdp,
            structure_file="solvated.gro",
            topology_file=top,
        ),
        ions_tpr,
    )
    runner._check(r, "grompp (ions)")

    r = runner.run_genion(
        ions_tpr, "ions.gro", top
    )
    runner._check(r, "genion")

    return SystemArtifacts(gro="ions.gro", top=top)


def run_md_stage(
    runner: GromacsRunner,
    stage_name: str,
    mdp: str,
    system: SystemArtifacts,
) -> SystemArtifacts:
    
    tpr = f"{stage_name}.tpr"
    r = runner.run_grompp(
        GROMACSInputs(
            mdp=mdp,
            structure_file=system.gro,
            topology_file=system.top,
        ),
        tpr,
    )
    runner._check(r, f"grompp ({stage_name})")

    r = runner.run_mdrun(tpr, stage_name)
    runner._check(r, f"mdrun ({stage_name})")

    return SystemArtifacts(
        gro=f"{stage_name}.gro",
        top=system.top,
        tpr=tpr,
    )