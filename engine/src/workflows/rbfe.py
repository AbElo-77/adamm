from core.artifacts import (
    FileRef, Trajectory, EnergySeries,
    LambdaWindow, SimulationRun
)
from provenance.store import ProvenanceStore
from provenance.metadata import ArtifactMetadata
from engines.gromacs.inputs import GROMACSInputs, GROMACSExecution, MDConfig
from engines.gromacs.runner import GromacsRunner, CommandResult
from engines.gromacs.outputs import parse_xvg, read_trajectory
from datetime import datetime

class GromacsSimulationNode:
    def __init__(self, runner: GromacsRunner, prov_store: ProvenanceStore):
        self.runner = runner
        self.prov = prov_store

from provenance.metadata import ArtifactMetadata
from core.artifacts import FileRef, LambdaWindow, SimulationRun

def run_lambda_window(self, inputs: GROMACSInputs, lambda_value: float):
    try:
        tpr_file = f"lambda_{lambda_value}.tpr"
        result_grompp = self.runner.run_grompp(inputs, tpr_file)
        grompp_fp = self.prov.add_artifact(
            FileRef(tpr_file),
            metadata=ArtifactMetadata(
                created_by="GromacsSimulationNode",
                created_at=result_grompp.start_time,
                engine=self.runner.options.gmx_binary,
                engine_version=self.runner.engine_version,
                extra={
                    "exit_code": str(result_grompp.returncode),
                    "stderr": result_grompp.stderr
                }
            )
        )
        if result_grompp.returncode != 0:
            print(f"[WARN] grompp failed for λ={lambda_value}")

        deffnm = f"lambda_{lambda_value}"
        result_mdrun = self.runner.run_mdrun(tpr_file, deffnm)
        traj_artifact = read_trajectory(f"{deffnm}.xtc", atom_count=inputs.structure_file)
        traj_fp = self.prov.add_artifact(traj_artifact)

        energy = parse_xvg(f"{deffnm}.xvg", energy_type="potential", units="kJ/mol")
        energy_fp = self.prov.add_artifact(energy)

        lw = LambdaWindow(
            lambda_value=lambda_value,
            energy_series={"potential": energy},
            trajectory=traj_artifact,
            exchange_acceptance=None
        )
        lw_fp = self.prov.add_artifact(lw, parents=[energy, traj_artifact])
        return lw, lw_fp
    except Exception as e:
        fail_fp = self.prov.add_artifact(
            FileRef(path=f"lambda_{lambda_value}_failed.log"),
            metadata=ArtifactMetadata(
                created_by="GromacsSimulationNode",
                created_at=datetime.utcnow().isoformat(),
                engine=self.runner.options.gmx_binary,
                engine_version=self.runner.engine_version,
                extra={"exception": str(e)}
            )
        )
        raise e 


class RBFEWorkflow:
    def __init__(self, runner: GromacsRunner, prov_store: ProvenanceStore):
        self.runner = runner
        self.prov = prov_store

    def run_simulation(self, input_sets: list[GROMACSInputs], lambdas: list[float]):
        lambda_windows = []

        node = GromacsSimulationNode(self.runner, self.prov)

        for i, (inp, lam) in enumerate(zip(input_sets, lambdas)):
            lw, lw_fp = node.run_lambda_window(inp, lam)
            lambda_windows.append(lw)

        sim_run = SimulationRun(
            engine="GROMACS",
            engine_version="unknown",
            input_files={f"lambda_{i}": FileRef(inp.structure_file) for i, inp in enumerate(input_sets)},
            lambda_windows=lambda_windows,
            log_file=None,
            start_time=datetime.utcnow().isoformat(),
            end_time=None,
            wall_clock_seconds=None
        )
        sim_fp = self.prov.add_artifact(sim_run, parents=lambda_windows)
        return sim_run, sim_fp
