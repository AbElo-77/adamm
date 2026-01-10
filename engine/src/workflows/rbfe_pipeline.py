from datetime import datetime
from typing import List

from core.artifacts import (
    FileRef, Trajectory, EnergySeries, LambdaWindow, SimulationRun
)
from provenance.store import ProvenanceStore
from provenance.metadata import ArtifactMetadata
from engines.gromacs.runner import GromacsRunner, CommandResult
from engines.gromacs.inputs import GROMACSInputs, MDPConfig
from engines.gromacs.outputs import parse_xvg, read_trajectory

class GromacsSimulationNode:
    """
    Node wrapper for a single λ-window GROMACS simulation.
    Produces all artifacts and records provenance.
    """
    def __init__(self, runner: GromacsRunner, prov_store: ProvenanceStore):
        self.runner = runner
        self.prov = prov_store

    def run_lambda_window(self, inputs: GROMACSInputs, lambda_value: float) -> LambdaWindow:
        try:
            tpr_file = f"lambda_{lambda_value}.tpr"
            result_grompp: CommandResult = self.runner.run_grompp(inputs, tpr_file)

            tpr_fp = self.prov.add_artifact(
                FileRef(path=tpr_file),
                metadata=ArtifactMetadata(
                    created_by="GromacsSimulationNode.grompp",
                    created_at=result_grompp.start_time,
                    engine="GROMACS",
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
            result_mdrun: CommandResult = self.runner.run_mdrun(tpr_file, deffnm)

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
            return lw
        except Exception as e:
            fail_fp = self.prov.add_artifact(
                FileRef(path=f"lambda_{lambda_value}_failed.log"),
                metadata=ArtifactMetadata(
                    created_by="GromacsSimulationNode",
                    created_at=datetime.utcnow().isoformat(),
                    engine="GROMACS",
                    engine_version=self.runner.engine_version,
                    extra={"exception": str(e)}
                )
            )
            raise e

class RBFEWorkflow:
    def __init__(self, runner: GromacsRunner, prov_store: ProvenanceStore):
        self.runner = runner
        self.prov = prov_store

    def run_simulation(self, input_sets: List[GROMACSInputs], lambdas: List[float]) -> SimulationRun:
        start_time = datetime.now().isoformat()
        node = GromacsSimulationNode(self.runner, self.prov)
        lambda_windows: List[LambdaWindow] = []

        for inp, lam in zip(input_sets, lambdas):
            lw = node.run_lambda_window(inp, lam)
            lambda_windows.append(lw)

        end_time = datetime.now().isoformat()

        sim_run = SimulationRun(
            engine="GROMACS",
            engine_version=self.runner.engine_version,
            input_files={f"lambda_{i}": FileRef(inp.structure_file) for i, inp in enumerate(input_sets)},
            lambda_windows=lambda_windows,
            log_file=None,
            start_time=start_time,
            end_time=end_time,
            wall_clock_seconds=None
        )
        sim_fp = self.prov.add_artifact(sim_run, parents=lambda_windows)
        print(f"[INFO] SimulationRun artifact stored with fingerprint {sim_fp}")
        return sim_run
