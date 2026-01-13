from typing import List
from core.execution import Executor
from core.artifacts import Artifact, SimulationRun
from engines.gromacs.runner import GromacsRunner
from provenance.store import ProvenanceStore
from workflows.dags.rbfe_gro import build_rbfe_dag

class RBFEWorkflow:
    def __init__(self, runner: GromacsRunner, prov_store: ProvenanceStore):
        self.runner = runner
        self.prov = prov_store
        self.executor = Executor(prov_store=self.prov, cache_enabled=True)

    def run_simulation(
        self,
        input_sets: List[Artifact],
        lambdas: List[float]
    ) -> SimulationRun:

        dag = build_rbfe_dag(
            runner=self.runner,
            prov_store=self.prov,
            input_artifacts=input_sets,
            lambdas=lambdas
        )

        all_artifacts = self.executor.execute_dag(
            dag=dag,
            initial_artifacts=input_sets
        )

        sim_runs = [
            a for a in all_artifacts
            if isinstance(a, SimulationRun)
        ]

        if len(sim_runs) != 1:
            raise RuntimeError(
                f"Expected exactly one SimulationRun, found {len(sim_runs)}"
            )

        return sim_runs[0]
