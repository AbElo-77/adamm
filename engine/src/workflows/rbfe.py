from typing import List, Any
from engine.src.core.artifacts import SimulationRun, LambdaWindow
from engine.src.core.graphs import GROMACS_RBFE_DAG
from engine.src.engines.gromacs.runner import GromacsRunner
from engine.src.provenance.store import ProvenanceStore
from engine.src.workflows.nodes.gromacs.simulation_run import SimulationRunNode
from engine.src.workflows.dags.rbfe_gro import build_rbfe_dag

class RBFEWorkflow:
    def __init__(self, runner: GromacsRunner, prov_store: ProvenanceStore):
        self.runner = runner
        self.prov = prov_store

    def run_simulation(
        self,
        input_sets: List[Any],
        lambdas: List[float]
    ) -> SimulationRun:

        if len(input_sets) != len(lambdas):
            raise ValueError("input_sets and lambdas must have equal length")

        lws: List[LambdaWindow] = []

        for inputs, lam in zip(input_sets, lambdas):
            dag: GROMACS_RBFE_DAG = build_rbfe_dag(
                runner=self.runner,
                prov_store=self.prov,
                lam=lam
            )
            lw = dag.execute(inputs)
            lws.append(lw)

        sim_node = SimulationRunNode(
            prov_store=self.prov,
            engine="GROMACS",
            engine_version=self.runner._detect_version()
        )

        return sim_node.execute(lws)

