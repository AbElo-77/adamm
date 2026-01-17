from typing import List
from engine.src.core.nodes import Node
from engine.src.core.artifacts import LambdaWindow, SimulationRun
from datetime import datetime

class SimulationRunNode(Node):
    def __init__(self, prov_store, engine, engine_version):
        super().__init__("simulation_run", prov_store)
        self.engine = engine
        self.engine_version = engine_version

    """
    inputs: List[LambdaWindows]; this is the simulation for each lambda. 
    outputs: SimulationRun; this is the entire RBFE simulation, including metadata.
    """
    def run(self, inputs: List[LambdaWindow]):
        start = datetime.now().isoformat()
        sim = SimulationRun(
            engine=self.engine,
            engine_version=self.engine_version,
            input_files={},
            lambda_windows=inputs,
            log_file=None,
            start_time=start,
            end_time=None,
            wall_clock_seconds=None
        )
        self.prov.add_artifact(sim, parents=inputs)
        return [sim]
