from core.nodes import Node
from core.artifacts import SimulationRun
from datetime import datetime

class SimulationRunNode(Node):
    def __init__(self, prov_store, engine, engine_version):
        super().__init__("simulation_run", prov_store)
        self.engine = engine
        self.engine_version = engine_version

    def run(self, inputs):
        start = datetime.utcnow().isoformat()
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
