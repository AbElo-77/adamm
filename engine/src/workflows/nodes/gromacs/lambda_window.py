from typing import List
from engine.src.core.nodes import Node
from engine.src.core.artifacts import Artifact, LambdaWindow

class LambdaWindowNode(Node):
    def __init__(self, prov_store, lambda_value: float):
        super().__init__(name=f"lambda_window_{lambda_value}", prov_store=prov_store)
        self.lambda_value = lambda_value

    """
    inputs: List[Artifacts] containing the trajectory, energies, and thermodynamics. 
    outputs: LambdaWindow representing the entire simulation for this lambda. 
    """
    def run(self, inputs: List[Artifact]):
        traj, energy, thermodynamics = inputs

        lw = LambdaWindow(
            lambda_value=self.lambda_value,
            thermodynamics=thermodynamics,
            energy_series=energy,
            trajectory=traj,
            exchange_acceptance=None
        )
        self.prov.add_artifact(lw, parents=inputs)
        return [lw]
