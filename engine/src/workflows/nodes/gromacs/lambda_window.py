from core.nodes import Node
from core.artifacts import LambdaWindow

class LambdaWindowNode(Node):
    def __init__(self, prov_store, lambda_value: float):
        super().__init__(name=f"lambda_window_{lambda_value}", prov_store=prov_store)
        self.lambda_value = lambda_value

    def run(self, inputs):
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
