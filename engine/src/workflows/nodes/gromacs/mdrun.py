from core.nodes import Node
from engines.gromacs.outputs import read_trajectory

class MdrunNode(Node):
    def __init__(self, runner, prov_store, lambda_value: float):
        super().__init__(name=f"mdrun_lambda_{lambda_value}", prov_store=prov_store)
        self.runner = runner
        self.lambda_value = lambda_value

    def run(self, inputs):
        tpr_file = f"lambda_{self.lambda_value}.tpr"
        deffnm = f"lambda_{self.lambda_value}"

        result = self.runner.run_mdrun(tpr_file, deffnm)

        traj = read_trajectory(f"{deffnm}.xtc", atom_count=None)
        self.prov.add_artifact(traj, parents=[tpr_file]) # review this here
        return [traj]
