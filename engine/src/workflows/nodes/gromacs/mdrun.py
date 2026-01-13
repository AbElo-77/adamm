from core.artifacts import FileRef
from core.nodes import Node
from engines.gromacs.outputs import read_trajectory
from provenance.metadata import ArtifactMetadata

from datetime import datetime

class MdrunNode(Node):
    def __init__(self, runner, prov_store, lambda_value: float):
        super().__init__(name=f"mdrun_lambda_{lambda_value}", prov_store=prov_store)
        self.runner = runner
        self.lambda_value = lambda_value

    """
    inputs: FileRef; this is the path to the trajectory file
    outputs: ThermodynamicState; this is the thermodynamic state being sampled. 
    """
    def run(self, inputs: FileRef):
        tpr_file = inputs.path
        deffnm = f"lambda_{self.lambda_value}"

        start = datetime.now().isoformat()
        result = self.runner.run_mdrun(tpr_file, deffnm)

        traj = read_trajectory(f"{deffnm}.xtc", atom_count=None)
        self.prov.add_artifact(
            traj, 
            metadata=ArtifactMetadata(
                created_by=self.name, 
                create_at = start,
                engine="GROMACS", 
                engine_version=self.runner._detect_version(), 
                extra={
                    "exit_code": result.returncode,
                    "stderr": result.stderr
                }
            ),
            parents=[inputs]) 
        return [traj]
