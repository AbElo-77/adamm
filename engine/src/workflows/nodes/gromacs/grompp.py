from engine.src.core.nodes import Node
from engine.src.core.artifacts import FileRef
from engine.src.engines.gromacs.runner import GromacsRunner
from engine.src.provenance.metadata import ArtifactMetadata

class GromppNode(Node):
    def __init__(self, runner, prov_store, lambda_value: float):
        super().__init__(name=f"grompp_lambda_{lambda_value}", prov_store=prov_store)
        self.runner: GromacsRunner = runner
        self.lambda_value = lambda_value

    """
    inputs: GROMACSInputs; this is the user-defined input to the process. 
    outputs: FileRef; this is the path to the trajectory.
    """
    def run(self, inputs):
        tpr_file = f"lambda_{self.lambda_value}.tpr"

        result = self.runner.run_grompp(inputs, tpr_file)

        tpr = FileRef(tpr_file)
        self.prov.add_artifact(
            tpr,
            metadata=ArtifactMetadata(
                created_by=self.name,
                engine="GROMACS",
                engine_version=self.runner._detect_version(),
                extra={
                    "exit_code": result.returncode,
                    "stderr": result.stderr
                }
            ),
            parents=[inputs]
        )
        return tpr
