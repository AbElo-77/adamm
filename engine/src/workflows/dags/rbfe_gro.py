from engine.src.core.artifacts import Artifact, LambdaWindow, GROMACSInputs
from engine.src.core.nodes import Node
from engine.src.core.graphs import DAG

from engine.src.workflows.nodes.gromacs.grompp import GromppNode
from engine.src.workflows.nodes.gromacs.mdrun import MdrunNode
from engine.src.workflows.nodes.gromacs.energy import EnergyExtractionNode
from engine.src.workflows.nodes.gromacs.lambda_window import LambdaWindowNode
from engine.src.workflows.nodes.gromacs.thermodynamics import ThermodynamicsNode

from typing import List, Dict

class GROMACS_RBFE_DAG(DAG):

    def __init__(self):
        self.nodes: List[Node] = []
        self.dependencies: Dict[Node, List[Node]] = {}

    def execute(self, initial_input: GROMACSInputs) -> LambdaWindow:
        executed_nodes: set[Node] = set()
        artifacts: Dict[Node, Artifact] = {}

        while len(executed_nodes) < len(self.nodes):
            progress = False

            for node in self.nodes:
                if node in executed_nodes:
                    continue
                parents = self.dependencies.get(node, [])
                if not all(p in executed_nodes for p in parents):
                    continue

                if not parents:
                    output = node.execute(initial_input)
                else:
                    parent_outputs = [artifacts[p] for p in parents]
                    print(parent_outputs)
                    output = node.execute(parent_outputs)

                if len(executed_nodes) == len(self.nodes):
                    lw = output
                artifacts[node] = output
                executed_nodes.add(node)
                progress = True

            if not progress:
                raise RuntimeError("Circular dependency or missing artifacts")

        return lw

"""
This is a helper function for building the DAG for RBFE with GROMACS. 

    inputs: GROMACSRunner, ProvenanceStore, lambda_value
    outputs: DAG with grompp, thermodynamics, mdrun, energy, and lambda_window
"""
def build_rbfe_dag(runner, prov_store, lam):
    dag = GROMACS_RBFE_DAG()

    grompp = GromppNode(runner, prov_store, lam)
    thermo = ThermodynamicsNode(prov_store,  lam)
    mdrun = MdrunNode(runner, prov_store, lam)
    energy = EnergyExtractionNode(prov_store, lam)
    lw = LambdaWindowNode(prov_store, lam)

    dag.add_node(grompp)
    dag.add_node(thermo, parents=[grompp])
    dag.add_node(mdrun, parents=[grompp])
    dag.add_node(energy, parents=[mdrun])
    dag.add_node(lw, parents=[mdrun, energy])

    return dag
