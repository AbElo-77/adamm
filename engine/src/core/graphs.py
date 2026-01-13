from typing import List, Dict, Any
from core.nodes import Node
from core.artifacts import Artifact, LambdaWindow
from engines.gromacs.inputs import GROMACSInputs

"""
This is the DAG used to store nodes sequentially. 
"""
class DAG:
    def __init__(self):
        self.nodes: List[Node] = []
        self.dependencies: Dict[Node, List[Node]] = {}  

    def add_node(self, node: Node, parents: List[Node] = None):
        self.nodes.append(node)
        self.dependencies[node] = parents or []

    def execute(self, initial_input: List[Any]) -> Artifact:
        raise NotImplementedError("Subclasses must implement the execute method.")
    
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
                    output = node.execute(parent_outputs)

                if len(executed_nodes) == len(self.nodes):
                    lw = output
                artifacts[node] = output
                executed_nodes.add(node)
                progress = True

            if not progress:
                raise RuntimeError("Circular dependency or missing artifacts")

        return lw

