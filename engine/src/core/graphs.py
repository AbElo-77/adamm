from typing import List, Dict
from core.nodes import Node
from core.artifacts import Artifact

class DAG:
    def __init__(self):
        self.nodes: List[Node] = []
        self.dependencies: Dict[Node, List[Node]] = {}  

    def add_node(self, node: Node, parents: List[Node] = None):
        self.nodes.append(node)
        self.dependencies[node] = parents or []

    def execute(self, initial_artifacts: List[Artifact] = None) -> List[Artifact]:
        executed_nodes = set()
        artifacts = initial_artifacts or []

        while len(executed_nodes) < len(self.nodes):
            progress = False
            for node in self.nodes:
                if node in executed_nodes:
                    continue
                parents = self.dependencies.get(node, [])
                if all(p in executed_nodes for p in parents):
                    node_outputs = node.execute(artifacts)
                    artifacts.extend(node_outputs)
                    executed_nodes.add(node)
                    progress = True
            if not progress:
                raise RuntimeError("Circular dependency detected or missing artifacts")
        return artifacts
