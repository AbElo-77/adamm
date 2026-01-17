from typing import List, Dict, Any
from engine.src.core.nodes import Node
from engine.src.core.artifacts import Artifact
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


