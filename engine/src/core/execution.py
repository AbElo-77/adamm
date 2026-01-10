from typing import List
from core.nodes import Node
from core.artifacts import Artifact
from provenance.store import ProvenanceStore

class Executor:
    def __init__(self, prov_store: ProvenanceStore, cache_enabled: bool = True):
        self.prov = prov_store
        self.cache_enabled = cache_enabled

    def run_node(self, node: Node, inputs: List[Artifact]) -> List[Artifact]:
        cache_fp = "-".join([i.fingerprint() for i in inputs] + [node.name])
        if self.cache_enabled:
            cached = self.prov.get_artifact(cache_fp)
            if cached:
                print(f"[INFO] Node {node.name} skipped due to cache")
                return cached

        outputs = node.execute(inputs)
        if self.cache_enabled and outputs:
            self.prov.add_artifact(outputs[0]) 
        return outputs

    def run_graph(self, nodes: List[Node], inputs: List[Artifact] = None) -> List[Artifact]:
        all_artifacts = inputs or []
        for node in nodes:
            node_outputs = self.run_node(node, all_artifacts)
            all_artifacts.extend(node_outputs)
        return all_artifacts
