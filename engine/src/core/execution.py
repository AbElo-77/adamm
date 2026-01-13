from core.graphs import DAG
from core.nodes import Node
from core.artifacts import Artifact
from provenance.store import ProvenanceStore
from typing import List

class Executor:
    def __init__(self, prov_store: ProvenanceStore, cache_enabled: bool = True):
        self.prov = prov_store
        self.cache_enabled = cache_enabled

    def execute_dag(
        self,
        dag: DAG,
        initial_artifacts: List[Artifact]
    ) -> List[Artifact]:
        artifacts = list(initial_artifacts)

        executed = set()

        while len(executed) < len(dag.nodes):
            progress = False

            for node in dag.nodes:
                if node in executed:
                    continue

                parents = dag.dependencies.get(node, [])
                if not all(p in executed for p in parents):
                    continue

                outputs = node.execute(artifacts)
                artifacts.extend(outputs)
                executed.add(node)
                progress = True

            if not progress:
                raise RuntimeError("DAG execution stalled (cycle or missing dependency)")

        return artifacts
