from typing import List, Optional
from datetime import datetime
from core.artifacts import Artifact
from provenance.store import ProvenanceStore
from provenance.metadata import ArtifactMetadata

class Node:
    def __init__(self, name: str, prov_store: ProvenanceStore):
        self.name = name
        self.prov = prov_store
        self.inputs: List[Artifact] = []
        self.outputs: List[Artifact] = []
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None

    def execute(self, inputs: List[Artifact]) -> List[Artifact]:
        self.inputs = inputs
        self.start_time = datetime.now().isoformat()
        outputs = self.run(inputs)
        self.end_time = datetime.now().isoformat()

        for out in outputs:
            self.prov.add_artifact(
                out,
                metadata=ArtifactMetadata(
                    created_by=self.name,
                    created_at=self.start_time
                ),
                parents=inputs
            )
        self.outputs = outputs
        return outputs

    def run(self, inputs: List[Artifact]) -> List[Artifact]:
        raise NotImplementedError("Subclasses must implement the run method.")