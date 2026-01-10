import os
import json
from pathlib import Path
from typing import Optional, Dict
from core.artifacts import Artifact
from .hashing import fingerprint
from .lineage import LineageRecord
from .metadata import ArtifactMetadata

class ProvenanceStore:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lineage: Dict[str, LineageRecord] = {}

    def add_artifact(
        self,
        artifact: Artifact,
        metadata: Optional[ArtifactMetadata] = None,
        parents: Optional[list[Artifact]] = None,
    ) -> str:
        fp = artifact.fingerprint()

        artifact_path = self.root / f"{fp}.json"
        artifact_path.write_text(json.dumps(artifact.to_dict(), indent=2))

        meta = metadata.to_dict() if metadata else {}
        meta_path = self.root / f"{fp}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        record = LineageRecord(fp)
        if parents:
            for p in parents:
                record.add_parent(p.fingerprint())
        self.lineage[fp] = record

        return fp

    def get_artifact(self, fp: str) -> dict:
        path = self.root / f"{fp}.json"
        if not path.exists():
            raise FileNotFoundError(f"Artifact {fp} not found")
        return json.loads(path.read_text())

    def get_metadata(self, fp: str) -> dict:
        path = self.root / f"{fp}.meta.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def get_lineage(self, fp: str) -> LineageRecord:
        return self.lineage.get(fp)
