from typing import Dict, List, Optional

class LineageRecord:
    def __init__(self, artifact_hash: str):
        self.artifact_hash = artifact_hash
        self.parents: List[str] = []
        self.children: List[str] = []

    def add_parent(self, parent_hash: str):
        if parent_hash not in self.parents:
            self.parents.append(parent_hash)

    def add_child(self, child_hash: str):
        if child_hash not in self.children:
            self.children.append(child_hash)
