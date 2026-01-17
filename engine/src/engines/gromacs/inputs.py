from dataclasses import dataclass
import hashlib
from typing import Dict, Optional
import json

@dataclass(frozen=True)
class MDConfig: 
    parameters: Dict[str,str]
    source_file: Optional[str] = None

@dataclass(frozen=True)
class GROMACSExecution:
    gromacs_binary: str = "gmx"
    ntmpi: Optional[int] = None
    ntomp: Optional[int] = None
    gpu_id: Optional[str] = None

@dataclass(frozen=True)
class GROMACSInputs: 
    """
    structure_file -> .gro, .pdb
    topology_file -> .top
    index_file -> .ndx
    restraint_file -> .itp
    """
    mdp: MDConfig
    structure_file: str
    topology_file: str
    index_file: Optional[str] = None
    restraint_file: Optional[str] = None

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            default=str
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()
    
    def to_dict(self):
        return {
            "mdp": {
                "parameters": self.mdp.parameters,
                "source_file": self.mdp.source_file
            },
            "structure_file": self.structure_file,
            "topology_file": self.topology_file,
            "index_file": self.index_file,
            "restraint_file": self.restraint_file
        }
    
@dataclass(frozen=True)
class SystemArtifacts:
    gro: str
    top: str
    tpr: Optional[str] = None