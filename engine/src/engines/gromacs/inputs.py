from dataclasses import dataclass
from typing import Dict, Optional

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
    index_file: Optional[str]
    restraint_file: Optional[str]
