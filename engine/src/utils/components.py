import numpy as np

from dataclasses import dataclass
from typing import Dict, List,  Any
from core.artifacts import Artifact, EnergySeries, ThermodynamicState, LambdaWindow

"""

"""
@dataclass
class FreeEnergyEstimate:
    value: float
    stderr: float
    estimator: str
    metadata: dict # should include reference states

"""

"""
@dataclass(frozen=True)
class Samples(Artifact): 
    sampled_energy = Dict[str, EnergySeries]
    thermodynamics = ThermodynamicState
    atom_count: int
    burn_in: float = 0.1
    inefficiency_criteria: str = 'block'
    reference_lambda: LambdaWindow

    def to_dict(self) -> Dict[str, Any]: 
        return {
            "sampled_traj": self.sampled_energy, 
            "thermodynamics": self.thermodynamics,
            "atom_count": self.atom_count,
            "burn_in": self.burn_in, 
            "inefficiency_criteria": self.inefficiency_criteria, 
            "reference_lambda": self.reference_lambda.to_dict()
        }

"""

"""    
@dataclass(frozen=True)
class ReducedPotentialDataset(Artifact):
    u_kn: np.ndarray
    sample_state_indices: np.ndarray
    state_fingerprints: List[str]
    source_runs: List[str]
    truncation_policy: Dict[str, Any]

    def to_dict(self):
        return {
            "u_kn": self.u_kn.tolist(),
            "sample_state_indices": self.sample_state_indices.tolist(),
            "state_fingerprints": self.state_fingerprints,
            "source_runs": self.source_runs,
            "truncation_policy": self.truncation_policy,
        }
