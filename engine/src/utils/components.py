from dataclasses import dataclass
from typing import Dict, Any
from core.artifacts import Artifact, EnergySeries, ThermodynamicState, LambdaWindow

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
            "reference_lambda": self.reference_lambda
        }