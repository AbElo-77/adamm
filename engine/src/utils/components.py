import numpy as np

from dataclasses import dataclass
from typing import Dict, List,  Any
from core.artifacts import Artifact, EnergySeries, ThermodynamicState, LambdaWindow

"""
This artifact stores information about sampled energies from a LambdaWindow. 

    sampled_energy -> energy reduced for statiscal efficiency 
    thermodynamics -> properties of the thermodynamic state for the lw
    n_eff -> effective number of samples after removing inefficiency 
    burn_in -> fraction removed from start of simulation
    ineff_criteria -> technique for removing inefficiency (e.g., block)
    lambda_fingerprint -> reference lambda state
"""
@dataclass(frozen=True)
class Samples(Artifact): 
    sampled_energy: Dict[str, EnergySeries]
    thermodynamics: ThermodynamicState
    n_eff: float
    burn_in: float = 0.1
    inefficiency_criteria: str = 'block'
    lambda_fingerprint: LambdaWindow.fingerprint

    def to_dict(self) -> Dict[str, Any]: 
        return {
            "sampled_traj": self.sampled_energy, 
            "thermodynamics": self.thermodynamics,
            "n_eff": self.n_eff,
            "burn_in": self.burn_in, 
            "inefficiency_criteria": self.inefficiency_criteria, 
            "lambda_fingerprint": self.lambda_fingerprint
        }

"""
This artifact stores information about the reduced potentials among a set of samples. 

    u_kn -> thermodynamic-potential array
    samples -> the states for which the reduced potential dataset is constructed
    truncation_policy -> method for bounding values
"""    
@dataclass(frozen=True)
class ReducedPotentialDataset(Artifact):
    u_kn: np.ndarray
    samples: List[Samples.fingerprint]
    truncation_policy: Dict[str, Any]

    def to_dict(self):
        return {
            "u_kn": self.u_kn.tolist(),
            "samples": self.samples,
            "truncation_policy": self.truncation_policy,
        }

"""
This artifact stores information about a free energy estimate. 

    value -> the free energy
    stderr -> the statistical uncertainty
    units -> ... 
    estimator -> the method used to compute the free energy (e.g., BAR)
    dataset: the ReducedPotentialDataset of the samples
    samples -> the Samples between which free energy difference is calculated
    reference_State -> the reference free energy state
"""
@dataclass
class FreeEnergyEstimate(Artifact):
    value: float
    stderr: float
    units: str = "kJ/mol"
    estimator: str
    dataset: ReducedPotentialDataset.fingerprint
    samples: List[Samples.fingerprint]
    reference_state: Samples.fingerprint

    def to_dict(self):
        return {
            "value": self.value, 
            "stderr": self.stderr, 
            "units": self.units, 
            "estimator": self.estimator, 
            "dataset":  self.dataset, 
            "samples": self.samples, 
            "reference_state": self.reference_state
        }