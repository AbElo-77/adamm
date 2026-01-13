from dataclasses import dataclass
from typing import Dict
from core.artifacts import EnergySeries
from core.nodes import Node
from utils.components import Samples

@dataclass
class Estimate:
    value: float
    stderr: float
    estimator: str
    metadata: dict

class Estimator:
    def estimate(self, fingerprint) -> Estimate:
        raise NotImplementedError("Subclasses must implement estimate().")

"""
Apply burn-in before sampling; determine g per energy type. 
Determine N_eff as len(energy["potential"]) / g. 
"""   
class SampleSelection(Node): 
    def __init__(self, prov_store): 
        super.__init__(name="sample_selector", prov=prov_store)
        self.prov = prov_store
    
    def generate_samples(self, fingerprint: str, burn_in = 0.1, ineff_criteria = 'block') -> Samples:
        lw = self.prov.get_artifact(fingerprint)
        energy_series = lw["energy_series"]
        thermodynamics = lw["thermodynamic_state"]

        g_ineff = self.compute_stat_ineff(energy_series)

        if ineff_criteria == 'block':
            sampled_energy = self.sample_energies_block(energy_series, thermodynamics, g_ineff)
        else: 
            sampled_energy = self.sample_energies_thin(energy_series, thermodynamics, g_ineff)

        return Samples(
            sampled_energy=sampled_energy, 
            thermodynamics=thermodynamics,
            atom_count=lw["trajectory"]["atom_count"],
            burn_in=burn_in, 
            inefficiency_criteria=ineff_criteria,
            reference_lambda=lw
        )
    
    def compute_stat_ineff(self, energies) -> float:
        pass
    
    def sample_energies_block(self, energies, thermodynamic_state, g) -> Dict[str, EnergySeries]:
        pass

    def sample_energies_thin(self, energies, thermodynamic_state, g) -> Dict[str, EnergySeries]:
        pass

class BAR(Estimator): 
    def estimate(self, fingerprints: Dict[str, str]):
        pass

class TI(Estimator): 
    def estimate(self, fingerprints: Dict[str, str]):
        pass

class MBAR(Estimator): 
    def estimate(self, fingerprint:  Dict[str, str]):
        pass