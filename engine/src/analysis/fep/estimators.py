import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from core.artifacts import Artifact, EnergySeries, ThermodynamicState
from core.nodes import Node
from utils.components import Samples, FreeEnergyEstimate, ReducedPotentialDataset
    
"""
inputs: fingerprint to the lambda window, burn_in fraction, and method for removing ineffiency. 
outputs: Samples containing the sampled_energies, n_eff, and reference lambda.
"""   
class SampleSelection(Node): 
    def __init__(self, prov_store): 
        super.__init__(name="sample_selector", prov=prov_store)
        self.prov = prov_store
    
    def run(self, fingerprint: str, burn_in = 0.1, ineff_criteria = 'block') -> Samples:
        lw = self.prov.get_artifact(fingerprint)
        energy_series = lw["energy_series"]
        thermodynamics = lw["thermodynamic_state"]

        energy_series = self.apply_burn_in(energy_series, burn_in)
        if len(energy_series["pontential"]) < 5000:
               
               g_ineff = self.compute_stat_ineff(energy_series)
        else:   
            g_ineff = self.compute_stat_ineff_block(energy_series)

        if ineff_criteria == 'block':
            sampled_energy = self.sample_energies_block(energy_series, g_ineff)
        else: 
            sampled_energy = self.sample_energies_thin(energy_series, g_ineff)

        sample = Samples(
            sampled_energy=sampled_energy, 
            thermodynamics=thermodynamics,
            n_eff = len(energy_series["potential"]) / g_ineff,
            burn_in=burn_in, 
            inefficiency_criteria=ineff_criteria,
            reference_lambda=fingerprint
        )
        self.prov.add_artifact(sample, parents=[lw])

        return [sample]

    def apply_burn_in(self, energy_series: Dict[str, EnergySeries], burn_in) -> Dict[str, EnergySeries]: 
            assert 0.0 <= burn_in < 1.0

            trimmed = {}
            for key, series in energy_series.items():
                n = len(series.values)
                start = int(n * burn_in)

                trimmed[key] = EnergySeries(
                    values=series.values[start:],
                    units=series.units,
                    sampling_interval=series.sampling_interval,
                    source_file=series.source_file,
                    metadata={
                        **series.metadata,
                        "burn_in_fraction": burn_in
                    }
                )

            return trimmed

    def compute_stat_ineff(self, energy_series: Dict[str, EnergySeries]) -> float:
        series = energy_series["potential"].values
        N = len(series)

        if N < 2:
            return 1.0

        mu = np.mean(series)
        var = np.mean((series - mu) ** 2)

        if var == 0:
            return 1.0

        g = 1.0
        C_prev = 1.0
        max_lag = N // 2

        for t in range(1, max_lag):
            cov = 0.0
            for i in range(N - t):
                cov += (series[i] - mu) * (series[i + t] - mu)

            C_t = cov / ((N - t) * var)
            if C_t + C_prev < 0:
                break

            g += 2.0 * C_t
            C_prev = C_t

        return max(g, 1.0)
    
    def compute_stat_ineff_block(self,
        energy_series: Dict[str, EnergySeries],
        min_blocks: int = 10,
        plateau_window: int = 3,
        plateau_rtol: float = 0.1,
    ) -> float:

        series = energy_series["potential"].values
        x = np.asarray(series, dtype=float)
        N = x.size

        if N < 2:
            return 1.0, float(N), {}, 1

        mu = x.mean()
        var0 = np.mean((x - mu) ** 2)

        if var0 == 0.0:
            return 1.0, float(N), {}, 1

        g_curve: Dict[int, float] = {}

        B = 1
        while True:
            M = N // B
            if M < min_blocks:
                break

            trimmed = x[:M * B]
            blocks = trimmed.reshape(M, B)
            block_means = blocks.mean(axis=1)

            varB = np.var(block_means, ddof=1)
            gB = (B * varB) / var0

            g_curve[B] = max(float(gB), 1.0)
            B *= 2

        if not g_curve:
            return 1.0, float(N), {}, 1
        
        g_values = list(g_curve.values())
        g = g_values[-1]

        if len(g_values) >= plateau_window:
            tail = np.array(g_values[-plateau_window:])
            spread = (tail.max() - tail.min()) / np.mean(tail)

            if spread <= plateau_rtol:
                g = float(np.median(tail))

        return max(g, 1.0)
    
    def sample_energies_block(self, energies, g) -> Dict[str, EnergySeries]:
        B = int(np.ceil(g))

        if B < 1:
            raise ValueError("Block size must be >= 1")

        blocked: Dict[str, EnergySeries] = {}

        for key, series in energies.items():
            x = np.asarray(series.values)

            M = len(x) // B
            if M < 2:
                raise ValueError(
                    f"Not enough samples to block observable '{key}': "
                    f"{len(x)} samples, block size {B}"
                )

            trimmed = x[:M * B]
            blocks = trimmed.reshape(M, B)
            block_means = blocks.mean(axis=1)

            blocked[key] = EnergySeries(
                values=block_means,
                units=series.units,
                sampling_interval=(
                    series.sampling_interval * B
                    if series.sampling_interval is not None
                    else None
                ),
                source_file=series.source_file,
                metadata={
                    **(series.metadata or {}),
                    "statistical_inefficiency": g,
                    "block_size": B,
                    "n_blocks": M,
                },
            )

        return blocked 
    
    def sample_energies_thin(self, energies, g) -> Dict[str, EnergySeries]:
        stride = np.ceil(g)

        thinned: Dict[str, EnergySeries] = {}
        for k, v in energies.items(): 
            thinned[k] = EnergySeries(
                    values=v.values[::stride],
                    units=v.units,
                    sampling_interval=v.sampling_interval,
                    source_file=v.source_file,
                    metadata=v.metadata
            )

        return thinned


"""
inputs: set of Samples corresponding to different lambdas. 
outputs: ReducedPotentialDataset; this is the general Estimator input. 
"""
class ReducedPotentialBuilder(Node): 
    def __init__(self, prov_store): 
        super.__init__(name="reduced_potential_builder", prov=prov_store)
        self.prov = prov_store
    
    def run(self, 
            samples: List[Samples], 
            trunc_policy
        ) -> ReducedPotentialDataset:

        energies = [sample.sampled_energy for sample in samples]
        thermodynamics = [sample.thermodynamics for sample in samples]

        assert len(energies) == len(thermodynamics)

        u_kn = self.generate_matrix(energies, thermodynamics=thermodynamics)

        dataset =  ReducedPotentialDataset(
            u_kn=u_kn,
            samples=[sample.fingerprint for sample in samples],
            truncation_policy=trunc_policy
        )
        self.prov.add_artifact(dataset, parents=[samples])

        return [dataset]
    
    def generate_matrix(
        self,
        energies: List[Dict[str, EnergySeries]],
        thermodynamics: List[ThermodynamicState],
    ) -> np.ndarray:

        k_B = 0.008314462618
        K = len(thermodynamics)

        n_samples_per_state = [
            len(e["potential"].values)
            for e in energies
        ]

        N_total = sum(n_samples_per_state)

        u_kn = np.full((K, N_total), np.nan, dtype=np.float64)

        betas = np.array([
            thermo.beta
            if thermo.beta is not None
            else 1.0 / (k_B * thermo.temperature)
            for thermo in thermodynamics
        ])

        col_offset = 0

        for i, energy in enumerate(energies):
            pot = energy["potential"].values
            dH_fwd = energy.get("delta_h_forward")
            dH_bwd = energy.get("delta_h_backward")

            n_i = len(pot)

            for n in range(n_i):
                col = col_offset + n

                u_kn[i, col] = betas[i] * pot[n]

                if dH_fwd is not None and i + 1 < K:
                    u_kn[i + 1, col] = betas[i + 1] * (pot[n] + dH_fwd.values[n])

                if dH_bwd is not None and i - 1 >= 0:
                    u_kn[i - 1, col] = betas[i - 1] * (pot[n] + dH_bwd.values[n])

            col_offset += n_i

        return u_kn

"""
This is a template estimator class to generate a free energy estimate.
"""
class Estimator:
    def estimate(self, inputs: ReducedPotentialDataset) -> FreeEnergyEstimate:
        raise NotImplementedError("Subclasses must implement estimate().")

"""
This class estimates free energy with Bennet's Acceptance Ratio (BAR).

    implemented in GROMACS, OpenMM, AMBER
    inputs: ReducedPotentialDataset for two Samples
    outputs: FreeEnergyEstimate
"""
class BAR(Estimator): 

    def estimate(self, inputs: ReducedPotentialDataset) -> FreeEnergyEstimate:
        pass

"""
This class estimates free energy with Multistate Bennet's Acceptance Ratio (MBAR).

    implemented in OpenMM, AMBER
    inputs: ReducedPotentialDataset for Samples corresponding to a simulation run
    outputs: FreeEnergyEstimate 
"""
class MBAR(Estimator): 
    def estimate(self, inputs: ReducedPotentialDataset) -> FreeEnergyEstimate:
        pass

"""
This class estiamtes free energy with thermodynamic integration (TI). 

    implemented in GROMACS
    inputs: Samples cooresponding to a simulation run
    outputs: FreeEnergyEstimate
"""
class TI(Estimator): 
    def estimate(self, inputs: List[Samples]) -> FreeEnergyEstimate:
        pass

"""

"""
@dataclass(frozen=True)
class EstimatorRun(Artifact):
    free_energy: FreeEnergyEstimate
    parameters: Dict[str, Any]
    solver_settings: Dict[str, Any]
    converged: bool
    iterations: Optional[int]
    residual: Optional[float]

    def to_dict(self):
        return {
            "free_energy": self.free_energy, 
            "parameters": self.parameters,
            "solver_settings": self.solver_settings,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
        }