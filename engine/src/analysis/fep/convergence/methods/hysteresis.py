import numpy as np

from engine.src.utils.components import Samples, FreeEnergyEstimate
from engine.src.analysis.fep.estimators import SampleSelection, BAR, TI, ReducedPotentialBuilder

from engine.src.core.artifacts import SimulationRun, DiagnosticResult

from typing import List

"""
inputs: SimulationRun
outputs: DiagnosticResult

modify this script to give a range for convergence...not true / false
"""
class ForwardBackwardConvergence:

    def __init__(self, prov_store): 
        self.prov = prov_store

    def conv_check(self, simulation_run: SimulationRun) -> DiagnosticResult: 
    
        sample_selection = SampleSelection(self.prov)

        samples: List[Samples] = []
        for lw in simulation_run.lambda_windows: 
                samples.append(sample_selection.run(lw, 0.1, ineff_criteria='block'))
    
        ti = TI(self.prov)

        (bar_hys, ti_hys) = self.forward_backward(samples)

    def forward_backward(self, samples: List[Samples]): 
        return (self.bar_fb(samples), self.ti_fb(samples))

    def bar_fb(self, samples: List[Samples]):
        bar = BAR(self.prov)
        rp_builder = ReducedPotentialBuilder(self.prov)
        rp_ds = rp_builder.run(samples, trunc_policy=None)

        forward_ds = rp_ds.select_direction("forward")
        backward_ds = rp_ds.select_direction("backward")
        

        forward: FreeEnergyEstimate = bar.estimate(forward_ds)
        backward: FreeEnergyEstimate = bar.estimate(backward_ds)

        hysteresis = np.abs(forward.value + backward.value)
    
    """
    not implemented for GROMACS.  
    """
    def ti_fb(self, samples: List[Samples]): 
        lambdas = [s.thermodynamics.lambda_value for s in samples]
        dhdl_means = [
            np.mean(s.sampled_energy["dhdl"].values)
            for s in samples
        ]

        lambdas = np.array(lambdas)
        dhdl = np.array(dhdl_means)

        deltaG_fwd = np.trapezoid(dhdl, lambdas)
        deltaG_bwd = np.trapezoid(dhdl[::-1], lambdas[::-1])

        hysteresis = deltaG_fwd + deltaG_bwd