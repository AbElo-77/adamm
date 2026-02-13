from engine.src.utils.components import Samples, FreeEnergyEstimate
from engine.src.analysis.fep.estimators import SampleSelection, ReducedPotentialBuilder, BAR, TI

from engine.src.core.artifacts import SimulationRun, DiagnosticResult, EnergySeries
from typing import List, Dict

import numpy as np

"""
inputs: SimulationRun
outputs: DiagnosticResult

modify this script to give a range for convergence...not true / false
"""
class TimeSlicedConvergence:

    def __init__(self, prov_store):
        self.prov = prov_store

    def conv_check(self, simulation_run: SimulationRun) -> DiagnosticResult: 

        sample_selection = SampleSelection(self.prov)

        samples: List[Samples] = []
        for lw in simulation_run.lambda_windows: 
            samples.append(sample_selection.run(lw, 0.1, ineff_criteria='block'))
        

        ts_intervals = [0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9, 1.0]
        outputs = self.estimate_intervals(samples, ts_intervals, simulation_run)
        
        decision = self.convergence_decision(outputs["bar_series"], outputs["ti_series"], ts_intervals)
        diag = DiagnosticResult(
            "time_sliced_convergence", 
            decision, 
            units=None, 
            context=None,
            source_run= simulation_run
            
        )
        self.prov.add_artifact(diag, parents=[simulation_run])

        return diag
    
    def convergence_decision(self, bar_series, ti_series, time_fractions) -> DiagnosticResult:
        bar_vals, bar_errs = self.extract_series(bar_series)
        ti_vals,  ti_errs  = self.extract_series(ti_series)

        diagnostics = {
            "bar_time_stability": self.time_stability(bar_vals, bar_errs),
            "ti_time_stability":  self.time_stability(ti_vals, ti_errs),
            "bar_uncertainty_sat": self.uncertainty_saturation(bar_errs),
            "ti_uncertainty_sat":  self.uncertainty_saturation(ti_errs),
            "estimators_agree": self.estimator_agreement(
                bar_vals[-1], bar_errs[-1],
                ti_vals[-1],  ti_errs[-1]
            ),
            "bar_drift_slope": self.drift_slope(time_fractions, bar_vals),
            "ti_drift_slope":  self.drift_slope(time_fractions, ti_vals),
        }

        converged = (
            diagnostics["bar_time_stability"] > 0.7 and
            diagnostics["ti_time_stability"] > 0.7 and
            diagnostics["estimators_agree"] and
            abs(diagnostics["bar_drift_slope"]) < 0.5 and
            abs(diagnostics["ti_drift_slope"]) < 0.5
        )

        return converged, diagnostics

    def estimate_intervals(
        self,
        full_samples: list[Samples],
        ts_intervals: list[float], 
        sim: SimulationRun
    ) -> Dict[str, List[FreeEnergyEstimate]]:

        rp_builder = ReducedPotentialBuilder(self.prov)
        bar = BAR(self.prov)
        ti = TI(self.prov)

        bar_series = []
        ti_series = []

        for ts in ts_intervals:
            
            sliced_samples = [
                self.slice_samples(s, ts) for s in full_samples
            ]

            rp_data = rp_builder.run(
                sliced_samples,
                trunc_policy="explicit"
            )
            bar_est = bar.estimate(rp_data)
            bar_series.append(bar_est)

            
            ti_est = ti.estimate(sliced_samples)
            ti_series.append(ti_est)

        return {
            "ti_series": ti_series, 
            "bar_series": bar_series
        }
    
    def slice_samples(self, samples: Samples, fraction: float) -> Samples:
        assert 0.0 < fraction <= 1.0

        sliced_energy = {}

        for name, series in samples.sampled_energy.items():
            values = series.values
            n = int(len(values) * fraction)

            sliced_energy[name] = EnergySeries(
                values=values[:n],
                units=series.units, 
                sampling_interval=series.sampling_interval, 
                source_file=series.source_file, 
                metadata=series.metadata
            )

        return Samples(
            sampled_energy=sliced_energy,
            thermodynamics=samples.thermodynamics,
            n_eff=samples.n_eff * fraction,
            burn_in=samples.burn_in,
            inefficiency_criteria=samples.inefficiency_criteria,
            lambda_fingerprint=samples.fingerprint
        )
    
    def extract_series(self, estimates: List[FreeEnergyEstimate]):
        values = np.array(e.value for e in estimates)
        errors = np.array(e.stderr for e in estimates)

        return (values, errors)
    
    def time_stability(self, values, errors, window=3, sigma=1.0):
        ref = values[-1]
        ref_err = errors[-1]

        stable = []
        for i in range(len(values) - window):
            delta = abs(values[i] - ref)
            tol = sigma * np.sqrt(errors[i]**2 + ref_err**2)
            stable.append(delta < tol)

        return np.mean(stable)
    
    def uncertainty_saturation(self, errors):
        ratios = errors[1:] / errors[:-1]
        return np.median(ratios)
    
    def estimator_agreement(self, bar_val, bar_err, ti_val, ti_err, sigma=1.0):
        delta = abs(bar_val - ti_val)
        tol = sigma * np.sqrt(bar_err**2 + ti_err**2)
        return delta < tol
    
    def drift_slope(self, time_fractions, values):
        coeff = np.polyfit(time_fractions, values, deg=1)
        return coeff[0]