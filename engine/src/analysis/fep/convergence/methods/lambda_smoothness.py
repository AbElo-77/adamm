import numpy as np
from typing import Any, Dict, List

from core.artifacts import SimulationRun, DiagnosticResult
from utils.components import Samples
from engine.src.analysis.fep.estimators import SampleSelection

"""
inputs: SimulationRun
outputs: DiagnosticResult

modify this script to give a range for convergence...not true / false
"""
class LambdaSmoothnessConvergence:

    def __init__(self, prov_store, burn_in: float = 0.1, ineff_criteria: str = "block"):
        self.prov = prov_store
        self.burn_in = burn_in
        self.ineff_criteria = ineff_criteria

    def conv_check(self, simulation_run: SimulationRun) -> DiagnosticResult:
        selector = SampleSelection(self.prov)
        samples: List[Samples] = []

        for lw in simulation_run.lambda_windows:
            lw_fp = self._as_fingerprint(lw)
            s = selector.run(lw_fp, burn_in=self.burn_in, ineff_criteria=self.ineff_criteria)
            if isinstance(s, list):
                s = s[0]
            samples.append(s)

        samples = sorted(samples, key=lambda s: s.thermodynamics.lambda_value)

        lambdas = np.array([s.thermodynamics.lambda_value for s in samples], dtype=float)
        means = np.array([np.mean(s.sampled_energy["dhdl"].values) for s in samples], dtype=float)

        variances = np.array([np.var(s.sampled_energy["dhdl"].values, ddof=1) for s in samples], dtype=float)
        stderrs = np.sqrt(variances / np.maximum(1.0, np.array([s.n_eff for s in samples], dtype=float)))

        curvature = self._curvature_metric(lambdas, means)
        endpoint = self._endpoint_inflation(lambdas, stderrs)
        quad = self._quadrature_sensitivity(lambdas, means)

        diag = DiagnosticResult(
            name="lambda_smoothness",
            value={
                "curve": [
                    {"lambda": float(lambdas[i]), "mean_dhdl": float(means[i]), "stderr_dhdl": float(stderrs[i])}
                    for i in range(len(lambdas))
                ],
                "curvature": curvature,
                "endpoint": endpoint,
                "quadrature_sensitivity": quad,
                "settings": {"burn_in": self.burn_in, "ineff_criteria": self.ineff_criteria},
            },
            units=None,
            context={"K": len(samples)},
            source_run=simulation_run,
        )

        self.prov.add_artifact(diag, parents=[simulation_run])
        return diag

    def _curvature_metric(self, lambdas: np.ndarray, means: np.ndarray) -> Dict[str, Any]:
        K = len(lambdas)
        if K < 3:
            return {"available": False, "notes": "Need >=3 lambdas for curvature."}

        curv = []
        for i in range(1, K - 1):
            l0, l1, l2 = lambdas[i - 1], lambdas[i], lambdas[i + 1]
            f0, f1, f2 = means[i - 1], means[i], means[i + 1]

            h1 = l1 - l0
            h2 = l2 - l1
            if h1 <= 0 or h2 <= 0:
                continue
            fpp = 2.0 * (((f2 - f1) / h2) - ((f1 - f0) / h1)) / (h1 + h2)
            curv.append((float(l1), float(fpp)))

        abs_vals = np.array([abs(c[1]) for c in curv], dtype=float) if curv else np.array([])
        return {
            "available": True,
            "points": [{"lambda": l, "second_derivative": v} for (l, v) in curv],
            "max_abs_second_derivative": float(abs_vals.max()) if abs_vals.size else None,
            "median_abs_second_derivative": float(np.median(abs_vals)) if abs_vals.size else None,
        }

    def _endpoint_inflation(self, lambdas: np.ndarray, stderrs: np.ndarray) -> Dict[str, Any]:
        if len(stderrs) < 3:
            return {"available": False, "notes": "Need >=3 lambdas for endpoint comparison."}

        mid = stderrs[1:-1]
        if mid.size == 0 or np.median(mid) == 0:
            return {"available": True, "ratio_endpoints_to_mid_median": None}

        ratio0 = float(stderrs[0] / np.median(mid))
        ratio1 = float(stderrs[-1] / np.median(mid))

        return {
            "available": True,
            "endpoint0_ratio": ratio0,
            "endpoint1_ratio": ratio1,
            "max_endpoint_ratio": float(max(ratio0, ratio1)),
        }

    def _quadrature_sensitivity(self, lambdas: np.ndarray, means: np.ndarray) -> Dict[str, Any]:
        trap = float(np.trapz(means, lambdas))

        K = len(lambdas)
        if K < 3 or (K % 2 == 0):
            return {"trap": trap, "simpson": None, "delta": None, "notes": "Simpson requires odd number of points."}

        dl = np.diff(lambdas)
        if np.any(dl <= 0):
            return {"trap": trap, "simpson": None, "delta": None, "notes": "Non-monotone lambdas."}

        r = (dl.max() - dl.min()) / max(1e-12, dl.mean())
        if r > 0.05:
            return {"trap": trap, "simpson": None, "delta": None, "notes": "Grid not uniform enough for Simpson."}

        h = float(dl.mean())
        f0, fN = means[0], means[-1]
        odd_sum = means[1:-1:2].sum()
        even_sum = means[2:-1:2].sum()
        simpson = float((h / 3.0) * (f0 + fN + 4.0 * odd_sum + 2.0 * even_sum))

        return {"trap": trap, "simpson": simpson, "delta": float(abs(trap - simpson))}

    def _as_fingerprint(self, lw_obj_or_fp: Any) -> str:
        if isinstance(lw_obj_or_fp, str):
            return lw_obj_or_fp
        if hasattr(lw_obj_or_fp, "fingerprint") and callable(getattr(lw_obj_or_fp, "fingerprint")):
            return lw_obj_or_fp.fingerprint()
        if hasattr(lw_obj_or_fp, "fingerprint"):
            return str(getattr(lw_obj_or_fp, "fingerprint"))
        raise TypeError(f"Cannot interpret lambda window reference: {type(lw_obj_or_fp)}")
