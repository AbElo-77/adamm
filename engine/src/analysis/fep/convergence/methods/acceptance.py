import numpy as np
from typing import Any, Dict, List

from core.artifacts import SimulationRun, DiagnosticResult

"""
inputs: SimulationRun
outputs: DiagnosticResult

modify this script to give a range for convergence...not true / false
"""
class ReplicaAcceptanceConvergence:

    def __init__(self, prov_store):
        self.prov = prov_store

    def conv_check(self, simulation_run: SimulationRun) -> DiagnosticResult:
        rows: List[Dict[str, Any]] = []
        acc_values = []

        lws = list(simulation_run.lambda_windows)
        lws = sorted(lws, key=lambda lw: lw.lambda_value)

        for lw in lws:
            lam = float(lw.lambda_value)
            acc = lw.exchange_acceptance
            rows.append({"lambda": lam, "exchange_acceptance": None if acc is None else float(acc)})
            if acc is not None:
                acc_values.append(float(acc))

        acc_arr = np.array(acc_values, dtype=float) if acc_values else np.array([])

        summary = {
            "n_windows": len(lws),
            "n_with_acceptance": int(acc_arr.size),
            "fraction_missing": float(1.0 - (acc_arr.size / max(1, len(lws)))),
            "min_acceptance": float(acc_arr.min()) if acc_arr.size else None,
            "median_acceptance": float(np.median(acc_arr)) if acc_arr.size else None,
        }

        diag = DiagnosticResult(
            name="replica_acceptance",
            value={"per_window": rows, "summary": summary},
            units=None,
            context={},
            source_run=simulation_run,
        )

        self.prov.add_artifact(diag, parents=[simulation_run])
        return diag
