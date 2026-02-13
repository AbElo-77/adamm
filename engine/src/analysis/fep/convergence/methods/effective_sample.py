import numpy as np
from typing import Any, Dict, List, Tuple

from core.artifacts import SimulationRun, DiagnosticResult
from utils.components import Samples, ReducedPotentialDataset
from engine.src.analysis.fep.estimators import SampleSelection, ReducedPotentialBuilder 

"""
inputs: SimulationRun
outputs: DiagnosticResult

modify this script to give a range for convergence...not true / false
"""
class EffectiveSampleSizeConvergence:

    def __init__(
        self,
        prov_store,
        burn_in: float = 0.1,
        ineff_criteria: str = "block",
        min_pair_samples: int = 50,
    ):
        self.prov = prov_store
        self.burn_in = burn_in
        self.ineff_criteria = ineff_criteria
        self.min_pair_samples = min_pair_samples

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

        rp_builder = ReducedPotentialBuilder(self.prov)
        ds = rp_builder.run(samples, trunc_policy={"purpose": "ess"})[0]

        window_ess = self._window_ess(samples)
        edge_ess = self._edge_reweighting_ess(ds, samples)

        summary = {
            "min_window_n_eff": float(min(w["n_eff"] for w in window_ess)) if window_ess else None,
            "median_window_n_eff": float(np.median([w["n_eff"] for w in window_ess])) if window_ess else None,
            "worst_edge_min_ess_frac": self._worst_edge_min_ess_frac(edge_ess),
        }

        diag = DiagnosticResult(
            name="effective_sample_size",
            value={
                "window_ess": window_ess,
                "edge_reweighting_ess": edge_ess,
                "summary": summary,
                "settings": {
                    "burn_in": self.burn_in,
                    "ineff_criteria": self.ineff_criteria,
                    "min_pair_samples": self.min_pair_samples,
                },
            },
            units=None,
            context={"K": len(samples)},
            source_run=simulation_run,
        )

        self.prov.add_artifact(diag, parents=[simulation_run, ds])
        return diag

    def _window_ess(self, samples: List[Samples]) -> List[Dict[str, Any]]:
        out = []
        for s in samples:
            lam = float(s.thermodynamics.lambda_value)
            raw_n = int(len(s.sampled_energy["potential"].values))
            out.append({
                "lambda": lam,
                "n_raw": raw_n,
                "n_eff": float(s.n_eff),
                "eff_fraction": float(s.n_eff / max(1, raw_n)),
            })
        return out

    def _edge_reweighting_ess(
        self,
        ds: ReducedPotentialDataset,
        samples_sorted: List[Samples],
    ) -> List[Dict[str, Any]]:
        u = np.asarray(ds.u_kn, dtype=float)
        owner = np.asarray(ds.state_of_column, dtype=int)
        K, N = u.shape

        lambdas = np.array([s.thermodynamics.lambda_value for s in samples_sorted], dtype=float)

        edges = []
        for i in range(K - 1):
            j = i + 1
            du_i, du_j, n_i_used, n_j_used = self._pair_du(u, owner, i, j)

            valid = (du_i.size >= self.min_pair_samples) and (du_j.size >= self.min_pair_samples)

            if not valid:
                edges.append({
                    "edge": {"i": i, "j": j, "lambda_i": float(lambdas[i]), "lambda_j": float(lambdas[j])},
                    "valid": False,
                    "counts": {"n_i_used": int(n_i_used), "n_j_used": int(n_j_used)},
                    "ess": None,
                    "ess_fraction": None,
                    "notes": "Insufficient neighbor-evaluated samples for ESS.",
                })
                continue

            ess_ij = self._ess_from_du(du_i)
            ess_ji = self._ess_from_du(du_j)

            edges.append({
                "edge": {"i": i, "j": j, "lambda_i": float(lambdas[i]), "lambda_j": float(lambdas[j])},
                "valid": True,
                "counts": {"n_i_used": int(du_i.size), "n_j_used": int(du_j.size)},
                "ess": {"ess_i_to_j": float(ess_ij), "ess_j_to_i": float(ess_ji)},
                "ess_fraction": {
                    "ess_frac_i_to_j": float(ess_ij / max(1, du_i.size)),
                    "ess_frac_j_to_i": float(ess_ji / max(1, du_j.size)),
                    "min_ess_frac": float(min(ess_ij / max(1, du_i.size), ess_ji / max(1, du_j.size))),
                },
            })

        return edges

    def _worst_edge_min_ess_frac(self, edge_ess: List[Dict[str, Any]]) -> Any:
        best = None
        for e in edge_ess:
            if not e.get("valid"):
                continue
            m = e["ess_fraction"]["min_ess_frac"]
            if best is None or m < best["min_ess_frac"]:
                best = {
                    "edge": e["edge"],
                    "min_ess_frac": float(m),
                }
        return best

    def _pair_du(self, u_kn: np.ndarray, owner: np.ndarray, i: int, j: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        cols_i = np.where(owner == i)[0]
        cols_j = np.where(owner == j)[0]

        mask_i = (~np.isnan(u_kn[i, cols_i])) & (~np.isnan(u_kn[j, cols_i]))
        mask_j = (~np.isnan(u_kn[i, cols_j])) & (~np.isnan(u_kn[j, cols_j]))

        cols_i = cols_i[mask_i]
        cols_j = cols_j[mask_j]

        du_i = u_kn[j, cols_i] - u_kn[i, cols_i]
        du_j = u_kn[i, cols_j] - u_kn[j, cols_j]

        return du_i, du_j, int(cols_i.size), int(cols_j.size)

    def _ess_from_du(self, du: np.ndarray) -> float:
        du = np.asarray(du, dtype=float)
        a = -du
        a = a - np.max(a)
        w = np.exp(a)
        s = np.sum(w)
        if s <= 0 or not np.isfinite(s):
            return 0.0
        w /= s
        return float(1.0 / np.sum(w**2))

    def _as_fingerprint(self, lw_obj_or_fp: Any) -> str:
        if isinstance(lw_obj_or_fp, str):
            return lw_obj_or_fp
        if hasattr(lw_obj_or_fp, "fingerprint") and callable(getattr(lw_obj_or_fp, "fingerprint")):
            return lw_obj_or_fp.fingerprint()
        if hasattr(lw_obj_or_fp, "fingerprint"):
            return str(getattr(lw_obj_or_fp, "fingerprint"))
        raise TypeError(f"Cannot interpret lambda window reference: {type(lw_obj_or_fp)}")