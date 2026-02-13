import numpy as np

from engine.src.utils.components import Samples, FreeEnergyEstimate
from engine.src.analysis.fep.estimators import SampleSelection, BAR, TI, ReducedPotentialBuilder

from engine.src.core.artifacts import SimulationRun, DiagnosticResult, EnergySeries
from engine.src.utils.components import Samples, ReducedPotentialDataset
from typing import List, Dict, Tuple, Any, Optional

"""
inputs: SimulationRun
outputs: DiagnosticResult

modify this script to give a range for convergence...not true / false
"""
class OverlapConvergence:

    def __init__(
        self,
        prov_store,
        burn_in: float = 0.1,
        ineff_criteria: str = "block",
        min_pair_samples: int = 50,
        hist_bins: int = 50,
    ):
        self.prov = prov_store
        self.burn_in = burn_in
        self.ineff_criteria = ineff_criteria
        self.min_pair_samples = min_pair_samples
        self.hist_bins = hist_bins

    def conv_check(self, simulation_run: SimulationRun) -> DiagnosticResult:

        selector = SampleSelection(self.prov)
        sample_list: List[Samples] = []

        for lw in simulation_run.lambda_windows:
            lw_fp = self._as_fingerprint(lw)
            s = selector.run(lw_fp, burn_in=self.burn_in, ineff_criteria=self.ineff_criteria)
            if isinstance(s, list):
                s = s[0]
            sample_list.append(s)

        sample_list = sorted(sample_list, key=lambda s: s.thermodynamics.lambda_value)

        rp_builder = ReducedPotentialBuilder(self.prov)
        ds_list = rp_builder.run(sample_list, trunc_policy={"purpose": "overlap"})
        dataset: ReducedPotentialDataset = ds_list[0] if isinstance(ds_list, list) else ds_list

        edge_metrics, summary = self._compute_overlap_metrics(dataset, sample_list)

        diag = DiagnosticResult(
            name="overlap_metrics",
            value={
                "edges": edge_metrics,
                "summary": summary,
                "settings": {
                    "burn_in": self.burn_in,
                    "ineff_criteria": self.ineff_criteria,
                    "min_pair_samples": self.min_pair_samples,
                    "hist_bins": self.hist_bins,
                },
            },
            units=None,
            context={
                "K": int(dataset.u_kn.shape[0]),
                "N_total": int(dataset.u_kn.shape[1]),
            },
            source_run=simulation_run,
        )

        self.prov.add_artifact(diag, parents=[simulation_run, dataset])
        return diag

    def _compute_overlap_metrics(
        self,
        dataset: ReducedPotentialDataset,
        samples_sorted: List[Samples],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        u = np.asarray(dataset.u_kn, dtype=float)
        owner = np.asarray(dataset.state_of_column, dtype=int)
        n_k = np.asarray(dataset.n_k, dtype=int)

        K, N = u.shape
        if owner.shape[0] != N:
            raise ValueError(f"state_of_column length {owner.shape[0]} != N_total {N}")
        if n_k.shape[0] != K:
            raise ValueError(f"n_k length {n_k.shape[0]} != K {K}")

        lambdas = np.array([s.thermodynamics.lambda_value for s in samples_sorted], dtype=float)

        edges: List[Dict[str, Any]] = []

        worst_by_min_ess_frac = None
        worst_by_max_z = None 

        for i in range(K - 1):
            j = i + 1

            du_i, du_j, n_i_used, n_j_used = self._pair_du(u, owner, i, j)

            valid = (du_i.size >= self.min_pair_samples) and (du_j.size >= self.min_pair_samples)

            metrics: Dict[str, Any] = {
                "edge": {"i": int(i), "j": int(j), "lambda_i": float(lambdas[i]), "lambda_j": float(lambdas[j])},
                "counts": {
                    "n_i_total": int(n_k[i]),
                    "n_j_total": int(n_k[j]),
                    "n_i_used": int(n_i_used),
                    "n_j_used": int(n_j_used),
                },
                "valid": bool(valid),
            }

            if not valid:
                metrics.update({
                    "ess": None,
                    "ess_fraction": None,
                    "separation_z": None,
                    "work_stats": None,
                    "hist_overlap": None,
                    "notes": "Insufficient neighbor-evaluated samples for robust overlap metrics.",
                })
                edges.append(metrics)
                continue

            ess_ij = self._ess_from_du(du_i)
            ess_ji = self._ess_from_du(du_j)

            ess_frac_ij = float(ess_ij / max(1, du_i.size))
            ess_frac_ji = float(ess_ji / max(1, du_j.size))

            z = self._separation_z(du_i, du_j)
            ovl = self._hist_overlap(du_i, du_j, bins=self.hist_bins)

            metrics.update({
                "ess": {
                    "ess_i_to_j": float(ess_ij),
                    "ess_j_to_i": float(ess_ji),
                },
                "ess_fraction": {
                    "ess_frac_i_to_j": ess_frac_ij,
                    "ess_frac_j_to_i": ess_frac_ji,
                    "ess_frac_min": float(min(ess_frac_ij, ess_frac_ji)),
                },
                "separation_z": float(z),
                "work_stats": {
                    "du_i_to_j": {"mean": float(np.mean(du_i)), "std": float(np.std(du_i, ddof=1))},
                    "du_j_to_i": {"mean": float(np.mean(du_j)), "std": float(np.std(du_j, ddof=1))},
                },
                "hist_overlap": float(ovl),
            })

            ess_min = min(ess_frac_ij, ess_frac_ji)
            if worst_by_min_ess_frac is None or ess_min < worst_by_min_ess_frac[0]:
                worst_by_min_ess_frac = (ess_min, i)

            if worst_by_max_z is None or z > worst_by_max_z[0]:
                worst_by_max_z = (z, i)

            edges.append(metrics)

        summary = {
            "worst_edge_by_min_ess_fraction": (
                None if worst_by_min_ess_frac is None
                else {"edge_i": int(worst_by_min_ess_frac[1]), "edge_j": int(worst_by_min_ess_frac[1] + 1),
                      "min_ess_fraction": float(worst_by_min_ess_frac[0])}
            ),
            "worst_edge_by_max_separation_z": (
                None if worst_by_max_z is None
                else {"edge_i": int(worst_by_max_z[1]), "edge_j": int(worst_by_max_z[1] + 1),
                      "separation_z": float(worst_by_max_z[0])}
            ),
            "n_edges": int(K - 1),
            "n_valid_edges": int(sum(1 for e in edges if e.get("valid"))),
        }

        return edges, summary

    def _pair_du(
        self,
        u_kn: np.ndarray,
        state_of_column: np.ndarray,
        i: int,
        j: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        cols_i = np.where(state_of_column == i)[0]
        cols_j = np.where(state_of_column == j)[0]

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
        w_sum = np.sum(w)
        if w_sum == 0.0 or not np.isfinite(w_sum):
            return 0.0
        w /= w_sum
        return float(1.0 / np.sum(w ** 2))

    def _separation_z(self, du_i: np.ndarray, du_j: np.ndarray) -> float:
        du_i = np.asarray(du_i, dtype=float)
        du_j = np.asarray(du_j, dtype=float)
        mu_i, mu_j = np.mean(du_i), np.mean(du_j)
        var = np.var(du_i, ddof=1) + np.var(du_j, ddof=1)
        return float(abs(mu_i - mu_j) / np.sqrt(var + 1e-12))

    def _hist_overlap(self, du_i: np.ndarray, du_j: np.ndarray, bins: int = 50) -> float:
        x = np.asarray(du_i, dtype=float)
        y = np.asarray(du_j, dtype=float)

        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return 0.0

        p, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
        q, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)

        widths = np.diff(edges)
        p_mass = p * widths
        q_mass = q * widths

        return float(np.sum(np.minimum(p_mass, q_mass)))

    def _as_fingerprint(self, lw_obj_or_fp: Any) -> str:
        if isinstance(lw_obj_or_fp, str):
            return lw_obj_or_fp
        if hasattr(lw_obj_or_fp, "fingerprint") and callable(getattr(lw_obj_or_fp, "fingerprint")):
            return lw_obj_or_fp.fingerprint()
        if hasattr(lw_obj_or_fp, "fingerprint"):
            return str(getattr(lw_obj_or_fp, "fingerprint"))
        raise TypeError(f"Cannot interpret lambda window reference: {type(lw_obj_or_fp)}")