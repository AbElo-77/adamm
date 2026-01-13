from __future__ import annotations

import subprocess
import tempfile
from typing import Dict, List

import numpy as np

from core.nodes import Node
from core.artifacts import EnergySeries, FileRef, ThermodynamicState
from provenance.metadata import ArtifactMetadata
from engines.gromacs.outputs import parse_xvg


class EnergyExtractionNode(Node):

    def __init__(
        self,
        prov_store,
        lambda_value: float,
        terms: Dict[str, str] | None = None,
    ):
        
        super().__init__(
            name=f"energy_extraction_lambda_{lambda_value}",
            prov_store=prov_store,
        )
        self.lambda_value = lambda_value

        self.terms = terms or {
            "potential": "Potential",
            "dhdl": "dH/dλ",
            "delta_h_forward": "ΔH λ→λ+1",
            "delta_h_backward": "ΔH λ→λ-1",
        }

    """
    inputs: Dict[str, object] containing FileRefs to the energy and trajectory files. 
    outputs: Dict[str, EnergySeries]; this contains the potential, dH/dl, dh_f, and dh_b.
    """
    def run(self, inputs: Dict[str, object]) -> Dict[str, EnergySeries]:
        
        edr_file: str = inputs["edr_file"]
        tpr_file: str = inputs["tpr_file"]
        thermo: ThermodynamicState = inputs["thermodynamic_state"]

        self._validate_inputs(edr_file, tpr_file, thermo)

        extracted = self._run_gmx_energy(edr_file)

        artifacts: Dict[str, EnergySeries] = []

        for logical_name, series in extracted.items():
            artifact = EnergySeries(
                values=series["values"],
                units="kJ/mol",
                sampling_interval=series["dt_ps"],
                source_file=FileRef(path=edr_file),
                metadata={
                    "term": logical_name,
                    "engine_term": series["engine_term"],
                    "lambda": self.lambda_value,
                    "thermodynamic_state": thermo.fingerprint(),
                },
            )
            self.prov.add_artifact(
                artifact,            
                metadata=ArtifactMetadata(
                created_by=self.name,
                engine="GROMACS"
                ),
                parents=list(inputs)
            )
            artifacts[logical_name] = artifact

        return artifacts

    def _validate_inputs(
        self,
        edr_file: str,
        tpr_file: str,
        thermo: ThermodynamicState,
    ):
        if thermo.lambda_value != self.lambda_value:
            raise ValueError(
                f"Lambda mismatch: node={self.lambda_value}, "
                f"ThermodynamicState={thermo.lambda_value}"
            )

        if thermo.source_tpr.path != tpr_file:
            raise ValueError(
                "ThermodynamicState does not correspond to provided TPR file"
            )

    def _run_gmx_energy(self, edr_file: str) -> Dict[str, dict]:

        results: Dict[str, dict] = {}

        with tempfile.TemporaryDirectory() as tmp:
            for logical_name, engine_label in self.terms.items():
                xvg_path = f"{tmp}/{logical_name}.xvg"

                self._invoke_gmx_energy(
                    edr_file=edr_file,
                    energy_label=engine_label,
                    output_xvg=xvg_path,
                )

                times, values = parse_xvg(xvg_path)
                dt_ps = self._infer_dt(times)

                results[logical_name] = {
                    "engine_term": engine_label,
                    "values": values,
                    "dt_ps": dt_ps,
                }

        return results

    def _invoke_gmx_energy(
        self,
        edr_file: str,
        energy_label: str,
        output_xvg: str,
    ):
        
        try:
            proc = subprocess.run(
                ["gmx", "energy", "-f", edr_file, "-o", output_xvg],
                input=f"{energy_label}\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"gmx energy failed for term '{energy_label}'"
            ) from e

    @staticmethod
    def _infer_dt(times: np.ndarray) -> float | None:
        if len(times) < 2:
            return None
        return float(times[1] - times[0])
