import subprocess
import re
import math
from core.nodes import Node
from core.artifacts import ThermodynamicState, FileRef

K_BOLTZMANN = 0.008314462618


class ThermodynamicsNode(Node):
    def __init__(self, prov_store, lambda_value: float):
        super().__init__(name=f"thermodynamics_lambda_{lambda_value}",
                         prov_store=prov_store)
        self.lambda_value = lambda_value

    def run(self, inputs):
        tpr_file = f"lambda_{self.lambda_value}.tpr"

        thermodynamics = self.parse_tpr(tpr_file)

        self.prov.add_artifact(
            thermodynamics,
            context={"lambda": self.lambda_value}
        )

        return thermodynamics

    def parse_tpr(self, source_file: str) -> ThermodynamicState:

        dump_text = self._dump_tpr(source_file)

        temperature = self._parse_temperature(dump_text)
        lambda_value = self._parse_lambda(dump_text)
        ensemble = self._parse_ensemble(dump_text)
        softcore = self._parse_softcore(dump_text)
        neighbors = self._parse_lambda_neighbors(dump_text)
        constraints = self._parse_constraints(dump_text)

        beta = 1.0 / (K_BOLTZMANN * temperature)\
        
        # need to change so that this is stored in the prov instead of returned
        return ThermodynamicState(
            temperature=temperature,
            beta=beta,
            lambda_value=lambda_value,
            ensemble=ensemble,
            softcore=softcore,
            calc_lambda_neighbors=neighbors,
            constraints=constraints,
            engine="gromacs",
            engine_version=self._parse_gromacs_version(dump_text),
            source_tpr=FileRef(path=source_file)
        )

    def _dump_tpr(self, tpr_file: str) -> str:
        try:
            result = subprocess.run(
                ["gmx", "dump", "-s", tpr_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to dump TPR {tpr_file}") from e

    def _parse_temperature(self, text: str) -> float:
        match = re.search(r"ref_t\s+=\s+([0-9.]+)", text)
        if not match:
            raise ValueError("Temperature not found in TPR")
        return float(match.group(1))

    def _parse_lambda(self, text: str) -> float:
        match = re.search(r"init_lambda_state\s+=\s+(\d+)", text)
        if not match:
            raise ValueError("Lambda state index not found")

        lambda_state = int(match.group(1))

        lambdas = re.search(r"fep-lambdas\s+=\s+\{([^\}]+)\}", text)
        if not lambdas:
            raise ValueError("Lambda schedule not found")

        lambda_values = [float(x) for x in lambdas.group(1).split()]
        return lambda_values[lambda_state]

    def _parse_lambda_neighbors(self, text: str) -> int:
        match = re.search(r"calc_lambda_neighbors\s+=\s+(\d+)", text)
        return int(match.group(1)) if match else 0

    def _parse_softcore(self, text: str) -> dict:
        params = {}
        for key in ["sc_alpha", "sc_power", "sc_sigma"]:
            match = re.search(fr"{key}\s+=\s+([0-9.eE+-]+)", text)
            if match:
                params[key] = float(match.group(1))
        return params

    def _parse_constraints(self, text: str) -> str:
        match = re.search(r"constraints\s+=\s+(\w+)", text)
        return match.group(1) if match else "none"

    def _parse_ensemble(self, text: str) -> str:
        if "parrinello-rahman" in text.lower():
            return "NPT"
        if "nose-hoover" in text.lower():
            return "NVT"
        return "unknown"

    def _parse_gromacs_version(self, text: str) -> str:
        match = re.search(r"GROMACS version:\s+([^\n]+)", text)
        return match.group(1).strip() if match else "unknown"
