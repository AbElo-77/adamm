from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import numpy as np

@dataclass(frozen=True)
class Artifact:
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            default=str
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("")
    
@dataclass(frozen=True)
class FileRef(Artifact):
    path: str
    checksum: Optional[str] = None 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "checksum": self.checksum,
        }

@dataclass(frozen=True)
class Trajectory(Artifact):
    file_ref: FileRef
    atom_count: int
    traj_type: str = '.gro'
    frame_count: Optional[int] = None
    time_step_ps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_ref": self.file_ref.to_dict(),
            "atom_count": self.atom_count,
            "frame_count": self.frame_count,
            "time_step_ps": self.time_step_ps,
        }
    
@dataclass(frozen=True)
class EnergySeries(Artifact):
    values: np.ndarray
    units: str
    sampling_interval: Optional[float]
    source_file: FileRef
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "units": self.units,
            "sampling_interval_ps": self.sampling_interval_ps,
            "source_file": self.source_file.to_dict(),
            "metadata": self.metadata,
        }
    
@dataclass(frozen=True)
class ThermodynamicState(Artifact):
    temperature: float
    beta: float
    lambda_value: float
    ensemble: str
    softcore: dict
    calc_lambda_neighbors: int
    constraints: str
    engine: str
    engine_version: str
    source_tpr: FileRef

    def to_dict(self):
        return {
            "temperature": self.temperature,
            "beta": self.beta,
            "lambda_value": self.lambda_value,
            "ensemble": self.ensemble,
            "softcore": self.softcore,
            "calc_lambda_neighbors": self.calc_lambda_neighbors,
            "constraints": self.constraints,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "source_tpr": self.source_tpr.to_dict(),
        }


@dataclass(frozen=True)
class LambdaWindow(Artifact):
    lambda_value: float
    thermodynamics: ThermodynamicState
    energy_series: Dict[str, EnergySeries]
    trajectory: Optional[Trajectory]
    exchange_acceptance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambda_value": self.lambda_value,
            "thermodynamic_state": self.thermodynamics,
            "energy_series": {
                k: v.to_dict() for k, v in self.energy_series.items()
            },
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "exchange_acceptance": self.exchange_acceptance,
        }

@dataclass(frozen=True)
class Protein(Artifact):
    name: str
    file_ref: FileRef

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_ref": self.file_ref.to_dict(),
        }

@dataclass(frozen=True)
class Ligand(Artifact):
    name: str
    file_ref: FileRef
    net_charge: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_ref": self.file_ref.to_dict(),
            "net_charge": self.net_charge,
        }

@dataclass(frozen=True)
class Mapping(Artifact):
    ligand_a: Ligand
    ligand_b: Ligand
    atom_map: Dict[int, int]
    method: str
    source_file: Optional[FileRef] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_a": self.ligand_a.to_dict(),
            "ligand_b": self.ligand_b.to_dict(),
            "atom_map": self.atom_map,
            "method": self.method,
            "source_file": self.source_file.to_dict() if self.source_file else None,
        }
    
@dataclass(frozen=True)
class ReducedPotentialDataset(Artifact):
    u_kn: np.ndarray
    sample_state_indices: np.ndarray
    state_fingerprints: List[str]
    source_runs: List[str]
    truncation_policy: Dict[str, Any]

    def to_dict(self):
        return {
            "u_kn": self.u_kn.tolist(),
            "sample_state_indices": self.sample_state_indices.tolist(),
            "state_fingerprints": self.state_fingerprints,
            "source_runs": self.source_runs,
            "truncation_policy": self.truncation_policy,
        }
    
@dataclass(frozen=True)
class SampleSelection(Artifact):
    total_samples: int
    discarded_equilibration: int
    thinning_interval: Optional[int]
    selection_method: str
    rationale: str

    def to_dict(self):
        return {
            "total_samples": self.total_samples,
            "discarded_equilibration": self.discarded_equilibration,
            "thinning_interval": self.thinning_interval,
            "selection_method": self.selection_method,
            "rationale": self.rationale,
        }

@dataclass(frozen=True)
class EstimatorRun(Artifact):
    method: str 
    input_dataset: ReducedPotentialDataset
    sample_selection: SampleSelection
    parameters: Dict[str, Any]
    solver_settings: Dict[str, Any]
    converged: bool
    iterations: Optional[int]
    residual: Optional[float]

    def to_dict(self):
        return {
            "method": self.method,
            "input_dataset": self.input_dataset.fingerprint(),
            "sample_selection": self.sample_selection.fingerprint(),
            "parameters": self.parameters,
            "solver_settings": self.solver_settings,
            "converged": self.converged,
            "iterations": self.iterations,
            "residual": self.residual,
        }
    
@dataclass(frozen=True)
class FreeEnergyEstimate(Artifact):
    value: float
    uncertainty: float
    units: str
    reference_states: Tuple[str, str]
    estimator_run: EstimatorRun

    def to_dict(self):
        return {
            "value": self.value,
            "uncertainty": self.uncertainty,
            "units": self.units,
            "reference_states": self.reference_states,
            "estimator_run": self.estimator_run.fingerprint(),
        }
    
@dataclass(frozen=True)
class SimulationRun(Artifact):
    engine: str
    engine_version: str
    input_files: Dict[str, FileRef]
    lambda_windows: list[LambdaWindow]
    log_file: Optional[FileRef]
    start_time: Optional[str]
    end_time: Optional[str]
    wall_clock_seconds: Optional[float]

    def to_dict(self):
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "input_files": {k: v.to_dict() for k, v in self.input_files.items()},
            "lambda_windows": [lw.to_dict() for lw in self.lambda_windows],
            "log_file": self.log_file.to_dict() if self.log_file else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "wall_clock_seconds": self.wall_clock_seconds
        }
    
@dataclass(frozen=True)
class DiagnosticResult(Artifact):
    name: str
    value: Any
    units: Optional[str]
    context: Dict[str, Any]
    source_run: SimulationRun

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "units": self.units,
            "context": self.context,
            "source_run": self.source_run.fingerprint(),
        }

@dataclass(frozen=True)
class DecisionRecord(Artifact):
    decision_type: str 
    confidence: Optional[float]
    diagnostics_used: List[DiagnosticResult]
    rationale: str
    overridden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "confidence": self.confidence,
            "diagnostics_used": [
                d.fingerprint() for d in self.diagnostics_used
            ],
            "rationale": self.rationale,
            "overridden": self.overridden,
        }
