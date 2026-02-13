from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import numpy as np

from engine.src.engines.gromacs.inputs import GROMACSInputs

"""
This is the general wrapper class for any object producted during the ADAMM lifecylce. 
"""
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

"""
This artifact stores the path to a file of interest, including trajectories and energies. 
"""   
@dataclass(frozen=True)
class FileRef(Artifact):
    path: str
    checksum: Optional[str] = None 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "checksum": self.checksum,
        }

"""
This artifact stores trajectory information. 

    file_ref -> path to trajectory
    traj_type -> type of trajectory file
    lambda_value -> the lambda
    atom_count -> number of atoms
    frame_count -> number of frames 
    time_step_ps -> timestep in picoseconds
"""
@dataclass(frozen=True)
class Trajectory(Artifact):
    file_ref: FileRef
    lambda_value: float
    atom_count: int
    traj_type: str = '.trr'
    frame_count: Optional[int] = None
    time_step_ps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_ref": self.file_ref.to_dict(),
            "traj_type": self.traj_type,
            "lambda_value": self.lambda_value,
            "atom_count": self.atom_count,
            "frame_count": self.frame_count,
            "time_step_ps": self.time_step_ps,
        }

"""
this artifact stores energy information from the *.edr file. 

    values -> the values for each frame of the specific energy data. 
    units -> the units for the specific energy. 
    sampling_interval -> how often the energy value is sampled. 
    source_file -> path to *.edr
    metadata -> (term, lambda, thermodynamic_state)
"""   
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

"""
This artifact stores thermodynamic information from the *.tpr file

    temperature -> initial state temperature
    beta -> 1/(k_boltzmann * temperature)
    lambda_value -> the lambda
    ensemble -> the type of ensemble (NVT or NPT)
    softcore -> sc parameters for free energy calculations
    constraints -> constraint parameters
    source_tpr -> path to *.tpr
"""    
@dataclass(frozen=True)
class ThermodynamicState(Artifact):
    temperature: float
    beta: float
    lambda_value: float
    ensemble: str
    softcore: dict
    constraints: str
    source_tpr: FileRef

    def to_dict(self):
        return {
            "temperature": self.temperature,
            "beta": self.beta,
            "lambda_value": self.lambda_value,
            "ensemble": self.ensemble,
            "softcore": self.softcore,
            "constraints": self.constraints,
            "source_tpr": self.source_tpr.to_dict(),
        }

"""
This artifact stores the trajectory, energy, and thermodynamics of a single lambda.

    lambda_value -> the lambda
    thermodynamics -> the ThermodynamicState
    energy_series -> the array of EnergySeries
    trajectory -> the trajectory file (.trr / .xtc)
    exchange_acceptance -> exchange rate if performing REMD or alchemical enhanced sampling
"""
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

"""
This artifact stores the protein name and the path to its .pdb
"""
@dataclass(frozen=True)
class Protein(Artifact):
    name: str
    file_ref: FileRef

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_ref": self.file_ref.to_dict(),
        }

"""
This artifact stores the ligand name and the path to its .pdb, along with its net_charge. 
"""
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

"""
This artifact stores the mapping between the two ligands when performing RBFE. 

    ligand_a -> first ligand
    ligand_b -> second ligand 
    atom_map -> the atom_map used for alchemical transformations
    method -> technique used to generate atom_map***
    source_file -> path to mapping file
"""
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

"""
This artifact stores the entire RBFE simulation ran. 

    engine -> the engine used to run the simulations (e.g. GROMACS)
    engine_version -> the version of the engine
    input_files -> the GROMACSInputs
    lambda_windows -> the set of LambdaWindows in the simulation
    log_file -> path to the log file
"""   
@dataclass(frozen=True)
class SimulationRun(Artifact):
    engine: str
    engine_version: str
    input_files: GROMACSInputs
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

"""
This artifact stores a diagnostic result.
"""   
@dataclass(frozen=True)
class DiagnosticResult(Artifact):
    name: str
    values: List[Any]
    confidences: List[Any]
    units: Optional[str]
    source_run: SimulationRun
    context: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.values,
            "confidences": self.confidences, 
            "units": self.units,
            "context": self.context,
            "source_run": self.source_run.fingerprint(),
        }

"""
This artifact stores the ultimate decision record.
"""
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
