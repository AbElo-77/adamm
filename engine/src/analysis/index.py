from dataclasses import dataclass
from core.artifacts import Artifact

@dataclass(frozen=True)
class AnalysisResult(Artifact):
    analysis_type: str
    inputs: list[str]   
    metrics: dict
    diagnostics: dict  
    warnings: list[str]

    def to_dict(self):
        return {
            "analysis_type": self.analysis_type,
            "inputs": self.inputs,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "warnings": self.warnings
        }
