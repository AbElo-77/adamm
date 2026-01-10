from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class ArtifactMetadata:
    created_by: Optional[str] = None
    created_at: Optional[str] = None 
    engine: Optional[str] = None
    engine_version: Optional[str] = None
    extra: Optional[Dict[str, str]] = None

    def to_dict(self):
        return {
            "created_by": self.created_by,
            "created_at": self.created_at,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "extra": self.extra or {},
        }
