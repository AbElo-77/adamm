class NodeExecutionError(Exception):
    """Raised when a Node fails during execution."""
    pass

class DAGExecutionError(Exception):
    """Raised when DAG cannot be executed (cycle, missing dependencies)."""
    pass

class ArtifactNotFoundError(Exception):
    """Raised when an artifact is missing in ProvenanceStore."""
    pass

class ProvenanceError(Exception):
    """Raised when provenance cannot be recorded or read."""
    pass
