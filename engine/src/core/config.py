from dataclasses import dataclass
from typing import Optional

@dataclass
class ADAMMConfig:
    prov_store_path: str = "./prov_store"
    gmx_binary: str = "gmx"
    n_threads: Optional[int] = None
    default_mdp: Optional[str] = None
    log_level: str = "INFO"
