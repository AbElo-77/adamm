import numpy as np
from core.artifacts import EnergySeries, Trajectory, LogFile

def parse_xvg(
    filepath: str,  
    units: str
) -> EnergySeries: 
    data = []
    with open(filepath) as f: 
        for line in f: 
            if line.startswith(('#','@')):
                continue
            _, val = line.split()[:2]
            data.append(val)
    
    return EnergySeries(
        values=np.ndarray(data), 
        units=units, 
        sampling_interval=None, 
        source_file=filepath
    )

def read_trajectory(
    filepath: str, 
    atom_count: int, 
    traj_type: str = '.gro'
) -> Trajectory: 
    return Trajectory(
        file_ref=filepath, 
        atom_count=atom_count,
        traj_type=traj_type
    )

def read_edr(
    filepath: str
) -> LogFile: 
    with open(filepath) as f: 
        return f.read()

def read_log(
    filepath: str
) -> LogFile: 
    with open(filepath) as f: 
        return f.read()