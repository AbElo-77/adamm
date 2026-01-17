## Augmented Dynamics and Analysis for Molecular Mechanics (ADAMM)

This is a toolkit for analyzing simulation results with regards to convergence, failure modes, and uncertainty.

# DEVELOPMENT NOTES:

- GROMACS Modulation and Convergence Pipeline (1/10 - 1/31)
    - Other
- API (2/1 - 2/28)
    - Define PROTOCOLS inside of APIs (MDStage + MDProtocol with preprocessing, stages, lambdas, and REMD). 
    - Should take in pdb files, the various needed mdp files, forefield choice, and water model. 
        - API
        - build_solvated_system()
        - [GROMACSInputs]{1-4}
        - DAG[Energy + LW]
        - Analysis Pipeline
- Docs and Examples (3/1 - 3/15)
    - Other

# FIXES: 

1. Resovle naming inconsistencies after full GROMACS RBFE pipeline development; for example, GromacsRunner -> GROMACSRunner. 
2. 