## Augmented Dynamics and Analysis for Molecular Mechanics (ADAMM)

This is a toolkit for analyzing simulation results with regards to convergence, failure modes, and uncertainty.

# DEVELOPMENT NOTES:

- GROMACS Modulation and Convergence Pipeline (3/10 - 3/31)
    - Other
- API (4/1 - 4/20)
    - Define PROTOCOLS inside of APIs (MDStage + MDProtocol with preprocessing, stages, lambdas, and REMD). 
    - Should take in pdb files, the various needed mdp files, forefield choice, and water model. 
        - API
        - build_solvated_system()
        - [GROMACSInputs]{1-4}
        - DAG[Energy + LW]
        - Analysis Pipeline
- Docs and Examples (5/1 - 5/15)
    - Other

# FIXES: 

1. Resovle naming inconsistencies after full GROMACS RBFE pipeline development; for example, GromacsRunner -> GROMACSRunner. 
2. Update ReducedPotentialBuilder to output state_of_column and n_k values. 
3. After debugging, set capture_output and text to **True**