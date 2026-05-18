## Augmented Dynamics and Analysis for Molecular Mechanics (ADAMM)

This is a toolkit for analyzing simulation results with regard to convergence, failure modes, and uncertainty.

# DEVELOPMENT NOTES:

- Integrate DuckDB for provenance, **alongside** local cache.

- GROMACS Modulation and Convergence Pipeline (5/15 - 5/31)
    - Other
- API (6/1 - 6/20)
    - Define PROTOCOLS inside APIs (MDStage + MDProtocol with preprocessing, stages, lambdas, and REMD). 
    - Should take in pdb files, the various needed mdp files, forefield choice, and water model. 
        - API
        - build_solvated_system()
        - [GROMACSInputs]{1-4}
        - DAG[Energy + LW]
        - Analysis Pipeline
- Docs and Examples (7/1 - 7/15)
    - Other

# FIXES: 

1. Resolve naming inconsistencies after full GROMACS RBFE pipeline development; for example, GromacsRunner -> GROMACSRunner. 
2. Update ReducedPotentialBuilder to output state_of_column and n_k values. 
3. After debugging, set capture_output and text to **True**
