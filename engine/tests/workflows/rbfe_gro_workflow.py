import os 

from engine.src.engines.gromacs.inputs import GROMACSInputs, GROMACSExecution, MDConfig, SystemArtifacts
from engine.src.engines.gromacs.runner import GromacsRunner, build_solvated_system
from engine.src.provenance.store import ProvenanceStore
from engine.src.workflows.rbfe import RBFEWorkflow

execution =  GROMACSExecution()
runner = GromacsRunner(execution)

prov_store = ProvenanceStore(os.getcwd())

def main(): 
    sys = build_solvated_system(
        runner,
        pdb="engine/tests/workflows/inputs/protein_no_wat.pdb",
        forcefield="charmm36-jul2022",
        water_model="tip3p",
        ions_mdp="engine/tests/workflows/inputs/ions.mdp"
    )

    mdp = MDConfig(
        source_file="engine/tests/workflows/inputs/md.mdp",
        parameters={
            "integrator": "md",
            "nsteps": 5000,
            "dt": 0.002,
            "tcoupl": "V-rescale",
            "tc-grps": "System",
            "tau_t": 0.1,
            "ref_t": 300,
            "pcoupl": "Parrinello-Rahman",
            "ref_p": 1.0,
            "compressibility": 4.5e-5,
            "gen_vel": "yes",
            "gen_temp": 300,
            "gen_seed": 42
        }
    )

    inputs = GROMACSInputs(
        mdp=mdp,
        structure_file=sys.gro,
        topology_file=sys.top,
        index_file=None,
        restraint_file=None
    )

    workflow = RBFEWorkflow(runner, prov_store)
    outputs = workflow.run_simulation([inputs], [1.0])

    print(outputs)

if __name__ == "__main__":
    main()