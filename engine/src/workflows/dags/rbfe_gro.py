from core.graphs import GROMACS_RBFE_DAG
from engine.src.workflows.nodes.gromacs.grompp import GromppNode
from engine.src.workflows.nodes.gromacs.mdrun import MdrunNode
from engine.src.workflows.nodes.gromacs.energy import EnergyExtractionNode
from engine.src.workflows.nodes.gromacs.lambda_window import LambdaWindowNode
from engine.src.workflows.nodes.gromacs.simulation_run import SimulationRunNode
from engine.src.workflows.nodes.gromacs.thermodynamics import ThermodynamicsNode

"""
This is a helpful function for building the DAG for RBFE with GROMACS. 

    inputs: GROMACSRunner, ProvenanceStore, GROMACSInputs, lambda_value
    outputs: DAG with grompp, thermodynamics, mdrun, energy, and lambda_window
"""
def build_rbfe_dag(runner, prov_store, inputs, lam):
    dag = GROMACS_RBFE_DAG()

    grompp = GromppNode(runner, prov_store, lam)
    thermo = ThermodynamicsNode(prov_store,  lam)
    mdrun = MdrunNode(runner, prov_store, lam)
    energy = EnergyExtractionNode(prov_store, lam)
    lw = LambdaWindowNode(prov_store, lam)

    dag.add_node(grompp)
    dag.add_node(thermo)
    dag.add_node(mdrun, parents=[grompp])
    dag.add_node(energy, parents=[mdrun])
    dag.add_node(lw, parents=[mdrun, energy])

    return dag
