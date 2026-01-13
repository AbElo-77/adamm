from core.graphs import DAG
from engine.src.workflows.nodes.gromacs.grompp import GromppNode
from engine.src.workflows.nodes.gromacs.mdrun import MdrunNode
from engine.src.workflows.nodes.gromacs.energy import EnergyExtractionNode
from engine.src.workflows.nodes.gromacs.lambda_window import LambdaWindowNode
from engine.src.workflows.nodes.gromacs.simulation_run import SimulationRunNode
from engine.src.workflows.nodes.gromacs.thermodynamics import ThermodynamicsNode

def build_rbfe_dag(runner, prov_store, input_artifacts, lambdas):
    dag = DAG()
    lambda_nodes = []

    for lam, _ in zip(lambdas, input_artifacts):
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

        lambda_nodes.append(lw)

    sim_node = SimulationRunNode(
        prov_store,
        engine="GROMACS",
        engine_version=runner.detect_version
    )
    dag.add_node(sim_node, parents=lambda_nodes)
    return dag
