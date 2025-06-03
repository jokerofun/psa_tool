from gboml import GbomlGraph
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.consumer_mock_data_gen import generate_consumption_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data
from examples.helpers.microgrid_setup import MicrogridSetup
import examples.helpers.gboml_build_domain as gboml_domain
from examples.helpers.file_writer import write

def run(setup:MicrogridSetup, microgrid_file_path="examples/GBOML/microgrid.txt"):
    gboml_setup = MicrogridSetup()
    gboml_setup = setup
    gboml_setup.T = setup.T + 1
    gboml_domain.build_microgrid(setup=gboml_setup, file_path=microgrid_file_path)

    home_predicted_consumption = {}
    for _ in range(gboml_setup.no_homes):
        home_predicted_consumption = predict_consumer_data(parameters={"hours": gboml_setup.T, "model_name":"consumer_model", "factor": 1})["gen_consumption"]["consumption_kWh"].values
    school_predicted_consumption = predict_consumer_data(parameters={"hours": gboml_setup.T, "model_name":"consumer_model", "factor": 100})["gen_consumption"]["consumption_kWh"].values
    demand =  home_predicted_consumption * gboml_setup.no_homes + school_predicted_consumption
    wind_data = get_wind_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    
    solar_prod = {}
    for _ in range(gboml_setup.no_solar_panels):
        solar_data = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
        solar_prod = generate_solar_panel_data(solar_data, parameters={"rated_power": gboml_setup.solar_capacity_home,})

    wind_prod = generate_wind_turbine_data(wind_data, parameters={"rated_power": gboml_setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})

    solar_data = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    big_solar_prod = generate_solar_panel_data(solar_data, parameters={"rated_power": gboml_setup.solar_capacity_school})

    np.savetxt("data/demand.csv", demand)
    np.savetxt("data/gen_wind.csv", wind_prod["gen_wind_data"]["energy_generated"][:gboml_setup.T].values)
    np.savetxt("data/gen_solar.csv", solar_prod["gen_solar_data"]["energy_generated"][:gboml_setup.T].values)
    np.savetxt("data/gen_big_solar.csv", big_solar_prod["gen_solar_data"]["energy_generated"][:gboml_setup.T].values)

    gboml_model = GbomlGraph(gboml_setup.T)
    nodes, edges, global_params = gboml_model.import_all_nodes_and_edges(microgrid_file_path)
    gboml_model.add_nodes_in_model(*nodes)
    gboml_model.add_hyperedges_in_model(*edges)
    gboml_model.build_model()

    # solution = gboml_model.solve_cbc(opt_file=microgrid_file_path)
    solution = gboml_model.solve_clp()
    details = gboml_model.turn_solution_to_dictionary(
        solver_data=solution[3], status=solution[2], 
        solution=solution[0], objective=solution[1])

    total_parameters = len(global_params)  # Global parameters
    total_variables = 0
    total_constraints = 0

    # Iterate through all nodes
    for node in nodes:
        total_parameters += len(node.get_parameters())
        total_variables += len(node.get_variables())
        total_constraints += len(node.get_constraints())

    # Iterate through all hyperedges
    for edge in edges:
        total_constraints += len(edge.get_constraints())
    
    stats = {
        "implementation": "GBOML",
        "solver": solution[3]['name'],
        "parameters": total_parameters,
        "constraints": total_constraints,
        "variables": total_variables,
        "status": solution[2],
        "result": solution[1]
    }
    write(gboml_setup.output_path, stats)
    # return (solution, details)

if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 72
    
    run(setup=setup)
    setup.T = 48
    run(setup=setup)
    # (result, details) = run(setup=setup)
    # print(result)

