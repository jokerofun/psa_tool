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
    
    home_demands = []
    for _ in range(gboml_setup.no_homes):
        home_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": gboml_setup.T, "model_name":"consumer_model", "factor": 1})
        home_demand = home_demand_dict["gen_consumption"]["consumption_kWh"]
        home_demands.append(home_demand)
    school_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": gboml_setup.T, "model_name":"consumer_model", "factor": 100})
    school_demand = school_demand_dict["gen_consumption"]["consumption_kWh"]
    demand =  [sum(group) for group in zip(*home_demands)] + school_demand
    np.savetxt("data/gboml_data/demand.csv", demand)

    wind_data_dict = get_wind_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_prod_dict = generate_wind_turbine_data(wind_data_dict, parameters={"rated_power": gboml_setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    wind_prod = wind_prod_dict["gen_wind_data"]["energy_generated"]
    np.savetxt("data/gboml_data/gen_wind.csv", wind_prod[:gboml_setup.T])
    
    solar_prod = {}
    for i in range(1, gboml_setup.no_solar_panels + 1):
        solar_data_dict = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})
        solar_prod_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": gboml_setup.solar_capacity_home,})
        solar_prod = solar_prod_dict["gen_solar_data"]["energy_generated"]
        np.savetxt(f"data/gboml_data/gen_solar_{i}.csv", solar_prod[:gboml_setup.T])

    solar_data_dict = get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    solar_prod_school_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": gboml_setup.solar_capacity_school})
    solar_prod_school = solar_prod_school_dict["gen_solar_data"]["energy_generated"]
    np.savetxt("data/gboml_data/gen_big_solar.csv", solar_prod_school[:gboml_setup.T])

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

    # FROM HERE - DON'T COUNT THESE CHARACTERS FOR PRODUCTIVITY EXPERIMENTS
    total_parameters = len(global_params)
    total_variables = 0
    total_constraints = 0

    for node in nodes:
        total_parameters += len(node.get_parameters())
        total_variables += len(node.get_variables())
        total_constraints += len(node.get_constraints())

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
    setup.T = 24
    
    run(setup=setup)
    # (result, details) = run(setup=setup)
    # print(result)

