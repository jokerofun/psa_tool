from gboml import GbomlGraph
import numpy as np
import gboml_build_domain

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.consumer_mock_data_gen import generate_consumption_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data

def run_gboml_microgrid(T, no_households, solar_capacity_home, solar_capacity_school, wind_capacity, 
                        no_batteries, battery_capacity, battery_power, battery_efficiency, microgrid_file_path="examples/psa_examples/microgrid_test.txt"):
    demand = predict_consumer_data()["gen_consumption"]["consumption_kWh"].values * no_households
    wind_data = get_wind_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    solar_data = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    wind_prod = generate_wind_turbine_data(wind_data, parameters={"rated_power": wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    solar_prod = generate_solar_panel_data(solar_data, parameters={"rated_power": solar_capacity_home,})

    np.savetxt("data/demand.csv", demand)
    np.savetxt("data/gen_wind.csv", wind_prod["gen_wind_data"]["energy_generated"][:T].values)
    np.savetxt("data/gen_solar.csv", solar_prod["gen_solar_data"]["energy_generated"][:T].values)

    gboml_model = GbomlGraph(T)
    nodes, edges, _ = gboml_model.import_all_nodes_and_edges(microgrid_file_path)
    gboml_model.add_nodes_in_model(*nodes)
    gboml_model.add_hyperedges_in_model(*edges)
    gboml_model.build_model()

    solution = gboml_model.solve_clp()
    details = gboml_model.turn_solution_to_dictionary(
        solver_data=solution[3], status=solution[2], 
        solution=solution[0], objective=solution[1])

    return (solution, details)

if __name__ == "__main__":
    T = 24 # time segments
    no_households = 50
    solar_capacity_home = 5 # kW
    solar_capacity_school = 25 # kW
    wind_capacity = 1000 # kW
    no_batteries = 3
    battery_capacity = 500 # kWh
    battery_power = 500 # kW
    battery_efficiency = 0.9 # in %

    gboml_build_domain.build_microgrid(time_length=T, solar_panels_no=no_households, batteries_no=no_batteries,
                                       wind_turbines_no=1, battery_charging_power=battery_power, 
                                       battery_discharging_power=battery_power, battery_capacity=battery_capacity, 
                                       battery_efficiency=battery_efficiency, battery_soc=0)
    
    (result, details) = run_gboml_microgrid(T=T, no_households=no_households, solar_capacity_home=solar_capacity_home, 
                        solar_capacity_school=solar_capacity_school, wind_capacity=wind_capacity, 
                        no_batteries=no_batteries, battery_capacity=battery_capacity, battery_power=battery_power,
                        battery_efficiency=battery_efficiency)
    print(result)

