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
from examples.psa_examples.microgrid_setup import MicrogridSetup

def run_gboml_microgrid(T, no_households, big_solar_panels_no, wind_turbines_no, 
                        solar_capacity_home, solar_capacity_school, wind_capacity, no_batteries, 
                        battery_soc, battery_capacity, battery_power, battery_efficiency, 
                        microgrid_file_path="examples/psa_examples/microgrid_test.txt"):
    gboml_build_domain.build_microgrid(time_length=T, solar_panels_no=no_households, big_solar_panels_no=big_solar_panels_no, 
                                    batteries_no=no_batteries, wind_turbines_no=wind_turbines_no, 
                                    battery_charging_power=battery_power, battery_discharging_power=battery_power, 
                                    battery_capacity=battery_capacity, battery_efficiency=battery_efficiency, 
                                    battery_soc=battery_soc)
    home_predicted_consumption = predict_consumer_data(parameters={"hours": 24, "model_name":"consumer_model", "factor": 1})["gen_consumption"]["consumption_kWh"].values
    school_predicted_consumption = predict_consumer_data(parameters={"hours": 24, "model_name":"consumer_model", "factor": 100})["gen_consumption"]["consumption_kWh"].values
    demand =  home_predicted_consumption * no_households + school_predicted_consumption
    wind_data = get_wind_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    solar_data = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    wind_prod = generate_wind_turbine_data(wind_data, parameters={"rated_power": wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    solar_prod = generate_solar_panel_data(solar_data, parameters={"rated_power": solar_capacity_home,})

    solar_data = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    big_solar_prod = generate_solar_panel_data(solar_data, parameters={"rated_power": solar_capacity_school})

    np.savetxt("data/demand.csv", demand)
    np.savetxt("data/gen_wind.csv", wind_prod["gen_wind_data"]["energy_generated"][:T].values)
    np.savetxt("data/gen_solar.csv", solar_prod["gen_solar_data"]["energy_generated"][:T].values)
    np.savetxt("data/gen_big_solar.csv", big_solar_prod["gen_solar_data"]["energy_generated"][:T].values)

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
    setup = MicrogridSetup()
    
    (result, details) = run_gboml_microgrid(T=setup.T, 
                                            no_households=setup.no_households, 
                                            big_solar_panels_no=setup.big_solar_panels_no,
                                            wind_turbines_no=setup.wind_turbines_no,
                                            solar_capacity_home=setup.solar_capacity_home, 
                                            solar_capacity_school=setup.solar_capacity_school, 
                                            wind_capacity=setup.wind_capacity,
                                            no_batteries=setup.no_batteries, 
                                            battery_soc=setup.battery_SoC,
                                            battery_capacity=setup.battery_capacity, 
                                            battery_power=setup.battery_power,
                                            battery_efficiency=setup.battery_efficiency)
    print(details)

