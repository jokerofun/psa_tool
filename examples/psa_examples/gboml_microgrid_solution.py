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

T = 24 # time segments
no_households = 50
solar_capacity_home = 5 # kW
solar_capacity_school = 25 # kW
wind_capacity = 1000 # kW
no_batteries = 3
battery_capacity = 500 # kWh
battery_power = 500 # kW
battery_efficiency = 0.9 # in %

demand = predict_consumer_data()["gen_consumption"]["consumption_kWh"].values * no_households
wind_dict = {}
solar_dict = {}
wind_data = get_wind_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
solar_data = get_irradiation_data(dataframe={}, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
wind_prod = generate_wind_turbine_data(wind_data, parameters={"rated_power": wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
solar_prod = generate_solar_panel_data(solar_data, parameters={"rated_power": solar_capacity_home,})

np.savetxt("data/demand.csv", demand)
np.savetxt("data/gen_wind.csv", wind_prod["gen_wind_data"]["energy_generated"].values)
np.savetxt("data/gen_solar.csv", solar_prod["gen_solar_data"]["energy_generated"].values)

gboml_model = GbomlGraph(T)
nodes, edges, _ = gboml_model.import_all_nodes_and_edges("examples/psa_examples/microgrid.txt")
gboml_model.add_nodes_in_model(*nodes)
gboml_model.add_hyperedges_in_model(*edges)
gboml_model.build_model()

solution = gboml_model.solve_clp()
print(solution)