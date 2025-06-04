import numpy as np
import cvxpy as cp
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data
from examples.helpers.microgrid_setup import MicrogridSetup
from examples.helpers.file_writer import write

import time
# from examples.helpers.plot_microgrid_solution import plot_microgrid_solution

def solve_microgrid_cvxpy(setup:MicrogridSetup):    
    start_time = time.time()
    home_demands = []
    setup_time = time.time()
    for _ in range(setup.no_homes):
        home_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1})
        home_demand = home_demand_dict["gen_consumption"]["consumption_kWh"]
        home_demands.append(home_demand)
    school_demand_dict = predict_consumer_data(dataframe={}, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100})
    school_demand = school_demand_dict["gen_consumption"]["consumption_kWh"]
    total_demand = [sum(group) for group in zip(*home_demands)] + school_demand

    wind_data_dict = get_wind_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_prod_dict = generate_wind_turbine_data(wind_data_dict, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    wind_prod = wind_prod_dict["gen_wind_data"]["energy_generated"]

    solar_data_dict = get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
    solar_prod_school_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_school})
    solar_prod_school = solar_prod_school_dict["gen_solar_data"]["energy_generated"]

    solar_prods = []
    for _ in range(setup.no_solar_panels):
        solar_data_dict =  get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
        solar_prod_dict = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_home,})
        solar_prod = solar_prod_dict["gen_solar_data"]["energy_generated"]
        solar_prods.append(solar_prod)
    dataflow_time = time.time()

    grid_import = cp.Variable(setup.T, nonneg=True)
    battery_charge = [cp.Variable(setup.T, nonneg=True) for _ in range(setup.no_batteries)]
    battery_discharge = [cp.Variable(setup.T, nonneg=True) for _ in range(setup.no_batteries)]
    battery_soc = [cp.Variable(setup.T+1, nonneg=True) for _ in range(setup.no_batteries)]
    c_homes = [cp.Variable(setup.T, nonneg=True) for _ in range(setup.no_homes)]
    c_school = cp.Variable(setup.T, nonneg=True)

    constraints = []
    for i in range(setup.no_batteries):
        constraints.append(battery_soc[i][0] == 0)

    for t in range(setup.T):
        for i in range(setup.no_homes):
            constraints.append(c_homes[i][t] <= 1)
        constraints.append(c_school[t] <= 1)

        total_hourly_battery_discharge = sum(battery_discharge[i][t] for i in range(setup.no_batteries))
        total_hourly_battery_charge = sum(battery_charge[i][t] for i in range(setup.no_batteries))
        total_hourly_solar_prod_homes = sum(c_homes[i][t] * solar_prods[i][t] for i in range(setup.no_solar_panels))
        total_hourly_solar_prod_school = c_school[t] * solar_prod_school[t]
        constraints.append(
            grid_import[t] + total_hourly_battery_discharge + total_hourly_solar_prod_homes + total_hourly_solar_prod_school + wind_prod[t]
            == total_hourly_battery_charge + total_demand[t])

        for i in range(setup.no_batteries):
            constraints.append(battery_charge[i][t] <= setup.battery_power)
            constraints.append(battery_discharge[i][t] <= setup.battery_power)

            constraints.append(
                battery_soc[i][t+1] == battery_soc[i][t] + battery_charge[i][t] *
                setup.battery_efficiency - battery_discharge[i][t] / setup.battery_efficiency)

            constraints.append(battery_soc[i][t+1] <= setup.battery_capacity)

    objective = cp.Minimize(cp.sum(grid_import))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=False)
    end_time = time.time()

    # FROM HERE - DON'T COUNT THESE CHARACTERS FOR PRODUCTIVITY EXPERIMENTS
    stats = {
        "implementation": "CVXPY",
        "solver": prob.solver_stats.solver_name,
        "parameters": sum(p.size for p in prob.parameters()),
        "constraints": len(prob.constraints),
        "variables": sum(v.size for v in prob.variables()),
        "status": prob.status,
        "result": prob.value,
        "T": setup.T,
        "no_homes": setup.no_homes,
        "setup_time": setup_time - start_time,
        "dataflow_time": dataflow_time - setup_time,
        "optimizer_time": end_time - dataflow_time
    }
    write(setup.output_path, stats)

    # Round results
    # grid_import.value = np.round(grid_import.value, 2)
    
    # print("Total grid import (kWh):", np.sum(grid_import.value))
    # print("Grid import per hour:", grid_import.value)

    # for i in range(setup.no_batteries):
    #     print(f"Battery {i} SoC:", np.round(battery_soc[i].value, 2))

    # plt.figure(figsize=(12, 6))
    # plt.plot(total_demand[:setup.T], label="Total Demand")
    # plt.plot(solar_prod[:setup.T], label="Solar Production")
    # plt.plot(wind_prod[:setup.T], label="Wind Production")
    # plt.plot(solar_prod[:setup.T] + wind_prod[:setup.T], label="Total Renewable Production")
    # for i in range(setup.no_batteries):
    #     plt.plot(battery_soc[i][:setup.T].value[:-1], label=f"Battery {i} SoC", linestyle="--")
    # plt.xlabel("Hour")
    # plt.ylabel("kWh")
    # plt.title("Demand and Production, and Battery SoC Profiles")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("demand_production_profiles.png")

    # return grid_import.value

if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 24
    solve_microgrid_cvxpy(setup)