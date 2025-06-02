from matplotlib import pyplot as plt
import numpy as np
import cvxpy as cp

import sys
import os

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.consumer_mock_data_gen import generate_consumption_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data
from examples.helpers.microgrid_setup import MicrogridSetup
from examples.helpers.file_writer import write
from benchmark.benchmark import Benchmark


def solve_microgrid(setup:MicrogridSetup):    
    # Actual data
    df: dict = {}
    df2: dict = {}

    # NOTE Using predicted consumer data
    home_demand = {}
    for _ in range(setup.no_homes):
        home_demand = predict_consumer_data(dataframe=df, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1})
    school_demand = predict_consumer_data(dataframe=df2, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100})
    df["total_demand"] = home_demand["gen_consumption"]["consumption_kWh"] * setup.no_homes + school_demand["gen_consumption"]["consumption_kWh"]
    total_demand = df["total_demand"].values

    # NOTE Using mock data
    # demand = generate_consumption_data(df, parameters={"A0" : 1, "A1": 3, "A2": 2, "phi0": 3, "phi1": 9})
    # demand["consumption"] = (demand["consumption"] * num_homes) / 100
    # total_demand = demand
    # df["total_demand"] = total_demand

    get_wind_data(df, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    get_irradiation_data(df, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    df2 = df.copy()
    wind_prod = generate_wind_turbine_data(df, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    wind_prod = wind_prod["gen_wind_data"]["energy_generated"].values
    solar_prod = {}
    for _ in range(setup.no_homes):
        solar_prod = generate_solar_panel_data(df, parameters={"rated_power": setup.solar_capacity_home,})
        solar_prod = (solar_prod["gen_solar_data"]["energy_generated"].values * setup.no_homes)
    solar_prod_school = generate_solar_panel_data(df2, parameters={"rated_power": setup.solar_capacity_school,})
    solar_prod = solar_prod + solar_prod_school["gen_solar_data"]["energy_generated"].values

    # Decision variables
    grid_import = cp.Variable(setup.T, nonneg=True)
    battery_charge = [cp.Variable(setup.T, nonneg=True) for _ in range(setup.no_batteries)]
    battery_discharge = [cp.Variable(setup.T, nonneg=True)
                         for _ in range(setup.no_batteries)]
    battery_soc = [cp.Variable(setup.T+1, nonneg=True) for _ in range(setup.no_batteries)]
    c = [cp.Variable(setup.T, nonneg=True) for _ in range(setup.no_homes+setup.no_big_solar_panels)]

    # Constraints
    constraints = []

    # Initial battery state for each battery
    for i in range(setup.no_batteries):
        # Initial state of charge is 0
        constraints.append(battery_soc[i][0] == 0)

    for t in range(setup.T):
        # Control variable for solar panel activation
        for i in range(setup.no_homes+setup.no_big_solar_panels):
            constraints.append(c[i][t] <= 1)

        # Power balance: sum all battery charge/discharge and grid import
        total_battery_discharge = sum(
            battery_discharge[i][t] for i in range(setup.no_batteries))
        total_battery_charge = sum(
            battery_charge[i][t] for i in range(setup.no_batteries))
        constraints.append(
            grid_import[t] + total_battery_discharge -
            total_battery_charge == total_demand[t] -
            ((solar_prod[t] * c[t]) + wind_prod[t])
        )

        for i in range(setup.no_batteries):
            # Battery charge/discharge limits
            constraints.append(battery_charge[i][t] <= setup.battery_power)
            constraints.append(battery_discharge[i][t] <= setup.battery_power)

            # Battery state of charge limits
            constraints.append(
                battery_soc[i][t+1] == battery_soc[i][t] + battery_charge[i][t] *
                setup.battery_efficiency -
                battery_discharge[i][t] / setup.battery_efficiency
            )

            # Battery SoC limits
            constraints.append(battery_soc[i][t+1] <= setup.battery_capacity)

    # Objective: minimize total grid import
    objective = cp.Minimize(cp.sum(grid_import))

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=False)

    stats = {
        "implementation": "CVXPY",
        "solver": prob.solver_stats.solver_name,
        "parameters": sum(p.size for p in prob.parameters()),
        "constraints": len(prob.constraints),
        "variables": sum(v.size for v in prob.variables()),
        "status": prob.status,
        "result": prob.value
    }
    write(setup.output_path, stats)

    # Round results
    # grid_import.value = np.round(grid_import.value, 2)
    
    # print("Total grid import (kWh):", np.sum(grid_import.value))
    # print("Grid import per hour:", grid_import.value)

    # for i in range(setup.no_batteries):
    #     # print(f"Battery {i} charge:", np.round(battery_charge[i].value, 2))
    #     # print(f"Battery {i} discharge:", np.round(battery_discharge[i].value, 2))
    #     print(f"Battery {i} SoC:", np.round(battery_soc[i].value, 2))

    # # Plot demand and production
    # plt.figure(figsize=(12, 6))
    # plt.plot(total_demand[:setup.T], label="Total Demand")
    # plt.plot(solar_prod[:setup.T], label="Solar Production")
    # plt.plot(wind_prod[:setup.T], label="Wind Production")
    # plt.plot(solar_prod[:setup.T] + wind_prod[:setup.T], label="Total Renewable Production")
    # # Add battery SoC to the plot
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


def solve_microgrid_with_mock_data(time_intervals=24, num_batteries=1):
    T = time_intervals
    num_homes = 50
    solar_capacity_home = 5  # kW
    solar_capacity_school = 25  # kW
    wind_capacity = 50  # kW
    n_batteries = num_batteries
    battery_capacity = 500  # kWh
    battery_power = 500  # kW max charge/discharge
    battery_efficiency = 0.9  # 90% efficiency

    # Example time series
    np.random.seed(42)
    # home_demand = np.random.uniform(1, 2, (num_homes, T))  # kW per home
    home_demand = np.random.uniform(3, 5, (num_homes, T))  # kW per home
    # school_demand = np.random.uniform(10, 20, T)  # kW
    school_demand = np.random.uniform(30, 50, T)  # kW
    solar_profile = np.clip(
        np.sin(np.linspace(0, np.pi, T)), 0, None)  # normalized
    wind_profile = np.clip(
        np.sin(np.linspace(0, 2 * np.pi, T) - 1), 0, None)  # normalized

    # Aggregate demand and production
    total_home_demand = home_demand.sum(axis=0)
    total_demand = total_home_demand + school_demand
    solar_prod = solar_profile * \
        (num_homes * solar_capacity_home + solar_capacity_school)
    wind_prod = wind_profile * wind_capacity

    # Decision variables
    grid_import = cp.Variable(T, nonneg=True)
    battery_charge = [cp.Variable(T, nonneg=True) for _ in range(n_batteries)]
    battery_discharge = [cp.Variable(T, nonneg=True)
                         for _ in range(n_batteries)]
    battery_soc = [cp.Variable(T+1, nonneg=True) for _ in range(n_batteries)]
    c = cp.Variable(T, nonneg=True)

    # Constraints
    constraints = []

    # Initial battery state for each battery
    for i in range(n_batteries):
        # Initial state of charge is 0
        constraints.append(battery_soc[i][0] == 0)

    for t in range(T):
        # Control variable for solar panel activation
        constraints.append(c[t] <= 1)

        # Power balance: sum all battery charge/discharge and grid import
        total_battery_discharge = sum(
            battery_discharge[i][t] for i in range(n_batteries))
        total_battery_charge = sum(
            battery_charge[i][t] for i in range(n_batteries))
        constraints.append(
            grid_import[t] + total_battery_discharge -
            total_battery_charge == total_demand[t] -
            ((solar_prod[t] * c[t]) + wind_prod[t])
        )

        for i in range(n_batteries):
            # Battery charge/discharge limits
            constraints.append(battery_charge[i][t] <= battery_power)
            constraints.append(battery_discharge[i][t] <= battery_power)

            # Battery state of charge limits
            constraints.append(
                battery_soc[i][t+1] == battery_soc[i][t] + battery_charge[i][t] *
                battery_efficiency -
                battery_discharge[i][t] / battery_efficiency
            )

            # Battery SoC limits
            constraints.append(battery_soc[i][t+1] >= 0)
            constraints.append(battery_soc[i][t+1] <= battery_capacity)

    # Objective: minimize total grid import
    objective = cp.Minimize(cp.sum(grid_import))

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CBC, verbose=False)

    # Round results
    grid_import.value = np.round(grid_import.value, 2)
    
    print("Total grid import (kWh):", np.sum(grid_import.value))
    print("Grid import per hour:", grid_import.value)
    
    for i in range(n_batteries):
        # print(f"Battery {i} charge:", np.round(battery_charge[i].value, 2))
        # print(f"Battery {i} discharge:", np.round(battery_discharge[i].value, 2))
        print(f"Battery {i} SoC:", np.round(battery_soc[i].value, 2))

    # Plot demand and production
    plt.figure(figsize=(12, 6))
    plt.plot(total_demand, label="Total Demand")
    plt.plot(solar_prod, label="Solar Production")
    plt.plot(wind_prod, label="Wind Production")
    plt.plot(solar_prod + wind_prod, label="Total Renewable Production")
    # Add battery SoC to the plot
    for i in range(n_batteries):
        plt.plot(battery_soc[i].value[:-1], label=f"Battery {i} SoC", linestyle="--")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Demand and Production, and Battery SoC Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("demand_production_profiles.png")

    # return grid_import.value


if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 24
    # Benchmark.run(solve_microgrid, setup, runs=1)
    solve_microgrid(setup)