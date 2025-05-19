import numpy as np
import cvxpy as cp

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from benchmark.benchmark import Benchmark


def solve_microgrid(time_intervals=24, num_batteries=1):
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
    home_demand = np.random.uniform(5, 8, (num_homes, T))  # kW per home
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
    prob.solve(solver=cp.OSQP, verbose=False)

    # Round results
    grid_import.value = np.round(grid_import.value, 2)
    
    print("Total grid import (kWh):", np.sum(grid_import.value))
    print("Grid import per hour:", grid_import.value)

    return grid_import.value


if __name__ == "__main__":
    Benchmark.run(solve_microgrid, runs=1)
    Benchmark.run(solve_microgrid, 48, 3, runs=1)
