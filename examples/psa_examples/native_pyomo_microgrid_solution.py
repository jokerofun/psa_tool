import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pyomo.environ as pyo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.dataflow_nodes.consumer_mock_data_gen import generate_consumption_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data

from benchmark.benchmark import Benchmark

def solve_microgrid_pyomo(time_intervals=24, num_batteries=1):
    T = time_intervals
    num_homes = 50
    solar_capacity_home = 5  # kW
    solar_capacity_school = 25  # kW
    wind_capacity = 100  # kW
    n_batteries = num_batteries
    battery_capacity = 500  # kWh
    battery_power = 500  # kW max charge/discharge
    battery_efficiency = 0.9  # 90% efficiency

    # Actual data
    df: dict = {}
    demand = generate_consumption_data(df, parameters={"A0" : 1, "A1": 3, "A2": 2, "phi0": 3, "phi1": 9})
    demand["consumption"] = (demand["consumption"] * num_homes) / 100
    total_demand = demand
    df["total_demand"] = total_demand
    wind_data = get_wind_data(df)
    irradiation_data = get_irradiation_data(df)
    df["wind_data"] = wind_data
    df["solar_irradation"] = irradiation_data
    wind_prod = generate_wind_turbine_data(df, parameters={"rated_power": wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    solar_prod = generate_solar_panel_data(df, parameters={"rated_power": solar_capacity_home,})

    total_demand = total_demand.values[:, 1]
    solar_prod = solar_prod.values[:, 1]
    wind_prod = wind_prod.values[:, 1]

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T-1)
    model.B = pyo.RangeSet(0, n_batteries-1)

    # Variables
    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.battery_charge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_discharge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_soc = pyo.Var(model.B, range(T+1), domain=pyo.NonNegativeReals)
    model.c = pyo.Var(model.T, bounds=(0,1))

    # Initial SoC
    def soc_init_rule(m, b):
        return m.battery_soc[b,0] == 0
    model.soc_init = pyo.Constraint(model.B, rule=soc_init_rule)

    # Constraints for each time step
    def power_balance_rule(m, t):
        total_battery_discharge = sum(m.battery_discharge[b, t] for b in model.B)
        total_battery_charge = sum(m.battery_charge[b, t] for b in model.B)
        return (m.grid_import[t] + total_battery_discharge - total_battery_charge ==
                total_demand[t] - ((solar_prod[t] * m.c[t]) + wind_prod[t]))
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def battery_charge_limit_rule(m, b, t):
        return m.battery_charge[b, t] <= battery_power
    model.battery_charge_limit = pyo.Constraint(model.B, model.T, rule=battery_charge_limit_rule)

    def battery_discharge_limit_rule(m, b, t):
        return m.battery_discharge[b, t] <= battery_power
    model.battery_discharge_limit = pyo.Constraint(model.B, model.T, rule=battery_discharge_limit_rule)

    def soc_update_rule(m, b, t):
        return m.battery_soc[b, t+1] == m.battery_soc[b, t] + m.battery_charge[b, t] * battery_efficiency - m.battery_discharge[b, t] / battery_efficiency
    model.soc_update = pyo.Constraint(model.B, range(T), rule=soc_update_rule)

    def soc_min_rule(m, b, t):
        return m.battery_soc[b, t+1] >= 0
    model.soc_min = pyo.Constraint(model.B, range(T), rule=soc_min_rule)

    def soc_max_rule(m, b, t):
        return m.battery_soc[b, t+1] <= battery_capacity
    model.soc_max = pyo.Constraint(model.B, range(T), rule=soc_max_rule)

    # Objective: minimize total grid import
    model.obj = pyo.Objective(expr=sum(model.grid_import[t] for t in model.T), sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    grid_import = np.array([pyo.value(model.grid_import[t]) for t in model.T])
    print("Total grid import (kWh):", np.sum(grid_import))
    print("Grid import per hour:", grid_import)

    # Plot demand and production
    plt.figure(figsize=(12, 6))
    plt.plot(total_demand[:T], label="Total Demand")
    plt.plot(solar_prod[:T], label="Solar Production")
    plt.plot(wind_prod[:T], label="Wind Production")
    plt.plot(solar_prod[:T] + wind_prod[:T], label="Total Renewable Production")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Demand and Production Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("demand_production_profiles.png")

    return grid_import


def solve_microgrid_pyomo_with_mock_data(time_intervals=24, num_batteries=1):
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

    total_home_demand = home_demand.sum(axis=0)
    total_demand = total_home_demand + school_demand
    solar_prod = solar_profile * (num_homes * solar_capacity_home + solar_capacity_school)
    wind_prod = wind_profile * wind_capacity

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T-1)
    model.B = pyo.RangeSet(0, n_batteries-1)

    # Variables
    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.battery_charge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_discharge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_soc = pyo.Var(model.B, range(T+1), domain=pyo.NonNegativeReals)
    model.c = pyo.Var(model.T, bounds=(0,1))

    # Initial SoC
    def soc_init_rule(m, b):
        return m.battery_soc[b,0] == 0
    model.soc_init = pyo.Constraint(model.B, rule=soc_init_rule)

    # Constraints for each time step
    def power_balance_rule(m, t):
        total_battery_discharge = sum(m.battery_discharge[b, t] for b in model.B)
        total_battery_charge = sum(m.battery_charge[b, t] for b in model.B)
        return (m.grid_import[t] + total_battery_discharge - total_battery_charge ==
                total_demand[t] - ((solar_prod[t] * m.c[t]) + wind_prod[t]))
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def battery_charge_limit_rule(m, b, t):
        return m.battery_charge[b, t] <= battery_power
    model.battery_charge_limit = pyo.Constraint(model.B, model.T, rule=battery_charge_limit_rule)

    def battery_discharge_limit_rule(m, b, t):
        return m.battery_discharge[b, t] <= battery_power
    model.battery_discharge_limit = pyo.Constraint(model.B, model.T, rule=battery_discharge_limit_rule)

    def soc_update_rule(m, b, t):
        return m.battery_soc[b, t+1] == m.battery_soc[b, t] + m.battery_charge[b, t] * battery_efficiency - m.battery_discharge[b, t] / battery_efficiency
    model.soc_update = pyo.Constraint(model.B, range(T), rule=soc_update_rule)

    def soc_min_rule(m, b, t):
        return m.battery_soc[b, t+1] >= 0
    model.soc_min = pyo.Constraint(model.B, range(T), rule=soc_min_rule)

    def soc_max_rule(m, b, t):
        return m.battery_soc[b, t+1] <= battery_capacity
    model.soc_max = pyo.Constraint(model.B, range(T), rule=soc_max_rule)

    # Objective: minimize total grid import
    model.obj = pyo.Objective(expr=sum(model.grid_import[t] for t in model.T), sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('gurobi')
    result = solver.solve(model, tee=False)

    grid_import = np.array([pyo.value(model.grid_import[t]) for t in model.T])
    print("Total grid import (kWh):", np.sum(grid_import))
    print("Grid import per hour:", grid_import)

    # Plot demand and production
    plt.figure(figsize=(12, 6))
    plt.plot(total_demand[:T], label="Total Demand")
    plt.plot(solar_prod[:T], label="Solar Production")
    plt.plot(wind_prod[:T], label="Wind Production")
    plt.plot(solar_prod[:T] + wind_prod[:T], label="Total Renewable Production")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Demand and Production Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("demand_production_profiles.png")

    return grid_import

if __name__ == "__main__":
    # Benchmark.run(solve_microgrid_pyomo, runs=1)
    Benchmark.run(solve_microgrid_pyomo, 24, 3, runs=1)