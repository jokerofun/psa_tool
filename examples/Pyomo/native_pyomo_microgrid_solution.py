import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import Var, Constraint, Param, value

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.consumer_mock_data_gen import generate_consumption_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data
from examples.helpers.microgrid_setup import MicrogridSetup
from examples.helpers.file_writer import write

# from benchmark.benchmark import Benchmark

def solve_microgrid_pyomo(setup:MicrogridSetup):

    # Actual data
    df: dict = {}
    df2: dict = {}

    # NOTE Using predicted consumer data
    home_demands = []
    for _ in range(setup.no_homes):
        home_demands.append(predict_consumer_data(dataframe=df, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1})["gen_consumption"]["consumption_kWh"])
    school_demand = predict_consumer_data(dataframe=df2, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100})
    df["total_demand"] = [sum(group) for group in zip(*home_demands)] + school_demand["gen_consumption"]["consumption_kWh"]
    total_demand = df["total_demand"].values

    get_wind_data(df, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    wind_prod = generate_wind_turbine_data(df, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25})
    wind_prod = wind_prod["gen_wind_data"]["energy_generated"].values

    get_irradiation_data(df, parameters={"latitude": 57.0488, "longitude": 9.9217})  # Aalborg, Denmark
    df2 = df.copy()
    solar_prod_school = generate_solar_panel_data(df2, parameters={"rated_power": setup.solar_capacity_school,})
    solar_prod_school = solar_prod_school["gen_solar_data"]["energy_generated"].values

    solar_prods = []
    for _ in range(setup.no_solar_panels):
        solar_data_dict =  get_irradiation_data({}, parameters={"latitude": 57.0488, "longitude": 9.9217})
        solar_prod = generate_solar_panel_data(solar_data_dict, parameters={"rated_power": setup.solar_capacity_home,})
        solar_prod = (solar_prod["gen_solar_data"]["energy_generated"].values)
        solar_prods.append(solar_prod)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, setup.T-1)
    model.B = pyo.RangeSet(0, setup.no_batteries-1)
    model.H = pyo.RangeSet(0, setup.no_homes-1)
    model.S = pyo.RangeSet(0, setup.no_big_solar_panels-1)

    # Variables
    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.battery_charge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_discharge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_soc = pyo.Var(model.B, range(setup.T+1), domain=pyo.NonNegativeReals)
    # model.c = pyo.Var(model.T, bounds=(0,1))
    model.c_home = pyo.Var(model.H, model.T, domain=pyo.NonNegativeReals)
    model.c_school = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # Initial SoC
    def soc_init_rule(m, b):
        return m.battery_soc[b,0] == 0
    model.soc_init = pyo.Constraint(model.B, rule=soc_init_rule)

    # Constraints for each time step
    def power_balance_rule(m, t):
        total_battery_discharge = sum(m.battery_discharge[b, t] for b in model.B)
        total_battery_charge = sum(m.battery_charge[b, t] for b in model.B)
        # total_hourly_solar_prod = m.c[t] * solar_prod[t]
        total_hourly_solar_prod = sum(m.c_home[h, t] * solar_prods[h][t] for h in model.H)
        hourly_solar_prod_school = m.c_school[t] * solar_prod_school[t]
        return (m.grid_import[t] + total_hourly_solar_prod + hourly_solar_prod_school + wind_prod[t] + total_battery_discharge == 
                total_battery_charge + total_demand[t])
    model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

    def battery_charge_limit_rule(m, b, t):
        return m.battery_charge[b, t] <= setup.battery_power
    model.battery_charge_limit = pyo.Constraint(model.B, model.T, rule=battery_charge_limit_rule)

    def battery_discharge_limit_rule(m, b, t):
        return m.battery_discharge[b, t] <= setup.battery_power
    model.battery_discharge_limit = pyo.Constraint(model.B, model.T, rule=battery_discharge_limit_rule)

    def soc_update_rule(m, b, t):
        return m.battery_soc[b, t+1] == m.battery_soc[b, t] + m.battery_charge[b, t] * setup.battery_efficiency - m.battery_discharge[b, t] / setup.battery_efficiency
    model.soc_update = pyo.Constraint(model.B, range(setup.T), rule=soc_update_rule)

    # def soc_min_rule(m, b, t):
    #     return m.battery_soc[b, t+1] >= 0
    # model.soc_min = pyo.Constraint(model.B, range(setup.T), rule=soc_min_rule)

    def soc_max_rule(m, b, t):
        return m.battery_soc[b, t+1] <= setup.battery_capacity
    model.soc_max = pyo.Constraint(model.B, range(setup.T), rule=soc_max_rule)

    def c_home_max_rule(m, h, t):
        return m.c_home[h, t] <= 1
    model.c_home_max = pyo.Constraint(model.H, range(setup.T), rule=c_home_max_rule)

    def c_school_max_rule(m, t):
        return m.c_school[t] <= 1
    model.c_school_max = pyo.Constraint(range(setup.T), rule=c_school_max_rule)

    # Objective: minimize total grid import
    model.obj = pyo.Objective(expr=sum(model.grid_import[t] for t in model.T), sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('cbc', executable="C:/Users/samue/Downloads/cbc/bin/cbc.exe")
    result = solver.solve(model, tee=False)

    stats = {
        "implementation": "Pyomo",
        "solver": solver.name,
        "parameters": len([1 for _ in model.component_data_objects(Param, active=True)]),
        "constraints": len([1 for _ in model.component_data_objects(Constraint, active=True)]),
        "variables": len([1 for _ in model.component_data_objects(Var, active=True)]),
        "status": result.solver.termination_condition,
        "result": value(model.obj)
    }
    write(setup.output_path, stats)

    # grid_import = np.array([pyo.value(model.grid_import[t]) for t in model.T])
    # print("Total grid import (kWh):", np.sum(grid_import))
    # print("Grid import per hour:", grid_import)

    # # Plot demand and production
    # plt.figure(figsize=(12, 6))
    # plt.plot(total_demand[:setup.T], label="Total Demand")
    # plt.plot(solar_prod[:setup.T], label="Solar Production")
    # plt.plot(wind_prod[:setup.T], label="Wind Production")
    # plt.plot(solar_prod[:setup.T] + wind_prod[:setup.T], label="Total Renewable Production")
    # # Add battery SoC to the plot
    # for i in range(setup.no_batteries):
    #     soc_values = [pyo.value(model.battery_soc[i, t]) for t in range(setup.T+1)]
    #     plt.plot(soc_values[:-1], label=f"Battery {i} SoC", linestyle="--")
    # plt.xlabel("Hour")
    # plt.ylabel("kWh")
    # plt.title("Demand and Production, and Battery SoC Profiles")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("figures/demand_production_profiles.png")
    # plt.show()

    # return grid_import


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
    # Add battery SoC to the plot
    for i in range(n_batteries):
        plt.plot(model.battery_soc[i].value[:-1], label=f"Battery {i} SoC", linestyle="--")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Demand and Production, and Battery SoC Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("demand_production_profiles.png")

    return grid_import

if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 48
    # Benchmark.run(solve_microgrid_pyomo, runs=1)
    # Benchmark.run(solve_microgrid_pyomo, setup, runs=1)
    solve_microgrid_pyomo(setup=setup)