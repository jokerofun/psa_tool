import os, sys
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import Var, Constraint, Param, value

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

def solve_microgrid_pyomo(setup:MicrogridSetup):
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

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, setup.T-1)
    model.B = pyo.RangeSet(0, setup.no_batteries-1)
    model.H = pyo.RangeSet(0, setup.no_homes-1)
    model.S = pyo.RangeSet(0, setup.no_big_solar_panels-1)

    model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
    model.battery_charge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_discharge = pyo.Var(model.B, model.T, domain=pyo.NonNegativeReals)
    model.battery_soc = pyo.Var(model.B, range(setup.T+1), domain=pyo.NonNegativeReals)
    model.c_home = pyo.Var(model.H, model.T, domain=pyo.NonNegativeReals)
    model.c_school = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    def soc_init_rule(m, b):
        return m.battery_soc[b,0] == 0
    model.soc_init = pyo.Constraint(model.B, rule=soc_init_rule)

    def power_balance_rule(m, t):
        total_hourly_battery_discharge = sum(m.battery_discharge[b, t] for b in model.B)
        total_hourly_battery_charge = sum(m.battery_charge[b, t] for b in model.B)
        total_hourly_solar_prod_homes = sum(m.c_home[h, t] * solar_prods[h][t] for h in model.H)
        total_hourly_solar_prod_school = m.c_school[t] * solar_prod_school[t]
        return (m.grid_import[t] + total_hourly_solar_prod_homes + total_hourly_solar_prod_school + wind_prod[t] + total_hourly_battery_discharge == 
                total_hourly_battery_charge + total_demand[t])
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

    def soc_max_rule(m, b, t):
        return m.battery_soc[b, t+1] <= setup.battery_capacity
    model.soc_max = pyo.Constraint(model.B, range(setup.T), rule=soc_max_rule)

    def c_home_max_rule(m, h, t):
        return m.c_home[h, t] <= 1
    model.c_home_max = pyo.Constraint(model.H, range(setup.T), rule=c_home_max_rule)

    def c_school_max_rule(m, t):
        return m.c_school[t] <= 1
    model.c_school_max = pyo.Constraint(range(setup.T), rule=c_school_max_rule)

    model.obj = pyo.Objective(expr=sum(model.grid_import[t] for t in model.T), sense=pyo.minimize)

    solver = pyo.SolverFactory('cbc', executable="examples/Pyomo/cbc/bin/cbc.exe")
    result = solver.solve(model, tee=False)
    end_time = time.time()

    # FROM HERE - DON'T COUNT THESE CHARACTERS FOR PRODUCTIVITY EXPERIMENTS
    stats = {
        "implementation": "Pyomo",
        "solver": solver.name,
        "parameters": len([1 for _ in model.component_data_objects(Param, active=True)]),
        "constraints": len([1 for _ in model.component_data_objects(Constraint, active=True)]),
        "variables": len([1 for _ in model.component_data_objects(Var, active=True)]),
        "status": result.solver.termination_condition,
        "result": value(model.obj),
        "T": setup.T,
        "no_homes": setup.no_homes,
        "setup_time": setup_time - start_time,
        "dataflow_time": dataflow_time - setup_time,
        "optimizer_time": end_time - dataflow_time
    }
    write(setup.output_path, stats)

    # TODO: plot needs to be corrected
    # grid_import = np.array([pyo.value(model.grid_import[t]) for t in model.T])             
    # all_soc_values = []
    # for i in range(setup.no_batteries):
    #     soc_values = np.array([pyo.value(model.battery_soc[i, t]) for t in range(setup.T+1)])
    #     all_soc_values.append(soc_values)

    # total_solar_prod_homes = np.array([sum(pyo.value(model.c_home[h, t]) * solar_prods[h][t] for h in model.H for t in model.T)])
    # total_solar_prod_school = np.array([pyo.value(model.c_school[t]) * solar_prod_school[t] for t in model.T])
    # total_solar_prod = total_solar_prod_homes + total_solar_prod_school
    # plot_microgrid_solution(setup=setup, solution="pyomo", grid_import=grid_import, total_demand=total_demand, 
    #                         solar_prod=total_solar_prod, wind_prod=wind_prod, soc_values=all_soc_values)
    # return grid_import

if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 24
    solve_microgrid_pyomo(setup=setup)