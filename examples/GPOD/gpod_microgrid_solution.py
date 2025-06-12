from datetime import datetime, timedelta
import cvxpy as cp
import sys, os

import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from examples.dataflow_nodes.spot_prices_data import fetch_spot_prices
from src.optimization.energy_domain import ConnectingNode
from src.optimization.graph_problem_class import GraphProblemClass
from src.optimization.energy_domain import ConnectingNode
from examples.GPOD.microgrid_domain import Consumer, MeteringPoint, SolarPanel, Battery, WindTurbine
from src.dataflow.dataflow_classes_v2 import DataProcessingTask
from src.dataflow.default_tasks import *
from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data
from examples.helpers.microgrid_setup import MicrogridSetup
from examples.helpers.file_writer import write
from examples.helpers.plot_microgrid_solution import plot_microgrid_solution

import time
# from examples.helpers.plot_microgrid_solution import plot_microgrid_solution

def solve_microgrid_gpod(setup: MicrogridSetup):
    start_time = time.time()
    problemClass = GraphProblemClass("microgrid_problem", time_length=setup.T)

    metering_point = MeteringPoint("grid1")
    # homes = [Consumer(f"household_{i+1}") for i in range(setup.no_homes)]
    # school = Consumer("school")
    # solar_panels_homes = [SolarPanel(f"solar_panel_{i+1}") for i in range(setup.no_homes)]
    # solar_panel_school = SolarPanel("solar_panel_school")
    # wind_turbine = WindTurbine("wind_turbine")
    # batteries = [Battery(f"battery_{i+1}", setup.battery_power, setup.battery_power, setup.battery_capacity, setup.battery_efficiency) for i in range(setup.no_batteries)]
    battery1 = Battery("battery1", 20, 20, 20, 0.9)
    battery2 = Battery("battery2", 20, 20, 60, 0.7)
    battery3 = Battery("battery3", 10, 10, 300, 0.45)


    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=setup.T)

    metering_point.dataflow.task("get_spot_prices", DataProcessingTask, process_func=fetch_spot_prices, parameters={"start_date": start, "end_date": end}, final=True)

    # for home in homes:
        # home.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1}, final=True)
    # school.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100}, final=True)
    # for solar_panel in solar_panels_homes:
        # task1 = solar_panel.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
        # task2 = solar_panel.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_home}, final=True)
        # task1 >> task2
    # solar_school_task1 = solar_panel_school.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217}) 
    # solar_school_task2 = solar_panel_school.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_school}, final=True)
    # solar_school_task1 >> solar_school_task2
    # wind_task1 = wind_turbine.dataflow.task("get_wind_data", DataProcessingTask, process_func=get_wind_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
    # wind_task2 = wind_turbine.dataflow.task("gen_wind_data", DataProcessingTask, process_func=generate_wind_turbine_data, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed": 14, "cut_out_speed": 25}, final=True)
    # wind_task1 >> wind_task2
    
    # problemClass.add_nodes([*homes, *solar_panels_homes, *batteries, 
    #                         school, solar_panel_school, wind_turbine, 
    #                         metering_point])
    
    # metering_point.connect_to([*homes, *solar_panels_homes, *batteries, school, 
                                #   solar_panel_school, wind_turbine])

    problemClass.add_nodes([battery1, battery2, battery3, metering_point])

    metering_point.connect_to([battery1, battery2, battery3])

    setup_time = time.time()
    result = problemClass.solve(solver=cp.CBC, objective="minimize", value="cost")
    end_time = time.time()
    dataflow_time = problemClass.dataflow_time
    # FROM HERE - DON'T COUNT THESE CHARACTERS FOR PRODUCTIVITY EXPERIMENTS
    stats = {
        "implementation": "GPO-D",
        "solver": result.solver_stats.solver_name,
        "parameters": sum(p.size for p in result.parameters()),
        "total_constraints": len(result.constraints),
        "variables": len(result.variables()),
        "total_variables": sum(v.size for v in result.variables()),
        "status": result.status,
        "result": result.value,
        "T": setup.T,
        "no_homes": setup.no_homes,
        "setup_time": setup_time - start_time,
        "dataflow_time": dataflow_time - setup_time,
        "optimizer_time": end_time - dataflow_time
    }
    write(setup.output_path, stats)
    
    # total_consumption = school.consumption_kWh + sum(household.consumption_kWh for household in homes)
    # solar_production = sum(solar_panel.max_power_output_kW for solar_panel in solar_panels_homes) + solar_panel_school.max_power_output_kW

    # print("Total grid import:", sum(metering_point.energy_import.value))
    
    plot_microgrid_solution(
        setup=setup,
        solution=result,
        grid_import=metering_point.energy_import.value,
        # total_demand=[
        #     sum(home.consumption_kWh[t] for home in homes) + school.consumption_kWh[t]
        #     for t in range(setup.T)
        # ],
        # solar_prod=[
        #     sum(solar_panel.max_power_output_kW[t] for solar_panel in solar_panels_homes) + solar_panel_school.max_power_output_kW[t]
        #     for t in range(setup.T)
        # ],
        # wind_prod=[wind_turbine.max_power_output_kW[t] for t in range(setup.T)],
        soc_values=[battery1.SoC, battery2.SoC, battery3.SoC],
        prices = metering_point.prices if hasattr(metering_point, 'prices') else None
    )

if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 24
    n = 50
    setup.no_homes = n
    setup.no_solar_panels = n
    solve_microgrid_gpod(setup=setup)