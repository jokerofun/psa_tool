import cvxpy as cp
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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

import time
# from examples.helpers.plot_microgrid_solution import plot_microgrid_solution

def solve_microgrid_gpod(setup: MicrogridSetup):
    start_time = time.time()
    problemClass = GraphProblemClass("microgrid_problem", time_length=setup.T)

    metering_point = MeteringPoint("grid1")
    homes = [Consumer(f"household_{i+1}") for i in range(setup.no_homes)]
    school = Consumer("school")
    solar_panels_homes = [SolarPanel(f"solar_panel_{i+1}") for i in range(setup.no_homes)]
    solar_panel_school = SolarPanel("solar_panel_school")
    wind_turbine = WindTurbine("wind_turbine")
    microgrid_balance = ConnectingNode("balance")
    batteries = [Battery(f"battery_{i+1}", setup.battery_power, setup.battery_power, setup.battery_capacity, setup.battery_efficiency) for i in range(setup.no_batteries)]
    
    for home in homes:
        home.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 1}, final=True)
    school.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": setup.T, "model_name":"consumer_model", "factor": 100}, final=True)
    for solar_panel in solar_panels_homes:
        task1 = solar_panel.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
        task2 = solar_panel.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_home}, final=True)
        task1 >> task2
    solar_school_task1 = solar_panel_school.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217}) 
    solar_school_task2 = solar_panel_school.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_school}, final=True)
    solar_school_task1 >> solar_school_task2
    wind_task1 = wind_turbine.dataflow.task("get_wind_data", DataProcessingTask, process_func=get_wind_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_task2 = wind_turbine.dataflow.task("gen_wind_data", DataProcessingTask, process_func=generate_wind_turbine_data, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed": 14, "cut_out_speed": 25}, final=True)
    wind_task1 >> wind_task2

    problemClass.add_nodes([*homes, *solar_panels_homes, *batteries, 
                            school, solar_panel_school, wind_turbine, 
                            metering_point, microgrid_balance])

    microgrid_balance.connect_nodes([*homes, *solar_panels_homes, *batteries, school, 
                           solar_panel_school, wind_turbine, metering_point])

    setup_time = time.time()
    result = problemClass.solve(solver=cp.CBC, objective="minimize", value="cost")
    end_time = time.time()
    dataflow_time = problemClass.dataflow_time
    # FROM HERE - DON'T COUNT THESE CHARACTERS FOR PRODUCTIVITY EXPERIMENTS
    stats = {
        "implementation": "GPO-D",
        "solver": result.solver_stats.solver_name,
        "parameters": sum(p.size for p in result.parameters()),
        "constraints": len(result.constraints),
        "variables": sum(v.size for v in result.variables()),
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
    
    # plot_microgrid_solution(
    #     setup.T,
    #     total_consumption=total_consumption,
    #     wind_production=wind_turbine.max_power_output_kW,
    #     solar_production=solar_production,
    #     batteries=batteries
    # )

if __name__ == "__main__":
    setup = MicrogridSetup()
    setup.T = 24
    n = 50
    setup.no_homes = n
    setup.no_solar_panels = n
    solve_microgrid_gpod(setup=setup)