from src.optimization.graph_problem_class import GraphProblemClass
from src.optimization.energy_domain import ConnectingNode
from microgrid_domain import Consumer, Grid, SolarPanel, Battery, WindTurbine
from src.dataflow.dataflow_classes_v2 import DataFetchingFromFileTask, DataProcessingTask
from src.dataflow.dataflow_manager_v2 import DataflowManager
from src.dataflow.dataflow_v2 import Dataflow
import numpy as np
from src.dataflow.default_tasks import *
from examples.dataflow_nodes.consumer_data_pred import predict_consumer_data
from examples.dataflow_nodes.consumer_mock_data_gen import generate_consumption_data
from examples.dataflow_nodes.openmeteo_irradiation import get_irradiation_data
from examples.dataflow_nodes.openmeteo_wind import get_wind_data
from examples.dataflow_nodes.solar_panel_generation import generate_solar_panel_data
from examples.dataflow_nodes.wind_turbine_generation import generate_wind_turbine_data
from examples.psa_examples.microgrid_setup import MicrogridSetup
import cvxpy as cp

def plot_results(t, total_consumption, wind_production, solar_production, batteries):
    import matplotlib.pyplot as plt
    # for battery in batteries:
    #     plt.plot(battery.SoC[:t].value[:-1], label=f'Battery {battery.name} SoC', linestyle='--')
    plt.plot(total_consumption[:t], label='Total Consumption', color='red')
    plt.plot(wind_production[:t], label='Wind Production', color='blue')
    plt.plot(solar_production[:t], label='Solar Production', color='orange')
    plt.plot(wind_production[:t] + solar_production[:t], label='Total Production', color='green')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.title('Microgrid Power Flow - Consumption vs. Production')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig("microgrid_result.png")

if __name__ == "__main__":
    # Define the parameters for the microgrid
    setup = MicrogridSetup()

    # Create optimization problem class
    problemClass = GraphProblemClass("microgrid_problem", time_length=setup.T)

    # Create microgrid models/components
    grid = Grid("grid1")
    household_consumers = [Consumer(f"household_{i+1}") for i in range(setup.no_households)]
    school = Consumer("school")
    solar_panels_household = [SolarPanel(f"solar_panel_{i+1}") for i in range(setup.no_households)]
    solar_panel_school = SolarPanel("solar_panel_school")
    wind_turbine = WindTurbine("wind_turbine")
    balance = ConnectingNode("balance")
    batteries = [Battery(f"battery_{i+1}", setup.battery_power, setup.battery_power, setup.battery_capacity, setup.battery_efficiency) for i in range(setup.no_batteries)]
    
    for household in household_consumers:
        # household.dataflow.task("gen_consumption", DataProcessingTask, process_func=generate_consumption_data, parameters={"A0": 1, "A1": 3, "A2": 2, "phi0": 3, "phi1": 9}, final=True)
        household.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": 24, "model_name":"consumer_model", "factor": 1}, final=True)
    school_task = school.dataflow.task("gen_consumption", DataProcessingTask, process_func=predict_consumer_data, parameters={"hours": 24, "model_name":"consumer_model", "factor": 100}, final=True)
    for solar_panel in solar_panels_household:
        task1 = solar_panel.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
        task2 = solar_panel.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_home}, final=True)
        task1 >> task2
    solar_school_task1 = solar_panel_school.dataflow.task("get_solar_data", DataProcessingTask, process_func=get_irradiation_data, parameters={"latitude": 57.0488, "longitude": 9.9217}) 
    solar_school_task2 = solar_panel_school.dataflow.task("gen_solar_data", DataProcessingTask, process_func=generate_solar_panel_data, parameters={"rated_power": setup.solar_capacity_school}, final=True)
    solar_school_task1 >> solar_school_task2
    wind_task1 = wind_turbine.dataflow.task("get_wind_data", DataProcessingTask, process_func=get_wind_data, parameters={"latitude": 57.0488, "longitude": 9.9217})
    wind_task2 = wind_turbine.dataflow.task("gen_wind_data", DataProcessingTask, process_func=generate_wind_turbine_data, parameters={"rated_power": setup.wind_capacity, "cut_in_speed": 3.5, "rated_speed": 14, "cut_out_speed": 25}, final=True)
    wind_task1 >> wind_task2

    # add components to the problem class
    problemClass.add_nodes(household_consumers)
    problemClass.add_nodes(solar_panels_household)
    problemClass.add_nodes([school, solar_panel_school, wind_turbine])
    problemClass.add_nodes(batteries)
    problemClass.add_node(grid)
    problemClass.add_node(balance)

    balance.connect_nodes(household_consumers)
    balance.connect_nodes(solar_panels_household)
    balance.connect_nodes([school, solar_panel_school, wind_turbine])
    balance.connect_nodes([grid])
    balance.connect_nodes(batteries)

    result = problemClass.solve(solver=cp.CBC,objective="minimize", value="cost")

    total_consumption = school.consumption_kWh + sum(household.consumption_kWh for household in household_consumers)
    solar_production = sum(solar_panel.max_power_output_kW for solar_panel in solar_panels_household) + solar_panel_school.max_power_output_kW

    print("Total grid import:", sum(grid.energy_import.value))
    
    plot_results(
        setup.T,
        total_consumption=total_consumption,
        wind_production=wind_turbine.max_power_output_kW,
        solar_production=solar_production,
        batteries=batteries
    )