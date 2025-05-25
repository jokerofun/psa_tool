from src.optimization.base_domain import GraphProblemClass
from src.optimization.energy_domain import ConnectingNode
from microgrid_domain import Consumer, Grid, SolarPanel, Battery, WindTurbine
from src.dataflow.dataflow_classes import DataFetchingFromFileNode
from src.dataflow.dataflow_manager import DataFlowManager
import numpy as np

def procFunc1(dfs):
    # do some processing
    print("Processing data")
    dfs = dfs["csv_prices"] 
    return dfs

def trainFunc1(dfs):
    # do some processing
    print("Training model")
    return dfs

if __name__ == "__main__":
    # Define the parameters for the microgrid
    T = 24 # time segments
    no_households = 50
    solar_capacity_household = 5 # kW
    solar_capacity_school = 25 # kW
    wind_capacity = 50 # kW
    no_batteries = 1
    battery_capacity = 500 # kWh
    battery_power = 500 # kW
    battery_efficiency = 0.9 # in %

    # Generate random data for microgrid components
    np.random.seed(42)
    household_consumption = np.random.uniform(3, 5, (no_households, T)) # kWh
    school_consumption = np.random.uniform(30, 50, T) # kWh
    solar_profile = np.clip(np.sin(np.linspace(0, np.pi, T)), 0, None) # kWh
    wind_profile = np.clip(np.sin(np.linspace(0, 2 * np.pi, T) - 1), 0, None) # kWh

    # Create optimization problem class
    problemClass = GraphProblemClass("microgrid_problem")

    # Create microgrid models/components
    grid = Grid("grid1")
    household_consumers = [Consumer(f"household_{i}", household_consumption[i]) for i in range(no_households)]
    school = Consumer("school", school_consumption)
    solar_panels_household = [SolarPanel(f"solar_panel_{i}", solar_capacity_household * solar_profile) for i in range(no_households)]
    solar_panel_school = SolarPanel("solar_panel_school", solar_capacity_school * solar_profile)
    wind_turbine = WindTurbine("wind_turbine", wind_capacity * wind_profile)
    balance = ConnectingNode("balance")
    batteries = [Battery(f"battery_{i}", battery_power, battery_power, battery_capacity, battery_efficiency) for i in range(no_batteries)]
    
    # add components to the problem class
    problemClass.add_nodes(household_consumers)
    problemClass.add_nodes(solar_panels_household)
    problemClass.add_nodes([school, solar_panel_school])
    problemClass.add_node(wind_turbine)
    problemClass.add_nodes(batteries)
    problemClass.add_node(grid)
    problemClass.add_node(balance)

    balance.connect_nodes(household_consumers)
    balance.connect_nodes(solar_panels_household)
    balance.connect_nodes([grid, school, solar_panel_school, wind_turbine])
    balance.connect_nodes(batteries)

    problemClass.set_time_length(T)

    # problemClass.get_objective_function("minimize").values("cost")
    # result = problemClass.solve()
    # allVariables = problemClass.get_all_variables()
    # print(allVariables)
    print(household_consumers[10].name)