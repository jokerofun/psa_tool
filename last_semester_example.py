# import all dataflow needs



# define batteries and power exhanges or load them from db

import numpy as np
from dataflow_manager.dataflow_classes import DataFetchingFromFileNode
from dataflow_manager.dataflow_manager import DataFlowManager
from archive.lib_descriptive import plot_battery_arbitrage_multiple
from optimization.energy_sector_classes import Battery, PowerExchange, TransmissionLine
from optimization.solver_classes import GraphProblemClass

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
    problemClass = GraphProblemClass("graph_problem")
    battery1 = Battery(problemClass, 50, 50, "bat1", 100 )
    battery2 = Battery(problemClass, 100, 100,"bat2",  200)
    battery3 = Battery(problemClass, 150, 150, "bat3", 300)
    power_exchange = PowerExchange(problemClass, 350, 350, "power_exchange")
    # transmission_line = TransmissionLine(problem_class=problemClass, capacity=100, transmission_loss=0.1)

    # Connect the nodes
    power_exchange - battery1
    power_exchange - battery2
    power_exchange - battery3
    # battery3 - transmission_line
    
    # define a dataflow for the power exchange 
    PE_dataflow = DataFlowManager.getInstance().newDataFlow(PowerExchange)
    
    PE_dataflow.node("csv_prices", DataFetchingFromFileNode, "dataflow_manager/test_data/pricesEUR.csv") >> PE_dataflow.node(name="prepoc")
    PE_dataflow.node("csv_prices_dkk", DataFetchingFromFileNode, "dataflow_manager/test_data/pricesDKK.csv") >> PE_dataflow.node(name="prepoc")
    PE_dataflow.node(name="prepoc") >> PE_dataflow.node(name="training", final=True)


    problemClass.set_time_len(5)
    # override process function for the porcessing nodes
    PE_dataflow.node("prepoc").process_func = procFunc1
    PE_dataflow.node("training").process_func = trainFunc1
    dfs = DataFlowManager.getInstance().getData(PowerExchange, 1)
    print(dfs)
    # convert to numpy array
    power_exchange.prices = dfs["csv_prices"].values.flatten()
    
    # abad = problemClass.getObjectiveFunction("minimize")
    # grrer = abad.values("cost")
    
    print(problemClass._nodes)
    # solve the problem and get the results
    problemClass.solve()
    allVariables = problemClass.getAllVariables()
    print(allVariables)



    # # For each battery, get the state of charge and the power flow and store it as ndimensioal array
    soc = []
    power_flow = []

    for item in allVariables:
        if isinstance(item, dict):  # Ensure the item is a dictionary
            for key, value in item.items():
                if 'SOC' in value and 'powerFlow' in value:  # Check if the key contains SOC and powerFlow
                    soc.append(value['SOC'])
                    power_flow.append(-value['powerFlow'])

    soc = np.array(soc)
    power_flow = np.array(power_flow)
    plot_battery_arbitrage_multiple(power_exchange.prices, soc, power_flow, 3)
    
    