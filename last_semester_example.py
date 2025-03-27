# import all dataflow needs



# define batteries and power exhanges or load them from db

from dataflow_manager.dataflow_classes import DataFetchingFromFileNode
from dataflow_manager.dataflow_manager import DataFlowManager
from optimization.energy_sector_classes import Battery, PowerExchange
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
    problemClass = GraphProblemClass()
    battery1 = Battery(problemClass, 50, 50, "bat1", 100 )
    battery2 = Battery(problemClass, 100, 100,"bat2",  200)
    battery3 = Battery(problemClass, 150, 150, "bat3", 300)
    power_exchange = PowerExchange(problemClass, 50, 50)

    # Connect the nodes
    power_exchange - battery1
    power_exchange - battery2
    power_exchange - battery3
    
    # define a dataflow for the power exchange 
    PE_dataflow = DataFlowManager.getInstance().newDataFlow(PowerExchange)
    
    PE_dataflow.node("csv_prices", DataFetchingFromFileNode, "dataflow_manager/test_data/pricesEUR.csv") >> PE_dataflow.node(name="prepoc")
    PE_dataflow.node("csv_prices_dkk", DataFetchingFromFileNode, "dataflow_manager/test_data/pricesDKK.csv") >> PE_dataflow.node(name="prepoc")
    PE_dataflow.node(name="prepoc") >> PE_dataflow.node(name="training", final=True)


    problemClass.setTimeLen(5)
    # override process function for the porcessing nodes
    PE_dataflow.node("prepoc").process_func = procFunc1
    PE_dataflow.node("training").process_func = trainFunc1
    dfs = DataFlowManager.getInstance().getData(PowerExchange, 1)
    print(dfs)
    # convert to numpy array
    power_exchange.prices = dfs["priceEUR"].values
    
    problemClass.getObjectiveFunction("minimize").values("cost")
    # solve the problem and get the results
    problemClass.solve()
    allVariables = problemClass.getAllVariables()
    print(allVariables)
    
    