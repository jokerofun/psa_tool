import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.dataflow.dataflow_classes import DataFetchingFromFileNode
from src.dataflow.dataflow_manager import DataFlowManager
from archive.examples.energy_domain_deprecated import PowerExchange
from persistence.db_manager import DBManager


def procFunc1(dfs, parameters = {}):
    # do some processing
    print("Processing data")
    dfs = dfs["results"] 
    return dfs

def trainFunc1(dfs, parameters = {}):
    # do some processing
    print("Training model")
    return dfs

if __name__ == "__main__":
    db_context = DBManager()
    problemClass = db_context.load_optimization_problem_with_nodes("batteryArbitrage")
    power_exchange = problemClass.get_node("powerExchange")
    PE_dataflow = DataFlowManager.getInstance().newDataFlow(PowerExchange)
    
    PE_dataflow.node("results", DataFetchingFromFileNode, "dataflow_manager/test_data/pricesEUR.csv") >> PE_dataflow.node(name="prepoc")
    PE_dataflow.node("csv_prices_dkk", DataFetchingFromFileNode, "dataflow_manager/test_data/pricesDKK.csv") >> PE_dataflow.node(name="prepoc")
    PE_dataflow.node(name="prepoc") >> PE_dataflow.node(name="training", final=True)


    problemClass.setTimeLen(5)
    # override process function for the porcessing nodes
    PE_dataflow.node("prepoc").process_func = procFunc1
    PE_dataflow.node("training").process_func = trainFunc1
    dfs = DataFlowManager.getInstance().getData(PowerExchange)
    print(dfs)
    problemClass.getObjectiveFunction("minimize").values("cost")
    # solve the problem and get the results
    problemClass.solve()
    allVariables = problemClass.getAllVariables()
    print(allVariables)