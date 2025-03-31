from dataflow_manager.dataflow_classes import DataFetchingFromFileNode
from dataflow_manager.dataflow_manager import DataFlowManager
from optimization.energy_sector_classes import PowerExchange
from persistence.db_manager import DBManager

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
    db_context = DBManager()
    problemClass = db_context.load_optimization_problem_with_nodes("batteryArbitrage")
    power_exchange = problemClass.get_node("powerExchange")
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