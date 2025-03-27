# import all dataflow needs



# define batteries and power exhanges or load them from db

from dataflow_manager.dataflow_manager import DataFlowManager
from optimization.energy_sector_classes import Battery, PowerExchange
from optimization.solver_classes import GraphProblemClass


if __name__ == "__main__":
    problemClass = GraphProblemClass()
    battery1 = Battery(problemClass, 50, 50, 100)
    battery2 = Battery(problemClass, 100, 100, 200)
    battery3 = Battery(problemClass, 150, 150, 300)
    power_exchange = PowerExchange(problemClass, 50, 50)

    # Connect the nodes
    power_exchange - battery1
    power_exchange - battery2
    power_exchange - battery3
    
    # define a dataflow for the power exchange 
    PE_dataflow = DataFlowManager.newDataFlow(PowerExchange)
    
    PE_dataflow.datafromcsvNode(name="csv_prices") >> PE_dataflow.processingNode(name="prepoc")
    PE_dataflow.datafromAPINOde(name="weather") >> PE_dataflow.processingNode(name="prepoc")
    PE_dataflow.processingNode(name="prepoc") >> PE_dataflow.processingNode(name="training", final=True)
    
    # override process function for the porcessing nodes
    PE_dataflow.node("prepoc").process = procFunc1
    PE_dataflow.node("training").process = trainFunc1
    
    # solve the problem and get the results
    problemClass.solve()
    allVariables = problemClass.getAllVariables()
    
    
    