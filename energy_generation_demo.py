from dataflow_manager.dataflow_classes import DataFetchingFromFileNode
from dataflow_manager.dataflow_manager import DataFlowManager
from examples.energy_generation import Generator, Area
from optimization.solver_classes import GraphProblemClass

def procFunc1(dfs):
    print("Processing data")
    dfs = dfs["future_demand"] 
    return dfs

def trainFunc1(dfs):
    print("Training future demand prediction model")
    return dfs

if __name__ == "__main__":
    problemClass = GraphProblemClass("energy_generation_optimization_demo1")
    generator1 = Generator("Generator1", 50, 10, 100)
    generator2 = Generator("Generator2", 60, 0, 60)
    generator3 = Generator("Generator3", 70, 20, 80)
    area1 = Area("DK0")
    area1.add_generators([generator1, generator2, generator3])
    problemClass.add_nodes([area1, generator1, generator2, generator3])

    dataflow = DataFlowManager.getInstance().newDataFlow(Area)

    dataflow.node("future_demand", DataFetchingFromFileNode, "examples/test_data/future_demand.csv") >> dataflow.node(name="prepoc")
    dataflow.node(name="prepoc") >> dataflow.node(name="training", final=True)

    problemClass.setTimeLen(24)
    dataflow.node("prepoc").process_func = procFunc1
    dataflow.node("training").process_func = trainFunc1
    future_demnad_df = DataFlowManager.getInstance().getData(Area, 1)
    print(future_demnad_df)
    area1.future_hourly_demand = future_demnad_df["future_demand"].values.flatten()

    print(problemClass._nodes)
    problemClass.getObjectiveFunction("minimize").values("cost")
    problemClass.solve()
    result = problemClass.getAllVariables()
    print(result)