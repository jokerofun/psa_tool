from src.dataflow.dataflow_classes import DataFetchingFromFileNode
from src.dataflow.dataflow_manager import DataFlowManager
from archive.examples.energy_generation.domain import Generator, Area
from archive.solver_classes_deprecated import GraphProblemClass

import matplotlib.pyplot as plt

def procFunc1(dfs):
    print("Processing data")
    dfs = dfs["future_demand"] 
    return dfs

def trainFunc1(dfs):
    print("Training future demand prediction model")
    return dfs

def plot_energy_generation_demo(area: Area):
    # Plot results
    plt.figure(figsize=(14, 6))
    for gen in area1.generators:
        plt.plot(range(24), gen.power_output.value, label=gen.name)
    plt.plot(range(24), area1.future_hourly_demand, 'k--', label="Demand", linewidth=1.5)
    plt.title("Optimal Generation Schedule")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/energy_generation_schedule.png")
    plt.show()

if __name__ == "__main__":
    problemClass = GraphProblemClass("energy_generation_optimization_demo1")
    generator1 = Generator("Producer1", 50, 10, 100)
    generator2 = Generator("Producer2", 60, 0, 60)
    generator3 = Generator("Producer3", 70, 20, 80)
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
    value = problemClass.solve()
    result = problemClass.getAllVariables()
    print(result)
    print(f"Total generation cost over 24 hours: ${value:.2f}")

    plot_energy_generation_demo(area1)