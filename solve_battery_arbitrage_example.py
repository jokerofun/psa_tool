import numpy as np
from lib_descriptive import plot_battery_arbitrage_multiple
from optimization.solver_classes import GraphProblemClass, Battery, PowerExchange
from dataflow_manager.dataflow_classes import DataProcessingNode, DataFetchingFromFileNode
import dataflow_manager.dataflow_manager as manager

if __name__ == "__main__":
    problemClass = GraphProblemClass()

    num_batteries = 3

    battery1 = Battery(problemClass, 50, 50, 100)
    battery2 = Battery(problemClass, 100, 100, 200)
    battery3 = Battery(problemClass, 150, 150, 300)
    power_exchange = PowerExchange(problemClass, 50, 50)

    # Connect the nodes
    power_exchange - battery1
    power_exchange - battery2
    power_exchange - battery3

    # Get the prices for the power exchange using the DataNodes
    prices_node = DataFetchingFromFileNode("prices", "data/future_prices.csv")

    # Process the prices data
    prices_processing_node = DataProcessingNode("prices_processed", None, "prices")
    prices_processing_node.process = lambda df: (df.drop(columns=["HourUTC", "PriceArea", "SpotPriceDKK"])).sort_values(by="HourDK")

    dfs = manager.execute()

    prices = dfs["prices_processed"]["SpotPriceEUR"].tolist()

    # Set the prices for the power exchange
    power_exchange.setPrices(prices)

    # Set the time length for the optimization problem
    problemClass.setTimeLen(dfs["prices_processed"].shape[0])

    # # Solve the optimization problem
    # problemClass.solve()

    # # Get output from optimization
    # allVariables = problemClass.getAllVariables()

    # # For each battery, get the state of charge and the power flow and store it as ndimensioal array
    # soc = []
    # power_flow = []
    # for i in range(num_batteries):
    #     soc.append(allVariables[i]['Battery']['SOC'])
    #     power_flow.append(allVariables[i]['Battery']['powerFlow'])

    # soc = np.array(soc)
    # power_flow = np.array(power_flow)
    # plot_battery_arbitrage_multiple(prices, soc, power_flow, num_batteries)

    # problemClass.printResults()