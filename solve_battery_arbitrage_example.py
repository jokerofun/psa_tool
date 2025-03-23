import time
import numpy as np
from lib_descriptive import plot_battery_arbitrage_multiple
from optimization.solver_classes import GraphProblemClass, Battery, PowerExchange
from dataflow_manager.dataflow_classes import DataProcessingNode, DataFetchingFromFileNode

if __name__ == "__main__":
    start_time = time.time()
    problemClass = GraphProblemClass()

    num_batteries = 6

    battery1 = Battery(problemClass, 50, 50, 100)
    battery2 = Battery(problemClass, 100, 100, 200)
    battery3 = Battery(problemClass, 150, 150, 300)
    battery4 = Battery(problemClass, 505, 502, 1007)
    battery5 = Battery(problemClass, 100, 100, 2009)
    battery6 = Battery(problemClass, 150, 150, 3000)
    power_exchange = PowerExchange(problemClass, 50, 50)

    # Connect the nodes
    power_exchange - battery1
    power_exchange - battery2
    power_exchange - battery3
    power_exchange - battery4
    power_exchange - battery5
    power_exchange - battery6

    dfs = {}

    # Get the prices for the power exchange using the DataNodes
    prices_node = DataFetchingFromFileNode("prices", "data/future_prices.csv")
    dfs = prices_node.execute(dfs)

    # Process the prices data
    prices_processing_node = DataProcessingNode("prices_processed", dfs["prices"])
    prices_processing_node.process = lambda df: (df.drop(columns=["HourUTC", "PriceArea", "SpotPriceDKK"])).sort_values(by="HourDK")
    dfs = prices_processing_node.execute(dfs)

    prices = dfs["prices_processed"]["SpotPriceEUR"].tolist()

    # Set the prices for the power exchange
    power_exchange.setPrices(prices)

    # Set the time length for the optimization problem
    problemClass.setTimeLen(dfs["prices_processed"].shape[0])

    # Solve the optimization problem
    problemClass.solve()

    end_time = time.time()  # End the timer
    print(f"Execution time: {end_time - start_time:.2f} seconds")  # Print the elapsed time

    allVariables = problemClass.getAllVariables()

    # For each battery, get the state of charge and the power flow and store it as ndimensioal array
    soc = []
    power_flow = []
    for i in range(num_batteries):
        soc.append(allVariables[i]['Battery']['SOC'])
        power_flow.append(allVariables[i]['Battery']['powerFlow'])

    soc = np.array(soc)
    power_flow = np.array(power_flow)
    plot_battery_arbitrage_multiple(prices, soc, power_flow, num_batteries)

    problemClass.printResults()