# Import the necessary classes from optimization/solver_classes
from optimization.solver_classes import GraphProblemClass, ConnectingNode, Producer, Consumer, Battery, TransmissionLine, PowerExchange
from dataflow_manager.dataflow_classes import DataProcessingNode, DataFetchingFromFileNode

if __name__ == "__main__":
    # Create a GraphProblemClass instance
    problemClass = GraphProblemClass()

    # Create ConnectingNode instances
    node1 = ConnectingNode(problemClass)
    node2 = ConnectingNode(problemClass)

    # Create Producer, Consumer, and Battery instances
    # Producer and Consumer instances are not needed for this problem
    producer = Producer(problemClass, node1, 5, 2)
    consumer = Consumer(problemClass, node2)
    battery1 = Battery(problemClass, node2, 5, 5, 10)
    battery2 = Battery(problemClass, node2, 5, 5, 10)
    battery3 = Battery(problemClass, node2, 5, 5, 10)

    # Create TransmissionLine and PowerExchange instances
    transmission_line = TransmissionLine(problemClass, node1, node2, 5)
    power_exchange = PowerExchange(problemClass, node1, 10, 10)

    # Set the time length for the optimization problem
    problemClass.setTimeLen(5)

    dfs = {}

    # Get the prices for the power exchange using the DataNodes
    prices_node = DataFetchingFromFileNode("prices", "prices.csv")
    dfs = prices_node.execute(dfs)

    # Process the prices data
    prices_processing_node = DataProcessingNode("prices_processed", dfs["prices"])
    prices_processing_node.process = lambda df: (df['price'] * 2)
    dfs = prices_processing_node.execute(dfs)

    # Set the prices for the power exchange
    power_exchange.setPrices(dfs["prices_processed"].tolist())

    # Set the consumption schedule for the consumer
    consumer.setConsumptionSchedule([5, 2, 8, 10, 8])

    # Solve the optimization problem
    problemClass.solve()

    # problemClass.printResults()