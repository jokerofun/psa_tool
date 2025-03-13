# I want to solve the optimization problem of solving the battery arbitrage problem. The problem consists of 3 batteries and a power exchange. Use the classes from optimization/solver_classes.

# Import the necessary classes from optimization/solver_classes
from optimization.solver_classes import GraphProblemClass, ConnectingNode, Producer, Consumer, Battery, TransmissionLine, PowerExchange
from dag_manager.dag_classes import DataProcessingNode, DataFetchingFromFileNode, DataFetchingFromDBNode, DataFetchingFromAPINode
import inspect

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

    # Get the prices for the power exchange using the DataNodes
    prices_node = DataFetchingFromFileNode("prices_node", "prices.csv", "some_key") # some key parameter
    prices_df = prices_node.execute(None)

    # Process the prices data
    prices_processing_node = DataProcessingNode("prices_processing_node", "prices_data")
    prices_processing_node.execute = lambda df: (df['price'] * 2).tolist()
    prices = prices_processing_node.execute(prices_df)

    # Set the prices for the power exchange
    # prices = [1, 2, 4, 5, 1]
    power_exchange.setPrices(prices)

    # Set the consumption schedule for the consumer
    consumer.setConsumptionSchedule([5, 2, 8, 10, 8])

    # Solve the optimization problem
    problemClass.solve()

    # print(inspect.signature(prices_processing_node.execute))
    # print(inspect.getsourcelines(prices_processing_node.execute))

    # inspect.getsourcelines

    problemClass.printResults()