from matplotlib import pyplot as plt
import numpy as np
from optimization.solver_classes import Node, GraphProblemClass
import cvxpy as cp

# https://medium.com/@gmarchetti/linear-programming-for-inventory-optimization-64aa674a13cc

class NewsvendorNode(Node):
    def __init__(self, problem_class, unit_cost, holding_cost, backorder_cost, demand_scenarios, demand_probabilities):
        super().__init__(problem_class)
        self.unit_cost = unit_cost
        self.holding_cost = holding_cost
        self.backorder_cost = backorder_cost
        self.demand_scenarios = demand_scenarios
        self.demand_probabilities = demand_probabilities
        self.order_quantity = cp.Variable(nonneg=True)

    @property
    def cost(self):
        expected_cost = 0
        for d, p in zip(self.demand_scenarios, self.demand_probabilities):
            holding = cp.pos(self.order_quantity - d) * self.holding_cost
            backorder = cp.pos(d - self.order_quantity) * self.backorder_cost
            expected_cost += p * (holding + backorder)
        return self.unit_cost * self.order_quantity + expected_cost

    @property
    def variables(self):
        return [self.order_quantity]
    
    def setTimeLen(self, time_len):
        self.time_len = time_len


def plot_stuff(newsvendor_node):
    # Access the optimal order quantity
    optimal_order = newsvendor_node.order_quantity.value
    print(f"Optimal order quantity: {optimal_order:.2f}")

    # Optional: Plot expected cost vs order quantity for a range
    order_quantities = np.linspace(min(newsvendor_node.demand_scenarios) - 10,
                                max(newsvendor_node.demand_scenarios) + 10,
                                100)
    expected_costs = []

    for x in order_quantities:
        total_cost = newsvendor_node.unit_cost * x
        for d, p in zip(newsvendor_node.demand_scenarios, newsvendor_node.demand_probabilities):
            holding = max(x - d, 0) * newsvendor_node.holding_cost
            backorder = max(d - x, 0) * newsvendor_node.backorder_cost
            total_cost += p * (holding + backorder)
        expected_costs.append(total_cost)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(order_quantities, expected_costs, label="Expected Total Cost")
    plt.axvline(optimal_order, color='red', linestyle='--', label=f'Optimal Order = {optimal_order:.2f}')
    plt.xlabel("Order Quantity")
    plt.ylabel("Expected Cost")
    plt.title("Newsvendor Problem - Optimal Inventory Decision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("newsvendor_opt.png")  # Save the plot as a file
    plt.show()


if __name__ == "__main__":
    # Define problem parameters
    unit_cost = 2.0
    holding_cost = 0.5
    backorder_cost = 2.5
    demand_scenarios = [15, 60, 72, 78, 82]
    demand_probabilities = [0.2, 0.3, 0.1, 0.1, 0.3]

    # Initialize the problem
    inventory_problem = GraphProblemClass("Newsvendor Problem")

    # Add the newsvendor node
    newsvendor_node = NewsvendorNode(inventory_problem, unit_cost, holding_cost, backorder_cost, demand_scenarios, demand_probabilities)

    inventory_problem.setTimeLen(5)

    # Define the objective function
    selector = inventory_problem.getObjectiveFunction("minimize").values("cost")
    # selector.add(newsvendor_node.cost)

    # Solve the problem
    inventory_problem.solve()

    # Print the results
    inventory_problem.printResults()
    # print(newsvendor_node.order_quantity.value)
    plot_stuff(newsvendor_node)
