from matplotlib import pyplot as plt
import numpy as np
from optimization.solver_classes import DeviceNode, Node, GraphProblemClass
import cvxpy as cp

# https://optimization.cbe.cornell.edu/index.php?title=Newsvendor_problem

class NewsvendorNode(DeviceNode):
    def __init__(self, problem_class, scenarios, costs, price, name="Newsvendor"):
        super().__init__(problem_class)
        self.name = name
        self.scenarios = scenarios  # List of demand scenarios
        self.probabilities = [s["prob"] for s in scenarios]
        self.demands = [s["demand"] for s in scenarios]
        self.c = costs["cost"]  # ordering cost per unit
        self.p = price  # selling price per unit
        self.Q = cp.Variable(name="Q", nonneg=True) # order quantity
        self.holding = [cp.Variable(name=f"h_{i}", nonneg=True) for i in range(len(scenarios))] # unsold inventory in scenario
        self.shortage = [cp.Variable(name=f"s_{i}", nonneg=True) for i in range(len(scenarios))] # unmet demand in scenario

    @property
    def variables(self):
        return [self.Q] + self.holding + self.shortage

    @property
    def cost(self): # total expected cost
        expected_cost = self.c * self.Q
        expected_loss = sum(
            self.probabilities[i] * (
                self.holding[i] * (self.p - self.c) + self.shortage[i] * self.p
            )
            for i in range(len(self.scenarios))
        )
        return expected_cost - expected_loss  # maximize profit, or minimize -profit

    def constraints(self, t):
        cons = []
        for i, d in enumerate(self.demands):
            cons.append(self.holding[i] >= self.Q - d)
            cons.append(self.shortage[i] >= d - self.Q)
        return cons

    def powerflow(self, t=0):
        return 0  # Not applicable here, just to satisfy interface
    
    def setTimeLen(self, time_len):
        self.time_len = time_len

    def plot_profit_curve(self):
        q_range = np.linspace(0, max(self.demands) * 1.5, 100)
        expected_profits = []

        for q in q_range:
            profit = -self.c * q
            for i, d in enumerate(self.demands):
                sold = min(q, d)
                unsold = max(q - d, 0)
                unmet = max(d - q, 0)
                profit += self.probabilities[i] * (self.p * sold - self.c * q)
            expected_profits.append(profit)

        plt.plot(q_range, expected_profits, label=f"Expected Profit for {self.name}")
        plt.axvline(x=self.Q.value, color='r', linestyle='--', label=f"Optimal Q for {self.name} = {self.Q.value:.2f}")
        plt.xlabel("Order Quantity (Q)")
        plt.ylabel("Expected Profit")
        plt.grid(True)

class ClusterNode(Node):
    def __init__(self, problem_class, budget=None, supplier_capacity=None):
        super().__init__(problem_class)
        self.connected_nodes = []
        self.name = "ClusterNode"
        self.budget = budget
        self.supplier_capacity = supplier_capacity

    def connect(self, node):
        self.connected_nodes.append(node)

    def constraints(self, t):
        cons = []

        if self.budget is not None:
            total_cost = cp.sum([node.Q * node.c for node in self.connected_nodes])
            cons.append(total_cost <= self.budget)

        if self.supplier_capacity is not None:
            total_units = cp.sum([node.Q for node in self.connected_nodes])
            cons.append(total_units <= self.supplier_capacity)

        return cons
    
    def setTimeLen(self, time_len):
        self.time_len = time_len

if __name__ == "__main__":
    scenarios1 = [
        {"demand": 700, "prob": 0.450},
        {"demand": 800, "prob": 0.300},
        {"demand": 900, "prob": 0.220},
        {"demand": 1000, "prob": 0.015},
        {"demand": 1100, "prob": 0.010},
    ]
    scenarios2 = [
        {"demand": 700, "prob": 0.450},
        {"demand": 800, "prob": 0.300},
        {"demand": 900, "prob": 0.220},
        {"demand": 1000, "prob": 0.015},
        {"demand": 1100, "prob": 0.010},
    ]
    scenarios3 = [
        {"demand": 700, "prob": 0.450},
        {"demand": 800, "prob": 0.300},
        {"demand": 900, "prob": 0.220},
        {"demand": 1000, "prob": 0.015},
        {"demand": 1100, "prob": 0.010},
    ]

    costs1 = {"cost": 0.005}  # per unit cost
    price1 = 0.08  # selling price
    costs2 = {"cost": 0.005}  # per unit cost
    price2 = 0.08  # selling price
    costs3 = {"cost": 0.005}  # per unit cost
    price3 = 0.08  # selling price

    problem = GraphProblemClass("NewsvendorProblem")
    connection = ClusterNode(None, budget=1000, supplier_capacity=500)
    product1 = NewsvendorNode(None, scenarios1, costs1, price1)
    product2 = NewsvendorNode(None, scenarios2, costs2, price2)
    product3 = NewsvendorNode(None, scenarios3, costs3, price3)
    connection.connect(product1)
    connection.connect(product2)
    connection.connect(product3)
    product1.setConnectingNode(connection)
    product2.setConnectingNode(connection)
    product3.setConnectingNode(connection)
    problem.add_nodes([product1, product2, product3, connection])
    problem.getObjectiveFunction("maximize").values("cost")
    problem.setTimeLen(1)
    problem.solve()
    problem.printResults()
    
    # Plot the profit curves for each product
    plt.figure(figsize=(10, 6))
    product1.plot_profit_curve()
    product2.plot_profit_curve()
    product3.plot_profit_curve()

    # Display combined plot
    plt.title("Expected Profit vs Order Quantity for Multiple Products")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"newsvendor_opt3.png")  # Save the plot as a file
