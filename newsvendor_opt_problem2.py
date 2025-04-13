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

        plt.figure(figsize=(8, 5))
        plt.plot(q_range, expected_profits, label="Expected Profit")
        plt.axvline(x=self.Q.value, color='r', linestyle='--', label=f"Optimal Q = {self.Q.value:.2f}")
        plt.title("Expected Profit vs Order Quantity")
        plt.xlabel("Order Quantity (Q)")
        plt.ylabel("Expected Profit")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("newsvendor_opt2.png")  # Save the plot as a file
        plt.show()



if __name__ == "__main__":
    scenarios = [
        {"demand": 700, "prob": 0.450},
        {"demand": 800, "prob": 0.300},
        {"demand": 900, "prob": 0.220},
        {"demand": 1000, "prob": 0.015},
        {"demand": 1100, "prob": 0.010},
    ]

    costs = {"cost": 0.005}  # per unit cost
    price = 0.08  # selling price

    problem = GraphProblemClass("NewsvendorProblem")
    node = NewsvendorNode(problem, scenarios, costs, price)
    problem.getObjectiveFunction("maximize").values("cost")
    problem.setTimeLen(1)
    problem.solve()
    problem.printResults()
    node.plot_profit_curve()
