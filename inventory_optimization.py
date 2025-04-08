from matplotlib import pyplot as plt
import numpy as np
from optimization.solver_classes import DeviceNode, GraphProblemClass
import cvxpy as cp


class WarehouseNode(DeviceNode):
    def __init__(self, problem_class, order_capacity, inventory_capacity, ordering_cost, holding_cost):
        super().__init__(problem_class)
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.order_capacity = order_capacity
        self.inventory_capacity = inventory_capacity
        self.name = "Warehouse"

    def setTimeLen(self, time_len):
        self.time_len = time_len
        self.inventory = cp.Variable(time_len, nonneg=True)
        self.orders = cp.Variable(time_len, nonneg=True)

    @property
    def variables(self):
        return [self.inventory, self.orders]
    
    @property
    def cost(self):
        return cp.sum(self.holding_cost * self.inventory + self.ordering_cost * self.orders)

    def powerflow(self, t):
        return -self.shipments[t]

    def constraints(self, t):
        constraints = []

        # Inventory balance
        if t == 0:
            constraints.append(self.inventory[t] == self.orders[t] - self.shipments[t])
        else:
            constraints.append(self.inventory[t] == self.inventory[t-1] + self.orders[t] - self.shipments[t])

        # Capacity constraint
        constraints.append(self.inventory[t] <= self.inventory_capacity)

        # Order limit constraint
        constraints.append(self.orders[t] <= self.order_capacity)

        return constraints
    
    def setShipments(self, shipments):
        self.shipments = shipments

class RetailerNode(DeviceNode):
    def __init__(self, problem_class, demand):
        super().__init__(problem_class)
        self.demand = demand
        self.name = "Retailer"

    def setTimeLen(self, time_len):
        self.time_len = time_len
        self.received = cp.Variable(time_len, nonneg=True)

    @property
    def variables(self):
        return [self.received]
    
    @property
    def cost(self):
        return 0

    def powerflow(self, t):
        return self.received[t]

    def constraints(self, t):
        return [self.received[t] == self.demand[t]]
    



def run_inventory_optimization():
    time_len = 7
    demand_series = [20, 25, 18, 30, 22, 24, 28]

    problem = GraphProblemClass("InventoryOptimization")

    warehouse = WarehouseNode(problem, holding_cost=1.0, ordering_cost=2.0, inventory_capacity=100, order_capacity=25)
    retailer = RetailerNode(problem, demand=demand_series)

    # Connect warehouse and retailer
    warehouse - retailer

    # Shipment variable between warehouse and retailer
    shipment = cp.Variable(time_len, nonneg=True)
    warehouse.setShipments(shipment)
    retailer.received = shipment

    # Set time length for nodes
    problem.setTimeLen(time_len)

    # Define objective (minimize cost)
    problem.getObjectiveFunction("minimize").values("cost")

    # Solve problem
    problem.solve()

    # Print results
    print("Inventory:", np.round(warehouse.inventory.value, 2))
    print("Orders:", warehouse.orders.value)
    print("Shipments:", shipment.value)

    plot_inventory_results(warehouse, retailer, shipment.value, demand_series)

def plot_inventory_results(warehouse, retailer, shipment, demand_series):
    t = np.arange(len(demand_series))
    inventory = warehouse.inventory.value
    orders = warehouse.orders.value
    shipments = shipment
    demand = demand_series

    plt.figure(figsize=(12, 6))

    # Plot Inventory
    plt.subplot(2, 1, 1)
    plt.plot(t, inventory, label="Inventory", marker='o')
    plt.plot(t, orders, label="Orders", marker='x')
    plt.title("Warehouse Inventory and Orders")
    plt.xlabel("Time")
    plt.ylabel("Units")
    plt.grid(True)
    plt.legend()

    # Plot Shipments vs Demand
    plt.subplot(2, 1, 2)
    plt.plot(t, shipments, label="Shipments to Retailer", marker='s')
    plt.plot(t, demand, label="Retailer Demand", linestyle='--', color='red')
    plt.title("Shipments vs Demand")
    plt.xlabel("Time")
    plt.ylabel("Units")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("inventory_plot.png")  # Save the plot as a file
    plt.show()


if __name__ == '__main__':
    run_inventory_optimization()