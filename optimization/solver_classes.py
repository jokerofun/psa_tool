import cvxpy as cp
import math
import numpy as np


class BaseSolverClass():
    pass

class GraphProblemClass():
    def __init__(self):
        self.nodes = []
    
    def add_node(self, node):
        self.nodes.append(node)

    def collectCosts(self):
        return cp.sum([node.getCost() for node in self.nodes])
    
    def collectConstraints(self, t):
        constraints = []
        for node in self.nodes:
            constraints.extend(node.getConstraints(t))
        # print(constraints)
        return constraints
    
    def solve(self):
        objective = cp.Maximize(self.collectCosts())
        constraints = []
        for t in range(self.time_len):
            constraints.extend(self.collectConstraints(t))
        problem = cp.Problem(objective, constraints)
        problem.solve()

    def printResults(self):
        for node in self.nodes:
            print([print(var) for var in node.getVariables()])

    def setTimeLen(self, time_len):
        self.time_len = time_len
        for node in self.nodes:
            node.setTimeLen(time_len)

class Node():
    def __init__(self, problem_class : GraphProblemClass):
        self.problem_class = problem_class
        problem_class.add_node(self)

    def getConstraints(self):
        return []
    
    def getVariables(self):
        return []
    
    def getCost(self):
        return 0
    
class ConnectingNode(Node):
    def __init__(self, problem_class):
        super().__init__(problem_class)
        self.connected_nodes = []

    def connect(self, node):
        self.connected_nodes.append(node)

    def setTimeLen(self, time_len):
        self.time_len = time_len

    def getConstraints(self, t):
        return [cp.sum([node.getPowerFlow(self, t) for node in self.connected_nodes]) == 0]
    
## find a better name for this class, because it is not a leaf     
class DeviceNode(Node):
    def __call__(self, problem_class):
        return super().__init__(problem_class)
    
    def getPowerFlow(self, connecting_node : ConnectingNode):
        return 0
    
    def getCost(self):
        return 0
    
    def getVariables(self):
        return []
        
class TransmissionLine(DeviceNode):
    def __init__(self, problem_class, connecting_node_1 : ConnectingNode, connecting_node_2 : ConnectingNode, capacity, transmission_loss = 0):
        super().__init__(problem_class)
        # direction is 1 -> 2
        self.connecting_node_1 = connecting_node_1
        self.connecting_node_2 = connecting_node_2
        connecting_node_1.connect(self)
        connecting_node_2.connect(self)
        self.transmission_loss = transmission_loss
        self.capacity = capacity  

    def setTimeLen(self, time_len):
        self.time_len = time_len
        self.power_flow = cp.Variable(time_len)

    def getConstraints(self, t):
        return [
            cp.pos(self.power_flow[t]) <= self.capacity,
            cp.neg(self.power_flow[t]) <= self.capacity
        ] 
    
    def getPowerFlow(self, connecting_node : ConnectingNode, t):
        if connecting_node == self.connecting_node_1:
            return cp.pos(self.power_flow[t]) * (1 - self.transmission_loss) + cp.neg(self.power_flow[t])
        elif connecting_node == self.connecting_node_2:
            return cp.neg(self.power_flow[t]) * (1 - self.transmission_loss) + cp.pos(self.power_flow[t])
        else:
            return 0
        
    def getVariables(self):
        return [self.power_flow]


class Producer(DeviceNode):
    def __init__(self, problem_class, connecting_node: ConnectingNode, production_capacity, price = 10):
        super().__init__(problem_class)
        self.connecting_node = connecting_node
        connecting_node.connect(self)
        self.production_capacity = production_capacity
        self.price = price

    def setTimeLen(self, time_len):
        self.production_schedule = cp.Variable(time_len, nonneg=True)

    def getConstraints(self, t):
        return [
            self.production_schedule[t] <= self.production_capacity
        ]
    
    def getPowerFlow(self, connecting_node, t):
        return self.production_schedule[t]
    
    def getCost(self):
        return cp.sum(self.production_schedule * self.price)
    
    def getVariables(self):
        return [self.production_schedule]
    
class Consumer(DeviceNode):
    def __init__(self, problem_class, connecting_node: ConnectingNode):
        super().__init__(problem_class)
        self.connecting_node = connecting_node
        connecting_node.connect(self)

    def setTimeLen(self, time_len):
        self.consumption_schedule = cp.Parameter(time_len, nonneg=True)

    def setConsumptionSchedule(self, consumption_schedule):
        self.consumption_schedule.value = consumption_schedule

    def getConstraints(self, t):
        return []

    def getPowerFlow(self, connecting_node, t):
        return -self.consumption_schedule[t]

class Prosumer(DeviceNode):
    def __init__(self, problem_class, connecting_node: ConnectingNode, production_capacity, consumption_capacity):
        super().__init__(problem_class)
        self.connecting_node = connecting_node
        connecting_node.connect(self)
        self.production_capacity = production_capacity
        self.consumption_capacity = consumption_capacity

class PowerExchange(Prosumer):
    def __init__(self, problem_class, connecting_node: ConnectingNode, production_capacity, consumption_capacity):
        super().__init__(problem_class, connecting_node, production_capacity, consumption_capacity)

    def setTimeLen(self, time_len):
        self.powerFlow = cp.Variable(time_len)

    def setPrices(self, prices : list):
        self.prices = prices

    def getConstraints(self, t):
        return [self.powerFlow[t] <= self.production_capacity, self.powerFlow[t] >= -self.consumption_capacity]

    def getPowerFlow(self, connecting_node, t):
        return self.powerFlow[t]
    
    def getCost(self):
        return cp.sum(self.prices @ self.powerFlow)
    
    def getVariables(self):
        return [self.powerFlow]
    
class Battery(Prosumer):
    def __init__(self, problem_class, connecting_node: ConnectingNode, production_capacity, consumption_capacity, battery_capacity, efficiency = 0.9):
        super().__init__(problem_class, connecting_node, production_capacity, consumption_capacity)
        self.battery_capacity = battery_capacity
        self.efficiency = efficiency

    def setTimeLen(self, time_len):
        self.SOC = cp.Variable(shape = (time_len), nonneg=True)
        self.powerFlow = cp.Variable(shape = (time_len), bounds=[-self.consumption_capacity, self.production_capacity])

    def getConstraints(self, t):
        constraints = [self.SOC[t] <= self.battery_capacity]
        if t == 0:
            constraints.append(
                self.SOC[t] == self.efficiency * cp.neg(self.powerFlow[t]) - (1 / self.efficiency) * cp.pos(self.powerFlow[t])
            )
        else:
            constraints.append(
                self.SOC[t] == self.SOC[t-1] + self.efficiency * cp.neg(self.powerFlow[t]) - (1 / self.efficiency) * cp.pos(self.powerFlow[t])
            )
        return constraints

    def getPowerFlow(self, connecting_node, t):
        return self.powerFlow[t]

    def getCost(self):
        0

    def getVariables(self):
        return [self.SOC, self.powerFlow]

    
if __name__ == "__main__":
    problemClass = GraphProblemClass()
    node1 = ConnectingNode(problemClass)
    node2 = ConnectingNode(problemClass)
    producer = Producer(problemClass, node1, 10)
    consumer = Consumer(problemClass, node2)
    battery = Battery(problemClass, node2, 5, 5, 10)
    transmission_line = TransmissionLine(problemClass, node1, node2, 10)
    power_exchange = PowerExchange(problemClass, node1, 10, 10)

    problemClass.setTimeLen(5)
    prices = [1,2,2,3,1]
    
    power_exchange.setPrices(prices)
    consumer.setConsumptionSchedule([1,2,2,3,1])


    problemClass.solve()
    problemClass.printResults()
    print("done")
    
    # play_around_with_optimization_problem

    
