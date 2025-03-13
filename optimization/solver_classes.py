import cvxpy as cp
import math
import numpy as np
import abc
import inspect

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
        objective = cp.Minimize(self.collectCosts())
        constraints = []
        for t in range(self.time_len):
            constraints.extend(self.collectConstraints(t))
        problem = cp.Problem(objective, constraints)
        problem.solve()

    def printResults(self):
        for node in self.nodes:
            print(node.name)
            for var in node.getVariables():
                print(var.value)

    def setTimeLen(self, time_len):
        self.time_len = time_len
        for node in self.nodes:
            node.setTimeLen(time_len)

class Node():
    def __init__(self, problem_class : GraphProblemClass):
        self.problem_class = problem_class
        problem_class.add_node(self)
        self.name = ""
        
    # @abc.abstractmethod
    def getConstraints(self):
        return []
    
    @abc.abstractmethod
    def getConnectingNode(self):
        return
    
    @abc.abstractmethod
    def setConnectingNode(self, connecting_node):
        return
    
    def getVariables(self):
        return []
    
    @abc.abstractmethod
    def getCost(self):
        return
    
class ConnectingNode(Node):
    def __init__(self, problem_class):
        super().__init__(problem_class)
        self.connected_nodes = []
        self.name = "ConnectingNode"

    def connect(self, node):
        self.connected_nodes.append(node)

    def setTimeLen(self, time_len):
        self.time_len = time_len

    def getConstraints(self, t):
        return [cp.sum([node.getPowerFlow(self, t) for node in self.connected_nodes]) == 0]
    
## find a better name for this class, because it is not a leaf     
class DeviceNode(Node):
    def __init__(self, problem_class):
        super().__init__(problem_class)
        self.connecting_node = None

    @abc.abstractmethod
    def getPowerFlow(self, connecting_node : ConnectingNode):
        return
    
    @abc.abstractmethod
    def getCost(self):
        return 
    
    def getVariables(self):
        return []
    
    def getConnectingNode(self):
        return self.connecting_node
    
    def setConnectingNode(self, connecting_node):
        self.connecting_node = connecting_node
    
    def __sub__(self, other: Node): 
        if self.connecting_node is None and other.getConnectingNode() is None:
            print("Connecting nodes are None: " + self.name)
            self.connecting_node = ConnectingNode(self.problem_class)
            other.setConnectingNode(self.connecting_node)
            self.connecting_node.connect(self)
            self.connecting_node.connect(other)
        elif self.connecting_node is None:
            print("Connecting node is None: " + self.name)
            self.connecting_node = other.getConnectingNode()
            self.connecting_node.connect(self)
        elif other.connecting_node is None:
            print("Other connecting node is None: " + self.name)
            other.setConnectingNode(self.connecting_node)
            self.connecting_node.connect(other)  
            
    # define rsub as doing nothing
    # def __rsub__(self, other: Node):
    #     return self
        
class TransmissionLine(DeviceNode):
    def __init__(self, problem_class, capacity, transmission_loss = 0):
        super().__init__(problem_class)
        # direction is left(1) -> right (2)
        # special case as it will have 2 connecting nodes. The left one will be the standard connecting node
        # right one will be connecting_node_right
        self.connecting_node_right = None
        self.transmission_loss = transmission_loss
        self.capacity = capacity  
        self.name = "TransmissionLine"
        
    def __sub__(self, other: Node):
        if self.connecting_node_right is None and other.getConnectingNode() is None:
            self.connecting_node_right = ConnectingNode(self.problem_class)
            other.setConnectingNode(self.connecting_node_right)
            self.connecting_node_right.connect(self)
            self.connecting_node_right.connect(other)
        elif self.connecting_node_right is None:
            self.connecting_node_right = other.getConnectingNode()
            self.connecting_node_right.connect(self)
        elif other.connecting_node is None:
            other.setConnectingNode(self.connecting_node_right)
            self.connecting_node_right.connect(other)
            
    def getConnectingNode(self):
        return self.connecting_node_right
    
    def setConnectingNode(self, connecting_node):
        self.connecting_node_right = connecting_node

    def setTimeLen(self, time_len):
        self.time_len = time_len
        self.power_flow_left_right = cp.Variable(time_len)
        self.power_flow_right_left = cp.Variable(time_len)

    def getConstraints(self, t):
        return [
            self.power_flow_left_right[t] <= self.capacity,
            self.power_flow_right_left[t] <= self.capacity
        ] 
    
    def getPowerFlow(self, connecting_node : ConnectingNode, t):
        if connecting_node == self.connecting_node_right:
            return (1-self.transmission_loss)*self.power_flow_left_right[t] - self.power_flow_right_left[t]
        elif connecting_node == self.connecting_node:
            return (1-self.transmission_loss)*self.power_flow_right_left[t] - self.power_flow_left_right[t]
        else:
            return 0
        
    def getVariables(self):
        return [self.power_flow_left_right - self.power_flow_right_left]


class Producer(DeviceNode):
    def __init__(self, problem_class, production_capacity, price = 10):
        super().__init__(problem_class)
        self.production_capacity = production_capacity
        self.price = price
        self.name = "Producer"

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
    def __init__(self, problem_class):
        super().__init__(problem_class)
        self.name = "Consumer"

    def setTimeLen(self, time_len):
        self.consumption_schedule = cp.Parameter(time_len, nonneg=True)

    def setConsumptionSchedule(self, consumption_schedule):
        self.consumption_schedule.value = consumption_schedule

    def getConstraints(self, t):
        return []

    def getPowerFlow(self, connecting_node, t):
        return -self.consumption_schedule[t]

class Prosumer(DeviceNode):
    def __init__(self, problem_class, production_capacity, consumption_capacity):
        super().__init__(problem_class)
        self.production_capacity = production_capacity
        self.consumption_capacity = consumption_capacity
        

class PowerExchange(Prosumer):
    def __init__(self, problem_class, production_capacity, consumption_capacity):
        super().__init__(problem_class, production_capacity, consumption_capacity)
        self.name = "PowerExchange"

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
    def __init__(self, problem_class, production_capacity, consumption_capacity, battery_capacity, efficiency = 0.9):
        super().__init__(problem_class, production_capacity, consumption_capacity)
        self.battery_capacity = battery_capacity
        self.efficiency = efficiency
        self.name = "Battery"

    def setTimeLen(self, time_len):
        self.charge = cp.Variable(time_len, nonneg=True)
        self.discharge = cp.Variable(time_len, nonneg=True)
        self.SOC = cp.Variable(shape = (time_len), nonneg=True)

    def getConstraints(self, t):
        constraints = []
        constraints.append(self.charge[t] <= self.production_capacity)
        constraints.append(self.discharge[t] <= self.consumption_capacity)

        constraints = [self.SOC[t] <= self.battery_capacity]
        if t == 0:
            constraints.append(
                self.SOC[t] == self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
            )
        else:
            constraints.append(
                self.SOC[t] == self.SOC[t-1] + self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
            )
        return constraints

    def getPowerFlow(self, connecting_node, t):
        return self.discharge[t] - self.charge[t]

    def getCost(self):
        0

    def getVariables(self):
        return [self.SOC, self.discharge - self.charge]

    
if __name__ == "__main__":
    problemClass = GraphProblemClass()
    producer = Producer(problemClass, production_capacity=5, price=2)
    consumer = Consumer(problemClass)
    battery = Battery(problemClass, production_capacity= 5, consumption_capacity= 5, battery_capacity= 5)
    transmission_line = TransmissionLine(problemClass, capacity=5, transmission_loss=0)
    power_exchange = PowerExchange(problemClass, production_capacity= 10, consumption_capacity=5)

    producer - transmission_line
    transmission_line - consumer
    producer - power_exchange
    battery - consumer
    

    problemClass.setTimeLen(5)    
    power_exchange.setPrices([1,2,4,5,1])
    consumer.setConsumptionSchedule([5,2,8,10,8])

    problemClass.solve()
    problemClass.printResults()
    
    # print(inspect.signature(producer))
    print(inspect.signature(Producer))
    print(inspect.signature(Producer.getPowerFlow))
    print(inspect.signature(producer.getPowerFlow))
    # print(inspect.signature(producer))
    print(inspect.getsource(Producer))
    print(inspect.getsource(Producer.getPowerFlow))
    print(inspect.getsource(producer.getPowerFlow))

    
