import cvxpy as cp
import math
import numpy as np
import abc
import inspect

from .selector import Selector
# from persistence.db_manager import DBManager

class BaseSolverClass():
    pass

class GraphProblemClass():
    _nodes = []
    _objective = ""
    _selector = None
    def __init__(self):
        self._nodes = []
    
    def add_node(self, node):
        self._nodes.append(node)

    def collectCosts(self):
        return cp.sum([node.cost for node in self._nodes])
    
    def collectConstraints(self, t):
        constraints = []
        for node in self._nodes:
            constraints.extend(node.constraints(t))
        return constraints
    
    # objective function builder, with minimize or maximize
    def getObjectiveFunction(self, objective : str = "minimize"):
        self._objective = objective.lower()
        # print(self._objective)
        if self._objective != "minimize" and self._objective != "maximize":
            raise ValueError("Objective function must be either minimize or maximize")
        self._selector = Selector(self._nodes)
        return self._selector
    
    def solve(self):
        objective = None
        # print(self._selector.get())
        if self._objective == "minimize":
            objective = cp.Minimize(cp.sum(self._selector.get()))
        elif self._objective == "maximize":
            objective = cp.Maximize(cp.sum(self._selector.get()))
        else:
            raise ValueError("_objective was not set")
        constraints = []
        for t in range(self.time_len):
            constraints.extend(self.collectConstraints(t))
        problem = cp.Problem(objective, constraints)
        problem.solve()

    def printResults(self):
        for node in self._nodes:
            print(node.variables)

    def getAllVariables(self):
        variables = []
        for node in self._nodes:
            variables.append(node.variables)
        return variables

    def setTimeLen(self, time_len):
        self.time_len = time_len
        for node in self._nodes:
            node.setTimeLen(time_len)

class Node():
    def __init__(self, problem_class : GraphProblemClass):
        self.problem_class = problem_class
        problem_class.add_node(self)
        self.name = ""
        
    def get_attr(self, attr):
        """
        Retrieve the attribute value by name.
        This works for both stored attributes and computed properties.
        """
        try:
            return getattr(self, attr)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{attr}'")
        
    def constraints(self, t):
        return []
    
    @abc.abstractmethod
    def getConnectingNode(self):
        return
    
    @abc.abstractmethod
    def setConnectingNode(self, connecting_node):
        return
    
    @property
    def variables(self):
        return []
    
    @property
    def cost(self):
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

    def powerflow(self, connecting_node : ConnectingNode):
        return
    
    @property
    def cost(self):
        return 
    
    @property
    def variables(self):
        return []
    
    def getConnectingNode(self):
        return self.connecting_node
    
    def setConnectingNode(self, connecting_node):
        self.connecting_node = connecting_node
    
    def __sub__(self, other: Node): 
        if self.connecting_node is None and other.getConnectingNode() is None:
            # print("Connecting nodes are None: " + self.name)
            self.connecting_node = ConnectingNode(self.problem_class)
            other.setConnectingNode(self.connecting_node)
            self.connecting_node.connect(self)
            self.connecting_node.connect(other)
        elif self.connecting_node is None:
            # print("Connecting node is None: " + self.name)
            self.connecting_node = other.getConnectingNode()
            self.connecting_node.connect(self)
        elif other.connecting_node is None:
            # print("Other connecting node is None: " + self.name)
            other.setConnectingNode(self.connecting_node)
            self.connecting_node.connect(other)  
            
    # define rsub as doing nothing
    # def __rsub__(self, other: Node):
    #     return self
        
    

    
