import cvxpy as cp
import abc

from .selector import Selector
# from persistence.db_manager import DBManager

class GraphProblemClass():
    def __init__(self, name):
        self.name = name
        self._nodes = []
        self._objective = ""
        self._selector = None

    def __repr__(self):
        return f"GraphProblemClass(name={self.name},nodes={self._nodes},objective={self._objective})"
    
    def add_node(self, node):
        self._nodes.append(node)

    def add_nodes(self, nodes):
        self._nodes.extend(nodes)

    def get_node(self, node_name):
        node = next((item for item in self._nodes if getattr(item, "name", None) == str(node_name)), None)

        return node

    def collect_costs(self):
        return cp.sum([node.cost for node in self._nodes])
    
    def collect_constraints(self, t):
        constraints = []
        for node in self._nodes:
            constraints.extend(node.constraints(t))
        
        return constraints
    
    # objective function builder, with minimize or maximize
    def get_objective_function(self, objective : str = "minimize"):
        self._objective = objective.lower()
        if self._objective != "minimize" and self._objective != "maximize":
            raise ValueError("Objective function must be either minimize or maximize")
        
        self._selector = Selector(self._nodes)
        return self._selector
    
    def solve(self):
        objective = None
        if self._objective == "minimize":
            objective = cp.Minimize(cp.sum(self._selector.get()))
        elif self._objective == "maximize":
            objective = cp.Maximize(cp.sum(self._selector.get()))
        else:
            raise ValueError("_objective was not set")
        
        constraints = []
        for t in range(self.time_length):
            constraints.extend(self.collect_constraints(t))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=True)

        if problem.status == cp.OPTIMAL:
            print(f"Result: {problem.value}")

        return problem.value

    def print_results(self):
        for node in self._nodes:
            print(node.variables)

    def get_all_variables(self):
        variables = []
        for node in self._nodes:
            variables.append(node.variables)
            
        return variables

    def set_time_length(self, time_length):
        self.time_length = time_length
        for node in self._nodes:
            node.set_time_length(time_length)

class Node():
    def __init__(self, name):
        self.name = name
        
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
    
    def get_class_name(self):
        return self.__class__.__name__
    
    def get_attributes(self):
        attributes = vars(self)
        primitive_attributes_only = {}

        for key, value in attributes.items():
            if not isinstance(value, (ConnectingNode)):
                primitive_attributes_only[key] = value
            else:
                # primitive_attributes_only[key] = value.__class__.__name__
                primitive_attributes_only[key] = None

        return primitive_attributes_only
    
    @abc.abstractmethod
    def getConnectingNode(self):
        pass
    
    @abc.abstractmethod
    def setConnectingNode(self, connecting_node):
        pass
    
    @property
    def variables(self):
        return []
    
    @property
    def cost(self):
        return
    
class ConnectingNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.connected_nodes = []

    def connect(self, node):
        self.connected_nodes.append(node)

    def connect_nodes(self, nodes):
        self.connected_nodes.extend(nodes)

    def set_time_length(self, time_length):
        self.time_len = time_length

    def constraints(self, t):
        # print([node.powerflow(t) for node in self.connected_nodes])
        return [cp.sum([node.powerflow(t) for node in self.connected_nodes]) == 0]

# TODO: to remove?
class DeviceNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.connecting_node = None
        self.connected_nodes = []

    def powerflow(self):
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
            self.connecting_node = ConnectingNode()
            other.setConnectingNode(self.connecting_node)
            self.connecting_node.connect(self)
            self.connecting_node.connect(other)
        elif self.connecting_node is None:
            self.connecting_node = other.getConnectingNode()
            self.connecting_node.connect(self)
        elif other.connecting_node is None:
            other.setConnectingNode(self.connecting_node)
            self.connecting_node.connect(other)  