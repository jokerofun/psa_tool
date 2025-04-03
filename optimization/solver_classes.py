import cvxpy as cp
import abc

from .selector import Selector
# from persistence.db_manager import DBManager

class BaseSolverClass():
    pass

class GraphProblemClass():
    def __init__(self, name):
        self.name = name
        self._nodes = []
        self.time_len = 0

    def __repr__(self):
        return f"GraphProblemClass(name={self.name}, nodes={self._nodes})"

    def add_node(self, node):
        self._nodes.append(node)

    def get_node(self, node_name):
        return next((node for node in self._nodes if getattr(node, "name", None) == node_name), None)

    def collect_costs(self):
        return cp.sum([node.cost for node in self._nodes])

    def collect_constraints(self, t):
        return [constraint for node in self._nodes for constraint in node.constraints(t)]

    def solve(self, objective="minimize"):
        if objective not in {"minimize", "maximize"}:
            raise ValueError("Objective must be 'minimize' or 'maximize'")
        
        obj_func = cp.Minimize if objective == "minimize" else cp.Maximize
        objective = obj_func(self.collect_costs())
        constraints = [self.collect_constraints(t) for t in range(self.time_len)]
        problem = cp.Problem(objective, [c for sublist in constraints for c in sublist])
        problem.solve()

    def set_time_len(self, time_len):
        self.time_len = time_len
        for node in self._nodes:
            node.set_time_len(time_len)

    def getObjectiveFunction(self, objective_type):
        """Retrieve the objective function for the problem."""
        if objective_type == "minimize":
            return cp.Minimize(self.collect_costs())
        elif objective_type == "maximize":
            return cp.Maximize(self.collect_costs())
        else:
            raise ValueError("Objective type must be 'minimize' or 'maximize'")

    def getAllVariables(self):
        """Retrieve all variables from the nodes."""
        # variables = {}
        # for node in self._nodes:
        #     variables[node.name] = node.variables
        # return variables
        variables = []

        for node in self._nodes:
            variables.append(node.variables)
        return variables

class Node():
    def __init__(self, problem_class, name="", is_connecting_node=False):
        self.problem_class = problem_class
        self.name = name
        self.is_connecting_node = is_connecting_node
        self.connected_nodes = [] if is_connecting_node else None
        self.connecting_node = None
        if problem_class:
            problem_class.add_node(self)

    def connect(self, other):
        """Connect this node to another node."""
        if self.is_connecting_node:
            self.connected_nodes.append(other)
        else:
            if not self.connecting_node and not other.connecting_node:
                connecting_node = Node(self.problem_class, name="ConnectingNode", is_connecting_node=True)
                connecting_node.connect(self)
                connecting_node.connect(other)
                self.connecting_node = connecting_node
                other.connecting_node = connecting_node
            elif not self.connecting_node:
                other.connecting_node.connect(self)
                self.connecting_node = other.connecting_node
            elif not other.connecting_node:
                self.connecting_node.connect(other)
                other.connecting_node = self.connecting_node

    def __sub__(self, other):
        """Overload the - operator to connect nodes."""
        if isinstance(other, Node):
            self.connect(other)
            return self
        raise ValueError("Can only connect to another Node instance")

    def constraints(self, t):
        if self.is_connecting_node:
            return [cp.sum([node.powerflow(t) for node in self.connected_nodes]) == 0]
        return []

    def set_time_len(self, time_len):
        self.time_len = time_len

    @property
    def cost(self):
        return

    @property
    def variables(self):
        return []

    def powerflow(self, t):
        return