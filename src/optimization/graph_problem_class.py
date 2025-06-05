import cvxpy as cp
from src.dataflow.dataflow_manager_v2 import DataflowManager
from .selector import Selector
# from persistence.db_manager import DBManager

import time

class GraphProblemClass():
    def __init__(self, name, time_length):
        self.name = name
        self.time_length = time_length
        self._nodes = []
        self._objective = ""
        self._selector = None
        self._dataflow_manager = DataflowManager()

    def __repr__(self):
        return f"GraphProblemClass(name={self.name},nodes={self._nodes},objective={self._objective})"
    
    def add_node(self, node):
        self._nodes.append(node)

    def add_nodes(self, nodes):
        self._nodes.extend(nodes)

    def get_node(self, node_name):
        node = next((item for item in self._nodes if getattr(item, "name", None) == str(node_name)), None)

        return node

    def fetch_dataflows(self):
        for node in self._nodes:
            if hasattr(node, "dataflow") and node.dataflow is not None:
                self._dataflow_manager.new_dataflow(node, node.dataflow)
    
    def set_time_length(self):
        for node in self._nodes:
            node.set_time_length(self.time_length)

    def collect_costs(self):
        return cp.sum([node.cost for node in self._nodes])
    
    def collect_constraints(self, t):
        constraints = []
        for node in self._nodes:
            constraints.extend(node.constraints(t))
        
        return constraints
    
    def collect_const_constraints(self):
        constraints = []
        for node in self._nodes:
            constraints.extend(node.const_constraints())

        return constraints
    
    # objective function builder, with minimize or maximize
    def get_objective_function(self, objective : str = "minimize"):
        self._objective = objective.lower()
        if self._objective != "minimize" and self._objective != "maximize":
            raise ValueError("Objective function must be either minimize or maximize")
        
        self._selector = Selector(self._nodes)
        return self._selector
    
    def solve(self, solver=cp.GUROBI, objective: str = "minimize", value: str = "cost"):
        # fetch dataflows from nodes and execute them
        self.fetch_dataflows()
        self._dataflow_manager.execute()
        self.dataflow_time = time.time()
        # assign result values from dataflows to parameters in nodes 
        for node in self._nodes:
            node.assign(self.time_length)

        # add connecting nodes to the problem class
        self.add_connecting_nodes()
        
        # set time length for all nodes (which basically initializes the decision variables for nodes)
        self.set_time_length()

        # build the objective function
        self.get_objective_function(objective).values(value)
        objective = None
        if self._objective == "minimize":
            objective = cp.Minimize(cp.sum(self._selector.get()))
        elif self._objective == "maximize":
            objective = cp.Maximize(cp.sum(self._selector.get()))
        else:
            raise ValueError("_objective was not set")

        # collect all constraints from the nodes
        constraints = []
        constraints.extend(self.collect_const_constraints())
        for t in range(self.time_length):
            constraints.extend(self.collect_constraints(t))
        
        # build and solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=solver, verbose=False)

        # if problem.status == cp.OPTIMAL:
        #     print(f"Result: {problem.value}")

        # print values of decision variables of the nodes in the GraphProblemClass
        # print("-" * 50)
        # self.print_results()
        # print("-" * 50)
        # print("END")

        return problem

    def add_connecting_nodes(self):
        """
        Checks each node for a connecting_node attribute and adds it to self._nodes if not already present.
        Ensures each connecting node is added only once.
        """
        existing_node_ids = set(id(node) for node in self._nodes)
        new_connecting_nodes = []
        for node in self._nodes:
            connecting_node = getattr(node, "connecting_node", None)
            if connecting_node is not None and id(connecting_node) not in existing_node_ids:
                new_connecting_nodes.append(connecting_node)
                existing_node_ids.add(id(connecting_node))
        self._nodes.extend(new_connecting_nodes)

    def print_results(self):
        for node in self._nodes:
            print(node.variables)
    

    def get_all_variables(self):
        variables = []
        for node in self._nodes:
            variables.append(node.variables)
            
        return variables
