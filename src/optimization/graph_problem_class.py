import cvxpy as cp
from dataflow.dataflow_manager_v2 import DataflowManager
from .selector import Selector
# from persistence.db_manager import DBManager

class GraphProblemClass():
    def __init__(self, name):
        self.name = name
        self._nodes = []
        self._objective = ""
        self._selector = None
        self._dataflow_manager = DataflowManager.getInstance()

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
    
    def solve(self, solver=cp.GUROBI, objective: str = "minimize", value: str = "cost"):
        # fetch dataflows from nodes and execute them
        self.fetch_dataflows()
        self._dataflow_manager.execute()
        
        # build the objective function
        self.get_objective_function(objective).values(value)
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
        problem.solve(solver=solver, verbose=True)

        if problem.status == cp.OPTIMAL:
            print(f"Result: {problem.value}")

        # print values of decision variables in the nodes of the GraphProblemClass
        print("-" * 50)
        self.print_results()
        print("-" * 50)
        print("END")

        return problem

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