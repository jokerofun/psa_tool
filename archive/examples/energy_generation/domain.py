from archive.solver_classes_deprecated import Node
import cvxpy as cp

class Generator(Node):
    def __init__(self, 
                 name, 
                 cost_per_mwh, 
                 min_power_output, 
                 max_power_output):
        self.name = name
        self.cost_per_mwh = cost_per_mwh  # in €/MWh
        self.min_power_output = min_power_output
        self.max_power_output = max_power_output

    def __repr__(self):
        return f"Generator({self.name}, {self.cost_per_mwh}, {self.min_power_output}, {self.max_power_output})"
    
    def constraints(self, t):
        constraints = []

        constraints.append(self.power_output[t] >= self.min_power_output)
        constraints.append(self.power_output[t] <= self.max_power_output)
        
        return constraints
    
    def setTimeLen(self, t):
        self.power_output = cp.Variable(t, nonneg=True)

    @property
    def cost(self):
        return cp.sum(self.cost_per_mwh * self.power_output)
    
    @property
    def variables(self):
        return {self.name: {"power_output": self.power_output.value.round(2)}}
    
class Area(Node):
    def __init__(self, name):
        self.name = name
        self.future_hourly_demand = []  # in MWh
        self.generators = []

    def __repr__(self):
        return f"Area({self.name})"

    def constraints(self, t):
        constraints = []

        # Total generation must meet total demand
        total_hourly_generation = sum(gen.power_output[t] for gen in self.generators)
        constraints.append(total_hourly_generation == self.future_hourly_demand[t])
        
        return constraints
    
    def add_generator(self, generator):
        self.generators.append(generator)

    def add_generators(self, generators):
        self.generators.extend(generators)

    def setTimeLen(self, t):
        pass

    @property
    def cost(self):
        return 0

    @property
    def variables(self):
        hourly_cost = sum(gen.cost_per_mwh * gen.power_output.value.round(2) for gen in self.generators)
        total_cost = sum(hourly_cost)
        return {self.name: {"future_hourly_demand": self.future_hourly_demand, "hourly_cost:": hourly_cost, "total_cost": total_cost}}