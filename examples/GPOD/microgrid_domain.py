import cvxpy as cp
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.optimization.energy_domain import Resource, ConnectingNode

class Consumer(Resource):
    def __init__(self, name):
        super().__init__(name)
        self.consumption_kWh = []

    def powerflow(self, t):
        return -self.consumption_kWh[t]
    
    @property
    def variables(self):
        return {self.name : {"powerFlow" : -self.consumption_kWh}}
    
    def assign(self, t):
        self.consumption_kWh = self.dataflow.results["gen_consumption"]["consumption_kWh"][:t]
    
class Producer(Resource):
    def __init__(self, name):
        super().__init__(name)
        self.max_power_output_kW = []

class SolarPanel(Producer):
    def __init__(self, name):
        super().__init__(name)

    def set_time_length(self, t):
        self.c = cp.Variable(t, nonneg=False)

    def powerflow(self, t):
        return self.c[t] * self.max_power_output_kW[t]
    
    def constraints(self, t):
        return [self.c[t] >= 0, self.c[t] <= 1]
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : self.c.value * self.max_power_output_kW}}

    def assign(self, t):
        self.max_power_output_kW = self.dataflow.results["gen_solar_data"]["energy_generated"][:t]

class WindTurbine(Producer):
    def __init__(self, name):
        super().__init__(name)
    
    def powerflow(self, t):
        return self.max_power_output_kW[t]
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : self.max_power_output_kW}}
    
    def assign(self, t):
        self.max_power_output_kW = self.dataflow.results["gen_wind_data"]["energy_generated"][:t]
    
class Battery(Resource):
    def __init__(self, name, charging_power_kW, discharging_power_kW, capacity_kWh, efficiency):
        super().__init__(name)
        self.charging_power_kW = charging_power_kW
        self.discharging_power_kW = discharging_power_kW
        self.capacity_kWh = capacity_kWh
        self.efficiency = efficiency

    def set_time_length(self, t):
        self.charge = cp.Variable(t, nonneg=False)
        self.discharge = cp.Variable(t, nonneg=False)
        self.SoC = cp.Variable(shape = (t+1), nonneg=False)

    def const_constraints(self):
        return [self.SoC[0] == 0]
    
    def constraints(self, t):
        constraints = [
            self.charge[t] >= 0,
            self.discharge[t] >= 0,
            self.SoC[t] >= 0,
            self.SoC[t+1] <= self.capacity_kWh,
            self.charge[t] <= self.charging_power_kW,
            self.discharge[t] <= self.discharging_power_kW,
            self.SoC[t+1] == self.SoC[t] + self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
        ]

        return constraints
    
    def powerflow(self, t):
        return self.discharge[t] - self.charge[t]
    
    @property
    def variables(self):
        return {self.name : {"SOC" : self.SoC.value, "powerFlow":  self.discharge.value - self.charge.value}}

class MeteringPoint(Resource):
    def __init__(self, name):
        super().__init__(name)
        # self.connect_nodes([self])

    def set_time_length(self, t):
        self.energy_import = cp.Variable(t, nonneg=False)

    def constraints(self, t):
        return [self.energy_import[t] >= 0]
    
    def powerflow(self, t):
        return self.energy_import[t]
    
    @property
    def cost(self):
        return cp.sum(self.energy_import)
    
    @property
    def variables(self):
        return {self.name : {"powerFlow" : self.energy_import.value}}
