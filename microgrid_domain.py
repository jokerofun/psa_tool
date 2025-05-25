from src.optimization.energy_domain import Resource

import cvxpy as cp

class Consumer(Resource):
    def __init__(self, name, consumption_kWh=[]):
        super().__init__(name)
        self.consumption_kWh = consumption_kWh # power consumption in kWh

    def set_time_length(self, t):
        pass

    def powerflow(self, t):
        return -self.consumption_kWh[t]
    
    @property
    def variables(self):
        return {self.name : {"powerFlow" : -self.consumption_kWh}}

    @property
    def cost(self):
        return 0
    
class Producer(Resource):
    def __init__(self, name, max_power_output_kW):
        super().__init__(name)
        self.max_power_output_kW = max_power_output_kW

    def set_time_length(self, t):
        pass

    @property
    def cost(self):
        return 0

class SolarPanel(Producer):
    def __init__(self, name, max_power_output_kW):
        super().__init__(name, max_power_output_kW)
        # self.init_variables(self)

    def set_time_length(self, t):
        self.c = cp.Variable(t, nonneg=True) # control variable c

    def powerflow(self, t):
        return self.c[t] * self.max_power_output_kW[t]
    
    def constraints(self, t):
        return [self.c[t] >= 0,
                self.c[t] <= 1]

    # def init_variables(self):
    #     self.add_variable(self.c) # control variable c
    
    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        # return {self.name : {"production_schedule" : self.c.value * self.max_power_output_kW}}
        return {self.name : {"production_schedule" : self.c.value}}

class WindTurbine(Producer):
    def __init__(self, name, max_power_output_kW):
        super().__init__(name, max_power_output_kW)

    def constraints(self, t):
        return []
    
    def powerflow(self, t):
        return self.max_power_output_kW[t]
    
    def set_time_length(self, t):
        self.time_length = t

    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : self.max_power_output_kW}}
    
class Battery(Resource):
    def __init__(self, name, charging_power_kW, discharging_power_kW, capacity_kWh, efficiency):
        super().__init__(name)
        self.charging_power_kW = charging_power_kW
        self.discharging_power_kW = discharging_power_kW
        self.capacity_kWh = capacity_kWh
        self.efficiency = efficiency

        # self.init_variables(self)

    # def init_variables(self):
    #     self.add_variable(self.charge)
    #     self.add_variable(self.discharge)
    #     self.add_variable(self.SoC)
    #     self.add_variable(self.mode) # control variable mode

    def set_time_length(self, t):
        self.charge = cp.Variable(t, nonneg=True)
        self.discharge = cp.Variable(t, nonneg=True)
        # self.mode = cp.Variable(time_len, boolean=True)
        self.SoC = cp.Variable(shape = (t), nonneg=True)

    def constraints(self, t):
        constraints = [
            self.charge[t] >= 0,
            self.discharge[t] >= 0,
            self.SoC[t] >= 0,
            self.capacity_kWh >= 0,
            self.SoC[t] <= self.capacity_kWh,
            self.charge[t] <= self.charging_power_kW, # * self.mode[t],
            self.discharge[t] <= self.discharging_power_kW, # * (1 - self.mode[t]),
        ]
        
        if t == 0:
            constraints.append(
                self.SoC[t] == self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t])
            constraints.append(self.SoC[t] == 0)
        else:
            constraints.append(
                self.SoC[t] == self.SoC[t-1] + self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t])
        
        return constraints
    
    def powerflow(self, t):
        return self.discharge[t] - self.charge[t]
    
    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        return {self.name : {"SOC" : self.SoC.value, "powerFlow":  self.discharge.value - self.charge.value}}

class Grid(Resource):
    def __init__(self, name):
        super().__init__(name)
        # self.init_variables(self)
    
    # def init_variables(self):
    #     self.add_variable(self.energy_import)

    def set_time_length(self, t):
        self.energy_import = cp.Variable(t, nonneg=True)

    def powerflow(self, t):
        return self.energy_import[t]
    
    @property
    def cost(self):
        return cp.sum(self.energy_import)
    
    @property
    def variables(self):
        return {self.name : {"powerFlow" : self.energy_import.value}}
