from src.optimization.base_domain import DeviceNode, ConnectingNode

import cvxpy as cp

# class Node():
#     def __init__(self, name):
#         self.name = name
#         self.parameters = []
#         self.variables = []

#     def constraints(self, t):
#         return []

#     def add_parameter(self, parameter):
#         self.parameters.append(parameter)

#     def add_variable(self, variable):
#         self.variables.append(variable)
    
#     @property
#     def cost(self):
#         return 0

class Consumer(DeviceNode):
    def __init__(self, name, consumption_kWh=[]):
        super().__init__(name)
        self.consumption_kWh = consumption_kWh # power consumption in kWh

    def set_time_length(self, time_len):
        pass

    def powerflow(self, t):
        return -self.consumption_kWh[t]
    
    def variables(self):
        return {self.name : {"powerFlow" : -self.consumption_kWh}}

    @property
    def cost(self):
        return 0
    
class Producer(DeviceNode):
    def __init__(self, name, max_power_output_kW):
        super().__init__(name)
        self.max_power_output_kW = max_power_output_kW
        # self.power_output_kWh:list # power output in kWh

    @property
    def cost(self):
        return 0

class SolarPanel(Producer):
    def __init__(self, name, max_power_output_kW):
        super().__init__(name, max_power_output_kW)
        # self.c = [] # control variable c
        # self.P_rated_kW = P_rated_kW # rated power capacity in kW
        # self.Q = Q # solar irradiance per kW/m^2
        # self.max_power_output_kW = max_power_output_kW # maximum power output in kW
        # self.c:list
        # self.power_output_kWh = [x * y for x, y in enumerate(zip(self.c.value, self.max_power_output_kW))] # power output in kWh
        # self.init_variables(self)

    def set_time_length(self, time_len):
        self.c = cp.Variable(time_len, nonneg=True)
        # self.power_output_kWh = cp.Variable(time_len, nonneg=True) # power output in kWh

    def powerflow(self, t):
        return self.c[t] * self.max_power_output_kW[t]
    
    def constraints(self, t):
        # self.power_output_kWh[t] = self.c[t] * self.max_power_output_kW[t] # power output in kWh
        return [self.c[t] >= 0,
                self.c[t] <= 1 
                # self.power_output_kWh[t] == self.c[t] * self.P_rated_kW * self.Q[t], 
                # self.power_output_kWh[t] == self.c[t] * self.max_power_output_kW[t]
            ]

    # def init_variables(self):
    #     self.add_variable(self.c) # control variable c
    
    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : 0}}

class WindTurbine(Producer):
    def __init__(self, name, max_power_output_kW):
        super().__init__(name, max_power_output_kW)

    def constraints(self, t):
        return []
    
    def powerflow(self, t):
        return self.max_power_output_kW[t]
    
    def set_time_length(self, time_len):
        pass

    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : self.max_power_output_kW}}
    
class Battery(DeviceNode):
    def __init__(self, name, charging_power_kW, discharging_power_kW, capacity_kWh, efficiency):
        super().__init__(name)
        self.charging_power_kW = charging_power_kW # charging power in kW
        self.discharging_power_kW = discharging_power_kW
        self.capacity_kWh = capacity_kWh
        self.efficiency = efficiency
        # self.charge: list
        # self.discharge: list
        # self.SoC: list
        # self.mode:bool

        # self.init_variables(self)

    # def init_variables(self):
    #     self.add_variable(self.charge)
    #     self.add_variable(self.discharge)
    #     self.add_variable(self.SoC)
    #     self.add_variable(self.mode) # control variable mode

    def set_time_length(self, time_len):
        self.charge = cp.Variable(time_len, nonneg=True)
        self.discharge = cp.Variable(time_len, nonneg=True)
        # self.mode = cp.Variable(time_len, boolean=True)
        self.SoC = cp.Variable(shape = (time_len), nonneg=True)

    def constraints(self, t):
        constraints = [
            self.charge[t] >= 0,
            self.discharge[t] >= 0,
            self.SoC[t] >= 0,
            self.capacity_kWh >= 0,
            self.SoC[t] <= self.capacity_kWh,
            self.charge[t] <= self.charging_power_kW,# * self.mode[t],
            self.discharge[t] <= self.discharging_power_kW# * (1 - self.mode[t]),
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
        return -(self.charge[t] - self.discharge[t])
    
    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        return {self.name : {"SOC" : self.SoC.value, "powerFlow":  self.discharge.value - self.charge.value}}

class Grid(ConnectingNode):
    def __init__(self, name):
        super().__init__(name)
        # self.energy_import = [] # energy import in kWh
        # self.consumption_units = []
        # self.production_units = []
        # self.storage_units = []

        # self.energy_import:list

        # self.init_variables(self)
    
    # def init_variables(self):
    #     self.add_variable(self.energy_import)

    def set_time_length(self, time_len):
        self.energy_import = cp.Variable(time_len, nonneg=True)

    def powerflow(self, t):
        return self.energy_import[t]
    
    @property
    def cost(self):
        return cp.sum(self.energy_import)
    
    # def constraints(self, t):
    #     energy_produced = sum([unit.power_output_kWh[t] for unit in self.production_units])
    #     energy_consumed = sum([unit.consumption_kWh[t] for unit in self.consumption_units])
    #     energy_charged = sum([unit.charge[t] for unit in self.storage_units])
    #     energy_discharged = sum([unit.discharge[t] for unit in self.storage_units])

    #     return [energy_produced + energy_charged + self.energy_import[t] == energy_consumed + energy_discharged]
