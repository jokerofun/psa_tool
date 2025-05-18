class Node():
    def __init__(self, name):
        self.name = name
        self.parameters = []
        self.variables = []

    def constraints(self, t):
        return []

    def add_parameter(self, parameter):
        self.parameters.append(parameter)

    def add_variable(self, variable):
        self.variables.append(variable)
    
    @property
    def cost(self):
        return 0

class Consumer(Node):
    def __init__(self, name):
        super().__init__(name)
        self.consumption_kWh = []

    # def constraints(self):
    #     pass

    # def variables(self):
    #     pass

    # def cost(self):
    #     pass
    
class Producer(Node):
    def __init__(self, name):
        super().__init__(name)
        self.power_output_kWh:list # power output in kWh

    # def __str__(self):
    #     return f"Producer {self.id}: Capacity: {self.capacity}, Output: {self.output}"

class SolarPanel(Producer):
    def __init__(self, name, max_power_output_kW:list):
        super().__init__(name)
        # self.P_rated_kW = P_rated_kW # rated power capacity in kW
        # self.Q = Q # solar irradiance per kW/m^2
        self.max_power_output_kW = max_power_output_kW # maximum power output in kW
        self.c:list

        self.init_variables(self)

    def constraints(self, t):
        return [0 <= self.c[t] <= 1, 
                # self.power_output_kWh[t] == self.c[t] * self.P_rated_kW * self.Q[t], 
                self.power_output_kWh[t] == self.c[t] * self.max_power_output_kW[t]]

    def init_variables(self):
        self.add_variable(self.c) # control variable c

class WindTurbine(Producer):
    def __init__(self, name):
        super().__init__(name)

    def constraints(self, t):
        return []
    
class Battery(Node):
    def __init__(self, name, charging_power_kW, discharging_power_kW, capacity_kWh, efficiency):
        super().__init__(name)
        self.charging_power_kW = charging_power_kW # charging power in kW
        self.discharging_power_kW = discharging_power_kW
        self.capacity_kWh = capacity_kWh
        self.efficiency = efficiency
        self.charge: list
        self.discharge: list
        self.SoC: list
        self.mode:bool

        self.init_variables(self)

    def init_variables(self):
        self.add_variable(self.charge)
        self.add_variable(self.discharge)
        self.add_variable(self.SoC)

    def constraints(self, t):
        constraints = [
            self.charge[t] >= 0,
            self.discharge[t] >= 0,
            self.SoC[t] >= 0,
            self.capacity_kWh >= 0,
            self.SoC[t] <= self.capacity_kWh,
            self.charge[t] <= self.charging_power_kW * self.mode[t],
            self.discharge[t] <= self.discharging_power_kW * (1 - self.mode[t]),
        ]
        
        if t == 0:
            constraints.append(
                self.SoC[t] == self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t])
            constraints.append(self.SOC[t] == 0)
        else:
            constraints.append(
                self.SoC[t] == self.SOC[t-1] + self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t])
        
        return constraints
    
class Grid(Node):
    def __init__(self, name):
        super().__init__(name)
        self.consumption_units = []
        self.production_units = []
        self.storage_units = []
        self.energy_import:list

        self.init_variables(self)

    def init_variables(self):
        self.add_variable(self.energy_import)

    def constraints(self, t):
        energy_produced = sum([unit.power_output_kWh[t] for unit in self.production_units])
        energy_consumed = sum([unit.consumption_kWh[t] for unit in self.consumption_units])
        energy_charged = sum([unit.charge[t] for unit in self.storage_units])
        energy_discharged = sum([unit.discharge[t] for unit in self.storage_units])

        return [energy_produced + energy_charged + self.energy_import[t] == energy_consumed + energy_discharged]

    @property
    def cost(self):
        return 0
    














class Household():
    def __init__(self, id, demand):
        self.id = id
        self.demand = demand
        self.solar = 0
        self.battery = 0
        self.grid = 0

    def __str__(self):
        return f"House {self.id}: Demand: {self.demand}, Solar: {self.solar}, Battery: {self.battery}, Grid: {self.grid}"
    
class School():
    def __init__(self, id, demand):
        self.id = id
        self.demand = demand
        self.solar = 0
        self.battery = 0
        self.grid = 0

    def __str__(self):
        return f"School {self.id}: Demand: {self.demand}, Solar: {self.solar}, Battery: {self.battery}, Grid: {self.grid}"
    
class SolarPanel():
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.output = 0

    def __str__(self):
        return f"Solar Panel {self.id}: Capacity: {self.capacity}, Output: {self.output}"
    
class WindTurbine():
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.output = 0

    def __str__(self):
        return f"Wind Turbine {self.id}: Capacity: {self.capacity}, Output: {self.output}"
    
class Battery():
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.state_of_charge = 0

    def __str__(self):
        return f"Battery {self.id}: Capacity: {self.capacity}, State of Charge: {self.state_of_charge}"
    
class Grid():
    def __init__(self, id):
        self.id = id
        self.power_flow = 0

    def __str__(self):
        return f"Grid {self.id}: Power Flow: {self.power_flow}"
    
