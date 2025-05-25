from archive.solver_classes_deprecated import Node, ConnectingNode, DeviceNode
import cvxpy as cp

class TransmissionLine(DeviceNode):
    def __init__(self, name, capacity, transmission_loss = 0):
        super().__init__(name)
        self.name = name
        # direction is left(1) -> right (2)
        # special case as it will have 2 connecting nodes. The left one will be the standard connecting node
        # right one will be connecting_node_right
        self.connecting_node_right = None
        self.transmission_loss = transmission_loss
        self.capacity = capacity
        
    def __sub__(self, other: Node):
        if self.connecting_node_right is None and other.getConnectingNode() is None:
            self.connecting_node_right = ConnectingNode(self.name)
            other.setConnectingNode(self.connecting_node_right)
            self.connecting_node_right.connect(self)
            self.connecting_node_right.connect(other)
        elif self.connecting_node_right is None:
            self.connecting_node_right = other.getConnectingNode()
            self.connecting_node_right.connect(self)
        elif other.connecting_node is None:
            other.setConnectingNode(self.connecting_node_right)
            self.connecting_node_right.connect(other)
            
    def get_connecting_node(self):
        return self.connecting_node_right
    
    def set_connecting_node(self, connecting_node):
        self.connecting_node_right = connecting_node

    def set_time_length(self, time_segment):
        self.time_len = time_segment
        self.power_flow_left_right = cp.Variable(time_segment)
        self.power_flow_right_left = cp.Variable(time_segment)

    def constraints(self, t):
        return [
            self.power_flow_left_right[t] <= self.capacity,
            self.power_flow_right_left[t] <= self.capacity
        ] 
    
    @property
    def powerflow(self, connecting_node : ConnectingNode, t):
        if connecting_node == self.connecting_node_right:
            return (1-self.transmission_loss)*self.power_flow_left_right[t] - self.power_flow_right_left[t]
        elif connecting_node == self.connecting_node:
            return (1-self.transmission_loss)*self.power_flow_right_left[t] - self.power_flow_left_right[t]
        else:
            return 0
        
    @property
    def variables(self):
        return {self.name : {"powerFlow" : self.power_flow_left_right.value - self.power_flow_right_left.value}}

class Producer(DeviceNode):
    def __init__(self, name, production_capacity, price = 10):
        super().__init__(name)
        self.name = name
        self.production_capacity = production_capacity
        self.price = price

    def set_time_length(self, time_len):
        self.production_schedule = cp.Variable(time_len, nonneg=True)

    def constraints(self, time_segment):
        return [self.production_schedule[time_segment] <= self.production_capacity]
    
    def power_flow(self, time_segment):
        return self.production_schedule[time_segment]
    
    @property
    def cost(self):
        return cp.sum(self.production_schedule * self.price)
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : self.production_schedule.value}}

class Consumer(DeviceNode):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def set_time_length(self, time_segment):
        self.consumption_schedule = cp.Parameter(time_segment, nonneg=True)

    def set_consumption_schedule(self, consumption_schedule):
        self.consumption_schedule.value = consumption_schedule

    def constraints(self, time_segment):
        return []

    def power_flow(self, time_segment):
        return -self.consumption_schedule[time_segment]
    
    def variables(self):
        return {self.name : {"powerFlow" : -self.consumption_schedule.value}}

class Prosumer(DeviceNode):
    def __init__(self, name, production_capacity, consumption_capacity):
        super().__init__(name)
        self.name = name
        self.production_capacity = production_capacity
        self.consumption_capacity = consumption_capacity

class PowerExchange(Prosumer):
    _prices = []
    def __init__(self, name, production_capacity, consumption_capacity):
        super().__init__(name, production_capacity, consumption_capacity)
        self.name = name

    def __repr__(self):
        return f"PowerExchange(name={self.name},production_capacity={self.production_capacity},consumption_capacity={self.consumption_capacity})"
    
    def set_time_length(self, time_length):
        self.powerFlow = cp.Variable(time_length)

    @property
    def prices(self):
        return self._prices

    @prices.setter
    def prices(self, prices):
        self._prices = prices

    def constraints(self, t):
        return [self.powerFlow[t] <= self.production_capacity, self.powerFlow[t] >= -self.consumption_capacity]

    def power_flow(self, t):
        return self.powerFlow[t]
    
    @property
    def cost(self):
        return cp.sum(self._prices @ self.powerFlow)
    
    @property
    def variables(self):
        return {self.name : {"powerFlow" : self.powerFlow.value}}
    
class Battery(Prosumer):
    def __init__(self, name, production_capacity, consumption_capacity, battery_capacity, efficiency = 0.9):
        super().__init__(name, production_capacity, consumption_capacity)
        self.battery_capacity = battery_capacity
        self.efficiency = efficiency
        self.name = name

    def __repr__(self):
        return f"Battery(name={self.name},production_capacity={self.production_capacity}, consumption_capacity={self.consumption_capacity}, battery_capacity={self.battery_capacity}, efficiency={self.efficiency})"

    def set_time_length(self, time_segment):
        self.charge = cp.Variable(time_segment, nonneg=True)
        self.discharge = cp.Variable(time_segment, nonneg=True)
        self.mode = cp.Variable(time_segment, boolean=True)
        self.SOC = cp.Variable(shape = (time_segment), nonneg=True)

    def constraints(self, t):
        constraints = []
        # Ensure that if mode[t] == 1 then only charging is allowed (discharge[t] is forced to 0)
        # If mode[t] == 0 then only discharging is allowed (charge[t] is forced to 0)
        # We use the production_capacity and consumption_capacity as big-M values.
        constraints.append(self.charge[t] <= self.production_capacity * self.mode[t])
        constraints.append(self.discharge[t] <= self.consumption_capacity * (1 - self.mode[t]))
        constraints.append(self.SOC[t] <= self.battery_capacity)
        
        if t == 0:
            constraints.append(
                self.SOC[t] == self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
            )
            constraints.append( self.SOC[t] == 0 )
        else:
            constraints.append(
                self.SOC[t] == self.SOC[t-1] + self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
            )
        return constraints

    def power_flow(self, t):
        return self.discharge[t] - self.charge[t]
    
    @property
    def cost(self):
        return 0

    @property
    def variables(self):
        return {self.name : {"SOC" : self.SOC.value, "powerFlow":  self.discharge.value - self.charge.value}}


    