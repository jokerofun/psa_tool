from solver_classes import Node, ConnectingNode, GraphProblemClass, DeviceNode
import cvxpy as cp

class TransmissionLine(DeviceNode):
    def __init__(self, problem_class, capacity, transmission_loss = 0):
        super().__init__(problem_class)
        # direction is left(1) -> right (2)
        # special case as it will have 2 connecting nodes. The left one will be the standard connecting node
        # right one will be connecting_node_right
        self.connecting_node_right = None
        self.transmission_loss = transmission_loss
        self.capacity = capacity  
        self.name = "TransmissionLine"
        
    def __sub__(self, other: Node):
        if self.connecting_node_right is None and other.getConnectingNode() is None:
            self.connecting_node_right = ConnectingNode(self.problem_class)
            other.setConnectingNode(self.connecting_node_right)
            self.connecting_node_right.connect(self)
            self.connecting_node_right.connect(other)
        elif self.connecting_node_right is None:
            self.connecting_node_right = other.getConnectingNode()
            self.connecting_node_right.connect(self)
        elif other.connecting_node is None:
            other.setConnectingNode(self.connecting_node_right)
            self.connecting_node_right.connect(other)
            
    def getConnectingNode(self):
        return self.connecting_node_right
    
    def setConnectingNode(self, connecting_node):
        self.connecting_node_right = connecting_node

    def setTimeLen(self, time_len):
        self.time_len = time_len
        self.power_flow_left_right = cp.Variable(time_len)
        self.power_flow_right_left = cp.Variable(time_len)

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
    def __init__(self, problem_class, production_capacity, price = 10):
        super().__init__(problem_class)
        self.production_capacity = production_capacity
        self.price = price
        self.name = "Producer"

    def setTimeLen(self, time_len):
        self.production_schedule = cp.Variable(time_len, nonneg=True)

    def constraints(self, t):
        return [
            self.production_schedule[t] <= self.production_capacity
        ]
    
    def powerflow(self, connecting_node, t):
        return self.production_schedule[t]
    
    @property
    def cost(self):
        return cp.sum(self.production_schedule * self.price)
    
    @property
    def variables(self):
        return {self.name : {"production_schedule" : self.production_schedule.value}}
        
    
class Consumer(DeviceNode):
    def __init__(self, problem_class):
        super().__init__(problem_class)
        self.name = "Consumer"

    def setTimeLen(self, time_len):
        self.consumption_schedule = cp.Parameter(time_len, nonneg=True)

    def setConsumptionSchedule(self, consumption_schedule):
        self.consumption_schedule.value = consumption_schedule

    def constraints(self, t):
        return []

    def powerflow(self, connecting_node, t):
        return -self.consumption_schedule[t]
    
    def variables(self):
        return {self.name : {"powerFlow" : -self.consumption_schedule.value}}

class Prosumer(DeviceNode):
    def __init__(self, problem_class, production_capacity, consumption_capacity):
        super().__init__(problem_class)
        self.production_capacity = production_capacity
        self.consumption_capacity = consumption_capacity
        

class PowerExchange(Prosumer):
    _prices = []
    def __init__(self, problem_class, production_capacity, consumption_capacity):
        super().__init__(problem_class, production_capacity, consumption_capacity)
        self.name = "PowerExchange"

    def setTimeLen(self, time_len):
        self.powerFlow = cp.Variable(time_len)

    @property
    def prices(self):
        return self._prices

    @prices.setter
    def prices(self, prices):
        self._prices = prices

    def constraints(self, t):
        return [self.powerFlow[t] <= self.production_capacity, self.powerFlow[t] >= -self.consumption_capacity]

    def powerflow(self, connecting_node, t):
        return self.powerFlow[t]
    
    @property
    def cost(self):
        return cp.sum(self._prices @ self.powerFlow)
    
    @property
    def variables(self):
        return {self.name : {"powerFlow" : self.powerFlow.value}}
    
class Battery(Prosumer):
    def __init__(self, problem_class, production_capacity, consumption_capacity, name, battery_capacity, efficiency = 0.9):
        super().__init__(problem_class, production_capacity, consumption_capacity)
        self.battery_capacity = battery_capacity
        self.efficiency = efficiency
        self.name = name

    def __repr__(self):
        return "Battery"

    def setTimeLen(self, time_len):
        self.charge = cp.Variable(time_len, nonneg=True)
        self.discharge = cp.Variable(time_len, nonneg=True)
        self.SOC = cp.Variable(shape = (time_len), nonneg=True)

    def constraints(self, t):
        constraints = []
        constraints.append(self.charge[t] <= self.production_capacity)
        constraints.append(self.discharge[t] <= self.consumption_capacity)

        constraints = [self.SOC[t] <= self.battery_capacity]
        if t == 0:
            constraints.append(
                self.SOC[t] == self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
            )
        else:
            constraints.append(
                self.SOC[t] == self.SOC[t-1] + self.efficiency * self.charge[t] - (1 / self.efficiency) * self.discharge[t]
            )
        return constraints

    def powerflow(self, connecting_node, t):
        return self.discharge[t] - self.charge[t]
    
    @property
    def cost(self):
        return 0

    @property
    def variables(self):
        return {self.name : {"SOC" : self.SOC.value, "powerFlow":  self.discharge.value - self.charge.value}}

    def get_class_info(self):
        return self.__class__.__name__
    
    def get_parameters(self):
        attributes = vars(self)
        primitive_attributes_only = {}

        for key, value in attributes.items():
            if not isinstance(value, (GraphProblemClass, ConnectingNode)):
                primitive_attributes_only[key] = value
            else:
                # primitive_attributes_only[key] = value.__class__.__name__
                primitive_attributes_only[key] = None

        return primitive_attributes_only
    
    
if __name__ == "__main__":
    problemClass = GraphProblemClass()
    producer = Producer(problemClass, production_capacity=5, price=2)
    consumer = Consumer(problemClass)
    battery = Battery(problemClass, production_capacity= 5, consumption_capacity= 5, name="bat1", battery_capacity= 5)
    transmission_line = TransmissionLine(problemClass, capacity=5, transmission_loss=0)
    power_exchange = PowerExchange(problemClass, production_capacity= 10, consumption_capacity=5)

    producer - transmission_line
    transmission_line - consumer
    producer - power_exchange
    battery - consumer
    

    problemClass.setTimeLen(5)    
    power_exchange.prices = [1,2,4,5,1]
    consumer.setConsumptionSchedule([5,2,8,10,8])

    problemClass.getObjectiveFunction("minimize").of_type(PowerExchange).values("cost")
    problemClass.solve()
    problemClass.printResults()
    