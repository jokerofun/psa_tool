from .solver_classes import Battery
import numpy as np

# battery extended with degradation
class BatteryDeg(Battery):
    def __init__(self, problem_class, production_capacity, consumption_capacity, battery_capacity, efficiency=0.9, D50 = 1.0, beta = 0.693):
        super().__init__(problem_class, production_capacity, consumption_capacity, battery_capacity, efficiency)
        self.D50 = D50
        self.beta = beta
        
    def getCost(self):
        return self.charge * self.D50 * np.exp(self.beta * ((self.SOC - 50) / 50))    
        
    