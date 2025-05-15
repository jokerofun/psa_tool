import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataflow_manager.dataflow_classes import DataProcessingNode

## main check
if __name__ == "__main__":
    # set up placeholder dictionary for nodeID : parameters
    # this would live in a duckdb table
    parameters = {
        0 : {"rated_power": 50, "cut_in_speed": 3.5, "rated_speed" : 14, "cut_out_speed": 25}, # wind turbine
        1 : {"rated_power" : 5}, # solar panel on a home
        2 : {"rated_power" : 25}, # solar panel on the school
        3 : {"A0" : 0.4, "A1": 0.8, "A2": 1.8, "phi0": 4.6, "phi1": 5.2}, # consumption for the school
    }
    number_of_houses = 50
    ## initialize random with seed
    np.random.seed(42)
    
    for i in range(number_of_houses):
        parameters[i + 4] = {"A0" : 0.2 , "A1": 0.4, "A2": 0.4, "phi0": 3.4 + np.random.rand(1) * 0.4, "phi1": 10+ np.random.rand(1) * 0.4}
        
    
        
    
    