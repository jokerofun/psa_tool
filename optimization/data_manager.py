import numpy as np


# Singleton class for a global data manager
class DataManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance
        
    def __init__(self):
        # here goes db connection stuff
        pass
        
    def getData(nodeID: int) -> np.array: 
        match nodeID:
            case 1:
                return [1,2,4,5,1]
            case 2:
                return [1,2,4,5,1]
            case 3:
                return [5,2,8,10,8]
            case _:
                return []
