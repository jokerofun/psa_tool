import abc
from src.dataflow.dataflow_v2 import Dataflow

class Node():
    def __init__(self, name):
        self.name = name
        self.dataflow = Dataflow(f"{self.name}_dataflow", self)
        self.dataflow.default_workflow(source="data/test_data/pricesEUR.csv")
    
    def __repr__(self):
        return f"Node(name={self.name})"
    
    def get_attr(self, attr):
        """
        Retrieve the attribute value by name.
        This works for both stored attributes and computed properties.
        """
        try:
            return getattr(self, attr)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{attr}'")
    
    def get_attributes(self):
        attributes = vars(self)
        primitive_attributes_only = {}

        for key, value in attributes.items():
            if not isinstance(value, Node):
                primitive_attributes_only[key] = value
            else:
                # primitive_attributes_only[key] = value.__class__.__name__
                primitive_attributes_only[key] = None

        return primitive_attributes_only
    
    def get_class_name(self):
        return self.__class__.__name__
    
    def constraints(self, t):
        return []

    @property
    def variables(self):
        pass
    
    @property
    def cost(self):
        pass
