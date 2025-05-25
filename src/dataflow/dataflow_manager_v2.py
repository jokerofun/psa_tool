from __future__ import annotations
from typing import Dict, TYPE_CHECKING

from .dataflow_v2 import Dataflow
from src.optimization.base_domain import Node

# singleton class
class DataflowManager:
    __instance = None

    @staticmethod
    def getInstance() -> DataflowManager:
        if DataflowManager.__instance == None:
            DataflowManager()
        return DataflowManager.__instance

    def __init__(self) -> None:
        if DataflowManager.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DataflowManager.__instance = self
            self.dataflows = {}
    
    def new_dataflow(self, object, dataflow = None) -> Dataflow:
        if object is None:
            raise Exception("Object cannot be None")
        if not isinstance(object, Node):
            raise Exception("Object should be a subclass of Node")
        if dataflow is not None and not isinstance(dataflow, Dataflow):
            raise Exception("Dataflow should be an instance of Dataflow class")

        if object.name in self.dataflows:
            return self.dataflows[object.name]
         
        if dataflow is None:
            dataflow = Dataflow(object.name, object)

        self.dataflows[object.name] = dataflow

        return dataflow
    
    def execute(self) -> None:
        # execute all dataflows
        for dataflow in self.dataflows.values():
            dataflow.execute()

    # get data from a specific task in an object's dataflow
    # note: object is associated with a dataflow class that has a list of tasks e.g. for fetching data, processing data, or ML
    def get_data(self, object, task_name):
        # get class of nodeClassInstance
        # nodeClass = nodeClassInstance.__class__
        # check if nodeclass is subclass of Node
        if not isinstance(object, Node):
            raise Exception("Object should be a subclass of Node")
        # check if nodeClass is in the dataFlows
        if object.name not in self.dataflows:
            raise Exception("Object doesn't have any dataflow")
        # get the dataflow instance
        dataflow = self.dataflows[object.name]
        return dataflow.get_data(task_name)
    
    # overload [] operator 
    def __getitem__(self, key):
        return self.new_dataflow(key)