from __future__ import annotations

from .dataflow import Dataflow
from optimization.solver_classes import Node

# singleton class
class DataFlowManager:
    __instance = None
    

    @staticmethod
    def getInstance() -> DataFlowManager:
        if DataFlowManager.__instance == None:
            DataFlowManager()
        return DataFlowManager.__instance

    def __init__(self) -> None:
        if DataFlowManager.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DataFlowManager.__instance = self
            self._dataflows = {}

    def newDataFlow(self, NodeClass) -> Dataflow:
        # check if NodeClass is Node class or its subclass
        if not issubclass(NodeClass, Node):
            raise Exception("NodeClass should be a subclass of Node")
        # check if NodeClass is already in the dataFlows
        if NodeClass in self._dataflows:
            return self._dataflows[NodeClass]
        dataflow = Dataflow(NodeClass)
        self._dataflows[NodeClass] = dataflow
        return self._dataflows[NodeClass]

    def getData(self, nodeClass, parameters = {}):
        # check if nodeclass is subclass of Node
        if not issubclass(nodeClass, Node):
            raise Exception(str(nodeClass) +
                            ": NodeClass should be a subclass of Node")
        # check if nodeClass is in the dataFlows
        if nodeClass not in self._dataflows:
            raise Exception("NodeClass is not in dataFlows")
        # get the dataflow instance
        dataflow = self._dataflows[nodeClass]
        ## assume that there are two columns one with date values and the other with values, convert the values to numpy array
        # dataflow['values'].values.flatten()
        # get the column name without the date column
        return dataflow.getData(parameters)

    # overload [] operator
    def __getitem__(self, key):
        if key not in self._dataflow:
            self._dataflow[key] = Dataflow(key)
        return self._dataflow[key]
