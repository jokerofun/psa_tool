from __future__ import annotations

from .dataflow import Dataflow
from src.optimization.base_domain import Node

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
            self.dataFlows = {}

    def newDataFlow(self, NodeClass) -> Dataflow:
        # check if NodeClass is Node class or its subclass
        if not issubclass(NodeClass, Node):
            raise Exception("NodeClass should be a subclass of Node")
        # check if NodeClass is already in the dataFlows
        if NodeClass in self.dataFlows:
            return self.dataFlows[NodeClass]
        dataflow = Dataflow(NodeClass)
        self.dataFlows[NodeClass] = dataflow
        return self.dataFlows[NodeClass]

    def getData(self, nodeClass, nodeID):
        # check if nodeclass is subclass of Node
        if not issubclass(nodeClass, Node):
            raise Exception(str(nodeClass) +
                            ": NodeClass should be a subclass of Node")
        # check if nodeClass is in the dataFlows
        if nodeClass not in self.dataFlows:
            raise Exception("NodeClass is not in dataFlows")
        # get the dataflow instance
        dataflow = self.dataFlows[nodeClass]
        return dataflow.getData(nodeID)

    # overload [] operator
    def __getitem__(self, key):
        if key not in self.dataFlows:
            self.dataFlows[key] = Dataflow(key)
        return self.dataFlows[key]
