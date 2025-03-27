from __future__ import annotations
import queue
from typing import Dict, TYPE_CHECKING

from dataflow_manager.dataflow import Dataflow
from optimization.solver_classes import Node

# if TYPE_CHECKING:
#     import pandas as pd
#     from dataflow_manager.dataflow_classes import DataflowNode

# nodes_queue = queue.Queue()

# def addNode(node: DataflowNode) -> None:
#     nodes_queue.put(node)

# def listQueue() -> None:
#     print(list(nodes_queue.queue))

# def addNodes(nodes: list[DataflowNode]) -> None:
#     for node in nodes:
#         nodes_queue.put(node)

# def execute() -> Dict[str, pd.DataFrame]:
#     dfs = {}
#     while not nodes_queue.empty():
#         node = nodes_queue.get()
#         try:
#             dfs = node.execute(dfs)
#         except Exception as e:
#             print(f"Error executing node {node.name}: {e}")
#             nodes_queue.put(node)
#             break
#     return dfs

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
        return Dataflow(NodeClass)

    def node(self, name: str) -> DataflowNode:
        for node in nodes_queue.queue:
            if node.name == name:
                return node
        return None

    def datafromcsvNode(self, name: str, file_path: str) -> DataflowNode:
        # node = DataFetchingFromFileNode(name, file_path)
        # addNode(node)
        # return node
        pass

    def datafromAPINode(self, name: str, url: str) -> DataflowNode:
        # node = DataFetchingFromAPI(name, url)
        # addNode(node)
        # return node
        pass 
    
    def processingNode(self, name: str, data: pd.DataFrame = None, data_name: str = None) -> DataflowNode:
        # node = DataProcessingNode(name, data, data_name)
        # addNode(node)
        # return node
        pass

    def datafromDBNode(self, name: str, table_name: str) -> DataflowNode:
        # node = DataFetchingFromDBNode(name, table_name)
        # addNode(node)
        # return node
        pass

    def getData(self, nodeClassInstance, nodeID):
        # get class of nodeClassInstance
        nodeClass = nodeClassInstance.__class__
        # check if nodeclass is subclass of Node
        if not issubclass(nodeClass, Node):
            raise Exception("NodeClass should be a subclass of Node")
        # check if nodeClass is in the dataFlows
        if nodeClass not in self.dataFlows:
            raise Exception("NodeClass is not in dataFlows")
        # get the dataflow instance
        dataflow = self.dataFlows[nodeClass]
        return dataflow.getData(nodeID)
    
    # overload [] operator 
    def __getitem__(self, key):
        return self.dataFlows[key]