

from .dataflow_classes import DataProcessingNode, DataflowNode


class Dataflow:
    def __init__(self, NodeClass) -> None:
        self.NodeClass = NodeClass
        self.nodes = {}
        
    # also include optional arguments for the constructor    
    def node(self, name: str, classType = None, *args, **kwargs) -> None:
        if name in self.nodes:
            return self.nodes[name]
        else:
            if classType is None:
                self.nodes[name] = DataProcessingNode(name, *args, **kwargs)
            else:
                self.nodes[name] = classType(name, *args, **kwargs)
            return self.nodes[name]
    
    # overload [] operator
    def __getitem__(self, name: str) -> DataflowNode:
        return self.node(name)
    
    def getData(self, nodeID):
        # find the one with nodes.final = true
        for node in self.nodes.values():
            if node._final:
                node.run()
                return node.getResults()