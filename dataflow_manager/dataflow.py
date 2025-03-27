

from dataflow_manager.dataflow_classes import DataflowNode


class Dataflow:
    def __init__(self, NodeClass) -> None:
        self.NodeClass = NodeClass
        self.nodes = {}
        
    def node(self, name: str, classType) -> None:
        if name in self.nodes:
            return self.nodes[name]
        else:
            self.nodes[name] = classType(name)
            return self.nodes[name]
    
    # overload [] operator
    def __getitem__(self, name: str) -> DataflowNode:
        return self.node(name)