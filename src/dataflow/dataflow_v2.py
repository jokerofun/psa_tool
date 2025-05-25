from .dataflow_classes import DataProcessingNode, DataflowNode


class Dataflow:
    def __init__(self, name, object) -> None:
        self.name = name
        self.object = object
        self.nodes = {}
        
    # also include optional arguments for the constructor    
    def node(self, name: str, task_node_type = None, *args, **kwargs) -> None:
        if name in self.nodes:
            return self.nodes[name]
        else:
            if task_node_type is None:
                self.nodes[name] = DataProcessingNode(name, *args, **kwargs)
            else:
                self.nodes[name] = task_node_type(name, *args, **kwargs)
            return self.nodes[name]
    
    def execute(self) -> None:
        # run all nodes in the dataflow
        for node in self.nodes.values():
            node.run()

    # overload [] operator
    def __getitem__(self, name: str) -> DataflowNode:
        return self.node(name)
    
    def get_data(self, nodeID):
        # find the one with nodes.final = true
        for node in self.nodes.values():
            if node._final:
                node.run()
                return node.get_results()