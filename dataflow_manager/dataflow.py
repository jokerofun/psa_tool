from .dataflow_classes import DataProcessingNode, DataflowNode


class Dataflow:
    def __init__(self, NodeClass) -> None:
        self._final_df_name = "results"
        self.NodeClass = NodeClass
        self.nodes = {}

    # also include optional arguments for the constructor
    def node(self, name: str, classType=None, *args, **kwargs) -> None:
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

    def getData(self, parameters: dict):
        # find the one with nodes.final = true
        self._parameters = parameters
        for node in self.nodes.values():
            if node._final:
                node.run(parameters)
                data = node.get_results()[self._final_df_name].values.flatten()
                return data
