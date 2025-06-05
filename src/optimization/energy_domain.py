from src.optimization.base_domain import Node
import cvxpy as cp

class ConnectingNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.connected_nodes = []

    def connect(self, node):
        self.connected_nodes.append(node)

    def connect_nodes(self, nodes):
        self.connected_nodes.extend(nodes)

    def set_time_length(self, time_length):
        self.time_len = time_length

    def constraints(self, t):
        return [cp.sum([node.powerflow(t) for node in self.connected_nodes]) == 0]
    
    @property
    def cost(self):
        return 0

class Resource(Node):
    def __init__(self, name):
        super().__init__(name)
        self.connecting_node = None
        self.connected_nodes = []

    def powerflow(self):
        return
    
    @property
    def cost(self):
        return 0
    
    @property
    def variables(self):
        return []
    
    def getConnectingNode(self):
        return self.connecting_node
    
    def setConnectingNode(self, connecting_node):
        self.connecting_node = connecting_node
    
    def connect_to(self, other):
        if isinstance(other, (list, tuple)):
            nodes = other
        else:
            nodes = [other]

        for node in nodes:
            if self.connecting_node is None and node.getConnectingNode() is None:
                self.connecting_node = ConnectingNode("balance")
                node.setConnectingNode(self.connecting_node)
                self.connecting_node.connect(self)
                self.connecting_node.connect(node)
            elif self.connecting_node is None:
                self.connecting_node = node.getConnectingNode()
                self.connecting_node.connect(self)
            elif node.getConnectingNode() is None:
                node.setConnectingNode(self.connecting_node)
                self.connecting_node.connect(node)
        return self  