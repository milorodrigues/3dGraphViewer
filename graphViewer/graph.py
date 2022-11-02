import networkx as nx

class Graph():
    def __init__(self, graph):
        self.graph = graph
        self.initializeGVFields(self.graph)

    def initializeGVFields(self, graph):
        nx.set_node_attributes(self.graph, (0.0, 0.0, 0.0), "GV_position")
        nx.set_node_attributes(self.graph, (1.0, 1.0, 1.0), "GV_color")