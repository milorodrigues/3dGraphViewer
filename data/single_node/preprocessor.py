import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

G = nx.DiGraph()
G.add_node(1)

data = json_graph.node_link_data(G)
with open('single_node.json', 'w') as f:
    json.dump(data, f)