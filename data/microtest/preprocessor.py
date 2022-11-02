import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

edge_list = [
    [1,2,3],
    [2,3,4],
    [2,6,1],
    [3,4,9],
    [4,2,8],
    [4,5,1],
    [5,6,2],
    [5,3,5],
    [7,6,5]
]

edge_tuples = []
for i in np.arange(len(edge_list)):
    edge_tuples.append((edge_list[i][0], edge_list[i][1], {'weight': edge_list[i][2]}))

G = nx.DiGraph()
G.add_edges_from(edge_tuples)

data = json_graph.node_link_data(G)
with open('microtest.json', 'w') as f:
    json.dump(data, f)