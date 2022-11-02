import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

#Perimeter for Tutte drawing: [1, 2, 3, 4, 5]

edge_list = [
    [1,2],
    [1,5],
    [1,6],
    [2,3],
    [2,7],
    [3,4],
    [3,8],
    [4,5],
    [4,9],
    [5,10],
    [6,7],
    [6,10],
    [6,11],
    [7,8],
    [7,11],
    [8,9],
    [8,11],
    [9,10],
    [9,11],
    [10,11]
]

edge_tuples = []
for i in np.arange(len(edge_list)):
    edge_tuples.append((edge_list[i][0], edge_list[i][1]))

G = nx.Graph()
G.add_edges_from(edge_tuples)
G.graph['GV_BarycentricFixedVertices'] = [1, 2, 3, 4, 5]

data = json_graph.node_link_data(G)
with open('3-connected.json', 'w') as f:
    json.dump(data, f)