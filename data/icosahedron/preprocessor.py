import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

edge_list = [
    [1,2],
    [1,3],
    [1,4],
    [1,5],
    [1,8],
    [2,3],
    [2,5],
    [2,10],
    [2,12],
    [3,8],
    [3,11],
    [3,12],
    [4,5],
    [4,6],
    [4,7],
    [4,8],
    [5,6],
    [5,10],
    [6,7],
    [6,9],
    [6,10],
    [7,8],
    [7,9],
    [7,11],
    [8,11],
    [9,10],
    [9,11],
    [9,12],
    [10,12],
    [11,12]
]

edge_tuples = []
for i in np.arange(len(edge_list)):
    edge_tuples.append((edge_list[i][0], edge_list[i][1]))

G = nx.Graph()
G.add_edges_from(edge_tuples)
G.graph['GV_BarycentricFixedVertices'] = [1, 2, 3]

data = json_graph.node_link_data(G)
with open('icosahedron.json', 'w') as f:
    json.dump(data, f)