import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

edge_list = [
    [1,2],
    [1,3],
    [1,4],
    [2,3],
    [2,4],
    [3,4],
]

edge_tuples = []
for i in np.arange(len(edge_list)):
    edge_tuples.append((edge_list[i][0], edge_list[i][1]))

G = nx.Graph()
G.add_edges_from(edge_tuples)
G.graph['GV_BarycentricFixedVertices'] = [1, 2, 3]

data = json_graph.node_link_data(G)
with open('triangle.json', 'w') as f:
    json.dump(data, f)