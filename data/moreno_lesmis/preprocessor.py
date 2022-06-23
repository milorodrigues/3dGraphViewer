import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

infile = open('out.moreno_lesmis_lesmis', 'r')
lines = infile.readlines()

edge_list = []
for i in np.arange(2,len(lines)):
    edge_list.append(list(map(int, lines[i].strip().split())))

edge_tuples = []
for i in np.arange(len(edge_list)):
    edge_tuples.append((edge_list[i][0], edge_list[i][1], {'weight': edge_list[i][2]}))

G = nx.Graph()
G.add_edges_from(edge_tuples)

data = json_graph.node_link_data(G)
with open('moreno_lesmis.json', 'w') as f:
    json.dump(data, f)