import networkx as nx
from networkx.readwrite import json_graph
import json

filepath = 'input data/moreno_lesmis/moreno_lesmis.json'
with open(filepath) as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)

print(G.number_of_nodes())
print(G.number_of_edges())