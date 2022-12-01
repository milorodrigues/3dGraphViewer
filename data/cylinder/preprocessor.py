import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

id = 0

nodesPerCircle = 30
circles = 50

g = nx.Graph()

previousCircle = []
for circle in range(circles):    
    currentCircle = []
    for node in range(nodesPerCircle):
        id += 1
        g.add_node(id)
        currentCircle.append(id)
    for i in range(1, len(currentCircle)+1):
        u = currentCircle[i-1]
        v = currentCircle[i % len(currentCircle)]
        g.add_edge(u, v, weight=1)

    if len(previousCircle) > 0:
        for i in range(nodesPerCircle):
            g.add_edge(currentCircle[i], previousCircle[i], weight=1)

    previousCircle = list(currentCircle)

data = json_graph.node_link_data(g)
with open('cylinder1500.json', 'w') as f:
    json.dump(data, f)