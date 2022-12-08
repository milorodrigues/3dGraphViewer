import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

id = 0

nodesPerCircle = 20
circles = 75

g = nx.Graph()

firstCircle = []
for i in range(nodesPerCircle):
    id += 1
    g.add_node(id)
    firstCircle.append(id)
for i in range(len(firstCircle)):
    g.add_edge(firstCircle[i], firstCircle[(i+1)%len(firstCircle)], weight=1)

previousCircle = list(firstCircle)
for circle in range(1, circles):
    currentCircle = []
    for node in range(nodesPerCircle):
        # add nodes of current circle
        id += 1
        g.add_node(id)
        currentCircle.append(id)
    for i in range(nodesPerCircle):
        # connect current circle
        g.add_edge(currentCircle[i], currentCircle[(i+1)%len(currentCircle)], weight=1)
    for i in range(nodesPerCircle):
        # connect to previous circle
        g.add_edge(currentCircle[i], previousCircle[i], weight=1)
    
    previousCircle = list(currentCircle)

# connect last circle to first to form a torus
for i in range(nodesPerCircle):
    g.add_edge(firstCircle[i], previousCircle[i], weight=1)
    print(f"Attaching {firstCircle[i]} to {previousCircle[i]}")
    #g.add_edge(firstCircle[i], previousCircle[nodesPerCircle - 1 -i], weight=1)
    #print(f"Attaching {firstCircle[i]} to {previousCircle[nodesPerCircle - 1 -i]}")

data = json_graph.node_link_data(g)
with open('torus1500.json', 'w') as f:
    json.dump(data, f)