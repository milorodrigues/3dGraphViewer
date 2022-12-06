import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

id = 0

nodesPerLine = 10
lines = 10

g = nx.Graph()

corners = [9, 90, 99]

previousLine = []
for i in range(lines):
    currentLine = []
    for j in range(nodesPerLine):
        id += 1

        g.add_node(id)
        currentLine.append(id)

        if (j > 0):
            g.add_edge(currentLine[-2], currentLine[-1], weight=1)

    if len(previousLine) > 0:
        for k in range(nodesPerLine):
            g.add_edge(previousLine[k], currentLine[k], weight=1)
    previousLine = list(currentLine)

g = nx.contracted_nodes(g, 1, 9, self_loops=False, copy=True)
g = nx.contracted_nodes(g, 90, 99, self_loops=False, copy=True)
g = nx.contracted_nodes(g, 1, 90, self_loops=False, copy=True)

data = json_graph.node_link_data(g)
with open('knottedmesh100.json', 'w') as f:
    json.dump(data, f)
    