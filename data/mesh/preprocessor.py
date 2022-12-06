import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

id = 0

nodesPerLine = 10
lines = 10

g = nx.Graph()

corners = []

previousLine = []
for i in range(lines):
    currentLine = []
    for j in range(nodesPerLine):
        id += 1
        g.add_node(id)
        currentLine.append(id)
        if (j > 0):
            g.add_edge(id-1, id, weight=1)
    
    if (i==0) or (i == lines - 1):
        corners.append(currentLine[0])
        corners.append(currentLine[-1])

    if len(previousLine) > 0:
        for k in range(nodesPerLine):
            g.add_edge(previousLine[k], currentLine[k], weight=1)
    previousLine = list(currentLine)

data = json_graph.node_link_data(g)
with open('mesh100.json', 'w') as f:
    json.dump(data, f)