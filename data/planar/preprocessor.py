import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import json

id = 0

sides = 20
levels = 10

g = nx.Graph()

corners = []

id += 1
previousLine = [id]
for i in range(levels):
    currentLine = []
    for j in range(sides):
        id += 1
        g.add_node(id)
        currentLine.append(id)
    
    for j in range(len(currentLine)):
        u = currentLine[j]
        v = currentLine[(j + 1) % len(currentLine)]
        g.add_edge(u, v, weight=1)

    if len(previousLine) == 1:
        for v in currentLine:
            g.add_edge(previousLine[0], v, weight = 1)
    elif len(previousLine) > 1:
        for i in range(sides):
            g.add_edge(previousLine[i], currentLine[i], weight = 1)

    previousLine = list(currentLine)

g.graph['GV_BarycentricFixedVertices'] = list(previousLine)

data = json_graph.node_link_data(g)
with open('planar201.json', 'w') as f:
    json.dump(data, f)