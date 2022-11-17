import networkx
from networkx.readwrite import json_graph
import json

import graphViewer.graphViewer as GV

#filepath = 'data/microtest/microtest.json'
#filepath = 'data/moreno_lesmis/moreno_lesmis.json'
filepath = 'data/3-connected/3-connected.json'
#filepath = 'data/triangle/triangle.json'
#filepath = 'data/single_node/single_node.json'
with open(filepath) as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)

#print(G.number_of_nodes())
#print(G.number_of_edges())

gv = GV.GraphViewer(graph = G)
gv.run()