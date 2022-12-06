import networkx
from networkx.readwrite import json_graph
import json
import argparse

import graphViewer.graphViewer as GV

class Parameters:
    def __init__(self, filepath, model, iterations, renderCameraTarget):
        self.filepath = filepath
        self.model = model
        self.iterations = iterations
        self.renderCameraTarget = renderCameraTarget

        self.iterationsLeft = self.iterations

argParser = argparse.ArgumentParser()
argParser.add_argument("-f", "--filepath", help="Relative path to the json file containing the graph", required=True)
argParser.add_argument("-i", "--iterations", help="Number of iterations (default 100)", default=100)
argParser.add_argument("-m", "--model", help="Drawing model (default spring)", default="spring")
argParser.add_argument("-t", "--target", help="Render camera target on screen (y/n) (default n)", default='n')
args = argParser.parse_args()

par = Parameters(
    filepath=args.filepath,
    model=args.model,
    iterations=int(args.iterations),
    renderCameraTarget=True if (args.target.lower() == 'y' or args.target.lower() == 'yes') else False
)

with open(par.filepath) as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)

gv = GV.GraphViewer(graph = G, parameters = par) 

print(G.number_of_edges())
print(G.number_of_nodes())
gv.run()

#filepath = 'data/microtest/microtest.json'
#filepath = 'data/moreno_lesmis/moreno_lesmis.json'
#filepath = 'data/3-connected/3-connected.json'
#filepath = 'data/triangle/triangle.json'
#filepath = 'data/single_node/single_node.json'