import numpy as np

class GraphPainter:
    def __init__(self):
        return
    
    def random(data):
        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_color'] = (
                np.random.uniform(low = 0.0, high = 1.00001),
                np.random.uniform(low = 0.0, high = 1.00001),
                np.random.uniform(low = 0.0, high = 1.00001))