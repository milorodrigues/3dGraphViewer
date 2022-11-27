import numpy as np
import math
import sys
from functools import reduce
import networkx as nx
import glm
from abc import ABC, abstractmethod
import pandas as pd
import random
import time

class GraphDrawer:
    def __init__(self, model):
        self.origin = glm.vec3(0.0, 0.0, 0.0)

        model = model.lower()
        if model == "multi-scale":
            self.drawer = MultiScaleDrawer(self.origin)
        elif model == "graph-distance":
            self.drawer = GraphDistanceDrawer(self.origin)
        elif model == "spring":
            self.drawer = SpringDrawer(self.origin)
        elif model == "barycentric":
            self.drawer = BarycentricDrawer(self.origin)
        else:
            self.drawer = RandomDrawer(self.origin)
    
    def initialize(self, data):
        self.drawer.initialize(data)

    def runLoop(self, data):
        self.drawer.runLoop(data)

    def eades(data):
        areaRadius = 1.5 # maximum distance from origin

        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = areaRadius*(-1), high = areaRadius),
                np.random.uniform(low = areaRadius*(-1), high = areaRadius),
                np.random.uniform(low = areaRadius*(-1), high = areaRadius))
        return

    def randomize(self, data, flat=None):
        areaRadius = 1.5 # maximum distance from origin
        
        factor = [1,1,1]
        if flat in [0,1,2]:
            factor[flat] = 0

        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = areaRadius*(-1), high = areaRadius) * factor[0],
                np.random.uniform(low = areaRadius*(-1), high = areaRadius) * factor[1],
                np.random.uniform(low = areaRadius*(-1), high = areaRadius) * factor[2])

class DrawerInterface(ABC):
    @abstractmethod
    def initialize(self, data):
        raise NotImplementedError

    @abstractmethod
    def runLoop(self, data):
        raise NotImplementedError

class MultiScaleDrawer(DrawerInterface):
    def __init__(self, origin):
        self.origin = origin
        self.areaRadius = 100

        self.threshold = 10
        self.ratio = 3

    def initialize(self, data):
        print(f"Initializing MultiScaleDrawer...")

        print(f"Computing shortest paths...")
        pathsList = []
        paths = nx.all_pairs_dijkstra_path_length(data.graph, weight='weight')
        for path in paths:
            for target in path[1]:
                pathsList.append([path[0], target, path[1][target]])

        self.shortestPaths = pd.DataFrame(pathsList, columns=['n1', 'n2', 'distance'])

        print(f"Initializing node positions...")
        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius))

        print(f"Moving to drawing...")

        start = time.time()

        k = self.threshold
        while k <= data.graph.number_of_nodes():
            print(f"k = {k} {time.time() - start}")
            print(self.kCenters(data, k))
            k *= self.ratio
            print(f"{time.time() - start}")

    def kCenters(self, data, k):
        centers = set()
        next = random.choice(list(data.graph.nodes))

        centers.add(next)

        df = self.shortestPaths[(self.shortestPaths['n1'].isin(centers)) ^ (self.shortestPaths['n2'].isin(centers))].loc[:,['n1', 'n2', 'distance']].copy()
        df['n1'] = df.apply(lambda x: int(x['n2']) if x['n1'] in centers else int(x['n1']), axis=1)
        df.drop(df[~df['n2'].isin(centers)].index, inplace=True)
        df.set_index('n1', inplace=True)

        for i in range(2, k+1):
            centers.add(next)
            df2 = self.shortestPaths[((self.shortestPaths['n1'] == next) & ~(self.shortestPaths['n2'].isin(centers))) ^ ((self.shortestPaths['n2'] == next) & ~(self.shortestPaths['n1'].isin(centers)))].loc[:,['n1', 'n2', 'distance']].copy()
            df2['n1'] = df2.apply(lambda x: int(x['n2']) if x['n1'] == next else int(x['n1']), axis=1)
            df2.drop(df2[df2['n2'] != next].index, inplace=True)
            df2.rename(columns={'distance': 'distance2'}, inplace=True)
            df2.set_index('n1', inplace=True)

            df = pd.concat([df['distance'], df2['distance2']], axis=1)
            df['distance'] = df.apply(lambda x: x['distance2'] if x['distance2'] < x['distance'] else x['distance'], axis=1)
            df.drop(['distance2'], axis=1, inplace=True)
            next = df[df['distance'] == df['distance'].max()].index.values.astype(int)[0]
            df.drop(next, inplace=True)
        
        return centers

    def localLayout(self, data, centers):
        k = 4
        return

    def runLoop(self, data):
        return

    def getWeight(self, a, b, attr):
        if 'weight' in attr:
            return attr['weight']
        else:
            return 1

class GraphDistanceDrawer(DrawerInterface):
    def __init__(self, origin):
        self.areaRadius = 1.0
        self.origin = origin
        self.rigidity = 1 # big K
        self.threshold = 0.1 # epsilon
    
    def initialize(self, data):
        print(f"Initializing GraphDistanceDrawer...")

        print(f"Computing shortest paths...")
        pathsList = []
        paths = nx.all_pairs_dijkstra_path_length(data.graph, weight='weight')
        for path in paths:
            for target in path[1]:
                pathsList.append([path[0], target, path[1][target]])

        self.shortestPaths = pd.DataFrame(pathsList, columns=['n1', 'n2', 'distance'])
        self.shortestPaths.drop(self.shortestPaths[self.shortestPaths['n1'] == self.shortestPaths['n2']].index, inplace=True)

        print(f"Computing ideal distances and spring constants...")
        goalLength = (self.areaRadius * 2) / self.shortestPaths['distance'].max() # big L
        self.shortestPaths['length'] = self.shortestPaths['distance'] * goalLength
        self.shortestPaths['rigidity'] = self.rigidity / (self.shortestPaths['distance'] ** 2)

        print(f"Initializing deltas...")
        self.deltas = {}
        for n in self.shortestPaths['n1'].unique():
            self.deltas[n] = 0

        print(f"Initializing node positions...")
        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius))

        print(f"Moving to drawing...")
        return

    def runLoop(self, data):
        for key in self.deltas:
            print(key)
        return

class SpringDrawer(DrawerInterface):
    def __init__(self, origin):
        self.areaRadius = 1.0 # Maximum distance from origin
        self.origin = origin
        self.c = [None, 1, 1, 1, 0.1]

    def initialize(self, data):
        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius))

    def runLoop(self, data):
        for u in data.graph.nodes:
            pos = glm.vec3(*data.graph.nodes[u]['GV_position'])
            totalForce = glm.vec3(0.0, 0.0, 0.0)

            for v in data.graph.nodes:
                if v != u:
                    totalForce = totalForce + (self.c[4] * self.calculateForceExerted(data, v, u))

            newPos = pos + totalForce
            data.graph.nodes[u]['GV_position'] = (newPos.x, newPos.y, newPos.z)

    def calculateForceExerted(self, data, source, target):
        sourcePos = glm.vec3(*data.graph.nodes[source]['GV_position'])
        targetPos = glm.vec3(*data.graph.nodes[target]['GV_position'])

        distance = self.euclidean(sourcePos, targetPos)

        if data.graph.has_edge(source, target) or data.graph.has_edge(target, source):
            direction = glm.normalize(sourcePos - targetPos)
            strength = self.c[1] * math.log(distance/self.c[2])
            force = direction * strength

        else:
            direction = glm.normalize(targetPos - sourcePos)
            strength = self.c[3] / (distance ** 2)
            force = direction * strength

        return force

    def euclidean(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

class BarycentricDrawer(DrawerInterface):
    def __init__(self, origin):
        self.areaRadius = 3 # Maximum distance from origin
        self.origin = origin

    def initialize(self, data):
        if 'GV_BarycentricFixedVertices' in data.graph.graph:
            self.fixedVertices = data.graph.graph['GV_BarycentricFixedVertices']
        else:
            cycles = self.cycleFinder(data)
            self.fixedVertices = cycles[np.argmax(np.array([len(c) for c in cycles]))]

        self.freeVertices = [x for x in list(data.graph.nodes) if x not in self.fixedVertices]
        self.positionFixedVertices(self.fixedVertices, data)
    
    def runLoop(self, data):
        for node in self.freeVertices:
            self.positionNode(data, node)
    
    def cycleFinder(self, data):
        aux = [(0, -1)] * len(list(data.graph.nodes)) # (color, parent) tuples
        cycles = []

        for node in list(data.graph.nodes):
            if aux[node-1][0] == 0:
                self.cycleFinder_dfs(data, aux, node, -1, cycles)

        return cycles
    
    def cycleFinder_dfs(self, data, aux, node, parent, cycles):
        if aux[node-1][0] == 2: # Node fully visited
            return

        elif aux[node-1][0] == 1: # Node visited, but not fully: found cycle. Backtrack through parents to find full path
            current = parent
            path = [current]
            while current != node:
                current = aux[current-1][1]
                path.append(current)
            cycles.append(path)
            return
            

        aux[node-1] = (1, parent) # Marking as partially visited and marking the parent

        for neighbor in [n for n in data.graph.neighbors(node)]:
            if neighbor == aux[node-1][1]:
                continue
            self.cycleFinder_dfs(data, aux, neighbor, node, cycles)
        
        aux[node-1] = (2, aux[node-1][1]) # Marking as completely visited
        return

    def positionFixedVertices(self, fixedVertices, data):
        edges = len(fixedVertices)
        thetas = [i/edges * math.tau for i in range(edges)]

        for n in range(len(thetas)):
            theta = thetas[n]
            coordinates = (self.areaRadius * math.cos(theta) + self.origin[0], self.areaRadius * math.sin(theta) + self.origin[1])
            data.graph.nodes[fixedVertices[n]]['GV_position'] = (coordinates[0], coordinates[1], 0.0)

    def positionNode(self, data, node):
        neighbors = Util.getAllNeighbors(data, node)
        sumNeighbors = [sum(i) for i in zip(*[list(data.graph.nodes[n]['GV_position']) for n in neighbors])]
        
        newPos = [c/data.graph.degree[node] for c in sumNeighbors]
        data.graph.nodes[node]['GV_position'] = (newPos[0], newPos[1], newPos[2])

class RandomDrawer(DrawerInterface):
    def __init__(self, origin):
        self.areaRadius = 3 # Maximum distance from origin
        self.origin = origin

    def initialize(self, data):
        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius))

    def runLoop(self, data):
        return

class Util():
    def __init__(self):
        return

    def getAllNeighbors(data, node):        
        if nx.is_directed(data.graph):
            neighbors = set()
            neighbors.update(list(data.graph.neighbors(node)))
            neighbors.update(list(data.graph.predecessors(node)))
            return list(neighbors)
        else:
            return list(data.graph.neighbors(node))
        