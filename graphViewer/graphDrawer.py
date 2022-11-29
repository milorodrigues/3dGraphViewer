import numpy as np
import math
import networkx as nx
import glm
from abc import ABC, abstractmethod
import pandas as pd
import random
import time
import queue

class GraphDrawer:
    def __init__(self, model):
        self.origin = glm.vec3(0.0, 0.0, 0.0)

        model = model.lower()
        if model == "multi-scale":
            self.drawer = GajerDrawer(self.origin)
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

class GajerDrawer(DrawerInterface):
    def __init__(self, origin):
        self.origin = origin

        # Parameters
        self.rounds = 15
        self.areaScaling = 1
        self.r = 0.15 # For heat calculation
        self.s = 3 # For heat calculation

        self.iterationsLeft = 0
        self.roundsLeft = 0
    
    def initialize(self, data):
        print(f"Computing shortest paths...")
        paths = nx.all_pairs_dijkstra_path_length(data.graph, weight='weight')
        self.distances = {}
        for p in paths:
            self.distances.setdefault(p[0], p[1])

        e = nx.eccentricity(data.graph, sp=self.distances)
        diameterWeighted = nx.diameter(data.graph, e)
        diameterUnweighted = nx.diameter(data.graph)

        print(f"Generating filtrations...")
        factor = math.log(diameterWeighted, 2) / (diameterUnweighted / math.log(diameterUnweighted, 2))
        exp = factor

        aux = list(data.graph.nodes)
        self.filtrations = []
        self.filtrations.append(list(aux))

        while 2**exp <= diameterWeighted:
            aux = list(self.filtrations[-1])
            self.filtrations.append([])

            while len(aux) > 0:
                next = random.choice(aux)
                aux.remove(next)
                self.filtrations[-1].append(next)

                maxDistance = 2**exp
                for v in aux:
                    if self.distances[next][v] <= maxDistance:
                        aux.remove(v)
            
            exp += factor

            # Cutting the smallest graph down to 3 vertices for triangulation later
            # If there was a graph smaller than 3, it's dropped but its vertices are kept when cutting down the next smallest one
            keep = set()
            while len(self.filtrations[-1]) < 3:
                set.update(set(self.filtrations[-1]))
                self.filtrations = self.filtrations[:-1]
            if len(self.filtrations[-1]) > 3:
                aux = set(self.filtrations[-1])
                aux = list(aux - keep)
                self.filtrations[-1] = list(keep) + random.sample(aux, 3 - len(keep))

        self.k = len(self.filtrations) - 1
        self.filtrations.append([])

        print(f"Computing neighborhoods...")

        # Neighborhoods dict structure:
        # {source: {k1: {size: x, members: [neighbors]}, ...}, ...}
        self.neighborhoods = {}

        degreeSum = 0
        for node, val in data.graph.degree(weight='weight'):
            degreeSum += val

        for i in range(self.k, -1, -1):
            size = self.nbrs(i, degreeSum)
            for v in self.filtrations[i]:
                self.neighborhoods.setdefault(v, {})
                self.neighborhoods[v].setdefault(i, {'size': 0, 'members': [v]})
                q = queue.PriorityQueue()
                for n in data.graph.neighbors(v):
                    q.put((self.distances[v][n], n, self.distances[v][n]))

                while self.neighborhoods[v][i]['size'] < size:
                    if q.empty(): break

                    closest = q.get()
                    while closest[1] in self.neighborhoods[v][i]['members']:
                        if q.empty(): break
                        closest = q.get()
                    if q.empty() and closest[1] in self.neighborhoods[v][i]['members']: break

                    self.neighborhoods[v][i]['size'] += closest[2]
                    self.neighborhoods[v][i]['members'].append(closest[1])

                    for n in data.graph.neighbors(closest[1]):
                        if n not in self.neighborhoods[v][i]['members']:
                            q.put((self.distances[v][n], n, data.graph[closest[1]][n]['weight']))

        print(f"Moving to drawing...")
        self.iterationsLeft = self.k
        self.placedVertices = {}
        return
    
    def runLoop(self, data):
        if self.roundsLeft == 0:
            if self.iterationsLeft > 0:
                self.runIteration(data)
                #self.iterationsLeft -= 1
                self.iterationsLeft = 0
                self.roundsLeft = self.rounds
        else:
            print(f"rounds left: {self.roundsLeft}")
            self.runRound(data)
            self.roundsLeft -= 1
        return

    def runIteration(self, data):
        print(f"starting iteration i = {self.iterationsLeft}")
        f = list(set(self.filtrations[self.iterationsLeft]) - set(self.filtrations[self.iterationsLeft+1]))

        if self.iterationsLeft == self.k: # First iteration, place 3 deepest vertices in a triangle around the origin
            data.graph.nodes[f[0]]['GV_position'] = (0.0, 0.0, 0.0)
            data.graph.nodes[f[1]]['GV_position'] = (self.distances[f[0]][f[1]] * self.areaScaling, 0.0, 0.0)

            x = self.distances[f[0]][f[2]] * ((self.distances[f[0]][f[2]]**2 + self.distances[f[0]][f[1]]**2 - self.distances[f[1]][f[2]]**2) / (2 * self.distances[f[0]][f[2]] * self.distances[f[0]][f[1]]))
            y = math.sqrt(self.distances[f[0]][f[2]]**2 - x**2)
            data.graph.nodes[f[2]]['GV_position'] = (x, y, 0.0)

            #print(f"Distance AB = {Util.magnitude(glm.vec3(*data.graph.nodes[f[1]]['GV_position']) - glm.vec3(*data.graph.nodes[f[0]]['GV_position']))} Graph distance AB = {self.distances[f[0]][f[1]]} Scaled graph distance AB = {self.distances[f[0]][f[1]] * self.areaScaling}")
            #print(f"Distance AC = {Util.magnitude(glm.vec3(*data.graph.nodes[f[2]]['GV_position']) - glm.vec3(*data.graph.nodes[f[0]]['GV_position']))} Graph distance AC = {self.distances[f[0]][f[2]]} Scaled graph distance AC = {self.distances[f[0]][f[2]] * self.areaScaling}")
            #print(f"Distance BC = {Util.magnitude(glm.vec3(*data.graph.nodes[f[2]]['GV_position']) - glm.vec3(*data.graph.nodes[f[1]]['GV_position']))} Graph distance BC = {self.distances[f[1]][f[2]]} Scaled graph distance BC = {self.distances[f[1]][f[2]] * self.areaScaling}")

            barycenter = (glm.vec3(*data.graph.nodes[f[0]]['GV_position']) + glm.vec3(*data.graph.nodes[f[1]]['GV_position']) + glm.vec3(*data.graph.nodes[f[2]]['GV_position'])) / 3
            delta = glm.vec3(0.0, 0.0, 0.0) - barycenter

            for v in f:
                newPos = glm.vec3(*data.graph.nodes[v]['GV_position']) + delta
                data.graph.nodes[v]['GV_position'] = (newPos.x, newPos.y, newPos.z)
                self.placedVertices.setdefault(v, {})

        else:
            for v in f:
                # find initial position pos[v] of v
                break
        return
    
    def runRound(self, data):
        i = self.iterationsLeft + 1 # Because we've already decreased self.iterationsLeft in runLoop before starting the rounds
        f = self.filtrations[i]

        for v in f:
            delta = self.calculateLocalForce(data, v, i)

            if 'heat' not in self.placedVertices[v]:
                heat = self.areaScaling/6
                self.placedVertices[v]['oldCos'] = 0
            else:
                heat = self.placedVertices[v]['heat']
                if Util.magnitude(delta) != 0 and Util.magnitude(self.placedVertices[v]['oldDelta']):
                    c = (delta * self.placedVertices[v]['oldDelta']) / Util.magnitude(delta) * Util.magnitude(self.placedVertices[v]['oldDelta'])
                    if self.placedVertices[v]['oldCos'] * c > 0:
                        heat += (1 + c * self.r * self.s)
                    else:
                        heat += (1 + c * self.r)
                    self.placedVertices[v]['oldCos'] = c
            
            delta = heat * (delta / Util.magnitude(delta))
            self.placedVertices[v]['oldDelta'] = delta

            newPos = glm.vec3(*data.graph.nodes[v]['GV_position']) + delta
            data.graph.nodes[v]['GV_position'] = (newPos.x, newPos.y, newPos.z)
            print(f"{v} {newPos}")
        return

    def calculateLocalForce(self, data, v, i):
        force = glm.vec3(0.0, 0.0, 0.0)
        neighborhood = self.neighborhoods[v][i]['members']
        pos = glm.vec3(*data.graph.nodes[v]['GV_position'])

        for n in neighborhood:
            if n == v:
                continue
            nPos = glm.vec3(*data.graph.nodes[n]['GV_position'])

            euclideanDistance = Util.magnitude(nPos - pos)
            graphDistance = self.distances[v][n]
            nForce = (euclideanDistance / (graphDistance * (self.areaScaling ** 2))) - 1
            nForce = nForce * (nPos - pos)
            force = force + nForce
        return force

    def nbrs(self, i, degreeSum):
        return (degreeSum / len(self.filtrations[i]))

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

    def magnitude(vector):
        return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)
        