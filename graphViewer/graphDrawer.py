import numpy as np
import math
import networkx as nx
import glm
from abc import ABC, abstractmethod
import pandas as pd
import random
import time

class GraphDrawer:
    def __init__(self, model, iterations):
        self.origin = glm.vec3(0.0, 0.0, 0.0)

        model = model.lower()
        if model == "multi-scale":
            self.drawer = GajerDrawer(self.origin)
        elif model == "centers":
            self.drawer = CentersDrawer(self.origin)
        elif model == "spring":
            self.drawer = SpringDrawer(self.origin, iterations)
        elif model == "barycentric":
            self.drawer = BarycentricDrawer(self.origin, iterations)
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

class CentersDrawer(DrawerInterface):
    def __init__(self, origin):
        self.origin = origin

        self.threshold = 5
        self.ratio = 3

        self.areaScaling = 3

        self.neighborhoodSize = 8
        self.localRounds = 40
    
    def initialize(self, data):
        print(f"Computing shortest paths...")
        paths = nx.all_pairs_dijkstra_path_length(data.graph, weight='weight')
        self.distances = {}
        for p in paths:
            self.distances.setdefault(p[0], p[1])

        print(f"Computing centers...")
        self.n = data.graph.number_of_nodes()
        k = self.threshold

        self.centers = []
        next = random.choice(list(data.graph.nodes))
        centersDistance = dict(self.distances[next])
        centersDistance.pop(next, None)
        self.centers.append(next)
        
        while k <= self.n:
            for i in range(k - len(self.centers)):
                next = max(centersDistance, key = lambda key: centersDistance[key])
                
                centersDistance.pop(next, None)
                self.centers.append(next)

                for d in centersDistance:
                    if centersDistance[d] > self.distances[next][d]:
                        centersDistance[d] = self.distances[next][d]
            
            print(f"k = {k} len(self.centers) = {len(self.centers)}")
            k *= self.ratio

        self.filtrations = []
        k = self.threshold
        while k <= self.n:
            self.filtrations.append(list(self.centers[:k]))
            k *= self.ratio
        self.filtrations.append(list(data.graph.nodes))
        
        print([len(l) for l in self.filtrations])

        print(f"Moving to drawing...")
        self.k = len(self.filtrations) - 1
        return
    
    def runLoop(self, data):
        if self.k >= 0:
            print(f"Running loop k = {self.k}")
            f = self.filtrations[len(self.filtrations) - 1 - self.k]
            print(len(f))

            if self.k == len(self.filtrations) - 1:
                # Initialize first set of centers
                drawingRadius = (nx.diameter(data.graph, nx.eccentricity(data.graph, sp=self.distances)) * self.areaScaling)/2
                for v in f:
                    randomPos = glm.vec3(np.random.uniform(low = drawingRadius * (-1), high = drawingRadius),
                                        np.random.uniform(low = drawingRadius * (-1), high = drawingRadius),
                                        np.random.uniform(low = drawingRadius * (-1), high = drawingRadius))
                    data.graph.nodes[v]['GV_position'] = (randomPos.x, randomPos.y, randomPos.z)

            for i in range(self.localRounds):
                print(f"Round {i+1}/{self.localRounds}")
                for v in f:
                    neighborhood = []
                    neighborhoodSize = 0

                    fSorted = sorted([(n, self.distances[v][n]) for n in f], key = lambda tup: tup[1])

                    for n in fSorted:
                        if neighborhoodSize > self.neighborhoodSize: break
                        neighborhood.append(n[0])
                        neighborhoodSize += n[1]

                    disp = self.calculateLocalDisplacement(data, v, neighborhood)
                    pos = glm.vec3(*data.graph.nodes[v]['GV_position'])
                    newPos = pos + disp
                    data.graph.nodes[v]['GV_position'] = (newPos.x, newPos.y, newPos.z)

            """for v in f:
                print(f"{v} {data.graph.nodes[v]['GV_position']}")"""

            for v in self.filtrations[-1]:
                if v in f: continue
                
                c = self.closestCenter(data, v, f)
                pos = glm.vec3(*data.graph.nodes[c]['GV_position'])
                noise = glm.vec3(np.random.uniform(low = -0.5, high = 0.5),
                                np.random.uniform(low = -0.5, high = 0.5),
                                np.random.uniform(low = -0.5, high = 0.5))
                newPos = pos + noise
                data.graph.nodes[v]['GV_position'] = (newPos.x, newPos.y, newPos.z)

            self.k -= 1
            #self.k = -1

            print(f"Centering drawing...")
            self.centerGraph(data)
        return

    def runIteration(self, data):
        return
    
    def runRound(self, data):
        return

    def calculateLocalDisplacement(self, data, v, neighborhood):
        pos = glm.vec3(*data.graph.nodes[v]['GV_position'])
        disp = glm.vec3(0.0, 0.0, 0.0)

        for n in neighborhood:
            nPos = glm.vec3(*data.graph.nodes[n]['GV_position'])
            delta = nPos - pos

            if Util.magnitude(delta) == 0: continue

            # Attract if too far, repel if too close
            if Util.magnitude(delta) < (self.distances[v][n] * self.areaScaling):
                delta = delta * (-1)

            nDisp = glm.normalize(delta) * math.log(Util.magnitude(delta)/(self.distances[v][n] * self.areaScaling))
            disp += nDisp

        return disp

    def closestCenter(self, data, v, centers):
        distances = sorted([(n, self.distances[v][n]) for n in centers], key = lambda tup: tup[1])
        return distances[0][0]

    def centerGraph(self, data):
        barycenter = glm.vec3(0.0, 0.0, 0.0)
        for node in data.graph.nodes:
            barycenter += glm.vec3(*data.graph.nodes[node]['GV_position'])
        
        n = data.graph.number_of_nodes()
        barycenter = glm.vec3(0.0, 0.0, 0.0) - (barycenter/n)

        for node in data.graph.nodes:
            newPos = barycenter + glm.vec3(*data.graph.nodes[node]['GV_position'])
            data.graph.nodes[node]['GV_position'] = (newPos.x, newPos.y, newPos.z)

class GajerDrawer(DrawerInterface):
    def __init__(self, origin):
        self.origin = origin

        # Parameters
        self.minRounds = 10
        self.maxRounds = 200
        self.areaScaling = 5
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
        self.diameterWeighted = nx.diameter(data.graph, e)
        self.diameterUnweighted = nx.diameter(data.graph)
        print(f"dW = {self.diameterWeighted} dU = {self.diameterUnweighted}")

        print(f"Generating filtrations...")
        #factor = math.log(diameterWeighted, 2) / (diameterUnweighted / math.log(diameterUnweighted, 2))
        factor = 1
        exp = factor

        aux = list(data.graph.nodes)
        self.filtrations = []
        self.filtrations.append(list(aux))

        #while 2**exp <= diameterWeighted:
        while 2**exp <= self.diameterUnweighted:
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

        print(f"{[len(f) for f in self.filtrations]}")

        # Cutting the smallest graph down to 5 vertices to define an R4 space for the first placement loop
        # If there was a graph smaller than 5, it's dropped but its vertices are kept when cutting down the next smallest one
        keep = set()
        while len(self.filtrations[-1]) < 5:
            set.update(set(self.filtrations[-1]))
            self.filtrations = self.filtrations[:-1]
        if len(self.filtrations[-1]) > 5:
            aux = set(self.filtrations[-1])
            aux = list(aux - keep)
            self.filtrations[-1] = list(keep) + random.sample(aux, 5 - len(keep))

        self.k = len(self.filtrations) - 1
        self.filtrations.append([])
        print(f"{[len(f) for f in self.filtrations]}")

        print(f"Computing neighborhoods...")

        # Neighborhoods dict structure:
        # {source: {k1: {size: x, members: [neighbors]}, ...}, ...}
        self.neighborhoods = {}

        degreeSum = 0
        for node, val in data.graph.degree(weight='weight'):
            degreeSum += val

        """for i in range(self.k, -1, -1):
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
                            q.put((self.distances[v][n], n, data.graph[closest[1]][n]['weight']))"""
    
        for i in range(self.k, -1, -1):
            size = math.ceil(self.nbrs(i, degreeSum))
            for v in self.filtrations[i]:
                #print(f"v = {v}")
                visited = set()
                visited.add(v)

                self.neighborhoods.setdefault(v, {})
                self.neighborhoods[v].setdefault(i, {'size': 0, 'members': [v]})
                di = [(node, self.distances[v][node]) for node in self.filtrations[i]]
                di = sorted(di, key = lambda tup: tup[1])

                begin = 0
                while di[begin][0] == v:
                    begin += 1
                
                end = begin + size + 1
                if end >= len(di):
                    end = len(di)
                
                self.neighborhoods[v][i]['members'] = [m[0] for m in di[begin:end]]

                #print(f"size = {self.neighborhoods[v][i]['size']} len(Ni(v)) = {len(self.neighborhoods[v][i]['members'])}")

        print(f"Building R4 space and initializing GV_position_R4 fields...")
        r4 = []

        # Generating 4 linearly independent 4D vectors
        while True:
            r4 = []
            r4.append(glm.vec4(np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0)))
            r4.append(glm.vec4(np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0)))
            r4.append(glm.vec4(np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0)))
            r4.append(glm.vec4(np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0), np.random.uniform(low = -1.0, high = 1.0)))
            
            mat = np.transpose(np.array([[*r4[0]], [*r4[1]], [*r4[2]], [*r4[3]]]))
            if np.linalg.det(mat) > 0.01 or np.linalg.det(mat) < -0.01:
                break
        
        # Orthonormalizing
        r4, r = np.linalg.qr(mat)
        r4 = np.transpose(r4)
        r4 = [glm.vec4(*v) for v in r4]

        self.r4 = r4

        for n in data.graph.nodes:
            data.graph.nodes[n]['GV_position_R4'] = (0.0, 0.0, 0.0, 0.0)
        
        print(f"Moving to drawing...")
        self.iterationsLeft = self.k
        self.placedVertices = {}
        return
    
    def runLoop(self, data):
        if self.roundsLeft == 0:
            if self.iterationsLeft >= 0:
                #if self.iterationsLeft == self.k - 2: return
                self.runIteration(data)
                self.roundsLeft = self.getRounds()
                self.iterationsLeft -= 1
            elif self.iterationsLeft == -1:
                print(f"Centering graph...")
                self.centerGraph(data)
                self.iterationsLeft -= 1
                print(f"Drawing finished.")
        else:
            print(f"rounds left: {self.roundsLeft}")
            self.runRound(data)
            self.roundsLeft -= 1
        return

    def getRounds(self):
        #return math.ceil((((self.maxRounds - self.minRounds)/self.k) * (self.k - self.iterationsLeft + 1)) + self.minRounds)
        #return 30
        base = (self.maxRounds/self.minRounds) ** (1/self.k)
        return math.ceil(base ** (self.k - self.iterationsLeft)) * self.minRounds

    def runIteration(self, data):
        print(f"starting iteration i = {self.iterationsLeft}")
        print(f"{[len(f) for f in self.filtrations]}")
        f = list(set(self.filtrations[self.iterationsLeft]) - set(self.filtrations[self.iterationsLeft+1]))
        #print(f"f = {f}")

        if self.iterationsLeft == self.k: # First iteration, place 5 deepest vertices around the origin
            """rand = False
            if (self.distances[f[0]][f[1]] + self.distances[f[0]][f[2]] > self.distances[f[1]][f[2]] and
                self.distances[f[1]][f[2]] + self.distances[f[0]][f[2]] > self.distances[f[0]][f[1]] and
                self.distances[f[0]][f[1]] + self.distances[f[1]][f[2]] > self.distances[f[0]][f[2]]):
                a = f[0]
                b = f[1]
                c = f[2]
                d = f[3]
                e = f[4]
            elif (self.distances[f[0]][f[1]] + self.distances[f[0]][f[3]] > self.distances[f[1]][f[3]] and
                self.distances[f[1]][f[3]] + self.distances[f[0]][f[3]] > self.distances[f[0]][f[1]] and
                self.distances[f[0]][f[1]] + self.distances[f[1]][f[3]] > self.distances[f[0]][f[3]]):
                a = f[0]
                b = f[1]
                d = f[2]
                c = f[3]
                e = f[4]
            elif (self.distances[f[0]][f[3]] + self.distances[f[0]][f[2]] > self.distances[f[3]][f[2]] and
                self.distances[f[3]][f[2]] + self.distances[f[0]][f[2]] > self.distances[f[0]][f[3]] and
                self.distances[f[0]][f[3]] + self.distances[f[3]][f[2]] > self.distances[f[0]][f[2]]):
                a = f[0]
                d = f[1]
                b = f[2]
                c = f[3]
                e = f[4]
            elif (self.distances[f[1]][f[3]] + self.distances[f[1]][f[2]] > self.distances[f[3]][f[2]] and
                self.distances[f[3]][f[2]] + self.distances[f[1]][f[2]] > self.distances[f[1]][f[3]] and
                self.distances[f[1]][f[3]] + self.distances[f[3]][f[2]] > self.distances[f[1]][f[2]]):
                d = f[0]
                a = f[1]
                b = f[2]
                c = f[3]
                e = f[4]
            else:
                print(f"Couldn't draw triangle, initializing randomly...")
                a = f[0]
                b = f[1]
                c = f[2]
                d = f[3]
                e = f[4]
                data.graph.nodes[a]['GV_position'] = (0.0, 0.0, 0.0)
                data.graph.nodes[b]['GV_position'] = (self.areaScaling, 0.0, 0.0)
                data.graph.nodes[c]['GV_position'] = (0.0, self.areaScaling, 0.0)
                data.graph.nodes[d]['GV_position'] = (0.0, 0.0, self.areaScaling)

                self.placedVertices.setdefault(a, {})
                self.placedVertices.setdefault(b, {})
                self.placedVertices.setdefault(c, {})
                self.placedVertices.setdefault(d, {})
                rand = True
                exit()

            if ~rand:
                aPos = glm.vec3(0.0, 0.0, 0.0)
                bPos = glm.vec3(self.distances[a][b] * self.areaScaling, 0.0, 0.0)

                ab = self.distances[a][b] * self.areaScaling
                ac = self.distances[a][c] * self.areaScaling
                bc = self.distances[b][c] * self.areaScaling

                cx = ac * ((ac**2 + ab**2 - bc**2)/(2 * ab * ac))
                cy = math.sqrt(ac**2 - cx**2)
                cPos = glm.vec3(cx, cy, 0.0)

                delta = glm.vec3(0.0, 0.0, 0.0) - ((aPos + bPos + cPos)/3) # Centering the ABC triangle to try to ensure D won't be coplanar with them later

                aPos += delta
                bPos += delta
                cPos += delta

                ad = self.distances[a][d]
                bd = self.distances[b][d]
                cd = self.distances[c][d]
                dPos = glm.vec3(0.0, 0.0, ((ad+bd+cd) * self.areaScaling)/3)
                
                temp = [(a, aPos), (b, bPos), (c, cPos), (d, dPos)]

                print(f"Refining initial positions of the starting filtration...")
                for i in range(20):
                    for j in range(4):
                        v = temp[j]
                        disp = glm.vec3(0.0, 0.0, 0.0)
                        for n in temp:
                            if v[0] == n[0]: continue
                            delta = n[1] - v[1]
                            if Util.magnitude(delta) <= (self.distances[v[0]][n[0]] * self.areaScaling):
                                delta = delta * (-1)
                            nDisp = math.log(Util.magnitude(delta)/(self.distances[v[0]][n[0]] * self.areaScaling))
                            disp = disp + (glm.normalize(delta) * nDisp)
                        temp[j] = (v[0], v[1] + disp)
                        print(temp)

                aPos = temp[0][1]
                bPos = temp[1][1]
                cPos = temp[2][1]
                dPos = temp[3][1]

                data.graph.nodes[a]['GV_position'] = (aPos.x, aPos.y, aPos.z)
                self.placedVertices.setdefault(a, {})
                data.graph.nodes[b]['GV_position'] = (bPos.x, bPos.y, bPos.z)
                self.placedVertices.setdefault(b, {})
                data.graph.nodes[c]['GV_position'] = (cPos.x, cPos.y, cPos.z)
                self.placedVertices.setdefault(c, {})
                data.graph.nodes[d]['GV_position'] = (dPos.x, dPos.y, dPos.z)
                self.placedVertices.setdefault(d, {})"""

            bound = self.diameterWeighted * self.areaScaling
            nodes = [(i, glm.vec4(np.random.uniform(low = bound * (-1), high = bound),
                                    np.random.uniform(low = bound * (-1), high = bound),
                                    np.random.uniform(low = bound * (-1), high = bound),
                                    np.random.uniform(low = bound * (-1), high = bound))) for i in f]

            print(f"Refining initial positions of the starting filtration...")
            
            for i in range(50):
                for j in range(len(nodes)):
                    v = nodes[j]
                    force = glm.vec4(0.0, 0.0, 0.0, 0.0)
                    for n in nodes:
                        if v[0] == n[0]: continue
                        delta = n[1] - v[1]
                        if Util.magnitude4(delta) <= (self.distances[v[0]][n[0]] * self.areaScaling):
                            delta = delta * (-1)
                        nForce = math.log(Util.magnitude4(delta)/(self.distances[v[0]][n[0]] * self.areaScaling))
                        force = force + (glm.normalize(delta) * nForce)
                    nodes[j] = (v[0], v[1] + force)                    

            for v in nodes:
                projection = self.project(v[1])
                data.graph.nodes[v[0]]['GV_position_R4'] = (v[1].x, v[1].y, v[1].z, v[1].w)
                data.graph.nodes[v[0]]['GV_position'] = (projection.x, projection.y, projection.z)
                self.placedVertices.setdefault(v[0], {})

        else:
            print(f"len(f) = {len(f)}")
            for v in f: # Find initial position pos[v] of v
                # Find closest 5 vertices among the ones already placed
                d = [(u, self.distances[u][v]) for u in self.placedVertices]
                d = sorted(d, key = lambda tup: tup[1])
                d = d[:5]

                # Place v in the barycenter of those 4
                newPos = glm.vec4(0.0, 0.0, 0.0, 0.0)
                for n in d:
                    newPos += glm.vec4(*data.graph.nodes[n[0]]['GV_position_R4'])
                newPos = newPos / 5

                data.graph.nodes[v]['GV_position_R4'] = (newPos.x, newPos.y, newPos.z, newPos.w)

                projection = self.project(newPos)
                data.graph.nodes[v]['GV_position'] = (projection.x, projection.y, projection.z)
                self.placedVertices.setdefault(v, {})
        return
    
    def runRound(self, data):
        i = self.iterationsLeft + 1 # Because we've already decreased self.iterationsLeft in runLoop before starting the rounds
        #i = self.k
        f = self.filtrations[i]
        #print(f"f = {f}")

        for v in f:
            if i > 0:
                delta = self.calculateLocalForceKK(data, v, i)
            else:
                delta = self.calculateLocalForceFR(data, v, i)
                #delta = self.calculateLocalForceKK(data, v, i)

            if 'heat' not in self.placedVertices[v]:
                heat = self.areaScaling/6
                self.placedVertices[v]['oldCos'] = 0
            else:
                heat = self.placedVertices[v]['heat']
                if Util.magnitude4(delta) != 0 and Util.magnitude4(self.placedVertices[v]['oldDelta']):
                    c = (delta * self.placedVertices[v]['oldDelta']) / Util.magnitude4(delta) * Util.magnitude4(self.placedVertices[v]['oldDelta'])
                    if self.placedVertices[v]['oldCos'] * c > 0:
                        heat += (1 + c * self.r * self.s)
                    else:
                        heat += (1 + c * self.r)
                    self.placedVertices[v]['oldCos'] = c
            
            delta = heat * (delta / Util.magnitude4(delta))
            self.placedVertices[v]['oldDelta'] = delta

            newPos = glm.vec4(*data.graph.nodes[v]['GV_position_R4']) + delta
            projection = self.project(newPos)

            data.graph.nodes[v]['GV_position_R4'] = (newPos.x, newPos.y, newPos.z, newPos.w)
            data.graph.nodes[v]['GV_position'] = (projection.x, projection.y, projection.z)
        return

    def calculateLocalForceKK(self, data, v, i):
        force = glm.vec4(0.0, 0.0, 0.0, 0.0)
        neighborhood = self.neighborhoods[v][i]['members']
        pos = glm.vec4(*data.graph.nodes[v]['GV_position_R4'])

        for n in neighborhood:
            if n == v:
                continue
            nPos = glm.vec4(*data.graph.nodes[n]['GV_position_R4'])

            euclideanDistance = Util.magnitude4(nPos - pos)
            graphDistance = self.distances[v][n]
            nForce = (euclideanDistance / (graphDistance * (self.areaScaling ** 2))) - 1
            nForce = nForce * (nPos - pos)
            force = force + nForce
        return force

    def calculateLocalForceFR(self, data, v, i):
        force = glm.vec4(0.0, 0.0, 0.0, 0.0)
        neighborhood = self.neighborhoods[v][i]['members']
        pos = glm.vec4(*data.graph.nodes[v]['GV_position_R4'])

        """for n in data.graph.neighbors(v):
            nPos = glm.vec3(*data.graph.nodes[n]['GV_position'])
            euclideanDistance = Util.magnitude(nPos - pos)
            nForce = ((euclideanDistance**2)/(self.areaScaling**2)) * (nPos - pos)
            force = force + nForce

        for n in neighborhood:
            if n == v:
                continue

            if euclideanDistance == 0:
                euclideanDistance = self.areaScaling / 1000
            nForce = 0.5 * ((self.areaScaling**2)/(euclideanDistance**2)) * (pos - nPos)

            force = force + nForce"""

        """for n in neighborhood:
            if n == v:
                continue
            nPos = glm.vec3(*data.graph.nodes[n]['GV_position'])
            delta = (pos - nPos)
            if (Util.magnitude(delta) == 0):
                continue

            nForce = glm.normalize(delta) * ((self.areaScaling**2) / Util.magnitude(delta))
            force = force + nForce

        for n in data.graph.neighbors(v):
            nPos = glm.vec3(*data.graph.nodes[n]['GV_position'])
            delta = (nPos - pos)
            if (Util.magnitude(delta) == 0):
                continue

            nForce = glm.normalize(delta) * ((Util.magnitude(delta)**2) / ((self.distances[v][n] * self.areaScaling)**2))
            if math.isnan(Util.magnitude(nForce)):
                print(f"v = {v} n = {n}")
                print(f"delta = {delta}")
                print(f"glm.normalize(delta) = {glm.normalize(delta)}")
                print(f"nForce = {nForce}")
                print(f"Util.magnitude(nForce) = {Util.magnitude(nForce)}")
                print(f"self.distances[{v}][{n}] = {self.distances[v][n]}")
                print(f"(self.distances[v][n] * self.areaScaling) = {(self.distances[v][n] * self.areaScaling)}")
                print(f"(self.distances[v][n] * self.areaScaling)**2 = {(self.distances[v][n] * self.areaScaling)**2}")
                exit()
            force = force + nForce"""

        for n in data.graph.neighbors(v):
            nPos = glm.vec4(*data.graph.nodes[n]['GV_position_R4'])
            delta = (nPos - pos)
            if (Util.magnitude4(delta) == 0): continue

            if Util.magnitude4(delta) <= (self.distances[v][n] * self.areaScaling):
                delta = delta * (-1)
            nForce = glm.normalize(delta) * math.log(Util.magnitude4(delta)/(self.distances[v][n] * self.areaScaling))
            force = force + nForce

        for n in neighborhood:
            nPos = glm.vec4(*data.graph.nodes[n]['GV_position_R4'])
            delta = (pos - nPos)
            if (Util.magnitude4(delta) == 0): continue

            nForce = glm.normalize(delta) * (1 / (Util.magnitude4(delta) ** 2))
            force = force + nForce

        if math.isnan(Util.magnitude4(force)):
            print(f"Stopping")
            quit()
        return force

    def nbrs(self, i, degreeSum):
        return 3 * (degreeSum / len(self.filtrations[i]))

    def centerGraph(self, data):
        barycenter = glm.vec3(0.0, 0.0, 0.0)
        for node in data.graph.nodes:
            barycenter += glm.vec3(*data.graph.nodes[node]['GV_position'])
        
        n = data.graph.number_of_nodes()
        barycenter = glm.vec3(0.0, 0.0, 0.0) - (barycenter/n)

        for node in data.graph.nodes:
            newPos = barycenter + glm.vec3(*data.graph.nodes[node]['GV_position'])
            data.graph.nodes[node]['GV_position'] = (newPos.x, newPos.y, newPos.z)

    def project(self, v):
        projection = v - ((glm.dot(self.r4[0], v)) * self.r4[0])
        x = glm.dot(self.r4[1], projection)
        y = glm.dot(self.r4[2], projection)
        z = glm.dot(self.r4[3], projection)
        return glm.vec3(x, y, z)

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
    def __init__(self, origin, iterations):
        self.areaRadius = 1.0 # Maximum distance from origin
        self.origin = origin
        self.c = [None, 1, 1, 1, 0.1]
        self.iterations = iterations

    def initialize(self, data):
        for node in data.graph.nodes:
            data.graph.nodes[node]['GV_position'] = (
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius),
                np.random.uniform(low = self.areaRadius*(-1), high = self.areaRadius))

    def runLoop(self, data):
        if self.iterations > 0:
            print(f"Iteration {self.iterations}")
            for u in data.graph.nodes:
                pos = glm.vec3(*data.graph.nodes[u]['GV_position'])
                totalForce = glm.vec3(0.0, 0.0, 0.0)

                for v in data.graph.nodes:
                    if v != u:
                        totalForce = totalForce + (self.c[4] * self.calculateForceExerted(data, v, u))

                newPos = pos + totalForce
                data.graph.nodes[u]['GV_position'] = (newPos.x, newPos.y, newPos.z)
            self.iterations -= 1

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
    def __init__(self, origin, iterations):
        self.areaRadius = 20 # Maximum distance from origin
        self.origin = origin
        self.iterations = iterations

    def initialize(self, data):
        if 'GV_BarycentricFixedVertices' in data.graph.graph:
            self.fixedVertices = data.graph.graph['GV_BarycentricFixedVertices']
        else:
            print(f"Face not provided, finding cycles...")
            cycles = self.cycleFinder(data)
            self.fixedVertices = cycles[np.argmax(np.array([len(c) for c in cycles]))]

        self.freeVertices = [x for x in list(data.graph.nodes) if x not in self.fixedVertices]
        self.positionFixedVertices(self.fixedVertices, data)
    
    def runLoop(self, data):
        if self.iterations > 0:
            print(f"Iteration {self.iterations}")
            for node in self.freeVertices:
                self.positionNode(data, node)
            self.iterations -= 1
    
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

    def magnitude4(vector):
        return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2 + vector.w ** 2)
        
