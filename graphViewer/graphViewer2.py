import networkx as nx
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class GraphViewer:
    def __init__(self, graph):
        self.displaySize = (800, 600)
        self.displayFOV = 50.0
        self.displayNear = 0.1
        self.displayFar = 50.0
        self.data = Data(graph)

    def run(self):
        pygame.init()

        pygame.display.set_mode(self.displaySize, DOUBLEBUF | OPENGL)

        glEnable(GL_DEPTH_TEST)

        gluPerspective(45, (self.displaySize[0] / self.displaySize[1]), 0.1, 50.0)
        # field of view, aspect ratio, near clipping plane, far clipping plane

        glTranslatef(0.0, 0.0, -10)
        # starting point of the camera, sounds like

        glRotatef(0, 0, 0, 0)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print('Quitting program...')
                    quit()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.render()
            pygame.display.flip()
            pygame.time.wait(10)

    def render(self):
        self.nodes = [
            Node((-1.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
            Node((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            Node((1.5, 0.0, 0.0), (0.0, 0.0, 1.0))
        ]
        
        glEnable(GL_COLOR_MATERIAL)

        for node in self.data.G.nodes:
            self.drawNode(self.data.G.nodes[node])

        for edge in self.data.G.edges:
            self.drawEdge(edge)

        glTranslatef(0.0, 0.0, 0.001)
    
    def drawNode(self, node):
        glPushMatrix()
        glTranslatef(*(node['GV_position']))

        q = gluNewQuadric()
        glColor3f(*(node['GV_color']))
        gluSphere(q, 0.1, 20, 20)
        glPopMatrix()

    def drawEdge(self, edge):
        outPos = self.data.G.nodes[edge[0]]['GV_position']
        inPos = self.data.G.nodes[edge[1]]['GV_position']
        
        glBegin(GL_LINES)
        glVertex3f(*outPos)
        glVertex3f(*inPos)
        glEnd()
        glFlush()
        return

class Data:
    def __init__(self, graph):
        self.G = graph
        nx.set_node_attributes(self.G, (0.0, 0.0, 0.0), "GV_position")
        nx.set_node_attributes(self.G, (1.0, 1.0, 1.0), "GV_color")

        #self.G.nodes[1]['GV_color'] = (1.0, 0.0, 0.0)
        #self.G.nodes[2]['GV_color'] = (0.0, 1.0, 0.0)
        #self.G.nodes[3]['GV_color'] = (0.0, 0.0, 1.0)
         
        self.eades()

        self.randomizeColors()

    def randomizeColors(self):
        for node in self.G.nodes:
            self.G.nodes[node]['GV_color'] = (  np.random.uniform(low = 0.0, high = 1.00001),
                                                np.random.uniform(low = 0.0, high = 1.00001),
                                                np.random.uniform(low = 0.0, high = 1.00001))

    def eades(self):
        areaRadius = 1.5 # maximum distance from origin

        for node in self.G.nodes:
            self.G.nodes[node]['GV_position'] = (   np.random.uniform(low = areaRadius*(-1), high = areaRadius),
                                                    np.random.uniform(low = areaRadius*(-1), high = areaRadius),
                                                    np.random.uniform(low = areaRadius*(-1), high = areaRadius))
        return


class Node:
    def __init__(self, position, color):
        self.position = position
        self.color = color