import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glm

from . import graph as G
from . import graphDrawer as GD
from . import graphPainter as GP
from .camera import Camera

class GraphViewer:
    def __init__(self, graph):
        self.displaySize = (800, 600)
        self.displayFOV = 50.0
        self.displayNear = 0.1
        self.displayFar = 50.0
        self.data = G.Graph(graph)

        self.cam = {}
    
    def run(self):
        graphDrawer = GD.GraphDrawer()
        graphDrawer.barycentric(self.data)
        GP.GraphPainter.random(self.data)

        #cam = Camera()
        #cam.myLookAt()

        pygame.init()
        pygame.display.set_mode(self.displaySize, DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)

        gluPerspective(45, (self.displaySize[0] / self.displaySize[1]), 0.1, 50.0)
        # field of view, aspect ratio, near clipping plane, far clipping plane

        self.cam['origin'] = glm.vec3(0.0, 0.0, 0.0)
        self.cam['position'] = glm.vec3(0.0, 0.0, -10.0)
        self.cam['up'] = glm.vec3(0.0, 1.0, 0.0)
        self.cam['look'] = self.cam['position'] * -1.0

        gluLookAt(
            self.cam['position'][0], self.cam['position'][1], self.cam['position'][2],
            self.cam['look'][0], self.cam['look'][1], self.cam['look'][2],
            self.cam['up'][0], self.cam['up'][1], self.cam['up'][2]
        )

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print('Quitting program...')
                    quit()

                if event.type == pygame.MOUSEMOTION:
                    curMousePos = pygame.mouse.get_pos()
                    if pygame.mouse.get_pressed()[2]: #Right click
                        print("dragging right click")

            
            keypress = pygame.key.get_pressed()
            if keypress[pygame.K_w]:
                glTranslatef(0,0,0.1)
            if keypress[pygame.K_s]:
                glTranslatef(0,0,-0.1)
            if keypress[pygame.K_d]:
                glTranslatef(-0.1,0,0)
            if keypress[pygame.K_a]:
                glTranslatef(0.1,0,0)

            #self.cam['origin'] = glm.vec3(0.0, 0.0, 0.0)
            #self.cam['position'] = glm.vec3(0.0, 0.0, -10.0)
                        
            #keypress = pygame.key.get_pressed()


            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.render()
            pygame.display.flip()
            pygame.time.wait(10)

    def render(self):
        for node in self.data.graph.nodes:
            self.drawNode(self.data.graph.nodes[node])
        
        for edge in self.data.graph.edges:
            self.drawEdge(edge)

    def drawNode(self, node):
        glPushMatrix()
        glTranslatef(*(node['GV_position']))

        q = gluNewQuadric()
        glColor3f(*(node['GV_color']))
        gluSphere(q, 0.1, 20, 20)
        glPopMatrix()

    def drawEdge(self, edge):
        outPos = self.data.graph.nodes[edge[0]]['GV_position']
        inPos = self.data.graph.nodes[edge[1]]['GV_position']
        
        glBegin(GL_LINES)
        glVertex3f(*outPos)
        glVertex3f(*inPos)
        glEnd()
        glFlush()
        return