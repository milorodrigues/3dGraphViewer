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

        pygame.init()
        pygame.display.set_mode(self.displaySize, DOUBLEBUF | OPENGL)
        glEnable(GL_DEPTH_TEST)

        self.cam = Camera()

        #print(f"{self.cam.look}\n{self.cam.front}")

        self.verbose = False
        
        while True:
            curMousePos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print('Quitting program...')
                    quit()

                if event.type == pygame.MOUSEMOTION:
                    if pygame.mouse.get_pressed()[1]: #Wheel click
                        pygame.event.set_grab(True)
                        oldMousePos = (curMousePos[0], curMousePos[1])
                        curMousePos = pygame.mouse.get_pos()
                        delta = tuple(map(lambda i, j: i - j, curMousePos, oldMousePos))
                        self.cam.dragOrbital(delta)
                    if not pygame.mouse.get_pressed()[1]: #Let go of wheel click
                        pygame.event.set_grab(False)

                    if pygame.mouse.get_pressed()[2]: #Right click
                        #self.verbose = True
                        pygame.event.set_grab(True)
                        oldMousePos = (curMousePos[0], curMousePos[1])
                        curMousePos = pygame.mouse.get_pos()
                        delta = tuple(map(lambda i, j: i - j, curMousePos, oldMousePos))
                        self.cam.dragFly(delta)
                    if not pygame.mouse.get_pressed()[2]: #Let go of right click
                        self.verbose = False
                        pygame.event.set_grab(False)

                if event.type == pygame.MOUSEWHEEL:
                    if (event.y != 0):
                        self.cam.zoom(event.y)

            
            """keypress = pygame.key.get_pressed()
            if keypress[pygame.K_w]:
                cam.dragOrbital(1.0, 0.0)
            if keypress[pygame.K_s]:
                cam.dragOrbital(-1.0, 0.0)
            if keypress[pygame.K_d]:
                cam.dragOrbital(0.0, -1.0)
            if keypress[pygame.K_a]:
                cam.dragOrbital(0.0, 1.0)"""
            

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (self.displaySize[0] / self.displaySize[1]), 0.0001, 5000.0) # field of view, aspect ratio, near clipping plane, far clipping plane
            #glOrtho(0, self.displaySize[0], self.displaySize[1], 0, -1, 5000.0) # left, right, bottom, top, near, far

            glMatrixMode(GL_MODELVIEW)

            self.cam.activate()

            self.drawGraph()
            
            pygame.display.flip()
            pygame.time.wait(30)

    def drawGraph(self):
        for node in self.data.graph.nodes:
            self.drawNode(self.data.graph.nodes[node])
        
        for edge in self.data.graph.edges:
            self.drawEdge(edge)

        glPushMatrix()
        glTranslate(*(self.cam.target))
        q = gluNewQuadric()
        glColor3f(1.0, 1.0, 1.0)
        gluSphere(q, 0.15, 20, 20)
        glPopMatrix()

    def drawNode(self, node):
        glPushMatrix()
        glTranslatef(*(node['GV_position']))

        if(self.verbose):
            a = (GLfloat * 16)()
            print(list(glGetFloatv(GL_MODELVIEW_MATRIX, a)))

        q = gluNewQuadric()
        glColor3f(*(node['GV_color']))
        gluSphere(q, 0.1, 20, 20)
        glPopMatrix()

    def drawEdge(self, edge):
        outPos = self.data.graph.nodes[edge[0]]['GV_position']
        inPos = self.data.graph.nodes[edge[1]]['GV_position']
        
        glPushMatrix()
        glBegin(GL_LINES)
        glVertex3f(*outPos)
        glVertex3f(*inPos)
        glEnd()
        glFlush()
        glPopMatrix()