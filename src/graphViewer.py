import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class GraphViewer:
    def __init__(self):
        self.displaySize = (800, 600)
        self.displayFOV = 50.0
        self.displayNear = 0.1
        self.displayFar = 50.0

    def run(self):
        pygame.init()

        pygame.display.set_mode(self.displaySize, DOUBLEBUF | OPENGL)

        glEnable(GL_DEPTH_TEST)

        gluPerspective(45, (self.displaySize[0] / self.displaySize[1]), 0.1, 50.0)
        # field of view, aspect ratio, near clipping plane, far clipping plane

        glTranslatef(0.0, 0.0, -5)
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
        self.drawNode()
    
    def drawNode(self):
        glEnable(GL_COLOR_MATERIAL)

        sphere = gluNewQuadric()
        glColor3f(1.0, 0.0, 0.0)
        gluSphere(sphere, 1.4, 20, 20)