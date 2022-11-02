import pygame
import glm
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Camera:
    def __init__(self):
        self.pos = glm.vec3(0.0, 0.0, -10)
        self.target = glm.vec3(0.0, 0.0, 0.0)
        self.direction = glm.normalize(self.pos - self.target)

        self.right = glm.vec3(0.0, 1.0, 0.0)
        self.up = glm.cross(self.direction, self.right)
        return

    def myLookAt(self):
        print(self.pos)
        print(self.target)
        print(self.direction)
        print(self.up)