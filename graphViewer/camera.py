import pygame
import glm
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class Camera:
    def __init__(self):
        self.pos = glm.vec3(0.0, 0.0, -10)
        self.target = glm.vec3(0.0, 0.0, 0.0)
        self.look = glm.normalize(self.target - self.pos)
        self.origin = glm.vec3(0.0, 0.0, 0.0)
        
        self.up = glm.vec3(0.0, 1.0, 0.0) #Constant

        gluLookAt(*(self.pos), *(self.look), *(self.up))

    """def lookAt(self, target):
        self.target = target
        self.direction = glm.normalize(self.target - self.pos)

        view = np.ndarray((4,4))
        view[0][0] = self.right.x
        view[1][0] = self.right.y
        view[2][0] = self.right.z
        view[0][1] = self.up.x
        view[1][1] = self.up.y
        view[2][1] = self.up.z
        view[0][2] = -self.direction.x
        view[1][2] = -self.direction.y
        view[2][2] = -self.direction.z
        view[3][0] = glm.dot(self.right, self.pos)
        view[3][1] = glm.dot(self.up, self.pos)
        view[3][2] = glm.dot(self.direction, self.pos)

        return(view)"""
