import pygame
import glm
import numpy as np
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Camera:
    def __init__(self):
        self.pos = glm.vec3(0.0, 0.0, -10)
        self.target = glm.vec3(0.0, 0.0, 0.0)
        self.look = self.target - self.pos

        #Constants
        self.origin = glm.vec3(0.0, 0.0, 0.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.speed = 0.1

        self.radius = self.magnitude(self.look)
        #self.theta = 0
        #self.phi = 90

    def zoom(self, zoomDir):
        #Known issue: you can zoom in past the object, so zooming out afterwards will make it appear to flip bc you've turned around
        #Can implement a check to only apply the zoom if it won't make direction change
        direction = glm.normalize(self.look)
        self.pos = self.pos + (direction * self.speed * zoomDir)
        self.updateLook()
        self.updateRadius()

    def dragTarget(self, mouseDelta):
        horizontalDir = mouseDelta[0]
        verticalDir = mouseDelta[1]

        right = glm.normalize(glm.cross(self.look, self.up)) * -1.0
        delta = ((self.up * verticalDir) + (right * horizontalDir)) * self.speed

        self.target = self.target + delta
        self.pos = self.pos + delta
        self.updateLook()
        self.updateRadius()

    def activate(self):
        glLoadIdentity()
        gluLookAt(*(self.pos), *(self.look), *(self.up))

    #Update functions

    def updateLook(self):
        self.look = self.target - self.pos
    
    def updateRadius(self):
        self.magnitude(self.pos - self.target)

    #Utilities

    def magnitude(self, vec):
        return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

    """def orbitalPos(self):
        return glm.vec3(
            self.radius * math.sin(math.radians(self.phi)) * math.sin(math.radians(self.theta)), 
            self.radius * math.cos(math.radians(self.phi)),
            self.radius * math.sin(math.radians(self.phi)) * math.cos(math.radians(self.theta))           
        )
    
    def lookAt(self, target):
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
