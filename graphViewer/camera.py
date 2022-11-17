import pygame
import glm
import numpy as np
import quaternion
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Camera:
    def __init__(self):
        self.pos = glm.vec3(0.0, 0.0, -10)
        self.target = glm.vec3(0.0, 0.0, 0.0)

        self.front = glm.normalize(self.target - self.pos)
        self.updateLook()

        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)

        #Constants
        self.origin = glm.vec3(0.0, 0.0, 0.0)
        self.speed = 0.05

        self.updateRadius()
        self.theta = 0 # affected by vertical drag
        self.phi = 270 # affected by horizontal drag

    def activate(self):
        glLoadIdentity()
        gluLookAt(*(self.pos), *(self.look), *(self.up))

    def moveForwardBack(self, zoomDir):
        #Known issue: you can zoom in past the object, so zooming out afterwards will make it appear to flip bc you've turned around
        #Can maybe implement a check to only apply the zoom if it won't make direction change

        self.pos = self.pos + (self.front * self.speed * zoomDir)
        self.updateLook()
        self.updateRadius()

    def dragFly(self, mouseDelta):
        horizontalDir = mouseDelta[0]
        verticalDir = mouseDelta[1]
        
        delta = ((self.up * verticalDir) + (self.right * horizontalDir)) * self.speed

        self.target = self.target + delta
        self.pos = self.pos + delta

        self.updateLook()

    def dragOrbital(self, mouseDelta):
        self.phi = ((self.phi - mouseDelta[1]) + 360) % 360
        self.theta = (self.theta - mouseDelta[0]) % 360

        self.pos = self.target + self.orbitalPos()

        oldFront = glm.vec3(*(self.front))
        self.front = glm.normalize(self.target - self.pos)

        w = math.sqrt((self.magnitude(oldFront) ** 2) * (self.magnitude(self.front) ** 2)) + glm.dot(oldFront, self.front)
        q = np.quaternion(w, *glm.cross(oldFront, self.front))

        qR = np.quaternion(0, *self.right)
        qR = np.quaternion.normalized(q * qR * q.conjugate())
        self.right = glm.normalize(glm.vec3(qR.x, qR.y, qR.z))

        qU = np.quaternion(0, *self.up)
        qU = np.quaternion.normalized(q * qU * q.conjugate())
        self.up = glm.normalize(glm.vec3(qU.x, qU.y, qU.z))

        self.updateLook()

    #Update functions

    def updateLook(self):
        self.look = self.pos + self.front
    
    def updateRadius(self):
        self.radius = self.magnitude(self.pos - self.target)

    #Utilities

    def magnitude(self, vec):
        return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

    def orbitalPos(self):
        return glm.vec3(
            self.radius * math.sin(math.radians(self.phi)) * math.sin(math.radians(self.theta)), 
            self.radius * math.cos(math.radians(self.phi)),
            self.radius * math.sin(math.radians(self.phi)) * math.cos(math.radians(self.theta))           
        )
