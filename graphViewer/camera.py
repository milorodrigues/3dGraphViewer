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
        self.angSpeed = 1

        self.radius = self.magnitude(self.look)
        self.theta = 0 # affected by vertical drag
        self.phi = 270 # affected by horizontal drag

    def activate(self):
        glLoadIdentity()
        gluLookAt(*(self.pos), *(self.look), *(self.up))

    def zoom(self, zoomDir):
        #Known issue: you can zoom in past the object, so zooming out afterwards will make it appear to flip bc you've turned around
        #Can maybe implement a check to only apply the zoom if it won't make direction change
        direction = glm.normalize(self.look)
        self.pos = self.pos + (direction * self.speed * zoomDir)
        self.updateLook()
        self.updateRadius()

    def dragFly(self, mouseDelta):
        horizontalDir = mouseDelta[0]
        verticalDir = mouseDelta[1]

        right = glm.normalize(glm.cross(self.look, self.up)) * -1.0
        delta = ((self.up * verticalDir) + (right * horizontalDir)) * self.speed

        self.target = self.target + delta
        self.pos = self.pos + delta
        self.updateLook()
        self.updateRadius()

    def dragOrbital(self, mouseDelta):
        #Known issue: object flips horizontally at phi = 0 and phi = 180. pending to find out why.
        print(f"in dragOrbital({mouseDelta})")

        self.phi = ((self.phi - mouseDelta[1]) + 360) % 360
        self.theta = (self.theta - mouseDelta[0]) % 360

        print(f"phi = {self.phi} theta = {self.theta}")

        self.pos = self.target + self.orbitalPos()

        self.updateLook()

    #Update functions

    def updateLook(self):
        self.look = self.target - self.pos
    
    def updateRadius(self):
        self.magnitude(self.pos - self.target)

    #Utilities

    def magnitude(self, vec):
        return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

    def orbitalPos(self):
        return glm.vec3(
            self.radius * math.sin(math.radians(self.phi)) * math.sin(math.radians(self.theta)), 
            self.radius * math.cos(math.radians(self.phi)),
            self.radius * math.sin(math.radians(self.phi)) * math.cos(math.radians(self.theta))           
        )
