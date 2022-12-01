import pygame
import glm
import numpy as np
#import quaternion
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Camera:
    def __init__(self):
        self.pos = glm.vec3(0.0, 0.0, -200)
        self.target = glm.vec3(0.0, 0.0, 0.0)

        self.front = glm.normalize(self.target - self.pos)
        self.updateLook()

        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.right = glm.vec3(1.0, 0.0, 0.0)

        #Constants
        self.origin = glm.vec3(0.0, 0.0, 0.0)
        self.speed = 0.05
        self.angSpeed = 0.5

        self.updateRadius()
        self.theta = 0 # affected by vertical drag
        self.phi = 270 # affected by horizontal drag

    def activate(self):
        glLoadIdentity()
        gluLookAt(*(self.pos), *(self.look), *(self.up))

    def moveForwardBack(self, zoomDir):
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

    def dragLook(self, mouseDelta):
        x = mouseDelta[0]
        y = mouseDelta[1] * -1
        return

    def dragOrbital(self, mouseDelta):
        mouseDelta = (mouseDelta[0], mouseDelta[1] * -1)
        delta = self.transformMouseDelta(*mouseDelta)
        #print(f"{mouseDelta} {delta}")

        self.phi = ((self.phi + (delta[1] * self.angSpeed)) + 360) % 360
        self.theta = (self.theta - (delta[0] * self.angSpeed) + 360) % 360

        self.pos = self.target + self.orbitalPos()

        oldFront = glm.vec3(*(self.front))
        self.front = glm.normalize(self.target - self.pos)

        w = math.sqrt((self.magnitude(oldFront) ** 2) * (self.magnitude(self.front) ** 2)) + glm.dot(oldFront, self.front)
        q = glm.quat(w, *glm.cross(oldFront, self.front))

        qR = glm.quat(0, *self.right)
        qR = glm.normalize(q * qR * glm.conjugate(q))
        self.right = glm.normalize(glm.vec3(qR.x, qR.y, qR.z))

        qU = glm.quat(0, *self.up)
        qU = glm.normalize(q * qU * glm.conjugate(q))
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

    def transformMouseDelta(self, x, y):
        v = glm.vec2(x, y)
        right = glm.vec2(1.0, 0.0)
        vMag = math.sqrt(x ** 2 + y ** 2)

        # Find the angle the vec2 makes with the positive 2d x-axis (relative to screen space)
        try:
            currentAngle = math.degrees(math.acos(glm.dot(v, right) / vMag))
            if y < 0:
                currentAngle = 360 - currentAngle
        except ZeroDivisionError:
            #print(f"ZeroDivisionError {v}")
            return v

        # Find the angle the camera up makes with the global up, i.e. by how much the camera has turned
        globalUp = glm.vec3(0.0, 1.0, 0.0)
        globalRight = glm.vec3(-1.0, 0.0, 0.0)

        angleUp = math.atan2(self.magnitude(glm.cross(self.up, globalUp)), glm.dot(self.up, globalUp))
        angleRight = math.atan2(self.magnitude(glm.cross(self.right, globalRight)), glm.dot(self.right, globalRight))

        # Apply angle to the vec2
        if math.isclose(angleUp, angleRight):
            newAngle = (currentAngle - angleUp + 360) % 360
        else:
            newAngle = (currentAngle + angleUp + 360) % 360

        # Generate new vec2
        a = vMag * math.cos(math.radians(newAngle))
        b = vMag * math.sin(math.radians(newAngle))

        return (a,b)