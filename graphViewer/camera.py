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
        self.origin = glm.vec3(0.0, 0.0, 0.0)
        
        self.up = glm.vec3(0.0, 1.0, 0.0) #Constant

        self.radius = 10.0
        self.theta = 0
        self.phi = 90

        self.pos = self.orbitalPos()
        self.update()        

    def update(self):
        #self.pos = self.orbitalPos()
        print(f"self.pos length {self.magnitude(self.pos)}")
        self.look = self.pos * (-1.0)
        gluLookAt(*(self.pos), *(self.look), *(self.up))

    def moveForward(self):
        print("in moveForward")
        direction = glm.normalize(self.look)
        speed = direction * 0.001
        self.pos = self.pos + speed
        self.radius = self.magnitude(self.pos)
        print(self.radius)

    def moveBack(self):
        print("in moveBack")
        direction = glm.normalize(self.look)
        speed = direction * (-0.001)
        self.pos = self.pos + speed
        self.radius = self.magnitude(self.pos)
        print(self.radius)

    def orbitalPos(self):
        return glm.vec3(
            self.radius * math.sin(math.radians(self.phi)) * math.sin(math.radians(self.theta)), 
            self.radius * math.cos(math.radians(self.phi)),
            self.radius * math.sin(math.radians(self.phi)) * math.cos(math.radians(self.theta))           
        )

    def magnitude(self, vec):
        return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

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
