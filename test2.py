import math
import glm

def magnitude(vec):
    return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)

def upAngle(up, right):
    globalUp = glm.vec3(0.0, 1.0, 0.0)
    globalRight = glm.vec3(-1.0, 0.0, 0.0)

    angleUp = math.atan2(magnitude(glm.cross(up, globalUp)), glm.dot(up, globalUp))
    angleRight = math.atan2(magnitude(glm.cross(right, globalRight)), glm.dot(right, globalRight))

    return math.degrees(angleUp), math.degrees(angleRight)

"""up = glm.vec3(-1.0, 1.0, 0.0)
right = glm.vec3(-1.0, -1.0, 0.0)
upAngle(up, right)

up = glm.vec3(1.0, 1.0, 0.0)
right = glm.vec3(-1.0, 1.0, 0.0)
upAngle(up, right)

up = glm.vec3(-1.0, -1.0, 0.0)
right = glm.vec3(1.0, -1.0, 0.0)
upAngle(up, right)

up = glm.vec3(0.0, -1.0, 0.0)
right = glm.vec3(1.0, 0.0, 0.0)
upAngle(up, right)"""

up = glm.vec3(0.0, -1.0, 0.0)
right = glm.vec3(1.0, 0.0, 0.0)
upAngle(up, right)

up = glm.vec3(0.0, -1.0, 0.0)
right = glm.vec3(-1.0, 0.0, 0.0)
upAngle(up, right)