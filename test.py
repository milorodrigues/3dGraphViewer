"""print(f"{mouseDelta} ", end='')
        alpha = math.acos(glm.dot(glm.vec3(0.0,1.0,0.0), self.up))
        beta = math.acos(glm.dot(glm.vec3(1.0,0.0,0.0), self.right))

        # Alpha being the angle between vectors a and b, alpha = arccosine(a dot b / (mag(a) * mag(b)))
        
        mouseDeltaMagnitude = math.sqrt(mouseDelta[0] ** 2 + mouseDelta[1] ** 2)
        mouseDeltaAngle = math.acos(glm.dot(glm.vec2(-1.0, 0.0), glm.vec2(*mouseDelta)) / mouseDeltaMagnitude)
        upAngle = math.acos(glm.dot(glm.vec3(0.0,1.0,0.0), self.up)) #both are unit vectors so denominator is 1*1=1

        deltaAngle = """

import math
import glm

# x points to viewer's right hand, y points up
#remember to invert the mouse's y in the program, bc in pygame y points down
def transform2Dvector(x, y, angle):
    v = glm.vec2(x, y)
    right = glm.vec2(1.0, 0.0)
    vMag = math.sqrt(x ** 2 + y ** 2)

    currentAngle = math.degrees(math.acos(glm.dot(v, right) / vMag))
    if y < 0:
        currentAngle = 360 - currentAngle

    newAngle = (currentAngle + angle + 360) % 360

    print(f"{(x, y)} {currentAngle} {newAngle}")

    a = vMag * math.cos(math.radians(newAngle))
    b = vMag * math.sin(math.radians(newAngle))

    return (a,b)

def transform2DvectorATan(x, y, angle):
    v = glm.vec2(x, y)
    right = glm.vec2(1.0, 0.0)
    vMag = math.sqrt(x ** 2 + y ** 2)
    
    cross = glm.cross(v, right)
    crossMag = math.sqrt(cross.x ** 2 + cross.y ** 2)
    dot = glm.dot(v, right)
    
    """currentAngle = math.degrees(math.acos(glm.dot(v, right) / vMag))
    if y < 0:
        currentAngle = 360 - currentAngle"""
    
    currentAngle = math.degress(math.atan2(crossMag, dot))
    print(f"{(x, y)} {currentAngle}")

    """newAngle = (currentAngle + angle + 360) % 360

    print(f"{(x, y)} {currentAngle} {newAngle}")

    a = vMag * math.cos(math.radians(newAngle))
    b = vMag * math.sin(math.radians(newAngle))

    return (a,b)"""

"""transform2Dvector(1.0, 1.0, 0)
transform2Dvector(-1.0, 1.0, 0)
transform2Dvector(-1.0, -1.0, 0)
transform2Dvector(1.0, -1.0, 0)"""

transform2Dvector(7.713, 9.193, 0)
transform2DvectorATan(7.713, 9.193, 0)
transform2Dvector(-7.713, 9.193, 0)
transform2DvectorATan(-7.713, 9.193, 0)
transform2Dvector(-7.713, -9.193, 0)
transform2DvectorATan(-7.713, -9.193, 0)
transform2Dvector(7.713, -9.193, 0)
transform2DvectorATan(7.713, -9.193, 0)

"""transform2Dvector(1.0, 0.0, 0)
transform2Dvector(0.0, 1.0, 0)
transform2Dvector(-1.0, 0.0, 0)
transform2Dvector(0.0, -1.0, 0)"""

#print(transform2Dvector(7.713, 9.193, -100))
"""print(math.sin(math.radians(50)))
print(math.cos(math.radians(50)))
print(math.sin(math.radians(130)))
print(math.cos(math.radians(130)))"""