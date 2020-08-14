import math
import numpy as np

class indexed_point:   
    
    roundFactor = 2
    
    def __init__(self, index = 0, x = 0.0, y = 0.0, radius = 0.0, angle = 0.0):
        if radius == 0.0 and angle == 0.0: 
            self.index = index
            self.x = round(x,self.roundFactor)
            self.y = round(y,self.roundFactor)
        elif x == 0.0 and y == 0.0:
            self.x = round(radius * math.cos(math.radians(angle)),self.roundFactor)
            self.y = round(radius * math.sin(math.radians(angle)),self.roundFactor)
            

    def asPolarCoordinate(self):
        radius = round(np.sqrt(self.x**2 + self.y**2),self.roundFactor)
        angle = round(np.degrees(np.arctan(self.y/self.x)),self.roundFactor)
        if angle < 0:
            angle = 180 + angle
        return(radius, angle)
    
    def pointIsEqualToAnotherPoint(self,anotherPoint):
        if self.x == anotherPoint.x and self.y == anotherPoint.y:
            return True
        else:
            return False