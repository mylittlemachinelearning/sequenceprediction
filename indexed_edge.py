class indexed_edge:
    
    def __init__(self, index, startPoint, endPoint):
        self.index = index
        self.startPoint = startPoint
        self.endPoint = endPoint
    
    def startsOrEndsWithPoint(self, point):
        if self.startPoint.x == point.x and self.startPoint.y == point.y :
            return True
        elif self.endPoint.x == point.x and self.endPoint.y == point.y :
            return True
        else:
            return False
    
    def edgeEqualsToAnotherEdge(self, anotherEdge):
        if self.startsOrEndsWithPoint(anotherEdge.startPoint) and self.startsOrEndsWithPoint(anotherEdge.endPoint):
            return True
        else:
            return False
    
    def switchOrientation(self):
        newStartpoint = self.endPoint
        newEndpoint = self.startPoint
        self.startPoint = newStartpoint
        self.endPoint = newEndpoint