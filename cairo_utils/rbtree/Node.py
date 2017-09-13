class Node:

    def __init__(self,id,value,parent=None,data=None):
        self.id = id
        #Children:
        self.left = None
        self.right = None
        #Parent:
        self.parent = parent
        #Node Date:
        self.colour = RED
        self.value = value
        self.data = data

    def getBlackHeight(self,parent=None):
        current = self
        height = 0
        while current is not None:
            if current.colour == BLACK:
                height += 1
            if current == parent:
                current = None
            else:
                current = current.parent
        return height
        
    def __str__(self):
        if self.colour == RED:
            colour = "R"
        else:
            colour = "B"
        return "({}_{} {} {})".format(colour,self.value,self.left,self.right)

    def getMin(self):
        current = self
        while current.left is not None:
            current = current.left
        return current

    def getMax(self):
        current = self
        while current.right is not None:
            current = current.right
        return current
