class Tree:

    def __init__(self,value,root=False,data=None):
        self.value = value
        self.data = data
        self.left = None
        self.right = None
        self.root = root

    def isLeaf(self):
        return self.left is None and self.right is None

    def isRoot(self):
        return self.root

    def insert(self,value,data=None):
        insertOnLeft = value < self.value
        if insertOnLeft:
            if self.left is None:
                self.left = Tree(value,data=data)
            else:
                self.left.insert(value,data=data)
        else:
            if self.right is None:
                self.right = Tree(value,data=data)
            else:
                self.right.insert(value,data=data)

                
    def __str__(self):
        if self.left is not None:
            leftString = self.left.__str__()
        else:
            leftString = "()"
        if self.right is not None:
            rightString = self.right.__str__()
        else:
            rightString = "()"            
        return "( V: {} Left: {},Right: {} )".format(self.value,leftString,rightString)


    def search(self,value):
        if value == self.value:
            return self
        elif self.isLeaf():
            return None
        elif value < self.value:
            if self.left is None:
                return None
            else:
                return self.left.search(value)
        else:
            if self.right is None:
                return None
            else:
                return self.right.search(value)

    def getRange(self,l,r):
        values = []
        if l < self.value and self.left is not None:
            values.extend(self.left.getRange(l,r))

        if l < self.value and self.value <= r:
            values.append(self)
            
        if self.value <= r and self.right is not None:
            values.extend(self.right.getRange(l,r))

        return values
            
