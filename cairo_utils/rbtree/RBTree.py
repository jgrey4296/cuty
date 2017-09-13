from .operations import *

class RBTree:

    def __init__(self):
        """ Initialise the rb tree container, ie: the node list """
        self.nodes = []
        self.root = None
        self.nextId = 0

    def insert(self,*args):
        for x in args:
            self.rb_insert(x)
        
    def rb_insert(self,value,data=None):
        """ Insert a value into the tree """
        newNode = Node(self.nextId,value,data=data)
        self.nodes.append(newNode)
        self.nextId += 1
        
        y = None
        x = self.root
        
        while x is not None:
            y = x
            if newNode.value < x.value:
                x = x.left
            else:
                x = x.right
        newNode.parent = y
        if y is None:
            self.root = newNode
        elif newNode.value < y.value:
            y.left = newNode
        else:
            y.right = newNode
        newNode.left = None
        newNode.right = None
        rbtreeFixup(self,newNode)

    def delete(self,node):
        """ Delete a value from the tree """
        rbTreeDelete(self,node)

    def search(self,value):
        """ Search the tree for a value """
        current = self.root
        while current is not None and current.value != value:
            if value > current.value:
                current = current.right
            else:
                current = current.left
        return current
                
    def min(self):
        """ Get the min value of the tree """
        return self.root.getMin()

    def max(self):
        """ Get the max value of the tree """
        return self.root.getMax()

    def countBlackHeight(self,node=None):
    """ Given a node, count all paths and check they have the same black height """
    if node is None:
        if self.root is None:
            return None
        node = self.root
    stack = [node]
    leaves = []
    while len(stack) > 0:
        current = stack.pop()
        if current.left is None and current.right is None:
            leaves.append(current)
        else:
            if current.left is not None:
                stack.append(current.left)
            if current.right is not None:
                stack.append(current.right)

    allHeights = [x.getBlackHeight(node) for x in leaves]
    return allHeights

