from .operations import *
from .Node import Node

class RBTree:

    def __init__(self, cmp=None):
        """ Initialise the rb tree container, ie: the node list """
        if cmp is None:
            cmp = lambda a,b: a < b
        
        self.nodes = []
        self.values = set()
        self.root = None
        self.nextId = 0
        self.cmp = cmp

    def __len__(self):
        return len(self.nodes)
        
    def insert(self,*args):
        for x in args:
            self.rb_insert(x)
        
    def rb_insert(self,value,data=None):
        """ Insert a value into the tree """
        newNode = Node(self.nextId,value,data=data)
        self.nodes.append(newNode)
        self.values.add(value)
        self.nextId += 1
        
        y = None
        x = self.root
        
        while x is not None:
            y = x
            if self.cmp(newNode.value,x.value):
                x = x.left
            else:
                x = x.right
        newNode.parent = y
        if y is None:
            self.root = newNode
        elif self.cmp(newNode.value, y.value):
            y.left = newNode
        else:
            y.right = newNode
        newNode.left = None
        newNode.right = None
        rbtreeFixup(self,newNode)

    def delete(self,node):
        """ Delete a value from the tree """
        assert(isinstance(node, Node))
        assert(node.value in self.values)
        rbTreeDelete(self,node)
        self.values.remove(node.value)
        self.nodes.remove(node)

    def search(self,value):
        """ Search the tree for a value """
        current = self.root
        while current is not None and current.value != value:
            if self.cmp(current.value, value):
                current = current.left
            else:
                current = current.right
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

