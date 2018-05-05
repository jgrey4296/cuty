from .operations import *
from .Node import Node

class RBTree:

    def __init__(self, cmpFunc=None, eqFunc=None):
        """ Initialise the rb tree container, ie: the node list """

        #Default Comparison and Equality functions with dummy data ignored
        if cmpFunc is None:
            cmpFunc = lambda a,b,cd: a < b
        if eqFunc is None:
            eqFunc = lambda a,b,cd: a.value == b
            
        self.nodes = []
        self.values = set()
        self.root = None
        self.cmpFunc = cmpFunc
        self.eqFunc = eqFunc

    def __len__(self):
        return len(self.nodes)
        
    def insert(self,*args, data=None, cmpData=None):
        nodes = []
        for x in args:
            newNode = self.rb_insert(x, data=data, cmpData=cmpData)
            nodes.append(nodes)
        return nodes
        
    def rb_insert(self,value,data=None, cmpData=None):
        """ Insert a value into the tree """
        newNode = Node(value,data=data)
        self.nodes.append(newNode)
        self.values.add(value)
        
        y = None
        x = self.root
        
        while x is not None:
            y = x
            if self.cmpFunc(newNode.value, x.value, cmpData):
                x = x.left
            else:
                x = x.right
        newNode.parent = y
        if y is None:
            self.root = newNode
        elif self.cmpFunc(newNode.value, y.value, cmpData):
            y.left = newNode
        else:
            y.right = newNode
        newNode.left = None
        newNode.right = None
        rbtreeFixup(self,newNode)
        return newNode

    def delete(self, *args):
        """ Delete a value from the tree """
        for node in args:
            assert(isinstance(node, Node))
            assert(node.value in self.values)
            rbTreeDelete(self,node)
            self.values.remove(node.value)
            self.nodes.remove(node)

    def search(self, value, cmpFunc=None, eqFunc=None,
               cmpData=None, closest=False):
        """ Search the tree for a value """
        if cmpFunc is None:
            cmpFunc = self.cmpFunc
        if eqFunc is None:
            eqFunc = self.eqFunc

        parent = None
        current = self.root
        while current is not None and not eqFunc(current,value, cmpData):
            parent = current
            if cmpFunc(current.value, value, cmpData):
                current = current.left
            else:
                current = current.right

        if closest and current is None:
            return parent
        else:
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

    def get_chain(self):
        """ Get the sequence of leaf values, from left to right """
        if self.root is None:
            return []
        chain = []
        current = self.root.getMin()
        while current is not None:
            chain.append(current)
            current = current.getSuccessor()
        return chain
