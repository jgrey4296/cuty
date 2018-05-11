from functools import partial
from types import FunctionType
import logging as root_logger

from .operations import *
from .Node import Node
from .ComparisonFunctions import *

logging = root_logger.getLogger(__name__)

class RBTree:
    """ A Red-Black Tree Implementation
    Properties of RBTrees:
    1) Every node is Red Or Black
    2) The root is black
    3) Every leaf is Black, leaves are null nodes
    4) If a node is red, it's children are black
    5) All paths from a node to its leaves contain the same number of black nodes
    """
    
    def __init__(self, cmpFunc=None, eqFunc=None, cleanupFunc=None):
        """ Initialise the rb tree container, ie: the node list """

        #Default Comparison and Equality functions with dummy data ignored
        if cmpFunc is None:
            cmpFunc = default_comparison
        if eqFunc is None:
            eqFunc = default_equality
        if cleanupFunc is None:
            cleanupFunc = lambda x: []
        assert(isinstance(cmpFunc, (partial, FunctionType)))
        assert(isinstance(eqFunc, (partial, FunctionType)))
        
        self.nodes = []
        self.root = None
        self.cmpFunc = cmpFunc
        self.eqFunc = eqFunc
        self.cleanupFunc = cleanupFunc

    #------------------------------
    # def Basic Access
    #------------------------------
    
    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        if self.root is None:
            return "RBTree(_)"

        return "RBTree( Len: {})".format(len(self))

    def min(self):
        """ Get the min value of the tree """
        if self.root is None:
            return None
        return self.root.min()

    def max(self):
        """ Get the max value of the tree """
        if self.root is None:
            return None
        return self.root.max()

    def get_chain(self):
        """ Get the sequence of leaf values, from left to right """
        if self.root is None:
            return []
        chain = []
        current = self.root.min()
        while current is not None:
            chain.append(current)
            current = current.getSuccessor()
        return chain

    def get_successor_triple(self,node):
        if node is None:
            return None
        a = node
        b = a.getSuccessor()
        if b != None:
            c = b.getSuccessor()
            if c != None:
                return (a,b,c)
        return None

    def get_predecessor_triple(self,node):
        if node == None:
            return None
        a = node
        b = a.getPredecessor()
        if b != None:
            c = b.getPredecessor()
            if c != None:
                return (c,b,a)
        return None
    
    #------------------------------
    # def Query
    #------------------------------

    def search(self, value, cmpFunc=None, eqFunc=None,
               cmpData=None, closest=False):
        """ Search the tree for a value """
        if cmpFunc is None:
            cmpFunc = self.cmpFunc
        if eqFunc is None:
            eqFunc = self.eqFunc
        parent = self.root
        current = self.root
        comp = Directions.CENTRE
        while current is not None and not eqFunc(current,value, cmpData):
            parent = current
            comp = cmpFunc(current, value, cmpData)
            if comp is Directions.LEFT:
                current = current.left
            elif comp is Directions.RIGHT:
                current = current.right
            else:
                break

        if closest and current is None:
            #closest non-exact match found
            return (parent, comp)
        elif current is None:
            #nothing found
            return (None, None)
        else:
            #exact match found
            return (current, comp)
    
    
    #------------------------------
    # def Public Update
    #------------------------------
    
    def update_values(self, func, funcData):
        """ Call a function on all stored values,
        function signature:  f(value, funcData)
        """
        assert(isinstance(func, (FunctionType, partial)))
        for node in self.nodes:
            func(node.value, funcData)
    
    def insert(self,*args, data=None, cmpData=None):
        nodes = []
        for x in args:
            newNode = self.__insert(x, data=data, cmpData=cmpData)
            nodes.append(newNode)
        return nodes

    def delete(self, *args, cleanupFunc=None):
        """ Delete a value from the tree """
        if cleanupFunc is None:
            cleanupFunc = self.cleanupFunc
        toRemove = set(args)
        while bool(toRemove):
            target = toRemove.pop()
            assert(isinstance(target, Node))
            if target not in self.nodes:
                continue
            toRemove.update(cleanupFunc(target))
            rbTreeDelete(self,target)
            self.nodes.remove(target)

    def delete_value(self,*args, cmpFunc=None, eqFunc=None, cleanupFunc=None,
                     cmpData=None):
        for val in args:
            node,direction = self.search(val, cmpFunc=cmpFunc, eqFunc=eqFunc, cmpData=cmpData)
            if node is not None:
                self.delete(node, cleanupFunc=cleanupFunc)
            
    #------------------------------
    # def Private Update
    #------------------------------
    def __insert(self,value,data=None, cmpData=None):
        """ Insert a value into the tree """
        parent, direction = self.search(value, closest=True, cmpData=cmpData)
        if direction is Directions.LEFT:
            return self.insert_predecessor(parent, value, data=data)
        else:
            return self.insert_successor(parent, value, data=data)


    def insert_successor(self,existing_node,newValue, data=None):
        assert(existing_node is None or isinstance(existing_node, Node))
        new_node = Node(newValue, data=data)
        self.nodes.append(new_node)
        if existing_node is None:
            self.root = new_node
        else:
            existing_node.add_right(new_node)
        self.__balance(new_node)
        return new_node

    def insert_predecessor(self,existing_node,newValue, data=None):
        assert(existing_node is None or isinstance(existing_node, Node))
        new_node = Node(newValue, data=data)
        self.nodes.append(new_node)
        if existing_node == None:
            self.root = new_node
        else:
            existing_node.add_left(new_node)
        self.__balance(new_node)
        return new_node

    def __balance(self, node):
        assert(isinstance(node, Node))
        rbtreeFixup(self, node)
    
    #------------------------------
    # def Debug
    #------------------------------

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


    #------------------------------
    # def DEPRECATED
    #------------------------------

    def update_arcs(self):
        raise Exception("Deprecated: Use update_values")

    def isEmpty(self):
        raise Exception("Deprecated: use len or bool")

    def insert_many(self, *values):
        raise Exception("Deprecated: use insert")


    def debug_chain(self):
        raise Exception("Deprecated")

    def delete_node(self):
        raise Exception("Deprecated: use delete")

    def balance(self):
        raise Exception("Deprecated: use __balance")
    
