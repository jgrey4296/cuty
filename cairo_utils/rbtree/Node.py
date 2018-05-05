from types import FunctionType
from functools import partial
import logging as root_logger
logging = root_logger.getLogger(__name__)

class Node:
    """ The Container for RBTree Data """
    i = 0
    
    def __init__(self,value,parent=None,data=None):
        self.id = Node.i
        Node.i += 1
        #Children:
        self.left = None
        self.right = None
        #Parent:
        self.parent = parent
        #Node Date:
        self.red = True
        self.value = value
        self.data = {}
        if data is not None:
            assert(isinstance(data,dict))
            self.data.update(data)

        #todo: create templates for data.
        #for arc/voronoi/beachline: left and right circle events


    #------------------------------
    # def Basic Info
    #------------------------------
    
    def __repr__(self):
        if self.red:
            colour = "R"
        else:
            colour = "B"
        return "({}_{} {} {})".format(colour,self.value,self.left,self.right)

    def getBlackHeight(self,parent=None):
        current = self
        height = 0
        while current is not None:
            if not current.red:
                height += 1
            if current == parent:
                current = None
            else:
                current = current.parent
        return height

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

    def getPredecessor(self):
        if self.left is not None:
            return self.left.getMax()
        if self.parent is not None and self.parent.right == self:
            return self.parent
        prev = self
        current = self.parent
        while current is not None and current.right != prev:
            prev = current
            current = current.parent

        if current is not self:
            return current
        else:
            return None

    def getSuccessor(self):
        if self.right is not None:
            return self.right.getMin()
        if self.parent is not None and self.parent.left == self:
            return self.parent
        prev = self
        current = self.parent
        while current is not None and current.left != prev:
            prev = current
            current = current.parent

        if current is not self:
            return current
        else:
            return None

    def getPredecessor_while(self, condition):
        assert(isinstance(condition, (FunctionType, partial)))
        results = []
        current = self.getPredecessor()
        while current is not None and condition(current):
            results.append(current)
            current = current.getPredecessor()
        return results
        

    def getSuccessor_while(self, condition):
        assert(isinstance(condition, (FunctionType, partial)))
        results = []
        current = self.getSuccessor()
        while current is not None and condition(current):
            results.append(current)
            current = current.getSuccessor()
        return results


    def getNeighbours_while(self, condition):
        results = []
        results += self.getPredecessor_while(condition)
        results += self.getSuccessor_while(condition)
        return results

    def isLeaf(self):
        return self.left is None and self.right is None

    #------------------------------
    # def Basic Update
    #------------------------------
    
    def add_left(self,node,force=False):
        logging.debug("{}: Adding {} to Left".format(self,node))
        if self == node:
            node = None
        if self.left == None or force:
            self.link_left(node)
        else:
            self.getPredecessor().add_right(node)
        
    def add_right(self,node,force=False):
        logging.debug("{}: Adding {} to Right".format(self,node))
        if self == node:
            node = None
        if self.right == None or force:
            self.link_right(node)
        else:
            self.getSuccessor().add_left(node)

    def link_left(self,node):
        logging.debug("{} L-> {}".format(self,node))
        if self == node:
            node = None
        self.left = node
        self.left.parent = self

    def link_right(self,node):
        logging.debug("{} R-> {}".format(self,node))
        if self == node:
            node = None
        self.right = node
        if self.right is not None:
            self.right.parent = self
        
    def disconnect_from_parent(self):
        if self.parent != None:
            if self.parent.left == self:
                logging.debug("Disconnecting {} L-> {}".format(self.parent,self))
                self.parent.left = None
            else:
                logging.debug("Disconnecting {} R-> {}".format(self.parent,self))
                self.parent.right = None
            self.parent = None
            
    def disconnect_sequence(self):
        self.disconnect_successor()
        self.disconnect_predecessor()

    def disconnect_hierarchy(self):
        return [self.disconnect_left(),self.disconnect_right()]

    def disconnect_left(self):
        logging.debug("{} disconnectin left: {}".format(self,self.left))
        if self.left != None:
            node = self.left
            self.left = None
            node.parent = None
            return node
        return None

    def disconnect_right(self):
        logging.debug("{} disconnecting right: {}".format(self,self.right))
        if self.right != None:
            node = self.right
            self.right = None
            node.parent = None
            return node
        return None

    #------------------------------
    # def Deprecated
    #------------------------------
    
    def get_predecessor(self):
        raise Exception("Deprecated: use getPredecessor")

    def get_successor(self):
        raise Exception("Deprecated: use getSuccessor")

    def compare_simple(self):
        raise Exception("Deprecated: use appropriate comparison function in rbtree")

    def intersect(self):
        raise Exception("Deprecated: use appropriate method in value")

    def update_arcs(self):
        raise Exception("Deprecated: use rbtree update_values with appropriate lambda")

    def countBlackHeight_null_add(self):
        raise Exception("Deprecated")

    def print_colour(self):
        raise Exception("Deprecated: check node.red ")

    def print_blackheight(self):
        raise Exception("Deprecated")

    def print_tree(self):
        raise Exception("Deprecated")

    def print_tree_plus(self):
        raise Exception("Deprecated")

    def getMinValue(self):
        raise Exception("Deprecated")

    def getMaxValue(self):
        raise Exception("Deprecated")

    
