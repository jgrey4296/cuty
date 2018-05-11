from types import FunctionType
from functools import partial
import logging as root_logger
from string import ascii_uppercase
import IPython
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

    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        assert(other is None or isinstance(other, Node))
        if other is None:
            return False
        return self.id == other.id

    def __repr__(self):
        if self.value is not None and hasattr(self.value, "id"):
            return "({}_{})".format(ascii_uppercase[self.value.id % 26], int(self.value.id/26))
        else:
            return "({})".format(self.value)
        

    def getBlackHeight(self,parent=None):
        current = self
        height = 0
        while current is not None:
            if not current.red:
                height += 1
            current = current.parent
        return height

    def min(self):
        current = self
        while current.left is not None:
            current = current.left
        return current

    def max(self):
        current = self
        while current.right is not None:
            current = current.right
        return current

    def getPredecessor(self):
        if self.left is not None:
            return self.left.max()
        if self.parent is not None and not self.parent.on_left(self):
            return self.parent
        prev = self
        current = self.parent
        count = 0
        while current is not None and current.right != prev:
            prev = current
            current = current.parent
            count += 1

        if current is not self:
            return current
        else:
            return None

    def getSuccessor(self):
        if self.right is not None:
            return self.right.min()
        if self.parent is not None and self.parent.on_left(self):
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
        if self == node:
            node = None
        if self.left == None or force:
            self.link_left(node)
        else:
            self.getPredecessor().add_right(node)
        logging.debug("{}: Adding {} to Left".format(self,node))

        
    def add_right(self,node,force=False):
        if self == node:
            node = None
        if self.right == None or force:
            self.link_right(node)
        else:
            self.getSuccessor().add_left(node)
        logging.debug("{}: Adding {} to Right".format(self,node))

    def link_left(self,node):
        assert(node is not self)
        if node is not None:
            assert(self.right is not node)
            assert(self.parent is not node)
            assert(node.left is not self)
            assert(node.right is not self)
        self.left = node
        if self.left is not None:
            self.left.parent = self
        logging.debug("{} L-> {}".format(self,node))


    def link_right(self,node):
        assert(node is not self)
        if node is not None:
            assert(self.parent is not node)
            assert(node.left is not self)
            assert(node.right is not self)
            assert(self.left is not node)
        self.right = node
        if self.right is not None:
            self.right.parent = self
        logging.debug("{} R-> {}".format(self,node))

        
    def disconnect_from_parent(self):
        parent = self.parent
        if self.parent != None:
            if self.parent.on_left(self):
                self.parent.left = None
            else:
                self.parent.right = None
            self.parent = None
        logging.debug("Disconnecting {} -> {}".format(parent,self))

            

    def disconnect_left(self):
        if self.left != None:
            node = self.left
            self.left = None
            node.parent = None
            logging.debug("{} disconnecting left".format(self))
            return node
        return None

    def disconnect_right(self):
        if self.right != None:
            node = self.right
            self.right = None
            node.parent = None
            logging.debug("{} disconnecting right".format(self))
            return node
        return None

    def on_left(self, node):
        assert(isinstance(node, Node))
        return node == self.left
    
    def rotate_right(self):
        setAsRoot = True
        orig_parent = None
        originally_on_left = False
        newHead = self.left
        newRight = self
        newLeft = newHead.right
        if self.parent is not None:
            setAsRoot = False
            originally_on_left = self.parent.on_left(self)
            orig_parent = self.parent
            newRight.disconnect_from_parent()
        newHead.disconnect_from_parent()
        if newLeft is not None:
            newLeft.disconnect_from_parent()

        newRight.link_left(newLeft)
        newHead.link_right(newRight)
        if orig_parent is not None:
            if originally_on_left:
                orig_parent.link_left(newHead)
            else:
                orig_parent.link_right(newHead)
        return setAsRoot, newHead

    def rotate_left(self):
        setAsRoot = True
        orig_parent = None
        originally_on_left = False
        newHead = self.right
        newLeft = self
        newRight = newHead.left
        if self.parent is not None:
            setAsRoot = False
            originally_on_left = self.parent.on_left(self)
            orig_parent = self.parent
            newLeft.disconnect_from_parent()
        newHead.disconnect_from_parent()
        if newRight is not None:
            newRight.disconnect_from_parent()

        newLeft.link_right(newRight)
        newHead.link_left(newLeft)
        if orig_parent is not None:
            if originally_on_left:
                orig_parent.link_left(newHead)
            else:
                orig_parent.link_right(newHead)
        return setAsRoot, newHead
        
        
        
        
    
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

    def getMin(self):
        raise Exception("Deprecated: use .min()")

    def getMax(self):
        raise Exception("Deprecated: use .max()")
    
    def disconnect_hierarchy(self):
        #return [self.disconnect_left(),self.disconnect_right()]
        raise Exception("Deprecated")

    def disconnect_sequence(self):
        # self.disconnect_right()
        # self.disconnect_left()
        raise Exception("Deprecated")
