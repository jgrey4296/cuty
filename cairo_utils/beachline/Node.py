import logging as root_logger
import math
from string import ascii_uppercase
from cairo_utils import Parabola
from .NilNode import NilNode
from .operations import Directions

logging = root_logger.getLogger(__name__)


class Node(object):
    """ The internal node class for the rbtree.  """
    i = 0
    
    def __init__(self,value,parent=NilNode,red=True,arc=True):
        assert(parent is NilNode or isinstance(parent, Node))
        if arc is True:
            assert(isinstance(value, Parabola))
        self.id = Node.i
        Node.i += 1
        #Node Data:
        self.red = red
        #arc: is the value an arc or a normal number?
        self.arc = arc
        self.value = value
        self.left_circle_event = None
        self.right_circle_event = None
        #Additional Data:
        self.data = {}
        #Children:
        self.left = NilNode
        self.right = NilNode
        #Parent:
        self.parent = parent

    def __str__(self):
        if self.arc:
            try:
                value = ascii_uppercase[self.value.id]
            except IndexError as e:
                value = "IE:{}".format(str(self.value))
            return value
        else:
            return str(self.value)

    def compare_simple(self,x):
        """ Compare two values """
        if x < self.value:
            return Directions.LEFT
        if x > self.value:
            return Directions.RIGHT
        return Directions.CENTRE
        
    def compare(self,x,d=None):
        """ A More complicated comparison, which looks at neighbours of the node,
        used to check this node's arc against the neighbour arcs 
        """
        logging.debug("Comparing {} to {}".format(x,self))
        if not self.arc:
            return self.compare_simple(x)
        
        pred = self.get_predecessor()
        succ = self.get_successor()
        logging.debug("Pred: {}, Succ: {}".format(pred,succ))
        pred_intersect = None
        succ_intersect = None
        the_range = [-math.inf,math.inf]
        if pred == NilNode and succ == NilNode: #Base case: single arc
            logging.debug("Single Arc: {}".format(self))
            return Directions.CENTRE

        if pred != NilNode and succ != NilNode:
            logging.debug("Trio of arcs is clockwise: {}".format(isClockwise(pred.value.get_focus(),\
                                                                           self.value.get_focus(),\
                                                                           succ.value.get_focus(),\
                                                                           cartesian=True)))
        #pred and successor are the same arc
        if pred != NilNode and succ != NilNode and pred.value == succ.value:
            intersect = pred.value.intersect(self.value)
            logging.debug("Predecessor and Successor are the same: {}".format(pred))
            logging.debug("Intersection result: {}".format(intersect))
            if len(intersect) != 2:
                raise Exception("Two parabolas arent intersecting correctly")
            if the_range[0] < intersect[:,0].min():
                the_range[0] = intersect[:,0].min()
            if the_range[1] > intersect[:,0].max():
                the_range[1] = intersect[:,0].max()
            

        else: #different arcs bookend
            if pred != NilNode:
                pred_intersect = self.value.intersect(pred.value)
                logging.debug("Pred intersect result: {}".format(pred_intersect))
                if len(pred_intersect.shape) == 2:
                    if pred_intersect.shape[0] == 1:
                        the_range[0] = pred_intersect[0,0]
                    else:
                        the_range[0] = pred_intersect[1,0]
                else:
                    the_range[0] = pred_intersect[0]
                    
            
            if succ != NilNode:
                succ_intersect = succ.value.intersect(self.value)
                logging.debug("Succ intersect result: {}".format(succ_intersect))
                if len(succ_intersect.shape) == 2:
                    if succ_intersect.shape[0] == 1:
                        the_range[1] = succ_intersect[0,0]
                    else:
                        the_range[1] = succ_intersect[1,0]
                else:
                    the_range[0] = succ_intersect[0]

        logging.debug("Testing: {} < {} < {}".format(the_range[0],x,the_range[1]))
        if the_range[0] < x and x <= the_range[1]:
            return Directions.CENTRE
        elif x < the_range[0]:
            return Directions.LEFT
        elif the_range[1] < x:
            return Directions.RIGHT
        else:
            logging.info("Comparison failure: {} < {} < {}".format(the_range[0],x,the_range[1]))
            IPython.embed()
            raise Exception("Comparison failure")

        
    def isLeaf(self):
        return self.left == NilNode and self.right == NilNode
    
    def intersect(self,node):
        assert(self.arc is True)
        assert(isinstance(node, Node))
        assert(node.arc is True)
        raise Exception("Not implemented yet: intersect")

    def update_arcs(self,d):
        assert(self.arc is True)
        self.value.update_d(d)
        
    def getBlackHeight(self,root=NilNode):
        """ Get the number of black nodes from self to the root  """
        #logging.debug("Getting black height for {} to {}".format(self,root))
        current = self
        height = 0
        while current != root and current != NilNode:
            #logging.debug("looping: {} -> {}".format(current,current.parent))
            if not current.red:
                height += 1
            current = current.parent
        return height

    def countBlackHeight_null_add(self):
        """ Given a node, count all paths and check they have the same black height """
        stack = [self]
        leaves = []
        while len(stack) > 0:
            current = stack.pop()
            if current.isLeaf():
                leaves.append(current)
            else:
                if current.left != NilNode:
                    stack.append(current.left)
                if current.right != NilNode:
                    stack.append(current.right)

        #plus one for the true 'leaf' nodes, the nil ones
        allHeights = [x.getBlackHeight(self)+1 for x in leaves]
        return allHeights
    
    def print_colour(self):
        """ String representation of the node """
        #logging.debug("printing colours")
        if self.red:
            colour = "R"
        else:
            colour = "B"
        if self.isLeaf():
            return "{}".format(colour)
        else:
            a = None
            b = None
            if self.left != NilNode:
                a = self.left.print_colour()
            if self.right != NilNode:
                b = self.right.print_colour()
            return "{}( {} {} )".format(colour,a,b)

    def print_blackheight(self):
        #logging.debug("Printing heights")
        if self.isLeaf():
            return "{}".format(self.getBlackHeight())
        else:
            a = None
            b = None
            if self.left != NilNode:
                a = self.left.print_blackheight()
            if self.right != NilNode:
                b = self.right.print_blackheight()
            return "{}( {} {})".format(self.getBlackHeight(), a,b)
        
    def print_tree(self):
        #logging.debug("Printing tree")
        if not self.arc:
            return ""
        elif self.isLeaf():
            return ascii_uppercase[self.value.id]
        else:
            i = ascii_uppercase[self.value.id]
            a = "Nil"
            b = "Nil"
            if self.left != NilNode:
                a = self.left.print_tree()
            if self.right != NilNode:
                b = self.right.print_tree()
            return "{}( {} {} )".format(i,a,b)

    def print_tree_plus(self):
        #logging.debug("printing tree plus")
        if not self.arc:
            return ""
        elif self.isLeaf():
            p = str(self.get_predecessor())
            s = str(self.get_successor())
            return "{}<-{}->{}".format(p,ascii_uppercase[self.value.id],s)
        else:
            i = ascii_uppercase[self.value.id]
            p = str(self.get_predecessor())
            s = str(self.get_successor())
            a = "Nil"
            b = "Nil"
            if self.left != NilNode:
                a = self.left.print_tree_plus()
            if self.right != NilNode:
                b = self.right.print_tree_plus()
            return "{}<-{}->{}( {} {})".format(p,i,s,a,b)

        
    def get_predecessor(self):
        if self.left != NilNode:
            return self.left.getMax()
        current = self
        found = False
        while not found:
            #logging.debug("predecessor loop: {}".format(current))
            if current.parent.right == current or current.parent == NilNode:
                found = True
            current = current.parent
        #logging.debug("pred loop fin: {}".format(current))
        return current

    def get_successor(self):
        if self.right != NilNode:
            return self.right.getMin()
        current = self
        found = False
        while not found:
            #logging.debug("successor loop: {}".format(current))
            if current.parent.left == current or current.parent == NilNode:
                found = True
            current = current.parent
        #logging.debug("succ loop fin: {}".format(current))
        return current
        
    def getMin(self):
        """ Get the smallest leaf from the subtree this node is root of """
        current = self
        while not current.isLeaf() and current.left != NilNode:
            current = current.left
        return current
    
    def getMax(self):
        """ Get the largest leaf from the subtree this node is root of """
        current = self
        while not current.isLeaf() and current.right != NilNode:
            current = current.right
        return current
    
    def getMinValue(self):
        return self.getMin().value

    def getMaxValue(self):
        return self.getMax().value
    
    def add_left(self,node,force=False):
        logging.debug("{}: Adding {} to Left".format(self,node))
        if self == node:
            node = NilNode
        if self.left == NilNode or force:
            self.link_left(node)
        else:
            self.get_predecessor().add_right(node)
        
    def add_right(self,node,force=False):
        logging.debug("{}: Adding {} to Right".format(self,node))
        if self == node:
            node = NilNode
        if self.right == NilNode or force:
            self.link_right(node)
        else:
            self.get_successor().add_left(node)
        
    def disconnect_from_parent(self):
        if self.parent != NilNode:
            if self.parent.left == self:
                logging.debug("Disconnecting {} L-> {}".format(self.parent,self))
                self.parent.left = NilNode
            else:
                logging.debug("Disconnecting {} R-> {}".format(self.parent,self))
                self.parent.right = NilNode
            self.parent = None

    def link_left(self,node):
        logging.debug("{} L-> {}".format(self,node))
        if self == node:
            node = NilNode
        self.left = node
        self.left.parent = self

    def link_right(self,node):
        logging.debug("{} R-> {}".format(self,node))
        if self == node:
            node = NilNode
        self.right = node
        self.right.parent = self
            
    def disconnect_sequence(self):
        self.disconnect_successor()
        self.disconnect_predecessor()

    def disconnect_hierarchy(self):
        return [self.disconnect_left(),self.disconnect_right()]

    def disconnect_left(self):
        logging.debug("{} disconnectin left: {}".format(self,self.left))
        if self.left != NilNode:
            node = self.left
            self.left = NilNode
            node.parent = NilNode
            return node
        return None

    def disconnect_right(self):
        logging.debug("{} disconnecting right: {}".format(self,self.right))
        if self.right != NilNode:
            node = self.right
            self.right = NilNode
            node.parent = NilNode
            return node
        return None


#utility function

def isClockwise(*args,cartesian=True):
    #based on stackoverflow.
    #sum over edges, if positive: CW. negative: CCW
    #assumes normal cartesian of y bottom = 0
    sum = 0
    p1s = args
    p2s = list(args[1:])
    p2s.append(args[0])
    pairs = zip(p1s,p2s)
    for p1,p2 in pairs:
        a = (p2[0,0]-p1[0,0]) * (p2[0,1]+p1[0,1])
        sum += a
    if cartesian:
        return sum >= 0
    else:
        return sum < 0
