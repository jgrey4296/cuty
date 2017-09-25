import logging as root_logger
from .operations import Directions, rbTreeDelete_textbook, rbtreeFixup
from .NilNode import NilNode
from .Node import Node
logging = root_logger.getLogger(__name__)


class BeachLine:
    """ A Red-Black Tree bastardisation into a Beach Line for fortune's algorithm
    Properties of RBTrees:
    1) Every node is Red Or Black
    2) The root is black
    3) Every leaf is Black, leaves are null nodes
    4) If a node is red, it's children are black
    5) All paths from a node to its leaves contain the same number of black nodes

    The beachline is intended to hold parabola arcs, and each node contains successor and
    predecessor links, to traverse the tree leaves laterally.
    The red-black properties are used to ensure arcs are ordered from left to right appropriately
    for their intersections
    """

    def __init__(self,arc=True):
        """ Initialise the rb tree container, ie: the node list """
        self.arc = arc
        self.nodes = []      #list of all nodes created 
        self.arcs_added = [] #list of all values added
        self.root = NilNode
        #init nilnode
        self.root.parent = NilNode
        self.root.left = NilNode
        self.root.right = NilNode
        self.root.canonical = NilNode
        
        
    def __str__(self):
        if self.root is NilNode:
            return "_"
        else:
            str = []
            str.append("RB Tree:\n")
            str.append("Colours: " + self.root.print_colour() + "\n")
            str.append("Heights: " + self.root.print_blackheight() + "\n")
            str.append("P_ids: " + self.root.print_tree() + "\n")
            str.append("Chain ids: " + self.print_chain())
            return "".join(str)
                
    def print_chain(self):
        if not self.arc:
            lst = [str(x) for x in self.get_chain()]
        else:
            lst = [ascii_uppercase[x.value.id] for x in self.get_chain()]
        return "-".join(lst)
        
    def update_arcs(self,d):
        for arc in self.arcs_added:
            arc.update_d(d)
        
    def isEmpty(self):
        return self.root is NilNode

    def insert_many(self,*values):
        for x in values:
            self.insert(x)
    
    def insert(self,value):
        if self.root is NilNode:
            self.arcs_added.append(value)
            self.root = Node(value,arc=self.arc)
            self.nodes.append(self.root)
            self.balance(self.root)
            return self.root
        else:
            node,direction = self.search(value)
            if direction is Directions.RIGHT or direction is directions.CENTRE:
                return self.insert_successor(node,value)
            else: #isinstance(direction,Left):
                return self.insert_predecessor(node,value)
        
    def insert_successor(self,existing_node,newValue):
        assert(existing_node is NilNode or isinstance(existing_node, Node))
        self.arcs_added.append(newValue)
        new_node = Node(newValue,arc=self.arc)
        self.nodes.append(new_node)
        if existing_node is NilNode:
            existing_node = self.root
        if existing_node is NilNode:
            self.root = new_node
        else:
            existing_node.add_right(new_node)
        self.balance(new_node)
        return new_node

    def insert_predecessor(self,existing_node,newValue):
        assert(existing_node is NilNode or isinstance(existing_node, Node))
        self.arcs_added.append(newValue)
        new_node = Node(newValue,arc=self.arc)
        self.nodes.append(new_node)
        if existing_node == NilNode :
            existing_node = self.root
        if existing_node == NilNode:
            self.root = new_node
        else:
            existing_node.add_left(new_node)
        self.balance(new_node)
        return new_node

    def delete_value(self,value):
        node,direction = self.search(value)
        self.delete_node(node)
    
    def delete_node(self,node):
        """ Delete a value from the tree """
        assert(node is NilNode or isinstance(node, Node))
        if node == NilNode:
            return        
        if self.arc:
            triple = [node.get_predecessor(),node,node.get_successor()]
            tripleString = "-".join([str(x) for x in triple if x])
            logging.debug("Deleting Arc: {}".format(tripleString))
        else:
            logging.debug("Deleting Value: {}".format(node.value))
        rbTreeDelete_textbook(self,node)
        if node in self.nodes:
            self.nodes.remove(node)
        #del node
        
    def search(self,x,d=None,verbose=False):
        """ Search the tree for a value, getting closest node to it, 
            returns (node,insertion_function)
        """
        current = self.root
        if current == NilNode:
            return None #not found
        parent = NilNode
        found = False
        while not found:
            comp = current.compare(x,d=d)
            logging.debug("Moving: {}".format(comp))
            if comp is Directions.LEFT:
                parent = current
                current = current.left
            elif comp is Directions.RIGHT:
                parent = current
                current = current.right
            elif comp is Directions.CENTRE:
                #ie: spot on
                parent = current
                found = True
            else: #type is none
                raise Exception("Comparison returned None")
            if current == NilNode:
                found = True
                
        return (parent, comp) #the existing parent and the side to add it
        
    def min(self):
        """ Get the min value of the tree """
        if self.root == NilNode:
            return None
        return self.root.getMin()

    def max(self):
        """ Get the max value of the tree """
        if self.root == NilNode:
            return None
        return self.root.getMax()
        
    def balance(self,node):
        assert(node is NilNode or isinstance(node, Node))
        rbtreeFixup(self,node)

    def get_chain(self):
        """ Get the sequence of leaf values, from left to right """
        #logging.debug("getting chain of nodes")
        if self.root == NilNode:
            return []
        chain = []
        current = self.root.getMin()
        while current != NilNode:
            #logging.debug("Get_chain: appending {} Pre: {}, Succ:{}".format(current,current.predecessor,current.get_successor()))
            chain.append(current)
            current = current.get_successor()
        return [x for x in chain]

    def get_successor_triple(self,node):
        if node is NilNode:
            return None
        a = node
        b = a.get_successor()
        if b != NilNode:
            c = b.get_successor()
            if c != NilNode:
                return (a,b,c)
        return None

    def get_predecessor_triple(self,node):
        if node == NilNode:
            return None
        a = node
        b = a.get_predecessor()
        if b != NilNode:
            c = b.get_predecessor()
            if c != NilNode:
                return (c,b,a)
        return None
        
    def countBlackHeight(self,node=None):
        """ Given a node, count all paths and check they have the same black height """
        if node is NilNode:
            if self.root is NilNode:
                return None
        node = self.root
        stack = [node]
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
                        
        allHeights = [x.getBlackHeight(node) for x in leaves]
        return allHeights
