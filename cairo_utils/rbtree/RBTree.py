"""
Provides the main RBTree class
"""
#pylint: disable=wildcard-import
#pylint: disable=no-self-use
#pylint: disable=too-many-arguments
from functools import partial
from types import FunctionType
import logging as root_logger

from .operations import rb_tree_delete, rb_tree_fixup
from .comparison_functions import Directions, default_comparison, default_equality
from .node import Node


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

    def __init__(self, cmp_func=None, eq_func=None, cleanup_func=None):
        """ Initialise the rb tree container, ie: the node list """
        #Default Comparison and Equality functions with dummy data ignored
        if cmp_func is None:
            cmp_func = default_comparison
        if eq_func is None:
            eq_func = default_equality
        if cleanup_func is None:
            cleanup_func = lambda x: []
        assert(isinstance(cmp_func, (partial, FunctionType)))
        assert(isinstance(eq_func, (partial, FunctionType)))

        self.nodes = []
        self.root = None
        self.cmp_func = cmp_func
        self.eq_func = eq_func
        self.cleanup_func = cleanup_func


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
        found = set()
        chain = []
        current = self.root.min()
        while current is not None:
            if current in found:
                raise Exception("This shouldn't happen")
            found.add(current)
            chain.append(current)
            current = current.get_successor()
        return chain

    def get_successor_triple(self, node):
        """ Get the trio of this node and two successors """
        if node is None:
            return None
        a = node
        b = a.get_successor()
        if b is not None:
            c = b.get_successor()
            if c is not None:
                return (a, b, c)
        return None

    def get_predecessor_triple(self, node):
        """ Get the trio of this node and two predecessors """
        if node is None:
            return None
        a = node
        b = a.getPredecessor()
        if b is not None:
            c = b.getPredecessor()
            if c is not None:
                return (c, b, a)
        return None


    #------------------------------
    # def Query
    #------------------------------
    def search(self, value, cmp_func=None, eq_func=None,
               cmp_data=None, closest=False, start=None):
        """ Search the tree for a value """
        logging.debug("Searching for: {}".format(value))

        if cmp_func is None:
            cmp_func = self.cmp_func
        if eq_func is None:
            eq_func = self.eq_func
        if start is None:
            start = self.root
        parent = start
        current = start
        comp = None
        while current is not None and not eq_func(current, value, cmp_data):
            parent = current
            comp = cmp_func(current, value, cmp_data)
            if comp is Directions.LEFT:
                logging.debug("left")
                current = current.left
            else:
                assert(comp is Directions.RIGHT)
                logging.debug("right")
                current = current.right

        if closest and current is None:
            #closest non-exact match found
            logging.debug("Found Closest: {}, {}".format(parent, comp))
            return (parent, comp)
        elif current is None:
            #nothing found
            logging.debug("Found Nothing")
            return (None, None)
        else:
            #exact match found
            logging.debug("Found Exact: {}, {}".format(current, comp))
            return (current, comp)


    #------------------------------
    # def Public Update
    #------------------------------
    def update_values(self, func, func_data):
        """ Call a function on all stored values,
        function signature:  f(value, func_data)
        """
        assert(isinstance(func, (FunctionType, partial)))
        for node in self.nodes:
            func(node.value, func_data)

    def insert(self, *args, data=None, cmp_data=None):
        """ Insert data into the tree """
        nodes = []
        for x in args:
            new_node = self.__insert(x, data=data, cmp_data=cmp_data)
            nodes.append(new_node)
        return nodes

    def delete(self, *args, cleanup_func=None):
        """ Delete a value from the tree """
        if cleanup_func is None:
            cleanup_func = self.cleanup_func
        to_remove = set(args)
        while bool(to_remove):
            target = to_remove.pop()
            logging.debug("Removing Target: {}".format(target))
            assert(isinstance(target, Node))
            if target not in self.nodes:
                continue
            to_remove.update(cleanup_func(target))
            rb_tree_delete(self, target)
            self.nodes.remove(target)

    def delete_value(self, *args, cmp_func=None, eq_func=None, cleanup_func=None,
                     cmp_data=None):
        """ Delete a value from the tree, and rebalance  """
        for val in args:
            node, _ = self.search(val, cmp_func=cmp_func,
                                  eq_func=eq_func, cmp_data=cmp_data)
            if node is not None:
                self.delete(node, cleanup_func=cleanup_func)
            else:
                logging.debug("Not Found: {}".format(val))


    #------------------------------
    # def Private Update
    #------------------------------
    def __insert(self, value, data=None, cmp_data=None):
        """ Insert a value into the tree """
        parent, direction = self.search(value, closest=True, cmp_data=cmp_data)
        if direction is Directions.LEFT:
            return self.insert_predecessor(parent, value, data=data)
        else:
            return self.insert_successor(parent, value, data=data)

    def insert_successor(self, existing_node, new_value, data=None):
        """ Insert a new value as a successor to a specific node """
        assert(existing_node is None or isinstance(existing_node, Node))
        logging.debug("Inserting Successor")
        new_node = Node(new_value, data=data)
        self.nodes.append(new_node)
        if existing_node is None:
            self.root = new_node
        else:
            existing_node.add_right(new_node)
        self.__balance(new_node)
        return new_node

    def insert_predecessor(self, existing_node, new_value, data=None):
        """ Insert a new value as a predecessor to the specific node """
        assert(existing_node is None or isinstance(existing_node, Node))
        logging.debug("Inserting predecessor")
        new_node = Node(new_value, data=data)
        self.nodes.append(new_node)
        if existing_node is None:
            self.root = new_node
        else:
            existing_node.add_left(new_node)
        self.__balance(new_node)
        return new_node

    def __balance(self, node):
        assert(isinstance(node, Node))
        rb_tree_fixup(self, node)


    #------------------------------
    # def Debug
    #------------------------------
    def count_black_height(self, node=None):
        """ Given a node, count all paths and check they have the same black height """
        if node is None:
            if self.root is None:
                return None
            node = self.root
        stack = [node]
        leaves = []
        while bool(stack):
            current = stack.pop()
            if current.left is None and current.right is None:
                leaves.append(current)
            else:
                if current.left is not None:
                    stack.append(current.left)
                if current.right is not None:
                    stack.append(current.right)

        all_heights = [x.getBlackHeight(node) for x in leaves]
        return all_heights
