"""
A Basic Node to use in a Red-Black Tree.
"""
from types import FunctionType
from functools import partial
import logging as root_logger
from string import ascii_uppercase

logging = root_logger.getLogger(__name__)

class Node:
    """ The Container for RBTree Data """
    i = 0

    def __init__(self, value, parent=None, data=None, eq_func=None):
        #pylint: disable=invalid-name
        self.id = Node.i
        #pylint: enable=invalid-name
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
        self.eq_func = eq_func
        if data is not None:
            assert(isinstance(data, dict))
            self.data.update(data)

        #todo: create templates for data.
        #for arc/voronoi/beachline: left and right circle events


    #------------------------------
    # def Basic Info
    #------------------------------

    def __hash__(self):
        if self.value is not None and hasattr(self.value, "id"):
            return self.value.id
        return self.id

    def __eq__(self, other):
        assert(other is None or isinstance(other, Node))
        result = False
        if other is None:
            return result
        result = self.id == other.id
        return result

    def __repr__(self):
        #pylint: disable=too-many-format-args
        if self.value is not None and hasattr(self.value, "id"):
            return "({}_{})".format(ascii_uppercase[self.value.id % 26],
                                    int(self.value.id/26), self.id)
        else:
            return "({}:{})".format(self.value, self.id)


    def get_black_height(self):
        """ Get the number of black nodes between self and the root """
        current = self
        height = 0
        while current is not None:
            if not current.red:
                height += 1
            current = current.parent
        return height

    def min(self):
        """ Get the leftmost node of the tree """
        current = self
        while current.left is not None:
            current = current.left
        return current

    def max(self):
        """ Get the rightmost node of the tree """
        current = self
        while current.right is not None:
            current = current.right
        return current

    def get_predecessor(self):
        """ Get the node to the immediate left """
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

    def get_successor(self):
        """ Get the node to the immediate right """
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

    def get_predecessor_while(self, condition):
        """ Collect predecessors while the condition is true """
        assert(isinstance(condition, (FunctionType, partial)))
        results = []
        current = self.get_predecessor()
        while current is not None and condition(current):
            results.append(current)
            current = current.get_predecessor()
        return results


    def get_successor_while(self, condition):
        """ Collect successors while the condition is true """
        assert(isinstance(condition, (FunctionType, partial)))
        results = []
        current = self.get_successor()
        while current is not None and condition(current):
            results.append(current)
            current = current.get_successor()
        return results


    def get_neighbours_while(self, condition):
        """ Collect left and right nodes whlie the condition is true """
        results = []
        results += self.get_predecessor_while(condition)
        results += self.get_successor_while(condition)
        return results

    def is_leaf(self):
        """ Test if this node has no children """
        return self.left is None and self.right is None

    #------------------------------
    # def Basic Update
    #------------------------------

    def add_left(self, node, force=False):
        """ Add a node to the immediate left """
        if self == node:
            node = None
        if self.left is None or force:
            self.link_left(node)
        else:
            self.get_predecessor().add_right(node)
        logging.debug("{}: Adding {} to Left".format(self, node))


    def add_right(self, node, force=False):
        """ Add a node to the immediate right """
        if self == node:
            node = None
        if self.right is None or force:
            self.link_right(node)
        else:
            self.get_successor().add_left(node)
        logging.debug("{}: Adding {} to Right".format(self, node))

    def link_left(self, node):
        """ Connect the passed in node as a predecessor """
        assert(node is not self)
        if node is not None:
            assert(self.right is not node)
            assert(self.parent is not node)
            assert(node.left is not self)
            assert(node.right is not self)
        self.left = node
        if self.left is not None:
            self.left.parent = self
        logging.debug("{} L-> {}".format(self, node))


    def link_right(self, node):
        """ Connect the passed in node as a successor """
        assert(node is not self)
        if node is not None:
            assert(self.parent is not node)
            assert(node.left is not self)
            assert(node.right is not self)
            assert(self.left is not node)
        self.right = node
        if self.right is not None:
            self.right.parent = self
        logging.debug("{} R-> {}".format(self, node))


    def disconnect_from_parent(self):
        """ Symmetrically disconnect this node from its parent """
        parent = self.parent
        if self.parent is not None:
            if self.parent.on_left(self):
                self.parent.left = None
            else:
                self.parent.right = None
            self.parent = None
        logging.debug("Disconnecting {} -> {}".format(parent, self))

    def disconnect_left(self):
        """ Symmetrically disconnect this node from is predecessor """
        if self.left is not None:
            node = self.left
            self.left = None
            node.parent = None
            logging.debug("{} disconnecting left".format(self))
            return node
        return None

    def disconnect_right(self):
        """ Symmetrically disconnect this node from is successor """
        if self.right is not None:
            node = self.right
            self.right = None
            node.parent = None
            logging.debug("{} disconnecting right".format(self))
            return node
        return None

    def on_left(self, node):
        """ Test to see if the passed in node is the predecessor """
        assert(isinstance(node, Node))
        return self.left == node

    def rotate_right(self):
        """ Trigger a rotation to the right centered on this node """
        set_as_root = True
        orig_parent = None
        originally_on_left = False
        new_head = self.left
        new_right = self
        new_left = new_head.right
        if self.parent is not None:
            set_as_root = False
            originally_on_left = self.parent.on_left(self)
            orig_parent = self.parent
            new_right.disconnect_from_parent()
        new_head.disconnect_from_parent()
        if new_left is not None:
            new_left.disconnect_from_parent()

        new_right.link_left(new_left)
        new_head.link_right(new_right)
        if orig_parent is not None:
            if originally_on_left:
                orig_parent.link_left(new_head)
            else:
                orig_parent.link_right(new_head)
        return set_as_root, new_head

    def rotate_left(self):
        """ Trigger a rotation to the left centered on this node """
        set_as_root = True
        orig_parent = None
        originally_on_left = False
        new_head = self.right
        new_left = self
        new_right = new_head.left
        if self.parent is not None:
            set_as_root = False
            originally_on_left = self.parent.on_left(self)
            orig_parent = self.parent
            new_left.disconnect_from_parent()
        new_head.disconnect_from_parent()
        if new_right is not None:
            new_right.disconnect_from_parent()

        new_left.link_right(new_right)
        new_head.link_left(new_left)
        if orig_parent is not None:
            if originally_on_left:
                orig_parent.link_left(new_head)
            else:
                orig_parent.link_right(new_head)
        return set_as_root, new_head
