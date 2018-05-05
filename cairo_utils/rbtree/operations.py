from enum import Enum
import logging as root_logger
from .Node import Node

logging = root_logger.getLogger(__name__)

Directions = Enum('Directions', 'LEFT RIGHT CENTRE')

""" Below are the functional implementations of Red-Black Tree operations """
#todo: integrate these into the beachline class 
def rotateLeft(tree,node):
    """ Rotate the given node left, making the new head be node.right """
    logging.debug("Rotating Left: {}".format(node))
    assert(isinstance(node,Node))
    if node.right is None:
        return #early exit if rotating is unnecessary
    newHead = node.right #Get the right subtree
    originalParent = node.parent
    #left subtree becomes the right subtree:
    node.right = newHead.left
    if node.right is not None:
        node.right.parent = node
    #move the original node to the left
    newHead.left = node
    newHead.left.parent = newHead
    if originalParent is None:            #update the root of the tree
        newHead.parent is None
        if tree:
            tree.root = newHead
    elif node == originalParent.left:  #update the parent's left subtree
        originalParent.left = newHead
        newHead.parent = originalParent
    else:
        originalParent.right = newHead
        newHead.parent = originalParent
    return newHead

def rotateRight(tree,node):
    """ Rotate the given node right, making the new head be node.left """
    logging.debug("Rotating Right: {}".format(node))
    assert(isinstance(node, Node))
    newHead = node.left
    originalParent = node.parent
    node.left = newHead.right
    if node.left != None:
        node.left.parent = node
    newHead.right = node
    newHead.right.parent = newHead    
    if originalParent == None:
        newHead.parent = None
        if tree:
            tree.root = newHead
    elif node == originalParent.left:
        originalParent.left = newHead
        newHead.parent = originalParent
    else:
        originalParent.right = newHead
        newHead.parent = originalParent
    return newHead
        
def rbtreeFixup(tree,node):
    """ Verify and fix the RB properties hold """
    while node.parent != None and node.parent.red:
        parent = node.parent
        grandParent = parent.parent
        if grandParent == None:
            break
        elif parent == grandParent.left:
            y = grandParent.right
            if y != None and y.red:
                parent.red = False
                y.red = False
                grandParent.red = True
                node = grandParent
            else:
                if node == parent.right:
                    node = parent
                    rotateLeft(tree,node)#invalidates parent and grandparent
                node.parent.red = False
                node.parent.parent.red = True
                rotateRight(tree,node.parent.parent)
        else:
            y = grandParent.left
            if y != None and y.red:
                parent.red = False
                y.red = False
                grandParent.red = True
                node = grandParent
            else:
                if node == parent.left:
                    node = parent
                    rotateRight(tree,node)#invalidates parent and grandparent
                node.parent.red = False
                node.parent.parent.red = True
                rotateLeft(tree,node.parent.parent)
    tree.root.red = False

def transplant(tree,u,v):
    """ Transplant the node v, and its subtree, in place of node u """
    logging.debug("Transplanting {} into {}".format(v,u))
    if u.parent == None:
        logging.debug("Setting root to {}".format(v))
        tree.root = v
        v.parent = None
    elif u == u.parent.left:
        logging.debug("Transplant linking left")
        parent = u.parent
        u.parent.link_left(v)
    else:
        logging.debug("Transplant linking right")
        parent = u.parent
        u.parent.link_right(v)


def rbTreeDelete(tree,z):
    y = z
    orig_pred = z.getPredecessor()
    orig_succ = z.getSuccessor()
    orig_parent = z.parent
    y_originally_red = y.red
    x = None
    if z.left == None:
        logging.debug("No left, transplanting right")
        x = z.right
        transplant(tree,z,z.right)
    elif z.right == None:
        logging.debug("No right, transplanting left")
        x = z.left
        transplant(tree,z,z.left)
    else:
        logging.debug("Both Left and right exist")
        y = z.right.getMin()
        y_originally_red = y.red
        x = y.right
        if y.parent == z:
            logging.debug("y.parent == z")
            x.parent = y
        else:
            logging.debug("y.parent != z")
            transplant(tree,y,y.right)
            y.link_right(z.right)
        transplant(tree,z,y)
        y.link_left(z.left)
        y.red = z.red
    if not y_originally_red:
        logging.debug("Fixingup up x: {}".format(x))
        rbDeleteFixup(tree,x)
    #collapse when two nodes are the same
    if orig_pred != None and orig_succ != None and orig_pred.value == orig_succ.value:
        logging.debug("Collapsing with successor {}".format(orig_succ))
        tree.delete_node(orig_succ)
    logging.debug("Finished deletion")
    
def rbDeleteFixup(tree,x):
    while x != tree.root and not x.red:
        if x == x.parent.left:
            w = x.parent.right
            if w.red:
                w.red = False
                x.parent.red = True
                rotateLeft(tree,x.parent)
                w = x.parent.right
            if not w.left.red and not w.right.red:
                w.red = True
                x = x.parent
            else:
                if not w.right.red:
                    w.left.red = False
                    w.red = True
                    rotateRight(tree,w)
                    w = x.parent.right
                w.red = x.parent.red
                x.parent.red = False
                w.right.red = False
                rotateLeft(tree,x.parent)
                x = tree.root
        else: #mirror for right
            w = x.parent.left
            if w.red:
                w.red = False
                x.parent.red = True
                rotateRight(tree,x.parent)
                w = x.parent.left
            if not w.right.red and not w.left.red:
                w.red = True
                x = x.parent
            else:
                if not w.left.red:
                    w.right.red = False
                    w.red = True
                    rotateLeft(tree,w)
                    w = x.parent.left
                w.red = x.parent.red
                x.parent.red = False
                w.left.red = False
                rotateRight(tree,x.parent)
                x = tree.root
    x.red = False
 


