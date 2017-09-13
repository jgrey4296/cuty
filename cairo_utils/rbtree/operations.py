# Helper functions

def rotateLeft(tree,node):
    """ Rotate the given node left, making the new head be node.right """
    if node.right is None:
        return
    newHead = node.right #Get the right subtree
    node.right = newHead.left        #left subtree becomes the right subtree 
    if newHead.left is not None:     #update the parent of the left subtree
        newHead.left.parent = node
    newHead.parent = node.parent     #update the parent of the new head
    if node.parent is None:          #update the root of the tree
        tree.root = newHead
    elif node == node.parent.left:  #update the parent's left subtree
        node.parent.left = newHead
    else:
        node.parent.right = newHead #update the parent's right subtree
    newHead.left = node           #move the original node to the left
    node.parent = newHead         #update the parent of the original node


def rotateRight(tree,node):
    """ Rotate the given node right, making the new head be node.left """
    if node.left is None:
        return
    newHead = node.left
    node.left = newHead.right
    if newHead.right is not None:
        newHead.right.parent = node
    newHead.parent = node.parent
    if node.parent is None:
        tree.root = newHead
    elif node == node.parent.right:
        node.parent.right = newHead
    else:
        node.parent.left = newHead
    newHead.right = node
    node.parent = newHead

def rbtreeFixup(tree,node):
    while node.parent is not None and node.parent.colour == RED:
        parent = node.parent
        parentParent = parent.parent
        if parentParent is None:
            break
        elif parent == parentParent.left:
            y = parentParent.right
            if y is not None and y.colour == RED:
                parent.colour = BLACK
                y.colour = BLACK
                parentParent.colour = RED
                node = parentParent
            else:
                if node == parent.right:
                    node = parent
                    rotateLeft(tree,node)
                parent.colour = BLACK
                parentParent.colour = RED
                rotateRight(tree,parentParent)
        else:
            y = parentParent.left
            if y is not None and y.colour == RED:
                parent.colour = BLACK
                y.colour = BLACK
                parentParent.colour = RED
                node = parentParent
            else:
                if node == parent.left:
                    node = parent
                    rotateRight(tree,node)
                parent.colour = BLACK
                parentParent.colour = RED
                rotateLeft(tree,parentParent)
    tree.root.colour = BLACK


def transplant(tree,u,v):
    """ Transplant the node v, and its subtree, in place of node u """
    if u.parent is None:
        tree.root = v
    elif u == u.parent.left:
        u.parent.left = v
    else:
        u.parent.right = v
    v.parent = u.parent
    

def rbTreeDelete(tree,z):
    """ Delete the node z from the tree """
    y = z
    origColour = y.colour
    x = None
    if z.left is None: #no left subtree, just move the right up
        x = z.right
        transplant(tree,z,z.right) 
    elif z.right is None: #no right subtree, move the left up
        x = z.left
        transplant(tree,z,z.left)
    else: #both subtrees exist
        y = z.right.getMin() #get the min of the right, and use that in place of the old head
        #could use the max of the left? might conflict with colours
        origColour = y.colour 
        x = y.right 
        if y.parent == z: #degenerate case: min of tree is z.right
            if x is not None:
                x.parent = y # surely this is redundant? x is already a child of y?
        else:
            transplant(tree,y,y.right) #move y'right subtree to where y is
            y.right = z.right #move the right subtree of the node to delete to the min of that subtree
            y.right.parent = y #update the parent
        transplant(tree,z,y) #move the new minimum to where z was
        y.left = z.left #take z's left subtree, move it to y
        y.left.parent = y #update the parent of the left subtree
        y.colour = z.colour #copy the colour over
    if origColour == BLACK:
        rbDeleteFixup(tree,x)


def rbDeleteFixup(tree,x):
    while x != tree.root and x.colour == BLACK: #keep going till you hit the root
        if x == x.parent.left: #Operate on the left subtree
            w = x.parent.right 
            if w.colour == RED: # opposite subtree is red
                w.colour = BLACK #switch colour of that tree and parent
                x.parent.colour = RED 
                rotateLeft(tree,x.parent) #then rotate
                w = x.parent.right #update the the opposite subtree to the new subtree
            if w.left.colour == BLACK and w.right.colour == BLACK: #if both subtrees are black
                w.colour = RED #recolour the subtree head
                x = x.parent #and move up
            else: #different colours on the subtrees
                if w.right.colour == BLACK: 
                    w.left.colour = BLACK #normalise the colours of the left and right
                    w.colour = RED #flip the parent colour
                    rotateRight(tree,w) #rotate
                    w = x.parent.right #update the subtree focus 
                w.colour = x.parent.colour 
                x.parent.colour = BLACK
                w.right.colour = BLACK
                rotateLeft(tree,x.parent) #rotate back if necessary
                x = tree.root 
        else: #mirror image for right subtree
            w = x.parent.left
            if w.colour == RED:
                w.colour = BLACK
                x.parent.colour = RED
                rotateRight(tree,x.parent)
                w = x.parent.left
            if w.right.colour == BLACK and w.left.colour == BLACK:
                w.colour = RED
                x = x.parent
            elif w.left.colour == BLACK:
                w.right.colour = BLACK
                w.colour = RED
                rotateLeft(tree,w)
                w = x.parent.left
            w.colour = x.parent.colour
            x.parent.colour = BLACK
            w.left.colour = BLACK
            rotateRight(tree,x.parent)
            x = Tree.root
    x.colour = BLACK
            
