"""
A Basic Tree Module
"""

class Tree:
    """ A Simple Binary Tree Class. Trees are nodes themselves.
    Each node can hold a *value* (used for searching, ordering etc),
    and *data*, that can be anything, especially a dictionary.
    """

    def __init__(self, value, root=False, data=None):
        self.value = value
        self.data = data
        self.left = None
        self.right = None
        self.root = root

    def is_leaf(self):
        """ Test if this object is a leaf node """
        return self.left is None and self.right is None

    def is_root(self):
        """ Test if this object is the root of a tree """
        return self.root

    def insert(self, value, data=None):
        """ Insert a node into the tree """
        insert_on_left = value < self.value
        if insert_on_left:
            if self.left is None:
                self.left = Tree(value, data=data)
            else:
                self.left.insert(value, data=data)
        else:
            if self.right is None:
                self.right = Tree(value, data=data)
            else:
                self.right.insert(value, data=data)

    def __str__(self):
        if self.left is not None:
            left_string = self.left.__str__()
        else:
            left_string = "()"
        if self.right is not None:
            right_string = self.right.__str__()
        else:
            right_string = "()"
        return "( V: {} Left: {}, Right: {} )".format(self.value, left_string, right_string)

    def search(self, value):
        """ Search the tree for a value """
        if value == self.value:
            return self
        elif self.is_leaf():
            return None
        elif value < self.value:
            if self.left is None:
                return None
            else:
                return self.left.search(value)
        else:
            if self.right is None:
                return None
            else:
                return self.right.search(value)

    def get_range(self, l, r):
        """ Get the leftmost and rightmost values of the tree """
        values = []
        if l < self.value and self.left is not None:
            values.extend(self.left.get_range(l, r))

        if l < self.value and self.value <= r:
            values.append(self)

        if self.value <= r and self.right is not None:
            values.extend(self.right.get_range(l, r))

        return values
