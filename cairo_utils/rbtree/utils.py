#!/usr/bin/env python3
"""

"""
##-- imports
from __future__ import annotations

import types
import abc
import datetime
import enum
import functools as ftz
import itertools as itz
import logging as logmod
import pathlib as pl
import re
import time
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Final, Generic,
                    Iterable, Iterator, Mapping, Match, MutableMapping,
                    Protocol, Sequence, Tuple, TypeAlias, TypeGuard, TypeVar,
                    cast, final, overload, runtime_checkable)
from uuid import UUID, uuid1
from weakref import ref

##-- end imports

##-- logging
logging = logmod.getLogger(__name__)
##-- end logging

class RBTreeUtils:

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

    def search(self, value, cmp_func=None, eq_func=None, cmp_data=None, closest=False, start=None):
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

    def delete_value(self, *args, cmp_func=None, eq_func=None, cleanup_func=None, cmp_data=None):
        """ Delete a value from the tree, and rebalance  """
        for val in args:
            node, _ = self.search(val, cmp_func=cmp_func,
                                  eq_func=eq_func, cmp_data=cmp_data)
            if node is not None:
                self.delete(node, cleanup_func=cleanup_func)
            else:
                logging.debug("Not Found: {}".format(val))

    def __insert(self, value, data=None, cmp_data=None):
        """ Insert a value into the tree """
        parent, direction = self.search(value, closest=True, cmp_data=cmp_data)
        if direction is Directions.LEFT:
            return self.insert_predecessor(parent, value, data=data)
        else:
            return self.insert_successor(parent, value, data=data)

    def __balance(self, node):
        assert(isinstance(node, Node))
        rb_tree_fixup(self, node)

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
