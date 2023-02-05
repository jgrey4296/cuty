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

class BeachLine:

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
