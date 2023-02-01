"""
A Basic Node to use in a Red-Black Tree.
"""
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from functools import partial
from string import ascii_uppercase
from types import FunctionType
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

from cairo_utils.tree import Tree

logging = root_logger.getLogger(__name__)

@dataclass
class Node(Tree):
    """ The Container for RBTree Data """

    red     : bool     = field(default=True)
    eq_func : Callable = field(default=None)

    #------------------------------
    # def Basic Info
    #------------------------------
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
