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

@dataclass
class LineSweepResult:
    """ Utility class to collect the start and end sets of halfedges,
    along with the set of intersections contained within halfedges,
    and the vertex describing that point.

    Starting and Ending are defined from Top -> Bottom, Left -> Right
    """

    vertex  : Vertex   = field()
    start   : HalfEdge = field()
    contain : Any      = field()
    end     : HalfEdge = field()

    def __repr__(self):
        return f"<Intersection: {self.vertex}\nStart: {self.start}\nContain: {self.contain}\nEnd: {self.end}>"

@dataclass
class IntersectorData:
    """ Processes a DCEL to intersect half_edges,
    in a self contained class
    """

    dcel            : 'DCEL'         = field()
    edge_set        : Set[HalfEdge]  = field(init=False, default_factory=set)
    lower_edges     : List[HalfEdge] = field(init=False, default_factory=list)
    results         : List[Any]      = field(init=False, default_factory=list)
    discovered      : Set[Any]       = field(init=False, default_factory=set)
    sweep_y         : float          = field(init=False, default=inf)
    sweep_y_prev    : float          = field(init=False, default=inf)
    #Tree to keep active edges in,
    #Sorted by the x's for the current y of the sweep line
    status_tree     : RBTree         = field(init=False, default=None)
    #Heap of (vert, edge) pairs,
    #with invariant : all([e.is_upper() for v, e in event_list])
    event_list      : List[Event]    = field(init=False, default_factory=list)

    def __init__(self, dcel):
        self.status_tree = RBTree(cmp_func=line_cmp, eq_func=line_eq)
