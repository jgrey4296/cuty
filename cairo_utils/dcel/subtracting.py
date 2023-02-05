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

class DCELSubtractor:

    def reset_frontier(self):
        """ Clear the general incremental algorithm frontier """
        self.frontier = set([])

    def clear_quad_tree(self):
        """ Clear the internal quadtree of vertices """
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
    def purge_edge(self, target):
        """ Clean up and delete an edge """
        assert(isinstance(target, HalfEdge))
        logging.debug("Purging Edge: {}".format(target.index))
        target_update = set()

        target.connect_next_to_prev()
        vert = target.origin
        target.origin = None
        if vert is not None:
            vert.unregister_half_edge(target)
            if vert.is_edgeless():
                vert.mark_for_cleanup()
                target_update.add(vert)

        if target.face is not None:
            face = target.face
            face.remove_edge(target)
            if not face.has_edges():
                face.mark_for_cleanup()
                target_update.add(face)

        if target.twin is not None:
            target.twin.mark_for_cleanup()
            target_update.add(target.twin)
            target.twin.twin = None
            target.twin = None


        self.half_edges.remove(target)

        return target_update

    def purge_vertex(self, target):
        """ Clean up and delete a vertex """
        assert(isinstance(target, Vertex))
        logging.debug("Purging Vertex: {}".format(target.index))
        target_update = set()

        half_edges = target.half_edges.copy()
        for edge in half_edges:
            assert(edge.origin == target)
            edge.origin = None
            target.unregister_half_edge(edge)
            edge.mark_for_cleanup()
            target_update.add(edge)

        self.vertices.remove(target)

        return target_update

    def purge_face(self, target):
        """ Clean up and delete a face """
        assert(isinstance(target, Face))
        logging.debug("Purging Face: {}".format(target.index))
        target_update = set()
        edges = target.get_edges()
        for edge in edges:
            target.remove_edge(edge)
            edge.mark_for_cleanup()
            target_update.add(edge)
        self.faces.remove(target)
        return target_update

    def purge(self, targets=None):
        """ Run all purge methods in correct order """
        if targets is None:
            #populate the targets:
            targets = set([])
            targets = targets.union([x for x in self.vertices if x.marked_for_cleanup])
            targets = targets.union([x for x in self.half_edges if x.marked_for_cleanup
                                     or x.is_infinite()])
            targets = targets.union([x for x in self.faces if x.marked_for_cleanup
                                     or not x.has_edges()])

        else:
            targets = set(targets)

        purged = set()
        while bool(targets):
            current = targets.pop()
            if current in purged:
                continue
            if isinstance(current, Vertex):
                targets = targets.union(self.purge_vertex(current))
            elif isinstance(current, HalfEdge):
                targets = targets.union(self.purge_edge(current))
            elif isinstance(current, Face):
                targets = targets.union(self.purge_face(current))
            purged.add(current)

        self.calculate_quad_tree()


    #------------------------------
    # def Vertex, Edge, HalfEdge Creation
    #------------------------------
