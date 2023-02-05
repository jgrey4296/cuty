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

class DCELAdder:

    def new_vertex(self, loc, data=None, force=False):
        """ Create a new vertex, or reuse an existing vertex.
        to force a new vertex instead of reusing, set force to True
        """
        assert(isinstance(loc, np.ndarray))
        new_vert = None
        matching_vertices = self.vertex_quad_tree.intersect(Vertex.free_bbox(loc))
        logging.debug("Quad Tree Size: {}".format(self.vertex_quad_tree.countmembers()))
        logging.debug("Query result: {}".format(matching_vertices))
        if bool(matching_vertices) and not force:
            #a good enough vertex exists
            new_vert = matching_vertices.pop()
            if data is not None:
                new_vert.data.update(data)
            logging.debug("Found a matching vertex: {}".format(new_vert))
        else:
            #no matching vertex, add this new one
            new_vert = Vertex(loc, data=data, dcel=self)
            logging.debug("No matching vertex, storing: {}, {}".format(new_vert, new_vert.bbox()))

        assert(new_vert is not None)
        return new_vert

    def new_edge(self, origin_vertex, twin_vertex, face=None, twin_face=None, prev_edge=None, twin_prev=None, next_edge=None, twin_next=None, edata=None, vdata=None):
        """ Create a new half edge pair, after specifying its start and end.
            Can set the faces, and previous edges of the new edge pair.
            Returns the outer edge
        """
        #todo: check for an already existing edge
        assert(origin_vertex is None or isinstance(origin_vertex, Vertex))
        assert(twin_vertex is None or isinstance(twin_vertex, Vertex))
        e1 = HalfEdge(origin_vertex, None, dcel=self)
        e2 = HalfEdge(twin_vertex, e1, dcel=self)
        e1.twin = e2
        #Connect with passed in details
        if face is not None:
            assert(isinstance(face, Face))
            face.add_edge(e1)
        if twin_face is not None:
            assert(isinstance(twin_face, Face))
            twin_face.add_edge(e2)
        if prev_edge is not None:
            assert(isinstance(prev_edge, HalfEdge))
            e1.add_prev(prev_edge)
        if twin_prev is not None:
            assert(isinstance(twin_prev, HalfEdge))
            e2.add_prev(twin_prev)
        if next_edge is not None:
            assert(isinstance(next_edge, HalfEdge))
            e1.add_next(next_edge)
        if twin_next is not None:
            assert(isinstance(twin_next, HalfEdge))
            e2.add_next(twin_next)
        if edata is not None:
            e1.data.update(edata)
            e2.data.update(edata)
        if vdata is not None:
            e1.origin.data.update(vdata)
            e2.origin.data.update(vdata)
        self.half_edges.update([e1, e2])
        logging.debug("Created Edge Pair: {}".format(e1.index))
        logging.debug("Created Edge Pair: {}".format(e2.index))
        return e1

    def new_face(self, site=None, edges=None, verts=None, coords=None, data=None):
        """ Creates a new face to link edges """
        used_list = [edges is not None, verts is not None, coords is not None]
        assert(len([x for x in used_list if x]) < 2)
        if site is None:
            site = np.array([0, 0])
        assert(isinstance(site, np.ndarray))
        new_face = Face(site=site, dcel=self, data=data)
        self.faces.add(new_face)
        #populate the face if applicable:
        coord_hull_gen = False
        if coords is not None:
            assert(isinstance(coords, np.ndarray))
            assert(coords.shape[1] == 2)
            hull_coords = Face.hull_from_coords(coords)
            verts = [self.new_vertex(x) for x in hull_coords]
            coord_hull_gen = True

        if verts is not None:
            if not coord_hull_gen:
                verts, _ = Face.hull_from_vertices(verts)
            edges = []
            for s, e in zip(verts, islice(cycle(verts), 1, None)):
                edges.append(self.new_edge(s, e))

        if edges is not None:
            new_face.add_edges(edges)
            self.link_edges_together(edges, loop=True)

        return new_face


    def create_edge(self, origin, end, edata=None, vdata=None, subdivs=0):
        """ Utility to create two vertices, and put them into a pair of halfedges,
        returning a halfedge
        subdivs specifies number of inner segments to the line"""
        assert(isinstance(origin, np.ndarray))
        assert(isinstance(end, np.ndarray))
        v1 = self.new_vertex(origin)
        v2 = self.new_vertex(end)
        e = self.new_edge(v1, v2)
        if vdata is not None:
            assert(isinstance(vdata, dict))
            v1.data.update(vdata)
            v2.data.update(vdata)
        if edata is not None:
            assert(isinstance(edata, dict))
            e.data.update(edata)
        return e

    def create_path(self, vs, close=False, edata=None, vdata=None):
        """ Create multiple half_edges, that connect to one another.
        With optional path closing """
        assert(isinstance(vs, np.ndarray))
        vertices = vs
        path_verts = zip(vertices, islice(cycle(vertices), 1, None))
        path = []
        for a, b in path_verts:
            if not close and (a == vertices[-1]).all():
                continue
            path.append(self.create_edge(a, b, edata=edata, vdata=vdata))
        return path

    def create_bezier(self, vs, edata=None, vdata=None, single=False):
        """ Takes a list of tuples (len 3 or 4), and creates
        approximation lines, that can be triggered later to
        draw the true bezier shape,
        Bezier Tuple: (Start, cps, End)"""
        assert(isinstance(vs, list))
        assert(all([isinstance(x, tuple) for x in vs]))
        if edata is None:
            edata = {}
        edges = []

        if single:
            #create a single, multi breakpoint line
            first = vs[0]
            last = vs[-1]
            e = self.create_edge(first[0], last[-1], edata=edata, vdata=vdata)
            e.data[EdgeE.BEZIER] = vs
            e.twin.data[EdgeE.NULL] = True
            return [e]


        for v in vs:
            if len(v) == 2 and all([isinstance(x, tuple) for x in v]):
                #is a single edge, with different control points for different
                #directions
                raise Exception("Dual Control Point Edges not yet supported")
            elif len(v) == 3:
                #is a single cp bezier
                a, _, b = v
                edge = self.create_edge(a, b, edata=edata, vdata=vdata)
                edge.data[EdgeE.BEZIER] = [v]
                edge.twin.data[EdgeE.NULL] = True
                edges.append(edge)
            elif len(v) == 4:
                #is a two cp bezier
                a, _, _, b = v
                edge = self.create_edge(a, b, edata=edata, vdata=vdata)
                edge.data[EdgeE.BEZIER] = [v]
                edge.twin.data[EdgeE.NULL] = True
            else:
                raise Exception("Unrecognised bezier type: {}".format(len(v)))

        return edges




    def add_to_current_quad_tree(self, verts):
        """ Add the passed in vertices into the current quad tree """
        assert(all([isinstance(x, Vertex) for x in verts]))
        assert(bool(self.quad_tree_stack))
        for x in verts:
            self.vertex_quad_tree.insert(item=x, bbox=x.bbox())
