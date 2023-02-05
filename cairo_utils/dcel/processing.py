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

class DCELProcessor:

    def calculate_quad_tree(self, subverts=None):
        """ Recalculate the quad tree with all vertices, or a subselection of vertices """
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
        verts = self.vertices
        if subverts is not None:
            assert(all([isinstance(x, Vertex) for x in subverts]))
            verts = subverts
        for vertex in verts:
            self.vertex_quad_tree.insert(item=vertex, bbox=vertex.bbox())

    def push_quad_tree(self):
        """ Add another vertex quad tree on top of the stack """
        self.quad_tree_stack.append(self.vertex_quad_tree)
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)

    def pop_quad_tree(self):
        """ Remove the top vertex quad tree from the stack """
        assert(bool(self.quad_tree_stack))
        sub_layer = self.quad_tree_stack.pop()
        if self.should_merge_stacks:
            for x in self.vertex_quad_tree.intersect(self.bbox):
                if x not in sub_layer.intersect(x.bbox()):
                    sub_layer.insert(item=x, bbox=x.bbox())
        self.vertex_quad_tree = sub_layer

    def link_edges_together(self, edges, loop=False):
        """ Given a list of half edges, set their prev and next fields in order """
        #TODO: check this
        assert(all([isinstance(x, HalfEdge) for x in edges]))
        if loop:
            other = islice(cycle(edges), 1, None)
        else:
            other = edges[1:]
        for (a, b) in zip(edges, other):
            a.add_next(b)

    def force_all_edge_lengths(self, l):
        """ Force all edges to be of length <= l. If over, split into multiple lines
        of length l. """
        assert(l > 0)
        processed = set()
        all_edges = list(self.half_edges)
        while bool(all_edges):
            current = all_edges.pop(0)
            assert(current.index not in processed)
            if current.get_length_sq() > l:
                _, new_edge = current.split_by_ratio(r=0.5)
                if new_edge.get_length_sq() > l:
                    all_edges.append(new_edge)
                else:
                    processed.add(new_edge.index)

            if current.get_length_sq() > l:
                all_edges.append(current)
            else:
                processed.add(current.index)

    def constrain_to_circle(self, centre, radius, candidates=None, force=False):
        """ Limit all faces, edges, and vertices to be within a circle,
        adding boundary verts and edges as necessary """
        assert(isinstance(centre, np.ndarray))
        assert(isinstance(radius, float))
        assert(centre.shape == (2, ))

        #constrain faces
        faces = self.faces.copy()
        for f in faces:
            logging.debug("Constraining Face: {}".format(f))
            f.constrain_to_circle(centre, radius, candidates=candidates, force=force)

        #constrain free edges
        hedges = self.half_edges.copy()
        for he in hedges:
            logging.debug("Constraining Hedge: {}".format(he))
            if he.face is not None or he.marked_for_cleanup:
                continue
            he.constrain_to_circle(centre, radius, candidates=candidates, force=force)

        #constrain free vertices
        vertices = self.vertices.copy()
        for v in vertices:
            logging.debug("Constraining Vertex: {}".format(v))
            if not v.is_edgeless():
                continue
            if not v.within_circle(centre, radius):
                v.mark_for_cleanup()

    def constrain_to_bbox(self, bbox, candidates=None, force=False):
        """ Constrain the entire dcel to a bbox """
        assert(isinstance(bbox, np.ndarray))
        assert(bbox.shape == (4, ))

        faces = self.faces.copy()
        for f in faces:
            logging.debug("Constraining Face: {}".format(f))
            f.constrain_to_bbox(bbox, candidates=candidates, force=force)

        #constrain free edges
        hedges = self.half_edges.copy()
        for he in hedges:
            logging.debug("Constraining Hedge: {}".format(he))
            if he.face is not None or he.marked_for_cleanup:
                continue
            he.constrain_to_bbox(bbox, candidates=candidates, force=force)

        #constrain free vertices
        vertices = self.vertices.copy()
        for v in vertices:
            logging.debug("Constraining Vertex: {}".format(v))
            if not v.is_edgeless():
                continue
            if not v.within(bbox):
                v.mark_for_cleanup()

    #------------------------------
    # def Utilities
    #------------------------------

    def intersect_half_edges(self, edge_set=None):
        """ run a sweep line over the dcel,
        getting back halfedge intersections """

        li = LineIntersector(self)
        return li(edge_set=edge_set)

    def order_vertices(self, focus, vertices):
        """ Given a focus point and a list of vertices, sort them
            by the counter-clockwise angle position they take relative """
        assert(all([isinstance(x, Vertex) for x in vertices]))
        assert(isinstance(focus, np.ndarray))
        relative_positions = [v.loc - focus for v in vertices]
        zipped = zip(relative_positions, vertices)
        angled = [((degrees(atan2(loc[1], loc[0])) + 360) % 360, vert) for loc, vert in zipped]
        sorted_angled = sorted(angled)
        # rads = (np.arctan2(verts[:, 1], verts[:, 0]) + TWOPI) % TWOPI
        # ordered = sorted(zip(rads, opp_hedges))
        return [vert for loc, vert in sorted_angled]

    def create_corner_vertex(self, e1, e2, bbox):
        """ Given two intersections (0-3) describing the corner,
        create the vertex at the boundary of the bbox """
        assert(isinstance(e1, int))
        assert(isinstance(e2, int))
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)

        if e1 == e2:
            raise Exception("Corner Edge Creation Error: edges are equal")
        if e1 % 2 == 0: #create the x vector
            v1 = np.array([bbox[e1], 0])
        else:
            v1 = np.array([0, bbox[e1]])
        if e2 % 2 == 0: #create the y vector
            v2 = np.array([bbox[e2], 0])
        else:
            v2 = np.array([0, bbox[e2]])
        #add together to get corner
        v3 = v1 + v2
        return self.new_vertex(*v3)

    def verify_all(self):
        """ Ensure all faces, edges and halfedges are coherent """
        reg_verts = set([x.index for x in self.vertices])
        reg_hedges = set([x.index for x in self.half_edges])
        reg_faces = set([x.index for x in self.faces])

        vert_hedges = set()
        for v in self.vertices:
            vert_hedges.update([x.index for x in v.half_edges])

        hedge_verts = set()
        hedge_nexts = set()
        hedge_prevs = set()
        hedge_faces = set()
        for h in self.half_edges:
            hedge_verts.add(h.origin.index)
            if h.next is not None:
                hedge_nexts.add(h.next.index)
            if h.prev is not None:
                hedge_prevs.add(h.prev.index)
            if h.face is not None:
                hedge_faces.add(h.face.index)

        face_edges = set()
        for f in self.faces:
            face_edges.update([x.index for x in f.edge_list])

        #differences:
        vert_hedge_diff = vert_hedges.difference(reg_hedges)
        hedge_vert_diff = hedge_verts.difference(reg_verts)
        hedge_nexts_diff = hedge_nexts.difference(reg_hedges)
        hedge_prevs_diff = hedge_prevs.difference(reg_hedges)
        hedge_faces_diff = hedge_faces.difference(reg_faces)
        face_edges_diff = face_edges.difference(reg_hedges)

        assert(all([not bool(x) for x in [vert_hedge_diff,
                                          hedge_vert_diff,
                                          hedge_nexts_diff,
                                          hedge_prevs_diff,
                                          hedge_faces_diff,
                                          face_edges_diff]]))
