""" The Top Level DCEL DataStructure. """
##-- imports
from __future__ import annotations
import logging as root_logger
import pickle
from collections import namedtuple
from dataclasses import InitVar, dataclass, field
from itertools import cycle, islice
from math import atan2, degrees
from os.path import isfile
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np
import pyqtree

from .constants import EdgeE, FaceE, VertE
from .face import Face
from .halfedge import HalfEdge
from .line_intersector import LineIntersector
from .vertex import Vertex

##-- end imports

logging = root_logger.getLogger(__name__)

#for importing data into the dcel:
DataPair = namedtuple('DataPair', 'key obj data')

@dataclass
class DCELData:
    """ The total DCEL data structure, stores vertices, edges, and faces,
    Based on de Berg's Computational Geometry
    """
    bbox                : np.ndarray     = field(default=np.array([-200, -200, 200, 200]))
    vertices            : Set[Vertex]    = field(default_factory=set)
    faces               : Set[Face]      = field(default_factory=set)
    half_edges          : Set[HalfEdge]  = field(default_factory=set)

    #todo               : make this a stack of quadtrees

    quad_tree_stack     : List[Any]      = field(default_factory=list)
    frontier            : Set[Any]       = field(default_factory=set)
    should_merge_stacks : bool           = field(default=True)
    data                : Dict[Any, Any] = field(default_factory=dict)

    vertex_quad_tree    : pyqtree        = field(init=False, default=None)
    Vertex                               = Vertex
    HalfEdge                             = HalfEdge
    Face                                 = Face

    def __post_init__(self, bbox=None):
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        #todo: make this a stack of quadtrees
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)


    def __enter__(self):
        """ Makes the Dcel a reusable context manager, that pushes
        and pops vertex quad trees for collision detection """
        self.push_quad_tree()

    def __exit__(self, e_type, value, traceback):
        self.pop_quad_tree()

    def __str__(self):
        """ Create a text description of the DCEL """
        #pylint: disable=too-many-locals
        vertices_description      = "Vertices: num: {}".format(len(self.vertices))
        edges_description         = "HalfEdges: num: {}".format(len(self.half_edges))
        faces_description         = "Faces: num: {}".format(len(self.faces))

        all_vertices              = [x.getVertices() for x in self.half_edges]
        flattened_vertices        = [x for (x, y) in all_vertices for x in (x, y)]
        set_of_vertices           = set(flattened_vertices)
        vertex_set                = "Vertex Set: num: {}/{}/{}".format(len(set_of_vertices),
                                                        len(flattened_vertices),
                                                        len(self.vertices))

        infinite_edges            = [x for x in self.half_edges if x.is_infinite()]
        infinite_edge_description = "Infinite Edges: num: {}".format(len(infinite_edges))

        complete_edges            = set()
        for x in self.half_edges:
            if not x in complete_edges and x.twin not in complete_edges:
                complete_edges.add(x)

        complete_edge_description        = "Complete Edges: num: {}".format(len(complete_edges))

        edgeless_vertices                = [x for x in self.vertices if x.is_edgeless()]
        edgeless_vertices_description    = "Edgeless vertices: num: {}".format(len(edgeless_vertices))

        edge_count_for_faces             = [str(len(f.edge_list)) for f in self.faces]
        edge_count_for_faces_description = \
                "Edge Counts for Faces: {}".format("-".join(edge_count_for_faces))

        purge_verts = "Verts to Purge: {}".format(len([x for x in self.vertices
                                                       if x.marked_for_cleanup]))
        purge_edges = "Hedges to Purge: {}".format(len([x for x in self.half_edges
                                                        if x.marked_for_cleanup]))
        purge_faces = "Faces to Purge: {}".format(len([x for x in self.faces
                                                       if x.marked_for_cleanup]))

        return "\n".join(["---- DCEL Description: ",
                          vertices_description,
                          edges_description,
                          faces_description,
                          vertex_set,
                          infinite_edge_description,
                          complete_edge_description,
                          edgeless_vertices_description,
                          edge_count_for_faces_description,
                          "-- Purging:",
                          purge_verts,
                          purge_edges,
                          purge_faces,
                          "----\n"])


    #------------------------------
    # def IO
    #------------------------------


    def __repr__(self):
        return "<DECL>"


