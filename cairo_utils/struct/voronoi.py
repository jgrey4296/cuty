""" Voronoi.py : Contains the Voronoi Class, which calculates a graphics independent DCEL
    of a Voronoi Diagram.
"""
##-- imports
from __future__ import annotations
import heapq
import logging as root_logger
import pickle
import sys
from dataclasses import InitVar, dataclass, field
from os.path import isfile
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import cairo_utils as utils
import numpy as np
import numpy.random as base_random
from cairo_utils import Parabola, rbtree
from cairo_utils.dcel import DCEL, HalfEdge
from cairo_utils.rbtree.comparison_functions import (Directions,
                                                     arc_comparison,
                                                     arc_equality)
from cairo_utils.umath import (bbox_centre, bound_line_in_bbox,
                               get_distance_raw, isClockwise)

from .events import CIRCLE_EVENTS, CircleEvent, SiteEvent, VEvent
from .voronoi_drawing import Voronoi_Debug

##-- end imports

logging = root_logger.getLogger(__name__)

#Constants and defaults
IMAGE_DIR = "imgs"
SAVENAME = "graph_data.pkl"
BBOX = np.array([0, 0, 1, 1]) #the bbox of the final image
EPSILON = sys.float_info.epsilon
MAX_STEPS = 100000
CARTESIAN = True

BASE_VORONOI_VERT_DATA = {"VORONOI_VERTEX" : True}
BASE_VORONOI_EDGE_DATA = {"VORONOI_EDGE"   : True}
BASE_VORONOI_FACE_DATA = {"VORONOI_FAE"    : True}

@dataclass
class VoronoiData:
    """ A Class to construct a voronoi diagram into a given dcel,
    after initial points are added
    in init_graph
    """

    node_size      : int        = field(default=10)
    max_steps      : int        = field(default=MAX_STEPS)
    bbox           : np.ndarray = field(default=BBOX)
    debug_draw     : bool       = field(default=False)
    #File name to pickle data to
    save_file_name : str        = field(default=SAVENAME)
    dcel           : DCEL       = field(default=None)

    current_step   : int                 = field(init=False, default=0)
    #Min Heap of site/circle events
    events         : list[Event]         = field(init=False, default_factory=list)
    #backup of the original sites
    sites          : list[Vertex]        = field(init=False, default_factory=list)
    #backup of all circle events
    circles        : list[Event]         = field(init=False, default_factory=list)
    #storage of breakpoint tuples -> halfedge
    halfedges      : dict[int, HalfEdge] = field(init=False, default_factory=dict)
    #The Beach Line Data Structure
    beachline      : Beachline           = field(init=False, default=None)
    #The sweep line position
    sweep_position : Vertex              = field(init=False, default=None)
    debug          : Voronoi_Debug       = field(init=False, default=None)

    def __post_init__(self, num_of_nodes=10, bbox=BBOX, save_name=SAVENAME, debug_draw=False, n=10, max_steps=MAX_STEPS, dcel=None):
        assert(isinstance(bbox, np.ndarray))
        assert(bbox.shape == (4, ))
        #The output voronoi diagram as a DCEL
        if dcel is None:
            self.__protect_dcel = False
            self.dcel = DCEL(bbox=bbox)
        else:
            self.__protect_dcel = True
            self.dcel = dcel
            if self.dcel.bbox != bbox:
                #todo: check this
                self.bbox = self.bbox

        VEvent.offset = self.bbox[3] - self.bbox[1]

        if self.debug_draw:
            self.debug = Voronoi_Debug(n, IMAGE_DIR, self)

    def reset(self):
        """ Reset the internal data structures """
        if not self.__protect_dcel:
            self.dcel       = DCEL(bbox=self.bbox)
        self.events         = []
        self.circles        = []
        self.halfedges      = {}
        self.sweep_position = None
        self.beachline      = rbtree.RBTree(cmp_func=arc_comparison,
                                            eq_func=arc_equality)

        self.current_step = 0
