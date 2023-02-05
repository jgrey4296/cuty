""" Vertex: The lowest level data structure in a dcel """
##-- imports
from __future__ import annotations

import logging as root_logger
from dataclasses import InitVar, dataclass, field
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import cairo_utils as cu
import numpy as np
from cairo_utils.constants import (ALLCLOSE_TOLERANCE, D_EPSILON, TOLERANCE,
                                   TWOPI, VERTEX, VERTRAD, EditE, VertE)
from cairo_utils.drawable import Drawable
from cairo_utils.drawing import draw_circle
from cairo_utils.umath import in_circle, rotate_point

##-- end imports

logging = root_logger.getLogger(__name__)

@dataclass
class Vertex(Drawable):
    """ A Simple vertex for two dimensions.
    Has a pair of coordinates, and stores the edges associated with it.
    """

    loc                               : np.ndarray     = field()
    dcel                              : DCEL           = field()
    #The edges this vertex is part of :
    half_edges                        : List[HalfEdge] = field(default_factory=list)
    #Custom data of the vertex        :
    data                              : Dict[Any, Any] = field(default_factory=dict)
    index                             : int            = field(default=None)

    marked_for_cleanup                : bool           = field(init=False, default=False)
    active                            : bool           = field(init=False, default=True)
    nextIndex = 0

    def __post_init__(self):
        assert(isinstance(loc, np.ndarray))
        assert(edges is None or isinstance(edges, list))

        if index is None:
            logging.debug("Creating vertex {} at: {:.3f} {:.3f}".format(Vertex.nextIndex,
                                                                        loc[0], loc[1]))
            self.index = Vertex.nextIndex
            Vertex.nextIndex += 1
        else:
            assert(isinstance(index, int))
            logging.debug("Re-Creating Vertex: {}".format(index))
            self.index = index
            if self.index >= Vertex.nextIndex:
                Vertex.nextIndex = self.index + 1

        if self.dcel is not None and self not in self.dcel.vertices:
            self.dcel.vertices.add(self)
            self.dcel.vertex_quad_tree.insert(item=self, bbox=self.bbox())



    def __repr__(self):
        edges = [x.index for x in self.half_edges]
        edges += [x.twin.index for x in self.half_edges]
        return "<Vertex V: {}, edges: {}, ({:.3f}, {:.3f})>".format(self.index, edges,
                                                                    self.loc[0], self.loc[1])

    def __lt__(self, other):
        """ Sorting top to bottom, left to right """
        assert(isinstance(other, Vertex))
        if np.allclose(self.loc[1], other.loc[1],
                       atol=ALLCLOSE_TOLERANCE[0],
                       rtol=ALLCLOSE_TOLERANCE[1]):
            return self.loc[0] < other.loc[0]
        else:
            return self.loc[1] > other.loc[1]

    def copy(self):
        """ Create an isolated copy of this vertex. Doesn't copy halfedge connections,
        but does copy data """
        new_vert = self.dcel.new_vertex(self.loc, data=self.data.copy())
        return new_vert

    def mark_for_cleanup(self):
        """ Schedule this object to be deleted on the next pass """
        self.marked_for_cleanup = True

    #------------------------------
    # def exporting
    #------------------------------
    def _export(self):
        """ Export identifiers instead of objects to allow reconstruction """
        logging.debug("Exporting Vertex: {}".format(self.index))
        enum_data     = {a.name:b for a, b in self.data.items() if a in VertE}
        non_enum_data = {a:b for a, b in self.data.items() if a not in VertE}

        return {
            'i'             : self.index,
            'x'             : self.loc[0],
            'y'             : self.loc[1],
            'half_edges'    : [x.index for x in self.half_edges],
            "enum_data"     : enum_data,
            "non_enum_data" : non_enum_data,
            "active"        : self.active
        }


    #------------------------------
    # def Human Readable Representations
    #------------------------------

    # def __str__(self):
    #     return "({:.3f}, {:.3f})".format(self.loc[0], self.loc[1])


    #------------------------------
    # def activation
    #------------------------------
    def activate(self):
        """ Activates the object, so it is drawn """
        self.active = True

    def deactivate(self):
        """ Removes the object from being drawn """
        self.active = False


    #------------------------------
    # def bboxes
    #------------------------------

    def bbox(self, e=D_EPSILON):
        """ Create a minimal bbox for the vertex,
        for dcel to find overlapping vertices using a quadtree  """
        return Vertex.free_bbox(self.loc, e=e)

    @staticmethod
    def free_bbox(loc, e=D_EPSILON):
        """ Static method utility to create a bbox.
        used for quad_tree checking without creating the vertex """
        assert(isinstance(loc, np.ndarray))
        loc = loc.astype(np.float64)
        return np.array([loc - e, loc + e]).flatten()


    #------------------------------
    #def queries
    #------------------------------
    def is_edgeless(self):
        """ asks whether a vertext has no registered halfedges  """
        return not bool(self.half_edges)

    def has_constraints(self, candidate_set=None):
        """ if a vertex is used by more than  """
        if candidate_set is None:
            candidate_set = set()
        assert(isinstance(candidate_set, set))
        return bool(self.half_edges.difference(candidate_set))

    def get_nearby_vertices(self, e=D_EPSILON):
        """ Utility method to get nearby vertices through the dcel reference "",
        returns the list of matches *including* self """
        assert(self.dcel is not None)
        return self.dcel.vertex_quad_tree.intersect(self.bbox(e=e))

    def within(self, bbox, tolerance=TOLERANCE):
        """ Check the vertex is within [x, y, x2, y2] """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        mod_bbox = bbox + np.array([-tolerance, -tolerance, tolerance, tolerance])
        in_x_bounds = mod_bbox[0] < self.loc[0] and self.loc[0] < mod_bbox[2]
        in_y_bounds = mod_bbox[1] < self.loc[1] and self.loc[1] < mod_bbox[3]
        return in_x_bounds and in_y_bounds

    def within_circle(self, centre, radius):
        """ Check the vertex is within the radius boundary of a point """
        return in_circle(centre, radius, self.to_array())[0]

    def outside(self, bbox):
        """ Check the vertex is entirely outside of the bbox [x, y, x2, y2] """
        return not self.within(bbox)


    #------------------------------
    # def HalfEdge Access and Registration
    #------------------------------

    def register_half_edge(self, he):
        """ register a halfedge as using this vertex
        will add the vertex into the first open slot of the halfedge
        """
        #Don't assert isinstance, as that would require importing halfedge
        assert(hasattr(he, 'index'))
        self.half_edges.add(he)
        logging.debug("Registered v{} to e{}".format(self.index, he.index))

    def unregister_half_edge(self, he):
        """ Remove a halfedge from the list that uses this vertex,
        also removes the vertex from the half_edges' slot
        """
        assert(hasattr(he, 'index'))
        if he in self.half_edges:
            self.half_edges.remove(he)
        logging.debug("Remaining edges: {}".format(len(self.half_edges)))

    def get_sorted_edges(self):
        """ return all half-edges that this vertex starts,
        sorted by angle. always relative to unit vector (right) """
        opp_hedges = {x.twin.origin : x for x in self.half_edges}
        verts = opp_hedges.keys()
        sorted_verts = self.dcel.order_vertices(self.loc, verts)
        return [opp_hedges[x] for x in sorted_verts]


    #------------------------------
    # def Coordinate access
    #------------------------------
    def to_array(self):
        """ Convert the Vertex's coords to a simple numpy array """
        return self.loc


    #------------------------------
    # Def Modifiers
    #------------------------------
    def extend_line_to(self, direction=None, length=None, rad=None, target=None, edge_data=None):
        """ create a line extending out from this vertex  """
        #TODO: calc target from direction, len, rad
        if target is None:
            raise Exception("Target is None")
        assert(isinstance(target, np.ndarray))
        assert(self.dcel is not None)
        new_edge = self.dcel.create_edge(self.to_array(),
                                        target,
                                        vdata=self.data,
                                        edata=edge_data)

        #make the edge have faces:
        self.register_half_edge(new_edge)
        return new_edge

    def translate(self, direction, d=1, absolute=False, candidates=None, force=False):
        """ Move the vertex by the vector dir, scaled by distance d,
        if absolute is true, just move to the specified point"""
        assert(isinstance(direction, np.ndarray))
        assert(direction.shape == (2, ))
        if not absolute:
            target = self.to_array() + (direction * d)
        else:
            target = direction
        if not force and self.has_constraints(candidates):
            return (self.dcel.new_vertex(target), EditE.NEW)
        else:
            self.loc = target
            return (self, EditE.MODIFIED)

    def rotate(self, c=None, r=0, candidates=None, force=False):
        """ Rotate the point around a target """
        assert(isinstance(c, np.ndarray))
        assert(c.shape == (2, ))
        assert(-TWOPI <= r <= TWOPI)
        new_loc = rotate_point(self.to_array(), cen=c, rads=r)
        if not force and self.has_constraints(candidates):
            return (self.dcel.new_vertex(new_loc), EditE.NEW)
        else:
            self.loc = new_loc
            return (self, EditE.MODIFIED)

    def draw(self, ctx, data_override=None):
        data = self.data.copy()
        if data_override is not None:
            data.update(data_override)

        vert_col = VERTEX
        vert_rad = VERTRAD
        sample_description = None

        if VertE.STROKE in data and isinstance(data[VertE.STROKE], (list, np.ndarray)):
            vert_col = data[VertE.STROKE]
        if VertE.RADIUS in data:
            vert_rad = data[VertE.RADIUS]
        if VertE.SAMPLE in data:
            sample_description = data[VertE.SAMPLE]

        if sample_description is not None:
            #draw as a sampled line
            sample_description(ctx, self)

        if VertE.NULL in data:
            return

        ctx.set_source_rgba(*vert_col)
        draw_circle(ctx, *self.loc, vert_rad)
