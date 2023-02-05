""" The highest level data structure in a dcel, apart from the dcel itself """
##-- imports
from __future__ import annotations
import logging as root_logger
from dataclasses import InitVar, dataclass, field
from itertools import cycle, islice
from typing import (Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator,
                    List, Mapping, Match, MutableMapping, Optional, Sequence,
                    Set, Tuple, TypeVar, Union, cast)

import numpy as np
from scipy.spatial import ConvexHull

from ..maths import umath as cumath
from ..constants import END, FACE, SMALL_RADIUS, START, TWOPI, VERTEX, WIDTH
from ..drawing import clear_canvas, draw_circle, draw_text
from ..umath import calc_bbox_corner, within_bbox
from .constants import EditE, FaceE
from .drawable import Drawable
from .halfedge import HalfEdge
from .vertex import Vertex

##-- end imports

logging = root_logger.getLogger(__name__)

@dataclass
class Face(Drawable):
    """
    A Face with a start point for its outer component list,
    and all of its inner components
    """

    #Site is the voronoi point that the face is built around
    site               : np.ndarray     = field()
    dcel               : DCEL           = field()
    #Primary list of ccw edges for this face
    edge_list          : List[Any]      = field(default_factory=list)
    coord_list         : List[Any]      = field(default_factory=list)
    #free vertices to build a convex hull from:
    free_vertices      : Set[Any]       = field(default_factory=set)
    data               : Dict[Any, Any] = field(default_factory=dict)
    index              : int            = field(default=None)

    marked_for_cleanup : bool           = field(init=False, default=False)
    nextIndex = 0

    def __post_init__(self):
        if site is not None:
            #site = np.array([0, 0])
            assert(isinstance(site, np.ndarray))

        if self.index is None:
            logging.debug("Creating Face {}".format(Face.nextIndex))
            self.index = Face.nextIndex
            Face.nextIndex += 1
        else:
            assert(isinstance(index, int))
            logging.debug("Re-creating Face: {}".format(index))
            self.index = index
            if self.index >= Face.nextIndex:
                Face.nextIndex = self.index + 1

        if self.dcel is not None and self not in self.dcel.faces:
            self.dcel.faces.add(self)


    def copy(self):
        """" Create a copy of the entire face """
        with self.dcel:
            #copy the halfedges
            es = [x.copy() for x in self.edge_list]
            #create a new face
            f = self.dcel.new_face(edges=es)
            #copy the data
            f.data.update(self.data)
            #return it
            return f

    #------------------------------
    # def hulls
    #------------------------------

    @staticmethod
    def hull_from_vertices(verts):
        """ Given a set of vertices, return the convex hull they form,
        and the vertices to discard """
        #TODO: put this into dcel?
        assert(all([isinstance(x, Vertex) for x in verts]))
        #convert to numpy:
        np_pairs = [(x.to_array(), x) for x in verts]
        hull = ConvexHull([x[0] for x in np_pairs])
        hull_verts = [np_pairs[x][1] for x in hull.vertices]
        discard_verts = set(verts).difference(hull_verts)
        assert(not bool(discard_verts.intersection(hull_verts)))
        assert(len(discard_verts) + len(hull_verts) == len(verts))
        return (hull_verts, list(discard_verts))

    @staticmethod
    def hull_from_coords(coords):
        """ Given a set of coordinates, return the hull they would form
        DOESN NOT RETURN DISCARDED, as the coords are not vertices yet
        """
        assert(isinstance(coords, np.ndarray))
        assert(coords.shape[1] == 2)
        hull = ConvexHull(coords)
        hull_coords = np.array([coords[x] for x in hull.vertices])
        return hull_coords


    #------------------------------
    # def Human Readable Representations
    #------------------------------


    def __str__(self):
        return "Face: {}".format(self.get_centroid())

    def __repr__(self):
        edge_list = len(self.edge_list)
        return "<Face: {}, edge_list: {}>".format(self.index, edge_list)

    def draw(self, ctx, clear=False, force_centre=False, text=False, data_override=None):
        """ Draw a single Face from a dcel.
        Can be the only thing drawn (clear=True),
        Can be drawn in the centre of the context for debugging (force_centre=True)
        """
        data = self.data.copy()
        if data_override is not None:
            assert(isinstance(data, dict))
            data.update(data_override)

        #early exits:
        if len(self.edge_list) < 2:
            return
        #Custom Clear
        if clear:
            clear_canvas(ctx)

        #Data Retrieval:
        line_wdith         = WIDTH
        vert_colour        = START
        vert_rad           = SMALL_RADIUS
        face_col           = FACE
        radius             = SMALL_RADIUS
        text_string        = "F: {}".format(self.index)
        should_offset_text = FaceE.TEXT_OFFSET in data
        centroid_col       = VERTEX
        draw_centroid      = FaceE.CENTROID in data
        sample_descr       = None

        if draw_centroid and isinstance(data[FaceE.CENTROID], (list, np.ndarray)):
            centroid_col = data[FaceE.CENTROID]
        if FaceE.STARTVERT in data and isinstance(data[FaceE.STARTVERT], (list, np.ndarray)):
            vert_colour = data[FaceE.STARTVERT]
        if FaceE.STARTRAD in data:
            vert_rad = data[FaceE.STARTRAD]
        if FaceE.FILL in data and isinstance(data[FaceE.FILL], (list, np.ndarray)):
            face_col = data[FaceE.FILL]
        if FaceE.CEN_RADIUS in data:
            radius = data[FaceE.CEN_RADIUS]
        if FaceE.TEXT in data:
            text_string = data[FaceE.TEXT]
        if FaceE.WIDTH in data:
            line_wdith = data[FaceE.WIDTH]
        if FaceE.SAMPLE in data:
            sample_descr = data[FaceE.SAMPLE]


        #Centre to context
        mid_point = (self.dcel.bbox[2:] - self.dcel.bbox[:2]) * 0.5
        face_centre = self.get_centroid()
        if force_centre:
            inv_centre = -face_centre
            ctx.translate(*inv_centre)
            ctx.translate(*mid_point)

        if sample_descr is not None:
            #draw as a sampled line
            sample_descr(ctx, self)

        if FaceE.NULL in data:
            return

        ctx.set_line_width(line_wdith)
        ctx.set_source_rgba(*face_col)
        #Setup Edges:
        initial = True
        for x in self.get_edges():
            v1, v2 = x.get_vertices()
            assert(v1 is not None)
            assert(v2 is not None)
            logging.debug("Drawing Face {} edge {}".format(self.index, x.index))
            logging.debug("Drawing Face edge from ({}, {}) to ({}, {})".format(v1.loc[0],
                                                                               v1.loc[1],
                                                                               v2.loc[0],
                                                                               v2.loc[1]))
            if initial:
                ctx.move_to(*v1.loc)
                initial = False
            ctx.line_to(*v2.loc)

            #todo move this out
            if FaceE.STARTVERT in data:
                ctx.set_source_rgba(*vert_colour)
                draw_circle(ctx, *v1.loc, vert_rad)


        #****Draw*****
        if FaceE.FILL not in data:
            ctx.stroke()
        else:
            ctx.close_path()
            ctx.fill()


        #Drawing the Centroid point
        ctx.set_source_rgba(*END)
        if draw_centroid:
            ctx.set_source_rgba(*centroid_col)
            draw_circle(ctx, *face_centre, radius)

        #Text Retrieval and drawing
        if text or FaceE.TEXT in data:
            draw_text(ctx, *face_centre, text_string, offset=should_offset_text)

        #Reset the forced centre
        if force_centre:
            ctx.translate(*(mid_point * -1))


    #------------------------------
    # def Exporting
    #------------------------------


    def _export(self):
        """ Export identifiers rather than objects, to allow reconstruction """
        logging.debug("Exporting face: {}".format(self.index))
        enum_data     = {a.name:b for a, b in self.data.items() if a in FaceE}
        non_enum_data = {a:b for a, b in self.data.items() if a not in FaceE}

        return {
            'i'             : self.index,
            'edges'         : [x.index for x in self.edge_list if x is not None],
            'sitex'         : self.site[0],
            'sitey'         : self.site[1],
            "enum_data"     : enum_data,
            "non_enum_data" : non_enum_data
        }


    def get_bbox(self):
        """ Get a rough bbox of the face """
        #TODO: fix this? its rough
        vertices      = [x.origin for x in self.edge_list]
        vertex_arrays = [x.to_array() for x in vertices if x is not None]
        if not bool(vertex_arrays):
            return np.array([[0, 0], [0, 0]])
        all_vertices = np.array([x for x in vertex_arrays])
        bbox         = np.array([np.min(all_vertices, axis=0),
                                 np.max(all_vertices, axis=0)])
        logging.debug("Bbox for Face {}  : {}".format(self.index, bbox))
        return bbox

    def mark_for_cleanup(self):
        """ Schedules the face for cleanup """
        self.marked_for_cleanup = True


    #------------------------------
    # def centroids
    #------------------------------

    def get_centroid(self):
        """ Get the user defined 'centre' of the face """
        if self.site is not None:
            return self.site.copy()
        else:
            return self.get_avg_centroid()

    def get_avg_centroid(self):
        """ Get the averaged centre point of the face from the vertices of the edges """
        k = len(self.edge_list)
        coords = np.array([x.origin.loc for x in self.edge_list])
        norm_coord = np.sum(coords, axis=0) / k
        if self.site is None:
            self.site = norm_coord
        return norm_coord

    def get_centroid_from_bbox(self):
        """ Alternate Centroid, the centre point of the bbox for the face"""
        bbox = self.get_bbox()
        difference = bbox[1, :] - bbox[0, :]
        centre = bbox[0, :] + (difference * 0.5)
        if self.site is None:
            self.site = centre
        return centre


    #------------------------------
    # def edge access
    #------------------------------

    def get_edges(self):
        """ Return a copy of the edgelist for this face """
        return self.edge_list.copy()

    def add_edges(self, edges):
        """ Add a list of edges to the face """
        assert(isinstance(edges, list))
        for x in edges:
            self.add_edge(x)

    def add_edge(self, edge):
        """ Add a constructed edge to the face """
        assert(isinstance(edge, HalfEdge))
        if edge.face is self:
            return
        if edge.face is not self and edge.face is not None:
            edge.face.remove_edge(edge)
        self.coord_list = None
        edge.face = self
        if edge not in self.edge_list:
            self.edge_list.append(edge)
        edge.marked_for_cleanup = False

    def remove_edge(self, edge):
        """ Remove an edge from this face, if the edge has this face
        registered, remove that too """
        assert(isinstance(edge, HalfEdge))
        #todo: should the edge be connecting next to prev here?
        if not bool(self.edge_list):
            return
        if edge in self.edge_list:
            self.edge_list.remove(edge)
        if edge.face is self:
            edge.face = None
        if edge.twin is None or edge.twin.face is None:
            edge.mark_for_cleanup()

    def sort_edges(self):
        """ Order the edges clockwise, by starting point, ie: graham scan """
        logging.debug("Sorting edges")
        centre = self.get_centroid()
        #verify all edges are ccw
        edges = self.edge_list.copy()
        for x in edges:
            if not x.he_ccw(centre):
                x.swapFaces()

        assert(all([x.he_ccw(centre) for x in self.edge_list]))
        # withDegrees = [(x.degrees(centre), x) for x in self.edge_list]
        # withDegrees.sort()
        # self.edge_list = [hedge for (deg, hedge) in withDegrees]
        self.edge_list.sort()

    def has_edges(self):
        """ Check if its a null face or has actual edges """
        return bool(self.edge_list)


    #------------------------------
    # def modifiers
    #------------------------------

    def subdivide(self, edge, ratio=None, angle=0):
        """ Bisect / Divide a face in half by creating a new line
        on the ratio point of the edge, at the angle specified, until it intersects
        a different line of the face.
        Angle is +- from 90 degrees.
        returns the new face
        """
        self.sort_edges()
        if ratio is None:
            ratio = 0.5
        assert(isinstance(edge, HalfEdge))
        assert(edge in self.edge_list)
        assert(0 <= ratio <= 1)
        assert(-90 <= angle <= 90)
        #split the edge
        new_point, new_edge = edge.split_by_ratio(ratio)

        #get the bisecting vector
        as_coords = edge.to_array()
        bisector = cumath.get_bisector(as_coords[0], as_coords[1])
        #get the coords of an extended line
        extended_end = cumath.extend_line(new_point.to_array(), bisector, 1000)
        el_coords = np.row_stack((new_point.to_array(), extended_end))

        #intersect with coords of edges
        intersection = None
        opp_edge = None
        for he in self.edge_list:
            if he in [edge, new_edge]:
                continue
            he_coords = he.to_array()
            intersection = cumath.intersect(el_coords, he_coords)
            if intersection is not None:
                opp_edge = he
                break
        assert(intersection is not None)
        assert(opp_edge is not None)
        #split that line at the intersection
        new_opp_point, new_opp_edge = opp_edge.split(intersection)

        #create the other face
        new_face = self.dcel.new_face()

        #create the subdividing edge:
        dividing_edge = self.dcel.new_edge(new_point, new_opp_point,
                                           face=self,
                                           twin_face=new_face,
                                           edata=edge.data,
                                           vdata=edge.origin.data)
        dividing_edge.add_prev(edge, force=True)
        dividing_edge.add_next(new_opp_edge, force=True)
        dividing_edge.twin.add_prev(opp_edge, force=True)
        dividing_edge.twin.add_next(new_edge, force=True)

        #divide the edges into new_opp_edge -> edge, new_edge -> opp_edge
        new_face_edge_group = []
        original_face_edge_update = []

        current = new_opp_edge
        while current != edge:
            assert(current.next is not None)
            original_face_edge_update.append(current)
            current = current.next
        original_face_edge_update.append(current)
        original_face_edge_update.append(dividing_edge)

        current = new_edge
        while current != opp_edge:
            assert(current.next is not None)
            new_face_edge_group.append(current)
            current.face = new_face
            current = current.next
        new_face_edge_group.append(current)
        current.face = new_face
        new_face_edge_group.append(dividing_edge.twin)

        #update the two faces edgelists
        self.edge_list = original_face_edge_update
        new_face.edge_list = new_face_edge_group

        #return both
        return (self, new_face)

    @staticmethod
    def merge_faces(*args):
        """ Calculate a convex hull from all passed in faces,
        creating a new face """
        assert(all([isinstance(x, Face) for x in args]))
        dc = args[0].dcel
        assert(dc is not None)
        all_verts = set()
        for f in args:
            all_verts.update(f.get_all_vertices())
        new_face = dc.new_face()
        #then build the convex hull
        hull, discarded = Face.hull_from_vertices(all_verts)
        for s, e in zip(hull, islice(cycle(hull), 1, None)):
            #create an edge
            dc.new_edge(s, e, face=new_face)
        #link the edges
        dc.link_edges_together(new_face.edge_list, loop=True)
        #return the face
        return (new_face, discarded)

    def translate_edge(self, transform, e=None, i=None, candidates=None, force=False):
        """ move an edge of the face """
        assert(e is None or e in self.edge_list)
        assert(i is None or 0 <= i < len(self.edge_list))
        assert(not (e is None and i is None))
        assert(isinstance(transform, np.ndarray))
        assert(transform.shape == (2, ))
        if i is None:
            i = self.edge_list.index(e)

        if not force and self.has_constraints(candidates):
            copied, _ = self.copy().translate_edge(transform, i=i, force=True)
            return (copied, EditE.NEW)

        self.edge_list[i].translate(transform, force=True)
        return (self, EditE.MODIFIED)

    def scale(self, amnt=None, target=None, vert_weights=None, edge_weights=None,
              force=False, candidates=None):
        """ Scale an entire face by amnt,
        or scale by vertex/edge normal weights """
        if not force and self.has_constraints(candidates):
            face_prime, _ = self.copy().scale(amnt=amnt, target=target,
                                              vert_weights=vert_weights,
                                              edge_weights=edge_weights,
                                              force=True)
            return (face_prime, EditE.NEW)

        if target is None:
            target = self.get_centroid_from_bbox()
        if amnt is None:
            amnt = np.ndarray([1, 1])
        assert(isinstance(amnt, np.ndarray))
        assert(amnt.shape == (2, ))
        if vert_weights is not None:
            assert(isinstance(vert_weights, np.ndarray))
        if edge_weights is not None:
            assert(isinstance(edge_weights, np.ndarray))

        verts = self.get_all_vertices()
        for vert in verts:
            loc = vert.loc.copy()
            loc -= target
            loc *= amnt
            loc += target
            vert.translate(loc, absolute=True, force=True)

        return (self, EditE.MODIFIED)

    def cut_out(self, candidates=None, force=False):
        """ Cut the Face out from its verts and halfedges that comprise it,
        creating new verts and edges, so the face can be moved and scaled
        without breaking the already existing structure """
        if not force and self.has_constraints(candidates):
            return (self.copy(), EditE.NEW)
        else:
            return (self, EditE.MODIFIED)

    def rotate(self, rads, target=None, candidates=None, force=False):
        """ copy and rotate the entire face by rotating each point """
        assert(-TWOPI <= rads <= TWOPI)
        if not force and self.has_constraints(candidates):
            face_prime, _ = self.copy().rotate(rads, target=target, force=True)
            return (face_prime, EditE.NEW)

        if target is None:
            target = self.get_centroid_from_bbox()
        assert(isinstance(target, np.ndarray))
        assert(target.shape == (2, ))

        for l in self.edge_list:
            l.rotate(c=target, r=rads, candidates=candidates, force=True)
        return (self, EditE.MODIFIED)

    def constrain_to_circle(self, centre, radius, candidates=None, force=False):
        """ Constrain the vertices and edges of a face to be within a circle """
        if not force and self.has_constraints(candidates):
            logging.debug("Face: Constraining a copy")
            face_prime, _ = self.copy().constrain_to_circle(centre, radius, force=True)
            return (face_prime, EditE.NEW)

        logging.debug("Face: Constraining edges")
        #constrain each edge
        edges = self.edge_list.copy()
        for e in edges:
            logging.debug("HE: {}".format(e))
            eprime, edit_e = e.constrain_to_circle(centre, radius, force=True)
            logging.debug("Result: {}".format(eprime))
            assert(edit_e == EditE.MODIFIED)
            assert(eprime in self.edge_list)
            if eprime.marked_for_cleanup:
                self.edge_list.remove(eprime)

        return (self, EditE.MODIFIED)

    #todo: possibly add a shrink/expand to circle method
    def constrain_to_bbox(self, bbox, candidates=None, force=False):
        """ Given a bbox, ensure all edges of the face are within """
        if not force and self.has_constraints(candidates):
            face_prime, _ = self.copy().constrain_to_bbox(bbox, force=True)
            return (face_prime, EditE.NEW)

        edges = self.edge_list.copy()

        for edge in edges:
            if edge.outside(bbox):
                self.remove_edge(edge)
                continue
            edge.constrain_to_bbox(bbox, candidates=candidates, force=True)

        return (self, EditE.MODIFIED)


    #------------------------------
    # def Vertex access
    #------------------------------

    def add_vertex(self, vert):
        """ Add a vertex, then recalculate the convex hull """
        assert(isinstance(vert, Vertex))
        self.free_vertices.add(vert)
        self.coord_list = None

    def get_all_vertices(self):
        """ Get all vertices of the face. both free and in halfedges """
        all_verts = set()
        all_verts.update(self.free_vertices)
        for e in self.edge_list:
            all_verts.update(e.get_vertices())
        return all_verts

    def get_all_coords(self):
        """ Get the sequence of coordinates for the edges """
        if self.coord_list is not None:
            return self.coord_list
        all_coords = np.array([x.to_array() for x in self.get_all_vertices()])
        self.coord_list = Face.hull_from_coords(all_coords)
        return self.coord_list


    #------------------------------
    # def verification
    #------------------------------

    def fixup(self, bbox=None):
        """ Verify and enforce correct designations of
        edge ordering, next/prev settings, and face settings """
        assert(bbox is not None)
        if not bool(self.edge_list):
            self.mark_for_cleanup()
            return []
        if len(self.edge_list) < 2:
            return []

        for e in self.edge_list:
            self.add_edge(e)

        altered = False
        centre = self.get_centroid()
        avg_centre = self.get_avg_centroid()
        if not within_bbox(centre, bbox) and within_bbox(avg_centre, bbox):
            altered = True
            self.site = avg_centre

        self.sort_edges()

        inferred_edges = []
        edges = self.edge_list.copy()
        prev = edges[-1]
        for e in edges:
            #enforce next and prev
            if e.prev is not prev:
                e.add_prev(prev, force=True)
            #if verts don't align AND they intersect the border of the bbox on separate edges:
            dont_align = not prev.connections_align(e)
            if dont_align:
                logging.debug("connections don't align")
                new_edge = self.dcel.new_edge(e.prev.twin.origin,
                                              e.origin,
                                              edata=e.data,
                                              vdata=e.origin.data)
                new_edge.face = self
                new_edge.add_prev(e.prev, force=True)
                new_edge.add_next(e, force=True)
                #insert that new edge into the edge_list
                index = self.edge_list.index(e)
                self.edge_list.insert(index, new_edge)
                inferred_edges.append(new_edge)

                #if the new_edge connects two different sides, split it and
                #force the middle vertex to the corner
                nib = new_edge.intersects_bbox(bbox)
                edge_es = set([ev for (coord, ev) in nib])
                if len(nib) == 2 and len(edge_es) > 1:
                    new_point, new_edge2 = new_edge.split_by_ratio(0.5, face_update=False)
                    new_edge2.face = self
                    self.edge_list.insert(index+1, new_edge2)
                    # if new_edge.twin.face is not None:
                    #     new_edge.twin.face.add_edge(new_edge2.twin)

                    move_to_coord = calc_bbox_corner(bbox, edge_es)
                    new_point.translate(move_to_coord, absolute=True, force=True)

            prev = e

        self.sort_edges()

        if altered:
            self.site = centre

        return inferred_edges

    def has_constraints(self, candidate_set=None):
        """ Tests whether the face's component edges and vertices are claimed by
        anything other than the face's own halfedges and their twins, and any passed in
        candidates """
        if candidate_set is None:
            candidate_set = set()
        candidates_plus_self = candidate_set.union([self],
                                                   self.edge_list,
                                                   [x.twin for x in self.edge_list
                                                    if x.twin is not None])
        return any([x.has_constraints(candidates_plus_self) for x in self.edge_list])

    def are_points_within(self, points):
        """ TODO: check whether a set of vertices are within the faces boundaries """
        assert(isinstance(points, np.ndarray))
        #see https://stackoverflow.com/questions/217578
        raise Exception("Unimplemented: are_points_within")
