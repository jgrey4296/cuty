""" HalfEdge: The intermediate level datastructure of the dcel """
#pylint: disable=too-many-arguments
#pylint: disable=no-self-use
import logging as root_logger
from math import pi, atan2, degrees
from itertools import islice, cycle
import numpy as np
from ..math import in_circle, intersect, sample_along_lines
from ..math import get_unit_vector, extend_line, rotate_point, is_point_on_line
from ..math import get_distance_raw, bbox_to_lines, get_midpoint, get_ranges
from ..constants import TWOPI, EPSILON, TOLERANCE
from ..constants import START, END, EDGE, D_EPSILON
from ..drawing import draw_circle, clear_canvas, draw_text
from .constants import EditE, EDGE_FOLLOW_GUARD, EdgeE
from .vertex import Vertex
from .line import Line
from .drawable import Drawable

logging = root_logger.getLogger(__name__)

PI = pi
TWOPI = 2 * PI
HALFPI = PI * 0.5
QPI = PI * 0.5

class HalfEdge(Drawable):
    """ A Canonical Half-Edge. Has an origin point, and a twin
    	half-edge for its end point,
        Auto-maintains counter-clockwise vertex order with it's twin.
    	Two HalfEdges make an Edge
    """
    nextIndex = 0

    def __init__(self, origin=None, twin=None, index=None, data=None, dcel=None):
        assert(origin is None or isinstance(origin, Vertex))
        assert(twin is None or isinstance(twin, HalfEdge))
        super().__init__()
        self.origin = origin
        self.twin = twin
        self.length_sq = -1
        #need to generate new faces:
        self.face = None
        #connected edges:
        self.next = None
        self.prev = None
        self.dcel = dcel

        if index is None:
            logging.debug("Creating Edge {}".format(HalfEdge.nextIndex))
            self.index = HalfEdge.nextIndex
            HalfEdge.nextIndex += 1
        else:
            assert(isinstance(index, int))
            logging.debug("Re-creating Edge: {}".format(index))
            self.index = index
            if self.index >= HalfEdge.nextIndex:
                HalfEdge.nextIndex = self.index + 1

        #register the halfedge with the vertex
        if origin is not None:
            self.origin.registerHalfEdge(self)

        #Additional:
        self.marked_for_cleanup = False
        self.constrained = False
        self.drawn = False
        self.fixed = False
        self.data = {}
        if data is not None:
            self.data.update(data)
        if self.dcel is not None and self not in self.dcel.halfEdges:
            self.dcel.halfEdges.add(self)

    def eq_verts(self, other):
        """ Test whether two halfedges share vertices """
        assert(isinstance(other, HalfEdge))
        s_verts = self.get_vertices()
        o_verts = other.get_vertices()
        return all([s == o for s, o in zip(s_verts, o_verts)])

    def copy(self):
        """ Copy the halfedge pair. sub-copies the vertexs too """
        assert(self.origin is not None)
        assert(self.twin is not None)
        #copy the vertices
        v1 = self.origin.copy()
        v2 = self.twin.origin.copy()
        #create the halfedge
        e = self.dcel.new_edge(v1, v2)
        #update next/prev?

        #copy data
        e.data.update(self.data)
        return e



    #------------------------------
    # def export
    #------------------------------


    def _export(self):
        """ Export identifiers instead of objects to allow reconstruction """
        logging.debug("Exporting Edge: {}".format(self.index))
        origin = self.origin
        if origin is not None:
            origin = origin.index
        twin = self.twin
        if twin is not None:
            twin = twin.index
        face = self.face
        if face is not None:
            face = face.index
        next_he = None
        prev_he = None
        if self.next is not None:
            next_he = self.next.index
        if self.prev is not None:
            prev_he = self.prev.index
        enum_data = {a.name:b for a, b in self.data.items() if a in EdgeE}
        non_enum_data = {a:b for a, b in self.data.items() if a not in EdgeE}


        return {
            'i' : self.index,
            'origin' : origin,
            'twin' : twin,
            'face' : face,
            'next' : next_he,
            'prev' : prev_he,
            "enum_data" : enum_data,
            "non_enum_data": non_enum_data
        }

    #------------------------------
    # def Human Readable Representations
    #------------------------------

    def __str__(self):
        return "HalfEdge {}: {} - {}".format(self.index, self.origin, self.twin.origin)

    def __repr__(self):
        origin = "n/a"
        twin = "n/a"
        if self.origin is not None:
            origin = self.origin.index
        if self.twin is not None:
            twin = self.twin.index
        n = "n/a"
        p = "n/a"
        if self.next is not None:
            n = self.next.index
        if self.prev is not None:
            p = self.prev.index
        f = "n/a"
        if self.face is not None:
            f = self.face.index

        coords = [str(x) for x in self.get_vertices()]

        data = (self.index, f, origin, twin, p, n, coords)
        return "(HE: {}, f: {}, O: {}, T: {}, P: {}, N: {}, XY: {})".format(*data)

    def draw(self, ctx, data_override=None, clear=False, text=False, width=None):
        #pylint: disable=too-many-locals
        logging.debug("Drawing Edge: {} | {}".format(self.index, self.twin.index))
        if clear:
            clear_canvas(ctx)
        data = self.data.copy()
        if data_override is not None:
            assert(isinstance(data_override, dict))
            data.update(data_override)

        #defaults
        colour = EDGE
        start_end_points = False
        start_col = START
        end_col = END
        start_rad = width
        end_rad = width
        write_text = "HE:{}.{}".format(self.index, self.twin.index)
        bezier = False
        bezier_simp = False
        sample_description = None

        #retrieve custom values
        if EdgeE.WIDTH in data:
            width = data[EdgeE.WIDTH]
            start_rad = width
            end_rad = width
        if EdgeE.STROKE in data:
            colour = data[EdgeE.STROKE]
        if EdgeE.START in data and isinstance(data[EdgeE.START], (list, np.ndarray)):
            start_col = data[EdgeE.START]
        if EdgeE.END in data and isinstance(data[EdgeE.END], (list, np.ndarray)):
            end_col = data[EdgeE.END]
        if EdgeE.START in data and EdgeE.END in data:
            start_end_points = True
        if EdgeE.STARTRAD in data:
            start_rad = data[EdgeE.STARTRAD]
        if EdgeE.ENDRAD in data:
            end_rad = data[EdgeE.ENDRAD]
        if EdgeE.TEXT in data:
            if isinstance(data[EdgeE.TEXT], str):
                write_text = data[EdgeE.TEXT]
            elif not data[EdgeE.TEXT]:
                write_text = None
        if EdgeE.BEZIER in data:
            bezier = data[EdgeE.BEZIER]
            assert(isinstance(bezier, list))
        if EdgeE.BEZIER_SIMPLIFY in data:
            bezier_simp = True
        if EdgeE.SAMPLE in data:
            sample_description = data[EdgeE.SAMPLE]

        #Get Start and end points
        v1, v2 = self.get_vertices()
        if v1 is None or v2 is None:
            #early exit if line is not completed
            return
        centre = get_midpoint(v1.to_array(), v2.to_array())
        logging.debug("Drawing HalfEdge {} : {}, {} - {}, {}".format(self.index,
                                                                     v1.loc[0],
                                                                     v1.loc[1],
                                                                     v2.loc[0],
                                                                     v2.loc[1]))
        if sample_description is not None:
            #draw as a sampled line
            sample_description(ctx, self)

        if EdgeE.NULL in data:
            return

        ctx.set_line_width(width)
        ctx.set_source_rgba(*colour)

        #draw as a line/curve
        #todo: allow beziers to be simplified to straight lines
        if bool(bezier):
            logging.debug("Drawing Bezier: {}".format(bezier))
            ctx.new_path()
            for b in bezier:
                ctx.move_to(*b[0])
                if bezier_simp:
                    ctx.line_to(*b[-1])
                    continue
                if len(b) == 3:
                    ctx.curve_to(*b[0], *b[1], *b[2])
                else:
                    assert(len(b) == 4)
                    ctx.curve_to(*b[1], *b[2], *b[3])

        else:
            logging.debug("Drawing Straight Line")
            ctx.move_to(*v1.loc)
            ctx.line_to(*v2.loc)

        ctx.stroke()

        if start_end_points:
            ctx.set_source_rgba(*start_col)
            draw_circle(ctx, *v1.loc, start_rad)
            ctx.set_source_rgba(*end_col)
            draw_circle(ctx, *v2.loc, end_rad)

        if text and write_text is not None:
            draw_text(ctx, *centre, write_text)


    #------------------------------
    # def Math
    #------------------------------

    def cross(self):
        """ Cross product of the halfedge """
        assert(self.origin is not None)
        assert(self.twin is not None)
        assert(self.twin.origin is not None)
        a = self.origin.to_array()
        b = self.twin.origin.to_array()
        return np.cross(a, b)

    def get_length_sq(self, force=False):
        """ Gets the calculated length, or calculate it. returns as a np.ndarray"""
        if not force and self.length_sq is not -1:
            return self.length_sq
        #otherwise calculate
        as_array = self.to_array()
        self.length_sq = get_distance_raw(as_array[0], as_array[1])
        return self.length_sq

    #------------------------------
    # def Modifiers
    #------------------------------

    def split(self, loc, copy_data=True, face_update=True):
        """ Take an s -> e, and make it now two edges s -> (x, y) -> e
        returns (firstHalf, new_point, secondHalf)"""
        assert(isinstance(loc, (np.ndarray, Vertex)))
        start = self.origin
        end = self.twin.origin
        if isinstance(loc, Vertex):
            new_point = loc
        else:
            new_point = self.dcel.new_vertex(loc)
        if copy_data:
            new_point.data.update(start.data)
        new_edge = self.dcel.new_edge(new_point, end)
        if copy_data:
            new_edge.data.update(self.data)
        #update the twin
        self.twin.origin = new_point
        #update registrations:
        end.unregisterHalfEdge(self)
        new_point.registerHalfEdge(self)
        new_point.registerHalfEdge(self.twin)
        end.unregisterHalfEdge(self.twin)
        end.registerHalfEdge(new_edge.twin)
        #recalculate length
        self.get_length_sq(force=True)
        self.twin.get_length_sq(force=True)
        #insert into next/prev ordering
        new_edge.add_next(self.next, force=True)
        new_edge.twin.add_prev(self.twin.prev, force=True)
        self.add_next(new_edge, force=True)
        new_edge.twin.add_next(self.twin, force=True)
        #update faces
        if face_update and self.face is not None:
            self.face.add_edge(new_edge)
        if face_update and self.twin.face is not None:
            self.twin.face.add_edge(new_edge.twin)
        return (new_point, new_edge)

    def split_by_ratio(self, r=0.5, face_update=True):
        """ Split an edge by a ratio of 0.0 - 1.0 : start - end.
        defaults to 0.5, the middle """
        point = sample_along_lines((self.to_array().flatten()), np.array([r]))[0]
        return self.split(point[0], face_update=face_update)

    def translate(self, direction, d=1, absolute=False, candidates=None, force=False):
        """ Move the edge by a vector and distance, or to an absoluteolute location """
        assert(isinstance(direction, np.ndarray))
        if not absolute:
            target = self.to_array() + (direction * d)
        else:
            assert(direction.shape == (2, 2))
            target = direction

        if not force and self.has_constraints(candidates):
            return (self.dcel.createEdge(target[0],
                                         target[1],
                                         edata=self.data,
                                         vdata=self.origin.data), EditE.NEW)
        else:
            _, edit1 = self.origin.translate(target[0], absolute=True, force=True)
            _, edit2 = self.twin.origin.translate(target[1], absolute=True, force=True)
            assert(edit1 == edit2)
            assert(edit1 == EditE.MODIFIED)
            return (self, EditE.MODIFIED)


    def extend(self, target=None, direction=None, rotate=None, d=1, in_sequence=True):
        """ Extend the line with a new line in the direction of 'target',
        or in the normalized direction 'direction', by distance d.
        if no target or direction is passed in, it extends in the line direction """
        start = self.origin.to_array()
        end = self.twin.origin.to_array()
        new_end = None
        if sum([1 for x in [target, direction, rotate] if x is not None]) > 1:
            raise Exception("HalfEdge.extend: Specify only one of target, direction, rotate")
        if target is not None:
            assert(isinstance(target, np.ndarray))
            assert(len(target) == 2)
            if d is not None:
                new_end = extend_line(end, target, d, from_start=False)
            else:
                new_end = target
        elif direction is not None:
            #use the direction raw
            assert(hasattr(direction, "__len__"))
            assert(len(direction) == 2)
            assert(d is not None)
            new_end = extend_line(end, end + direction, d)
        elif rotate is not None:
            #rotate the vector of the existing line and extend by that
            unit_vector = get_unit_vector(start, end)
            rotated = rotate_point(unit_vector, np.array([0, 0]), rads=rotate)
            new_end = extend_line(end, end + rotated, d)
        else:
            assert(d is not None)
            #get the normalized direction of self.origin -> self.twin.origin
            new_end = extend_line(start, end, from_start=False)
        #Then create a point at (dir * d), create a new edge to it
        new_vert = self.dcel.new_vertex(new_end)

            #todo: twinNext is the next ccw edge for the correct face

        new_edge = self.dcel.new_edge(self.twin.origin, new_vert,
                                      edata=self.data, vdata=self.origin.data)
        new_edge.fix_faces(self)

        return new_edge

    def rotate(self, c=None, r=0, candidates=None, force=False):
        """ return Rotated coordinates as if the edge was rotated around a point by rads """
        assert(isinstance(c, np.ndarray))
        assert(c.shape == (2, ))
        assert(-TWOPI <= r <= TWOPI)
        as_array = self.to_array()
        rotated_coords = rotate_point(as_array, cen=c, rads=r)

        if not force and self.has_constraints(candidates):
            return (self.dcel.createEdge(rotated_coords[0],
                                         rotated_coords[1],
                                         edata=self.data,
                                         vdata=self.origin.data), EditE.NEW)
        else:
            _, edit1 = self.origin.translate(rotated_coords[0], abs=True, force=True)
            _, edit2 = self.twin.origin.translate(rotated_coords[1], abs=True, force=True)
            assert(edit1 == edit2)
            assert(edit1 == EditE.MODIFIED)
            return (self, EditE.MODIFIED)

    def constrain_to_circle(self, centre, radius, candidates=None, force=False):
        """ Modify or create a new edge that is constrained to within a circle,
        while also marking the original edge for cleanup if necessary """
        #pylint: disable=too-many-locals
        #todo: handle sequences
        assert(isinstance(centre, np.ndarray))
        assert(centre.shape == (2, ))
        assert(radius >= 0)
        results = self.within_circle(centre, radius)
        logging.debug("HE: within_circle? {}".format(results))
        if all(results):
            #nothing needs to be done
            logging.debug("HE: fine")
            return (self, EditE.MODIFIED)
        if not any(results):
            logging.debug("HE: to remove")
            self.mark_for_cleanup()
            return (self, EditE.MODIFIED)

        closer, further = self.get_closer_and_further(centre)
        as_line = Line.new_line(np.array([closer.to_array(), further.to_array()]))
        intersections = as_line.intersect_with_circle(centre, radius)

        distances = get_distance_raw(further.to_array(), intersections)
        closest = intersections[np.argmin(distances)]
        vert_target = None

        if not force and self.has_constraints(candidates):
            edit_e = EditE.NEW
            vert_target = self.dcel.new_vertex(closest)
            target = self.copy()
            self.mark_for_cleanup()
        else:
            edit_e = EditE.MODIFIED
            target = self

        if further == self.origin:
            if vert_target is not None:
                target.replace_vertex(vert_target)
            else:
                target.origin.loc = closest
        else:
            if vert_target is not None:
                target.twin.replace_vertex(vert_target)
            else:
                target.twin.origin.loc = closest


        return (target, edit_e)

    def constrain_to_bbox(self, bbox, candidates=None, force=False):
        """ force this edge within a bbox, or a new edge to be """
        #pylint: disable=too-many-locals
        if not force and self.has_constraints(candidates):
            edge_prime, _ = self.copy().constrain_to_bbox(bbox, force=True)
            return (edge_prime, EditE.NEW)

        #get intersections with bbox
        intersections = self.intersects_bbox(bbox)
        verts = self.get_vertices()
        vert_coords = self.to_array()

        if self.within(bbox):
            logging.debug("Ignoring halfedge: is within bbox")
        elif self.outside(bbox):
            self.mark_for_cleanup()
        elif not bool(intersections):
            raise Exception("Edge Constraining: Part in and out, with no intersection")
        elif len(intersections) == 1:
            logging.debug("One intersection, moving outside vertex")
            intersect_coords, _ = intersections[0]
            outside_verts = [x for x in verts if not x.within(bbox)]
            assert(len(outside_verts) == 1)
            if outside_verts[0] == self.origin:
                d = self.origin.data
                target = self
            else:
                d = self.twin.origin.data
                target = self.twin
            new_vert = self.dcel.new_vertex(intersect_coords, data=d)
            target.replace_vertex(new_vert)

        elif len(intersections) == 2:
            logging.debug("Two intersections, moving both vertices")
            for i_c, _ in intersections:
                vert_to_move = verts[np.argmin(get_distance_raw(vert_coords, i_c))]
                if vert_to_move == self.origin:
                    d = self.origin.data
                    target = self
                else:
                    d = self.twin.origin.data
                    target = self.twin
                new_vert = self.dcel.new_vertex(intersect_coords, data=d)
                target.replace_vertex(new_vert)


        return (self, EditE.MODIFIED)




    #------------------------------
    # def Comparison
    #------------------------------

    def intersect(self, other_edge):
        """ Intersect two edges mathematically,
        returns intersection point or None """
        assert(isinstance(other_edge, HalfEdge))
        line_segment_1 = self.to_array()
        line_segment_2 = other_edge.to_array()
        return intersect(line_segment_1, line_segment_2)

    def intersects_bbox(self, bbox, tolerance=TOLERANCE):
        """ Return an enum of the edges of a bbox the line intersects
        returns a cairo_utils.constants.IntersectEnum
        returns a list. empty list is no intersections

            bbox is [min_x, min_y, max_x, max_y]
        """
        #calculate intersection points for each of the 4 edges of the bbox,
        #return as tuple of tuples: [( IntersectEnum, np.array(coordinates) )]

        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        if self.origin is None or self.twin.origin is None:
            raise Exception("Invalid line boundary test ")
        #adjust the bbox by an epsilon? not sure why. TODO: test this
        bbox_lines = bbox_to_lines(bbox)
        self_line_segment = self.to_array()
        start, end = self.to_array()

        logging.debug("Checking edge intersection:\n {}\n {}\n->{}\n----".format(start,
                                                                                 end,
                                                                                 bbox))
        result = []
        #run the 4 intersections
        for (curr_line, enum_value) in bbox_lines:
            intersected = intersect(self_line_segment, curr_line, tolerance=tolerance)
            if intersected is not None:
                result.append((intersected, enum_value))

        assert(len(result) < 3)
        return result


    def point_is_on_line(self, point):
        """ Test to see if a particular x, y coord is on a line """
        assert(isinstance(point, np.ndarray))
        assert(point.shape == (2, ))
        coords = self.to_array()
        return is_point_on_line(point, coords)


    def __call__(self, x=None, y=None):
        """ Pass in a value and calculate the other """
        assert(any([a is not None for a in [x, y]]))
        assert(not all([a is not None for a in [x, y]]))
        the_line = Line.new_line(self.to_array())
        return the_line(x=x, y=y)

    def get_ranges(self):
        """ Get the bbox of the halfedge """
        arr = get_ranges(self.to_array())
        return arr

    @staticmethod
    def compare_edges(center, a, b):
        """ Compare two halfedges against a centre point,
        returning whether a is CCW, equal, or CW from b
        """
        assert(isinstance(center, np.ndarray))
        assert(isinstance(a, HalfEdge))
        assert(isinstance(b, HalfEdge))

        offset_a = a.origin.to_array() - center
        offset_b = b.origin.to_array() - center

        deg_a = (degrees(atan2(offset_a[1], offset_a[0])) + 360) % 360
        deg_b = (degrees(atan2(offset_b[1], offset_b[0])) + 360) % 360

        return deg_a <= deg_b

    def degrees(self, centre):
        """ how many degrees from the center is this halfedge? """
        offset = self.origin.to_array() - centre
        deg = (degrees(atan2(offset[1], offset[0])) + 360) % 360
        return deg

    @staticmethod
    def ccw(a, b, c):
        """ Test for left-turn on three points of a triangle """
        assert(all([isinstance(x, np.ndarray) for x in [a, b, c]]))
        offset_b = b - a
        offset_c = c - a
        crossed = np.cross(offset_b, offset_c)
        return crossed >= 0

    @staticmethod
    def ccw_e(a, b, c):
        """ Test a centre point and two halfedges for ccw ordering """
        assert(isinstance(a, np.ndarray))
        assert(isinstance(b, HalfEdge))
        assert(isinstance(c, HalfEdge))
        first_origin = b.origin.to_array()
        second_origin = c.origin.to_array()
        offset_b = first_origin - a
        offset_c = second_origin - a
        crossed = np.cross(offset_b, offset_c)
        return crossed

    def __lt__(self, other):
        return HalfEdge.compare_edges(self.face.getCentroid(), self, other)


    def he_ccw(self, centre):
        """ Verify the halfedge is ccw ordered """
        assert(isinstance(centre, np.ndarray))
        return HalfEdge.ccw(centre, self.origin.to_array(), self.twin.origin.to_array())

    def is_upper(self):
        """ Is this halfedge the higher of the pair? """
        verts = self.get_vertices()
        return verts[0] < verts[1]

    def is_flat(self):
        """ is this halfedge flat? """
        arr = self.to_array()
        return arr[0, 1] == arr[1, 1]

    def contains_vertex(self, vert, tolerance=D_EPSILON):
        """ is the vertex passed in on the halfedge? """
        assert(isinstance(vert, Vertex))
        verts = self.to_array()
        if vert in verts:
            return True
        l = self.to_array()
        p = vert.to_array()
        return is_point_on_line(p, l)
        # l = Line.new_line(verts)
        # xprime = l(y=vert.loc[1])[0]
        # diff = abs(vert.loc[0]) - abs(xprime)
        # return -(tolerance) <= diff <= tolerance


    #------------------------------
    # def Utilities
    #------------------------------

    @staticmethod
    def avg_direction(edges):
        """ Get the average normalised direction vector of each component of the
        total line segment """
        assert(isinstance(edges, list))
        assert(all([isinstance(x, HalfEdge) for x in edges]))
        all_lines = [Line.new_line(x.to_array()) for x in edges]
        all_directions = np.array([x.direction for x in all_lines])
        direction = all_directions.sum(axis=0) / len(edges)
        return direction

    def vertex_intersections(self, e=EPSILON):
        """ Create a bbox for the total line segment, and intersect check that with the
        dcel quadtree """
        raise Exception("Unimplemented")

    def follow_sequence(self, backwards=False, guard=EDGE_FOLLOW_GUARD):
        """ Follow the .next or .prev chain to completion or loop """
        count = 1
        edges = [self]
        getter = lambda x: x.next
        if backwards:
            getter = lambda x: x.prev
        current = getter(self)
        #todo: possibly use a set and stop on any loop
        while count < guard and current is not None and current is not self:
            edges.append(current)
            current = getter(current)
            count += 1

        return edges




    #------------------------------
    # def Verification
    #------------------------------

    def fix_faces(self, originator):
        """ Infer faces by side on a vertex,
        leftmost means to fix on the right instead """
        extended_from = originator
        all_twins = [x.twin.origin for x in self.origin.halfEdges]
        edge_lookup = {x.twin.origin : x.twin for x in self.origin.halfEdges}
        assert(extended_from.origin in all_twins)
        ordered = self.dcel.orderVertices(self.origin.loc, all_twins)
        extended_index = ordered.index(extended_from.origin)
        zipped = zip(islice(cycle(ordered), extended_index, len(ordered) + extended_index),
                     islice(cycle(ordered), extended_index+1, len(ordered) + extended_index + 1))


        for a, b in zipped:
            a_edge = edge_lookup[a]
            b_edge = edge_lookup[b]
            a_edge.twin.add_prev(b_edge, force=True)

        if self.prev.face is None:
            new_face = self.dcel.newFace()
            new_face.add_edge(self.prev)
        if originator.twin.face is None:
            orig_twin_face = self.dcel.newFace()
            orig_twin_face.add_edge(originator.twin)


        self.prev.face.add_edge(self)
        f2_sequence = self.twin.follow_sequence()
        if originator.twin in f2_sequence:
            originator.twin.face.add_edge(self.twin)
        else:
            twin_face = self.dcel.newFace()
            for e in f2_sequence:
                self.prev.face.remove_edge(e)
                twin_face.add_edge(e)





    def has_constraints(self, candidate_set=None):
        """ Tests whether the halfedge, and its vertices, are used by things other than the
        faces, halfedges, and vertices passed in as the candidate set """
        if candidate_set is None:
            candidate_set = set()
        assert(isinstance(candidate_set, set))
        if self.twin is not None:
            candidates_plus_self = candidate_set.union([self, self.twin])
        else:
            candidates_plus_self = candidate_set.union([self])
        is_constrained = self.face is not None and self.face not in candidates_plus_self
        if self.origin is not None:
            is_constrained = is_constrained \
                            or self.origin.has_constraints(candidates_plus_self)
        if self.twin is not None:
            if self.twin.face is not None:
                is_constrained = is_constrained or self.twin.face not in candidates_plus_self
            if self.twin.origin is not None:
                is_constrained = is_constrained \
                                or self.twin.origin.has_constraints(candidates_plus_self)
        return is_constrained

    def is_infinite(self):
        """ If a halfedge has only one defined point, it stretches
            off into infinity """
        return self.origin is None or self.twin is None or self.twin.origin is None

    def connections_align(self, other):
        """ Verify that this and another halfedge's together form a full edge """
        assert(isinstance(other, HalfEdge))
        if self.twin.origin is None or other.origin is None:
            raise Exception("Invalid connection test")

        return self.twin.origin == other.origin

    def is_constrained(self):
        """ Check whether the edge has been forced within a bbox or circle"""
        return self.constrained or self.twin.constrained

    def set_constrained(self):
        """ Mark the full edge as forced within a bbox or circle """
        self.constrained = True
        self.twin.constrained = True

    def within(self, bbox):
        """ Check that both points in an edge are within the bbox """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        return self.origin.within(bbox) and self.twin.origin.within(bbox)

    def within_circle(self, centre, radius):
        """ is this halfedge within a given circle? """
        points = self.to_array()
        return in_circle(centre, radius, points)

    def outside(self, bbox):
        """ is this halfedge outside a given bbox? """
        verts = [x for x in self.get_vertices() if x is not None]
        return all([x.outside(bbox) for x in verts])


    def to_constrained(self, bbox):
        """ get the coords of the half-edge to within the
            bounding box of [min_x, min_y, max_x, max_y]
        """
        assert(self.origin is not None)
        assert(self.twin is not None)
        assert(self.twin.origin is not None)

        #Convert to an actual line representation, for intersection
        logging.debug("Constraining {} - {}".format(self.index, self.twin.index))
        as_line = Line.new_line(self.to_array())
        return as_line.constrain(*bbox)

    def swap_faces(self):
        """ Swap the registered face between the halfedges, to keep the halfedge
        as the external boundary of the face, and ordered ccw  """
        assert(self.face is not None)
        assert(self.twin is not None)
        #assert(self.twin.face is not None)
        origin_faces = self.face
        twin_faces = self.twin.face
        origin_faces.remove_edge(self)
        if twin_faces is not None:
            twin_faces.remove_edge(self.twin)
            twin_faces.add_edge(self)
        origin_faces.add_edge(self.twin)


    #------------------------------
    # def Vertex Access
    #------------------------------

    def add_vertex(self, vertex):
        """ Place a vertex into the first available slot of the full edge """
        assert(isinstance(vertex, Vertex))
        if self.origin is None:
            self.origin = vertex
            self.origin.registerHalfEdge(self)
        elif self.twin.origin is None:
            self.twin.origin = vertex
            self.twin.origin.registerHalfEdge(self.twin)
        else:
            raise Exception("trying to add a vertex to a full edge")

    def clear_vertices(self):
        """ remove vertices from the edge, clearing the vertex->edge references as well   """
        v1 = self.origin
        v2 = None
        self.origin = None
        if self.twin is not None:
            v2 = self.twin.origin
            self.twin.origin = None

        if v1 is not None:
            logging.debug("Clearing vertex {} from edge {}".format(v1.index, self.index))
            v1.unregisterHalfEdge(self)
        if v2 is not None:
            logging.debug("Clearing vertex {} from edge {}".format(v2.index, self.twin.index))
            v2.unregisterHalfEdge(self.twin)


    def replace_vertex(self, new_vert):
        """ Replace the vertex of this halfedge with a new one, unregistering the old """
        assert(isinstance(new_vert, Vertex))
        self.origin.unregisterHalfEdge(self)
        self.origin = new_vert
        self.origin.registerHalfEdge(self)


    def get_vertices(self):
        """ Get a tuple of the vertices of this halfedge """
        if self.twin is None:
            return (self.origin, None)
        return (self.origin, self.twin.origin)

    def to_array(self):
        """ Get an ndarray of the bounds of the edge """
        return np.row_stack((self.origin.to_array(), self.twin.origin.to_array()))

    def get_closer_and_further(self, centre):
        """ Return the edge vertices ordered to be [nearer, further] from a point,
        with a flag of whether the points have been switched from the edge ordering """
        assert(isinstance(centre, np.ndarray))
        distances = get_distance_raw(centre, self.to_array())
        if distances[0] < distances[1]:
            return (self.origin, self.twin.origin)
        else:
            return (self.twin.origin, self.origin)


    #------------------------------
    # def Edge Sequencing
    #------------------------------

    def add_next(self, next_edge, force=False):
        """ Add a halfedge to be the next in the chain """
        assert(next_edge is None or isinstance(next_edge, HalfEdge))
        if not force:
            assert(self.next is None)
            assert(next_edge is None or next_edge.prev is None)
        if self.next is not None:
            self.next.prev = None
        self.next = next_edge
        if self.next is not None:
            self.next.prev = self

    def add_prev(self, prev_edge, force=False):
        """ Set the half edge prior to this one in the CCW ordering """
        assert(prev_edge is None or isinstance(prev_edge, HalfEdge))
        if not force:
            assert(self.prev is None)
            assert(prev_edge is None or prev_edge.next is None)
        if self.prev is not None:
            self.prev.next = None
        self.prev = prev_edge
        if self.prev is not None:
            self.prev.next = self

    def connect_next_to_prev(self):
        """ Removes this Halfedge from the ordering """
        hprev = self.prev
        hnext = self.next
        if hprev is not None:
            hprev.next = hnext
        if hnext is not None:
            hnext.prev = hprev

    #------------------------------
    # def Cleanup
    #------------------------------

    def mark_for_cleanup(self):
        """ Marks this halfedge for cleanup. NOT for the twin,
        due to degenerate cases of hedges at boundaries """
        self.marked_for_cleanup = True
