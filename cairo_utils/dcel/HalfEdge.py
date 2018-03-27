""" HalfEdge: The intermediate level datastructure of the dcel """
import sys
import logging as root_logger
from math import pi, atan2, copysign, degrees
import numpy as np
import IPython
from cairo_utils.math import inCircle, get_distance, intersect, sampleAlongLine, get_normal, extend_line, rotatePoint, is_point_on_line
from cairo_utils.constants import TWOPI

from .Vertex import Vertex
from .Line import Line


logging = root_logger.getLogger(__name__)

EPSILON = sys.float_info.epsilon
PI = pi
TWOPI = 2 * PI
HALFPI = PI * 0.5
QPI = PI * 0.5

class HalfEdge:
    """ A Canonical Half-Edge. Has an origin point, and a twin
    	half-edge for its end point,
        Auto-maintains counter-clockwise vertex order with it's twin.
    	Two HalfEdges make an Edge
    """
    nextIndex = 0

    def __init__(self, origin=None, twin=None, index=None, data=None, dcel=None):
        assert(origin is None or isinstance(origin, Vertex))
        assert(twin is None or isinstance(twin, HalfEdge))
        self.origin = origin
        self.twin = twin
        #need to generate new faces:
        self.face = None
        #connected edges:
        #todo: separate into next *within* segment, and next *segment*?
        #todo: switch these back to individual next/prev
        self.nexts = set()
        self.prevs = set()
        self.dcel=dcel

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
        self.markedForCleanup = False
        self.constrained = False
        self.drawn = False
        self.fixed = False
        self.data = {}
        if data is not None:
            self.data.update(data)

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
        nextHEs = [x.index for x in self.nexts]
        prevHEs = [x.index for x in self.prevs]

        return {
            'i' : self.index,
            'origin' : origin,
            'twin' : twin,
            'face' : face,
            'nexts' : nextHEs,
            'prevs' : prevHEs,
            'data' : self.data
        }

    def __str__(self):
        return "HalfEdge: {} - {}".format(self.origin, self.twin.origin)

    def __repr__(self):
        origin = self.origin is not None
        if self.twin is not None:
            twin = self.twin.index
            twinOrigin = self.twin.origin is not None
        else:
            twin = False
            twinOrigin = False
        n = [x.index for x in self.nexts]
        p = [x.index for x in self.prevs]
                        
        coords = [str(x) for x in self.getVertices()]

        data = (self.index, origin, twin, twinOrigin, p, n, coords)
        return "(HE: {}, O: {}, T: {}, TO: {}, P: {}, N: {}, XY: {})".format(*data)
    
    def atan(self, centre=None):
        """ Get the radian to the half edge origin, from the centroid
        of the face it is part of, used to ensure clockwise ordering """
        assert(self.face is not None)
        assert(self.origin is not None)
        if centre is None:
            assert(hasattr(self.face, 'getCentroid'))
            centre = self.face.getCentroid()
        a = self.origin.toArray()
        #multiplying to... deal with inversion of cairo? TODO: check this
        centre *= [1, -1]
        a *= [1, -1]
        centre += [0, 1]
        a += [0, 1]
        o_a = a - centre
        a1 = atan2(o_a[1], o_a[0])
        return a1


    def __lt__(self, other):
        """ Compare to another half edge,
        returns true if self is anticlockwise in comparison to other,
        relative to a face
        todo: Verify this
        """
        #TODO: use ccw
        #todo: self.origin -> (self.twin | other.twin)
        raise Exception("Deprecated: HalfEdge.__lt__")


    def split(self, loc, copy_data=True):
        """ Take an s -> e, and make it now two edges s -> (x,y) -> e 
        returns (firstHalf, newPoint, secondHalf)"""
        assert(isinstance(loc, np.ndarray))
        start = self.origin
        end = self.twin.origin
        newPoint = self.dcel.newVertex(loc)
        if copy_data:
            newPoint.data.update(start.data)
        newEdge = self.dcel.newEdge(newPoint, end)
        if copy_data:
            newEdge.data.update(self.data)
        self.twin.origin = newPoint
        #update registrations:
        end.unregisterHalfEdge(self)
        newPoint.registerHalfEdge(self)        
        return (newPoint, newEdge)

    def split_by_ratio(self, r=0.5):
        """ Split an edge by a ratio of 0.0 - 1.0 : start - end.
        defaults to 0.5, the middle """
        point = sampleAlongLine(*(self.toArray().flatten()), r)
        return self.split(point[0])        

        
    def intersect(self, otherEdge):
        """ Intersect two edges mathematically,
        returns intersection point or None """
        assert(isinstance(otherEdge, HalfEdge))
        lineSegment1 = self.toArray().flatten()
        lineSegment2 = otherEdge.toArray().flatten()
        return intersect(lineSegment1, lineSegment2)

    def intersects_edge(self,bbox):
        raise Exception("Deprecated, use intersects_bbox")


    @staticmethod
    def calculate_bbox_coords(self, bbox):
        """ Assuming a bbox is defined as in constants: mix, miy, max, may,
        returns [ p1, p2, p3, p4] form the ccw set of vert coords for 4 lines
        """
        assert(isinstance(bbox,np.ndarray))
        assert(len(bbox) == 4)
        bl = bbox[:2]
        br = bbox[:2] + np.array([bbox[2], 0])
        tr = bbox[:2] + bbox[2:]
        tl = bbox[:2] + np.array([0, bbox[3]])

        return np.row_stack((bl, br, tr, tl))
                                
        
    
    def intersects_bbox(self, bbox):
        """ Return an integer 0-3 of the edge of a bbox the line intersects
        0 : Left Vertical Edge
        1 : Top Horizontal Edge
        2 : Right Vertical Edge
        3 : Bottom Horizontal Edge
        None: no intersection
            bbox is [min_x, min_y, max_x, max_y]
        """
        #calculate intersection points for each of the 4 edges of the bbox,
        #return as tuple of tuples: [( IntersectEnum, np.array(coordinates) )]
        
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        if self.origin is None or self.twin.origin is None:
            raise Exception("Invalid line boundary test ")
        #adjust the bbox by an epsilon? not sure why. TODO: test this
        bbox = bbox + np.array([-EPSILON, -EPSILON, EPSILON, EPSILON])
        start, end = self.toArray()

        logging.debug("Checking edge intersection:\n {}\n {}\n->{}\n----".format(start,
                                                                                 end,
                                                                                 bbox))
        result = []
        #convert the bbox
        #create the test lines
        #run the 4 intersections
        #package
        #return

        
        #return result
        raise Exception("Broken")


    def connections_align(self, other):
        """ Verify that this and another halfedge's together form a full edge """
        assert(isinstance(other, HalfEdge))
        if self.twin.origin is None or other.origin is None:
            raise Exception("Invalid connection test")
        
        if self.origin == other.origin \
           or self.twin.origin == other.twin.origin:
            raise Exception("Unexpected Connection Alignment")
        
        return self.twin.origin == other.origin

    def isConstrained(self):
        """ Check whether the edge has been forced within a bbox """
        return self.constrained or self.twin.constrained

    def setConstrained(self):
        """ Mark the full edge as forced within a bbox """
        self.constrained = True
        self.twin.constrained = True

    def within(self, bbox):
        """ Check that both points in an edge are within the bbox """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        return self.origin.within(bbox) and self.twin.origin.within(bbox)

    def within_circle(self, centre, radius):
        points = np.row_stack((self.origin.toArray(), self.twin.origin.toArray()))
        return all(inCircle(centre, radius, points))

    
    def outside(self, bbox):
        return self.origin.outside(bbox) and self.twin.origin.outside(bbox)

    def to_constrained(self, bbox):
        """ get the coords of the half-edge to within the
            bounding box of [min_x, min_y, max_x, max_y]
        """
        assert(self.origin is not None)
        assert(self.twin is not None)
        assert(self.twin.origin is not None)

        #Convert to an actual line representation, for intersection
        logging.debug("Constraining {} - {}".format(self.index, self.twin.index))
        asLine = Line.newLine(self.origin, self.twin.origin)
        return asLine.constrain(*bbox)

    def addVertex(self, vertex):
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

    def clearVertices(self):
        """ remove vertices from the edge, clearing the vertex->edge references as well   """
        v1 = self.origin
        v2 = self.twin.origin
        self.origin = None
        self.twin.origin = None
        if v1:
            logging.debug("Clearing vertex {} from edge {}".format(v1.index, self.index))
            v1.unregisterHalfEdge(self)
        if v2:
            logging.debug("Clearing vertex {} from edge {}".format(v2.index, self.twin.index))
            v2.unregisterHalfEdge(self.twin)

    def getVertices(self):
        """ Get a tuple of the vertices of this halfedge """
        assert(self.twin is not None)
        return (self.origin, self.twin.origin)
        
    def fixup(self):
        """ Fix the clockwise/counter-clockwise property of the edge """
        raise Exception("Deprecated: HalfEdge.fixup. Use methods in Face instead")
            
    def swapFaces(self):
        """ Swap the registered face between the halfedges, to keep the halfedge
        as the external boundary of the face, and ordered ccw  """
        assert(self.face is not None)
        assert(self.twin is not None)
        assert(self.twin.face is not None)
        oldFace = self.face
        self.face = self.twin.face
        self.twin.face = oldFace

    def addNext(self, nextEdge):
        assert(isinstance(nextEdge, HalfEdge))
        self.nexts.add(nextEdge)
        nextEdge.prevs.add(self)
        self.twin.prevs.add(nextEdge.twin)
        nextEdge.twin.nexts.add(self.twin)

    def addPrev(self, prevEdge):
        """ Set the half edge prior to this one in the CCW ordering """
        assert(isinstance(prevEdge, HalfEdge))
        assert(prevEdge.twin.origin == self.origin)
        self.prevs.add(prevEdge)
        prevEdge.nexts.add(self)
        self.twin.nexts.add(nextEdge.twin)
        nextEdge.twin.prevs.add(self.twin)

    def connectNextToPrev(self):
        """ Removes this Halfedge from the ordering """
        #for all prevs: connect to all nexts?
        #for all nexts: connect to all prevs?
        raise Exception("Unimplemented")

    def toArray(self):
        """ Get an ndarray of the bounds of the edge """
        return np.row_stack((self.origin.toArray(), self.twin.origin.toArray()))
    
    def isInfinite(self):
        """ If a halfedge has only one defined point, it stretches
            off into infinity """
        return self.origin is None or self.twin is None or self.twin.origin is None

    def markForCleanup(self):
        """ Marks this halfedge for cleanup. NOT for the twin, due to degenerate cases of hedges at boundaries """
        self.markedForCleanup = True

    def getCloserAndFurther(self, centre):
        """ Return the edge vertices ordered to be [nearer, further] from a point,
        with a flag of whether the points have been switched from the edge ordering """
        assert(isinstance(centre, np.ndarray))
        originD = get_distance(centre, self.origin.toArray())
        twinD = get_distance(centre, self.twin.origin.toArray())
        if originD <= twinD:
            return (np.row_stack((self.origin.toArray(), self.twin.origin.toArray())), False)
        else:
            return (np.row_stack((self.twin.origin.toArray(), self.origin.toArray())), True)

    def rotate(self, c=None, r=0):
        """ return Rotated coordinates as if the edge was rotated around a point by rads """
        assert(isinstance(c, np.ndarray))
        assert(c.shape == (2,))
        assert(-TWOPI <= r <= TWOPI)
        asArray = self.toArray()
        return rotatePoint(asArray, cen=c, rads=r)

        
    def extend(self, target=None, direction=None, rotate=None, d=1):
        """ Extend the line with a new line in the direction of 'target',
        or in the normalized direction 'direction', by distance d. 
        if no target or direction is passed in, it extends in the line direction """
        newEnd = None
        if sum([1 for x in [target, direction, rotate] if x is not None]) > 1:
            raise Exception("HalfEdge.extend: Specify only one of target, direction, rotate")
        if target is not None:
            assert(isinstance(target, np.ndarray))
            assert(len(target) == 2)
            if d is not None:
                newEnd = extend_line(self.twin.origin.toArray(), target, d, fromStart=False)
            else:
                newEnd = target
        elif direction is not None:
            #use the direction raw
            assert(hasattr(direction, "__len__"))
            assert(len(direction) == 2)
            assert(d is not None)
            end = self.twin.origin.toArray()
            newEnd = extend_line(end, end + direction, d)
        elif rotate is not None:
            #rotate the vector of the existing line and extend by that
            end = self.twin.origin.toArray()
            norm_vector = get_normal(self.origin.toArray(), self.twin.origin.toArray())
            rotated = rotatePoint(norm_vector, np.array([0,0]), rads=rotate)
            newEnd = extend_line(end, end + rotated, d)
        else:
            assert(d is not None)
            #get the normalized direction of self.origin -> self.twin.origin
            newEnd = extend_line(self.origin.toArray(), self.twin.origin.toArray(), d, fromStart=False)
        #Then create a point at (dir * d), create a new edge to it
        newEdge = self.dcel.createEdge(self.twin.origin.toArray(), newEnd)
        #get all edges from this edge,
        #A.Start
        #A.End, all of further edges start
        #EdgeEnds = [x.twin for x in edges if x.origin == A.End]
        
        #sort ccw order
        #[HalfEdge.ccw(A.Start, A.End, x) for x in EdgeEnds]
        #apply rules to infer faces
        left_most = False
        self.fix_faces(newEdge, left_most=left_most)
        
        self.addNext(newEdge)
        return newEdge

    def avg_direction_of_subsegments(self):
        """ Get the average normalised direction vector of each component of the 
        total line segment """
        raise Exception("Unimplemented")

    def bbox_intersect(self, e=EPSILON):
        """ Create a bbox for the total line segment, and intersect check that with the
        dcel quadtree """
        raise Exception("Unimplemented")

    def point_is_on_line(self, point):
        """ Test to see if a particular x,y coord is on a line """
        assert(isinstance(point, np.ndarray))
        assert(point.shape == (2,))
        coords = self.toArray()
        return is_point_on_line(point, coords)
    

    @staticmethod
    def compareEdges(center, a, b):
        """ Compare two halfedges against a centre point, returning whether a is CCW, equal, or CW
        from b 
        """
        assert(isinstance(center, np.ndarray))
        assert(isinstance(a, HalfEdge))
        assert(isinstance(b, HalfEdge))

        offset_a = a.origin.toArray() - center
        offset_b = b.origin.toArray() - center

        deg_a = (degrees(atan2(offset_a[1], offset_a[0])) + 360) % 360
        deg_b = (degrees(atan2(offset_b[1], offset_b[0])) + 360) % 360

        return deg_a <= deg_b

    @staticmethod
    def ccw(a, b, c):
        """ Test for left-turn on three points of a triangle """
        assert(all([isinstance(x, np.ndarray) for x in [a,b,c]]))
        offset_b = b - a
        offset_c = c - a
        crossed = np.cross(offset_b, offset_c)
        return crossed > 0

    @staticmethod
    def ccw_e(a, b, c):
        """ Test a centre point and two halfedges for ccw ordering """
        assert(isinstance(a, np.ndarray))
        assert(isinstance(b, HalfEdge))
        assert(isinstance(c, HalfEdge))
        firstOrigin = b.origin.toArray()
        secondOrigin = c.origin.toArray()
        offset_b = firstOrigin - a
        offset_c = secondOrigin - a
        crossed = np.cross(offset_b, offset_c)
        return crossed

    def __lt__(self, other):
        return HalfEdge.compareEdges(self.face.site, self, other)

    
    def he_ccw(self, centre):
        """ Verify the halfedge is ccw ordered """
        assert(isinstance(centre, np.ndarray))
        return HalfEdge.ccw(centre, self.origin.toArray(), self.twin.origin.toArray())
    
    def cross(self):
        """ Cross product of the halfedge """
        assert(self.origin is not None)
        assert(self.twin is not None)
        assert(self.twin.origin is not None)
        a = self.origin.toArray()
        b = self.twin.origin.toArray()
        return np.cross(a,b)
    

    def fix_faces(self, he, left_most=False):
        """ Infer faces by side on a vertex,
        leftmost means to fix on the right instead """
        #get mid point
        #rotate right and left and translate by a small delta
        #check the translated points are closer to the face centroid than further away
        #if they aren't, swap faces
        #raise error if they still aren't closer
        
        #self.face = he.face
        #he.face = newface
        #self.twin.face = he.face
        raise Exception("unimplemented")

        
