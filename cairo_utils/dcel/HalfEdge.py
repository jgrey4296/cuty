""" HalfEdge: The intermediate level datastructure of the dcel """

import sys
import logging as root_logger
from math import pi, atan2, copysign
import numpy as np
import IPython
from cairo_utils.math import inCircle, get_distance

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

    def __init__(self, origin=None, twin=None, index=None):
        assert(origin is None or isinstance(origin, Vertex))
        assert(twin is None or isinstance(twin, HalfEdge))
        self.origin = origin
        self.twin = twin
        self.face = None
        self.next = None
        self.prev = None

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
        nextHE = self.next
        if nextHE is not None:
            nextHE = nextHE.index
        prevHE = self.prev
        if prevHE is not None:
            prevHE = prevHE.index

        return {
            'i' : self.index,
            'origin' : origin,
            'twin' : twin,
            'face' : face,
            'next' : nextHE,
            'prev' : prevHE
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
        if self.next is not None:
            n = self.next.index
        else:
            n = False
        if self.prev is not None:
            p = self.prev.index
        else:
            p = False
                        
        
        return "(HE: {}, Origin: {}, Twin: {}, Twin.Origin: {}, Prev: {}, Next: {})".format(self.index,
                                                                                            origin,
                                                                                            twin,
                                                                                            twinOrigin,
                                                                                            p,
                                                                                            n)
    
    def atan(self, centre=None):
        """ Get the angle to the half edge origin, from the centroid
        of the face it is part of, used to ensure clockwise ordering """
        assert(self.face is not None)
        assert(self.origin is not None)
        if centre is None:
            assert(hasattr(self.face, 'getCentroid'))
            centre = self.face.getCentroid()
        a = self.origin.toArray()
        centre *= [1, -1]
        a *= [1, -1]
        centre += [0, 1]
        a += [0, 1]
        o_a = a - centre
        a1 = atan2(o_a[1], o_a[0])
        return a1


    def __lt__(self, other):
        """ Compare to another half edge,
        returns true if self is anticlockwise in comparison to other
        todo: Verify this
        """
        assert(isinstance(other, HalfEdge))
        assert(hasattr(self.face, 'getCentroid'))
        centre = self.face.getCentroid()
        a = self.origin.toArray()
        b = other.origin.toArray()
        logging.debug("Lt: {}, {}, {}".format(centre, a, b))
        centre *= [1, -1]
        a *= [1, -1]
        b *= [1, -1]
        centre += [0, 1]
        a += [0, 1]
        b += [0, 1]

        c = centre
        atob = b - a
        btoc = c - b
        cross = np.cross(atob, btoc)
        return cross <= 0.0


    def intersects_edge(self, bbox):
        """ Return an integer 0-3 of the edge of a bbox the line intersects
        0 : Left Vertical Edge
        1 : Top Horizontal Edge
        2 : Right Vertical Edge
        3 : Bottom Horizontal Edge
            bbox is [min_x, min_y, max_x, max_y]
        """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        if self.origin is None or self.twin.origin is None:
            raise Exception("Invalid line boundary test ")
        bbox = bbox + np.array([EPSILON, EPSILON, -EPSILON, -EPSILON])
        start = self.origin.toArray()
        end = self.twin.origin.toArray()
        logging.debug("Checking edge intersection:\n {}\n {}\n->{}\n----".format(start,
                                                                                 end,
                                                                                 bbox))
        if start[0] <= bbox[0] or end[0] <= bbox[0]:
            return 0
        elif start[1] <= bbox[1] or end[1] <= bbox[1]:
            return 1
        elif start[0] >= bbox[2] or end[0] >= bbox[2]:
            return 2
        elif start[1] >= bbox[3] or end[1] >= bbox[3]:
            return 3
        else:
            return None #no intersection

    def connections_align(self, other):
        """ Verify that this and another halfedge's together form a full edge """
        assert(isinstance(other, HalfEdge))
        if self.twin.origin is None or other.origin is None:
            raise Exception("Invalid connection test")
        if self.origin == other.origin \
           or self.twin.origin == other.twin.origin \
           or self.origin == other.twin.origin:
            logging.debug("Unexpected connection alignment")
            #There is a reason this no longer raises.
            #todo: find out why
            #raise Exception("Unexpected Connection Alignment")
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
        return inCircle(centre, radius, points)

    
    def outside(self, bbox):
        return self.origin.outside(bbox) and self.twin.origin.outside(bbox)

    def constrain(self, bbox):
        """ Constrain the half-edge to be with a
            bounding box of [min_x, min_y, max_x, max_y]
        """
        assert(self.origin is not None)
        assert(self.twin is not None)
        assert(self.twin.origin is not None)

        #Convert to an actual line representation, for intersection
        logging.debug("Constraining {} - {}".format(self.index, self.twin.index))
        asLine = Line.newLine(self.origin, self.twin.origin, bbox)
        asLine.constrain(*bbox)
        return asLine.bounds()

    def addVertex(self, vertex):
        """ Place a vertex into the first available slot of the full edge """
        assert(isinstance(vertex, Vertex))
        if self.origin is None:
            self.origin = vertex
            self.origin.registerHalfEdge(self)
        elif self.twin.origin is None:
            self.twin.origin = vertex
            self.twin.origin.registerHalfEdge(self)
        else:
            raise Exception("trying to add a vertex to a full edge")

    def fixup(self):
        """ Fix the clockwise/counter-clockwise property of the edge """
        #Swap to maintain counter-clockwise property
        if self.fixed or self.twin.fixed:
            logging.debug("Fixing an already fixed line")
            return
        
        if self.origin is not None and self.twin.origin is not None:
            if self.face == self.twin.face:
                raise Exception("Duplicate faces?")
            selfcmp = self < self.twin
            othercmp = self.twin < self
            logging.debug("Cmp Pair: {} - {}".format(selfcmp, othercmp))
            if selfcmp != othercmp:
                logging.debug("Mismatched Indices: {}-{}".format(self.index,
                                                                 self.twin.index))
                logging.debug("Mismatched: {} - {}, ({} | {})".format(self,
                                                                      self.twin,
                                                                      self.face.getCentroid(),
                                                                      self.twin.face.getCentroid()))
                raise Exception("Mismatched orientations")
            logging.debug("CMP: {}".format(selfcmp))
            if not selfcmp:
                logging.debug("Swapping the vertices of line {} and {}".format(self.index,
                                                                               self.twin.index))
                #unregister
                self.twin.origin.unregisterHalfEdge(self.twin)
                self.origin.unregisterHalfEdge(self)
                #cache
                temp = self.twin.origin
                #switch
                self.twin.origin = self.origin
                self.origin = temp
                #re-register
                self.twin.origin.registerHalfEdge(self.twin)
                self.origin.registerHalfEdge(self)

                reCheck = self < self.twin
                reCheck_opposite = self.twin < self
                #TODO: sort this out
                if False: #not reCheck or not reCheck_opposite:
                    logging.warn("Re-Orientation failed")
                    raise Exception("Re-Orientation failed")

            self.fixed = True
            self.twin.fixed = True

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

    def swapFaces(self):
        """ Swap the registered face between the halfedges, to keep the halfedge
        as the external boundary of the face, and ordered clockwise  """
        assert(self.face is not None)
        assert(self.twin is not None)
        assert(self.twin.face is not None)
        oldFace = self.face
        self.face = self.twin.face
        self.twin.face = oldFace

    def setNext(self, nextEdge):
        assert(isinstance(nextEdge ,HalfEdge))
        self.next = nextEdge
        self.next.prev = self

    def setPrev(self, prevEdge):
        """ Set the half edge prior to this one in the CW ordering """
        assert(isinstance(prevEdge, HalfEdge))
        assert(prevEdge.twin.origin == self.origin)
        self.prev = prevEdge
        self.prev.next = self

    def connectNextToPrev(self):
        """ Removes this Halfedge from the ordering """
        if self.prev is not None:
            self.prev.next = self.next
        if self.next is not None:
            self.next.prev = self.prev

    def getVertices(self):
        """ Get a tuple of the vertices of this halfedge """
        assert(self.twin is not None)
        return (self.origin, self.twin.origin)

    def isInfinite(self):
        """ If a halfedge has only one defined point, it stretches
            off into infinity """
        return self.origin is None or self.twin is None or self.twin.origin is None

    def markForCleanup(self):
        self.markedForCleanup = True
