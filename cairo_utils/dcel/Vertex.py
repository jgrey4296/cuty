""" Vertex: The lowest level data structure in a dcel """
import sys
from numbers import Number
import logging as root_logger
import numpy as np

from cairo_utils.math import inCircle

logging = root_logger.getLogger(__name__)
EPSILON = sys.float_info.epsilon

class Vertex:
    """ A Simple vertex for two dimensions.
    Has a pair of coordinates, and stores the edges associated with it. 
    """

    nextIndex = 0

    def __init__(self, x, y, iEdge=None, index=None, dcel=None):
        assert(isinstance(x, Number))
        assert(isinstance(y, Number))
        assert(iEdge is None or isinstance(iEdge, int))
        
        self.x = x
        self.y = y
        self.incidentEdge = iEdge
        #The edges this vertex is part of:
        self.halfEdges = []
        #Custom data of the vertex:
        self.data = {}
        #Reference back to the dcel
        self.dcel = dcel
        
        self.active = True
        if index is None:
            logging.debug("Creating vertex {} at: {:.3f} {:.3f}".format(Vertex.nextIndex, x, y))
            self.index = Vertex.nextIndex
            Vertex.nextIndex += 1
        else:
            assert(isinstance(index, int))
            logging.debug("Re-Creating Vertex: {}".format(index))
            self.index = index
            if self.index >= Vertex.nextIndex:
                Vertex.nextIndex = self.index + 1

    def _export(self):
        """ Export identifiers instead of objects to allow reconstruction """
        logging.debug("Exporting Vertex: {}".format(self.index))
        #todo: add 'active', and data
        return {
            'i': self.index,
            'x': self.x,
            'y': self.y,
            'halfEdges' : [x.index for x in self.halfEdges]
        }

    def __str__(self):
        return "({:.3f},{:.3f})".format(self.x, self.y)

    def __repr__(self):
        if self.incidentEdge is not None:
            incident = self.incidentEdge.index
        else:
            incident = False
        
        return "(V: {}, incident:{}, edges: {})".format(self.index, incident, len(self.halfEdges))
    
    def isEdgeless(self):
        return len(self.halfEdges) == 0

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def bbox(self, e=EPSILON):
        """ Create a minimal bbox for the vertex,
        for dcel to find overlapping vertices using a quadtree  """
        return np.array([self.x-e,
                         self.y-e,
                         self.x+e,
                         self.y+e])

    @staticmethod
    def free_bbox(x, y, e=EPSILON):
        """ Static method utility to create a bbox. used for quad_tree checking without creating the vertex """
        return np.array([x-e,
                         y-e,
                         x+e,
                         y+e])
        
    def get_nearby_vertices(self, e=EPSILON):
        """ Utility method to get nearby vertices through the dcel reference """
        return self.dcel.vertex_quad_tree.intersect(self.bbox(e=e))

    
    def registerHalfEdge(self, he):
        """ register a halfedge as using this vertex """
        #Don't assert isinstance, as that would require importing halfedge
        assert(hasattr(he, 'index'))
        self.halfEdges.append(he)
        logging.debug("Registered v{} to e{}".format(self.index, he.index))

    def unregisterHalfEdge(self, he):
        """ Remove a halfedge from the list that uses this vertex """
        assert(hasattr(he, 'index'))
        if he in self.halfEdges:
            self.halfEdges.remove(he)
        logging.debug("Remaining edges: {}".format(len(self.halfEdges)))


    def within(self, bbox):
        """ Check the vertex is within [x,y,x2,y2] """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        inXBounds = bbox[0] <= self.x and self.x <= bbox[2]
        inYBounds = bbox[1] <= self.y and self.y <= bbox[3]
        return inXBounds and inYBounds

    def within_circle(self, centre, radius):
        """ Check the vertex is within the radius boundary of a point """
        return inCircle(centre, radius, self.toArray())[0]
    
    def outside(self, bbox):
        """ Check the vertex is entirely outside of the bbox [x,y,x2,y2] """
        return not self.within(bbox)

    def toArray(self):
        """ Convert the Vertex's coords to a simple numpy array """
        return np.array([self.x, self.y])
