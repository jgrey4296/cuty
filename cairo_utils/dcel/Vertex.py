""" Vertex: The lowest level data structure in a dcel """
import sys
import IPython
from numbers import Number
import logging as root_logger
import numpy as np

from .constants import EditE
from ..math import inCircle, rotatePoint
from ..constants import TWOPI, D_EPSILON

logging = root_logger.getLogger(__name__)

class Vertex:
    """ A Simple vertex for two dimensions.
    Has a pair of coordinates, and stores the edges associated with it. 
    """

    nextIndex = 0

    def __init__(self, loc, edges=None, index=None, data=None, dcel=None, active=None):
        assert(isinstance(loc, np.ndarray))
        assert(edges is None or isinstance(edges, list))

        self.loc = loc
        #The edges this vertex is part of:
        self.halfEdges = set()
        if edges is not None:
            self.halfEdges.update(edges)
        #Custom data of the vertex:
        self.data = {}
        if data is not None:
            self.data.update(data)
        #Reference back to the dcel
        self.dcel = dcel
        self.markedForCleanup = False
        
        self.active = True
        if active is not None:
            assert(isinstance(active, bool))
            self.active = active
        
        if index is None:
            logging.debug("Creating vertex {} at: {:.3f} {:.3f}".format(Vertex.nextIndex, loc[0], loc[1]))
            self.index = Vertex.nextIndex
            Vertex.nextIndex += 1
        else:
            assert(isinstance(index, int))
            logging.debug("Re-Creating Vertex: {}".format(index))
            self.index = index
            if self.index >= Vertex.nextIndex:
                Vertex.nextIndex = self.index + 1

    
    def copy(self):
        """ Create an isolated copy of this vertex. Doesn't copy halfedge connections, 
        but does copy data """
        newVert = self.dcel.newVertex(self.loc, data=self.data.copy())
        return newVert

    def markForCleanup(self):
        self.markedForCleanup = True

    #------------------------------
    # def exporting
    #------------------------------
    
    def _export(self):
        """ Export identifiers instead of objects to allow reconstruction """
        logging.debug("Exporting Vertex: {}".format(self.index))
        return {
            'i': self.index,
            'x': self.loc[0],
            'y': self.loc[1],
            'halfEdges' : [x.index for x in self.halfEdges],
            "data" : self.data,
            "active" : self.active
        }

    #------------------------------
    # def Human Readable Representations
    #------------------------------
    
    def __str__(self):
        return "({:.3f},{:.3f})".format(self.loc[0], self.loc[1])

    def __repr__(self):
        return "(V: {}, edges: {}, ({:.3f}, {:.3f})".format(self.index, len(self.halfEdges),
                                                           self.loc[0], self.loc[1])


    #------------------------------
    # def activation
    #------------------------------
    def activate(self):
        self.active = True

    def deactivate(self):
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
        """ Static method utility to create a bbox. used for quad_tree checking without creating the vertex """
        assert(isinstance(loc, np.ndarray))
        loc = loc.astype(np.float64)
        return np.array([loc - e, loc + e]).flatten()

    #------------------------------
    #def queries
    #------------------------------

        """ Utility method to get nearby vertices through the dcel reference "",
        returns the list of matches *including* self """
        assert(self.dcel is not None)
        return self.dcel.vertex_quad_tree.intersect(self.bbox(e=e))

    
    #------------------------------
    # def HalfEdge Access and Registration
    #------------------------------

    def registerHalfEdge(self, he):
        """ register a halfedge as using this vertex
        will add the vertex into the first open slot of the halfedge
        """
        #Don't assert isinstance, as that would require importing halfedge
        assert(hasattr(he,'index'))
        self.halfEdges.add(he)
        logging.debug("Registered v{} to e{}".format(self.index, he.index))

    def unregisterHalfEdge(self, he):
        """ Remove a halfedge from the list that uses this vertex,
        also removes the vertex from the halfEdges' slot
        """
        assert(hasattr(he,'index'))
        if he in self.halfEdges:
            self.halfEdges.remove(he)
        logging.debug("Remaining edges: {}".format(len(self.halfEdges)))


    def within(self, bbox):
        """ Check the vertex is within [x,y,x2,y2] """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        inXBounds = bbox[0] <= self.loc[0] and self.loc[0] <= bbox[2]
        inYBounds = bbox[1] <= self.loc[1] and self.loc[1] <= bbox[3]
        return inXBounds and inYBounds

    def within_circle(self, centre, radius):
        """ Check the vertex is within the radius boundary of a point """
        return inCircle(centre, radius, self.toArray())[0]
    
    def outside(self, bbox):
        """ Check the vertex is entirely outside of the bbox [x,y,x2,y2] """
        return not self.within(bbox)

        
    #------------------------------
    # def Coordinate access
    #------------------------------
    def toArray(self):
        """ Convert the Vertex's coords to a simple numpy array """
        return self.loc

    #------------------------------
    # Def Modifiers
    #------------------------------
    
    def extend_line_to(self, dir=None, len=None, rad=None, target=None, edge_data=None):
        """ create a line extending out from this vertex  """
        #TODO: calc target from dir, len, rad
        if target is None:
            raise Exception("Target is None")
        assert(isinstance(target, np.ndarray))
        assert(self.dcel is not None)
        newEdge = self.dcel.createEdge(self.toArray(),
                                     target,
                                     vdata=self.data,
                                     edata=edge_data)

        #make the edge have faces:
        self.registerHalfEdge(newEdge)
        return newEdge

    def get_sorted_edges(self):
        """ return all half-edges that this vertex starts,
        sorted by angle. always relative to unit vector (right) """
        opp_hedges = [x.twin for x in self.halfEdges]
        verts = np.array([x.origin.toArray() for x in opp_hedges])
        rads = (np.arctan2(verts[:,1], verts[:,0]) + TWOPI) % TWOPI
        ordered = sorted(zip(rads, opp_hedges))
        return [y.twin for x,y in ordered]

    
