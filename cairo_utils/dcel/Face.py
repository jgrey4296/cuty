""" The highest level data structure in a dcel, apart from the dcel itself """
import logging as root_logger
import numpy as np
from numbers import Number

from .HalfEdge import HalfEdge

logging = root_logger.getLogger(__name__)

class Face(object):
    """ A Face with a start point for its outer component list,
    and all of its inner components """

    nextIndex = 0

    def __init__(self, site_x, site_y, index=None):
        assert(isinstance(site_x, Number))
        assert(isinstance(site_y, Number))
        #Site is the voronoi point that the face is built around
        self.site = np.array([site_x, site_y])
        #Starting point for bounding edges, going anti-clockwise
        self.outerComponent = None
        #Clockwise inner loops
        self.innerComponents = []
        self.edgeList = []
        #mark face for cleanup:
        self.markedForCleanup = False
        #Additional Data:
        self.data = {}
        if index is None:
            logging.debug("Creating Face {}".format(Face.nextIndex))
            self.index = Face.nextIndex
            Face.nextIndex += 1
        else:
            assert(isinstance(index, int))
            logging.debug("Re-creating Face: {}".format(index))
            self.index = index
            if self.index >= Face.nextIndex:
                Face.nextIndex = self.index + 1


    def __str__(self):
        return "Face: {}".format(self.getCentroid())

    def _export(self):
        """ Export identifiers rather than objects, to allow reconstruction """
        logging.debug("Exporting face: {}".format(self.index))
        return {
            'i' : self.index,
            'edges' : [x.index for x in self.edgeList if x is not None],
            'sitex' : self.site[0],
            'sitey' : self.site[1],
        }

    def removeEdge(self, edge):
        assert(isinstance(edge, HalfEdge))
        #todo: should the edge be connecting next to prev here?
        self.innerComponents.remove(edge)
        self.edgeList.remove(edge)

    def get_bbox(self):
        #TODO: fix this? its rough
        vertices = [x.origin for x in self.edgeList]
        vertexArrays = [x.toArray() for x in vertices if x is not None]
        if not bool(vertexArrays):
            return np.array([[0, 0], [0, 0]])
        allVertices = np.array([x for x in vertexArrays])
        bbox = np.array([[allVertices[:, 0].min(), allVertices[:, 1].min()],
                         [allVertices[:, 0].max(), allVertices[:, 1].max()]])
        logging.debug("Bbox for Face {}  : {}".format(self.index, bbox))
        return bbox

    def getAvgCentroid(self):
        """ Get the averaged centre point of the face """
        k = len(self.edgeList)
        xs = [x.origin.x for x in self.edgeList]
        ys = [x.origin.y for x in self.edgeList]
        norm_x = sum(xs) / k
        norm_y = sum(ys) / k
        return np.array([norm_x, norm_y])


    def getCentroid(self):
        return self.site.copy()

    def getCentroidFromBBox(self):
        """ Alternate Centroid, the centre point of the bbox for the face"""
        bbox = self.get_bbox()
        #max - min /2
        norm = bbox[1, :] + bbox[0, :]
        centre = norm * 0.5
        return centre


    def __getCentroid(self):
        """ An iterative construction of the centroid """
        vertices = [x.origin for x in self.edgeList if x.origin is not None]
        centroid = np.array([0.0, 0.0])
        signedArea = 0.0
        for i, v in enumerate(vertices):
            if i+1 < len(vertices):
                n_v = vertices[i+1]
            else:
                n_v = vertices[0]
            a = v.x*n_v.y - n_v.x*v.y
            signedArea += a
            centroid += [(v.x+n_v.x)*a, (v.y+n_v.y)*a]

        signedArea *= 0.5
        if signedArea != 0:
            centroid /= (6*signedArea)
        return centroid

    def getEdges(self):
        return self.edgeList.copy()

    def add_edge(self, edge):
        """ Add a constructed edge to the face """
        assert(isinstance(edge, HalfEdge))
        if edge.face is None:
            edge.face = self
        self.innerComponents.append(edge)
        self.edgeList.append(edge)

    def sort_edges(self):
        """ Order the edges anti-clockwise, by starting point """
        logging.debug("Sorting edges")
        centre = self.getCentroid()
        atanEdges = [(x.atan(), x) for x in self.edgeList]
        atanEdges = sorted(atanEdges)
        atanEdges.reverse()
        self.edgeList = [e for (a, e) in atanEdges]
        logging.debug("Sorted edges: {}".format([str(x.index) for x in self.edgeList]))
