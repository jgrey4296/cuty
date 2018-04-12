""" The highest level data structure in a dcel, apart from the dcel itself """
import logging as root_logger
import numpy as np
from numbers import Number
from itertools import cycle, islice
from functools import partial, cmp_to_key
from math import radians
from scipy.spatial import ConvexHull
import IPython

from .constants import EditE
from .Vertex import Vertex
from .HalfEdge import HalfEdge
from ..constants import TWOPI
from .. import math as cumath
from ..math import rotatePoint

logging = root_logger.getLogger(__name__)

class Face(object):
    """ A Face with a start point for its outer component list,
    and all of its inner components """

    nextIndex = 0

    def __init__(self, site=None, index=None, data=None, dcel=None):
        if site is None:
            site = np.array([0,0])
        assert(isinstance(site, np.ndarray))
        #Site is the voronoi point that the face is built around
        self.site = site
        #Primary list of ccw edges for this face
        self.edgeList = []
        self.coord_list = None
        #mark face for cleanup:
        self.markedForCleanup = False
        #Additional Data:
        self.data = {}
        if data is not None:
            self.data.update(data)
        self.dcel = dcel

        #free vertices to build a convex hull from:
        self.free_vertices = set()
        
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
                
    def copy(self):
        with self.dcel:
            #copy the halfedges
            es = [x.copy() for x in self.edgeList]
            #create a new face
            f = self.dcel.newFace(edges=es)
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
        npPairs = [(x.toArray(),x) for x in verts]
        hull = ConvexHull([x[0] for x in npPairs])
        hullVerts = [npPairs[x][1] for x in hull.vertices]
        discardVerts = set(verts).difference(hullVerts)
        assert(len(discardVerts.intersection(hullVerts)) == 0)
        assert(len(discardVerts) + len(hullVerts) == len(verts))
        return (hullVerts, list(discardVerts))

    @staticmethod
    def hull_from_coords(coords):
        """ Given a set of coordinates, return the hull they would form 
        DOESN NOT RETURN DISCARDED, as the coords are not vertices yet
        """
        assert(isinstance(coords, np.ndarray))
        assert(coords.shape[1] == 2)
        hull = ConvexHull(coords)
        hullCoords = np.array([coords[x] for x in hull.vertices])
        return hullCoords

    #------------------------------
    # def Human Readable Representations
    #------------------------------
    
                
    def __str__(self):
        return "Face: {}".format(self.getCentroid())

    def __repr__(self):
        edgeList = len(self.edgeList)
        return "(Face: {}, edgeList: {})".format(self.index, edgeList)        

    #------------------------------
    # def Exporting
    #------------------------------
    
    
    def _export(self):
        """ Export identifiers rather than objects, to allow reconstruction """
        logging.debug("Exporting face: {}".format(self.index))
        return {
            'i' : self.index,
            'edges' : [x.index for x in self.edgeList if x is not None],
            'sitex' : self.site[0],
            'sitey' : self.site[1],
            'data'  : self.data
        }


            
    def get_bbox(self):
        """ Get a rough bbox of the face """
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

    
    #------------------------------
    # def centroids
    #------------------------------
        
    def getAvgCentroid(self):
        """ Get the averaged centre point of the face from the vertices of the edges """
        k = len(self.edgeList)
        xs = [x.origin.loc[0] for x in self.edgeList]
        ys = [x.origin.loc[1] for x in self.edgeList]
        norm_x = sum(xs) / k
        norm_y = sum(ys) / k
        self.site = np.array([norm_x, norm_y])
        return self.site


    def getCentroid(self):
        """ Get the user defined 'centre' of the face """
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
        raise Exception("Deprecated")
        # vertices = [x.origin for x in self.edgeList if x.origin is not None]
        # centroid = np.array([0.0, 0.0])
        # signedArea = 0.0
        # for i, v in enumerate(vertices):
        #     if i+1 < len(vertices):
        #         n_v = vertices[i+1]
        #     else:
        #         n_v = vertices[0]
        #     a = v.loc[0]*n_v.loc[1] - n_v.loc[0]*v.loc[1]
        #     signedArea += a
        #     centroid += [(v.loc[0]+n_v.loc[0])*a, (v.loc[1]+n_v.loc[1])*a]

        # signedArea *= 0.5
        # if signedArea != 0:
        #     centroid /= (6*signedArea)
        # return centroid

    #------------------------------
    # def edge access
    #------------------------------
            
    def getEdges(self):
        """ Return a copy of the edgelist for this face """
        return self.edgeList.copy()

    def add_edges(self, edges):
        assert(isinstance(edges, list))
        for x in edges:
            self.add_edge(x)
    
    def add_edge(self, edge):
        """ Add a constructed edge to the face """
        assert(isinstance(edge, HalfEdge))
        if edge.face is None:
            edge.face = self
        self.outerBoundaryEdges.append(edge)
        self.edgeList.append(edge)

    def remove_edge(self, edge):
        """ Remove an edge from this face, if the edge has this face
        registered, remove that too """
        assert(isinstance(edge, HalfEdge))
        #todo: should the edge be connecting next to prev here?
        if not bool(self.outerBoundaryEdges) and not bool(self.edgeList):
            return
        if edge in self.outerBoundaryEdges:
            self.outerBoundaryEdges.remove(edge)
        if edge in self.edgeList:
            self.edgeList.remove(edge)
        if edge.face is self:
            edge.face = None
        
    def sort_edges(self):
        """ Order the edges clockwise, by starting point, ie: graham scan """
        logging.debug("Sorting edges")
        centre = self.getAvgCentroid()
        #verify all edges are ccw
        assert(all([x.he_ccw(centre) for x in self.edgeList]))
        
        self.edgeList.sort()
        #ensure all edges line up
        paired = zip(self.edgeList, islice(cycle(self.edgeList), 1, None))
        try:
            for a,b in paired:
                assert(a.twin.origin == b.origin)
        except AssertionError as e:
            IPython.embed(simple_prompt=True)

    def has_edges(self):
        """ Check if its a null face or has actual edges """
        innerEdges = bool(self.outerBoundaryEdges)
        outerEdges = bool(self.edgeList)
        return bool(innerEdges and outerEdges)

    def markForCleanup(self):
        self.markedForCleanup = True

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
        if ratio is None:
            ratio = 0.5
        assert(isinstance(edge, HalfEdge))
        assert(edge in self.edgeList)
        assert(0 <= ratio <= 1)
        assert(-90 <= angle <= 90)
        #split the edge
        newPoint, newEdge = edge.split_by_ratio(ratio)
                
        #extend a line from the new vertex, long enough to intersect the other side
        bisecting_edge = newEdge.extend(rotate=radians(angle), d=10000)
        
        #intersect with all lines of the face until the intersection is found
        intersections = [a for a in [bisecting_edge.intersect(x) for x in self.edgeList if x is not newEdge and x is not edge] if a is not None]
        assert(len(intersections) == 1)

        #split that line at the intersection
        
        #add the line in

        #sort the vertices

        #pick the new vertex, follow it around until hitting its twin
        #get the disjoint set of those vertices, those are the two faces
        

        #clean up the massive line?
        
        raise Exception("Unimplemented")

    def add_vertex(self, vert):
        """ Add a vertex, then recalculate the convex hull """
        assert(isinstance(vert, Vertex))
        self.free_vertices.add(vert)

    def get_all_vertices(self):
        """ Get all vertices of the face. both free and in halfedges """
        all_verts = set()
        all_verts.update(self.free_vertices)
        for e in self.edgeList:
            all_verts.update(e.getVertices())
        return all_verts

        
    def merge_faces(self, *args):
        """ Calculate a convex hull from all passed in faces,
        creating a new face """
        all_verts = self.get_all_vertices()
        for f in args:
            all_verts.update(f.get_all_vertices())
        f = self.dcel.newFace()
        f.free_vertices.update(all_verts)
        #then build the convex hull
        hull, discarded = Face.hull_from_vertices(f.free_vertices)
        #create the edges
                
        #assign to face

        #return the face
        raise Exception("unimplemented")
        
    def translate_edge(self, e, transform):
        assert(e in self.edgeList)
        assert(isinstance(transform, np.ndarray))
        assert(transform.shape == (2,))
        
        raise Exception("Unimplemented")

    def scale(self, amnt=None, vert_weights=None, edge_weights=None):
        """ Scale an entire face by amnt,
        or scale by vertex/edge normal weights """
        raise Exception("unimplemented")

    def cut_out(self):
        """ Cut the Face out from its verts and halfedges that comprise it,
        creating new verts and edges, so the face can be moved and scaled
        without breaking the already existing structure """
        return self.copy()

    def rotate(self, rads):
        """ copy and rotate the entire face """
        assert(-TWOPI <= rads <= TWOPI)
        raise Exception("Unimplemented")
    
            
    #------------------------------
    # def Vertex access
    #------------------------------
        
    def add_vertex(self, vert):
    #------------------------------
    # def verification
    #------------------------------
        
    def fixup(self, bbox=None):
