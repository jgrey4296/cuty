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
from ..math import rotatePoint, calc_bbox_corner, within_bbox

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
                
        if self.dcel is not None and self not in self.dcel.faces:
            self.dcel.faces.add(self)

                
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
        bbox = np.array([np.min(allVertices, axis=0),
                         np.max(allVertices, axis=0)])
        logging.debug("Bbox for Face {}  : {}".format(self.index, bbox))
        return bbox

    def markForCleanup(self):
        self.markedForCleanup = True
    
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
        #return self.site.copy()
        raise Exception("deprecated: use getAvgCentroid or getCentroidFromBBox")

    def getCentroidFromBBox(self):
        """ Alternate Centroid, the centre point of the bbox for the face"""
        bbox = self.get_bbox()
        difference = bbox[1,:] - bbox[0,:]
        centre = bbox[0,:] + (difference * 0.5)
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
        self.coord_list = None
        if edge.face is None:
            edge.face = self
        self.edgeList.append(edge)

    def remove_edge(self, edge):
        """ Remove an edge from this face, if the edge has this face
        registered, remove that too """
        assert(isinstance(edge, HalfEdge))
        #todo: should the edge be connecting next to prev here?
        if not bool(self.edgeList):
            return
        # if edge in self.outerBoundaryEdges:
        #     self.outerBoundaryEdges.remove(edge)
        if edge in self.edgeList:
            self.edgeList.remove(edge)
        if edge.face is self:
            edge.face = None
        edge.markForCleanup()
        
    def sort_edges(self):
        """ Order the edges clockwise, by starting point, ie: graham scan """
        logging.debug("Sorting edges")
        centre = self.getAvgCentroid()
        #verify all edges are ccw
        assert(all([x.he_ccw(centre) for x in self.edgeList]))
        self.edgeList.sort()

    def has_edges(self):
        """ Check if its a null face or has actual edges """
        return bool(self.edgeList)


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
        assert(edge in self.edgeList)
        assert(0 <= ratio <= 1)
        assert(-90 <= angle <= 90)
        #split the edge
        newPoint, newEdge = edge.split_by_ratio(ratio)

        #get the bisecting vector
        asCoords = edge.toArray()
        bisector = cumath.get_bisector(asCoords[0], asCoords[1])
        #get the coords of an extended line
        extended_end = cumath.extend_line(newPoint.toArray(), bisector, 1000)
        el_coords = np.row_stack((newPoint.toArray(), extended_end))

        #intersect with coords of edges
        intersection = None
        oppEdge = None
        for he in self.edgeList:
            if he in [edge, newEdge]:
                continue
            he_coords = he.toArray()
            intersection = cumath.intersect(el_coords, he_coords)
            if intersection is not None:
                oppEdge = he
                break
        assert(intersection is not None)
        assert(oppEdge is not None)
        #split that line at the intersection
        newOppPoint, newOppEdge = oppEdge.split(intersection)

        #create the other face
        newFace = self.dcel.newFace()
        
        #create the subdividing edge:
        dividingEdge = self.dcel.newEdge(newPoint, newOppPoint,
                                         face=self,
                                         twinFace=newFace,
                                         edata=edge.data,
                                         vdata=edge.origin.data)
        dividingEdge.addPrev(edge, force=True)
        dividingEdge.addNext(newOppEdge, force=True)
        dividingEdge.twin.addPrev(oppEdge, force=True)
        dividingEdge.twin.addNext(newEdge, force=True)

        #divide the edges into newOppEdge -> edge,  newEdge -> oppEdge
        newFace_Edge_Group = []
        originalFace_Edge_Update = []

        current = newOppEdge
        while current != edge:
            assert(current.next is not None)
            originalFace_Edge_Update.append(current)
            current = current.next
        originalFace_Edge_Update.append(current)
        originalFace_Edge_Update.append(dividingEdge)
        
        current = newEdge
        while current != oppEdge:
            assert(current.next is not None)
            newFace_Edge_Group.append(current)
            current.face = newFace
            current = current.next
        newFace_Edge_Group.append(current)
        current.face = newFace
        newFace_Edge_Group.append(dividingEdge.twin)        
        
        #update the two faces edgelists
        self.edgeList = originalFace_Edge_Update
        newFace.edgeList = newFace_Edge_Group

        #return both
        return (self, newFace)

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
        newFace = dc.newFace()
        #then build the convex hull
        hull, discarded = Face.hull_from_vertices(all_verts)
        for s,e in zip(hull, islice(cycle(hull),1, None)):
            #create an edge
            newEdge = dc.newEdge(s,e, face=newFace)
        #link the edges
        dc.linkEdgesTogether(newFace.edgeList, loop=True)
        #return the face
        return (newFace, discarded)
        
    def translate_edge(self, e, transform, candidates=None, force=False):
        assert(e in self.edgeList)
        assert(isinstance(transform, np.ndarray))
        assert(transform.shape == (2,))
        
        raise Exception("Unimplemented")
    
        if not force and self.has_constraints(candidates):
            copied = self.copy()
            return (copied, EditE.NEW)
        else:
            return (self, EditE.MODIFIED)
        

    def scale(self, amnt=None, target=None, vert_weights=None, edge_weights=None,
              force=False, candidates=None):
        """ Scale an entire face by amnt,
        or scale by vertex/edge normal weights """
        if not force and self.has_constraints(candidates):
            facePrime, edit_type = self.copy().scale(amnt=amnt, target=target,
                                                     vert_weights=vert_weights,
                                                     edge_weights=edge_weights,
                                                     force=True)
            return (facePrime, EditE.NEW)

        raise Exception("Unimplemented")
        if target is None:
            target = self.getCentroidFromBBox()
        if amnt is None:
            amnt = np.ndarray([1,1])
        assert(isinstance(amnt, np.ndarray))
        assert(amnt.shape == (2,))
        if vert_weights is not None:
            assert(isinstance(vert_weights, np.ndarray))
        if edge_weights is not None:
            assert(isinstance(edge_weights, np.ndarray))
                    
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
            facePrime, edit_e = self.copy().rotate(rads, target=target, force=True)
            return (facePrime, EditE.NEW)
            
        if target is None:
            target = self.getCentroidFromBBox()
        assert(isinstance(target, np.ndarray))
        assert(target.shape == (2,))

        for l in self.edgeList:
            l.rotate(c=target, r=rads, candidates=candidates, force=True)
        return (self, EditE.MODIFIED)


    def constrain_to_circle(self, centre, radius, candidates=None, force=False):
        """ Constrain the vertices and edges of a face to be within a circle """
        if not force and self.has_constraints(candidates):
            logging.warning("Face: Constraining a copy")
            facePrime, edit_type = self.copy().constrain_to_circle(centre, radius, force=True)
            return (facePrime, EditE.NEW)

        logging.debug("Face: Constraining edges")
        #constrain each edge            
        edges = self.edgeList.copy()        
        for e in edges:
            logging.debug("HE: {}".format(e))
            eprime, edit_e = e.constrain_to_circle(centre, radius, force=True)
            logging.debug("Result: {}".format(eprime))
            assert(edit_e == EditE.MODIFIED)
            assert(eprime in self.edgeList)
            if eprime.markedForCleanup:
                self.edgeList.remove(eprime)

        return (self, EditE.MODIFIED)

    #todo: possibly add a shrink/expand to circle method

    def constrain_to_bbox(self, bbox, cadidates=None, force=False):
        if not force and self.has_constraints(candidates):
            facePrime, edit_type = self.copy().constrain_to_bbox(bbox, force=True)
            return (facePrime, EditE.NEW)
            
        edges = self.edgeList.copy()

        for edge in edges:
            if not edge.within(bbox):
                self.remove_edge(edge)
                continue

            eprime, edit_e = edge.constrain_to_bbox(bbox, candidates=candidates, force=True)

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
        for e in self.edgeList:
            all_verts.update(e.getVertices())
        return all_verts

    def get_all_coords(self):
        """ Get the sequence of coordinates for the edges """
        if self.coord_list is not None:
            return self.coord_list
        all_coords = np.array([x.toArray() for x in self.get_all_vertices()])
        self.coord_list = Face.hull_from_coords(all_coords)
        return self.coord_list


    #------------------------------
    # def verification
    #------------------------------
        
    def fixup(self, bbox=None):
        """ Verify and enforce correct designations of
        edge ordering, next/prev settings, and face settings """
        if not bool(self.edgeList):
            self.markForCleanup()
            return []
        if len(self.edgeList) < 2:
            return []

        self.sort_edges()
        
        inferred_edges = []
        edges = self.edgeList.copy()
        prev = edges[-1]
        for e in edges:
            #register face to edge
            e.face = self
            #enforce next and prev
            if e.prev is not prev:
                e.addPrev(prev, force=True)
            #if the vertices don't align, create an additional edge to connect
            if not prev.connections_align(e):
                logging.debug("connections don't align")
                newEdge = self.dcel.newEdge(e.prev.twin.origin,
                                            e.origin,
                                            twinFace=e.twin.face,
                                            edata=e.data,
                                            vdata=e.origin.data)
                newEdge.face = self
                newEdge.addPrev(e.prev, force=True)
                newEdge.addNext(e, force=True)
                #insert that new edge into the edgeList
                index = self.edgeList.index(e)
                self.edgeList.insert(index, newEdge)
                inferred_edges.append(newEdge)

            prev = e
                
        return inferred_edges

    def has_constraints(self, candidateSet=None):
        """ Tests whether the face's component edges and vertices are claimed by
        anything other than the face's own halfedges and their twins, and any passed in 
        candidates """
        if candidateSet is None:
            candidateSet = set()
        candidatesPlusSelf = candidateSet.union([self], self.edgeList, [x.twin for x in self.edgeList if x.twin is not None])
        return any([x.has_constraints(candidatesPlusSelf) for x in self.edgeList])
        
        
            
        # #if they intersect with different bounding walls,  they need a corner
        # intersect_1 = current_edge.intersects_edge(bbox)
        # intersect_2 = prior_edge.intersects_edge(bbox)
        # logging.debug("Intersect Values: {} {}".format(intersect_1, intersect_2))

        # if intersect_1 is None or intersect_2 is None:
        #     logging.debug("Non- side intersecting lines")

        # #Simple connection requirement, straight line between end points
        # if intersect_1 == intersect_2 or any([x is None for x in [intersect_1, intersect_2]]):
        #     logging.debug("Match, new simple edge between: {}={}".format(current_edge.index,
        #                                                                  prior_edge.index))
        #     #connect together with simple edge
        #     #twin face is not set because theres no face outside of bounds
        #     newEdge = self.newEdge(prior_edge.twin.origin,
        #                            current_edge.origin,
        #                            face=f,
        #                            prev=prior_edge)
        #     newEdge.setPrev(prior_edge)
        #     current_edge.setPrev(newEdge)

        # else:
        #     logging.debug("Creating a corner edge connection between: {}={}".format(current_edge.index, prior_edge.index))
        #     #Connect the edges via an intermediate, corner vertex
        #     newVert = self.create_corner_vertex(intersect_1, intersect_2, bbox)
        #     logging.debug("Corner Vertex: {}".format(newVert))
        #     newEdge_1 = self.newEdge(prior_edge.twin.origin, newVert, face=f, prev=prior_edge)
        #     newEdge_2 = self.newEdge(newVert, current_edge.origin, face=f, prev=newEdge_1)
            
        #     current_edge.addPrev(newEdge_2)
        #     newEdge_2.addPrev(newEdge_1)
        #     newEdge_1.addPrev(prior_edge)
