""" The Top Level DCEL DataStructure. """
from collections import namedtuple
from math import atan2, degrees
from numbers import Number
from os.path import isfile
from random import random
from itertools import cycle, islice
import IPython
import math
import numpy as np
import pickle
import pyqtree
import sys

from ..math import get_distance
from .Face import Face
from .HalfEdge import HalfEdge
from .Vertex import Vertex
from .Line import Line
from .constants import EdgeE
from .line_intersector import LineIntersector

import logging as root_logger
logging = root_logger.getLogger(__name__)


#for importing data into the dcel:
DataPair = namedtuple('DataPair', 'key obj data')

class DCEL(object):
    """ The total DCEL data structure,  stores vertices,  edges,  and faces,
    Based on de Berg's Computational Geometry
    """
    Vertex = Vertex
    HalfEdge = HalfEdge
    Face = Face
    
    
    def __init__(self, bbox=None):
        if bbox is None:
            bbox = np.array([-200, -200, 200, 200])
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)
        self.vertices = set([])
        self.faces = set()
        self.halfEdges = set([])
        self.bbox = bbox
        #todo: make this a stack of quadtrees
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
        self.quad_tree_stack = []
        self.frontier = set()
        self.should_merge_stacks = True

        self.data = {}
        
    def reset_frontier(self):
        self.frontier = set()

    def copy(self):
        newDCEL = DCEL(self.bbox)
        newDCEL.import_data(self.export_data())        
        return newDCEL

    def __str__(self):
        """ Create a text description of the DCEL """
        verticesDescription = "Vertices: num: {}".format(len(self.vertices))
        edgesDescription = "HalfEdges: num: {}".format(len(self.halfEdges))
        facesDescription = "Faces: num: {}".format(len(self.faces))

        allVertices = [x.getVertices() for x in self.halfEdges]
        flattenedVertices = [x for (x, y) in allVertices for x in (x, y)]
        setOfVertices = set(flattenedVertices)
        vertexSet = "Vertex Set: num: {}/{}/{}".format(len(setOfVertices), len(flattenedVertices),len(self.vertices))

        infiniteEdges = [x for x in self.halfEdges if x.isInfinite()]
        infiniteEdgeDescription = "Infinite Edges: num: {}".format(len(infiniteEdges))

        completeEdges = set()
        for x in self.halfEdges:
            if not x in completeEdges and x.twin not in completeEdges:
                completeEdges.add(x)

        completeEdgeDescription = "Complete Edges: num: {}".format(len(completeEdges))

        edgelessVertices = [x for x in self.vertices if x.isEdgeless()]
        edgelessVerticesDescription = "Edgeless vertices: num: {}".format(len(edgelessVertices))

        edgeCountForFaces = [str(len(f.edgeList)) for f in self.faces]
        edgeCountForFacesDescription = "Edge Counts for Faces: {}".format("-".join(edgeCountForFaces))

        purgeVerts = "Verts to Purge: {}".format(len([x for x in self.vertices if x.markedForCleanup]))
        purgeEdges = "Hedges to Purge: {}".format(len([x for x in self.halfEdges if x.markedForCleanup]))
        purgeFaces = "Faces to Purge: {}".format(len([x for x in self.faces if x.markedForCleanup]))
        
        return "\n".join(["---- DCEL Description: ",
                          verticesDescription,
                          edgesDescription,
                          facesDescription,
                          vertexSet,
                          infiniteEdgeDescription,
                          completeEdgeDescription,
                          edgelessVerticesDescription,
                          edgeCountForFacesDescription,
                          "-- Purging:",
                          purgeVerts,
                          purgeEdges,
                          purgeFaces,
                          "----\n"])

    
    #------------------------------
    # def IO
    #------------------------------
        
    def export_data(self):
        """ Export a simple format to define vertices,  halfedges,  faces,
        uses identifiers instead of objects,  allows reconstruction """
        data = {
            'vertices' : [x._export() for x in self.vertices],
            'halfEdges' : [x._export() for x in self.halfEdges],
            'faces' : [x._export() for x in self.faces],
            'bbox' : self.bbox
        }
        return data

    def import_data(self, data):
        """ Import the data format of identifier links to actual links,  
        of export output from a dcel """
        assert(all([x in data for x in ['vertices', 'halfEdges', 'faces', 'bbox']]))
        self.bbox = data['bbox']
        #dictionarys:
        local_vertices = {}
        local_edges = {}
        local_faces = {}

        #construct vertices by index
        logging.info("Re-Creating Vertices: {}".format(len(data['vertices'])))
        for vData in data['vertices']:
            newVert = Vertex(np.array([vData['x'], vData['y']]), index=vData['i'], data=vData['data'], dcel=self, active=vData['active'])
            logging.debug("Re-created vertex: {}".format(newVert.index))
            local_vertices[newVert.index] = DataPair(newVert.index, newVert, vData)
        #edges by index
        logging.info("Re-Creating Edges: {}".format(len(data['halfEdges'])))
        for eData in data['halfEdges']:
            newEdge = HalfEdge(index=eData['i'], data=eData['data'], dcel=self)
            logging.debug("Re-created Edge: {}".format(newEdge.index))
            local_edges[newEdge.index] = DataPair(newEdge.index, newEdge, eData)
        #faces by index
        logging.info("Re-Creating Faces: {}".format(len(data['faces'])))
        for fData in data['faces']:
            newFace = Face(site=np.array([fData['sitex'], fData['sitey']]), index=fData['i'], data=fData['data'], dcel=self)
            logging.debug("Re-created face: {}".format(newFace.index))
            local_faces[newFace.index] = DataPair(newFace.index, newFace, fData)
        #Everything now exists,  so:
        #connect the objects up,  instead of just using ids
        try:
            #connect vertices to their edges
            for vertex in local_vertices.values():
                vertex.obj.halfEdges.update([local_edges[x].obj for x in vertex.data['halfEdges']])
        except Exception as e:
            logging.info("Error for vertex")
            IPython.embed(simple_prompt=True)
            
        try:
            #connect edges to their vertices, and neighbours, and face
            for edge in local_edges.values():
                if edge.data['origin'] is not None:
                    edge.obj.origin = local_vertices[edge.data['origin']].obj
                if edge.data['twin'] is not None:
                    edge.obj.twin = local_edges[edge.data['twin']].obj
                if edge.data['next'] is not None:
                    edge.obj.next = local_edges[edge.data['next']].obj
                if edge.data['prev'] is not None:
                    edge.obj.prev = local_edges[edge.data['prev']].obj 
                if edge.data['face'] is not None:
                    edge.obj.face = local_faces[edge.data['face']].obj
        except Exception as e:
            logging.info("Error for edge")
            IPython.embed(simple_prompt=True)
        try:
            #connect faces to their edges
            for face in local_faces.values():
                face.obj.edgeList = [local_edges[x].obj for x in face.data['edges']]
        except Exception as e:
            logging.info("Error for face")
            IPython.embed(simple_prompt=True)

        #Now transfer into the actual object:
        self.vertices = [x.obj for x in local_vertices.values()]
        self.halfEdges = [x.obj for x in local_edges.values()]
        self.faces = [x.obj for x in local_faces.values()]
        self.calculate_quad_tree()

    @staticmethod
    def loadfile(filename):
        """ Create a DCEL from a saved pickle """
        if not isfile("{}.dcel".format(filename)):
            raise Exception("Non-existing filename to load into dcel")
        with open("{}.dcel".format(filename), 'rb') as f:
            dcel_data = pickle.load(f)
        the_dcel = DCEL()
        the_dcel.import_data(dcel_data)
        return the_dcel

    def savefile(self, filename):
        """ Save dcel data to a pickle """
        theData = self.export_data()
        with open("{}.dcel".format(filename), 'wb') as f:
            pickle.dump(theData, f)

        

    #------------------------------
    # def quadtree
    #------------------------------
    
    def clear_quad_tree(self):
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)

    def calculate_quad_tree(self, subverts=None):
        """ Recalculate the quad tree with all vertices, or a subselection of vertices """
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
        verts = self.vertices
        if subverts is not None:
            assert(all([isinstance(x, Vertex) for x in subverts]))
            verts = subverts
        for vertex in verts:
            self.vertex_quad_tree.insert(item=vertex, bbox=vertex.bbox())
        
    def push_quad_tree(self):
        self.quad_tree_stack.append(self.vertex_quad_tree)
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)

    def pop_quad_tree(self):
        assert(len(self.quad_tree_stack) > 0)
        sub_layer = self.quad_tree_stack.pop()
        if self.should_merge_stacks:
            for x in self.vertex_quad_tree.intersect(self.bbox):
                if x not in sub_layer.intersect(x.bbox()):                
                    sub_layer.insert(item=x, bbox=x.bbox())
        self.vertex_quad_tree = sub_layer

    def add_to_current_quad_tree(self, verts):
        """ Add the passed in vertices into the current quad tree """
        assert(all([isinstance(x, Vertex) for x in verts]))
        assert(len(self.quad_tree_stack) > 0)
        for x in verts:
            self.vertex_quad_tree.insert(item=x, bbox=x.bbox())
        
        
    def __enter__(self):
        """ Makes the Dcel a reusable context manager, that pushes
        and pops vertex quad trees for collision detection """
        self.push_quad_tree()

    def __exit__(self, type, value, traceback):
        self.pop_quad_tree()
        

    #------------------------------
    # def PURGING
    #------------------------------
            
    def purge_edge(self, target):
        assert(isinstance(target, HalfEdge))
        logging.debug("Purging Edge: {}".format(target.index))
        target_update = set()

        target.connectNextToPrev()        
        vert = target.origin
        target.origin = None
        if vert is not None:
            vert.unregisterHalfEdge(target)
            if vert.isEdgeless():
                vert.markForCleanup()
                target_update.add(vert)
        
        if target.face is not None:
            face = target.face
            face.remove_edge(target)
            if not face.has_edges():
                face.markForCleanup()
                target_update.add(face)

        if target.twin is not None:
            target.twin.markForCleanup()
            target_update.add(target.twin)
            target.twin.twin = None
            target.twin = None
            

        self.halfEdges.remove(target)
        
        return target_update
        

    def purge_vertex(self, target):
        assert(isinstance(target, Vertex))
        logging.debug("Purging Vertex: {}".format(target.index))
        target_update = set()
        
        halfEdges = target.halfEdges.copy()
        for edge in halfEdges:
            assert(edge.origin == target)
            edge.origin = None
            target.unregisterHalfEdge(edge)
            edge.markForCleanup()
            target_update.add(edge)
            
        self.vertices.remove(target)
        
        return target_update


    def purge_face(self, target):
        assert(isinstance(target, Face))
        logging.debug("Purging Face: {}".format(target.index))
        target_update = set()
        edges = target.getEdges()
        for edge in edges:
            target.remove_edge(edge)
            edge.markForCleanup()
            target_update.add(edge)
        self.faces.remove(target)
        return target_update

    
    def purge(self, targets=None):
        """ Run all purge methods in correct order """
        if targets is None:
            #populate the targets:
            targets = set([])
            targets = targets.union([x for x in self.vertices if x.markedForCleanup])
            targets = targets.union([x for x in self.halfEdges if x.markedForCleanup or x.isInfinite()])
            targets = targets.union([x for x in self.faces if x.markedForCleanup or not x.has_edges()])
            
        else:
            targets = set(targets)

        purged = set()
        while bool(targets):
            current = targets.pop()
            if current in purged:
                continue
            if type(current) is Vertex:
                targets = targets.union(self.purge_vertex(current))
            elif type(current) is HalfEdge:
                targets = targets.union(self.purge_edge(current))
            elif type(current) is Face:
                targets = targets.union(self.purge_face(current))
            purged.add(current)

        self.calculate_quad_tree()

    #------------------------------
    # def Vertex, Edge, HalfEdge Creation
    #------------------------------
    
    def newVertex(self, loc, data=None, force=False):
        """ Create a new vertex,  or reuse an existing vertex.
        to force a new vertex instead of reusing, set force to True
        """
        assert(isinstance(loc, np.ndarray))
        newVert = None
        matchingVertices = self.vertex_quad_tree.intersect(Vertex.free_bbox(loc))
        logging.debug("Quad Tree Size: {}".format(self.vertex_quad_tree.countmembers()))
        logging.debug("Query result: {}".format(matchingVertices))
        if bool(matchingVertices) and not force:
            #a good enough vertex exists
            newVert = matchingVertices.pop()
            if data is not None:
                newVert.data.update(data)
            logging.debug("Found a matching vertex: {}".format(newVert))
        else:
            #no matching vertex,  add this new one
            newVert = Vertex(loc, data=data, dcel=self)
            logging.debug("No matching vertex,  storing: {}, {}".format(newVert, newVert.bbox()))
            self.vertices.add(newVert)
            self.vertex_quad_tree.insert(item=newVert, bbox=newVert.bbox())
        assert(newVert is not None)
        return newVert

    def newEdge(self, originVertex, twinVertex, face=None, twinFace=None,
                prev=None, twinPrev=None, next=None, twinNext=None,
                edata=None, vdata=None):
        """ Create a new half edge pair,  after specifying its start and end.
            Can set the faces,  and previous edges of the new edge pair.
            Returns the outer edge
        """
        #todo: check for an already existing edge
        assert(originVertex is None or isinstance(originVertex, Vertex))
        assert(twinVertex is None or isinstance(twinVertex, Vertex))
        e1 = HalfEdge(originVertex, None, dcel=self)
        e2 = HalfEdge(twinVertex, e1, dcel=self)
        e1.twin = e2
        #Connect with passed in details
        if face is not None:
            assert(isinstance(face, Face))
            face.add_edge(e1)
        if twinFace is not None:
            assert(isinstance(twinFace, Face))
            twinFace.add_edge(e2)
        if prev is not None:
            assert(isinstance(prev, HalfEdge))
            e1.addPrev(prev)
        if twinPrev is not None:
            assert(isinstance(twinPrev, HalfEdge))
            e2.addPrev(twinPrev)
        if next is not None:
            assert(isinstance(next, HalfEdge))
            e1.addNext(next)
        if twinNext is not None:
            assert(isinstance(twinNext, HalfEdge))
            e2.addNext(next)
        if edata is not None:
            e1.data.update(edata)
            e2.data.update(edata)
        if vdata is not None:
            e1.origin.data.update(vdata)
            e2.origin.data.update(vdata)            
        self.halfEdges.update([e1, e2])
        logging.debug("Created Edge Pair: {}".format(e1.index))
        logging.debug("Created Edge Pair: {}".format(e2.index))
        return e1

    def newFace(self, site=None, edges=None, verts=None, coords=None):
        """ Creates a new face to link edges """
        usedList = [edges is not None, verts is not None, coords is not None]
        assert(len([x for x in usedList if x]) < 2)
        if site is None:
            site = np.array([0,0])
        assert(isinstance(site, np.ndarray))
        newFace = Face(site=site, dcel=self)
        self.faces.add(newFace)
        #populate the face if applicable:
        coordHullGen = False
        if coords is not None:
            assert(isinstance(coords, np.ndarray))
            assert(coords.shape[1] == 2)
            hullCoords = Face.hull_from_coords(coords)
            verts = [self.newVertex(x) for x in hullCoords]
            coordHullGen = True
            
        if verts is not None:
            if not coordHullGen:
                verts, discarded = Face.hull_from_verts(verts)
            edges = []
            for s,e in zip(verts, islice(cycle(verts), 1, None)):
                edges.append(self.newEdge(s,e))
                    
        if edges is not None:
            newFace.add_edges(edges)
            self.linkEdgesTogether(edges, loop=True)
            
        return newFace


    def createEdge(self, origin, end, edata=None, vdata=None, subdivs=0):
        """ Utility to create two vertices, and put them into a pair of halfedges,
        returning a halfedge
        subdivs specifies number of inner segments to the line"""
        assert(isinstance(origin, np.ndarray))
        assert(isinstance(end, np.ndarray))
        v1 = self.newVertex(origin)
        v2 = self.newVertex(end)
        e = self.newEdge(v1, v2)
        if vdata is not None:
            assert(isinstance(vdata, dict))
            v1.data.update(vdata)
            v2.data.update(vdata)
        if edata is not None:
            assert(isinstance(edata, dict))
            e.data.update(edata)
        return e

    def createPath(self, vs, close=False, edata=None, vdata=None):
        """ Create multiple halfEdges, that connect to one another.
        With optional path closing """
        assert(isinstance(vs, np.ndarray))
        vertices = vs
        pathVerts = zip(vertices, islice(cycle(vertices), 1, None))
        path = []
        for a,b in pathVerts:
            if not close and (a == vertices[-1]).all():
                continue
            path.append(self.createEdge(a,b, edata=edata, vdata=vdata))
        return path        
    
    #------------------------------
    # def Coherence Utils
    #------------------------------
    
    def linkEdgesTogether(self, edges, loop=False):
        """ Given a list of half edges, set their prev and next fields in order """
        #TODO: check this
        assert(all([isinstance(x, HalfEdge) for x in edges]))
        if loop:
            other = islice(cycle(edges), 1, None)
        else:
            other = edges[1:]
        for (a,b) in zip(edges, other):
            a.addNext(b)

    def force_all_edge_lengths(self, l):
        """ Force all edges to be of length <= l. If over, split into multiple lines
        of length l. """
        assert(l > 0)
        processed = set()
        allEdges = list(self.halfEdges)
        while len(allEdges) > 0:
            current = allEdges.pop(0)
            assert(current.index not in processed)
            if current.getLength_sq() > l:
                newPoint, newEdge = current.split_by_ratio(r=0.5)
                if newEdge.getLength_sq() > l:
                    allEdges.append(newEdge)
                else:
                    processed.add(newEdge.index)
                    
            if current.getLength_sq() > l:
                allEdges.append(current)
            else:
                processed.add(current.index)
    

        
    def constrain_to_circle(self, centre, radius, candidates=None, force=False):
        """ Limit all faces, edges, and vertices to be within a circle,
        adding boundary verts and edges as necessary """
        assert(isinstance(centre, np.ndarray))
        assert(isinstance(radius, float))
        assert(centre.shape == (2,))

        #constrain faces
        faces = self.faces.copy()
        for f in faces:
            logging.debug("Constraining Face: {}".format(f))
            f.constrain_to_circle(centre, radius, candidates=candidates, force=force)
        
        #constrain free edges
        hedges = self.halfEdges.copy()
        for he in hedges:
            logging.debug("Constraining Hedge: {}".format(he))
            if he.face is not None or he.markedForCleanup:
                continue
            he.constrain_to_circle(centre, radius, candidates=candidates, force=force)
        
        #constrain free vertices
        vertices = self.vertices.copy()
        for v in vertices:
            logging.debug("Constraining Vertex: {}".format(v))
            if not v.isEdgeless():
                continue
            if not v.within_circle(centre, radius):
                v.markForCleanup()


    def constrain_to_bbox(self, bbox, candidates=None, force=False):
        assert(isinstance(bbox, np.ndarray))
        assert(bbox.shape == (4,))

        faces = self.faces.copy()
        for f in faces:
            logging.debug("Constraining Face: {}".format(f))
            f.constrain_to_bbox(bbox, candidates=candidates, force=force)
        
        #constrain free edges
        hedges = self.halfEdges.copy()
        for he in hedges:
            logging.debug("Constraining Hedge: {}".format(he))
            if he.face is not None or he.markedForCleanup:
                continue
            he.constrain_to_bbox(bbox, candidates=candidates, force=force)
        
        #constrain free vertices
        vertices = self.vertices.copy()
        for v in vertices:
            logging.debug("Constraining Vertex: {}".format(v))
            if not v.isEdgeless():
                continue
            if not v.within(bbox):
                v.markForCleanup()



        
    #------------------------------
    # def Utilities
    #------------------------------
    
    def orderVertices(self, focus, vertices):
        """ Given a focus point and a list of vertices,  sort them
            by the counter-clockwise angle position they take relative """
        assert(all([isinstance(x, Vertex) for x in vertices]))
        assert(isinstance(focus, np.ndarray))
        relativePositions = [v.loc - focus for v in vertices]
        zipped = zip(relativePositions, vertices)
        angled = [((degrees(atan2(loc[1], loc[0])) + 360) % 360, vert) for loc,vert in zipped]
        sortedAngled = sorted(angled)
        # rads = (np.arctan2(verts[:,1], verts[:,0]) + TWOPI) % TWOPI
        # ordered = sorted(zip(rads, opp_hedges))
        return [vert for loc,vert in sortedAngled]


    def create_corner_vertex(self, e1, e2, bbox):
        """ Given two intersections (0-3) describing the corner,  
        create the vertex at the boundary of the bbox """
        assert(isinstance(e1, int))
        assert(isinstance(e2, int))
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)

        if e1 == e2:
            raise Exception("Corner Edge Creation Error: edges are equal")
        if e1 % 2 == 0: #create the x vector
            v1 = np.array([bbox[e1], 0])
        else:
            v1 = np.array([0, bbox[e1]])
        if e2 % 2 == 0: #create the y vector
            v2 = np.array([bbox[e2], 0])
        else:
            v2 = np.array([0, bbox[e2]])
        #add together to get corner
        v3 = v1 + v2
        return self.newVertex(*v3)

    def verify_all(self):
        reg_verts = set([x.index for x in self.vertices])
        reg_hedges = set([x.index for x in self.halfEdges])
        reg_faces = set([x.index for x in self.faces])

        vert_hedges = set()
        for v in self.vertices:
            vert_hedges.update([x.index for x in v.halfEdges])

        hedge_verts = set()
        hedge_nexts = set()
        hedge_prevs = set()
        hedge_faces = set()
        for h in self.halfEdges:
            hedge_verts.add(h.origin.index) 
            if h.next is not None:
                hedge_nexts.add(h.next.index)
            if h.prev is not None:
                hedge_prevs.add(h.prev.index)
            if h.face is not None:
                hedge_faces.add(h.face.index)

        face_edges = set()
        for f in self.faces:
            face_edges.update([x.index for x in f.edgeList])

        #differences:
        vert_hedge_diff = vert_hedges.difference(reg_hedges)
        hedge_vert_diff = hedge_verts.difference(reg_verts)
        hedge_nexts_diff = hedge_nexts.difference(reg_hedges)
        hedge_prevs_diff = hedge_prevs.difference(reg_hedges)
        hedge_faces_diff = hedge_faces.difference(reg_faces)
        face_edges_diff = face_edges.difference(reg_hedges)

        try:
            assert(all([len(x) == 0 for x in [vert_hedge_diff,
                                              hedge_vert_diff,
                                              hedge_nexts_diff,
                                              hedge_prevs_diff,
                                              hedge_faces_diff,
                                              face_edges_diff]]))
        except AssertionError as e:
            IPython.embed(simple_prompt=True)
        

    
    #------------------------------
    # def deprecated
    #------------------------------

    def fixup_halfedges(self):
        """ Fix all halfedges to ensure they are counter-clockwise ordered """
        raise Exception("deprecated: fixup faces instead") 

    def intersect_halfedges(self):
        """ run a sweep line over the dcel, getting back halfedge intersections """
        raise Exception("unimplemented")

    def calculate_edge_connections(self, current_edge, prior_edge, bbox, f):
        """ Connect two edges within a bbox, with a corner if necessary  """
        raise Exception("Deprecated: use DCEL.integrate_bbox_edges")

    def verify_edges(self):
        raise Exception("deprecated, use HalfEdge.fixup")

    def verify_faces_and_edges(self):
        raise Exception("Deprecated, use face and halfedge verify methods")

    def purge_faces(self):
        """ Same as purging halfedges or vertices,  but for faces """
        raise Exception("deprecated, use dcel.purge")
            
    def purge_vertices(self):
        """ Remove all vertices that aren't connected to at least one edge"""
        raise Exception("deprecated, use dcel.purge")

    def purge_edges(self):
        """ Remove all edges that have been marked for cleanup """
        raise Exception("Deprecated: use dcel.purge")

    def integrate_bbox_edges(self, current_edge, prior_edge, bbox, f):
        raise Exception("Deprecated: Use dcel.constrain_to_bbox")

    def constrain_half_edges(self, bbox):
        raise Exception("Deprecated: use dcel.constrain_to_bbox")
