""" The Top Level DCEL DataStructure. """
from collections import namedtuple
from math import atan2
from numbers import Number
from os.path import isfile
from random import random
import IPython
import logging
import math
import numpy as np
import pickle
import pyqtree
import sys
from itertools import cycle, islice

from cairo_utils.math import get_distance
from .Face import Face
from .HalfEdge import HalfEdge
from .Vertex import Vertex
from .Line import Line
from .constants import EdgeE

EPSILON = sys.float_info.epsilon
CENTRE = np.array([[0.5, 0.5]])
PI = math.pi
TWOPI = 2 * PI
HALFPI = PI * 0.5
QPI = PI * 0.5

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
        self.vertices = []
        self.faces = []
        self.halfEdges = []
        self.bbox = bbox
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
        self.frontier = set()

    def reset_frontier(self):
        self.frontier = set()


    def __copy__(self):
        newDCEL = DCEL(self.bbox)
        newDCEL.import_data(self.export_data())        
        return newDCEL
        
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
            newVert = Vertex(vData['x'], vData['y'], index=vData['i'], data=vData['data'], dcel=self)
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
            newFace = Face(fData['sitex'], fData['sitey'], index=fData['i'], data=fData['data'], dcel=self)
            logging.debug("Re-created face: {}".format(newFace.index))
            local_faces[newFace.index] = DataPair(newFace.index, newFace, fData)
        #Everything now exists,  so:
        #connect the objects up,  instead of just using ids
        try:
            #connect vertices to their edges
            for vertex in local_vertices.values():
                vertex.obj.halfEdges = [local_edges[x].obj for x in vertex.data['halfEdges']]
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
                if edge.data['nexts'] is not None:
                    edge.obj.nexts = [local_edges[x].obj for x in edge.data['nexts']]
                if edge.data['prevs'] is not None:
                    edge.obj.prevs = [local_edges[x].obj for x in edge.data['prevs']]
                if edge.data['face'] is not None:
                    edge.obj.face = local_faces[edge.data['face']].obj
        except Exception as e:
            logging.info("Error for edge")
            IPython.embed(simple_prompt=True)
        try:
            #connect faces to their edges
            for face in local_faces.values():
                face.obj.edgeList = [local_edges[x].obj for x in face.data['edges']]
                face.obj.outerBoundaryEdges = [x.twin for x in face.obj.edgeList if x.twin.face is not None]
        except Exception as e:
            logging.info("Error for face")
            IPython.embed(simple_prompt=True)

        #Now transfer into the actual object:
        self.vertices = [x.obj for x in local_vertices.values()]
        self.halfEdges = [x.obj for x in local_edges.values()]
        self.faces = [x.obj for x in local_faces.values()]
        self.calculate_quad_tree()


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

    def __str__(self):
        """ Create a text description of the DCEL """
        verticesDescription = "Vertices: num: {}".format(len(self.vertices))
        edgesDescription = "HalfEdges: num: {}".format(len(self.halfEdges))
        facesDescription = "Faces: num: {}".format(len(self.faces))

        allVertices = [x.getVertices() for x in self.halfEdges]
        flattenedVertices = [x for (x, y) in allVertices for x in (x, y)]
        setOfVertices = set(flattenedVertices)
        vertexSet = "Vertex Set: num: {}/{}".format(len(setOfVertices), len(flattenedVertices))

        infiniteEdges = [x for x in self.halfEdges if x.isInfinite()]
        infiniteEdgeDescription = "Infinite Edges: num: {}".format(len(infiniteEdges))

        completeEdges = []
        for x in self.halfEdges:
            if not x in completeEdges and x.twin not in completeEdges:
                completeEdges.append(x)

        completeEdgeDescription = "Complete Edges: num: {}".format(len(completeEdges))

        edgelessVertices = [x for x in self.vertices if x.isEdgeless()]
        edgelessVerticesDescription = "Edgeless vertices: num: {}".format(len(edgelessVertices))

        edgeCountForFaces = [str(len(f.outerBoundaryEdges)) for f in self.faces]
        edgeCountForFacesDescription = "Edge Counts for Faces: {}".format("-".join(edgeCountForFaces))

        return "\n".join(["---- DCEL Description: ",
                          verticesDescription,
                          edgesDescription,
                          facesDescription,
                          vertexSet,
                          infiniteEdgeDescription,
                          completeEdgeDescription,
                          edgelessVerticesDescription,
                          edgeCountForFacesDescription,
                          "----\n"])


    #--------------------
    # MAIN VERTEX, HALFEDGE, FACE CREATION:
    #--------------------
    def newVertex(self, loc, data=None):
        """ Create a new vertex,  or reuse an existing vertex """
        assert(isinstance(loc, np.ndarray))
        newVert = None
        matchingVertices = self.vertex_quad_tree.intersect(Vertex.free_bbox(loc))
        if matchingVertices:
            #a good enough vertex exists
            newVert = matchingVertices.pop()
            if data is not None:
                newVert.data.update(data)
            logging.debug("Found a matching vertex: {}".format(newVert))
        else:
            #no matching vertex,  add this new one
            newVert = Vertex(loc, data=data, dcel=self)
            logging.debug("No matching vertex,  storing: {}".format(newVert))
            self.vertices.append(newVert)
            self.vertex_quad_tree.insert(item=newVert, bbox=newVert.bbox())
        assert(newVert is not None)
        return newVert

    def newEdge(self, originVertex, twinVertex, face=None, twinFace=None, prev=None, prev2=None):
        """ Create a new half edge pair,  after specifying its start and end.
            Can set the faces,  and previous edges of the new edge pair.
            Returns the outer edge
        """
        #todo: check for an already existing edge
        assert(originVertex is None or isinstance(originVertex, Vertex))
        assert(twinVertex is None or isinstance(twinVertex, Vertex))
        e1 = HalfEdge(originVertex, None, dcel=self)
        e2 = HalfEdge(twinVertex, e1, dcel=self)
        e1.twin = e2 #fixup
        #Connect with passed in details
        if face is not None:
            assert(isinstance(face, Face))
            e1.face = face
            face.add_edge(e1)
        if twinFace is not None:
            assert(isinstance(twinFace, Face))
            e2.face = twinFace
            twinFace.add_edge(e2)
        if prev is not None:
            assert(isinstance(prev, HalfEdge))
            e1.addPrev(prev)
        if prev2 is not None:
            assert(isinstance(prev2, HalfEdge))
            e2.addPrev(prev2)
        self.halfEdges.extend([e1, e2])
        logging.debug("Created Edge Pair: {}".format(e1.index))
        logging.debug("Created Edge Pair: {}".format(e2.index))
        return e1

    def newFace(self, site=None, edges=None):
        """ Creates a new face to link edges """
        if site is None:
            site = np.array([0,0])
        assert(isinstance(site, np.ndarray))
        newFace = Face(site=site, dcel=self)
        self.faces.append(newFace)
        if edges is not None:
            newFace.add_edges(edges)
                
        
        return newFace

    #--------------------
    # UTILITY CREATION METHODS
    #--------------------
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
        vertices = vs
        pathVerts = zip(vertices, islice(cycle(vertices), 1, None))
        path = []
        for a,b in pathVerts:
            if not close and a == vertices[-1]:
                continue
            path.append(self.createEdge(a,b, edata=edata, vdata=vdata))
        return path        
    
    #--------------------
    #MISC METHODS:
    #--------------------
    def linkEdgesTogether(self, edges):
        """ Given a list of half edges, set their prev and next fields in order """
        #TODO: check this
        assert(all([isinstance(x, HalfEdge) for x in edges]))
        for (a,b) in zip(edges, islice(cycle(edges), 1, None)):
            a.addNext(b)

    def setFaceForEdgeLoop(self, face, edge, isInnerComponentList=False):
        """ For a face and a list of edges,  link them together
            If the edges are the outer components,  just put the first edge in the face,
            Otherwise places all the edges in the face """
        assert(isinstance(face, Face))
        assert(isinstance(edge, HalfEdge))
        #TODO: implement this
        raise Exception("Reimplement this")


    def orderVertices(self, focus, vertices):
        """ Given a focus point and a list of vertices,  sort them
            by the counter-clockwise angle position they take relative """
        #TODO: Rewrite this, make focus optional
        #todo: rename this to make_convex_hull
        assert(all([isinstance(x, Vertex) for x in vertices]))
        relativePositions = [[x-focus[0], y-focus[1]] for x, y in vertices]
        angled = [(atan2(yp, xp), x, y) for xp, yp, x, y in zip(relativePositions, vertices)]
        sortedAngled = sorted(angled)
        #todo: use scipy.spatial.ConvexHull /graham scan
        return sortedAngled

    def constrain_half_edges(self, bbox):
        """ For each halfedge,  shorten it to be within the bounding box  """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)

        logging.debug("\n---------- Constraint Checking")
        numEdges = len(self.halfEdges)
        for e in self.halfEdges:

            logging.debug("\n---- Checking constraint for: {}/{} {}".format(e.index, numEdges, e))
            #If its going to be removed, ignore it
            if e.markedForCleanup:
                continue
            #if both vertices are within the bounding box,  don't touch
            if e.isConstrained():
                continue
            if e.within(bbox):
                continue
            #if both vertices are out of the bounding box,  clean away entirely
            if e.outside(bbox):
                logging.debug("marking for cleanup: e{}".format(e.index))
                e.markForCleanup()
                continue

            #constrain the point outside the bounding box:
            logging.debug("Constraining")
            newBounds = e.constrain(bbox)
            orig_1, orig_2 = e.getVertices()
            e.clearVertices()
            v1 = self.newVertex(*newBounds[0])
            v2 = self.newVertex(*newBounds[1])
            #This doesn't account for an edge that crosses the entire bbox
            if not (v1 == orig_1 or v1 == orig_2 or v2 == orig_1 or v2 == orig_2):
                logging.debug("Vertex Changes upon constraint:")
                logging.debug(" Originally: {},  {}".format(orig_1, orig_2))
                logging.debug(" New: {},  {}".format(v1, v2))
                raise Exception("One vertex shouldn't change")

            e.addVertex(v1)
            e.addVertex(v2)
            e.setConstrained()
            logging.debug("Result: {}".format(e))


    def purge_infinite_edges(self):
        """ Remove all edges that don't have a start or end """
        logging.debug("Purging infinite edges")
        edges_to_purge = [x for x in self.halfEdges if x.isInfinite()]
        logging.info("Purging {} infinite edges".format(len(edges_to_purge)))
        for e in edges_to_purge:
            logging.debug("Purging infinite: e{}".format(e.index))
            e.clearVertices()
            self.halfEdges.remove(e)
            e.face.removeEdge(e)
            e.connectNextToPrev()

    def purge_edges(self):
        """ Remove all edges that have been marked for cleanup """
        logging.debug("Purging edges marked for cleanup")
        edges_to_purge = [x for x in self.halfEdges if x.markedForCleanup]
        twin_edges_to_purge = []
        for e in edges_to_purge:
            twin_edges_to_purge.append(e.twin)
            logging.debug("Purging: e{}".format(e.index))
            e.clearVertices()
            self.halfEdges.remove(e)
            if e.face is not None:
                e.face.removeEdge(e)
            if e.twin.face is not None:
                e.twin.face.removeEdge(e)
            e.connectNextToPrev()
        #don't remove twins, they may be used separately on boundaries
            
    def purge_vertices(self):
        """ Remove all vertices that aren't connected to at least one edge"""
        used_vertices = [x for x in self.vertices if not x.isEdgeless()]
        self.vertices = used_vertices


    def complete_faces(self, bbox=None):
        """ Verify each face,  connecting non-connected edges,  taking into account
            corners of the bounding box passed in,  which connects using a pair of edges
        """
        #TODO: Check this
        logging.debug("---------- Completing faces")
        if bbox is None:
            bbox = np.array([0, 0, 1, 1])

        for f in self.faces:
            logging.debug("Completing face: {}".format(f.index))
            #sort edges going anti-clockwise
            f.sort_edges()
            edgeList = f.getEdges()
            if not bool(edgeList):
                #cleanup if the face is empty
                f.markForCleanup()
                continue
            #reverse to allow popping off
            #edgeList.reverse() ? unneeded? 
            first_edge = edgeList[-1]
            while len(edgeList) > 1:
                #pop off in anti-clockwise order
                current_edge = edgeList.pop()
                prior_edge = edgeList[-1]
                logging.debug("---- Edge Pair: {} - {}".format(current_edge.index, prior_edge.index))
                self.calculate_edge_connections(current_edge, prior_edge, bbox, f)
                
            #after everything, connect the ends of the loop
            self.calculate_edge_connections(edgeList.pop(), first_edge, bbox, f)
            logging.debug("Final sort of face: {}".format(f.index))
            f.sort_edges()
            logging.debug("Result: {}".format([x.index for x in f.getEdges()]))
            logging.debug("----")

            #Get the opposites
            f.outerBoundaryEdges = [x.twin for x in f.edgeList]
            

    def calculate_edge_connections(self, current_edge, prior_edge, bbox, f):
        """ Connect two edges within a bbox, with a corner if necessary  """
        assert(isinstance(f, Face))
        if prior_edge.connections_align(current_edge):
            current_edge.setPrev(prior_edge)
            return

        logging.debug("Edges do not align:\n\t e1: {} \n\t e2: {}".format(current_edge.twin.origin,
                                                                          prior_edge.origin))
        #if they intersect with different bounding walls,  they need a corner
        intersect_1 = current_edge.intersects_edge(bbox)
        intersect_2 = prior_edge.intersects_edge(bbox)
        logging.debug("Intersect Values: {} {}".format(intersect_1, intersect_2))

        if intersect_1 is None or intersect_2 is None:
            logging.debug("Non- side intersecting lines")

            #Simple connection requirement, straight line between end points
        if intersect_1 == intersect_2 or any([x is None for x in [intersect_1, intersect_2]]):
            logging.debug("Match, new simple edge between: {}={}".format(current_edge.index,
                                                                         prior_edge.index))
            #connect together with simple edge
            #twin face is not set because theres no face outside of bounds
            newEdge = self.newEdge(prior_edge.twin.origin,
                                   current_edge.origin,
                                   face=f,
                                   prev=prior_edge)
            newEdge.data[EdgeE.COLOUR] = [1, 0, 0, 1]
            newEdge.setPrev(prior_edge)
            current_edge.setPrev(newEdge)

        else:
            logging.debug("Creating a corner edge connection between: {}={}".format(current_edge.index, prior_edge.index))
            #Connect the edges via an intermediate, corner vertex
            newVert = self.create_corner_vertex(intersect_1, intersect_2, bbox)
            logging.debug("Corner Vertex: {}".format(newVert))
            newEdge_1 = self.newEdge(prior_edge.twin.origin, newVert, face=f, prev=prior_edge)
            newEdge_2 = self.newEdge(newVert, current_edge.origin, face=f, prev=newEdge_1)

            newEdge_1.data[EdgeE.COLOUR] = [0, 1, 0, 1]
            newEdge_2.data[EdgeE.COLOUR] = [0, 0, 1, 0]
            
            current_edge.addPrev(newEdge_2)
            newEdge_2.addPrev(newEdge_1)
            newEdge_1.addPrev(prior_edge)


    def purge_faces(self):
        """ Same as purging halfedges or vertices,  but for faces """
        to_clean = [x for x in self.faces if x.markedForCleanup or not x.has_edges()]
        self.faces = [x for x in self.faces if not x.markedForCleanup or x.has_edges()]
        for face in to_clean:
            edges = face.getEdges()
            for edge in edges:
                face.removeEdge(edge)
            for edge in face.outerBoundaryEdges.copy():
                face.removeEdge(edge)
                

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

    def fixup_halfedges(self):
        """ Fix all halfedges to ensure they are counter-clockwise ordered """
        logging.debug("---- Fixing order of vertices for each edge")
        for e in self.halfEdges:
            e.fixup()

    def verify_edges(self):
        """ check all edges... are well formed?  """
        #TODO: Fix
        raise Exception("Broken")
        #make sure every halfedge is only used once
        logging.debug("Verifying edges")
        troublesomeEdges = [] #for debugging
        usedEdges = {}
        for f in self.faces:
            for e in f.edgeList:
                if e.isInfinite():
                    raise Exception("Edge {} is infinite when it shouldn't be".format(e.index))
                if e < e.twin:
                    #for e's face, e -> e twin should be a left turn
                    raise Exception("Edge {} is not anti-clockwise".format(e.index))
                    #raise Exception("Edge {} is not anti clockwise".format(e.index))
                if e.index not in usedEdges:
                    usedEdges[e.index] = f.index
                else:
                    raise Exception("Edge {} in {} already used in {}".format(e.index, f.index, usedEdges[e.index]))
        logging.debug("Edges verified")
        return troublesomeEdges

    @staticmethod
    def loadfile(filename):
        """ Create a DCEL from a saved pickle """
        if not isfile(filename):
            raise Exception("Non-existing filename to load into dcel")
        with open(filename, 'rb') as f:
            dcel_data = pickle.load(f)
        the_dcel = DCEL()
        the_dcel.import_data(dcel_data)
        return the_dcel

    def savefile(self, filename):
        """ Save dcel data to a pickle """
        theData = self.export_data()
        with open(filename, 'wb') as f:
            pickle.dump(theData, f)
        

    def verify_faces_and_edges(self):
        """ Verify all faces and edges are well formed """
        all_face_edges = set()
        all_face_edge_twins = set()
        all_face_edgelist = set()
        all_face_edgelist_twins = set()
        all_edges = set([x for x in self.halfEdges])
        all_edge_twins = set([x.twin for x in self.halfEdges])
        for face in self.faces:
            all_face_edges.update([x for x in face.outerBoundaryEdges])
            all_face_edge_twins.update([x.twin for x in face.outerBoundaryEdges])
            all_face_edgelist.update([x for x in face.edgeList])
            all_face_edgelist_twins.update([x.twin for x in face.edgeList])

        if (all_face_edges == all_edges):
            IPython.embed(simple_prompt=True)
            

    def constrain_to_circle(self, centre, radius):
        """ Limit all faces, edges, and vertices to be within a circle,
        adding boundary verts and edges as necessary """
        assert(isinstance(centre, np.ndarray))
        assert(isinstance(radius, float))
        assert(centre.shape == (2,))
        removed_edges = []
        modified_edges = []
        
        for he in self.halfEdges:
            results = he.within_circle(centre, radius)
            arr = he.origin.toArray()
            if all(results): #if both within circle: leave
                continue
            elif not any(results): #if both without: remove
                he.markForCleanup()
                removed_edges.append(he)
            else: #one within, one without, modify
                #Get the further point
                closer, further, isOrigin = he.getCloserAndFurther(centre, radius)
                #create a line
                if isOrigin:
                    asLine = Line.newLine(he.origin, he.twin.origin, np.array([0,0,1,1]))
                else:
                    asLine = Line.newLine(he.twin.origin, he.origin, np.array([0,0,1,1]))
                #solve in relation to the circle
                intersection = asLine.intersect_with_circle(centre, radius)
                #get the appropriate intersection
                if intersection[0] is None:
                    closest = intersection[1]
                elif intersection[1] is None:
                    closest = intersection[0]
                else:
                    closest = intersection[np.argmin(get_distance(np.array(intersection), further))]
                #Create a new vertex to replace the old out of bounds vertex
                newVert = self.newVertex(*closest)
                orig1, orig2 = he.getVertices()
                he.clearVertices()
                #re-add the old vertex and new vertex to the half edge
                if isOrigin:
                    #origin is closer, replace the twin
                    he.addVertex(orig1)
                    he.addVertex(newVert)
                else:
                    #twin is closer, replace the origin
                    he.addVertex(newVert)
                    he.addVertex(orig2)
                    modified_edges.append(he)

        #todo: fixup faces
        
        self.purge_edges()
        self.purge_vertices()
        self.purge_faces()
        self.purge_infinite_edges()
        self.complete_faces()

        
            
