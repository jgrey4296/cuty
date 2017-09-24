""" The Top Level DCEL DataStructure. """
import math
from math import atan2
from random import random
import logging
import sys
from collections import namedtuple
from numbers import Number
import IPython
import numpy as np
import pyqtree
from os.path import isfile
import pickle

from cairo_utils import get_distance
from .Face import Face
from .HalfEdge import HalfEdge
from .Vertex import Vertex
from .Line import Line

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
            newVertex = Vertex(vData['x'], vData['y'], index=vData['i'])
            logging.debug("Re-created vertex: {}".format(newVertex.index))
            local_vertices[newVertex.index] = DataPair(newVertex.index, newVertex, vData)
        #edges by index
        logging.info("Re-Creating Edges: {}".format(len(data['halfEdges'])))
        for eData in data['halfEdges']:
            newEdge = HalfEdge(index=eData['i'])
            logging.debug("Re-created Edge: {}".format(newEdge.index))
            local_edges[newEdge.index] = DataPair(newEdge.index, newEdge, eData)
        #faces by index
        logging.info("Re-Creating Faces: {}".format(len(data['faces'])))
        for fData in data['faces']:
            newFace = Face(fData['sitex'], fData['sitey'], index=fData['i'])
            logging.debug("Re-created face: {}".format(newFace.index))
            local_faces[newFace.index] = DataPair(newFace.index, newFace, fData)
        #Everything now exists,  so:
        #connect the objects up,  instead of just using ids
        try:
            for vertex in local_vertices.values():
                vertex.obj.halfEdges = [local_edges[x].obj for x in vertex.data['halfEdges']]
        except Exception as e:
            logging.info("Error for vertex")
            IPython.embed(simple_prompt=True)
        try:
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
            for face in local_faces.values():
                face.obj.edgeList = [local_edges[x].obj for x in face.data['edges']]
                face.obj.innerComponents = [x.twin for x in face.obj.edgeList if x.twin.face is not None]
        except Exception as e:
            logging.info("Error for face")
            IPython.embed(simple_prompt=True)

        #Now transfer into the actual object:
        self.vertices = [x.obj for x in local_vertices.values()]
        self.halfEdges = [x.obj for x in local_edges.values()]
        self.faces = [x.obj for x in local_faces.values()]
        self.calculate_quad_tree()


    def clear_quad_tree(self):
        self.vertex_quad_tree = None

    def calculate_quad_tree(self):
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
        for vertex in self.vertices:
            self.vertex_quad_tree.insert(item=vertex, bbox=vertex.bbox())

    def __str__(self):
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

        edgeCountForFaces = [str(len(f.innerComponents)) for f in self.faces]
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



    def newVertex(self, x, y):
        """ Get a new vertex,  or reuse an existing vertex """
        assert(isinstance(x, Number))
        assert(isinstance(y, Number))
        newVertex = Vertex(x, y)
        matchingVertices = self.vertex_quad_tree.intersect(newVertex.bbox())
        if matchingVertices:
            #a good enough vertex exists
            newVertex = matchingVertices.pop()
            logging.debug("Found a matching vertex: {}".format(newVertex))
        else:
            #no matching vertex,  add this new one
            logging.debug("No matching vertex,  storing: {}".format(newVertex))
            self.vertices.append(newVertex)
            self.vertex_quad_tree.insert(item=newVertex, bbox=newVertex.bbox())
        return newVertex

    def newEdge(self, originVertex, twinVertex, face=None, twinFace=None, prev=None, prev2=None):
        """ Get a new half edge pair,  after specifying its start and end.
            Can set the faces,  and previous edges of the new edge pair.
            Returns the outer edge
        """
        assert(originVertex is None or isinstance(originVertex, Vertex))
        assert(twinVertex is None or isinstance(twinVertex, Vertex))
        e1 = HalfEdge(originVertex, None)
        e2 = HalfEdge(twinVertex, e1)
        e1.twin = e2 #fixup
        if face:
            assert(isinstance(face, Face))
            e1.face = face
            face.add_edge(e1)
        if twinFace:
            assert(isinstance(twinFace, Face))
            e2.face = twinFace
            twinFace.add_edge(e2)
        if prev:
            assert(isinstance(prev, HalfEdge))
            e1.prev = prev
            prev.next = e1
        if prev2:
            assert(isinstance(prev2, HalfEdge))
            e2.prev = prev2
            prev2.next = e2
        self.halfEdges.extend([e1, e2])
        logging.debug("Created Edge Pair: {}".format(e1.index))
        logging.debug("Created Edge Pair: {}".format(e2.index))
        return e1

    def newFace(self, site_x, site_y):
        """ Creates a new face to link edges """
        assert(isinstance(site_x, Number))
        assert(isinstance(site_y, Number))
        newFace = Face(site_x, site_y)
        self.faces.append(newFace)
        return newFace

    def linkEdgesTogether(self, edges):
        assert(all([isinstance(x, HalfEdge) for x in edges]))
        for i, e in enumerate(edges):
            e.prev = edges[i-1]
            e.prev.next = e

    def setFaceForEdgeLoop(self, face, edge, isInnerComponentList=False):
        """ For a face and a list of edges,  link them together
            If the edges are the outer components,  just put the first edge in the face,
            Otherwise places all the edges in the face """
        assert(isinstance(face, Face))
        assert(isinstance(edge, HalfEdge))

        start = edge
        current = edge.next
        if isInnerComponentList:
            face.innerComponents.append(start)
        else:
            face.outerComponent = start
        start.face = face
        while current is not start and current.next is not None:
            current.face = face
            current = current.next
            if isInnerComponentList:
                face.innerComponents.append(current)


    def orderVertices(self, focus, vertices):
        """ Given a focus point and a list of vertices,  sort them
            by the counter-clockwise angle position they take relative """
        assert(all([isinstance(x, Vertex) for x in vertices]))
        relativePositions = [[x-focus[0], y-focus[1]] for x, y in vertices]        
        angled = [(atan2(yp, xp), x, y) for xp, yp, x, y in zip(relativePositions, vertices)]
        sortedAngled = sorted(angled)
        return sortedAngled

    def constrain_half_edges(self, bbox):
        """ For each halfedge,  shorten it to be within the bounding box  """
        assert(isinstance(bbox, np.ndarray))
        assert(len(bbox) == 4)

        logging.debug("\n---------- Constraint Checking")
        numEdges = len(self.halfEdges)
        for e in self.halfEdges:

            logging.debug("\n---- Checking constraint for: {}/{} {}".format(e.index, numEdges, e))
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
            else:
                logging.debug("Constraining")
                #else constrain the point outside the bounding box:
                newBounds = e.constrain(bbox)
                orig_1, orig_2 = e.getVertices()
                e.clearVertices()
                v1 = self.newVertex(newBounds[0][0], newBounds[0][1])
                v2 = self.newVertex(newBounds[1][0], newBounds[1][1])
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

        twin_edges_to_purge = [x for x in twin_edges_to_purge if x in self.halfEdges]            
        for e in twin_edges_to_purge:
            logging.debug("Purging Twin: e{}".format(e.index))
            e.clearVertices()
            if e.face is not None:
                e.face.removeEdge(e)
            if e.twin.face is not None:
                e.twin.face.removeEdge(e)
            e.connectNextToPrev()

            

    def purge_vertices(self):
        """ Remove all vertices that aren't connected to at least one edge"""
        used_vertices = [x for x in self.vertices if len(x.halfEdges) > 0]
        self.vertices = used_vertices


    def complete_faces(self, bbox=None):
        """ Verify each face,  connecting non-connected edges,  taking into account
            corners of the bounding box passed in,  which connects using a pair of edges
        """
        logging.debug("---------- Completing faces")
        if bbox is None:
            bbox = np.array([0, 0, 1, 1])
        assert(bbox is not None)
        for f in self.faces:
            logging.debug("Completing face: {}".format(f.index))
            #sort edges going anti-clockwise
            f.sort_edges()
            edgeList = f.getEdges()
            if not bool(edgeList):
                f.markForCleanup()
                continue
            #reverse to allow popping off
            edgeList.reverse()
            first_edge = edgeList[-1]
            while len(edgeList) > 1:
                #pop off in anti-clockwise order
                current_edge = edgeList.pop()
                nextEdge = edgeList[-1]
                logging.debug("---- Edge Pair: {} - {}".format(current_edge.index, nextEdge.index))
                if current_edge.connections_align(nextEdge):
                    current_edge.setNext(nextEdge)
                else:
                    logging.debug("Edges do not align:\n\t e1: {} \n\t e2: {}".format(current_edge.twin.origin, nextEdge.origin))
                    #if they intersect with different bounding walls,  they need a corner
                    intersect_1 = current_edge.intersects_edge(bbox)
                    intersect_2 = nextEdge.intersects_edge(bbox)
                    logging.debug("Intersect Values: {} {}".format(intersect_1, intersect_2))

                    if intersect_1 is None or intersect_2 is None:
                        logging.debug("Non- side intersecting lines")

                    if intersect_1 == intersect_2 or intersect_1 is None or intersect_2 is None:
                        logging.debug("Intersects match,  creating a simple edge between: {}={}".format(current_edge.index, nextEdge.index))
                        #connect together with simple edge
                        newEdge = self.newEdge(current_edge.twin.origin, nextEdge.origin, face=f, prev=current_edge)
                        current_edge.setNext(newEdge)
                        newEdge.setNext(nextEdge)

                    else:
                        logging.debug("Creating a corner edge connection between: {}={}".format(current_edge.index, nextEdge.index))
                        #connect via a corner
                        newVertex = self.create_corner_vertex(intersect_1, intersect_2, bbox)
                        logging.debug("Corner Edge: {}".format(newVertex))
                        newEdge_1 = self.newEdge(current_edge.twin.origin, newVertex, face=f, prev=current_edge)
                        newEdge_2 = self.newEdge(newVertex, nextEdge.origin, face=f, prev=newEdge_1)

                        current_edge.setNext(newEdge_1)
                        newEdge_1.setNext(newEdge_2)
                        newEdge_2.setNext(nextEdge)

            #at this point,  only the last hasn't been processed
            #as above,  but:
            logging.debug("Checking final edge pair")
            current_edge = edgeList.pop()
            if current_edge.connections_align(first_edge):
                current_edge.setNext(first_edge)
            else:
                intersect_1 = current_edge.intersects_edge(bbox)
                intersect_2 = first_edge.intersects_edge(bbox)
                if intersect_1 is None or intersect_2 is None:
                    logging.debug("Edge Intersection is None")
                elif intersect_1 == intersect_2:
                    logging.debug("Intersects match,  creating final simple edge")
                    #connect with simple edge
                    newEdge = self.newEdge(current_edge.twin.origin, first_edge.origin, face=f, prev=current_edge)
                    current_edge.setNext(newEdge)
                    newEdge.setNext(first_edge)
                else:
                    logging.debug("Creating final corner edge connection between: {}={}".format(current_edge.index, first_edge.index))
                    #connect via a corner
                    newVertex = self.create_corner_vertex(intersect_1, intersect_2, bbox)
                    newEdge_1 = self.newEdge(current_edge.twin.origin, newVertex, face=current_edge.face, prev=current_edge)
                    newEdge_2 = self.newEdge(newVertex, first_edge.origin, face=current_edge.face, prev=newEdge_1)
                    #newEdge_1 = self.newEdge(current_edge.twin.origin, newVertex, face=current_edge.face, prev=current_edge)
                    #newEdge_2 = self.newEdge(newVertex, nextEdge.origin, face=current_edge.face, prev=newEdge_1)
                    current_edge.setNext(newEdge_1)
                    newEdge_1.setNext(newEdge_2)
                    newEdge_2.setNext(first_edge)

            logging.debug("Final sort of face: {}".format(f.index))
            f.sort_edges()
            logging.debug("Result: {}".format([x.index for x in f.getEdges()]))
            logging.debug("----")

            f.innerComponents = [x.twin for x in f.edgeList]
            


    def purge_faces(self):
        """ Same as purging halfedges or vertices,  but for faces """
        to_clean = [x for x in self.faces if x.markedForCleanup]
        self.faces = [x for x in self.faces if not x.markedForCleanup or x.has_edges()]
        for face in to_clean:
            edges = face.getEdges()
            for edge in edges:
                face.removeEdge(edge)
            for edge in face.innerComponents.copy():
                face.removeEdge(edge)
                

    def create_corner_vertex(self, e1, e2, bbox):
        """ Given two intersections (0-3),  create the vertex that corners them """
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
        logging.debug("---- Fixing order of vertices for each edge")
        for e in self.halfEdges:
            e.fixup()

    def verify_edges(self):
        #make sure every halfedge is only used once
        logging.debug("Verifying edges")
        troublesomeEdges = [] #for debugging
        usedEdges = {}
        for f in self.faces:
            for e in f.edgeList:
                if e.isInfinite():
                    raise Exception("Edge {} is infinite when it shouldn't be".format(e.index))
                if e < e.twin:
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
        if not isfile(filename):
            raise Exception("Non-existing filename to load into dcel")
        with open(filename, 'rb') as f:
            dcel_data = pickle.load(f)
        the_dcel = DCEL()
        the_dcel.import_data(dcel_data)
        return the_dcel
        

    def verify_faces_and_edges(self):
        all_face_edges = set()
        all_face_edge_twins = set()
        all_face_edgelist = set()
        all_face_edgelist_twins = set()
        all_edges = set([x.index for x in self.halfEdges])
        all_edge_twins = set([x.twin.index for x in self.halfEdges])
        for face in self.faces:
            all_face_edges.update([x.index for x in face.innerComponents])
            all_face_edge_twins.update([x.twin.index for x in face.innerComponents])
            all_face_edgelist.update([x.index for x in face.edgeList])
            all_face_edgelist_twins.update([x.twin.index for x in face.edgeList])
            
        IPython.embed(simple_prompt=True)
        return all_face_edges == all_edges
    
