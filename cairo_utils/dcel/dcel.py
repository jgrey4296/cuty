""" The Top Level DCEL DataStructure. """
#pylint: disable=too-many-public-methods
import logging as root_logger
from collections import namedtuple
from math import atan2, degrees
from os.path import isfile
from itertools import cycle, islice
import pickle
import numpy as np
import pyqtree

from .face import Face
from .halfedge import HalfEdge
from .vertex import Vertex
from .constants import EdgeE, VertE, FaceE
from .line_intersector import LineIntersector

logging = root_logger.getLogger(__name__)


#for importing data into the dcel:
DataPair = namedtuple('DataPair', 'key obj data')

class DCEL:
    """ The total DCEL data structure, stores vertices, edges, and faces,
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
        self.faces = set([])
        self.half_edges = set([])
        self.bbox = bbox
        #todo: make this a stack of quadtrees
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)
        self.quad_tree_stack = []
        self.frontier = set([])
        self.should_merge_stacks = True

        self.data = {}

    def reset_frontier(self):
        """ Clear the general incremental algorithm frontier """
        self.frontier = set([])

    def copy(self):
        """ Completely duplicate the dcel """
        new_dcel = DCEL(self.bbox)
        new_dcel.import_data(self.export_data())
        return new_dcel

    def __str__(self):
        """ Create a text description of the DCEL """
        #pylint: disable=too-many-locals
        vertices_description = "Vertices: num: {}".format(len(self.vertices))
        edges_description = "HalfEdges: num: {}".format(len(self.half_edges))
        faces_description = "Faces: num: {}".format(len(self.faces))

        all_vertices = [x.getVertices() for x in self.half_edges]
        flattened_vertices = [x for (x, y) in all_vertices for x in (x, y)]
        set_of_vertices = set(flattened_vertices)
        vertex_set = "Vertex Set: num: {}/{}/{}".format(len(set_of_vertices),
                                                        len(flattened_vertices),
                                                        len(self.vertices))

        infinite_edges = [x for x in self.half_edges if x.isInfinite()]
        infinite_edge_description = "Infinite Edges: num: {}".format(len(infinite_edges))

        complete_edges = set()
        for x in self.half_edges:
            if not x in complete_edges and x.twin not in complete_edges:
                complete_edges.add(x)

        complete_edge_description = "Complete Edges: num: {}".format(len(complete_edges))

        edgeless_vertices = [x for x in self.vertices if x.isEdgeless()]
        edgeless_vertices_description = "Edgeless vertices: num: {}".format(len(edgeless_vertices))

        edge_count_for_faces = [str(len(f.edgeList)) for f in self.faces]
        edge_count_for_faces_description = \
                "Edge Counts for Faces: {}".format("-".join(edge_count_for_faces))

        purge_verts = "Verts to Purge: {}".format(len([x for x in self.vertices
                                                       if x.markedForCleanup]))
        purge_edges = "Hedges to Purge: {}".format(len([x for x in self.half_edges
                                                        if x.markedForCleanup]))
        purge_faces = "Faces to Purge: {}".format(len([x for x in self.faces
                                                       if x.markedForCleanup]))

        return "\n".join(["---- DCEL Description: ",
                          vertices_description,
                          edges_description,
                          faces_description,
                          vertex_set,
                          infinite_edge_description,
                          complete_edge_description,
                          edgeless_vertices_description,
                          edge_count_for_faces_description,
                          "-- Purging:",
                          purge_verts,
                          purge_edges,
                          purge_faces,
                          "----\n"])


    #------------------------------
    # def IO
    #------------------------------

    def export_data(self):
        """ Export a simple format to define vertices, halfedges, faces,
        uses identifiers instead of objects, allows reconstruction """
        data = {
            'vertices' : [x._export() for x in self.vertices],
            'half_edges' : [x._export() for x in self.half_edges],
            'faces' : [x._export() for x in self.faces],
            'bbox' : self.bbox
        }
        return data

    def import_data(self, data):
        """ Import the data format of identifier links to actual links,
        of export output from a dcel """
        #pylint: disable=too-many-locals
        #pylint: disable=too-many-statements
        assert(all([x in data for x in ['vertices', 'half_edges', 'faces', 'bbox']]))
        self.bbox = data['bbox']

        #dictionarys used to store {newIndex : (newIndex, newObject, oldData)}
        local_vertices = {}
        local_edges = {}
        local_faces = {}
        output_mapping = {}

        #-----
        # Reconstruct Verts, Edges, Faces:
        #-----
        #construct vertices by index
        logging.info("Re-Creating Vertices: {}".format(len(data['vertices'])))
        for v_data in data['vertices']:
            combined_data = {}
            combined_data.update({VertE.__members__[a] : b for a, b in v_data['enumData'].items()})
            combined_data.update(v_data['nonEnumData'])

            new_vert = Vertex(np.array([v_data['x'], v_data['y']]),
                              index=v_data['i'], data=combined_data,
                              dcel=self, active=v_data['active'])
            logging.debug("Re-created vertex: {}".format(new_vert.index))
            local_vertices[new_vert.index] = DataPair(new_vert.index, new_vert, v_data)

        #edges by index
        logging.info("Re-Creating HalfEdges: {}".format(len(data['half_edges'])))
        for e_data in data['half_edges']:
            combined_data = {}
            combined_data.update({EdgeE.__members__[a] : b for a, b in e_data['enumData'].items()})
            combined_data.update(e_data['nonEnumData'])
            new_edge = HalfEdge(index=e_data['i'], data=combined_data, dcel=self)
            logging.debug("Re-created Edge: {}".format(new_edge.index))
            local_edges[new_edge.index] = DataPair(new_edge.index, new_edge, e_data)

        #faces by index
        logging.info("Re-Creating Faces: {}".format(len(data['faces'])))
        for f_data in data['faces']:
            combined_data = {}
            combined_data.update({FaceE.__members__[a] : b for a, b in f_data['enumData'].items()})
            combined_data.update(f_data['nonEnumData'])
            new_face = Face(site=np.array([f_data['sitex'], f_data['sitey']]), index=f_data['i'],
                            data=combined_data, dcel=self)
            logging.debug("Re-created face: {}".format(new_face.index))
            local_faces[new_face.index] = DataPair(new_face.index, new_face, f_data)

        #-----
        # def Upon reconstruction, reattach ids to the same objects
        #-----
        #this only update standard connections, not user connections
        #TODO: PASS OUT A MAPPING OF OLD IDS TO NEW FOR USER UPDATES
        try:
            #connect vertices to their edges
            for vertex in local_vertices.values():
                vertex.obj.half_edges.update( \
                    [local_edges[x].obj for x in vertex.data['half_edges']])
        except Exception:
            logging.warning("Import Error for vertex")

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
        except Exception:
            logging.warning("Import Error for edge")

        try:
            #connect faces to their edges
            for face in local_faces.values():
                face.obj.edgeList = [local_edges[x].obj for x in face.data['edges']]
        except Exception:
            logging.warning("Import Error for face")

        #Now recalculate the quad tree as necessary
        self.calculate_quad_tree()

        #todo: pass the mapping back
        output_mapping['verts'] = {x.data['i'] : x.key for x in local_vertices.values()}
        output_mapping['edges'] = {x.data['i'] : x.key for x in local_edges.values()}
        output_mapping['faces'] = {x.data['i'] : x.key for x in local_faces.values()}
        return output_mapping

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
        the_data = self.export_data()
        with open("{}.dcel".format(filename), 'wb') as f:
            pickle.dump(the_data, f)



    #------------------------------
    # def quadtree
    #------------------------------

    def clear_quad_tree(self):
        """ Clear the internal quadtree of vertices """
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
        """ Add another vertex quad tree on top of the stack """
        self.quad_tree_stack.append(self.vertex_quad_tree)
        self.vertex_quad_tree = pyqtree.Index(bbox=self.bbox)

    def pop_quad_tree(self):
        """ Remove the top vertex quad tree from the stack """
        assert(bool(self.quad_tree_stack))
        sub_layer = self.quad_tree_stack.pop()
        if self.should_merge_stacks:
            for x in self.vertex_quad_tree.intersect(self.bbox):
                if x not in sub_layer.intersect(x.bbox()):
                    sub_layer.insert(item=x, bbox=x.bbox())
        self.vertex_quad_tree = sub_layer

    def add_to_current_quad_tree(self, verts):
        """ Add the passed in vertices into the current quad tree """
        assert(all([isinstance(x, Vertex) for x in verts]))
        assert(bool(self.quad_tree_stack))
        for x in verts:
            self.vertex_quad_tree.insert(item=x, bbox=x.bbox())


    def __enter__(self):
        """ Makes the Dcel a reusable context manager, that pushes
        and pops vertex quad trees for collision detection """
        self.push_quad_tree()

    def __exit__(self, e_type, value, traceback):
        self.pop_quad_tree()


    #------------------------------
    # def PURGING
    #------------------------------

    def purge_edge(self, target):
        """ Clean up and delete an edge """
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


        self.half_edges.remove(target)

        return target_update


    def purge_vertex(self, target):
        """ Clean up and delete a vertex """
        assert(isinstance(target, Vertex))
        logging.debug("Purging Vertex: {}".format(target.index))
        target_update = set()

        half_edges = target.half_edges.copy()
        for edge in half_edges:
            assert(edge.origin == target)
            edge.origin = None
            target.unregisterHalfEdge(edge)
            edge.markForCleanup()
            target_update.add(edge)

        self.vertices.remove(target)

        return target_update


    def purge_face(self, target):
        """ Clean up and delete a face """
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
            targets = targets.union([x for x in self.half_edges if x.markedForCleanup
                                     or x.isInfinite()])
            targets = targets.union([x for x in self.faces if x.markedForCleanup
                                     or not x.has_edges()])

        else:
            targets = set(targets)

        purged = set()
        while bool(targets):
            current = targets.pop()
            if current in purged:
                continue
            if isinstance(current, Vertex):
                targets = targets.union(self.purge_vertex(current))
            elif isinstance(current, HalfEdge):
                targets = targets.union(self.purge_edge(current))
            elif isinstance(current, Face):
                targets = targets.union(self.purge_face(current))
            purged.add(current)

        self.calculate_quad_tree()

    #------------------------------
    # def Vertex, Edge, HalfEdge Creation
    #------------------------------

    def new_vertex(self, loc, data=None, force=False):
        """ Create a new vertex, or reuse an existing vertex.
        to force a new vertex instead of reusing, set force to True
        """
        assert(isinstance(loc, np.ndarray))
        new_vert = None
        matching_vertices = self.vertex_quad_tree.intersect(Vertex.free_bbox(loc))
        logging.debug("Quad Tree Size: {}".format(self.vertex_quad_tree.countmembers()))
        logging.debug("Query result: {}".format(matching_vertices))
        if bool(matching_vertices) and not force:
            #a good enough vertex exists
            new_vert = matching_vertices.pop()
            if data is not None:
                new_vert.data.update(data)
            logging.debug("Found a matching vertex: {}".format(new_vert))
        else:
            #no matching vertex, add this new one
            new_vert = Vertex(loc, data=data, dcel=self)
            logging.debug("No matching vertex, storing: {}, {}".format(new_vert, new_vert.bbox()))

        assert(new_vert is not None)
        return new_vert

    def new_edge(self, origin_vertex, twin_vertex, face=None, twin_face=None,
                 prev_edge=None, twin_prev=None, next_edge=None, twin_next=None,
                 edata=None, vdata=None):
        """ Create a new half edge pair, after specifying its start and end.
            Can set the faces, and previous edges of the new edge pair.
            Returns the outer edge
        """
        #todo: check for an already existing edge
        assert(origin_vertex is None or isinstance(origin_vertex, Vertex))
        assert(twin_vertex is None or isinstance(twin_vertex, Vertex))
        e1 = HalfEdge(origin_vertex, None, dcel=self)
        e2 = HalfEdge(twin_vertex, e1, dcel=self)
        e1.twin = e2
        #Connect with passed in details
        if face is not None:
            assert(isinstance(face, Face))
            face.add_edge(e1)
        if twin_face is not None:
            assert(isinstance(twin_face, Face))
            twin_face.add_edge(e2)
        if prev_edge is not None:
            assert(isinstance(prev_edge, HalfEdge))
            e1.add_prev(prev_edge)
        if twin_prev is not None:
            assert(isinstance(twin_prev, HalfEdge))
            e2.add_prev(twin_prev)
        if next_edge is not None:
            assert(isinstance(next_edge, HalfEdge))
            e1.add_next(next_edge)
        if twin_next is not None:
            assert(isinstance(twin_next, HalfEdge))
            e2.add_next(twin_next)
        if edata is not None:
            e1.data.update(edata)
            e2.data.update(edata)
        if vdata is not None:
            e1.origin.data.update(vdata)
            e2.origin.data.update(vdata)
        self.half_edges.update([e1, e2])
        logging.debug("Created Edge Pair: {}".format(e1.index))
        logging.debug("Created Edge Pair: {}".format(e2.index))
        return e1

    def new_face(self, site=None, edges=None, verts=None, coords=None, data=None):
        """ Creates a new face to link edges """
        used_list = [edges is not None, verts is not None, coords is not None]
        assert(len([x for x in used_list if x]) < 2)
        if site is None:
            site = np.array([0, 0])
        assert(isinstance(site, np.ndarray))
        new_face = Face(site=site, dcel=self, data=data)
        self.faces.add(new_face)
        #populate the face if applicable:
        coord_hull_gen = False
        if coords is not None:
            assert(isinstance(coords, np.ndarray))
            assert(coords.shape[1] == 2)
            hull_coords = Face.hull_from_coords(coords)
            verts = [self.new_vertex(x) for x in hull_coords]
            coord_hull_gen = True

        if verts is not None:
            if not coord_hull_gen:
                verts, _ = Face.hull_from_vertices(verts)
            edges = []
            for s, e in zip(verts, islice(cycle(verts), 1, None)):
                edges.append(self.new_edge(s, e))

        if edges is not None:
            new_face.add_edges(edges)
            self.link_edges_together(edges, loop=True)

        return new_face


    def create_edge(self, origin, end, edata=None, vdata=None, subdivs=0):
        """ Utility to create two vertices, and put them into a pair of halfedges,
        returning a halfedge
        subdivs specifies number of inner segments to the line"""
        assert(isinstance(origin, np.ndarray))
        assert(isinstance(end, np.ndarray))
        v1 = self.new_vertex(origin)
        v2 = self.new_vertex(end)
        e = self.new_edge(v1, v2)
        if vdata is not None:
            assert(isinstance(vdata, dict))
            v1.data.update(vdata)
            v2.data.update(vdata)
        if edata is not None:
            assert(isinstance(edata, dict))
            e.data.update(edata)
        return e

    def create_path(self, vs, close=False, edata=None, vdata=None):
        """ Create multiple half_edges, that connect to one another.
        With optional path closing """
        assert(isinstance(vs, np.ndarray))
        vertices = vs
        path_verts = zip(vertices, islice(cycle(vertices), 1, None))
        path = []
        for a, b in path_verts:
            if not close and (a == vertices[-1]).all():
                continue
            path.append(self.create_edge(a, b, edata=edata, vdata=vdata))
        return path

    def create_bezier(self, vs, edata=None, vdata=None, single=False):
        """ Takes a list of tuples (len 3 or 4), and creates
        approximation lines, that can be triggered later to
        draw the true bezier shape,
        Bezier Tuple: (Start, cps, End)"""
        assert(isinstance(vs, list))
        assert(all([isinstance(x, tuple) for x in vs]))
        if edata is None:
            edata = {}
        edges = []

        if single:
            #create a single, multi breakpoint line
            first = vs[0]
            last = vs[-1]
            e = self.create_edge(first[0], last[-1], edata=edata, vdata=vdata)
            e.data[EdgeE.BEZIER] = vs
            e.twin.data[EdgeE.NULL] = True
            return [e]


        for v in vs:
            if len(v) == 2 and all([isinstance(x, tuple) for x in v]):
                #is a single edge, with different control points for different
                #directions
                raise Exception("Dual Control Point Edges not yet supported")
            elif len(v) == 3:
                #is a single cp bezier
                a, _, b = v
                edge = self.create_edge(a, b, edata=edata, vdata=vdata)
                edge.data[EdgeE.BEZIER] = [v]
                edge.twin.data[EdgeE.NULL] = True
                edges.append(edge)
            elif len(v) == 4:
                #is a two cp bezier
                a, _, _, b = v
                edge = self.create_edge(a, b, edata=edata, vdata=vdata)
                edge.data[EdgeE.BEZIER] = [v]
                edge.twin.data[EdgeE.NULL] = True
            else:
                raise Exception("Unrecognised bezier type: {}".format(len(v)))

        return edges



    #------------------------------
    # def Coherence Utils
    #------------------------------

    def link_edges_together(self, edges, loop=False):
        """ Given a list of half edges, set their prev and next fields in order """
        #TODO: check this
        assert(all([isinstance(x, HalfEdge) for x in edges]))
        if loop:
            other = islice(cycle(edges), 1, None)
        else:
            other = edges[1:]
        for (a, b) in zip(edges, other):
            a.add_next(b)

    def force_all_edge_lengths(self, l):
        """ Force all edges to be of length <= l. If over, split into multiple lines
        of length l. """
        assert(l > 0)
        processed = set()
        all_edges = list(self.half_edges)
        while bool(all_edges):
            current = all_edges.pop(0)
            assert(current.index not in processed)
            if current.getLength_sq() > l:
                _, new_edge = current.split_by_ratio(r=0.5)
                if new_edge.getLength_sq() > l:
                    all_edges.append(new_edge)
                else:
                    processed.add(new_edge.index)

            if current.getLength_sq() > l:
                all_edges.append(current)
            else:
                processed.add(current.index)



    def constrain_to_circle(self, centre, radius, candidates=None, force=False):
        """ Limit all faces, edges, and vertices to be within a circle,
        adding boundary verts and edges as necessary """
        assert(isinstance(centre, np.ndarray))
        assert(isinstance(radius, float))
        assert(centre.shape == (2, ))

        #constrain faces
        faces = self.faces.copy()
        for f in faces:
            logging.debug("Constraining Face: {}".format(f))
            f.constrain_to_circle(centre, radius, candidates=candidates, force=force)

        #constrain free edges
        hedges = self.half_edges.copy()
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
        """ Constrain the entire dcel to a bbox """
        assert(isinstance(bbox, np.ndarray))
        assert(bbox.shape == (4, ))

        faces = self.faces.copy()
        for f in faces:
            logging.debug("Constraining Face: {}".format(f))
            f.constrain_to_bbox(bbox, candidates=candidates, force=force)

        #constrain free edges
        hedges = self.half_edges.copy()
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

    def intersect_half_edges(self, edge_set=None):
        """ run a sweep line over the dcel,
        getting back halfedge intersections """

        li = LineIntersector(self)
        return li(edge_set=edge_set)


    def order_vertices(self, focus, vertices):
        """ Given a focus point and a list of vertices, sort them
            by the counter-clockwise angle position they take relative """
        assert(all([isinstance(x, Vertex) for x in vertices]))
        assert(isinstance(focus, np.ndarray))
        relative_positions = [v.loc - focus for v in vertices]
        zipped = zip(relative_positions, vertices)
        angled = [((degrees(atan2(loc[1], loc[0])) + 360) % 360, vert) for loc, vert in zipped]
        sorted_angled = sorted(angled)
        # rads = (np.arctan2(verts[:, 1], verts[:, 0]) + TWOPI) % TWOPI
        # ordered = sorted(zip(rads, opp_hedges))
        return [vert for loc, vert in sorted_angled]


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
        return self.new_vertex(*v3)

    def verify_all(self):
        """ Ensure all faces, edges and halfedges are coherent """
        reg_verts = set([x.index for x in self.vertices])
        reg_hedges = set([x.index for x in self.half_edges])
        reg_faces = set([x.index for x in self.faces])

        vert_hedges = set()
        for v in self.vertices:
            vert_hedges.update([x.index for x in v.half_edges])

        hedge_verts = set()
        hedge_nexts = set()
        hedge_prevs = set()
        hedge_faces = set()
        for h in self.half_edges:
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

        assert(all([not bool(x) for x in [vert_hedge_diff,
                                          hedge_vert_diff,
                                          hedge_nexts_diff,
                                          hedge_prevs_diff,
                                          hedge_faces_diff,
                                          face_edges_diff]]))
